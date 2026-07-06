import glob
import multiprocessing as mp
import os

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from util import load_keypoints, normalize_coords


COCO_JOINT_WEIGHTS = np.array([
    3.0, 3.0, 3.0, 2.0, 2.0,
    4.0, 4.0, 2.0, 2.0,
    4.0, 4.0, 2.0, 2.0,
    1.0, 1.0, 1.0, 1.0,
], dtype=np.float32)


class KeypointDataset(Dataset):
    def __init__(self, args, max_interval=None):
        self.args = args
        self.seq_dir = args.data_dir
        self.seq_name = os.path.basename(self.seq_dir.rstrip('/'))
        self.img_dir = os.path.join(self.seq_dir, 'color')
        self.keypoint_dir = args.keypoint_dir or os.path.join(self.seq_dir, 'keypoints')
        if not os.path.isdir(self.keypoint_dir):
            raise FileNotFoundError('Keypoint directory not found: {}'.format(self.keypoint_dir))

        img_names = sorted(os.listdir(self.img_dir))
        self.num_imgs = min(self.args.num_imgs, len(img_names))
        self.img_names = img_names[:self.num_imgs]
        self.img_files = [os.path.join(self.img_dir, name) for name in self.img_names]

        h, w, _ = imageio.imread(self.img_files[0]).shape
        self.h, self.w = h, w
        max_interval = self.num_imgs - 1 if not max_interval else max_interval
        self.max_interval = mp.Value('i', max_interval)
        self.num_pts = self.args.num_pts
        self.num_joints = self.args.num_joints
        self.min_keypoint_conf = self.args.min_keypoint_conf
        self.patch_size = max(1, int(self.args.patch_size))
        self.use_mask = bool(getattr(self.args, 'foreground_hard_mask', False))

        self.mask_files = [img_file.replace('color', 'mask').replace('.jpg', '.png') for img_file in self.img_files]
        self.has_mask = len(self.mask_files) > 0 and os.path.exists(self.mask_files[0])
        self.keypoints, self.confidences, self.valid = self._load_all_keypoints()
        self.offsets, self.offset_weights = self._build_patch_offsets(self.patch_size)
        if self.num_joints == len(COCO_JOINT_WEIGHTS):
            self.joint_weights = COCO_JOINT_WEIGHTS
        else:
            self.joint_weights = np.ones((self.num_joints,), dtype=np.float32)

    def __len__(self):
        return self.num_imgs * 100000

    def set_max_interval(self, max_interval):
        self.max_interval.value = min(max_interval, self.num_imgs - 1)

    def increase_max_interval_by(self, increment):
        curr_max_interval = self.max_interval.value
        self.max_interval.value = min(curr_max_interval + increment, self.num_imgs - 1)

    def _load_all_keypoints(self):
        keypoints = []
        confidences = []
        valid = []
        for img_name in self.img_names:
            stem = os.path.splitext(img_name)[0]
            json_path = os.path.join(self.keypoint_dir, stem + '.json')
            if not os.path.exists(json_path):
                points = np.zeros((self.num_joints, 2), dtype=np.float32)
                conf = np.zeros((self.num_joints,), dtype=np.float32)
                valid_mask = np.zeros((self.num_joints,), dtype=bool)
            else:
                points, conf, valid_mask = load_keypoints(json_path,
                                                         num_joints=self.num_joints,
                                                         keypoint_format=self.args.keypoint_format)
            keypoints.append(points)
            confidences.append(conf)
            valid.append(valid_mask)
        return np.stack(keypoints), np.stack(confidences), np.stack(valid)

    def _build_patch_offsets(self, patch_size):
        radius = patch_size // 2
        offsets = []
        weights = []
        sigma = max(radius / 2.0, 1.0)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                offsets.append([dx, dy])
                weights.append(np.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma)))
        weights = np.asarray(weights, dtype=np.float32)
        weights /= np.maximum(weights.max(), 1e-6)
        return np.asarray(offsets, dtype=np.float32), weights

    def _sample_pair(self, idx):
        id1 = idx % self.num_imgs
        max_interval = min(self.max_interval.value, self.num_imgs - 1)
        lo = max(0, id1 - max_interval)
        hi = min(self.num_imgs, id1 + max_interval + 1)
        candidates = [i for i in range(lo, hi) if i != id1]
        if not candidates:
            candidates = [i for i in range(self.num_imgs) if i != id1]
        if not candidates:
            return id1, id1
        candidates = np.asarray(candidates)
        weights = np.ones((len(candidates),), dtype=np.float32)
        weights[np.abs(candidates - id1) <= 1] = 3.0
        weights /= weights.sum()
        id2 = int(np.random.choice(candidates, p=weights))
        return id1, id2

    def _load_mask(self, frame_id):
        if not self.has_mask or not self.use_mask:
            return None
        mask = imageio.imread(self.mask_files[frame_id])
        if mask.ndim == 3:
            mask = mask[..., :3].sum(axis=-1)
        return mask > 0

    def _inside_image(self, pt):
        return 0 <= pt[0] <= self.w - 1 and 0 <= pt[1] <= self.h - 1

    def _mask_contains(self, mask, pt):
        if mask is None:
            return True
        x = int(np.clip(np.round(pt[0]), 0, self.w - 1))
        y = int(np.clip(np.round(pt[1]), 0, self.h - 1))
        return bool(mask[y, x])

    def _sample_rgb(self, image, pts):
        pts_tensor = torch.from_numpy(pts).float()
        pts_normed = normalize_coords(pts_tensor, self.h, self.w)[None, None]
        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1)[None]
        rgb = F.grid_sample(image_tensor, pts_normed, align_corners=True).squeeze(0).squeeze(1).T
        return rgb

    def _build_candidates(self, id1, id2):
        kp1 = self.keypoints[id1]
        kp2 = self.keypoints[id2]
        conf1 = self.confidences[id1]
        conf2 = self.confidences[id2]
        valid = self.valid[id1] & self.valid[id2] & (conf1 >= self.min_keypoint_conf) & (conf2 >= self.min_keypoint_conf)

        mask1 = self._load_mask(id1)
        mask2 = self._load_mask(id2)

        pts1 = []
        pts2 = []
        weights = []
        for joint_id in np.where(valid)[0]:
            base_weight = min(conf1[joint_id], conf2[joint_id]) * self.joint_weights[joint_id]
            for offset, offset_weight in zip(self.offsets, self.offset_weights):
                p1 = kp1[joint_id] + offset
                p2 = kp2[joint_id] + offset
                if not self._inside_image(p1) or not self._inside_image(p2):
                    continue
                if not self._mask_contains(mask1, p1) or not self._mask_contains(mask2, p2):
                    continue
                pts1.append(p1)
                pts2.append(p2)
                weights.append(base_weight * offset_weight)
        if not pts1:
            return None, None, None
        return np.asarray(pts1, dtype=np.float32), np.asarray(pts2, dtype=np.float32), np.asarray(weights, dtype=np.float32)

    def __getitem__(self, idx):
        id1, id2 = self._sample_pair(idx)
        frame_interval = max(abs(id1 - id2), 1)
        max_interval = max(min(self.max_interval.value, self.num_imgs - 1), 1)
        pair_weight = np.cos((frame_interval - 1.0) / max_interval * np.pi / 2)

        img1 = imageio.imread(self.img_files[id1]) / 255.0
        img2 = imageio.imread(self.img_files[id2]) / 255.0

        pts1_all, pts2_all, weight_all = self._build_candidates(id1, id2)
        invalid = pts1_all is None or len(pts1_all) == 0
        if invalid:
            pts1 = np.zeros((self.num_pts, 2), dtype=np.float32)
            pts2 = np.zeros((self.num_pts, 2), dtype=np.float32)
            sampled_weight = np.zeros((self.num_pts,), dtype=np.float32)
        else:
            prob = weight_all.copy()
            prob_sum = prob.sum()
            prob = None if prob_sum <= 0 else prob / prob_sum
            select_ids = np.random.choice(len(pts1_all), self.num_pts, replace=(len(pts1_all) < self.num_pts), p=prob)
            pts1 = pts1_all[select_ids]
            pts2 = pts2_all[select_ids]
            sampled_weight = weight_all[select_ids]
            sampled_weight /= np.maximum(sampled_weight.max(), 1e-6)

        pts1 = torch.from_numpy(pts1).float()
        pts2 = torch.from_numpy(pts2).float()
        gt_rgb1 = self._sample_rgb(img1, pts1.numpy())
        gt_rgb2 = self._sample_rgb(img2, pts2.numpy())
        weights = torch.from_numpy(sampled_weight[:, None] * pair_weight).float()
        covisible_mask = (weights > 0).float()

        if invalid:
            weights.zero_()
            covisible_mask.zero_()

        if np.random.choice([0, 1]):
            id1, id2, pts1, pts2, gt_rgb1, gt_rgb2 = id2, id1, pts2, pts1, gt_rgb2, gt_rgb1

        return {
            'ids1': id1,
            'ids2': id2,
            'pts1': pts1,
            'pts2': pts2,
            'gt_rgb1': gt_rgb1,
            'gt_rgb2': gt_rgb2,
            'weights': weights,
            'covisible_mask': covisible_mask,
        }
