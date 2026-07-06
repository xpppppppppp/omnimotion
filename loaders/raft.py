import os
import glob
import json
import imageio
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import multiprocessing as mp
from util import normalize_coords, gen_grid_np, load_keypoints, load_query_points


def get_sample_weights(flow_stats):
    sample_weights = {}
    for k in flow_stats.keys():
        sample_weights[k] = {}
        total_num = np.array(list(flow_stats[k].values())).sum()
        for j in flow_stats[k].keys():
            sample_weights[k][j] = 1. * flow_stats[k][j] / total_num
    return sample_weights


class RAFTExhaustiveDataset(Dataset):
    def __init__(self, args, max_interval=None):
        self.args = args
        self.seq_dir = args.data_dir
        self.seq_name = os.path.basename(self.seq_dir.rstrip('/'))
        self.img_dir = os.path.join(self.seq_dir, 'color')
        self.flow_dir = os.path.join(self.seq_dir, 'raft_exhaustive')
        img_names = sorted(os.listdir(self.img_dir))
        self.num_imgs = min(self.args.num_imgs, len(img_names))
        self.img_names = img_names[:self.num_imgs]

        h, w, _ = imageio.imread(os.path.join(self.img_dir, img_names[0])).shape
        self.h, self.w = h, w
        max_interval = self.num_imgs - 1 if not max_interval else max_interval
        self.max_interval = mp.Value('i', max_interval)
        self.num_pts = self.args.num_pts
        self.grid = gen_grid_np(self.h, self.w)
        flow_stats = json.load(open(os.path.join(self.seq_dir, 'flow_stats.json')))
        self.sample_weights = get_sample_weights(flow_stats)

        self.train_use_mask = bool(getattr(self.args, 'train_use_mask', False))
        self.train_mask_erosion = max(int(getattr(self.args, 'train_mask_erosion', 0)), 0)
        self.train_keypoint_bias = bool(getattr(self.args, 'train_keypoint_bias', False))
        self.train_keypoint_radius = max(int(getattr(self.args, 'train_keypoint_radius', 24)), 1)
        self.train_keypoint_focus_ratio = float(np.clip(getattr(self.args, 'train_keypoint_focus_ratio', 0.75), 0.0, 1.0))
        self.train_query_frame_only = bool(getattr(self.args, 'train_query_frame_only', False))
        self.train_query_frame_prob = float(np.clip(getattr(self.args, 'train_query_frame_prob', 0.5), 0.0, 1.0))
        self.query_frame_id = int(np.clip(getattr(self.args, 'query_frame_id', 0), 0, self.num_imgs - 1))
        self.mask_dir = os.path.join(self.seq_dir, 'mask')
        self.fg_masks = None
        if self.train_use_mask and os.path.isdir(self.mask_dir):
            self.fg_masks = []
            kernel = None
            if self.train_mask_erosion > 0:
                kernel = np.ones((self.train_mask_erosion, self.train_mask_erosion), np.uint8)
            for img_name in self.img_names:
                mask_path = os.path.join(self.mask_dir, img_name.replace('.jpg', '.png'))
                if os.path.exists(mask_path):
                    fg_mask = imageio.imread(mask_path)
                    if fg_mask.ndim == 3:
                        fg_mask = fg_mask[..., :3].sum(axis=-1) > 0
                    else:
                        fg_mask = fg_mask > 0
                else:
                    fg_mask = np.ones((self.h, self.w), dtype=bool)
                fg_mask = fg_mask.astype(np.uint8)
                if kernel is not None:
                    fg_mask = cv2.erode(fg_mask, kernel, iterations=1)
                self.fg_masks.append(fg_mask > 0)

        self.train_keypoint_track_dir = getattr(self.args, 'train_keypoint_track_dir', '') or getattr(self.args, 'keypoint_dir', '')
        self.keypoint_region_masks = None
        self.has_dynamic_keypoint_tracks = False
        if self.train_keypoint_bias and os.path.isdir(self.train_keypoint_track_dir):
            self.keypoint_region_masks = []
            for frame_idx, img_name in enumerate(self.img_names):
                kp_path = os.path.join(self.train_keypoint_track_dir, os.path.splitext(img_name)[0] + '.json')
                region_mask = np.zeros((self.h, self.w), dtype=np.uint8)
                if os.path.exists(kp_path):
                    pts, _, valid = load_keypoints(kp_path,
                                                   num_joints=getattr(self.args, 'num_joints', 17),
                                                   keypoint_format=getattr(self.args, 'keypoint_format', 'auto'),
                                                   min_conf=getattr(self.args, 'min_keypoint_conf', 0.0))
                    for pt, is_valid in zip(pts, valid):
                        if not is_valid:
                            continue
                        center = (int(round(pt[0])), int(round(pt[1])))
                        cv2.circle(region_mask, center, self.train_keypoint_radius, 1, -1)
                region_mask = region_mask > 0
                if self.fg_masks is not None:
                    region_mask &= self.fg_masks[frame_idx]
                self.keypoint_region_masks.append(region_mask)
            self.has_dynamic_keypoint_tracks = any(mask.sum() > 0 for mask in self.keypoint_region_masks)

        self.query_keypoint_path = getattr(self.args, 'query_keypoint_path', '')
        self.query_region_mask = None
        if self.train_keypoint_bias and not self.has_dynamic_keypoint_tracks and self.query_keypoint_path and os.path.exists(self.query_keypoint_path):
            query_pts, _ = load_query_points(self.query_keypoint_path,
                                             num_joints=getattr(self.args, 'num_joints', 17),
                                             keypoint_format=getattr(self.args, 'keypoint_format', 'auto'),
                                             min_conf=getattr(self.args, 'min_keypoint_conf', 0.0))
            if len(query_pts) > 0:
                region_mask = np.zeros((self.h, self.w), dtype=np.uint8)
                for pt in query_pts:
                    center = (int(round(pt[0])), int(round(pt[1])))
                    cv2.circle(region_mask, center, self.train_keypoint_radius, 1, -1)
                region_mask = region_mask > 0
                if self.fg_masks is not None:
                    region_mask &= self.fg_masks[self.query_frame_id]
                self.query_region_mask = region_mask

    def _sample_from_candidates(self, candidate_ids, num_samples, prob=None):
        if len(candidate_ids) == 0 or num_samples <= 0:
            return np.array([], dtype=np.int64)
        replace = len(candidate_ids) < num_samples
        if prob is not None:
            prob = np.asarray(prob, dtype=np.float64)
            prob_sum = prob.sum()
            prob = None if prob_sum <= 0 else prob / prob_sum
        return np.random.choice(candidate_ids, num_samples, replace=replace, p=prob)

    def _sample_training_ids(self, masked_count, preferred_ids=None, base_prob=None):
        all_ids = np.arange(masked_count)
        if preferred_ids is None or len(preferred_ids) == 0 or self.train_keypoint_focus_ratio <= 0:
            return self._sample_from_candidates(all_ids, self.num_pts, base_prob)

        num_focus = int(round(self.num_pts * self.train_keypoint_focus_ratio))
        num_focus = min(max(num_focus, 1), self.num_pts)
        focus_prob = None if base_prob is None else base_prob[preferred_ids]
        focus_ids = self._sample_from_candidates(preferred_ids, num_focus, focus_prob)

        num_rest = self.num_pts - len(focus_ids)
        if num_rest <= 0:
            return focus_ids

        rest_ids = self._sample_from_candidates(all_ids, num_rest, base_prob)
        return np.concatenate([focus_ids, rest_ids], axis=0)

    def _get_keypoint_preferred_ids(self, id1, mask):
        if self.has_dynamic_keypoint_tracks and self.keypoint_region_masks is not None:
            preferred = self.keypoint_region_masks[id1] & mask
            if preferred.sum() > 0:
                return np.flatnonzero(preferred[mask])
        if self.query_region_mask is None or id1 != self.query_frame_id:
            return None
        preferred = self.query_region_mask & mask
        if preferred.sum() == 0:
            return None
        return np.flatnonzero(preferred[mask])

    def _get_training_mask(self, id1, id2, coord2):
        if self.fg_masks is None:
            return None

        src_mask = self.fg_masks[id1].copy()
        tgt_mask = self.fg_masks[id2]

        x2 = np.round(coord2[..., 0]).astype(np.int32)
        y2 = np.round(coord2[..., 1]).astype(np.int32)
        in_range = (x2 >= 0) & (x2 < self.w) & (y2 >= 0) & (y2 < self.h)
        tgt_valid = np.zeros_like(src_mask, dtype=bool)
        tgt_valid[in_range] = tgt_mask[y2[in_range], x2[in_range]]
        return src_mask & in_range & tgt_valid

    def __len__(self):
        return self.num_imgs * 100000

    def set_max_interval(self, max_interval):
        self.max_interval.value = min(max_interval, self.num_imgs - 1)

    def increase_max_interval_by(self, increment):
        curr_max_interval = self.max_interval.value
        self.max_interval.value = min(curr_max_interval + increment, self.num_imgs - 1)

    def __getitem__(self, idx):
        cached_flow_pred_dir = os.path.join('out', '{}_{}'.format(self.args.expname, self.seq_name), 'flow')
        cached_flow_pred_files = sorted(glob.glob(os.path.join(cached_flow_pred_dir, '*')))
        flow_error_file = os.path.join(os.path.dirname(cached_flow_pred_dir), 'flow_error.txt')
        if self.train_query_frame_only and (not self.has_dynamic_keypoint_tracks) and self.query_region_mask is not None:
            id1 = self.query_frame_id
        elif self.train_keypoint_bias and (not self.has_dynamic_keypoint_tracks) and self.query_region_mask is not None and np.random.rand() < self.train_query_frame_prob:
            id1 = self.query_frame_id
        elif os.path.exists(flow_error_file):
            flow_error = np.loadtxt(flow_error_file)
            id1_sample_weights = flow_error / np.sum(flow_error)
            id1 = np.random.choice(self.num_imgs, p=id1_sample_weights)
        else:
            id1 = idx % self.num_imgs

        img_name1 = self.img_names[id1]
        max_interval = min(self.max_interval.value, self.num_imgs - 1)
        img2_candidates = sorted(list(self.sample_weights[img_name1].keys()))
        img2_candidates = img2_candidates[max(id1 - max_interval, 0):min(id1 + max_interval, self.num_imgs - 1)]

        # sample more often from i-1 and i+1
        id2s = np.array([self.img_names.index(n) for n in img2_candidates])
        sample_weights = np.array([self.sample_weights[img_name1][i] for i in img2_candidates])
        sample_weights /= np.sum(sample_weights)
        sample_weights[np.abs(id2s - id1) <= 1] = 0.5
        sample_weights /= np.sum(sample_weights)

        img_name2 = np.random.choice(img2_candidates, p=sample_weights)
        id2 = self.img_names.index(img_name2)
        frame_interval = abs(id1 - id2)

        # read image, flow and confidence
        img1 = imageio.imread(os.path.join(self.img_dir, img_name1)) / 255.
        img2 = imageio.imread(os.path.join(self.img_dir, img_name2)) / 255.

        flow_file = os.path.join(self.flow_dir, '{}_{}.npy'.format(img_name1, img_name2))
        flow = np.load(flow_file)
        mask_file = flow_file.replace('raft_exhaustive', 'raft_masks').replace('.npy', '.png')
        masks = imageio.imread(mask_file) / 255.

        coord1 = self.grid
        coord2 = self.grid + flow

        cycle_consistency_mask = masks[..., 0] > 0
        occlusion_mask = masks[..., 1] > 0

        if frame_interval == 1:
            mask = np.ones_like(cycle_consistency_mask)
        else:
            mask = cycle_consistency_mask | occlusion_mask

        if self.train_use_mask and self.fg_masks is not None:
            fg_mask = self._get_training_mask(id1, id2, coord2)
            masked = mask & fg_mask
            if masked.sum() > 0:
                mask = masked
            elif fg_mask.sum() > 0:
                mask = fg_mask

        if mask.sum() == 0:
            invalid = True
            mask = np.ones_like(cycle_consistency_mask)
        else:
            invalid = False

        preferred_ids = self._get_keypoint_preferred_ids(id1, mask)
        base_prob = None
        if len(cached_flow_pred_files) > 0 and self.args.use_error_map:
            cached_flow_pred_file = cached_flow_pred_files[id1]
            assert img_name1 + '_' in cached_flow_pred_file
            sup_flow_file = os.path.join(self.flow_dir, os.path.basename(cached_flow_pred_file))
            pred_flow = np.load(cached_flow_pred_file)
            sup_flow = np.load(sup_flow_file)
            error_map = np.linalg.norm(pred_flow - sup_flow, axis=-1)
            error_map = cv2.GaussianBlur(error_map, (5, 5), 0)
            base_prob = error_map[mask]
        elif self.args.use_count_map:
            count_map = imageio.imread(os.path.join(self.seq_dir, 'count_maps', img_name1.replace('.jpg', '.png')))
            base_prob = 1 / np.sqrt(count_map + 1.)
            base_prob = base_prob[mask]

        select_ids = self._sample_training_ids(int(mask.sum()), preferred_ids=preferred_ids, base_prob=base_prob)

        pair_weight = np.cos((frame_interval - 1.) / max_interval * np.pi / 2)

        pts1 = torch.from_numpy(coord1[mask][select_ids]).float()
        pts2 = torch.from_numpy(coord2[mask][select_ids]).float()
        pts2_normed = normalize_coords(pts2, self.h, self.w)[None, None]

        covisible_mask = torch.from_numpy(cycle_consistency_mask[mask][select_ids]).float()[..., None]
        weights = torch.ones_like(covisible_mask) * pair_weight

        gt_rgb1 = torch.from_numpy(img1[mask][select_ids]).float()
        gt_rgb2 = F.grid_sample(torch.from_numpy(img2).float().permute(2, 0, 1)[None], pts2_normed,
                                align_corners=True).squeeze().T

        if invalid:
            weights = torch.zeros_like(weights)

        if np.random.choice([0, 1]):
            id1, id2, pts1, pts2, gt_rgb1, gt_rgb2 = id2, id1, pts2, pts1, gt_rgb2, gt_rgb1
            weights[covisible_mask == 0.] = 0

        data = {'ids1': id1,
                'ids2': id2,
                'pts1': pts1,  # [n_pts, 2]
                'pts2': pts2,  # [n_pts, 2]
                'gt_rgb1': gt_rgb1,  # [n_pts, 3]
                'gt_rgb2': gt_rgb2,
                'weights': weights,  # [n_pts, 1]
                'covisible_mask': covisible_mask,  # [n_pts, 1]
                }
        return data