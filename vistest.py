import os
import sys
import argparse
import imageio
import glob
import numpy as np
import torch
import cv2
import json
import util

from config import config_parser
from trainer import BaseTrainer
from matplotlib import cm
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

color_map = cm.get_cmap("jet")


def load_training_args(ckpt_path):
    ckpt_dir = os.path.dirname(ckpt_path)
    args_path = os.path.join(ckpt_dir, 'args.txt')
    parsed = {}
    if not os.path.exists(args_path):
        return parsed
    with open(args_path, 'r') as f:
        for line in f:
            key, sep, value = line.partition(' = ')
            if not sep:
                continue
            parsed[key.strip()] = value.strip()
    return parsed


def parse_bool(value, default=False):
    if value is None:
        return default
    return str(value).lower() in {'1', 'true', 'yes'}


def vis_single_point_trail(images, kpts, save_path, fps=10, point_radius=3, line_thickness=3, point_color=(0, 255, 0)):
    num_imgs, num_pts = kpts.shape[:2]
    frames = []

    for i in range(num_imgs):
        img_curr = images[i].copy()

        # Only draw points on current frame, no trail lines
        for j in range(num_pts):
            pt = kpts[i, j]
            p = (int(round(pt[0])), int(round(pt[1])))
            cv2.circle(img_curr, p, point_radius, point_color, -1, lineType=16)

        frames.append(img_curr)

    imageio.mimwrite(save_path, frames, quality=8, fps=fps)
    print("Video saved to " + save_path)


def save_points_to_json(kpts, output_dir, img_files):
    num_imgs, num_pts = kpts.shape[:2]
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num_imgs):
        shapes = []
        for j in range(num_pts):
            pt = kpts[i, j]
            shape = {
                "label": str(j),
                "points": [[float(pt[0]), float(pt[1])]],
                "group_id": None,
                "description": "",
                "shape_type": "point",
                "flags": {},
                "mask": None
            }
            shapes.append(shape)
        
        json_data = {
            "version": "5.6.1",
            "flags": {},
            "shapes": shapes
        }
        
        # Use image filename (without extension) for JSON
        img_path = img_files[i]
        img_name = os.path.basename(img_path)
        json_name = os.path.splitext(img_name)[0] + ".json"
        json_path = os.path.join(output_dir, json_name)
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
    
    print(f"Saved {num_imgs} JSON files to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Visualize point tracking from .pth checkpoint")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to .pth checkpoint file")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to video sequence directory")
    parser.add_argument("--points", type=str, required=True,
                        help="Points to track: x1,y1 x2,y2, N to sample N random points, or keypoints to use per-frame pose joints")
    parser.add_argument("--query_frame_id", type=int, default=0, help="Query frame ID")
    parser.add_argument("--output_dir", type=str, default="/mnt/home/caixiang/carMove-pose/json_omni", help="Output directory for JSON files")
    parser.add_argument("--keypoint_dir", type=str, default="", help="Optional keypoint json directory")
    parser.add_argument("--query_keypoint_path", type=str, default="", help="Optional keypoint json file for the query frame only")
    parser.add_argument("--rolling_query", action="store_true", help="Use frame-to-frame propagation instead of direct source-to-target querying")
    parser.add_argument("--use_point_head", action="store_true", help="Use the trained point head for point tracking")
    parser.add_argument("--use_max_loc", action="store_true", help="Use max location")
    parser.add_argument("--vis_occlusion", action="store_true", help="Visualize occlusion")
    parser.add_argument("--occlusion_th", type=float, default=0.99, help="Occlusion threshold")
    parser.add_argument("--fps", type=int, default=10, help="Output video fps")
    parser.add_argument("--trail_point_radius", type=int, default=3, help="Point radius for trail visualization")
    parser.add_argument("--trail_line_thickness", type=int, default=3, help="Line thickness for trail visualization")
    parser.add_argument("--trail_color", type=str, default="red", help="Color for trail (green, red, blue, yellow)")
    parser.add_argument("--corr_radius", type=int, default=6, help="Point radius for correspondence visualization")
    parser.add_argument("--save_video", action="store_true", help="Also save video output")

    args = parser.parse_args()

    # Parse color
    color_map = {
        "green": (0, 255, 0),
        "red": (0, 0, 255),
        "blue": (255, 0, 0),
        "yellow": (0, 255, 255),
    }
    trail_color = color_map.get(args.trail_color.lower(), (0, 255, 0))

    # Parse points
    use_keypoints = args.points.lower() in {"keypoints", "pose", "skeleton"}
    if use_keypoints:
        num_pts = 0
        pts = None
        print("Will query tracked pose keypoints")
    elif args.points.replace("-", "").replace(".", "").replace(" ", "").isdigit():
        num_pts = int(args.points)
        pts = None
        print("Will sample " + str(num_pts) + " random points")
    else:
        point_strs = args.points.split()
        pts = []
        for ps in point_strs:
            coords = ps.split(",")
            if len(coords) == 2:
                x, y = float(coords[0]), float(coords[1])
                pts.append([x, y])
        pts = np.array(pts)
        num_pts = len(pts)
        print("Tracking " + str(num_pts) + " points: " + str(pts))

    class SimpleArgs:
        def __init__(self):
            self.data_dir = args.data_dir
            self.expname = "test"
            self.save_dir = "out/"
            self.ckpt_path = args.ckpt_path
            self.no_reload = False
            self.distributed = 0
            self.local_rank = 0
            self.load_opt = 0
            self.load_scheduler = 0
            self.lr_feature = 1e-3
            self.lr_deform = 1e-4
            self.lr_color = 3e-4
            self.lr_point = 1e-4
            self.lrate_decay_steps = 20000
            self.lrate_decay_factor = 0.5
            self.pe_freq = 4
            self.use_affine = True
            self.use_point_head = args.use_point_head
            self.use_point_head_for_query = args.use_point_head
            self.point_head_hidden = 256
            self.point_head_layers = 3
            self.point_head_residual = True
            self.point_delta_scale = 0.25
            self.point_use_rgb_patch = False
            self.point_rgb_patch_size = 5
            self.point_use_dino_feature = False
            self.point_dino_dir = ""
            self.point_dino_dim = 384
            self.point_dino_l2_normalize = False
            self.point_use_dino_correlation = False
            self.point_corr_radius = 12
            self.point_corr_stride = 2
            self.point_corr_temperature = 10.0
            self.point_corr_update_base = False
            self.point_loss_weight = 0.0
            self.point_conf_weight = 0.01
            self.point_num_pairs = 8
            self.point_max_interval = 0
            self.point_supervision = "flow"
            self.chunk_size = 40000
            self.use_max_loc = args.use_max_loc
            self.query_frame_id = args.query_frame_id
            self.vis_occlusion = args.vis_occlusion
            self.occlusion_th = args.occlusion_th
            self.num_imgs = 250
            self.num_samples_ray = 32
            self.min_depth = 0
            self.max_depth = 2
            self.grad_clip = 0
            self.mask_near = False
            self.max_padding = 0
            self.use_error_map = False
            self.foreground_mask_path = ""
            self.start_interval = 20
            self.keypoint_dir = args.keypoint_dir
            self.keypoint_format = "auto"
            self.num_joints = 17
            self.min_keypoint_conf = 0.5
            self.patch_size = 5
            self.foreground_hard_mask = False
            self.query_pts_source = "keypoints" if use_keypoints else "mask"
            self.query_keypoint_path = args.query_keypoint_path
            self.rolling_query = args.rolling_query or use_keypoints

    simple_args = SimpleArgs()

    train_args = load_training_args(args.ckpt_path)
    if 'use_affine' in train_args:
        simple_args.use_affine = parse_bool(train_args.get('use_affine'), simple_args.use_affine)
    if 'num_imgs' in train_args:
        simple_args.num_imgs = int(train_args['num_imgs'])
    if 'pe_freq' in train_args:
        simple_args.pe_freq = int(train_args['pe_freq'])
    if 'min_depth' in train_args:
        simple_args.min_depth = float(train_args['min_depth'])
    if 'max_depth' in train_args:
        simple_args.max_depth = float(train_args['max_depth'])
    if 'use_point_head' in train_args:
        simple_args.use_point_head = args.use_point_head or parse_bool(train_args.get('use_point_head'), False)
        simple_args.use_point_head_for_query = simple_args.use_point_head
    if 'point_head_hidden' in train_args:
        simple_args.point_head_hidden = int(train_args['point_head_hidden'])
    if 'point_head_layers' in train_args:
        simple_args.point_head_layers = int(train_args['point_head_layers'])
    if 'point_head_residual' in train_args:
        simple_args.point_head_residual = parse_bool(train_args['point_head_residual'], True)
    if 'point_delta_scale' in train_args:
        simple_args.point_delta_scale = float(train_args['point_delta_scale'])
    if 'point_use_rgb_patch' in train_args:
        simple_args.point_use_rgb_patch = parse_bool(train_args['point_use_rgb_patch'], False)
    if 'point_rgb_patch_size' in train_args:
        simple_args.point_rgb_patch_size = int(train_args['point_rgb_patch_size'])
    if 'point_use_dino_feature' in train_args:
        simple_args.point_use_dino_feature = parse_bool(train_args['point_use_dino_feature'], False)
    if 'point_dino_dir' in train_args:
        simple_args.point_dino_dir = train_args['point_dino_dir']
    if 'point_dino_dim' in train_args:
        simple_args.point_dino_dim = int(train_args['point_dino_dim'])
    if 'point_dino_l2_normalize' in train_args:
        simple_args.point_dino_l2_normalize = parse_bool(train_args['point_dino_l2_normalize'], False)
    if 'point_use_dino_correlation' in train_args:
        simple_args.point_use_dino_correlation = parse_bool(train_args['point_use_dino_correlation'], False)
    if 'point_corr_radius' in train_args:
        simple_args.point_corr_radius = int(train_args['point_corr_radius'])
    if 'point_corr_stride' in train_args:
        simple_args.point_corr_stride = int(train_args['point_corr_stride'])
    if 'point_corr_temperature' in train_args:
        simple_args.point_corr_temperature = float(train_args['point_corr_temperature'])
    if 'point_corr_update_base' in train_args:
        simple_args.point_corr_update_base = parse_bool(train_args['point_corr_update_base'], False)

    print("Loading checkpoint from " + args.ckpt_path)
    trainer = BaseTrainer(simple_args)
    print("Loaded model with " + str(trainer.num_imgs) + " images")

    # Get original images
    img_dir = os.path.join(args.data_dir, "color")
    img_files = sorted(list(glob.glob(os.path.join(img_dir, "*"))))
    images = np.array([imageio.imread(img_file) for img_file in img_files])
    print("Loaded " + str(len(images)) + " images")

    print("Running point tracking...")
    if use_keypoints and args.query_keypoint_path:
        pts, _ = util.load_query_points(args.query_keypoint_path,
                                        num_joints=simple_args.num_joints,
                                        keypoint_format=simple_args.keypoint_format,
                                        min_conf=simple_args.min_keypoint_conf)
        num_pts = len(pts)
        if num_pts == 0:
            raise ValueError("No valid query keypoints found in " + args.query_keypoint_path)

    if pts is not None:
        frames, kpts = trainer.eval_video_correspondences(
            args.query_frame_id,
            pts=pts,
            num_pts=num_pts,
            use_max_loc=args.use_max_loc,
            vis_occlusion=args.vis_occlusion,
            occlusion_th=args.occlusion_th,
            radius=args.corr_radius,
            return_kpts=True,
            rolling_query=simple_args.rolling_query
        )
    else:
        frames, kpts = trainer.eval_video_correspondences(
            args.query_frame_id,
            num_pts=num_pts,
            use_max_loc=args.use_max_loc,
            vis_occlusion=args.vis_occlusion,
            occlusion_th=args.occlusion_th,
            radius=args.corr_radius,
            return_kpts=True,
            use_keypoints=use_keypoints,
            rolling_query=simple_args.rolling_query
        )

    # Save JSON files with names matching image files
    kpts = kpts.cpu().numpy()
    save_points_to_json(kpts, args.output_dir, img_files)

    # Save correspondence video if requested
    if args.save_video:
        video_output = os.path.join(args.output_dir, "correspondence.mp4")
        imageio.mimwrite(video_output, frames, quality=8, fps=args.fps)
        print("Correspondence video saved to " + video_output)

        # Save point trail visualization (points only, no lines)
        trail_output = os.path.join(args.output_dir, "trails.mp4")
        vis_single_point_trail(images, kpts, trail_output, fps=args.fps, point_radius=args.trail_point_radius, line_thickness=args.trail_line_thickness, point_color=trail_color)


if __name__ == "__main__":
    main()
