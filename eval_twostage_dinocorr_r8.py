import argparse
import glob
import json
import os
import time

import numpy as np
from PIL import Image
import torch

from trainer import BaseTrainer


def load_labelme_points(path):
    with open(path, "r") as f:
        data = json.load(f)
    points = {}
    for shape in data.get("shapes", []):
        if shape.get("shape_type") == "point" and shape.get("points"):
            points[str(shape.get("label"))] = np.array(shape["points"][0], dtype=np.float32)
    return points


def load_query_points(path):
    point_dict = load_labelme_points(path)
    labels = sorted(point_dict.keys(), key=lambda x: int(x) if x.isdigit() else x)
    points = np.stack([point_dict[label] for label in labels], axis=0).astype(np.float32)
    return labels, points


def save_labelme_points(labels, points, image_path, output_path):
    image = Image.open(image_path)
    data = {
        "version": "5.0.1",
        "flags": {},
        "shapes": [],
        "imagePath": os.path.basename(image_path),
        "imageData": None,
        "imageHeight": image.height,
        "imageWidth": image.width,
    }
    for label, point in zip(labels, points):
        data["shapes"].append(
            {
                "label": str(label),
                "points": [[float(point[0]), float(point[1])]],
                "group_id": None,
                "shape_type": "point",
                "flags": {},
            }
        )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def compute_metrics(pred_dir, gt_dir):
    dists = []
    per_frame = []
    for gt_path in sorted(glob.glob(os.path.join(gt_dir, "*.json"))):
        name = os.path.basename(gt_path)
        pred_path = os.path.join(pred_dir, name)
        if not os.path.exists(pred_path):
            continue
        gt = load_labelme_points(gt_path)
        pred = load_labelme_points(pred_path)
        frame_dists = []
        for label, gt_point in gt.items():
            if label not in pred:
                continue
            dist = float(np.linalg.norm(pred[label] - gt_point))
            dists.append(dist)
            frame_dists.append(dist)
        if frame_dists:
            per_frame.append((name, float(np.mean(frame_dists)), float(np.median(frame_dists)), len(frame_dists)))

    arr = np.array(dists, dtype=np.float32)
    return {
        "num_points": int(arr.size),
        "mean_px": float(arr.mean()) if arr.size else float("nan"),
        "median_px": float(np.median(arr)) if arr.size else float("nan"),
        "pck5": float((arr <= 5).mean()) if arr.size else float("nan"),
        "pck10": float((arr <= 10).mean()) if arr.size else float("nan"),
        "pck20": float((arr <= 20).mean()) if arr.size else float("nan"),
        "per_frame": per_frame,
    }


def write_metrics(run_id, metrics, times, meta, bench_dir):
    os.makedirs(bench_dir, exist_ok=True)
    with open(os.path.join(bench_dir, f"{run_id}_metrics.txt"), "w") as f:
        f.write(f"run_id: {run_id}\n")
        for key in ["num_points", "mean_px", "median_px", "pck5", "pck10", "pck20"]:
            f.write(f"{key}: {metrics[key]}\n")
        f.write("\nper_frame_name mean_px median_px n\n")
        for name, mean_px, median_px, count in metrics["per_frame"]:
            f.write(f"{name} {mean_px:.6f} {median_px:.6f} {count}\n")

    with open(os.path.join(bench_dir, f"{run_id}_times.txt"), "w") as f:
        for key, value in times.items():
            f.write(f"{key}: {value}\n")

    with open(os.path.join(bench_dir, f"{run_id}_meta.txt"), "w") as f:
        for key, value in meta.items():
            f.write(f"{key}: {value}\n")


class Args:
    pass


def build_trainer_args(cli_args):
    args = Args()
    args.data_dir = cli_args.data_dir
    args.expname = "eval_twostage_dinocorr_r8"
    args.local_rank = 0
    args.save_dir = "out/"
    args.ckpt_path = cli_args.ckpt_path
    args.no_reload = False
    args.distributed = 0
    args.num_iters = 0
    args.num_workers = 0
    args.load_opt = 0
    args.load_scheduler = 0
    args.loader_seed = 12

    args.dataset_types = "flow"
    args.dataset_weights = [1.0]
    args.num_imgs = cli_args.num_imgs
    args.num_pairs = 1
    args.num_pts = 8

    args.keypoint_dir = ""
    args.keypoint_format = "auto"
    args.num_joints = 17
    args.min_keypoint_conf = 0.5
    args.patch_size = 5
    args.foreground_hard_mask = False
    args.query_pts_source = "keypoints"
    args.query_keypoint_path = cli_args.query_path
    args.rolling_query = cli_args.rolling

    args.lr_feature = 1e-3
    args.lr_deform = 1e-4
    args.lr_color = 3e-4
    args.lrate_decay_steps = 20000
    args.lrate_decay_factor = 0.5
    args.grad_clip = 0

    args.use_point_head = True
    args.use_point_head_for_query = True
    args.lr_point = 1e-4
    args.point_head_hidden = 256
    args.point_head_layers = 3
    args.point_head_residual = True
    args.point_delta_scale = 0.0
    args.point_use_rgb_patch = False
    args.point_rgb_patch_size = 5
    args.point_use_dino_feature = False
    args.point_dino_dir = cli_args.dino_dir
    args.point_dino_dim = 384
    args.point_dino_l2_normalize = True
    args.point_use_dino_correlation = True
    args.point_corr_radius = 8
    args.point_corr_stride = 2
    args.point_corr_temperature = 10.0
    args.point_corr_update_base = True
    args.point_loss_weight = 0.0
    args.point_conf_weight = 0.01
    args.point_num_pairs = 1
    args.point_max_interval = 0
    args.point_supervision = "flow"

    args.use_error_map = False
    args.use_count_map = False
    args.train_use_mask = False
    args.train_mask_erosion = 0
    args.train_keypoint_bias = False
    args.train_keypoint_track_dir = ""
    args.train_keypoint_radius = 24
    args.train_keypoint_focus_ratio = 0.75
    args.train_query_frame_only = False
    args.train_query_frame_prob = 0.5
    args.use_affine = False
    args.mask_near = False
    args.num_samples_ray = 32
    args.pe_freq = 4
    args.min_depth = 0
    args.max_depth = 2
    args.start_interval = 20
    args.max_padding = 0

    args.chunk_size = 40000
    args.use_max_loc = False
    args.query_frame_id = cli_args.query_frame_id
    args.vis_occlusion = False
    args.occlusion_th = 0.99
    args.foreground_mask_path = ""
    args.skip_checkpoint_visualization = True
    return args


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate A: two-stage OmniMotion checkpoint + DINO local correlation r=8."
    )
    parser.add_argument("--data_dir", default="/mnt/home/caixiang/4.12-carMove")
    parser.add_argument("--ckpt_path", default="out/twostage5000_bench_20260608_1435_4.12-carMove/model_005000.pth")
    parser.add_argument("--query_path", default="/mnt/home/caixiang/carMove-pose/json_gt/00410.json")
    parser.add_argument("--gt_dir", default="/mnt/home/caixiang/carMove-pose/json_gt")
    parser.add_argument("--output_dir", default="out/twostage_dinocorr_r8_A/json")
    parser.add_argument("--bench_dir", default="out/bench_logs")
    parser.add_argument("--run_id", default="twostage_dinocorr_r8_A")
    parser.add_argument("--query_frame_id", type=int, default=0)
    parser.add_argument("--num_imgs", type=int, default=250)
    parser.add_argument("--dino_dir", default="")
    parser.add_argument("--rolling", action="store_true")
    args = parser.parse_args()

    labels, query_points = load_query_points(args.query_path)
    image_files = sorted(glob.glob(os.path.join(args.data_dir, "color", "*.jpg")))
    if not image_files:
        image_files = sorted(glob.glob(os.path.join(args.data_dir, "color", "*.png")))

    trainer_args = build_trainer_args(args)
    trainer = BaseTrainer(trainer_args)

    start = time.time()
    with torch.no_grad():
        output = trainer.eval_video_correspondences(
            args.query_frame_id,
            pts=query_points,
            num_pts=len(labels),
            return_kpts=True,
            rolling_query=args.rolling,
        )
    if isinstance(output, tuple):
        output = output[1]
    pred = output.detach().cpu().numpy() if isinstance(output, torch.Tensor) else np.asarray(output)
    elapsed = time.time() - start

    if pred.ndim != 3 or pred.shape[-1] != 2:
        raise RuntimeError(f"unexpected prediction shape {pred.shape}")
    if pred.shape[0] == len(labels) and pred.shape[1] == len(image_files):
        pred = np.transpose(pred, (1, 0, 2))

    for idx, image_path in enumerate(image_files[: pred.shape[0]]):
        name = os.path.splitext(os.path.basename(image_path))[0] + ".json"
        save_labelme_points(labels, pred[idx], image_path, os.path.join(args.output_dir, name))

    metrics = compute_metrics(args.output_dir, args.gt_dir)
    write_metrics(
        args.run_id,
        metrics,
        {"additional_train_sec": 0.0, "eval_sec": elapsed, "train_plus_eval_sec": elapsed},
        {
            "method": "two-stage OmniMotion + DINO local correlation r=8",
            "checkpoint": args.ckpt_path,
            "supervision": "no additional training; GT is used only for final evaluation",
            "point_delta_scale": 0.0,
            "point_corr_radius": 8,
            "point_corr_stride": 2,
            "point_corr_temperature": 10.0,
            "point_corr_update_base": True,
            "rolling": args.rolling,
        },
        args.bench_dir,
    )

    print({k: metrics[k] for k in ["num_points", "mean_px", "median_px", "pck5", "pck10", "pck20"]})
    print({"eval_sec": elapsed, "output_dir": args.output_dir})


if __name__ == "__main__":
    main()
