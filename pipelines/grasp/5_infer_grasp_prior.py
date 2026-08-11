#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as scio
import torch
from PIL import Image

from common.config import get_default_camera_to_base_path


def resolve_graspnet_root(project_root: Path, graspnet_root_argument: str | None) -> Path:
    if graspnet_root_argument:
        return Path(graspnet_root_argument).expanduser().resolve()
    return (project_root / "third-party" / "graspnet-baseline").resolve()


def append_graspnet_paths(graspnet_root: Path) -> None:
    sys.path.append(str(graspnet_root / "models"))
    sys.path.append(str(graspnet_root / "dataset"))
    sys.path.append(str(graspnet_root / "utils"))


def load_transform_json(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    transform_matrix = np.asarray(payload["T_cam2base"], dtype=float)
    if transform_matrix.shape != (4, 4):
        raise ValueError(f"T_cam2base must be shape (4, 4), got {transform_matrix.shape}")
    return transform_matrix


def load_demo_inputs(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    color = np.array(Image.open(data_dir / "color.png"), dtype=np.float32) / 255.0
    depth = np.array(Image.open(data_dir / "depth.png"))
    workspace_mask = np.array(Image.open(data_dir / "workspace_mask.png"))
    meta = scio.loadmat(str(data_dir / "meta.mat"))
    return color, depth, workspace_mask, meta


def build_end_points(
    color: np.ndarray,
    depth: np.ndarray,
    workspace_mask: np.ndarray,
    meta: dict[str, Any],
    num_points: int,
) -> tuple[dict[str, Any], np.ndarray]:
    from data_utils import CameraInfo, create_point_cloud_from_depth_image

    intrinsic = meta["intrinsic_matrix"]
    factor_depth = float(meta["factor_depth"])
    camera = CameraInfo(
        1280.0,
        720.0,
        intrinsic[0][0],
        intrinsic[1][1],
        intrinsic[0][2],
        intrinsic[1][2],
        factor_depth,
    )
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    valid_mask = (workspace_mask > 0) & (depth > 0)
    cloud_masked = cloud[valid_mask]
    color_masked = color[valid_mask]
    if len(cloud_masked) == 0:
        raise ValueError("No valid points from workspace_mask and depth.")

    if len(cloud_masked) >= num_points:
        sampled_indices = np.random.choice(len(cloud_masked), num_points, replace=False)
    else:
        sampled_indices_first = np.arange(len(cloud_masked))
        sampled_indices_second = np.random.choice(
            len(cloud_masked),
            num_points - len(cloud_masked),
            replace=True,
        )
        sampled_indices = np.concatenate([sampled_indices_first, sampled_indices_second], axis=0)

    cloud_sampled = np.ascontiguousarray(cloud_masked[sampled_indices], dtype=np.float32)
    color_sampled = color_masked[sampled_indices]

    end_points: dict[str, Any] = {}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    point_cloud_tensor = torch.from_numpy(cloud_sampled[np.newaxis]).to(device).contiguous()
    end_points["point_clouds"] = point_cloud_tensor
    end_points["cloud_colors"] = color_sampled
    return end_points, cloud_masked


def get_model(checkpoint_path: str, num_view: int):
    from graspnet import GraspNet

    model = GraspNet(
        input_feature_dim=0,
        num_view=num_view,
        num_angle=12,
        num_depth=4,
        cylinder_radius=0.05,
        hmin=-0.02,
        hmax_list=[0.01, 0.02, 0.03, 0.04],
        is_training=False,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def infer_grasp_array(model, end_points: dict[str, Any], debug: bool = False) -> np.ndarray:
    from graspnet import pred_decode

    with torch.no_grad():
        end_points["point_clouds"] = end_points["point_clouds"].contiguous()
        if debug:
            point_cloud_tensor = end_points["point_clouds"]
            print(
                "[grasp] point_clouds shape={} dtype={} device={} contiguous={}".format(
                    tuple(point_cloud_tensor.shape),
                    point_cloud_tensor.dtype,
                    point_cloud_tensor.device,
                    point_cloud_tensor.is_contiguous(),
                )
            )
        end_points = model(end_points)
        grasp_predictions = pred_decode(end_points)
    return grasp_predictions[0].detach().cpu().numpy()


def topk_by_score(grasp_array: np.ndarray, top_k: int) -> np.ndarray:
    if grasp_array.ndim != 2 or grasp_array.shape[1] < 16:
        raise ValueError(f"Unexpected grasp prediction shape: {grasp_array.shape}")
    if grasp_array.shape[0] == 0:
        return grasp_array
    top_k = max(1, min(int(top_k), grasp_array.shape[0]))
    sort_indices = np.argsort(-grasp_array[:, 0])
    return grasp_array[sort_indices[:top_k]]


def convert_top_grasp_to_payload(grasp_array_row: np.ndarray, camera_to_base_transform: np.ndarray) -> dict[str, Any]:
    score_value = float(grasp_array_row[0])
    width_value = float(grasp_array_row[1])
    height_value = float(grasp_array_row[2])
    depth_value = float(grasp_array_row[3])
    rotation_camera = grasp_array_row[4:13].reshape(3, 3)
    translation_camera = grasp_array_row[13:16].reshape(3)

    rotation_base = camera_to_base_transform[:3, :3] @ rotation_camera
    translation_base = (
        camera_to_base_transform[:3, :3] @ translation_camera
        + camera_to_base_transform[:3, 3]
    )

    return {
        "score": score_value,
        "width": width_value,
        "height": height_value,
        "depth": depth_value,
        "rotation_cam_3x3": rotation_camera.tolist(),
        "translation_cam_xyz": translation_camera.tolist(),
        "rotation_base_3x3": rotation_base.tolist(),
        "translation_base_xyz": translation_base.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer grasp prior from graspnet-baseline demo-format data.")
    parser.add_argument("--data-dir", required=True, help="folder containing color.png/depth.png/workspace_mask.png/meta.mat")
    parser.add_argument("--checkpoint-path", required=True, help="graspnet-baseline checkpoint path")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--graspnet-root", default=None, help="optional graspnet-baseline root path")
    parser.add_argument(
        "--transform-json",
        default=str(get_default_camera_to_base_path()),
        help="camera-to-base transform JSON path",
    )
    parser.add_argument("--num-point", type=int, default=20000)
    parser.add_argument("--num-view", type=int, default=300)
    parser.add_argument("--collision-thresh", type=float, default=0.01)
    parser.add_argument("--voxel-size", type=float, default=0.01)
    parser.add_argument("--top-k", type=int, default=20, help="sort and keep top-k candidates before taking top-1")
    parser.add_argument("--debug", action="store_true", help="print tensor/layout debug info")
    args = parser.parse_args()

    if args.debug:
        os.environ["GRASPNET_DEBUG_CONTIGUOUS"] = "1"

    project_root = Path(__file__).resolve().parent.parent.parent
    graspnet_root = resolve_graspnet_root(project_root, args.graspnet_root)
    append_graspnet_paths(graspnet_root)

    data_dir = Path(args.data_dir).expanduser().resolve()
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    camera_to_base_transform = load_transform_json(args.transform_json)
    color, depth, workspace_mask, meta = load_demo_inputs(data_dir)
    end_points, cloud_points = build_end_points(
        color=color,
        depth=depth,
        workspace_mask=workspace_mask,
        meta=meta,
        num_points=args.num_point,
    )
    model = get_model(args.checkpoint_path, num_view=args.num_view)
    grasp_array = infer_grasp_array(model, end_points, debug=args.debug)
    grasp_array = topk_by_score(grasp_array, top_k=args.top_k)
    if grasp_array.shape[0] == 0:
        raise RuntimeError("No valid grasp predictions.")

    top_grasp_payload = convert_top_grasp_to_payload(grasp_array[0], camera_to_base_transform)
    payload = {
        "source": "graspnet-baseline",
        "data_dir": str(data_dir),
        "checkpoint_path": str(Path(args.checkpoint_path).expanduser().resolve()),
        "candidate_count_after_filter": int(grasp_array.shape[0]),
        "top_grasp": top_grasp_payload,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved grasp prior JSON: {output_path}")
    print(f"[INFO] Top grasp score: {top_grasp_payload['score']:.4f}")


if __name__ == "__main__":
    main()
