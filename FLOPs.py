import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def _to_gflops(flops: float) -> float:
    return float(flops) / 1e9


def _to_mparams(params: int) -> float:
    return float(params) / 1e6


def _sum_flops_from_prof(prof: torch.profiler.profile) -> int:
    total = 0
    for event in prof.key_averages():
        if hasattr(event, "flops") and event.flops is not None:
            total += int(event.flops)
    return total


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total_params), int(trainable_params)


@torch.no_grad()
def profile_set_image_flops(
    predictor: SAM2ImagePredictor, image_np: np.ndarray, device: torch.device
) -> int:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(activities=activities, with_flops=True) as prof:
        predictor.set_image(image_np)
    return _sum_flops_from_prof(prof)


@torch.no_grad()
def profile_one_point_predict_flops(
    predictor: SAM2ImagePredictor,
    x: float,
    y: float,
    device: torch.device,
) -> int:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    point_coords = np.array([[x, y]], dtype=np.float32)
    point_labels = np.array([1], dtype=np.int32)

    with torch.profiler.profile(activities=activities, with_flops=True) as prof:
        predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )
    return _sum_flops_from_prof(prof)


def estimate_total_flops(
    set_image_flops: int,
    one_point_flops: int,
    n_per_side_list: tuple[int, ...],
) -> int:
    num_points = sum(n * n for n in n_per_side_list)
    return int(set_image_flops + num_points * one_point_flops)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate FLOPs + parameters for the SAM2 hint-generation pipeline."
    )
    parser.add_argument(
        "--image",
        default="/NAS_data/yjy/Parallel-GAN-main/Parallel-GAN-main/datasets/sar2opt/testB/11_1200_480.jpg",
        help="Input image path.",
    )
    parser.add_argument(
        "--sam2_checkpoint",
        default="/data/yjy_data/SAM2/checkpoints/sam2.1_hiera_large.pt",
        help="Path to the SAM2 checkpoint (e.g., sam2.1_hiera_large.pt).",
    )
    parser.add_argument(
        "--model_cfg",
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
        help="Path to SAM2 model config (e.g., configs/sam2.1/sam2.1_hiera_l.yaml).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device, e.g. cuda / cuda:0 / cpu.",
    )
    parser.add_argument(
        "--n_per_side_list",
        type=int,
        nargs="+",
        default=[16],
        help="n_per_side values used in your loop, e.g. --n_per_side_list 48 64.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    h, w = image_np.shape[:2]

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print("Building SAM2 model...")
    sam2_model = build_sam2(args.model_cfg, args.sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    total_params, trainable_params = count_parameters(sam2_model)

    print("Profiling set_image FLOPs...")
    set_image_flops = profile_set_image_flops(predictor, image_np, device)

    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    print("Profiling one-point predict FLOPs...")
    one_point_flops = profile_one_point_predict_flops(predictor, cx, cy, device)

    total_flops = estimate_total_flops(
        set_image_flops=set_image_flops,
        one_point_flops=one_point_flops,
        n_per_side_list=tuple(args.n_per_side_list),
    )
    num_points = sum(n * n for n in args.n_per_side_list)

    print("\n===== Compute Estimation =====")
    print(f"Image size                     : {h}x{w}")
    print(f"n_per_side_list               : {args.n_per_side_list}")
    print(f"Total point prompts           : {num_points}")
    print(
        f"Total parameters              : {total_params:,} ({_to_mparams(total_params):.3f} M)"
    )
    print(
        f"Trainable parameters          : {trainable_params:,} ({_to_mparams(trainable_params):.3f} M)"
    )
    print(
        f"set_image FLOPs               : {set_image_flops:,} ({_to_gflops(set_image_flops):.3f} GFLOPs)"
    )
    print(
        f"one-point predict FLOPs       : {one_point_flops:,} ({_to_gflops(one_point_flops):.3f} GFLOPs)"
    )
    print(
        f"Estimated total FLOPs         : {total_flops:,} ({_to_gflops(total_flops):.3f} GFLOPs)"
    )
    print("\nNote: This estimates the dominant SAM2 compute only.")
    print("      NumPy/OpenCV post-processing FLOPs are not included.")


if __name__ == "__main__":
    main()