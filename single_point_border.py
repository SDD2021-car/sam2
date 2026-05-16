import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def make_grid_points_torch(H, W, n_per_side=16, margin=1, device="cuda"):
    xs = torch.linspace(margin, W - 1 - margin, n_per_side, device=device)
    ys = torch.linspace(margin, H - 1 - margin, n_per_side, device=device)
    yv, xv = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xv.reshape(-1), yv.reshape(-1)], dim=1).float()


def segment_everything_with_predictor_gpu_post(
    predictor,
    H,
    W,
    n_per_side=16,
    score_thr=0.3,
    min_area=500,
    device="cuda",
):
    cand_masks = []
    cand_scores = []

    pts = make_grid_points_torch(H, W, n_per_side=n_per_side, margin=1, device=device)
    for p in pts:
        p_np = p.view(1, 2).detach().cpu().numpy().astype("float32")
        masks, scores, _ = predictor.predict(
            point_coords=p_np,
            point_labels=np.array([1], dtype=np.int32),
            multimask_output=True,
        )
        order = np.argsort(scores)[::-1]
        for k in order:
            s = float(scores[k])
            m = masks[k].astype(bool)
            if s >= score_thr and m.sum() >= min_area:
                cand_masks.append(m)
                cand_scores.append(s)
                break

    if not cand_masks:
        return np.zeros((H, W), dtype=np.uint16)

    masks = np.stack(cand_masks, axis=0)
    scores = np.array(cand_scores)
    areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
    priority = scores * np.power(np.clip(areas, 1, None), 0.7)
    order = np.argsort(priority)[::-1]

    label_map = np.zeros((H, W), dtype=np.uint16)
    cur_id = 1
    for idx in order:
        fill = masks[idx] & (label_map == 0)
        if fill.sum() < min_area:
            continue
        label_map[fill] = cur_id
        cur_id += 1
        if cur_id >= 65535:
            break
    return label_map


def histogram_peak_ratio(image, mask, bins=16):
    bins = max(1, min(int(bins), 256))
    pixels = image[mask]
    if pixels.size == 0:
        return 0.0
    step = max(1, 256 // bins)
    quant = (pixels // step).astype(np.int32)
    quant = np.clip(quant, 0, bins - 1)
    idx = quant[:, 0] * bins * bins + quant[:, 1] * bins + quant[:, 2]
    hist = np.bincount(idx, minlength=bins**3)
    total = hist.sum()
    if total == 0:
        return 0.0
    return float(hist.max()) / float(total)


def passes_histogram_threshold(image, dot_mask, histogram_bins=16, histogram_threshold=0.3):
    if image is None:
        return True
    peak_ratio = histogram_peak_ratio(image, dot_mask, bins=histogram_bins)
    return peak_ratio >= histogram_threshold


def _pick_dot_center(mask, candidate_ys, candidate_xs, rng, image_rgb, histogram_bins, histogram_threshold, require_histogram_pass):
    if len(candidate_ys) == 0:
        return None

    order = rng.permutation(len(candidate_ys))
    fallback = (int(candidate_ys[order[0]]), int(candidate_xs[order[0]]))

    for idx in order:
        cy, cx = int(candidate_ys[idx]), int(candidate_xs[idx])
        dot_tmp = np.zeros_like(mask, dtype=np.uint8)
        cv2.circle(dot_tmp, (cx, cy), 1, 1, thickness=-1)
        dot_tmp = dot_tmp.astype(bool) & (mask > 0)

        passed = passes_histogram_threshold(
            image_rgb,
            dot_tmp,
            histogram_bins=histogram_bins,
            histogram_threshold=histogram_threshold,
        )
        if passed == require_histogram_pass:
            return (cy, cx)

    return fallback


def build_dual_dots_with_boundary_constraint(
    image_rgb,
    label_map_u16,
    dot_radius=6,
    rng_seed=0,
    histogram_bins=16,
    histogram_threshold=0.3,
    require_histogram_pass=True,
):
    """生成两个点：一个满足边界距离约束，一个故意不满足边界距离约束。"""
    label_ids = np.unique(label_map_u16)
    label_ids = label_ids[label_ids != 0]
    if len(label_ids) == 0:
        h, w = label_map_u16.shape
        empty = np.zeros((h, w), dtype=bool)
        return {
            "far_dot": empty,
            "near_dot": empty,
            "far_center": None,
            "near_center": None,
            "instance_mask": empty,
        }

    best_id = max(label_ids, key=lambda i: int((label_map_u16 == i).sum()))
    mask = (label_map_u16 == best_id).astype(np.uint8)
    rng = np.random.default_rng(rng_seed)

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    ys_far, xs_far = np.where(dist >= float(dot_radius))
    ys_near, xs_near = np.where((mask > 0) & (dist < float(dot_radius)))

    if len(ys_far) == 0:
        ys_far, xs_far = np.where(mask > 0)
    if len(ys_near) == 0:
        ys_near, xs_near = np.where(mask > 0)

    far_center = _pick_dot_center(
        mask, ys_far, xs_far, rng, image_rgb, histogram_bins, histogram_threshold, require_histogram_pass
    )
    near_center = _pick_dot_center(
        mask, ys_near, xs_near, rng, image_rgb, histogram_bins, histogram_threshold, require_histogram_pass
    )

    far_dot = np.zeros_like(mask, dtype=np.uint8)
    near_dot = np.zeros_like(mask, dtype=np.uint8)

    if far_center is not None:
        fy, fx = far_center
        cv2.circle(far_dot, (fx, fy), dot_radius, 1, thickness=-1)
    if near_center is not None:
        ny, nx = near_center
        cv2.circle(near_dot, (nx, ny), dot_radius, 1, thickness=-1)

    far_dot = far_dot.astype(bool) & (mask > 0)
    near_dot = near_dot.astype(bool)
    # near_dot = near_dot.astype(bool) & (mask > 0)

    return {
        "far_dot": far_dot,
        "near_dot": near_dot,
        "far_center": far_center,
        "near_center": near_center,
        "instance_mask": mask.astype(bool),
    }


def validate_checkpoint_file(checkpoint_path):
    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"SAM2 checkpoint 不存在: {ckpt}")
    if not ckpt.is_file():
        raise ValueError(f"SAM2 checkpoint 不是文件: {ckpt}")
    if ckpt.stat().st_size == 0:
        raise ValueError(f"SAM2 checkpoint 文件为空: {ckpt}")


def save_dual_point_outputs(image, outputs, output_root, image_name, dot_radius=6):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    stem = Path(image_name).stem

    far_dot = outputs["far_dot"]
    near_dot = outputs["near_dot"]
    far_center = outputs["far_center"]
    near_center = outputs["near_center"]

    dual_cut = np.zeros_like(image)
    # dual_cut[far_dot] = image[far_dot]
    dual_cut[near_dot] = image[near_dot]
    Image.fromarray(dual_cut).save(output_root / f"{stem}_dual_point_cut.png")

    overlay = image.copy()
    if far_center is not None:
        fy, fx = far_center
        cv2.circle(overlay, (fx, fy), dot_radius, (0, 255, 0), thickness=2)
    if near_center is not None:
        ny, nx = near_center
        cv2.circle(overlay, (nx, ny), dot_radius, (255, 0, 0), thickness=2)
    Image.fromarray(overlay).save(output_root / f"{stem}_dual_point_overlay.png")

    combined = far_dot | near_dot
    ys, xs = np.where(combined)
    if len(ys) > 0:
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        crop = dual_cut[y0:y1, x0:x1]
        Image.fromarray(crop).save(output_root / f"{stem}_dual_point_crop.png")


def process_one_image(
    image_path,
    sam2_checkpoint,
    model_cfg,
    output_dir,
    device="cuda",
    rng_seed=0,
    histogram_bins=16,
    histogram_threshold=0.6,
    require_histogram_pass=True,
):
    image = np.array(Image.open(image_path).convert("RGB"))
    device = torch.device(device)
    validate_checkpoint_file(sam2_checkpoint)
    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    try:
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    except RuntimeError as err:
        msg = str(err)
        if "PytorchStreamReader failed reading zip archive" in msg:
            raise RuntimeError(
                "无法读取 SAM2 checkpoint（zip central directory 错误）。"
                "这通常表示 checkpoint 文件损坏或下载不完整，请重新下载该 .pt 文件后重试。"
                f"\ncheckpoint: {sam2_checkpoint}"
            ) from err
        raise

    predictor = SAM2ImagePredictor(sam2_model)
    predictor.set_image(image)

    h, w = image.shape[:2]
    label_map = segment_everything_with_predictor_gpu_post(
        predictor,
        H=h,
        W=w,
        n_per_side=16,
        score_thr=0.3,
        min_area=500,
        device=device.type,
    )

    outputs = build_dual_dots_with_boundary_constraint(
        image,
        label_map,
        dot_radius=4,
        rng_seed=rng_seed,
        histogram_bins=histogram_bins,
        histogram_threshold=histogram_threshold,
        require_histogram_pass=require_histogram_pass,
    )
    save_dual_point_outputs(
        image,
        outputs,
        output_root=output_dir,
        image_name=Path(image_path).name,
        dot_radius=6,
    )


def main():
    parser = argparse.ArgumentParser(
        description="每张图绘制两个点：绿色点满足边界距离约束；蓝色点不满足边界距离约束。"
    )
    parser.add_argument(
        "--input",
        default="/NAS_data/yjy/Parallel-GAN-main/Parallel-GAN-main/datasets/sar2opt/testB/11_1200_960.jpg",
        help="输入图片路径",
    )
    parser.add_argument(
        "--sam2_checkpoint",
        default="/data/yjy_data/SAM2/checkpoints/sam2.1_hiera_large.pt",
        help="SAM2 checkpoint 路径",
    )
    parser.add_argument(
        "--model_cfg",
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
        help="SAM2 配置路径",
    )
    parser.add_argument("--output-dir", default="single_point_outputs_border", help="输出目录")
    parser.add_argument("--device", default="cuda", help="cuda 或 cpu")
    parser.add_argument("--rng-seed", type=int, default=66, help="随机点采样种子")
    parser.add_argument("--histogram-bins", type=int, default=16, help="RGB 直方图分箱数量")
    parser.add_argument("--histogram-threshold", type=float, default=0.6, help="histogram 峰值比例阈值")
    parser.add_argument(
        "--require-histogram-fail",
        action="store_false",
        dest="require_histogram_pass",
        help="若设置，则改为选择不符合 histogram 条件的点",
    )
    parser.set_defaults(require_histogram_pass=True)
    args = parser.parse_args()

    process_one_image(
        image_path=args.input,
        sam2_checkpoint=args.sam2_checkpoint,
        model_cfg=args.model_cfg,
        output_dir=args.output_dir,
        device=args.device,
        rng_seed=args.rng_seed,
        histogram_bins=args.histogram_bins,
        histogram_threshold=args.histogram_threshold,
        require_histogram_pass=args.require_histogram_pass,
    )


if __name__ == "__main__":
    main()