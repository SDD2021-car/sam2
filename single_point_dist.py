import argparse
from pathlib import Path

import cv2
import numpy as np
from contextlib import nullcontext
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

def select_instance_id(label_map_u16, rng, instance_mode="largest", instance_id=None):
    label_ids = np.unique(label_map_u16)
    label_ids = label_ids[label_ids != 0]
    if len(label_ids) == 0:
        return None

    if instance_id is not None:
        if instance_id in label_ids:
            return int(instance_id)
        raise ValueError(f"指定的 instance_id={instance_id} 不存在，可选: {label_ids.tolist()}")

    if instance_mode == "largest":
        return int(max(label_ids, key=lambda i: int((label_map_u16 == i).sum())))
    if instance_mode == "random":
        return int(rng.choice(label_ids))
    if instance_mode == "first":
        return int(label_ids.min())

    raise ValueError(f"未知 instance_mode: {instance_mode}")


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


def _pick_second_center_with_point_distance(candidate_ys, candidate_xs, rng, first_center, min_point_distance, want_pass):
    if len(candidate_ys) == 0 or first_center is None:
        return None

    fy, fx = first_center
    order = rng.permutation(len(candidate_ys))

    for idx in order:
        cy, cx = int(candidate_ys[idx]), int(candidate_xs[idx])
        dist = float(np.sqrt((cy - fy) ** 2 + (cx - fx) ** 2))
        if (dist >= min_point_distance) == want_pass:
            return (cy, cx)

    return None


def build_dual_dots_with_point_distance_constraint(
    image_rgb,
    label_map_u16,
    dot_radius=6,
    min_point_distance=20,
    rng_seed=0,
    instance_mode="largest",
    instance_id=None,
    histogram_bins=16,
    histogram_threshold=0.3,
    require_histogram_pass=True,
):
    """生成两个点：一个满足点间最小距离约束，一个故意不满足点间最小距离约束。"""
    label_ids = np.unique(label_map_u16)
    label_ids = label_ids[label_ids != 0]
    if len(label_ids) == 0:
        h, w = label_map_u16.shape
        empty = np.zeros((h, w), dtype=bool)
        return {
            "initial_dot": empty,
            "pass_dot": empty,
            "fail_dot": empty,
            "far_center": None,
            "near_center": None,
            "initial_center": None,
            "instance_mask": empty,
        }

    # best_id = max(label_ids, key=lambda i: int((label_map_u16 == i).sum()))
    # mask = (label_map_u16 == best_id).astype(np.uint8)
    rng = np.random.default_rng(rng_seed)
    chosen_id = select_instance_id(
        label_map_u16,
        rng=rng,
        instance_mode=instance_mode,
        instance_id=instance_id,
    )
    if chosen_id is None:
        h, w = label_map_u16.shape
        empty = np.zeros((h, w), dtype=bool)
        return {
            "initial_dot": empty,
            "pass_dot": empty,
            "fail_dot": empty,
            "far_center": None,
            "near_center": None,
            "initial_center": None,
            "instance_mask": empty,
            "instance_id": None,
        }

    mask = (label_map_u16 == chosen_id).astype(np.uint8)
    ys_all, xs_all = np.where(mask > 0)
    if len(ys_all) == 0:
        h, w = label_map_u16.shape
        empty = np.zeros((h, w), dtype=bool)
        return {
            "initial_dot": empty,
            "pass_dot": empty,
            "fail_dot": empty,
            "far_center": None,
            "near_center": None,
            "initial_center": None,
            "instance_mask": empty,
        }

    initial_center = _pick_dot_center(
        mask, ys_all, xs_all, rng, image_rgb, histogram_bins, histogram_threshold, require_histogram_pass
    )
    fail_center = _pick_second_center_with_point_distance(
        ys_all,
        xs_all,
        rng,
        initial_center,
        min_point_distance=min_point_distance,
        want_pass=False,
    )
    pass_center = _pick_second_center_with_point_distance(
        ys_all,
        xs_all,
        rng,
        initial_center,
        min_point_distance=min_point_distance,
        want_pass=True,
    )

    initial_dot = np.zeros_like(mask, dtype=np.uint8)
    pass_dot = np.zeros_like(mask, dtype=np.uint8)
    fail_dot = np.zeros_like(mask, dtype=np.uint8)

    if initial_center is not None:
        iy, ix = initial_center
        cv2.circle(initial_dot, (ix, iy), dot_radius, 1, thickness=-1)
    if pass_center is not None:
        py, px = pass_center
        cv2.circle(pass_dot, (px, py), dot_radius, 1, thickness=-1)
    if fail_center is not None:
        fy, fx = fail_center
        cv2.circle(fail_dot, (fx, fy), dot_radius, 1, thickness=-1)
    initial_dot = initial_dot.astype(bool)
    pass_dot = pass_dot.astype(bool)
    fail_dot = fail_dot.astype(bool)
    # initial_dot = initial_dot.astype(bool) & (mask > 0)
    # pass_dot = pass_dot.astype(bool) & (mask > 0)
    # fail_dot = fail_dot.astype(bool) & (mask > 0)

    return {
        "initial_dot": initial_dot,
        "pass_dot": pass_dot,
        "fail_dot": fail_dot,
        "initial_center": initial_center,
        "pass_center": pass_center,
        "fail_center": fail_center,
        "far_center": pass_center,
        "near_center": fail_center,
        "instance_mask": mask.astype(bool),
        "instance_id": int(chosen_id),
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

    initial_dot = outputs["initial_dot"]
    pass_dot = outputs["pass_dot"]
    fail_dot = outputs["fail_dot"]
    initial_center = outputs["initial_center"]
    pass_center = outputs.get("pass_center", outputs.get("far_center"))
    fail_center = outputs.get("fail_center", outputs.get("near_center"))

    dual_cut = np.zeros_like(image)
    dual_cut[initial_dot] = image[initial_dot]
    dual_cut[pass_dot] = image[pass_dot]
    dual_cut[fail_dot] = image[fail_dot]
    Image.fromarray(dual_cut).save(output_root / f"{stem}_dual_point_cut.png")

    overlay = image.copy()
    if initial_center is not None:
        iy, ix = initial_center
        cv2.circle(overlay, (ix, iy), dot_radius, (255, 255, 0), thickness=2)
        cv2.drawMarker(
            overlay,
            (ix, iy),
            color=(255, 255, 0),
            markerType=cv2.MARKER_STAR,
            markerSize=max(10, dot_radius * 2),
            thickness=2,
        )
        cv2.putText(
            overlay,
            f"init({ix},{iy})",
            (ix + 4, max(12, iy - 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )
    if pass_center is not None:
        fy, fx = pass_center
        cv2.circle(overlay, (fx, fy), dot_radius, (0, 255, 0), thickness=2)
        cv2.drawMarker(
            overlay,
            (fx, fy),
            color=(0, 255, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=max(8, dot_radius * 2),
            thickness=2,
        )
        cv2.putText(
            overlay,
            f"pass({fx},{fy})",
            (fx + 4, max(12, fy - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.circle(overlay, (fx, fy), dot_radius, (0, 255, 0), thickness=2)
    if fail_center is not None:
        ny, nx = fail_center
        cv2.circle(overlay, (nx, ny), dot_radius, (255, 0, 0), thickness=2)
        cv2.drawMarker(
            overlay,
            (nx, ny),
            color=(255, 0, 0),
            markerType=cv2.MARKER_TILTED_CROSS,
            markerSize=max(8, dot_radius * 2),
            thickness=2,
        )
        cv2.putText(
            overlay,
            f"fail({nx},{ny})",
            (nx + 4, min(image.shape[0] - 6, ny + 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )
    Image.fromarray(overlay).save(output_root / f"{stem}_dual_point_overlay.png")

    points_only = np.zeros_like(image)
    if initial_center is not None:
        iy, ix = initial_center
        cv2.circle(points_only, (ix, iy), 2, (255, 255, 0), thickness=-1)
    if pass_center is not None:
        fy, fx = pass_center
        cv2.circle(overlay, (nx, ny), dot_radius, (255, 0, 0), thickness=2)
    if fail_center is not None:
        ny, nx = fail_center
        cv2.circle(points_only, (nx, ny), 2, (255, 0, 0), thickness=-1)
    Image.fromarray(points_only).save(output_root / f"{stem}_dual_points_only.png")

    # combined = initial_dot | pass_dot | fail_dot
    combined = fail_dot
    ys, xs = np.where(combined)
    if len(ys) > 0:
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        crop = dual_cut[y0:y1, x0:x1]
        Image.fromarray(crop).save(output_root / f"{stem}_dual_point_crop_fail.png")

    with open(output_root / f"{stem}_dual_point_centers.txt", "w", encoding="utf-8") as f:
        f.write(f"initial_center(y,x): {initial_center}\n")
        f.write(f"pass_center(y,x): {pass_center}\n")
        f.write(f"fail_center(y,x): {fail_center}\n")


def process_one_image(
    image_path,
    sam2_checkpoint,
    model_cfg,
    output_dir,
    device="cuda",
    rng_seed=0,
    min_point_distance=10,
    instance_mode="largest",
    instance_id=None,
    histogram_bins=16,
    histogram_threshold=0.6,
    require_histogram_pass=True,
):
    image = np.array(Image.open(image_path).convert("RGB"))
    device = torch.device(device)
    validate_checkpoint_file(sam2_checkpoint)
    if device.type == "cuda" and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    with autocast_ctx:
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

    outputs = build_dual_dots_with_point_distance_constraint(
        image,
        label_map,
        dot_radius=3,
        min_point_distance=min_point_distance,
        rng_seed=rng_seed,
        instance_mode=instance_mode,
        instance_id=instance_id,
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
        description="每张图绘制两个点：绿色点满足点间最小距离约束；蓝色点不满足点间最小距离约束。"
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
    parser.add_argument("--output-dir", default="single_point_outputs_dist", help="输出目录")
    parser.add_argument("--device", default="cuda", help="cuda 或 cpu")
    parser.add_argument("--rng-seed", type=int, default=11, help="随机点采样种子")
    parser.add_argument("--min-point-distance", type=float, default=10.0, help="两点之间的最小距离阈值")
    parser.add_argument(
        "--instance-mode",
        choices=["largest", "random", "first"],
        default="random",
        help="选择实例的策略：largest=面积最大；random=随机；first=最小ID",
    )
    parser.add_argument(
        "--instance-id",
        type=int,
        default=None,
        help="指定实例 ID（优先级高于 --instance-mode）",
    )
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