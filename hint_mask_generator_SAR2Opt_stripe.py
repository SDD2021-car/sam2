import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional progress bar dependency
    tqdm = None


def remove_speckles_by_cc(label_map_u16, min_cc_area=500):
    fg = (label_map_u16 > 0).astype(np.uint8)
    num, cc, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] < min_cc_area:
            label_map_u16[cc == i] = 0
    return label_map_u16


def output_exists(output_dir, image_path):
    stem = image_path.stem
    output_dir = Path(output_dir)
    return (output_dir / "color_hint" / f"{stem}.png").exists()


@torch.no_grad()
def make_grid_points_torch(H, W, n_per_side=64, margin=1, device="cuda"):
    xs = torch.linspace(margin, W - 1 - margin, n_per_side, device=device)
    ys = torch.linspace(margin, H - 1 - margin, n_per_side, device=device)
    yv, xv = torch.meshgrid(ys, xs, indexing="ij")
    pts = torch.stack([xv.reshape(-1), yv.reshape(-1)], dim=1).float()
    return pts


@torch.no_grad()
def nms_masks_torch(masks_bool, scores, iou_thr=0.7, topk=None):
    device = masks_bool.device
    N = masks_bool.shape[0]
    if N == 0:
        return torch.empty((0,), dtype=torch.long, device=device)

    order = torch.argsort(scores, descending=True)
    if topk is not None:
        order = order[:topk]

    masks_f = masks_bool.flatten(1)
    areas = masks_f.sum(dim=1).float()

    keep = []
    suppressed = torch.zeros((N,), dtype=torch.bool, device=device)

    for ii in order.tolist():
        if suppressed[ii]:
            continue
        keep.append(ii)
        mi = masks_f[ii]
        ai = areas[ii]
        if ai == 0:
            continue

        rest = order
        rest = rest[~suppressed[rest]]
        inter = (masks_f[rest] & mi).sum(dim=1).float()
        union = areas[rest] + ai - inter
        iou = inter / (union + 1e-6)

        to_suppress = rest[iou > iou_thr]
        suppressed[to_suppress] = True
        suppressed[ii] = False

    return torch.tensor(keep, dtype=torch.long, device=device)


@torch.no_grad()
def masks_to_nonoverlap_label_map_torch(
    masks_bool,
    scores,
    min_area=0,
    alpha=0.5,
    max_regions=65534,
):
    device = masks_bool.device

    if masks_bool.numel() == 0 or masks_bool.shape[0] == 0:
        empty = torch.zeros((1, 1), dtype=torch.uint16, device=device)
        return empty, torch.empty((0,), dtype=torch.long, device=device)

    N, H, W = masks_bool.shape
    masks_bool = masks_bool.bool()

    areas = masks_bool.flatten(1).sum(dim=1).float()
    priority = scores * torch.pow(torch.clamp(areas, min=1.0), alpha)
    order = torch.argsort(priority, descending=True)

    label_map = torch.zeros((H, W), dtype=torch.int32, device=device)

    kept = []
    cur_id = 1
    for idx in order.tolist():
        if cur_id > max_regions:
            break

        m = masks_bool[idx]
        fill = m & (label_map == 0)
        if min_area > 0 and int(fill.sum().item()) < min_area:
            continue

        label_map[fill] = cur_id
        kept.append(idx)
        cur_id += 1

    kept_idx = (
        torch.tensor(kept, dtype=torch.long, device=device)
        if kept
        else torch.empty((0,), dtype=torch.long, device=device)
    )
    label_map_u16 = label_map.clamp(0, max_regions).to(torch.uint16)
    return label_map_u16, kept_idx


@torch.no_grad()
def segment_everything_with_predictor_gpu_post(
    predictor,
    H,
    W,
    n_per_side_list=(48, 64),
    score_thr=0.6,
    min_area=3000,
    nms_iou_thr=0.75,
    topk_per_point=1,
    max_keep=2000,
    device="cuda",
):
    cand_masks = []
    cand_scores = []

    for nps in n_per_side_list:
        pts = make_grid_points_torch(H, W, n_per_side=nps, margin=1, device=device)

        for p in pts:
            p_np = p.view(1, 2).detach().cpu().numpy().astype("float32")
            masks, scores, _ = predictor.predict(
                point_coords=p_np,
                point_labels=np.array([1], dtype=np.int32),
                multimask_output=True,
            )

            order = np.argsort(scores)[::-1]
            picked = 0
            for k in order:
                s = float(scores[k])
                if s < score_thr:
                    continue
                m = masks[k].astype(bool)
                if m.sum() < min_area:
                    continue
                cand_masks.append(m)
                cand_scores.append(s)
                picked += 1
                if picked >= topk_per_point:
                    break

    if len(cand_masks) == 0:
        label_map = torch.zeros((H, W), dtype=torch.uint16, device=device)
        return [], [], label_map

    masks_t = torch.from_numpy(np.stack(cand_masks, axis=0)).to(
        device=device, dtype=torch.bool
    )
    scores_t = torch.tensor(cand_scores, device=device, dtype=torch.float32)

    keep = nms_masks_torch(masks_t, scores_t, iou_thr=nms_iou_thr, topk=max_keep)
    masks_kept = masks_t[keep]
    scores_kept = scores_t[keep]

    label_map, kept_idx2 = masks_to_nonoverlap_label_map_torch(
        masks_kept,
        scores_kept,
        min_area=min_area,
        alpha=0.7,
        max_regions=65534,
    )

    final_masks = masks_kept[kept_idx2]
    final_scores = scores_kept[kept_idx2]

    return final_masks, final_scores, label_map


def label_map_to_masks(label_map_u16):
    label_ids = np.unique(label_map_u16)
    label_ids = label_ids[label_ids != 0]
    masks = [(label_map_u16 == idx) for idx in label_ids]
    return masks, label_ids


def compute_mask_allocations(masks, target_ratio=0.05):
    total_area = sum(int(mask.sum()) for mask in masks)
    if total_area == 0:
        return [0 for _ in masks], 0
    target_total = int(round(target_ratio * masks[0].size))
    allocations = [
        int(round(target_total * (mask.sum() / total_area))) for mask in masks
    ]
    return allocations, target_total


def build_candidate_centers(
    mask,
    min_border_distance=0,
    min_point_distance=10,
    max_candidates=2000,
    rng=None,
):
    mask_u8 = mask.astype(np.uint8)
    dist = cv2.distanceTransform(mask_u8, distanceType=cv2.DIST_L2, maskSize=3)
    coords = np.column_stack(np.where(mask_u8 > 0))
    if coords.size == 0:
        return []
    rng = np.random.default_rng() if rng is None else rng
    rng.shuffle(coords)

    selected = []
    for y, x in coords:
        if len(selected) >= max_candidates:
            break
        if dist[y, x] <= min_border_distance:
            continue
        if selected:
            dists = np.sqrt(
                (np.array(selected)[:, 0] - y) ** 2
                + (np.array(selected)[:, 1] - x) ** 2
            )
            if np.any(dists < min_point_distance):
                continue
        selected.append((y, x))
    return selected


def dominant_color_from_mask(image, mask, bins=16):
    pixels = image[mask]
    if pixels.size == 0:
        return np.array([0, 0, 0], dtype=np.uint8)
    step = 256 // bins
    quant = (pixels // step).astype(np.int32)
    idx = quant[:, 0] * bins * bins + quant[:, 1] * bins + quant[:, 2]
    hist = np.bincount(idx, minlength=bins**3)
    max_idx = int(np.argmax(hist))
    r = max_idx // (bins * bins)
    g = (max_idx // bins) % bins
    b = max_idx % bins
    color = np.array([r, g, b], dtype=np.int32) * step + step // 2
    return np.clip(color, 0, 255).astype(np.uint8)


def histogram_peak_ratio(image, mask, bins=16):
    pixels = image[mask]
    if pixels.size == 0:
        return 0.0
    step = 256 // bins
    quant = (pixels // step).astype(np.int32)
    idx = quant[:, 0] * bins * bins + quant[:, 1] * bins + quant[:, 2]
    hist = np.bincount(idx, minlength=bins**3)
    total = hist.sum()
    if total == 0:
        return 0.0
    return float(hist.max()) / float(total)


def stripe_mask_from_center(center_yx, length, thickness, theta, shape):
    h, w = shape
    cy, cx = center_yx

    radius = 0.5 * math.sqrt(length * length + thickness * thickness)
    y_min = max(0, int(cy - radius))
    y_max = min(h, int(cy + radius) + 1)
    x_min = max(0, int(cx - radius))
    x_max = min(w, int(cx + radius) + 1)

    if y_min >= y_max or x_min >= x_max:
        return np.zeros((h, w), dtype=bool)

    yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
    x_rel = xx - cx
    y_rel = yy - cy

    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    x_rot = x_rel * cos_t + y_rel * sin_t
    y_rot = -x_rel * sin_t + y_rel * cos_t

    local_mask = (np.abs(x_rot) <= length / 2.0) & (np.abs(y_rot) <= thickness / 2.0)

    mask = np.zeros((h, w), dtype=bool)
    mask[y_min:y_max, x_min:x_max] = local_mask
    return mask


def build_hint_masks_from_label_map(
    label_map_u16,
    target_ratio=0.05,
    min_border_distance=5,
    min_point_distance=10,
    stripe_length=16,
    stripe_length_range=None,
    stripe_thickness=3,
    stripe_thickness_range=(1.0, 4.0),
    stripe_theta_range=(0.0, math.pi),
    min_delta_area=10,
    rng_seed=0,
    image=None,
    histogram_threshold=0.7,
    histogram_bins=16,
):
    """Build per-instance hint masks while enforcing a hard global area cap.

    Hard guarantee: the final union mask area is never greater than
    `round(target_ratio * H * W)`.
    """

    rng = np.random.default_rng(rng_seed)
    masks, label_ids = label_map_to_masks(label_map_u16)
    allocations, target_total = compute_mask_allocations(
        masks, target_ratio=target_ratio
    )

    hint_masks = [np.zeros_like(label_map_u16, dtype=bool) for _ in masks]
    hint_all = np.zeros_like(label_map_u16, dtype=bool)
    area_now = 0
    candidate_pools = []
    pool_indices = []

    for mask in masks:
        candidates = build_candidate_centers(
            mask,
            min_border_distance=min_border_distance,
            min_point_distance=min_point_distance,
            rng=rng,
        )
        candidate_pools.append(candidates)
        pool_indices.append(0)

    def passes_histogram_threshold(region_mask):
        if image is None:
            return True
        peak_ratio = histogram_peak_ratio(image, region_mask, bins=histogram_bins)
        return peak_ratio >= histogram_threshold

    def add_delta_with_cap(mask_idx, region_mask):
        nonlocal area_now, hint_all
        if area_now >= target_total:
            return 0

        delta_mask = region_mask & (~hint_masks[mask_idx])
        delta = int(delta_mask.sum())
        if delta == 0:
            return 0

        remain = target_total - area_now
        if delta > remain:
            ys, xs = np.where(delta_mask)
            pick = rng.choice(len(ys), size=remain, replace=False)
            partial = np.zeros_like(delta_mask, dtype=bool)
            partial[ys[pick], xs[pick]] = True
            delta_mask = partial
            delta = remain

        if delta < min_delta_area:
            return 0

        hint_masks[mask_idx] |= delta_mask
        hint_all |= delta_mask
        area_now += delta
        return delta

    # Phase 1: per-instance quota
    for idx, (mask, allocation) in enumerate(zip(masks, allocations)):
        hint_area = 0
        while (
            hint_area < allocation
            and pool_indices[idx] < len(candidate_pools[idx])
            and area_now < target_total
        ):
            center = candidate_pools[idx][pool_indices[idx]]
            pool_indices[idx] += 1

            length = stripe_length
            if stripe_length_range is not None:
                length = rng.uniform(stripe_length_range[0], stripe_length_range[1])
            thickness = stripe_thickness
            if stripe_thickness_range is not None:
                thickness = rng.uniform(
                    stripe_thickness_range[0], stripe_thickness_range[1]
                )
            theta = rng.uniform(stripe_theta_range[0], stripe_theta_range[1])

            raw_stripe = stripe_mask_from_center(center, length, thickness, theta, mask.shape)
            if np.any(raw_stripe & ~mask):
                continue
            stripe = raw_stripe & mask
            if not passes_histogram_threshold(stripe):
                continue

            added = add_delta_with_cap(idx, stripe)
            hint_area += added

    # Phase 2: greedy fill remaining budget
    while area_now < target_total:
        best_idx = None
        best_mask = None
        best_delta = 0

        for i, mask in enumerate(masks):
            if pool_indices[i] >= len(candidate_pools[i]):
                continue
            center = candidate_pools[i][pool_indices[i]]

            length = stripe_length
            if stripe_length_range is not None:
                length = rng.uniform(stripe_length_range[0], stripe_length_range[1])
            thickness = stripe_thickness
            if stripe_thickness_range is not None:
                thickness = rng.uniform(
                    stripe_thickness_range[0], stripe_thickness_range[1]
                )
            theta = rng.uniform(stripe_theta_range[0], stripe_theta_range[1])

            raw_stripe = stripe_mask_from_center(center, length, thickness, theta, mask.shape)
            if np.any(raw_stripe & ~mask):
                continue
            stripe = raw_stripe & mask
            if not passes_histogram_threshold(stripe):
                continue

            delta = int(np.logical_and(stripe, ~hint_masks[i]).sum())
            if delta > best_delta:
                best_delta = delta
                best_idx = i
                best_mask = stripe

        if best_idx is None or best_delta < min_delta_area:
            break

        pool_indices[best_idx] += 1
        add_delta_with_cap(best_idx, best_mask)

    return hint_masks, hint_all, label_ids, allocations, target_total


def generate_color_hint(image, hint_mask):
    color_hint = np.zeros_like(image)
    color_hint[hint_mask] = image[hint_mask]
    return color_hint


def generate_color_hint_by_regions(image, hint_masks, bins=16):
    color_hint = np.zeros_like(image)
    for mask in hint_masks:
        if not np.any(mask):
            continue
        color = dominant_color_from_mask(image, mask, bins=bins)
        color_hint[mask] = color
    return color_hint


def generate_hint_masks(
    image_path,
    sam2_checkpoint,
    model_cfg,
    *,
    device=None,
    segment_params=None,
    hint_params=None,
):
    image = Image.open(image_path)
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))
    else:
        image = np.asarray(image)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    predictor.set_image(image)

    seg_defaults = dict(
        n_per_side_list=(16,),
        score_thr=0.3,
        min_area=500,
        nms_iou_thr=0.75,
        topk_per_point=1,
        max_keep=3000,
    )
    if segment_params:
        seg_defaults.update(segment_params)

    H, W = image.shape[:2]
    _, _, label_map_t = segment_everything_with_predictor_gpu_post(
        predictor,
        H=H,
        W=W,
        device=device.type,
        **seg_defaults,
    )

    label_map = label_map_t.detach().cpu().numpy().astype(np.uint16)
    label_map = remove_speckles_by_cc(label_map, min_cc_area=300)

    hint_defaults = dict(
        target_ratio=0.05,
        min_border_distance=2,
        min_point_distance=10,
        stripe_length=16,
        stripe_length_range=(5.0, 30.0),
        stripe_thickness=3,
        stripe_thickness_range=(1.0, 4.0),
        stripe_theta_range=(0.0, math.pi),
        min_delta_area=10,
        rng_seed=33,
        image=image,
        histogram_threshold=0.6,
        histogram_bins=16,
    )
    if hint_params:
        hint_defaults.update(hint_params)

    hint_masks, hint_all, label_ids, allocations, target_total = (
        build_hint_masks_from_label_map(label_map, **hint_defaults)
    )
    color_hint = generate_color_hint(image, hint_all)
    color_hint_by_regions = generate_color_hint_by_regions(image, hint_masks, bins=16)

    return (
        image,
        color_hint,
        color_hint_by_regions,
        hint_masks,
        hint_all,
        label_ids,
        allocations,
        target_total,
        label_map,
    )


@torch.no_grad()
def save_label_overlay(
    image, label_map, out_path="seg_overlay.png", seed=0, alpha=0.55
):
    rng = np.random.default_rng(seed)
    overlay = image.astype(np.float32) / 255.0

    max_id = int(label_map.max())
    if max_id == 0:
        Image.fromarray(image).save(out_path)
        return

    colors = rng.random((max_id + 1, 3), dtype=np.float32)
    colors[0] = 0

    color_img = colors[label_map]
    mask = (label_map > 0)[..., None].astype(np.float32)

    out = overlay * (1 - alpha * mask) + color_img * (alpha * mask)
    out = (out * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(out).save(out_path)


def save_hint_outputs(
    output_root,
    image_name,
    color_hint,
    color_hint_by_regions,
    hint_masks,
    label_map,
    image,
):
    output_root = Path(output_root)
    color_hint_dir = output_root / "color_hint"
    color_hint_by_regions_dir = output_root / "color_hint_by_dots"
    hint_masks_dir = output_root / "hint_masks"
    label_map_dir = output_root / "label_map"
    label_overlay_dir = output_root / "label_overlay"

    color_hint_dir.mkdir(parents=True, exist_ok=True)
    color_hint_by_regions_dir.mkdir(parents=True, exist_ok=True)
    hint_masks_dir.mkdir(parents=True, exist_ok=True)
    label_map_dir.mkdir(parents=True, exist_ok=True)
    label_overlay_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(image_name).stem
    Image.fromarray(color_hint).save(color_hint_dir / f"{stem}.png")
    Image.fromarray(color_hint_by_regions).save(color_hint_by_regions_dir / f"{stem}.png")

    masks_stack = (
        np.stack(hint_masks, axis=0).astype(np.uint8)
        if hint_masks
        else np.zeros((0, 1, 1), dtype=np.uint8)
    )
    np.savez_compressed(hint_masks_dir / f"{stem}.npz", masks=masks_stack)
    Image.fromarray(label_map.astype(np.uint16)).save(label_map_dir / f"{stem}.png")
    save_label_overlay(
        image,
        label_map,
        out_path=label_overlay_dir / f"{stem}.png",
    )


def iter_image_files(input_path):
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path]

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return [
        path
        for path in sorted(input_path.iterdir())
        if path.is_file() and path.suffix.lower() in extensions
    ]


def _process_images_worker(
    image_paths,
    device,
    sam2_checkpoint,
    model_cfg,
    output_dir,
    queue,
    show_progress,
):
    rows = []
    if show_progress and tqdm is not None:
        iterator = tqdm(image_paths, desc=f"GPU {device}", unit="image")
    else:
        iterator = image_paths

    for image_path in iterator:
        (
            image,
            color_hint,
            color_hint_by_regions,
            hint_masks,
            hint_all,
            _label_ids,
            _allocations,
            _target_total,
            label_map,
        ) = generate_hint_masks(
            image_path,
            sam2_checkpoint,
            model_cfg,
            device=device,
        )
        save_hint_outputs(
            output_dir,
            image_path.name,
            color_hint,
            color_hint_by_regions,
            hint_masks,
            label_map,
            image=image,
        )
        ratio = float(hint_all.sum()) / float(hint_all.size) if hint_all.size else 0.0
        rows.append((image_path.name, f"{ratio:.6f}"))

    queue.put(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Generate stripe hint masks using SAM2 and save outputs."
    )
    parser.add_argument(
        "--input",
        default="/NAS_data/yjy/Parallel-GAN-main/Parallel-GAN-main/datasets/sar2opt/testB",
        help="Path to input image or directory of images.",
    )
    parser.add_argument(
        "--sam2_checkpoint",
        default="/data/yjy_data/SAM2/checkpoints/sam2.1_hiera_large.pt",
        help="Path to the SAM2 checkpoint.",
    )
    parser.add_argument(
        "--model_cfg",
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
        help="Path to the SAM2 model config.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to run on (e.g. cuda, cpu). Defaults to auto-detect.",
    )
    parser.add_argument(
        "--output-dir",
        default="hint_outputs_test_stripe",
        help="Root directory for output folders.",
    )
    parser.add_argument(
        "--summary-csv",
        default="hint_outputs_test_stripe/summary.csv",
        help="Optional path to save the per-image hint ratio table as CSV.",
    )
    parser.add_argument(
        "--devices",
        default="1,2,3,4,5,6",
        help="Comma-separated CUDA device indices for multi-GPU processing (e.g. 0,1).",
    )
    args = parser.parse_args()

    image_paths = iter_image_files(args.input)
    if not image_paths:
        raise ValueError(f"No images found at {args.input}.")

    output_root = Path(args.output_dir)
    image_paths = [path for path in image_paths if not output_exists(output_root, path)]
    if not image_paths:
        print("All images already have outputs. Nothing to process.")
        return

    rows = [("image", "hint_mask_ratio")]
    processes = []

    if args.devices is None or args.device != "cuda":
        iterator = image_paths if tqdm is None else tqdm(image_paths, desc="Processing", unit="image")

        for image_path in iterator:
            (
                image,
                color_hint,
                color_hint_by_regions,
                hint_masks,
                hint_all,
                _label_ids,
                _allocations,
                _target_total,
                label_map,
            ) = generate_hint_masks(
                image_path,
                args.sam2_checkpoint,
                args.model_cfg,
                device=args.device,
            )
            save_hint_outputs(
                args.output_dir,
                image_path.name,
                color_hint,
                color_hint_by_regions,
                hint_masks,
                label_map,
                image=image,
            )
            ratio = (
                float(hint_all.sum()) / float(hint_all.size) if hint_all.size else 0.0
            )
            rows.append((image_path.name, f"{ratio:.6f}"))
    else:
        device_ids = [
            int(part.strip())
            for part in args.devices.split(",")
            if part.strip() != ""
        ]
        if not device_ids:
            raise ValueError("--devices provided but no valid device indices found.")
        image_chunks = [image_paths[i:: len(device_ids)] for i in range(len(device_ids))]
        queue = mp.Queue()

        for device_id, chunk in zip(device_ids, image_chunks):
            if not chunk:
                continue
            device = f"cuda:{device_id}"
            process = mp.Process(
                target=_process_images_worker,
                args=(
                    chunk,
                    device,
                    args.sam2_checkpoint,
                    args.model_cfg,
                    args.output_dir,
                    queue,
                    True,
                ),
            )
            process.start()
            processes.append(process)

        for _ in processes:
            rows.extend(queue.get())

        for process in processes:
            process.join()

    rows = [rows[0]] + sorted(rows[1:], key=lambda x: x[0])

    col_widths = [
        max(len(row[0]) for row in rows),
        max(len(row[1]) for row in rows),
    ]
    header = f"{rows[0][0].ljust(col_widths[0])} | {rows[0][1].rjust(col_widths[1])}"
    separator = f"{'-' * col_widths[0]}-+-{'-' * col_widths[1]}"
    print(header)
    print(separator)
    for row in rows[1:]:
        print(f"{row[0].ljust(col_widths[0])} | {row[1].rjust(col_widths[1])}")

    summary_path = args.summary_csv
    if summary_path is None:
        summary_path = Path(args.output_dir) / "hint_mask_ratio.csv"
    else:
        summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("image,hint_mask_ratio\n")
        for row in rows[1:]:
            f.write(f"{row[0]},{row[1]}\n")


if __name__ == "__main__":
    main()