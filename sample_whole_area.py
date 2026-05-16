import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import cv2
# ---------------------------
# Device + AMP settings
# ---------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
# ---------------------------
# Visualization helpers
# ---------------------------
import cv2
import numpy as np

def keep_largest_cc(mask_bool, min_cc_area=0, fill_holes=False):
    """
    mask_bool: (H,W) bool
    return: (H,W) bool, 只保留最大连通域（可加 min_cc_area）
    """
    m = mask_bool.astype(np.uint8)
    if m.max() == 0:
        return mask_bool

    num, cc, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return mask_bool

    # 找最大前景连通域（跳过0背景）
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))
    if stats[best, cv2.CC_STAT_AREA] < min_cc_area:
        return np.zeros_like(mask_bool, dtype=bool)

    out = (cc == best)

    if fill_holes:
        # 填洞：用 flood fill 或形态学（简单起见用 flood fill）
        out_u8 = out.astype(np.uint8)
        h, w = out_u8.shape
        flood = out_u8.copy()
        mask = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(flood, mask, seedPoint=(0,0), newVal=1)  # 填充背景
        holes = (flood == 0)
        out = out | holes

    return out

def remove_speckles_by_cc(label_map_u16, min_cc_area=500):
    # 把所有非0区域当作前景做连通域
    fg = (label_map_u16 > 0).astype(np.uint8)
    num, cc, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    # stats: [label, x, y, w, h, area]
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] < min_cc_area:
            label_map_u16[cc == i] = 0
    return label_map_u16


def show_mask(mask, ax, random_color=False, borders=True, color=None, alpha=0.6):
    """
    mask: (H,W) bool/0-1
    color: RGBA np.array([r,g,b,a]) in [0,1], if provided overrides random_color/default
    """
    if color is None:
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([alpha])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, alpha])
    else:
        color = color.copy()
        if color.shape[0] == 3:
            color = np.concatenate([color, np.array([alpha])], axis=0)

    h, w = mask.shape[-2:]
    mask_u8 = mask.astype(np.uint8)
    mask_image = mask_u8.reshape(h, w, 1) * color.reshape(1, 1, -1)

    if borders:
        import cv2
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.7), thickness=2)

    ax.imshow(mask_image, interpolation="nearest")  # 防止插值造成“散点”伪影


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    if len(pos_points) > 0:
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*',
                   s=marker_size, edgecolor='white', linewidth=1.25)
    if len(neg_points) > 0:
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*',
                   s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green',
                               facecolor=(0, 0, 0, 0), lw=2))


def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None,
               borders=True, save_prefix="test_mask"):
    """
    保存每个候选mask为独立文件，避免覆盖
    """
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {float(score):.3f}", fontsize=18)
        plt.axis('off')
        plt.savefig(f"{save_prefix}_{i+1}.png", dpi=300, bbox_inches="tight")
        plt.close()


# ---------------------------
# Segment-everything helpers
# ---------------------------
@torch.no_grad()
def make_grid_points_torch(H, W, n_per_side=64, margin=1, device="cuda"):
    xs = torch.linspace(margin, W - 1 - margin, n_per_side, device=device)
    ys = torch.linspace(margin, H - 1 - margin, n_per_side, device=device)
    yv, xv = torch.meshgrid(ys, xs, indexing="ij")
    pts = torch.stack([xv.reshape(-1), yv.reshape(-1)], dim=1).float()  # (x,y)
    return pts


def mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union + 1e-6)


@torch.no_grad()
def nms_masks_torch(masks_bool, scores, iou_thr=0.7, topk=None):
    """
    masks_bool: (N,H,W) bool GPU tensor
    scores: (N,) float GPU tensor
    返回 keep indices (1D long tensor)
    """
    device = masks_bool.device
    N = masks_bool.shape[0]
    if N == 0:
        return torch.empty((0,), dtype=torch.long, device=device)

    order = torch.argsort(scores, descending=True)
    if topk is not None:
        order = order[:topk]

    # 预计算面积（GPU）
    masks_f = masks_bool.flatten(1)
    areas = masks_f.sum(dim=1).float()  # (N,)

    keep = []
    suppressed = torch.zeros((N,), dtype=torch.bool, device=device)

    # O(K^2) 但全 GPU bit 运算，比 numpy/CPU 快很多
    for ii in order.tolist():
        if suppressed[ii]:
            continue
        keep.append(ii)
        mi = masks_f[ii]  # (HW,)
        ai = areas[ii]
        if ai == 0:
            continue

        # 只对剩余候选计算 IoU
        rest = order
        rest = rest[~suppressed[rest]]
        # 计算 inter / union（GPU）
        inter = (masks_f[rest] & mi).sum(dim=1).float()
        union = areas[rest] + ai - inter
        iou = inter / (union + 1e-6)

        # 抑制与当前 mask 高重叠的
        to_suppress = rest[iou > iou_thr]
        suppressed[to_suppress] = True

        # 自己不要抑制
        suppressed[ii] = False

    return torch.tensor(keep, dtype=torch.long, device=device)

@torch.no_grad()
def masks_to_nonoverlap_label_map_torch(
    masks_bool, scores,
    min_area=0,
    alpha=0.5,
    max_regions=65534
):
    device = masks_bool.device

    if masks_bool.numel() == 0 or masks_bool.shape[0] == 0:
        return torch.zeros((1, 1), dtype=torch.uint16, device=device), torch.empty((0,), dtype=torch.long, device=device)

    N, H, W = masks_bool.shape
    masks_bool = masks_bool.bool()  # 确保是 bool

    areas = masks_bool.flatten(1).sum(dim=1).float()
    priority = scores * torch.pow(torch.clamp(areas, min=1.0), alpha)
    order = torch.argsort(priority, descending=True)

    label_map = torch.zeros((H, W), dtype=torch.int32, device=device)

    kept = []
    cur_id = 1
    for idx in order.tolist():
        if cur_id > max_regions:
            break

        m = masks_bool[idx]  # (H,W) bool tensor

        # 只填充 아직未占用的像素，保证 non-overlap
        fill = m & (label_map == 0)
        if min_area > 0 and int(fill.sum().item()) < min_area:
            continue

        label_map[m] = cur_id
        kept.append(idx)
        cur_id += 1

    kept_idx = torch.tensor(kept, dtype=torch.long, device=device) if kept else torch.empty((0,), dtype=torch.long, device=device)
    label_map_u16 = label_map.clamp(0, max_regions).to(torch.uint16)
    return label_map_u16, kept_idx



@torch.no_grad()
def segment_everything_with_predictor_gpu_post(
    predictor,
    H, W,
    n_per_side_list=(48, 64),
    score_thr=0.6,
    min_area=3000,
    nms_iou_thr=0.75,
    topk_per_point=1,
    max_keep=2000,
    device="cuda"
):
    cand_masks = []
    cand_scores = []

    for nps in n_per_side_list:
        pts = make_grid_points_torch(H, W, n_per_side=nps, margin=1, device=device)

        # predictor.predict 需要 numpy 坐标（多数实现），因此这里逐点仍是 Python 循环
        for p in pts:
            p_np = p.view(1, 2).detach().cpu().numpy().astype("float32")
            masks, scores, _ = predictor.predict(
                point_coords=p_np,
                point_labels=np.array([1], dtype=np.int32),
                multimask_output=True,
            )

            # 先在 CPU 过滤少量，再一次性转 GPU（减少传输量）
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

    # ---- 一次性搬到 GPU ----
    masks_t = torch.from_numpy(np.stack(cand_masks, axis=0)).to(device=device, dtype=torch.bool)   # (N,H,W)
    scores_t = torch.tensor(cand_scores, device=device, dtype=torch.float32)                       # (N,)

    # ---- GPU NMS ----
    keep = nms_masks_torch(masks_t, scores_t, iou_thr=nms_iou_thr, topk=max_keep)
    masks_kept = masks_t[keep]
    scores_kept = scores_t[keep]

    # ---- GPU non-overlap label map ----
    label_map, kept_idx2 = masks_to_nonoverlap_label_map_torch(
        masks_kept, scores_kept,
        min_area=min_area,
        alpha=0.7,
        max_regions=65534
    )

    # 返回“最终非重叠 masks / scores”
    final_masks = masks_kept[kept_idx2]         # (M,H,W) bool GPU
    final_scores = scores_kept[kept_idx2]       # (M,) GPU

    return final_masks, final_scores, label_map


def save_label_overlay(image, label_map, out_path="seg_overlay.png", seed=0, alpha=0.55):
    """
    将 label_map (H,W) 可视化为随机颜色叠加图，0为背景不着色
    """
    rng = np.random.default_rng(seed)
    H, W = label_map.shape
    overlay = image.astype(np.float32) / 255.0

    max_id = int(label_map.max())
    if max_id == 0:
        Image.fromarray(image).save(out_path)
        return

    # 生成颜色表（1..max_id）
    colors = rng.random((max_id + 1, 3), dtype=np.float32)
    colors[0] = 0  # 背景

    color_img = colors[label_map]  # (H,W,3)
    mask = (label_map > 0)[..., None].astype(np.float32)

    out = overlay * (1 - alpha * mask) + color_img * (alpha * mask)
    out = (out * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(out).save(out_path)


# ---------------------------
# Load image
# ---------------------------
image = Image.open('/NAS_data/yjy/Parallel-GAN-main/Parallel-GAN-main/datasets/sar2opt/testB/13_3360_960.jpg')
image = np.array(image.convert("RGB"))
H, W = image.shape[:2]

plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis('off')
plt.savefig("test_image2.png", dpi=300, bbox_inches="tight")
plt.close()


# ---------------------------
# Build SAM2 + predictor
# ---------------------------
sam2_checkpoint = "/data/yjy_data/SAM2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)
predictor.set_image(image)

print(predictor._features["image_embed"].shape, predictor._features["image_embed"][-1].shape)

# ---------------------------
# (A) Your original single-point demo (kept)
# ---------------------------
input_point = np.array([[100, 375]], dtype=np.float32)
input_label = np.array([1], dtype=np.int32)

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

sorted_ind = np.argsort(scores)[::-1]
masks = masks[sorted_ind]
scores = scores[sorted_ind]
logits = logits[sorted_ind]

# 保存三个候选mask：test_mask_1.png, test_mask_2.png, test_mask_3.png
show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label,
           borders=True, save_prefix="test_mask")


# ---------------------------
# (B) Segment Everything -> multi-region label map (NEW)
# ---------------------------
final_masks_t, final_scores_t, label_map_t = segment_everything_with_predictor_gpu_post(
    predictor,
    H=H, W=W,
    n_per_side_list=(32,),   # n_per_side_list 决定的是 “用多少个提示点去扫整张图像”,几个数就是几次
    score_thr=0.5, # 只保留模型认为“像一个有效目标”的 mask，分数低于 0.05 的直接丢弃。
    min_area=2000,
    nms_iou_thr=0.7,
    topk_per_point=1,
    max_keep=5000,
    device=device.type  # "cuda" or "cpu"
)

# 兼容：final_masks_t 可能是 torch.Tensor，也可能是 list
num_regions = final_masks_t.shape[0] if torch.is_tensor(final_masks_t) else len(final_masks_t)

# 兼容：label_map_t 可能是 uint16 / int32 / 在GPU上
label_max_id = int(label_map_t.to(torch.int64).max().item()) if torch.is_tensor(label_map_t) else int(label_map_t.max())

print(f"[SegmentEverything] kept regions: {num_regions} | label max id: {label_max_id}")


# 保存 overlay：这一步要转回 CPU/np，因为 PIL/matplotlib 在 CPU
label_map = label_map_t.detach().cpu().numpy().astype(np.uint16)
label_map = remove_speckles_by_cc(label_map, min_cc_area=500)  # 500~2000 试
save_label_overlay(image, label_map, out_path="seg_overlay10.png", seed=0, alpha=0.55)

