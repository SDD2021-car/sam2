# import torch
# from sam2.build_sam import build_sam2
# from sam2.sam2_image_predictor import SAM2ImagePredictor
# from PIL import Image
#
# checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
# model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
# predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
#
# with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
#     image = Image.open("/NAS_data/yjy/Parallel-GAN-main/Parallel-GAN-main/datasets/sar2opt/testB/11_1200_720.jpg")
#     predictor.set_image(image)
#     masks, _, _ = predictor.predict("tree")
import os
# import numpy as np
# import torch
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# from PIL import Image
# from sam2.build_sam import build_sam2
# from sam2.sam2_image_predictor import SAM2ImagePredictor
#
# if torch.cuda.is_available():
#     device = torch.device("cuda")
# else:
#     device = torch.device("cpu")
# print(f"using device: {device}")
# if device.type == "cuda":
#     # use bfloat16 for the entire notebook
#     torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
#     # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
#     if torch.cuda.get_device_properties(0).major >= 8:
#         torch.backends.cuda.matmul.allow_tf32 = True
#         torch.backends.cudnn.allow_tf32 = True
#
# def show_mask(mask, ax, random_color=False, borders = True):
#     if random_color:
#         color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
#     else:
#         color = np.array([30/255, 144/255, 255/255, 0.6])
#     h, w = mask.shape[-2:]
#     mask = mask.astype(np.uint8)
#     mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
#     if borders:
#         import cv2
#         contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#         # Try to smooth contours
#         contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
#         mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
#     ax.imshow(mask_image)
#
# def show_points(coords, labels, ax, marker_size=375):
#     pos_points = coords[labels==1]
#     neg_points = coords[labels==0]
#     ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
#     ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
#
# def show_box(box, ax):
#     x0, y0 = box[0], box[1]
#     w, h = box[2] - box[0], box[3] - box[1]
#     ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
#
# def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
#     for i, (mask, score) in enumerate(zip(masks, scores)):
#         plt.figure(figsize=(10, 10))
#         plt.imshow(image)
#         show_mask(mask, plt.gca(), borders=borders)
#         if point_coords is not None:
#             assert input_labels is not None
#             show_points(point_coords, input_labels, plt.gca())
#         if box_coords is not None:
#             # boxes
#             show_box(box_coords, plt.gca())
#         if len(scores) > 1:
#             plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
#         plt.axis('off')
#         plt.savefig("test_mask.png", dpi=300, bbox_inches="tight")
#         plt.close()
#
# def make_grid_points(H, W, n_per_side=64, margin=0):
#     """生成均匀网格点，坐标格式为 (x, y)"""
#     xs = np.linspace(margin, W - 1 - margin, n_per_side)
#     ys = np.linspace(margin, H - 1 - margin, n_per_side)
#     xv, yv = np.meshgrid(xs, ys)
#     pts = np.stack([xv.reshape(-1), yv.reshape(-1)], axis=1).astype(np.float32)
#     return pts
#
# def mask_iou(a, b):
#     """a,b: bool mask (H,W)"""
#     inter = np.logical_and(a, b).sum()
#     union = np.logical_or(a, b).sum()
#     return float(inter) / float(union + 1e-6)
#
# def nms_masks(masks, scores, iou_thr=0.7, topk=None):
#     """
#     masks: list[bool(H,W)]
#     scores: list[float]
#     return: keep indices
#     """
#     idxs = np.argsort(scores)[::-1]
#     if topk is not None:
#         idxs = idxs[:topk]
#
#     keep = []
#     for i in idxs:
#         ok = True
#         for j in keep:
#             if mask_iou(masks[i], masks[j]) > iou_thr:
#                 ok = False
#                 break
#         if ok:
#             keep.append(i)
#     return keep
#
#
# image = Image.open('/NAS_data/yjy/Parallel-GAN-main/Parallel-GAN-main/datasets/sar2opt/testB/11_1200_720.jpg')
# image = np.array(image.convert("RGB"))
# plt.figure(figsize=(10, 10))
# plt.imshow(image)
# plt.axis('off')
# plt.savefig("test_image.png", dpi=300, bbox_inches="tight")
# plt.close()
#
# sam2_checkpoint = "/data/yjy_data/SAM2/checkpoints/sam2.1_hiera_large.pt"
# model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
#
# sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
#
# predictor = SAM2ImagePredictor(sam2_model)
# predictor.set_image(image)
# input_point = np.array([[100, 375]])
# input_label = np.array([1])
#
# plt.figure(figsize=(10, 10))
# plt.imshow(image)
# show_points(input_point, input_label, plt.gca())
# plt.axis('off')
# plt.savefig("test_image1.png", dpi=300, bbox_inches="tight")
# plt.close()
#
# print(predictor._features["image_embed"].shape, predictor._features["image_embed"][-1].shape)
# masks, scores, logits = predictor.predict(
#     point_coords=input_point,
#     point_labels=input_label,
#     multimask_output=True,
# )
# sorted_ind = np.argsort(scores)[::-1]
# masks = masks[sorted_ind]
# scores = scores[sorted_ind]
# logits = logits[sorted_ind]
# show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label, borders=True)
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


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
def make_grid_points(H, W, n_per_side=64, margin=1):
    xs = np.linspace(margin, W - 1 - margin, n_per_side)
    ys = np.linspace(margin, H - 1 - margin, n_per_side)
    xv, yv = np.meshgrid(xs, ys)
    pts = np.stack([xv.reshape(-1), yv.reshape(-1)], axis=1).astype(np.float32)  # (x,y)
    return pts


def mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union + 1e-6)


def nms_masks(masks, scores, iou_thr=0.7, topk=None):
    idxs = np.argsort(scores)[::-1]
    if topk is not None:
        idxs = idxs[:topk]

    keep = []
    for i in idxs:
        ok = True
        for j in keep:
            if mask_iou(masks[i], masks[j]) > iou_thr:
                ok = False
                break
        if ok:
            keep.append(i)
    return keep


def segment_everything_with_predictor(
    predictor,
    H, W,
    n_per_side_list=(48, 64),   # 你可以改成 (64, 96) 更细但更慢
    score_thr=0.6,              # 过滤低分mask，减少碎片
    min_area=3000,              # 去除小碎片（按分辨率调）
    nms_iou_thr=0.75,           # 去重阈值
    topk_per_point=1,           # 每个点取几个候选mask（1=取最高分）
    max_keep=2000               # 最多保留多少mask（防止爆内存/太慢）
):
    cand_masks = []
    cand_scores = []

    for nps in n_per_side_list:
        pts = make_grid_points(H, W, n_per_side=nps, margin=1)

        for p in pts:
            masks, scores, _ = predictor.predict(
                point_coords=p.reshape(1, 2),
                point_labels=np.array([1], dtype=np.int32),
                multimask_output=True,
            )
            order = np.argsort(scores)[::-1]

            for k in order[:topk_per_point]:
                s = float(scores[k])
                if s < score_thr:
                    continue
                m = masks[k].astype(bool)
                if m.sum() < min_area:
                    continue
                cand_masks.append(m)
                cand_scores.append(s)

    if len(cand_masks) == 0:
        label_map = np.zeros((H, W), dtype=np.uint16)
        return [], [], label_map

    # NMS 去重
    keep = nms_masks(cand_masks, cand_scores, iou_thr=nms_iou_thr, topk=max_keep)
    kept_masks = [cand_masks[i] for i in keep]
    kept_scores = [cand_scores[i] for i in keep]

    # 生成 label map：高分优先填充未占用区域
    order2 = np.argsort(kept_scores)[::-1]
    label_map = np.zeros((H, W), dtype=np.uint16)
    final_masks, final_scores = [], []
    cur_id = 1

    for idx in order2:
        m = kept_masks[idx]
        fill = np.logical_and(m, label_map == 0)
        if fill.sum() < min_area:
            continue
        label_map[fill] = cur_id
        final_masks.append(m)
        final_scores.append(kept_scores[idx])
        cur_id += 1
        if cur_id >= 65535:  # uint16 上限保护
            break

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
image = Image.open('/NAS_data/yjy/Parallel-GAN-main/Parallel-GAN-main/datasets/sar2opt/testA/13_3360_960.jpg')
image = np.array(image.convert("RGB"))
H, W = image.shape[:2]

plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis('off')
plt.savefig("test_image1.png", dpi=300, bbox_inches="tight")
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

plt.figure(figsize=(10, 10))
plt.imshow(image)
show_points(input_point, input_label, plt.gca())
plt.axis('off')
plt.savefig("test_image1_1.png", dpi=300, bbox_inches="tight")
plt.close()

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
final_masks, final_scores, label_map = segment_everything_with_predictor(
    predictor,
    H=H, W=W,
    n_per_side_list=(96, 128),   # 更细可改成 (64, 96) 但会慢
    score_thr=0.05,
    min_area=100,
    nms_iou_thr=0.9,
    topk_per_point=8,
    max_keep=10000
)

print(f"[SegmentEverything] kept regions: {len(final_masks)} | label max id: {int(label_map.max())}")

# 保存 label map（uint16，避免 >255 溢出）
Image.fromarray(label_map.astype(np.uint16)).save("label_map_uint161.png")

# 保存叠加可视化图
save_label_overlay(image, label_map, out_path="seg_overlay1.png", seed=0, alpha=0.55)
