from PIL import Image
import numpy as np
import cv2
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
def _filter_valid_pixels(image_rgb, valid_mask, ignore_black=True):
    """提取有效像素，并可选去除纯黑像素。"""
    pixels = image_rgb[valid_mask]

    if ignore_black and pixels.size > 0:
        not_black_mask = np.any(pixels != 0, axis=1)
        pixels = pixels[not_black_mask]
    return pixels

def dominant_color_from_valid_region(image_rgb, valid_mask, bins=16, ignore_black=True):
    """
    只在 valid_mask=True 的区域内统计最大 color bin。
    可选忽略纯黑像素 (0, 0, 0)，避免背景黑色干扰直方图。
    """
    pixels = _filter_valid_pixels(image_rgb, valid_mask, ignore_black=ignore_black)
    if pixels.size == 0:
        return np.array([0, 0, 0], dtype=np.uint8), 0, None

    step = 256 // bins
    quant = (pixels // step).astype(np.int32)

    idx = quant[:, 0] * bins * bins + quant[:, 1] * bins + quant[:, 2]
    hist = np.bincount(idx, minlength=bins**3)

    max_idx = int(np.argmax(hist))
    r = max_idx // (bins * bins)
    g = (max_idx // bins) % bins
    b = max_idx % bins

    color = np.array([r, g, b], dtype=np.int32) * step + step // 2
    color = np.clip(color, 0, 255).astype(np.uint8)

    return color, max_idx, hist

def save_joint_color_histogram(hist, out_path):
    plt.figure(figsize=(12, 4))
    x = np.arange(len(hist))  # 4096
    plt.plot(x, hist, linewidth=1)
    plt.title("Joint RGB Quantized Histogram")
    plt.xlabel("Color Bin Index")
    plt.ylabel("Count")
    plt.xlim(0, len(hist) - 1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def save_rgb_histogram(image_rgb, valid_mask, out_hist_path, ignore_black=True, smooth_sigma=2.0):
    """按 R/G/B 三通道绘制直方图，并平滑成曲线后保存。"""
    pixels = _filter_valid_pixels(image_rgb, valid_mask, ignore_black=ignore_black)

    if pixels.size == 0:
        print("No valid pixels for histogram, skip saving histogram image.")
        return False

    out_hist_path = Path(out_hist_path)
    out_hist_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    channel_names = ["R", "G", "B"]
    channel_colors = ["red", "green", "blue"]
    # 用柱子中点作为 x 坐标，并用曲线连接这些中点。
    x_centers = np.arange(256) + 0.5

    for i, (name, color) in enumerate(zip(channel_names, channel_colors)):
        hist = np.bincount(pixels[:, i], minlength=256).astype(np.float32)
        if smooth_sigma and smooth_sigma > 0:
            hist = cv2.GaussianBlur(hist[None, :], ksize=(0, 0), sigmaX=smooth_sigma).ravel()
        plt.plot(x_centers, hist, color=color, linewidth=0, label=name)
        plt.fill_between(x_centers, hist, 0, color=color, alpha=0.8)


    plt.title("RGB Histogram (valid region)")
    plt.xlabel("Pixel Value")
    plt.ylabel("Count")
    plt.xlim([0, 255])
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_hist_path, dpi=150)
    plt.close()

    print(f"RGB histogram saved to: {out_hist_path}")
    return True

def make_color_circle_canvas(shape, color, radius=None, bg_color=(0, 0, 0)):
    """
    生成原图大小的纯色圆图
    """
    h, w = shape[:2]
    if radius is None:
        radius = min(h, w) // 2

    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:] = np.array(bg_color, dtype=np.uint8)

    center = (w // 2, h // 2)
    cv2.circle(canvas, center, radius, color.tolist(), thickness=-1)

    return canvas


def process_rgba_circle_image(image_path, out_path, bins=16, ignore_black=True, hist_out_path=None):
    img = Image.open(image_path).convert("RGBA")
    arr = np.array(img)

    rgb = arr[..., :3]
    alpha = arr[..., 3]

    # 只统计非透明区域
    valid_mask = alpha > 0

    color, max_idx, hist = dominant_color_from_valid_region(
        rgb, valid_mask, bins=bins, ignore_black=ignore_black
    )
    save_joint_color_histogram(hist, "/data/yjy_data/SAM2/1/joint_hist_4096_not_pass.png")
    if hist_out_path is None:
        out_path_obj = Path(out_path)
        hist_out_path = out_path_obj.with_name(f"{out_path_obj.stem}_rgb_hist.png")

    save_rgb_histogram(
        rgb,
        valid_mask,
        out_hist_path=hist_out_path,
        ignore_black=ignore_black,
    )
    # 半径也可以直接依据有效区域来估算
    ys, xs = np.where(valid_mask)
    if len(ys) == 0:
        radius = min(rgb.shape[:2]) // 2
    else:
        h = ys.max() - ys.min() + 1
        w = xs.max() - xs.min() + 1
        radius = min(h, w) // 2

    circle_img = make_color_circle_canvas(rgb.shape, color, radius=radius)
    Image.fromarray(circle_img).save(out_path)

    print("dominant bin color:", color.tolist())
    print("max_idx:", max_idx)
    print("histogram path:", hist_out_path)
    return color, circle_img


if __name__ == "__main__":
    process_rgba_circle_image(
        image_path="/data/yjy_data/SAM2/single_point_outputs/11_1200_960_point_crop_not_pass.png",
        out_path="/data/yjy_data/SAM2/1/circle_new.png",
        bins=16,
        ignore_black=True,
        hist_out_path="/data/yjy_data/SAM2/1/circle_new_rgb_hist.png",
    )