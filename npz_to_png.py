import os
from pathlib import Path
import numpy as np
from PIL import Image

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

def color_dots_to_white(img: Image.Image, thr: int = 20, keep_alpha: bool = True) -> Image.Image:
    """
    thr: 判定为“点”的亮度阈值。越大越严格（避免把暗噪声当点）。
    keep_alpha: 若原图有 alpha，是否保留 alpha 通道。
    """
    # 用 RGBA 统一处理（兼容 PNG alpha）
    rgba = img.convert("RGBA")
    arr = np.array(rgba)  # HxWx4, uint8
    rgb = arr[..., :3]
    a   = arr[..., 3]

    # 点判定：任一通道 > thr，并且(若有alpha) alpha>0
    is_dot = (rgb.max(axis=-1) > thr) & (a > 0)

    # 把点像素改成白色
    rgb[is_dot] = 255

    if keep_alpha:
        out = np.dstack([rgb, a])
        return Image.fromarray(out, mode="RGBA")
    else:
        return Image.fromarray(rgb, mode="RGB")

def batch(input_dir: str, output_dir: str, thr: int = 20):
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted([p for p in in_dir.rglob("*") if p.suffix.lower() in IMG_EXTS])
    print(f"Found {len(paths)} images.")

    for p in paths:
        try:
            img = Image.open(p)
            out_img = color_dots_to_white(img, thr=thr, keep_alpha=True)

            rel = p.relative_to(in_dir)
            out_path = out_dir / rel.with_suffix(".png")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_img.save(out_path)
        except Exception as e:
            print(f"[SKIP] {p} reason: {e}")

    print("Done.")

if __name__ == "__main__":
    batch("/data/yjy_data/SAM2/hint_outputs_test/color_hint_by_dots", "/data/yjy_data/SAM2/hint_outputs_test/hint_masks_png", thr=20)
