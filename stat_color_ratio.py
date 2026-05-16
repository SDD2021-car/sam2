import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def color_ratio_nonblack(img_path: Path, black_rgb_thresh: int = 25) -> dict:
    """
    色点=非黑像素
    近黑判定：R,G,B 都 < black_rgb_thresh 认为是黑
    """
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        arr = np.asarray(im, dtype=np.uint8)

    h, w, _ = arr.shape
    total = h * w

    r = arr[..., 0]
    g = arr[..., 1]
    b = arr[..., 2]

    near_black = (r < black_rgb_thresh) & (g < black_rgb_thresh) & (b < black_rgb_thresh)
    non_black = ~near_black

    color_pixels = int(non_black.sum())
    ratio = color_pixels / total if total else 0.0

    return {
        "file": str(img_path),
        "filename": img_path.name,
        "color_pixels": color_pixels,
        "total_pixels": total,
        "color_ratio": ratio,
    }


def main():
    parser = argparse.ArgumentParser(description="统计黑底图片非黑像素占比，并导出表格")
    parser.add_argument("--input_dir", default="/data/yjy_data/SAM2/hint_outputs_train_scene_try2/color_hint_by_dots", help="输入文件夹（递归遍历）")
    parser.add_argument("--out", type=str, default="color_ratio_train_scene2.xlsx", help="输出Excel路径")
    parser.add_argument("--csv", type=str, default="", help="可选：同时输出CSV路径")
    parser.add_argument("--black_rgb_thresh", type=int, default=25, help="近黑RGB阈值(越大越把暗像素当黑)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"输入路径不是文件夹：{input_dir}")

    rows = []
    for root, _, files in os.walk(input_dir):
        for fn in files:
            p = Path(root) / fn
            if not is_image_file(p):
                continue
            try:
                rows.append(color_ratio_nonblack(p, black_rgb_thresh=args.black_rgb_thresh))
            except Exception as e:
                rows.append({
                    "file": str(p),
                    "filename": p.name,
                    "color_pixels": None,
                    "total_pixels": None,
                    "color_ratio": None,
                    "error": str(e),
                })

    df = pd.DataFrame(rows).sort_values("file", ignore_index=True)

    out_xlsx = Path(args.out)
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out_xlsx, index=False)

    if args.csv:
        out_csv = Path(args.csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"完成：共处理 {len(df)} 张图片")
    print(f"已保存：{out_xlsx}")
    if args.csv:
        print(f"已保存：{args.csv}")


if __name__ == "__main__":
    main()
