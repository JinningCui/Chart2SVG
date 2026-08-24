#!/usr/bin/env python3
"""Compare Beagle source images with PNGs rendered from cleaned SVGs.

The source image is normalized with the same 512 px resize-and-center rule used by
generate_syntactic_svg.py.  Metrics are intended for screening; the lowest-score
rows can then be inspected visually before making SVG changes.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops
from skimage.metrics import structural_similarity


ROOT = Path(__file__).resolve().parent / "data" / "Beagle"
TARGET_SIZE = 512
METRIC_SIZE = 256


def _rgb_on_white(path: Path) -> Image.Image:
    with Image.open(path) as image:
        image.load()
        if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
            rgba = image.convert("RGBA")
            white = Image.new("RGBA", rgba.size, "white")
            return Image.alpha_composite(white, rgba).convert("RGB")
        return image.convert("RGB")


def _normalize_source(image: Image.Image) -> Image.Image:
    width, height = image.size
    scale = TARGET_SIZE / max(width, height)
    new_size = (int(width * scale), int(height * scale))
    resized = image.resize(new_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (TARGET_SIZE, TARGET_SIZE), "white")
    canvas.paste(
        resized,
        ((TARGET_SIZE - new_size[0]) // 2, (TARGET_SIZE - new_size[1]) // 2),
    )
    return canvas


def _content_bbox(image: Image.Image):
    diff = ImageChops.difference(image, Image.new("RGB", image.size, "white"))
    # Ignore tiny antialiasing/compression differences close to white.
    mask = np.max(np.asarray(diff), axis=2) > 8
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _bbox_iou(left, right) -> float:
    if left is None or right is None:
        return 1.0 if left == right else 0.0
    ix0, iy0 = max(left[0], right[0]), max(left[1], right[1])
    ix1, iy1 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 1.0


def _compare(task):
    chart_id, source_path, rendered_path, normalize_rendered = task
    try:
        source = _normalize_source(_rgb_on_white(Path(source_path)))
        rendered_source = _rgb_on_white(Path(rendered_path))
        rendered = (
            _normalize_source(rendered_source)
            if normalize_rendered
            else rendered_source.resize(
                (TARGET_SIZE, TARGET_SIZE), Image.Resampling.LANCZOS
            )
        )
        content_bboxes = _content_bbox(source), _content_bbox(rendered)
        source_small = source.resize((METRIC_SIZE, METRIC_SIZE), Image.Resampling.LANCZOS)
        rendered_small = rendered.resize((METRIC_SIZE, METRIC_SIZE), Image.Resampling.LANCZOS)
        source_arr = np.asarray(source_small, dtype=np.float32)
        rendered_arr = np.asarray(rendered_small, dtype=np.float32)
        mae = float(np.mean(np.abs(source_arr - rendered_arr)) / 255.0)
        source_gray = np.asarray(source_small.convert("L"), dtype=np.uint8)
        rendered_gray = np.asarray(rendered_small.convert("L"), dtype=np.uint8)
        ssim = float(structural_similarity(source_gray, rendered_gray, data_range=255))
        return {
            "id": chart_id,
            "ssim": ssim,
            "mae": mae,
            "bbox_iou": _bbox_iou(*content_bboxes),
            "error": "",
        }
    except Exception as exc:
        return {
            "id": chart_id,
            "ssim": "",
            "mae": "",
            "bbox_iou": "",
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--ids", help="comma-separated chart IDs to compare")
    parser.add_argument("--ids-file", help="text file containing one chart ID per line")
    parser.add_argument(
        "--report-suffix",
        default="",
        help="optional suffix for report filenames, e.g. raw_source",
    )
    parser.add_argument(
        "--normalize-rendered",
        action="store_true",
        help="resize and center rendered images like sources (for raw SVG diagnostics)",
    )
    parser.add_argument("--workers", type=int, default=max(1, min(10, os.cpu_count() or 1)))
    args = parser.parse_args()

    dataset = ROOT / args.dataset
    images = dataset / "images"
    charts = dataset / "charts"
    tasks = []
    missing_render = []
    selected_ids = {
        item.strip() for item in (args.ids or "").split(",") if item.strip()
    }
    if args.ids_file:
        selected_ids.update(
            line.strip()
            for line in Path(args.ids_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    for source in sorted(images.glob("*.png")):
        chart_id = source.stem
        if selected_ids and chart_id not in selected_ids:
            continue
        rendered = charts / chart_id / f"{chart_id}.png"
        if rendered.exists():
            tasks.append(
                (chart_id, str(source), str(rendered), args.normalize_rendered)
            )
        else:
            missing_render.append(chart_id)

    # ThreadPool avoids macOS sandbox semaphore restrictions. Pillow and NumPy
    # release the GIL for the expensive decode/resize/numeric operations here.
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(_compare, tasks, chunksize=20))

    valid = [row for row in rows if not row["error"]]
    valid.sort(key=lambda row: (row["ssim"], -row["mae"], row["bbox_iou"]))
    suffix = f"_{args.report_suffix}" if args.report_suffix else ""
    output_csv = dataset / f"consistency_report{suffix}.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "ssim", "mae", "bbox_iou", "error"])
        writer.writeheader()
        writer.writerows(valid + [row for row in rows if row["error"]])

    ssims = np.array([row["ssim"] for row in valid])
    maes = np.array([row["mae"] for row in valid])
    ious = np.array([row["bbox_iou"] for row in valid])
    summary = {
        "dataset": args.dataset,
        "compared": len(valid),
        "errors": len(rows) - len(valid),
        "missing_render": len(missing_render),
        "ssim": {key: float(value) for key, value in zip(
            ["min", "p01", "p05", "median", "mean"],
            [ssims.min(), *np.quantile(ssims, [0.01, 0.05, 0.5]), ssims.mean()],
        )} if len(ssims) else {},
        "mae": {key: float(value) for key, value in zip(
            ["max", "p99", "p95", "median", "mean"],
            [maes.max(), *np.quantile(maes, [0.99, 0.95, 0.5]), maes.mean()],
        )} if len(maes) else {},
        "bbox_iou": {key: float(value) for key, value in zip(
            ["min", "p01", "p05", "median", "mean"],
            [ious.min(), *np.quantile(ious, [0.01, 0.05, 0.5]), ious.mean()],
        )} if len(ious) else {},
        "screened_mismatch": sum(
            row["ssim"] < 0.55 and (row["mae"] > 0.10 or row["bbox_iou"] < 0.55)
            for row in valid
        ),
        "worst_ids": [row["id"] for row in valid[:20]],
        "report": str(output_csv),
    }
    output_json = dataset / f"consistency_summary{suffix}.json"
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
