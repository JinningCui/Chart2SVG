#!/usr/bin/env python3
"""Build standard Qwen conversation JSON from per-chart train_svg.txt files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SYSTEM_PROMPT = (
    "You are a world-class SVG Expert and Data Visualization Engineer. "
    "Your primary objective is to interpret rasterized chart images and "
    "reconstruct them into high-quality, semantically correct SVG code."
)
DEFAULT_DATASETS = (
    "chartblocks,fusion_clean,graphiq_clean,plotly_export,echarts"
)


def output_name(dataset: str) -> str:
    return "graphiq" if dataset == "graphiq_clean" else dataset


def process_dataset(base_dir: Path, dataset: str) -> tuple[int, int]:
    charts_dir = base_dir / dataset / "charts"
    output_dir = base_dir / "train_json"
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    skipped = 0

    for chart_dir in sorted(charts_dir.iterdir()):
        if not chart_dir.is_dir():
            continue
        semantic_path = chart_dir / "train_svg.txt"
        png_path = chart_dir / f"{chart_dir.name}.png"
        if not semantic_path.is_file() or not png_path.is_file():
            skipped += 1
            continue
        semantic_svg = semantic_path.read_text(encoding="utf-8").strip()
        if not semantic_svg:
            skipped += 1
            continue
        records.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": "<image>Convert this image to SVG code."},
                    {"role": "assistant", "content": semantic_svg},
                ],
                "images": [str(png_path.resolve())],
            }
        )

    destination = output_dir / f"{output_name(dataset)}.json"
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
    print(
        f"dataset={dataset} records={len(records)} skipped={skipped} "
        f"output={destination}"
    )
    return len(records), skipped


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "Beagle_Plus",
    )
    parser.add_argument("--datasets", default=DEFAULT_DATASETS)
    args = parser.parse_args()
    datasets = [name.strip() for name in args.datasets.split(",") if name.strip()]
    for dataset in datasets:
        process_dataset(args.base_dir.resolve(), dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
