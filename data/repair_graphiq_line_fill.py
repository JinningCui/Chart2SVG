#!/usr/bin/env python3
"""Prevent Graphiq line-series paths from falling back to black SVG fill."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from lxml import etree


BASE = Path(__file__).resolve().parent / "data" / "Beagle" / "graphiq_clean" / "charts"
INVALID = {"", "null", "undefined", "nan", "infinity", "-infinity"}


def repair(path: Path) -> bool:
    root = etree.fromstring(path.read_bytes(), parser=etree.XMLParser(huge_tree=True))
    changed = False
    for element in root.xpath(
        "//*[local-name()='path' and contains(concat(' ', normalize-space(@class), ' '), ' gfx-ln-path ')]"
    ):
        fill = element.get("fill", "").strip().lower()
        if fill in INVALID:
            element.set("fill", "none")
            changed = True
    if changed:
        output = etree.tostring(root, encoding="utf-8")
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as tmp:
            tmp.write(output)
            temp_path = Path(tmp.name)
        temp_path.replace(path)
    return changed


def main() -> int:
    parser = argparse.ArgumentParser()
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--ids", help="comma-separated chart IDs")
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--sync-svg", action="store_true")
    parser.add_argument(
        "--report",
        type=Path,
        default=BASE.parent / "repaired_line_fill_ids.txt",
    )
    args = parser.parse_args()
    ids = (
        sorted(path.parent.name for path in BASE.glob("*/cleaned_svg.txt"))
        if args.all
        else [item.strip() for item in args.ids.split(",") if item.strip()]
    )
    repaired = []
    for chart_id in ids:
        cleaned = BASE / chart_id / "cleaned_svg.txt"
        if cleaned.exists() and repair(cleaned):
            repaired.append(chart_id)
            if args.sync_svg:
                shutil.copyfile(cleaned, cleaned.parent / "svg.txt")
    args.report.write_text("\n".join(repaired) + ("\n" if repaired else ""), encoding="utf-8")
    print(f"selected={len(ids)} repaired={len(repaired)}")
    print(f"report={args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
