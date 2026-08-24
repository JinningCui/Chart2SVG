#!/usr/bin/env python3
"""Assign an explicit CJK-capable font stack to SVG text containing CJK glyphs."""

from __future__ import annotations

import argparse
import re
import shutil
import tempfile
from pathlib import Path

from lxml import etree


BASE = Path(__file__).resolve().parent / "data" / "Beagle"
CJK_RE = re.compile(
    "[\u2e80-\u2fff\u3040-\u30ff\u31f0-\u31ff\u3400-\u4dbf"
    "\u4e00-\u9fff\uf900-\ufaff]"
)
FONT_STACK = "'PingFang TC','PingFang SC','Songti TC','Songti SC',sans-serif"


def repair(path: Path) -> bool:
    root = etree.fromstring(path.read_bytes(), parser=etree.XMLParser(huge_tree=True))
    changed = False
    for element in root.xpath("//*[local-name()='text']"):
        text = "".join(element.itertext())
        if not CJK_RE.search(text):
            continue
        style = element.get("style", "")
        declaration = f"font-family: {FONT_STACK}"
        if re.search(r"(?:^|;)\s*font-family\s*:", style, flags=re.I):
            updated = re.sub(
                r"((?:^|;)\s*font-family\s*:)\s*[^;]*",
                rf"\1 {FONT_STACK}",
                style,
                flags=re.I,
            )
        else:
            updated = f"{style.rstrip(' ;')}; {declaration}".lstrip("; ")
        if updated != style or element.get("font-family") != FONT_STACK:
            element.set("style", updated)
            element.set("font-family", FONT_STACK)
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
    parser.add_argument("dataset")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--ids", help="comma-separated chart IDs")
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--sync-svg", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    charts = BASE / args.dataset / "charts"
    ids = (
        sorted(path.parent.name for path in charts.glob("*/cleaned_svg.txt"))
        if args.all
        else [item.strip() for item in args.ids.split(",") if item.strip()]
    )
    repaired = []
    for chart_id in ids:
        cleaned = charts / chart_id / "cleaned_svg.txt"
        if cleaned.exists() and repair(cleaned):
            repaired.append(chart_id)
            if args.sync_svg:
                shutil.copyfile(cleaned, cleaned.parent / "svg.txt")
    if args.report:
        args.report.write_text("\n".join(repaired) + ("\n" if repaired else ""), encoding="utf-8")
    print(f"selected={len(ids)} repaired={len(repaired)}")
    if len(repaired) <= 100:
        print("repaired_ids=" + ",".join(repaired))
    elif args.report:
        print(f"repaired_ids_report={args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
