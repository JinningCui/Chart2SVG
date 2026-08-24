#!/usr/bin/env python3
"""Canonicalize degenerate Chartblocks axis-domain paths.

Chartblocks emits paths such as ``M2,0V0H618V0``. They are valid in browsers,
but some SVG renderers drop the zero-length segments and fail to paint the axis.
This script replaces only bottom/top and left/right domain paths with equivalent
explicit horizontal or vertical lines.
"""

from __future__ import annotations

import argparse
import re
import shutil
import tempfile
from pathlib import Path

from lxml import etree


BASE = Path(__file__).resolve().parent / "data" / "Beagle" / "chartblocks" / "charts"
NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def axis_kind(element) -> str:
    current = element.getparent()
    while current is not None:
        classes = set(current.get("class", "").split())
        if "axis" in classes:
            if classes & {"bottom", "top"}:
                return "horizontal"
            if classes & {"left", "right"}:
                return "vertical"
        current = current.getparent()
    return ""


def repair(path: Path) -> bool:
    root = etree.fromstring(path.read_bytes(), parser=etree.XMLParser(huge_tree=True))
    changed = False
    for element in root.xpath(
        "//*[(local-name()='path' or local-name()='line') and contains(concat(' ', normalize-space(@class), ' '), ' domain ')]"
    ):
        geometry = element.get("d", "") or " ".join(
            element.get(key, "") for key in ("x1", "y1", "x2", "y2")
        )
        values = [float(item) for item in NUMBER_RE.findall(geometry)]
        if not values:
            continue
        kind = axis_kind(element)
        if kind == "horizontal":
            extent = max(values)
            coordinates = {"x1": "0", "y1": "-0.5", "x2": f"{extent:g}", "y2": "-0.5"}
        elif kind == "vertical":
            extent = max(values)
            coordinates = {"x1": "0", "y1": "0", "x2": "0", "y2": f"{extent:g}"}
        else:
            continue
        if etree.QName(element).localname != "line" or any(
            element.get(key) != value for key, value in coordinates.items()
        ):
            namespace = etree.QName(element).namespace
            element.tag = f"{{{namespace}}}line" if namespace else "line"
            element.attrib.pop("d", None)
            for key, value in coordinates.items():
                element.set(key, value)
            element.set("fill", "none")
            element.set("stroke", "#bcbcbc")
            element.set("stroke-width", "1")
            style = element.get("style", "")
            style = re.sub(r"(?:^|;)\s*stroke(?:-width)?\s*:[^;]*", "", style)
            style = re.sub(r"(?:^|;)\s*shape-rendering\s*:[^;]*", "", style)
            style = style.strip(" ;")
            if style:
                element.set("style", style)
            elif "style" in element.attrib:
                del element.attrib["style"]
            changed = True
    if changed:
        root.set("data-preserve-rendering", "true")
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
    selection.add_argument("--all", action="store_true", help="scan every chart")
    parser.add_argument("--sync-svg", action="store_true", help="copy repaired cleaned SVG to svg.txt")
    parser.add_argument("--report", type=Path, help="write repaired IDs, one per line")
    args = parser.parse_args()
    ids = (
        sorted(path.parent.name for path in BASE.glob("*/cleaned_svg.txt"))
        if args.all
        else [item.strip() for item in args.ids.split(",") if item.strip()]
    )
    repaired = []
    for chart_id in ids:
        path = BASE / chart_id / "cleaned_svg.txt"
        if path.exists() and repair(path):
            repaired.append(chart_id)
            if args.sync_svg:
                shutil.copyfile(path, path.parent / "svg.txt")
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
