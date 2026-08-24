#!/usr/bin/env python3
"""Repair ECharts full circles encoded as a single zero-length SVG arc.

Some generated ECharts SVG paths use ``M x y A rx ry 0 1 1 x y``. Browsers
may display those paths as circles, but standards-compliant rasterizers treat
an arc whose start and end points coincide as empty. Replace only that exact
shape with an equivalent ellipse while retaining styling and transforms.
"""

from __future__ import annotations

import argparse
import copy
import math
import re
import shutil
import tempfile
from pathlib import Path

from lxml import etree


BASE = Path(__file__).resolve().parent / "data" / "Beagle" / "echarts"
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
FULL_CIRCLE = re.compile(
    rf"^\s*M\s*({NUMBER})[ ,]+({NUMBER})\s*"
    rf"A\s*({NUMBER})[ ,]+({NUMBER})[ ,]+({NUMBER})[ ,]+1[ ,]+1[ ,]+"
    rf"({NUMBER})[ ,]+({NUMBER})\s*[zZ]?\s*$"
)


def fmt(value: float) -> str:
    if abs(value) < 1e-12:
        value = 0.0
    return f"{value:.12g}"


def repair_file(path: Path) -> tuple[int, int]:
    parser = etree.XMLParser(remove_blank_text=False, huge_tree=True)
    root = etree.parse(str(path), parser).getroot()
    repaired = 0
    paint_order_repaired = 0

    for node in list(root.iter()):
        if etree.QName(node).localname != "path":
            continue
        match = FULL_CIRCLE.match(node.get("d", ""))
        if not match:
            continue
        x1, y1, rx, ry, rotation, x2, y2 = map(float, match.groups())
        if (
            rx <= 0
            or ry <= 0
            or not math.isclose(rotation % 360, 0.0, abs_tol=1e-9)
            or not math.isclose(x1, x2, abs_tol=1e-7)
            or not math.isclose(y1, y2, abs_tol=1e-7)
        ):
            continue

        namespace = etree.QName(node).namespace
        tag = f"{{{namespace}}}ellipse" if namespace else "ellipse"
        replacement = etree.Element(tag)
        replacement.set("cx", fmt(x1 - rx))
        replacement.set("cy", fmt(y1))
        replacement.set("rx", fmt(rx))
        replacement.set("ry", fmt(ry))
        for key, value in node.attrib.items():
            if key != "d":
                replacement.set(key, value)
        replacement.text = node.text
        replacement.tail = node.tail
        parent = node.getparent()
        parent.replace(node, replacement)
        repaired += 1

    # CairoSVG does not consistently honor paint-order="stroke". Split the
    # element into a stroke-only copy followed by a fill-only original so the
    # visual stacking is explicit and renderer-independent.
    stroke_keys = {
        "stroke",
        "stroke-width",
        "stroke-opacity",
        "stroke-linecap",
        "stroke-linejoin",
        "stroke-miterlimit",
        "stroke-dasharray",
        "stroke-dashoffset",
    }
    for node in list(root.iter()):
        paint_order = node.get("paint-order", "")
        stroke = node.get("stroke", "")
        if "stroke" not in paint_order or not stroke or stroke == "none":
            continue
        parent = node.getparent()
        if parent is None:
            continue
        stroke_copy = copy.deepcopy(node)
        stroke_copy.set("fill", "none")
        stroke_copy.attrib.pop("paint-order", None)
        stroke_copy.attrib.pop("id", None)
        for key in stroke_keys:
            node.attrib.pop(key, None)
        node.attrib.pop("paint-order", None)
        parent.insert(parent.index(node), stroke_copy)
        paint_order_repaired += 1

    if repaired or paint_order_repaired:
        output = etree.tostring(root, encoding="utf-8", xml_declaration=False)
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as tmp:
            tmp.write(output)
            temp_path = Path(tmp.name)
        temp_path.replace(path)
    return repaired, paint_order_repaired


def main() -> int:
    argp = argparse.ArgumentParser()
    argp.add_argument("--ids", default="", help="optional comma-separated chart IDs")
    argp.add_argument(
        "--sync-svg",
        action="store_true",
        help="copy repaired cleaned_svg.txt to svg.txt for rendering",
    )
    args = argp.parse_args()
    selected = {item.strip() for item in args.ids.split(",") if item.strip()}

    files = 0
    shapes = 0
    paint_order_shapes = 0
    charts = BASE / "charts"
    synced = 0
    for chart_dir in sorted(charts.iterdir()):
        if not chart_dir.is_dir() or (selected and chart_dir.name not in selected):
            continue
        cleaned = chart_dir / "cleaned_svg.txt"
        if not cleaned.is_file():
            continue
        count, paint_count = repair_file(cleaned)
        if count or paint_count:
            files += 1
            shapes += count
            paint_order_shapes += paint_count
            print(
                f"repaired {chart_dir.name}/cleaned_svg.txt: "
                f"circles={count} paint_order={paint_count}"
            )
        # A prior interrupted run may already have repaired the ellipses. The
        # presence check makes --sync-svg resumable without touching charts
        # that do not need this Cairo compatibility fallback.
        cleaned_bytes = cleaned.read_bytes()
        if args.sync_svg and (b"<ellipse" in cleaned_bytes or count or paint_count):
            shutil.copyfile(cleaned, chart_dir / "svg.txt")
            synced += 1
    print(
        f"files_repaired={files} circles_repaired={shapes} "
        f"paint_order_repaired={paint_order_shapes} svg_synced={synced}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
