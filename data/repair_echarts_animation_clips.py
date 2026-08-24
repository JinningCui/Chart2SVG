#!/usr/bin/env python3
"""Repair ECharts SVGs captured at the first frame of a reveal animation.

Affected exports contain complete series geometry inside a clipPath whose
animated width or height is still zero. This script replaces only degenerate
clip paths referenced by visible groups with a canvas-sized clip and adds the
white ECharts canvas background. The repaired result is written to both
cleaned_svg.txt and svg.txt.
"""

from __future__ import annotations

import argparse
import re
import tempfile
from pathlib import Path

from lxml import etree


BASE = Path(__file__).resolve().parent / "data" / "Beagle" / "echarts" / "charts"
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
ZERO_REVEAL = re.compile(
    rf"^\s*M\s*({NUMBER})[ ,]+({NUMBER})\s*"
    rf"l\s*0[ ,]+0\s*l\s*0[ ,]+({NUMBER})\s*l\s*0[ ,]+0\s*[zZ]\s*$",
    re.IGNORECASE,
)


def atomic_write(path: Path, content: bytes) -> None:
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as tmp:
        tmp.write(content)
        temp_path = Path(tmp.name)
    temp_path.replace(path)


def repair(chart_dir: Path) -> int:
    source = chart_dir / "cleaned_svg.txt"
    if not source.is_file():
        source = chart_dir / "svg.txt"
    if not source.is_file():
        return 0

    root = etree.parse(str(source), etree.XMLParser(huge_tree=True)).getroot()
    repaired = 0
    for node in root.iter():
        if etree.QName(node).localname != "clipPath":
            continue
        for child in node:
            if etree.QName(child).localname != "path":
                continue
            if ZERO_REVEAL.match(child.get("d", "")):
                child.set("d", "M0 0H512V512H0Z")
                repaired += 1

    if not repaired:
        return 0

    # ECharts examples are rendered on a white page. A transparent root looks
    # black in alpha-aware viewers and is not equivalent to the source image.
    namespace = etree.QName(root).namespace
    rect_tag = f"{{{namespace}}}rect" if namespace else "rect"
    background = etree.Element(
        rect_tag, x="0", y="0", width="512", height="512", fill="white"
    )
    root.insert(0, background)
    root.set("viewBox", "0 0 512 512")
    root.set("width", "512")
    root.set("height", "512")
    output = etree.tostring(root, encoding="utf-8", xml_declaration=False)
    atomic_write(chart_dir / "cleaned_svg.txt", output)
    atomic_write(chart_dir / "svg.txt", output)
    return repaired


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", default="", help="optional comma-separated chart IDs")
    args = parser.parse_args()
    selected = {item.strip() for item in args.ids.split(",") if item.strip()}

    files = clips = 0
    for chart_dir in sorted(BASE.iterdir()):
        if not chart_dir.is_dir() or (selected and chart_dir.name not in selected):
            continue
        count = repair(chart_dir)
        if count:
            files += 1
            clips += count
            print(f"repaired {chart_dir.name}: degenerate_clips={count}")
    print(f"files_repaired={files} clips_repaired={clips}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
