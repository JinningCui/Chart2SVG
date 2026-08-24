#!/usr/bin/env python3
"""Restore selected raw Beagle SVGs as 512px, aspect-preserving cleaned SVGs.

This is a conservative fallback for charts where structural cleanup changes the
rendered appearance. It keeps the original SVG tree intact and only normalizes
the outer viewport to 512x512 with xMidYMid meet semantics.
"""

from __future__ import annotations

import argparse
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

from lxml import etree


BASE = Path(__file__).resolve().parent / "data" / "Beagle"
ARCHIVES = {
    "chartblocks": ("chartblocks.zip", "chartblocks"),
    "fusion_clean": ("fusioncharts.zip", "fusion_clean"),
    "graphiq_clean": ("graphiq.zip", "graphiq_clean"),
    "plotly_export": ("plotly.zip", "plotly_export"),
}
NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def numeric_length(value: str | None) -> float | None:
    if not value:
        return None
    match = NUMBER_RE.match(value.strip())
    return float(match.group()) if match else None


def normalize(source: bytes) -> bytes:
    if b"xlink:" in source:
        opening_end = source.find(b">")
        opening = source[:opening_end]
        if b"xmlns:xlink" not in opening:
            source = source[:4] + b' xmlns:xlink="http://www.w3.org/1999/xlink"' + source[4:]
    root = etree.fromstring(source, parser=etree.XMLParser(recover=False, huge_tree=True))
    view_box = root.get("viewBox", "").replace(",", " ").split()
    if len(view_box) != 4:
        width = numeric_length(root.get("width"))
        height = numeric_length(root.get("height"))
        if not width or not height:
            raise ValueError("source SVG has neither a valid viewBox nor numeric dimensions")
        root.set("viewBox", f"0 0 {width:g} {height:g}")
    root.set("x", "0")
    root.set("y", "0")
    root.set("width", "512")
    root.set("height", "512")
    root.set("preserveAspectRatio", "xMidYMid meet")
    namespace = etree.QName(root).namespace or "http://www.w3.org/2000/svg"
    wrapper = etree.Element(
        f"{{{namespace}}}svg",
        nsmap={None: namespace},
        viewBox="0 0 512 512",
        width="512",
        height="512",
        preserveAspectRatio="xMidYMid meet",
    )
    wrapper.set("data-raw-normalized", "true")
    wrapper.set("data-preserve-rendering", "true")
    etree.SubElement(
        wrapper,
        f"{{{namespace}}}rect",
        x="0",
        y="0",
        width="512",
        height="512",
        fill="white",
    )
    wrapper.append(root)
    return etree.tostring(wrapper, encoding="utf-8", xml_declaration=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=ARCHIVES)
    parser.add_argument("--ids", required=True, help="comma-separated chart IDs")
    parser.add_argument("--sync-svg", action="store_true", help="also update svg.txt")
    args = parser.parse_args()
    ids = [item.strip() for item in args.ids.split(",") if item.strip()]
    archive_name, member_root = ARCHIVES[args.dataset]
    dataset = BASE / args.dataset
    restored, failed = [], []
    with zipfile.ZipFile(BASE / archive_name) as archive:
        for chart_id in ids:
            member = f"{member_root}/charts/{chart_id}/svg.txt"
            target = dataset / "charts" / chart_id / "cleaned_svg.txt"
            try:
                output = normalize(archive.read(member))
                with tempfile.NamedTemporaryFile("wb", dir=target.parent, delete=False) as tmp:
                    tmp.write(output)
                    temp_path = Path(tmp.name)
                temp_path.replace(target)
                if args.sync_svg:
                    shutil.copyfile(target, target.parent / "svg.txt")
                restored.append(chart_id)
            except Exception as exc:
                failed.append((chart_id, str(exc)))
    print(f"selected={len(ids)} restored={len(restored)} failed={len(failed)}")
    print("restored_ids=" + ",".join(restored))
    for chart_id, error in failed:
        print(f"failed {chart_id}: {error}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
