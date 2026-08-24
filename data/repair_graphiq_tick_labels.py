#!/usr/bin/env python3
"""Restore formatted x-axis labels for known Graphiq source-SVG ID mismatches."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from lxml import etree


BASE = Path(__file__).resolve().parent / "data" / "Beagle" / "graphiq_clean" / "charts"
MAPPINGS = {
    "0a7d7e7b-3870-4da2-9bbc-808b205b5bc3": {
        0: "Q1 '16",
        1: "Q2 '16",
        2: "Q3 '16",
        3: "Q4 '16",
    },
    "0cc0d53a-afec-447e-95f3-36cf0f0e2508": {
        2: "Q1 '05",
        19: "Q2 '09",
        32: "Q2 '12",
        47: "Q4 '16",
    },
}


def repair(chart_id: str) -> None:
    path = BASE / chart_id / "cleaned_svg.txt"
    root = etree.fromstring(path.read_bytes(), parser=etree.XMLParser(huge_tree=True))
    ticks = []
    for group in root.xpath(
        "//*[local-name()='g' and contains(concat(' ', normalize-space(@class), ' '), ' tick ')]"
    ):
        value = "".join(group.itertext()).strip()
        if value.startswith("$"):
            break
        ticks.append(group)
    mapping = MAPPINGS[chart_id]
    for index, group in enumerate(ticks):
        if index not in mapping:
            group.set("display", "none")
            continue
        group.attrib.pop("display", None)
        text_nodes = group.xpath(".//*[local-name()='text']")
        if text_nodes:
            text_nodes[0].text = mapping[index]
            for child in list(text_nodes[0]):
                text_nodes[0].remove(child)
    output = etree.tostring(root, encoding="utf-8")
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as tmp:
        tmp.write(output)
        temp_path = Path(tmp.name)
    temp_path.replace(path)
    shutil.copyfile(path, path.parent / "svg.txt")


def main() -> int:
    for chart_id in MAPPINGS:
        repair(chart_id)
    print(f"repaired={len(MAPPINGS)}")
    print("repaired_ids=" + ",".join(MAPPINGS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
