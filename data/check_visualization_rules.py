#!/usr/bin/env python3
"""Audit cleaned Beagle SVGs for conservative visualization-rule violations.

Checks:
  * invalid/null numeric and paint attributes;
  * 512x512 canvas and root-level overflow clipping;
  * paired x/y axes for charts that expose one Cartesian axis;
  * Plotly Cartesian trace layers that are missing an available plot clipPath.

With --fix, only deterministic fixes are applied. Missing axes are reported but
not synthesized because tick locations/scales cannot be inferred safely.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from lxml import etree


BASE = Path(__file__).resolve().parent / "data" / "Beagle"
ALL_DATASETS = ["chartblocks", "fusion_clean", "graphiq_clean", "plotly_export"]
INVALID = {"null", "undefined", "nan", "infinity", "-infinity"}
X_AXIS_RE = re.compile(
    r"(?:^|[\s_-])x(?:axis|axes)(?:$|[\s_-])|xaxislayer|highcharts-xaxis|axis-grid\s+x",
    re.I,
)
Y_AXIS_RE = re.compile(
    r"(?:^|[\s_-])y(?:axis|axes)(?:$|[\s_-])|yaxislayer|highcharts-yaxis|axis-grid\s+y",
    re.I,
)
TRACE_RE = re.compile(r"(?:^|\s)(?:trace|scatter|bar|box|violin|histogram)(?:\s|$)", re.I)


def local_name(element) -> str:
    return etree.QName(element).localname


def has_clip_in_ancestry(element, root) -> bool:
    current = element
    while current is not None:
        if current.get("clip-path"):
            return True
        if current is root:
            break
        current = current.getparent()
    return False


def find_plot_clip(root):
    candidates = []
    for clip in root.xpath("//*[local-name()='clipPath']"):
        clip_id = clip.get("id")
        class_name = clip.get("class", "")
        if clip_id and ("plotclip" in class_name.lower() or "plot" in clip_id.lower()):
            if clip.xpath("./*[local-name()='rect']"):
                candidates.append(clip_id)
    # A document with several plot clips normally contains subplots. Applying
    # the first clip to every trace would move or hide valid marks, so only
    # auto-fix the unambiguous single-plot case.
    return candidates[0] if len(candidates) == 1 else None


def audit_one(task):
    dataset_name, chart_id, path_string, fix = task
    path = Path(path_string)
    result = {
        "dataset": dataset_name,
        "id": chart_id,
        "parse_error": "",
        "invalid_attributes": 0,
        "canvas_not_512": 0,
        "root_overflow_unclipped": 0,
        "has_x_axis": 0,
        "has_y_axis": 0,
        "missing_cartesian_axis": "",
        "unclipped_plotly_layers": 0,
        "fixed": 0,
    }
    try:
        parser = etree.XMLParser(recover=False, huge_tree=True)
        root = etree.fromstring(path.read_bytes(), parser=parser)
    except Exception as exc:
        result["parse_error"] = f"{type(exc).__name__}: {exc}"
        return result

    changed = False
    for element in root.iter():
        for attr, value in list(element.attrib.items()):
            if value.strip().lower() in INVALID:
                result["invalid_attributes"] += 1
                if fix:
                    del element.attrib[attr]
                    changed = True
        if "style" in element.attrib:
            style = element.attrib["style"]
            cleaned_style = re.sub(
                r"(?:^|;)\s*[-\w]+\s*:\s*(?:null|undefined|nan|[-+]?infinity)\s*(?=;|$)",
                "",
                style,
                flags=re.I,
            ).strip(" ;")
            if cleaned_style != style.strip(" ;"):
                result["invalid_attributes"] += 1
                if fix:
                    if cleaned_style:
                        element.attrib["style"] = cleaned_style
                    else:
                        del element.attrib["style"]
                    changed = True

    view_box = root.get("viewBox", "").replace(",", " ").split()
    width = root.get("width", "")
    height = root.get("height", "")
    try:
        valid_view_box = len(view_box) == 4 and float(view_box[2]) > 0 and float(view_box[3]) > 0
    except ValueError:
        valid_view_box = False
    # The rendered viewport must be 512px square. A different valid viewBox is
    # intentional for aspect-preserving raw fallbacks and is not a violation.
    if not valid_view_box or width != "512" or height != "512":
        result["canvas_not_512"] = 1
        if fix:
            root.set("viewBox", "0 0 512 512")
            root.set("width", "512")
            root.set("height", "512")
            changed = True

    style_parts = {
        item.split(":", 1)[0].strip().lower(): item.split(":", 1)[1].strip()
        for item in root.get("style", "").split(";")
        if ":" in item
    }
    overflow_value = root.get("overflow", "") or style_parts.get("overflow", "")
    if overflow_value and overflow_value.lower() not in {"hidden", "clip", "auto"}:
        result["root_overflow_unclipped"] = 1
        if fix:
            root.set("overflow", "hidden")
            changed = True

    class_text = " ".join(element.get("class", "") for element in root.iter())
    has_x = bool(X_AXIS_RE.search(class_text))
    has_y = bool(Y_AXIS_RE.search(class_text))
    result["has_x_axis"] = int(has_x)
    result["has_y_axis"] = int(has_y)
    if has_x != has_y:
        result["missing_cartesian_axis"] = "y" if has_x else "x"

    # Restrict automatic axis-boundary clipping to Plotly trace markup. Generic
    # SVG classes called "series" are too ambiguous (pie/map series included).
    if dataset_name == "plotly_export" and has_x and has_y:
        plot_clip_id = find_plot_clip(root)
        for element in root.iter():
            if not TRACE_RE.search(element.get("class", "")):
                continue
            if has_clip_in_ancestry(element, root):
                continue
            result["unclipped_plotly_layers"] += 1
            if fix and plot_clip_id:
                # Prefer the nearest Plotly plot container so all marks in the
                # same subplot receive one consistent boundary.
                target = element
                current = element
                while current is not None and current is not root:
                    if "plot" in current.get("class", "").split():
                        target = current
                        break
                    current = current.getparent()
                target.set("clip-path", f"url(#{plot_clip_id})")
                changed = True

    if fix and changed:
        output = etree.tostring(root, encoding="unicode", pretty_print=False)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False
        ) as temp:
            temp.write(output)
            temp_path = Path(temp.name)
        temp_path.replace(path)
        result["fixed"] = 1
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        default=",".join(ALL_DATASETS),
        help="comma-separated dataset names",
    )
    parser.add_argument("--workers", type=int, default=max(1, min(10, os.cpu_count() or 1)))
    parser.add_argument("--fix", action="store_true")
    args = parser.parse_args()

    dataset_names = [item.strip() for item in args.datasets.split(",") if item.strip()]
    tasks = []
    for dataset_name in dataset_names:
        charts = BASE / dataset_name / "charts"
        for chart_dir in sorted(charts.iterdir()):
            if not chart_dir.is_dir():
                continue
            cleaned = chart_dir / "cleaned_svg.txt"
            if cleaned.exists():
                tasks.append((dataset_name, chart_dir.name, str(cleaned), args.fix))

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(audit_one, tasks, chunksize=20))

    for dataset_name in dataset_names:
        dataset_rows = [row for row in rows if row["dataset"] == dataset_name]
        report = BASE / dataset_name / "visualization_rules_report.csv"
        with report.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(dataset_rows[0]) if dataset_rows else [])
            if dataset_rows:
                writer.writeheader()
                writer.writerows(dataset_rows)
        summary = {
            "dataset": dataset_name,
            "checked": len(dataset_rows),
            "parse_errors": sum(bool(row["parse_error"]) for row in dataset_rows),
            "invalid_attributes": sum(row["invalid_attributes"] for row in dataset_rows),
            "canvas_not_512": sum(row["canvas_not_512"] for row in dataset_rows),
            "root_overflow_unclipped": sum(
                row["root_overflow_unclipped"] for row in dataset_rows
            ),
            "missing_cartesian_axis": sum(
                bool(row["missing_cartesian_axis"]) for row in dataset_rows
            ),
            "unclipped_plotly_layers": sum(
                row["unclipped_plotly_layers"] for row in dataset_rows
            ),
            "files_fixed": sum(row["fixed"] for row in dataset_rows),
            "report": str(report),
        }
        summary_path = BASE / dataset_name / "visualization_rules_summary.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
