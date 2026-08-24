#!/usr/bin/env python3
"""Restore FusionCharts gauges and gradient-map charts with raw SVG fidelity.

These chart families depend on dense transformation/path structures that are
easy to damage during generic syntactic cleanup. The fallback preserves the raw
tree and only wraps it in a white 512px aspect-preserving viewport.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

from restore_raw_normalized_cleaned import normalize


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "data" / "Beagle"
DATASET = BASE / "fusion_clean"
ARCHIVE = BASE / "fusioncharts.zip"
MARKERS = ("-pointers", "fc-gradient-legend")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=DATASET / "repaired_complex_chart_ids.txt",
    )
    args = parser.parse_args()
    selected = []
    for cleaned in sorted((DATASET / "charts").glob("*/cleaned_svg.txt")):
        content = cleaned.read_text(encoding="utf-8", errors="ignore")
        if any(marker in content for marker in MARKERS):
            selected.append(cleaned.parent.name)

    repaired, failed = [], []
    with zipfile.ZipFile(ARCHIVE) as archive:
        for chart_id in selected:
            chart_dir = DATASET / "charts" / chart_id
            cleaned = chart_dir / "cleaned_svg.txt"
            try:
                source = archive.read(f"fusion_clean/charts/{chart_id}/svg.txt")
                output = normalize(source)
                with tempfile.NamedTemporaryFile("wb", dir=chart_dir, delete=False) as tmp:
                    tmp.write(output)
                    temp_path = Path(tmp.name)
                temp_path.replace(cleaned)
                shutil.copyfile(cleaned, chart_dir / "svg.txt")
                repaired.append(chart_id)
            except Exception as exc:
                failed.append((chart_id, str(exc)))
    args.report.write_text("\n".join(repaired) + ("\n" if repaired else ""), encoding="utf-8")
    print(f"selected={len(selected)} repaired={len(repaired)} failed={len(failed)}")
    print(f"report={args.report}")
    for chart_id, error in failed:
        print(f"failed {chart_id}: {error}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
