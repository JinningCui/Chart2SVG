#!/usr/bin/env bash
# Generate semantic SVG token files and merged Qwen JSON for Beagle datasets.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
BASE_DIR="${BEAGLE_PLUS_DIR:-$SCRIPT_DIR/data/Beagle_Plus}"
DATASETS="${BEAGLE_DATASETS:-chartblocks,fusion_clean,graphiq_clean,plotly_export,echarts}"
WORKERS="${GEN_SVG_WORKERS:-8}"

usage() {
  cat <<'EOF'
Usage: gen_semantic_svg.sh [options]

Run gen_svg_qwen.py followed by gen_std_svg_json.py.

Options:
  --base-dir PATH     Dataset root containing dataset folders.
  --datasets NAMES    Comma-separated dataset names.
  --workers N         Worker processes used by gen_svg_qwen.py.
  --python PATH       Python executable.
  -h, --help          Show this help.

Environment equivalents:
  BEAGLE_PLUS_DIR, BEAGLE_DATASETS, GEN_SVG_WORKERS, PYTHON_BIN

Example:
  ./gen_semantic_svg.sh \
    --base-dir /path/to/Beagle_Plus \
    --datasets echarts,chartblocks \
    --workers 8
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-dir)
      [[ $# -ge 2 ]] || { echo "Missing value for --base-dir" >&2; exit 2; }
      BASE_DIR="$2"
      shift 2
      ;;
    --datasets)
      [[ $# -ge 2 ]] || { echo "Missing value for --datasets" >&2; exit 2; }
      DATASETS="$2"
      shift 2
      ;;
    --workers)
      [[ $# -ge 2 ]] || { echo "Missing value for --workers" >&2; exit 2; }
      WORKERS="$2"
      shift 2
      ;;
    --python)
      [[ $# -ge 2 ]] || { echo "Missing value for --python" >&2; exit 2; }
      PYTHON_BIN="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -d "$BASE_DIR" ]]; then
  echo "Dataset root not found: $BASE_DIR" >&2
  exit 1
fi

BASE_DIR="$(cd "$BASE_DIR" && pwd)"

# CairoSVG on macOS may need the Conda Cairo dynamic libraries. Override this
# location with CAIRO_LIBRARY_DIR when using another environment.
CAIRO_LIBRARY_DIR="${CAIRO_LIBRARY_DIR:-/opt/anaconda3/envs/sketchpad/lib}"
if [[ -d "$CAIRO_LIBRARY_DIR" ]]; then
  export DYLD_FALLBACK_LIBRARY_PATH="$CAIRO_LIBRARY_DIR${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}"
fi

echo "[1/2] Generate semantic SVG: datasets=$DATASETS base_dir=$BASE_DIR"
"$PYTHON_BIN" "$SCRIPT_DIR/gen_svg_qwen.py" \
  --base-dir "$BASE_DIR" \
  --datasets "$DATASETS" \
  --workers "$WORKERS"

echo "[2/2] Generate standard Qwen JSON"
"$PYTHON_BIN" "$SCRIPT_DIR/gen_std_svg_json.py" \
  --base-dir "$BASE_DIR" \
  --datasets "$DATASETS"

echo "Semantic SVG generation completed: $BASE_DIR/train_json"
