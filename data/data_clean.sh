#!/usr/bin/env bash
# One-shot Beagle SVG cleaning, dataset-specific repair, optimization and render.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BEAGLE_DIR="$SCRIPT_DIR/data/Beagle"

PYTHON_BIN="${PYTHON_BIN:-python3}"
NODE_BIN="${NODE_BIN:-node}"
DATASETS="${BEAGLE_DATASETS:-chartblocks,fusion_clean,graphiq_clean,plotly_export,echarts}"
DATASETS="${DATASETS// /}"
WORKERS="${BEAGLE_WORKERS:-10}"
RENDER_SHARDS="${RENDER_SHARDS:-2}"
RUN_CONSISTENCY_CHECK="${RUN_CONSISTENCY_CHECK:-1}"
SVGO_MULTIPASS_VALUE="${SVGO_MULTIPASS:-false}"
SVGO_MAX_BYTES_VALUE="${SVGO_MAX_BYTES:-1000000}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found: $PYTHON_BIN" >&2
  exit 1
fi
if ! command -v "$NODE_BIN" >/dev/null 2>&1; then
  echo "Node.js not found: $NODE_BIN" >&2
  exit 1
fi

NODE_PATH_VALUE="${NODE_PATH:-}"
if [[ -z "$NODE_PATH_VALUE" ]]; then
  for candidate in "$PROJECT_DIR/node_modules" "$PROJECT_DIR/divi/node_modules" "$PROJECT_DIR/rebuttal/node_modules"; do
    if [[ -f "$candidate/svgo/package.json" ]]; then
      NODE_PATH_VALUE="$candidate"
      break
    fi
  done
fi
if [[ -z "$NODE_PATH_VALUE" ]]; then
  echo "Cannot locate the svgo package. Set NODE_PATH to its node_modules directory." >&2
  exit 1
fi

CAIRO_LIBRARY_DIR="${CAIRO_LIBRARY_DIR:-/opt/anaconda3/envs/sketchpad/lib}"
if [[ -d "$CAIRO_LIBRARY_DIR" ]]; then
  export DYLD_FALLBACK_LIBRARY_PATH="$CAIRO_LIBRARY_DIR${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}"
fi

dataset_enabled() {
  case ",$DATASETS," in
    *",$1,"*) return 0 ;;
    *) return 1 ;;
  esac
}

run_python() {
  "$PYTHON_BIN" "$@"
}

echo "[1/6] Generate syntactic SVGs: $DATASETS"
export BEAGLE_DATASETS="$DATASETS"
export BEAGLE_FORCE="${BEAGLE_FORCE:-1}"
export BEAGLE_WORKERS="$WORKERS"
"$PYTHON_BIN" "$SCRIPT_DIR/generate_syntactic_svg.py"

echo "[2/6] Apply deterministic rule and dataset-specific repairs"
run_python "$SCRIPT_DIR/check_visualization_rules.py" \
  --datasets "$DATASETS" --workers "$WORKERS" --fix

if dataset_enabled chartblocks; then
  run_python "$SCRIPT_DIR/repair_chartblocks_axes.py" --all --sync-svg \
    --report "$BEAGLE_DIR/chartblocks/repaired_axis_ids.txt"
  run_python "$SCRIPT_DIR/repair_cjk_fonts.py" chartblocks --all --sync-svg \
    --report "$BEAGLE_DIR/chartblocks/repaired_cjk_font_ids.txt"
  # Source image and generic geometry cleanup disagree for this chart; retain
  # the source SVG tree and only normalize its viewport.
  run_python "$SCRIPT_DIR/restore_raw_normalized_cleaned.py" chartblocks \
    --ids a097c24a-31a7-43bb-bdde-a80008d545fe --sync-svg
fi

if dataset_enabled fusion_clean; then
  run_python "$SCRIPT_DIR/repair_fusion_complex_charts.py"
fi

if dataset_enabled graphiq_clean; then
  run_python "$SCRIPT_DIR/repair_graphiq_line_fill.py" --all --sync-svg \
    --report "$BEAGLE_DIR/graphiq_clean/repaired_line_fill_ids.txt"
  run_python "$SCRIPT_DIR/repair_graphiq_tick_labels.py"
fi

if dataset_enabled plotly_export; then
  PLOTLY_RAW_FALLBACK_IDS="4312,4246,12123,4453,5825,8222,5821,5806,1836,4223,1678,11985,11988,5890,5889,3162,5888,3163,4525"
  run_python "$SCRIPT_DIR/restore_raw_normalized_cleaned.py" plotly_export \
    --ids "$PLOTLY_RAW_FALLBACK_IDS" --sync-svg
  run_python "$SCRIPT_DIR/repair_cjk_fonts.py" plotly_export --all --sync-svg \
    --report "$BEAGLE_DIR/plotly_export/repaired_cjk_font_ids.txt"
fi

echo "[3/6] Optimize SVGs with SVGO"
NODE_PATH="$NODE_PATH_VALUE" \
SVGO_DATASETS="$DATASETS" \
SVGO_MULTIPASS="$SVGO_MULTIPASS_VALUE" \
SVGO_MAX_BYTES="$SVGO_MAX_BYTES_VALUE" \
"$NODE_BIN" "$SCRIPT_DIR/svgo_optimizer.js"

# The ECharts SVG renderer sometimes serializes a full circle as one arc whose
# start and end points coincide. Cairo treats that standards-invalid arc as
# empty, so repair it after SVGO and before PNG rendering.
if dataset_enabled echarts; then
  run_python "$SCRIPT_DIR/repair_echarts_full_circle_paths.py" --sync-svg
  run_python "$SCRIPT_DIR/repair_echarts_animation_clips.py"
fi

echo "[4/6] Render PNGs with $RENDER_SHARDS shard(s)"
render_pids=()
render_logs=()
for ((shard_index=0; shard_index<RENDER_SHARDS; shard_index++)); do
  log_path="$BEAGLE_DIR/data_clean_svg2png_shard_${shard_index}.log"
  render_logs+=("$log_path")
  SVG2PNG_DATASETS="$DATASETS" \
  SVG2PNG_SHARD_COUNT="$RENDER_SHARDS" \
  SVG2PNG_SHARD_INDEX="$shard_index" \
  "$PYTHON_BIN" -u "$SCRIPT_DIR/svg2png.py" >"$log_path" 2>&1 &
  render_pids+=("$!")
done

render_status=0
for pid in "${render_pids[@]}"; do
  if ! wait "$pid"; then
    render_status=1
  fi
done
for log_path in "${render_logs[@]}"; do
  echo "--- $(basename "$log_path")"
  tail -n 3 "$log_path"
done
if [[ "$render_status" -ne 0 ]]; then
  echo "At least one PNG rendering shard failed." >&2
  exit "$render_status"
fi

echo "[5/6] Audit visualization rules"
run_python "$SCRIPT_DIR/check_visualization_rules.py" \
  --datasets "$DATASETS" --workers "$WORKERS"

echo "[6/6] Compare rendered PNGs with source images"
if [[ "$RUN_CONSISTENCY_CHECK" == "1" ]]; then
  IFS=',' read -r -a dataset_items <<< "$DATASETS"
  for dataset_name in "${dataset_items[@]}"; do
    run_python "$SCRIPT_DIR/check_beagle_png_consistency.py" \
      "$dataset_name" --workers "$WORKERS"
  done
else
  echo "Skipped (RUN_CONSISTENCY_CHECK=$RUN_CONSISTENCY_CHECK)"
fi

echo "Data cleaning completed successfully."
