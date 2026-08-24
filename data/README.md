# Data cleaning pipeline

`data_clean.sh` runs the following stages in order:

1. Generate normalized `cleaned_svg.txt` files.
2. Apply visualization-rule and dataset-specific repairs.
3. Optimize SVG with SVGO.
4. Render chart PNG files, optionally with multiple shards.
5. Audit canvas, overflow, attribute, and axis rules.
6. Compare rendered PNG files with the source images.

## Expected input layout

Place datasets under `data/data/Beagle` relative to the repository root:

```text
data/data/Beagle/
├── chartblocks/
├── fusion_clean/
├── graphiq_clean/
├── plotly_export/
└── echarts/
```

Each dataset normally contains:

```text
<dataset>/
├── images/<chart-id>.png
└── charts/<chart-id>/svg.txt
```

The legacy datasets may also contain `urls.txt` and their source ZIP archives.
Those archives are used by conservative fallback repairs.

## Configuration

- `BEAGLE_DATASETS`: comma-separated dataset names.
- `BEAGLE_WORKERS`: Python worker count; default `10`.
- `RENDER_SHARDS`: parallel PNG rendering processes; default `2`.
- `BEAGLE_FORCE`: regenerate existing cleaned SVG; default `1`.
- `RUN_CONSISTENCY_CHECK`: use `0` to skip PNG comparison.
- `PYTHON_BIN`: Python executable; default `python3`.
- `NODE_PATH`: `node_modules` directory containing `svgo` when it cannot be
  discovered automatically.
- `CAIRO_LIBRARY_DIR`: optional Cairo library directory on macOS.

Run the full pipeline from the repository root:

```bash
PYTHON_BIN="$PWD/.venv/bin/python" ./data/data_clean.sh
```

After cleaning, generate semantic SVG training records with:

```bash
./data/gen_semantic_svg.sh \
  --base-dir "$PWD/data/data/Beagle" \
  --python "$PWD/.venv/bin/python"
```

Split the merged dataset JSON files into per-chart files with:

```bash
"$PWD/.venv/bin/python" data/split_dataset.py \
  --source-dir "$PWD/data/data/Beagle/train_json"
```
