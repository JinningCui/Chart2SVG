# Chart2SVG

Chart2SVG provides a reproducible pipeline for cleaning chart SVG datasets and
turning the cleaned charts into multimodal instruction-tuning data. The current
workflow supports five Beagle_Plus sources:

- ChartBlocks (`chartblocks`)
- FusionCharts (`fusion_clean`)
- Graphiq (`graphiq_clean`)
- Plotly (`plotly_export`)
- Apache ECharts (`echarts`)

All pipeline code is stored in [`data/`](data/). Dataset files are intentionally
excluded from Git and can be downloaded separately from
[Beagle_Plus on Hugging Face](https://huggingface.co/datasets/syslocker/Beagle_Plus).

## Workflow

```text
raw svg.txt + source PNG
          │
          ▼
  data/data_clean.sh
          │
          ├── cleaned_svg.txt
          ├── optimized svg.txt
          ├── rendered chart PNG
          └── rule and consistency reports
          │
          ▼
data/gen_semantic_svg.sh
          │
          ├── train_svg.txt
          └── train_json/<dataset>.json
          │
          ▼
 data/split_dataset.py
          │
          └── train_json/<dataset>/<chart-id>.json
```

## Repository layout

```text
Chart2SVG/
├── README.md
├── package.json
└── data/
    ├── data_clean.sh
    ├── gen_semantic_svg.sh
    ├── gen_svg_qwen.py
    ├── gen_std_svg_json.py
    ├── split_dataset.py
    ├── semantic_tokens.py
    ├── generate_syntactic_svg.py
    ├── svgo_optimizer.js
    ├── svg2png.py
    ├── check_visualization_rules.py
    ├── check_beagle_png_consistency.py
    ├── repair_*.py
    └── requirements.txt
```

## 1. Installation

Python 3.10+, Node.js, npm, Cairo, and its native libraries are required.

```bash
git clone https://github.com/JinningCui/Chart2SVG.git
cd Chart2SVG

python3 -m venv .venv
source .venv/bin/activate
pip install -r data/requirements.txt
npm install
```

On macOS, Cairo can be installed with Homebrew if it is not already available:

```bash
brew install cairo pango libffi
```

If Cairo is provided by Conda, pass its library directory through
`CAIRO_LIBRARY_DIR` when running the scripts.

## 2. Prepare the datasets

Extract the datasets into the directory expected by the cleaning pipeline:

```text
data/data/Beagle/
├── chartblocks/
│   ├── charts/<chart-id>/svg.txt
│   └── images/<chart-id>.png
├── fusion_clean/
├── graphiq_clean/
├── plotly_export/
└── echarts/
```

Legacy fallback repairs also use the original archives when available:

```text
data/data/Beagle/
├── chartblocks.zip
├── fusioncharts.zip
├── graphiq.zip
└── plotly.zip
```

## 3. Clean the SVG datasets

Run all five datasets:

```bash
PYTHON_BIN="$PWD/.venv/bin/python" ./data/data_clean.sh
```

Run selected datasets and control parallelism:

```bash
BEAGLE_DATASETS=echarts,chartblocks \
BEAGLE_WORKERS=10 \
RENDER_SHARDS=2 \
PYTHON_BIN="$PWD/.venv/bin/python" \
./data/data_clean.sh
```

`data_clean.sh` executes these stages in order:

1. `generate_syntactic_svg.py` creates normalized `cleaned_svg.txt` files.
2. Visualization-rule and dataset-specific repair scripts fix known structural
   and renderer-compatibility problems.
3. `svgo_optimizer.js` optimizes SVG while preserving rendering-critical files.
4. `svg2png.py` renders chart PNG previews with configurable parallel shards.
5. `check_visualization_rules.py` audits canvas size, invalid attributes,
   overflow, clipping, and Cartesian axes.
6. `check_beagle_png_consistency.py` compares rendered PNG files with source
   images using SSIM, MAE, and bounding-box IoU.

Important environment variables:

| Variable | Default | Description |
|---|---:|---|
| `BEAGLE_DATASETS` | all five datasets | Comma-separated dataset names |
| `BEAGLE_WORKERS` | `10` | Python worker count |
| `RENDER_SHARDS` | `2` | Parallel PNG renderer processes |
| `BEAGLE_FORCE` | `1` | Regenerate existing cleaned SVG |
| `RUN_CONSISTENCY_CHECK` | `1` | Set to `0` to skip PNG comparison |
| `PYTHON_BIN` | `python3` | Python executable |
| `NODE_PATH` | auto-detected | `node_modules` containing SVGO |
| `CAIRO_LIBRARY_DIR` | Conda Sketchpad path | Optional Cairo library directory |

Cleaning outputs are written inside each chart directory:

```text
charts/<chart-id>/
├── svg.txt
├── cleaned_svg.txt
├── <chart-id>.png
└── ...
```

Audit reports are written at the dataset root as CSV and JSON files.

## 4. Generate semantic SVG training data

`gen_semantic_svg.sh` runs these two programs in sequence:

1. `gen_svg_qwen.py`: converts each eligible SVG into the specialized semantic
   SVG token stream, saves `train_svg.txt`, and reconstructs a 512×512 PNG.
2. `gen_std_svg_json.py`: builds one standard Qwen conversation JSON per
   dataset from `train_svg.txt` and its chart PNG.

Point it at the cleaned dataset root from the previous step:

```bash
./data/gen_semantic_svg.sh \
  --base-dir "$PWD/data/data/Beagle" \
  --python "$PWD/.venv/bin/python"
```

Use another dataset location, selected datasets, or a different worker count:

```bash
./data/gen_semantic_svg.sh \
  --base-dir /path/to/Beagle_Plus \
  --datasets echarts,chartblocks \
  --workers 8 \
  --python "$PWD/.venv/bin/python"
```

The same options can be supplied through environment variables:

| CLI option | Environment variable |
|---|---|
| `--base-dir` | `BEAGLE_PLUS_DIR` |
| `--datasets` | `BEAGLE_DATASETS` |
| `--workers` | `GEN_SVG_WORKERS` |
| `--python` | `PYTHON_BIN` |

SVG content longer than 80,000 characters after embedded bitmap removal is
skipped to keep training examples within practical model token limits.

The generated files are:

```text
<dataset-root>/
├── <dataset>/charts/<chart-id>/train_svg.txt
└── train_json/
    ├── chartblocks.json
    ├── fusion_clean.json
    ├── graphiq.json
    ├── plotly_export.json
    └── echarts.json
```

## 5. Split merged JSON into per-chart files

To create one JSON file per training sample:

```bash
"$PWD/.venv/bin/python" data/split_dataset.py \
  --source-dir "$PWD/data/data/Beagle/train_json"
```

Select only part of the generated data:

```bash
"$PWD/.venv/bin/python" data/split_dataset.py \
  --source-dir /path/to/Beagle_Plus/train_json \
  --datasets echarts,chartblocks
```

The resulting layout is:

```text
train_json/
├── echarts.json
└── echarts/
    ├── <chart-id-1>.json
    ├── <chart-id-2>.json
    └── ...
```

Each record follows the multimodal conversation format:

```json
{
  "messages": [
    {"role": "system", "content": "You are a world-class SVG Expert..."},
    {"role": "user", "content": "<image>Convert this image to SVG code."},
    {"role": "assistant", "content": "[<|START_OF_SVG|>]..."}
  ],
  "images": ["/absolute/path/to/chart.png"]
}
```

## License

The pipeline code is released under the [MIT License](LICENSE). The source
datasets retain their respective licenses and terms of use.
