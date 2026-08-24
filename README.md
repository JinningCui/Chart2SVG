# Chart2SVG

Chart2SVG contains the reproducible SVG cleaning pipeline used for the
Beagle_Plus chart datasets.

The implementation is under [`data/`](data/). It generates syntactically
normalized SVG, applies deterministic dataset-specific repairs, optimizes SVG
with SVGO, renders PNG previews, audits visualization rules, and compares the
renders with source images.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r data/requirements.txt
npm install

mkdir -p data/data/Beagle
# Put dataset folders or source archives under data/data/Beagle.

PYTHON_BIN="$PWD/.venv/bin/python" ./data/data_clean.sh
```

Select datasets and parallelism with environment variables:

```bash
BEAGLE_DATASETS=echarts,chartblocks \
BEAGLE_WORKERS=10 \
RENDER_SHARDS=2 \
PYTHON_BIN="$PWD/.venv/bin/python" \
./data/data_clean.sh
```

See [`data/README.md`](data/README.md) for the expected input layout and
pipeline stages.
