<h1 align="center">Chart2SVG: Editable SVG Generation from Raster Chart Images</h1>

<p align="center">
  <strong>Accepted to IEEE VIS 2026</strong>
</p>

<p align="center">
  <img alt="IEEE VIS 2026" src="https://img.shields.io/badge/IEEE%20VIS%202026-Accepted-success?style=flat-square">
  <a href="https://huggingface.co/datasets/syslocker/Beagle_Plus"><img alt="Hugging Face Dataset" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Beagle__Plus-FFD21E?style=flat-square"></a>
  <a href="https://github.com/JinningCui/Chart2SVG/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-blue?style=flat-square"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/Python-%E2%89%A53.10-3776AB?style=flat-square&amp;logo=python&amp;logoColor=white">
  <a href="https://swift.readthedocs.io/zh-cn/latest/GetStarted/SWIFT-installation.html"><img alt="Framework: ms-swift" src="https://img.shields.io/badge/Framework-ms--swift-6C63FF?style=flat-square"></a>
</p>

<p align="center">
  <img src="assets/figures/fig1.png" alt="Chart2SVG converts raster chart images into reconstructed SVG and supports chart editing" width="100%">
</p>

Chart2SVG is an end-to-end toolkit for converting chart images into structured
SVG. It covers dataset cleaning, semantic SVG tokenization, Qwen3-VL model
initialization, supervised fine-tuning (SFT), GRPO reinforcement learning, LoRA
merging, inference, and SVG rendering. The data pipeline currently supports five
Beagle_Plus sources:

- ChartBlocks (`chartblocks`)
- FusionCharts (`fusion_clean`)
- Graphiq (`graphiq_clean`)
- Plotly (`plotly_export`)
- Apache ECharts (`echarts`)

Dataset files are intentionally excluded from Git and can be downloaded from
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
          │
          ▼
 preare_svg_qwen.py
          │
          ▼
scripts/sft/lora_sft.sh
          │
          ├── merge_lora.sh (optional)
          ▼
 scripts/grpo/grpo.sh
          │
          ▼
 inference + SVG rendering
```

## Repository layout

```text
Chart2SVG/
├── README.md
├── package.json
├── assets/figures/fig1.png         # Project overview figure
├── preare_svg_qwen.py              # Initialize semantic SVG tokens
├── merge_lora.sh                   # Merge LoRA into the base model
├── data/
│   ├── data_clean.sh
│   ├── gen_semantic_svg.sh
│   ├── gen_svg_qwen.py
│   ├── gen_std_svg_json.py
│   ├── split_dataset.py
│   ├── semantic_tokens.py
│   ├── token_config.py
│   ├── tokenizer.py
│   ├── svg_dataset.py
│   ├── generate_syntactic_svg.py
│   ├── svgo_optimizer.js
│   ├── svg2png.py
│   ├── check_visualization_rules.py
│   ├── check_beagle_png_consistency.py
│   ├── repair_*.py
│   ├── requirements.txt            # Lightweight data-cleaning dependencies
│   └── requirement.txt             # Pinned training/GRPO environment
├── svglib/
│   ├── io.py
│   └── preprocess.py
└── scripts/
    ├── sft/lora_sft.sh
    ├── grpo/
    │   ├── grpo.sh
    │   ├── roll_out.sh
    │   ├── plugin.py
    │   └── prompt.txt
    └── inference/
        ├── run_inference_and_render.py
        └── batch_inference.py
```

## 1. Installation

### 1.1 Runtime requirements

Python is required for every stage. The
[official ms-swift installation guide](https://swift.readthedocs.io/zh-cn/latest/GetStarted/SWIFT-installation.html)
requires Python 3.10 or newer and recommends Python 3.12. The exported training
environment in [`data/requirement.txt`](data/requirement.txt) was created with
Python 3.12.12.

The data-cleaning pipeline additionally requires Node.js, npm, Cairo, and its
native libraries. Training and inference require a CUDA-enabled PyTorch
environment; GRPO rollout also requires a vLLM build compatible with the
installed CUDA and PyTorch versions.

### 1.2 Clone the repository

```bash
git clone https://github.com/JinningCui/Chart2SVG.git
cd Chart2SVG
```

### 1.3 Data-cleaning environment

Use the lightweight dependency file when only cleaning data and generating
semantic SVG:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r data/requirements.txt
npm install
```

On macOS, Cairo can be installed with Homebrew if it is not already available:

```bash
brew install cairo pango libffi
```

If Cairo is provided by Conda, pass its library directory through
`CAIRO_LIBRARY_DIR` when running the scripts.

### 1.4 Training, GRPO, and inference environment

The full lock file includes PyTorch 2.8.0, Transformers 4.57.6, vLLM 0.11.0,
ms-swift's training stack, and the SVG reward dependencies. It also contains
`-e ./ms-swift`, so clone ms-swift into the Chart2SVG repository root before
installing it:

```bash
python3.12 -m venv .venv-training
source .venv-training/bin/activate
python -m pip install -U pip

git clone https://github.com/modelscope/ms-swift.git ms-swift
# Checkout the ms-swift revision required by your experiment when applicable.

python -m pip install -r data/requirement.txt
```

After ms-swift is configured, replace the corresponding scripts under
`ms-swift/examples/train/` with this repository's customized training scripts,
including `scripts/sft/`, `scripts/grpo/`, and the other matching entries,
before starting training.

This lock file targets a Linux CUDA 12 environment and includes CUDA-specific
packages such as FlashAttention and vLLM. Do not use it unchanged for macOS,
CPU-only, NPU, or a different CUDA/PyTorch combination.

For a fresh, unpinned ms-swift installation, the official wheel command is:

```bash
python -m pip install 'ms-swift' -U
```

Follow the official guide when choosing a source revision, Docker image, or
hardware-specific environment. In particular, install vLLM as a version
compatible with the selected PyTorch and CUDA stack rather than mixing versions
from unrelated environments.

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
  --base-dir "Path to your Beagle_Plus dataset" \
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
  --source-dir "Path to your train_json directory" \
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
  "images": ["Path to your chart image"]
}
```

## 6. Initialize Qwen3-VL semantic SVG tokens

[`preare_svg_qwen.py`](preare_svg_qwen.py) collects the semantic tokens defined
by the data package, adds them to the tokenizer, resizes the model embeddings,
and initializes new embeddings from their English semantic descriptions.

Its default relative paths are:

| Purpose | Path |
|---|---|
| Base model | `models/Qwen3-VL-4B-Instruct` |
| Initialized output | `models/Qwen3_Chart_4B_Initialized` |

Place the base model at that location or edit `MODEL_PATH` and `SAVE_PATH`, then
run:

```bash
python preare_svg_qwen.py
```

The script currently loads the model on `cuda:1`; adjust `device_map` for your
GPU layout.

## 7. Supervised fine-tuning

Before running SFT, replace every descriptive placeholder in
[`scripts/sft/lora_sft.sh`](scripts/sft/lora_sft.sh):

| Placeholder | Replace with |
|---|---|
| `Path to your model` | Initialized model directory |
| `Path to your dataset` | One split training-data directory |

Run LoRA SFT with ms-swift:

```bash
bash scripts/sft/lora_sft.sh
```

The provided configuration uses Qwen3-VL, LoRA on all linear layers, preserves
`embed_tokens` and `lm_head`, and uses an 8,192-token maximum sequence length.
It sets LoRA rank 8, LoRA alpha 32, two epochs, and gradient accumulation of two
steps. Adjust GPU IDs, `CUDA_VISIBLE_DEVICES`, `NPROC_PER_NODE`, batch sizes,
and `--output_dir` for your environment.

## 8. Merge a LoRA checkpoint

[`merge_lora.sh`](merge_lora.sh) resolves paths relative to its own directory:

| Variable | Default relative path |
|---|---|
| `CKPT_DIR` | `../svg_output/v12-20260102-163107/checkpoint-3189` |
| `OUTPUT_DIR` | `models/Qwen3_Chart_Stage1` |

Edit those variables when your checkpoint layout differs, then run:

```bash
bash merge_lora.sh
```

The script calls `swift export --merge_lora true` and writes the merged model to
`OUTPUT_DIR`.

## 9. GRPO reinforcement learning

GRPO uses ms-swift, a vLLM rollout server, and the custom `svg_pipeline` reward.
Before running it, replace:

- `Path to your model` in `scripts/grpo/roll_out.sh` and
  `scripts/grpo/grpo.sh`;
- `Path to your plugin` in `scripts/grpo/grpo.sh`, normally with
  `scripts/grpo/plugin.py`; and
- the relative `grpo_train_json/...` entries with your GRPO datasets.

Start the rollout server:

```bash
bash scripts/grpo/roll_out.sh
```

In another terminal, start GRPO training:

```bash
bash scripts/grpo/grpo.sh
```

The configuration uses four GPUs, vLLM server mode on port 8000, LoRA, four
generations per prompt, and an 8,192-token completion limit. Keep
`CUDA_VISIBLE_DEVICES`, `NPROC_PER_NODE`, and vLLM tensor parallelism consistent
with your hardware.

### SVG reward

[`scripts/grpo/plugin.py`](scripts/grpo/plugin.py) registers
`SVGDependentReward` as `svg_pipeline`. For each completion it:

1. extracts standard or semantic SVG output;
2. rejects missing, invalid, oversized, timed-out, or non-renderable SVG;
3. renders valid SVG in a spawned sandbox process;
4. compares it with the reference image using relaxed IoU, SSIM, and PSNR; and
5. returns the mean visual score for valid SVG, while invalid SVG receives
   `-1.0`.

The implementation computes a repetition statistic but does not currently
apply it to the final reward value.

## 10. Inference and rendering

Run inference for one JSON or JSONL dataset:

```bash
python scripts/inference/run_inference_and_render.py \
  --checkpoint_path "Path to your checkpoint" \
  --dataset_path "Path to your dataset JSON" \
  --output_dir "Path to your inference output" \
  --resized_dir "Path to your normalized images"
```

Useful optional arguments include:

- `--limit N`: process only the first `N` pending samples;
- `--model_type qwen3_vl`: select the model type;
- `--infer_backend pt|vllm|lmdeploy`: select the inference backend;
- `--max_batch_size N`: control inference batching;
- `--render_workers N`: control parallel SVG rendering; and
- `--stream` or `--no_stream`: control streaming behavior.

The script normalizes input images to 512×512, generates four candidates per
sample, converts semantic SVG to standard SVG, restores original dimensions,
and saves SVG, PNG, and JSON artifacts.

For multiple datasets, edit `datasets` and `script_path` in
[`scripts/inference/batch_inference.py`](scripts/inference/batch_inference.py),
ensure the checkpoint default in `run_inference_and_render.py` is configured,
and run:

```bash
python scripts/inference/batch_inference.py
```

## Path placeholders

The public scripts intentionally use descriptive placeholders instead of
private server paths:

| Placeholder | Meaning |
|---|---|
| `Path to your model` | Base or trained model directory |
| `Path to your dataset` | Training dataset directory |
| `Path to your checkpoint` | SFT/GRPO checkpoint directory |
| `Path to your plugin` | Absolute or relative path to `plugin.py` |
| `Path to your inference script` | Path to the inference entry point |

Replace them before running the corresponding command. Do not commit access
tokens, private filesystem paths, or model credentials.

## License

The pipeline code is released under the [MIT License](LICENSE). The source
datasets retain their respective licenses and terms of use. The training
scripts include adaptations of ms-swift examples; follow the applicable
upstream licenses when redistributing those portions.
