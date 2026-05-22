# Optical Flow Benchmark

An efficiency benchmark harness for optical flow models on arbitrary video datasets. It measures **throughput** (GPU-hours per input-hour), **GPU utilization**, and **storage overhead** — not accuracy — across multiple models using a unified preprocessing step.

## Repository structure

```
benchmark_optical_flow/
  run_benchmark_optical_flow.sh          # main entry point (also a SLURM job script)
  models.tsv                             # model registry
  scripts/
    preprocess_video.py                  # FPS probe + frame resampling/striding
    infer_ptlflow_video.py               # PTLFlow inference on a single video
    infer_megaflow_video.py              # MegaFlow inference on a single video
    collect_benchmark_metrics.py         # GPU sampling, per-run JSON, aggregate CSV
    generate_viewer.py                   # (standalone) HTML result viewer
    compare_compressed_flow_video.py     # (standalone) compression comparison tool
  ptlflow/                               # submodule — hmorimitsu/ptlflow
  other_models/
    megaflow/                            # submodule — cvg/megaflow
```

## Setup

### Clone with submodules

```bash
git clone --recurse-submodules <repo-url>
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

### Python environments

The benchmark requires two separate Conda environments — one per backend. Install each inside its submodule directory following that project's own instructions.

| Environment | Variable | Used for |
|---|---|---|
| `ptflow` | `PTLFLOW_PYTHON` | PTLFlow models + all preprocessing |
| `megaflow` | `MEGAFLOW_PYTHON` | MegaFlow model |

Default Python paths assume Miniforge under `$SCRATCH`:
- PTLFlow: `$SCRATCH/miniforge3/envs/ptflow/bin/python`
- MegaFlow: `$SCRATCH/conda_envs/megaflow/bin/python`

Override with environment variables (see [Configuration](#configuration)).

## Models

Models are listed in `models.tsv`. Each enabled row is run against every input video.

| Column | Description |
|---|---|
| `enabled` | `1` to run, `0` or lines starting with `#` to skip |
| `name` | Display name used in output paths and the results CSV |
| `backend` | `ptlflow` or `megaflow` |
| `repo_model` | Model identifier passed to the backend |
| `ckpt` | Checkpoint tag (PTLFlow only; leave blank for MegaFlow) |

Default models: **FlowSeek**, **MEMFOF**, **ReCoVEr**, **WAFT**, **DPFlow** (PTLFlow backend), **MegaFlow** (MegaFlow backend).

## Running

### On a SLURM cluster

```bash
sbatch run_benchmark_optical_flow.sh
```

### Interactively on a GPU node

```bash
bash run_benchmark_optical_flow.sh
```

### Custom dataset or output path

```bash
DATASET_DIR=/path/to/videos RESULTS_DIR=/path/to/results bash run_benchmark_optical_flow.sh
```

## Configuration

All parameters are environment variables with defaults defined at the top of the script.

| Variable | Default | Description |
|---|---|---|
| `DATASET_DIR` | *(scratch path)* | Root directory of input videos |
| `RESULTS_DIR` | `$DATASET_DIR/results` | Output root |
| `MODELS_TSV` | `models.tsv` in this repo | Model registry file |
| `PTLFLOW_PYTHON` | Miniforge `ptflow` env | Python binary for PTLFlow models |
| `MEGAFLOW_PYTHON` | conda `megaflow` env | Python binary for MegaFlow |
| `COMMON_FPS` | `0` | Resample all videos to this FPS before inference (`0` = preserve source) |
| `FRAME_STRIDE` | `1` | Keep every Nth frame (`1` = all frames; takes precedence over `COMMON_FPS`) |
| `SAVE_FRAMES` | `0` | Also save per-frame PNG visualizations and `.flo` files |
| `SAVE_RAW_NPZ` | `1` | Save raw flow vectors as `.npz` |
| `OVERWRITE` | `0` | Re-run a model even if `flow_viz.mp4` already exists |
| `CONTINUE_ON_ERROR` | `1` | Keep going after a model failure instead of aborting |
| `MEGAFLOW_FIX_WIDTH` | `952` | MegaFlow input width in pixels (height scaled proportionally) |
| `MEGAFLOW_WINDOW_SIZE` | `4` | MegaFlow temporal window size |
| `MEGAFLOW_ITERS` | `8` | MegaFlow refinement iterations |

## Dataset layout

Videos are discovered recursively under `DATASET_DIR`. The immediate parent directory of each video file becomes its **category** in the results tree.

```
DATASET_DIR/
  <category>/
    <video_stem>.<ext>
```

Supported extensions: `mp4`, `mov`, `mkv`, `avi`, `webm` (override with `VIDEO_EXTENSIONS`).

## Outputs

### Per model × video

```
results/<Model>/<Category>/<VideoStem>/
  flow_viz.mp4      # color-coded optical flow visualization
  flow_raw.npz      # raw flow vectors, float16, shape [T, H, W, 2]
  metadata.txt      # run parameters and frame counts
```

With `SAVE_FRAMES=1`, per-frame files are written under `frames/flow_viz/` and `frames/flow/`.

### Aggregate CSV

`results/benchmark_metrics.csv` — one row per model×video (`row_type=per_video`) plus one aggregate row per model (`row_type=aggregate`):

| Column | Description |
|---|---|
| `runtime_seconds` | Wall-clock inference time |
| `input_duration_seconds` | Duration of the processed video |
| `gpu_hours_per_input_hour` | GPU time per hour of input (lower = faster) |
| `average_gpu_utilization_percent` | Mean GPU % sampled during inference |
| `input_size_bytes` / `output_size_bytes` | File sizes |
| `relative_storage` | Output-to-input size ratio |
| `status` | `ok` or `failed` |

### Preprocessing cache

Every model receives the same preprocessed MP4. Preprocessed files are cached under `results/_preprocessed/` so re-running with different models does not re-encode. By default, source FPS and frame sequence are preserved.

### Failure log

Any failed model runs are appended to `results/_logs/failures.txt` with the path to the corresponding `.log` file.
