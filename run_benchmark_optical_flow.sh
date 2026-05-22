#!/bin/bash
#SBATCH --job-name=oflow_qual_bench
#SBATCH --account=a143
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

BENCH_DIR="/users/mlopezescoriza/projects/pseudolabelers/benchmark_optical_flow"
DATASET_DIR="${DATASET_DIR:-/capstor/scratch/cscs/mlopezescoriza/dataset_benchmark_optical_flow}"
RESULTS_DIR="${RESULTS_DIR:-/capstor/scratch/cscs/mlopezescoriza/dataset_benchmark_optical_flow/results}"
PTLFLOW_REPO="${PTLFLOW_REPO:-${BENCH_DIR}/ptlflow}"
MEGAFLOW_REPO="${MEGAFLOW_REPO:-${BENCH_DIR}/other_models/megaflow}"
MODELS_TSV="${MODELS_TSV:-${BENCH_DIR}/models.tsv}"
METRICS_SCRIPT="${METRICS_SCRIPT:-${BENCH_DIR}/scripts/collect_benchmark_metrics.py}"

PTLFLOW_PYTHON="${PTLFLOW_PYTHON:-/capstor/scratch/cscs/${USER}/miniforge3/envs/ptflow/bin/python}"
MEGAFLOW_PYTHON="${MEGAFLOW_PYTHON:-/capstor/scratch/cscs/${USER}/conda_envs/megaflow/bin/python}"

# 0 preserves the source video FPS/frame sequence. Set COMMON_FPS=12, etc. to
# force every model to receive the same sampled FPS for every input video.
COMMON_FPS="${COMMON_FPS:-0}"
# 1 keeps every frame. Set FRAME_STRIDE=4 to use frames 0,4,8,12,...
# FRAME_STRIDE takes precedence over COMMON_FPS when greater than 1.
FRAME_STRIDE="${FRAME_STRIDE:-1}"
VIDEO_EXTENSIONS="${VIDEO_EXTENSIONS:-mp4,mov,mkv,avi,webm}"
SAVE_FRAMES="${SAVE_FRAMES:-0}"
SAVE_RAW_NPZ="${SAVE_RAW_NPZ:-1}"
OVERWRITE="${OVERWRITE:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
MEGAFLOW_FIX_WIDTH="${MEGAFLOW_FIX_WIDTH:-952}"
MEGAFLOW_WINDOW_SIZE="${MEGAFLOW_WINDOW_SIZE:-4}"
MEGAFLOW_ITERS="${MEGAFLOW_ITERS:-8}"

PREPROCESSED_DIR="${RESULTS_DIR}/_preprocessed"
LOG_DIR="${RESULTS_DIR}/_logs"
MANIFEST="${RESULTS_DIR}/manifest.tsv"
BENCHMARK_CSV="${RESULTS_DIR}/benchmark_metrics.csv"

mkdir -p "${RESULTS_DIR}" "${PREPROCESSED_DIR}" "${LOG_DIR}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/${USER}_matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/${USER}_cache}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

if [[ -f "/capstor/scratch/cscs/${USER}/miniforge3/etc/profile.d/conda.sh" ]]; then
  source "/capstor/scratch/cscs/${USER}/miniforge3/etc/profile.d/conda.sh"
fi

echo "Running on host: $(hostname)"
echo "Dataset: ${DATASET_DIR}"
echo "Results: ${RESULTS_DIR}"
echo "Common FPS: ${COMMON_FPS}"
echo "Frame stride: ${FRAME_STRIDE}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

video_find_expr=()
IFS=',' read -ra extensions <<< "${VIDEO_EXTENSIONS}"
for ext in "${extensions[@]}"; do
  video_find_expr+=(-iname "*.${ext}" -o)
done
video_find_expr=("${video_find_expr[@]:0:${#video_find_expr[@]}-1}")

find "${DATASET_DIR}" \
  -path "${RESULTS_DIR}" -prune -o \
  -type f \( "${video_find_expr[@]}" \) -print | sort > "${RESULTS_DIR}/videos.txt"

printf "category\tvideo_stem\tinput_video\tpreprocessed_video\tfps\n" > "${MANIFEST}"

preprocess_video() {
  local input_video="$1"
  local category="$2"
  local stem="$3"
  local output_video="${PREPROCESSED_DIR}/${category}/${stem}.mp4"

  local source_fps
  source_fps="$("${PTLFLOW_PYTHON}" "${BENCH_DIR}/scripts/preprocess_video.py" --input "${input_video}" --print_fps_only)"

  if [[ "${FRAME_STRIDE}" == "1" && "${COMMON_FPS}" == "0" ]]; then
    printf "%s\t%s" "${input_video}" "${source_fps}"
  else
    mkdir -p "$(dirname "${output_video}")"
    if [[ "${OVERWRITE}" == "1" || ! -s "${output_video}" ]]; then
      "${PTLFLOW_PYTHON}" "${BENCH_DIR}/scripts/preprocess_video.py" \
        --input "${input_video}" \
        --output "${output_video}" \
        --target_fps "${COMMON_FPS}" \
        --frame_stride "${FRAME_STRIDE}" >/dev/null
    fi
    if [[ ! -s "${output_video}" ]]; then
      echo "Failed to create preprocessed video: ${output_video}" >&2
      return 1
    fi
    if [[ "${FRAME_STRIDE}" != "1" ]]; then
      output_fps="$("${PTLFLOW_PYTHON}" "${BENCH_DIR}/scripts/preprocess_video.py" --input "${output_video}" --print_fps_only)"
    else
      output_fps="${COMMON_FPS}"
    fi
    printf "%s\t%s" "${output_video}" "${output_fps}"
  fi
}

run_ptlflow() {
  local display_name="$1"
  local repo_model="$2"
  local ckpt="$3"
  local input_video="$4"
  local output_dir="$5"
  local fps="$6"
  local category="$7"
  local stem="$8"

  local args=(
    "${BENCH_DIR}/scripts/infer_ptlflow_video.py"
    --model "${repo_model}"
    --input_path "${input_video}"
    --output_dir "${output_dir}"
    --extract_fps 0
    --output_fps "${fps}"
  )
  if [[ -n "${ckpt}" ]]; then
    args+=(--ckpt_path "${ckpt}")
  fi
  if [[ "${SAVE_FRAMES}" == "1" ]]; then
    args+=(--save_frames)
  fi
  if [[ "${SAVE_RAW_NPZ}" != "1" ]]; then
    args+=(--no_save_raw_npz)
  fi

  (
    cd "${PTLFLOW_REPO}"
    PYTHONPATH="${PTLFLOW_REPO}:${PYTHONPATH:-}" "${PTLFLOW_PYTHON}" "${METRICS_SCRIPT}" run \
      --log-path "${output_dir}.log" \
      --metrics-path "${output_dir}/benchmark_metrics.json" \
      --input-video "${input_video}" \
      --output-dir "${output_dir}" \
      --model "${display_name}" \
      --category "${category}" \
      --video-stem "${stem}" \
      -- "${PTLFLOW_PYTHON}" "${args[@]}"
  )
}

run_megaflow() {
  local display_name="$1"
  local repo_model="$2"
  local input_video="$3"
  local output_dir="$4"
  local fps="$5"
  local category="$6"
  local stem="$7"

  local args=(
    "${BENCH_DIR}/scripts/infer_megaflow_video.py"
    --model_name "${repo_model}"
    --input_path "${input_video}"
    --output_dir "${output_dir}"
    --output_fps "${fps}"
    --fix_width "${MEGAFLOW_FIX_WIDTH}"
    --window_size "${MEGAFLOW_WINDOW_SIZE}"
    --iters "${MEGAFLOW_ITERS}"
  )
  if [[ "${SAVE_FRAMES}" == "1" ]]; then
    args+=(--save_frames)
  fi
  if [[ "${SAVE_RAW_NPZ}" != "1" ]]; then
    args+=(--no_save_raw_npz)
  fi

  (
    cd "${MEGAFLOW_REPO}"
    PYTHONPATH="${MEGAFLOW_REPO}:${PYTHONPATH:-}" "${MEGAFLOW_PYTHON}" "${METRICS_SCRIPT}" run \
      --log-path "${output_dir}.log" \
      --metrics-path "${output_dir}/benchmark_metrics.json" \
      --input-video "${input_video}" \
      --output-dir "${output_dir}" \
      --model "${display_name}" \
      --category "${category}" \
      --video-stem "${stem}" \
      -- "${MEGAFLOW_PYTHON}" "${args[@]}"
  )
}

while IFS= read -r input_video; do
  category="$(basename "$(dirname "${input_video}")")"
  stem="$(basename "${input_video}")"
  stem="${stem%.*}"
  preprocess_result="$(preprocess_video "${input_video}" "${category}" "${stem}")"
  preprocessed_video="$(printf "%s" "${preprocess_result}" | cut -f1)"
  fps="$(printf "%s" "${preprocess_result}" | cut -f2)"

  printf "%s\t%s\t%s\t%s\t%s\n" "${category}" "${stem}" "${input_video}" "${preprocessed_video}" "${fps}" >> "${MANIFEST}"

  while IFS=$'\t' read -r enabled display_name backend repo_model ckpt; do
    [[ -z "${enabled}" || "${enabled}" =~ ^# ]] && continue
    [[ "${enabled}" == "1" ]] || continue

    output_dir="${RESULTS_DIR}/${display_name}/${category}/${stem}"
    if [[ "${OVERWRITE}" != "1" && -s "${output_dir}/flow_viz.mp4" ]]; then
      echo "Skipping existing ${display_name}/${category}/${stem}"
      continue
    fi
    mkdir -p "${output_dir}"

    echo "Running ${display_name} on ${category}/${stem}"
    set +e
    if [[ "${backend}" == "ptlflow" ]]; then
      run_ptlflow "${display_name}" "${repo_model}" "${ckpt:-}" "${preprocessed_video}" "${output_dir}" "${fps}" "${category}" "${stem}"
      status=$?
    elif [[ "${backend}" == "megaflow" ]]; then
      run_megaflow "${display_name}" "${repo_model}" "${preprocessed_video}" "${output_dir}" "${fps}" "${category}" "${stem}"
      status=$?
    else
      echo "Unknown backend '${backend}' for ${display_name}" >&2
      exit 2
    fi
    set -e

    if [[ "${status}" != "0" ]]; then
      echo "FAILED ${display_name} on ${category}/${stem}; see ${output_dir}.log" | tee -a "${LOG_DIR}/failures.txt"
      if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
        exit "${status}"
      fi
    fi
  done < "${MODELS_TSV}"
done < "${RESULTS_DIR}/videos.txt"

echo "Writing benchmark CSV to ${BENCHMARK_CSV}"
"${PTLFLOW_PYTHON}" "${METRICS_SCRIPT}" aggregate \
  --results-dir "${RESULTS_DIR}" \
  --manifest "${MANIFEST}" \
  --output-csv "${BENCHMARK_CSV}"

echo "Done. Outputs are under ${RESULTS_DIR}"
