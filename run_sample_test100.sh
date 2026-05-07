#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CKPT="./logs_diffusion_full/targetdiff_cjkim_full_gpu/checkpoints/62000.pt"
RUNTIME_CFG="./sampling.yml"
OUT_ROOT="./sampling_results_full_test100"

NUM_SAMPLES="${NUM_SAMPLES:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_STEPS="${NUM_STEPS:-1000}"
SAMPLE_NUM_ATOMS="${SAMPLE_NUM_ATOMS:-prior}"

if [[ ! -f "${CKPT}" ]]; then
  echo "[Error] checkpoint not found: ${CKPT}" >&2
  exit 1
fi

if [[ ! -f "./data/crossdocked_pocket10_pose_split.pt" ]]; then
  echo "[Error] split file not found" >&2
  exit 1
fi

if [[ ! -f "./data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb" ]]; then
  echo "[Error] lmdb file not found" >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}"

echo "[Sampling] checkpoint=${CKPT}"
echo "[Sampling] output=${OUT_ROOT}"
echo "[Sampling] num_samples=${NUM_SAMPLES}, batch_size=${BATCH_SIZE}, num_steps=${NUM_STEPS}, sample_num_atoms=${SAMPLE_NUM_ATOMS}"

for i in $(seq 0 99); do
  out_dir="${OUT_ROOT}/data_$(printf '%03d' "${i}")"

  if [[ -f "${out_dir}/result.pt" ]]; then
    echo "[Skip] data_id=${i}: ${out_dir}/result.pt exists"
    continue
  fi

  echo "[Sampling] data_id=${i} -> ${out_dir}"
  "${PYTHON_BIN}" sampling.py "${RUNTIME_CFG}" \
    --data_id "${i}" \
    --device cuda:0 \
    --num_samples "${NUM_SAMPLES}" \
    --batch_size "${BATCH_SIZE}" \
    --num_steps "${NUM_STEPS}" \
    --sample_num_atoms "${SAMPLE_NUM_ATOMS}" \
    --split_path ./data/crossdocked_pocket10_pose_split.pt \
    --lmdb_path ./data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb \
    --result_path "${out_dir}"
done

echo "[Sampling] completed"
