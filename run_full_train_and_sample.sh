#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

TRAIN_MAX_ITERS="${TRAIN_MAX_ITERS:-71000}"
TRAIN_LOGDIR="${TRAIN_LOGDIR:-./logs_diffusion_full}"
TRAIN_TAG="${TRAIN_TAG:-cjkim_full_gpu}"
SAMPLE_OUT_ROOT="${SAMPLE_OUT_ROOT:-./sampling_results_full_test100}"
SAMPLE_NUM_SAMPLES="${SAMPLE_NUM_SAMPLES:-100}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-16}"
SAMPLE_NUM_STEPS="${SAMPLE_NUM_STEPS:-1000}"
SAMPLE_NUM_ATOMS="${SAMPLE_NUM_ATOMS:-prior}"

mkdir -p "${TRAIN_LOGDIR}" "${SAMPLE_OUT_ROOT}"

echo "[Full Pipeline] Starting training"
echo "[Full Pipeline] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[Full Pipeline] TRAIN_MAX_ITERS=${TRAIN_MAX_ITERS}"

"${PYTHON_BIN}" train_diffusion.py \
  --config ./config.yml \
  --device cuda \
  --logdir "${TRAIN_LOGDIR}" \
  --tag "${TRAIN_TAG}" \
  --max_iters "${TRAIN_MAX_ITERS}"

CKPT_DIR="${TRAIN_LOGDIR}/targetdiff_${TRAIN_TAG}/checkpoints"
CKPT_PATH="$(find "${CKPT_DIR}" -maxdepth 1 -name '*.pt' -printf '%f\n' | sort -n | tail -n 1)"
if [[ -z "${CKPT_PATH}" ]]; then
  echo "[Full Pipeline] No checkpoint found in ${CKPT_DIR}"
  exit 1
fi
CKPT_PATH="${CKPT_DIR}/${CKPT_PATH}"
echo "[Full Pipeline] Training complete. Using checkpoint: ${CKPT_PATH}"

RUNTIME_CFG="./sampling_runtime_full.yml"
"${PYTHON_BIN}" -c "import yaml; d=yaml.safe_load(open('sampling.yml')); d['model']['checkpoint']='${CKPT_PATH}'; yaml.safe_dump(d, open('${RUNTIME_CFG}','w'), sort_keys=False)"

echo "[Full Pipeline] Starting test set sampling"
for i in $(seq 0 99); do
  out_dir="${SAMPLE_OUT_ROOT}/data_$(printf '%03d' "${i}")"
  echo "[Full Pipeline] Sampling data_id=${i} -> ${out_dir}"
  "${PYTHON_BIN}" sampling.py "${RUNTIME_CFG}" \
    --data_id "${i}" \
    --device cuda:0 \
    --num_samples "${SAMPLE_NUM_SAMPLES}" \
    --batch_size "${SAMPLE_BATCH_SIZE}" \
    --num_steps "${SAMPLE_NUM_STEPS}" \
    --sample_num_atoms "${SAMPLE_NUM_ATOMS}" \
    --split_path ./data/crossdocked_pocket10_pose_split.pt \
    --lmdb_path ./data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb \
    --result_path "${out_dir}"
done

echo "[Full Pipeline] Completed all sampling jobs"
