#!/usr/bin/env bash
set -euo pipefail

# Unitree G1 finetune entrypoint for FLUX.2 ImageWAM.
#
# Chains: action-expert init generation -> Qwen3 text cache -> training.
# Weights are initialised from the dim-expanded RoboTwin checkpoint via `resume`,
# which Wan22Trainer loads before accelerator.prepare.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"
imagewam_init "${SCRIPT_DIR}/../.."

GPU_PER_NODE="${GPU_PER_NODE:-4}"
ZERO_STAGE="${ZERO_STAGE:-1}"
TASK_NAME="g1_flux2_klein_4b_base_imagewam"
MODEL_CONFIG="configs/model/imagewam_flux2_klein_4b_base.yaml"
PRECOMPUTE_QWEN3_CACHE="${PRECOMPUTE_QWEN3_CACHE:-false}"

imagewam_require_env FLUX2_SRC
imagewam_require_env FLUX2_MODEL_PATH
imagewam_require_env FLUX2_AE_MODEL_PATH
imagewam_require_env G1_ROOT
imagewam_require_env G1_INIT_CKPT

FLUX2_QWEN3_MODEL_SPEC="${FLUX2_QWEN3_MODEL_SPEC:-Qwen/Qwen3-4B}"
QWEN_CACHE_DIR="${QWEN_CACHE_DIR:-${G1_ROOT}/flux2_qwen3_cache_4b}"
FRAME_FILTER_PATH="${FRAME_FILTER_PATH:-${G1_ROOT}/g1_frame_filter.json}"
ACTION_INIT="${ACTION_INIT:-${MODEL_ROOT:-${REPO_ROOT}/checkpoints}/action_dit_flux2_4b_g1_init.pt}"

export PYTHONPATH="${REPO_ROOT}/src:${FLUX2_SRC}/src:${FLUX2_SRC}${PYTHONPATH:+:${PYTHONPATH}}"
export ZERO_STAGE

DATASET_OVERRIDES=(
  "data.train.dataset_dirs=[${G1_ROOT}]"
  "data.train.nonidle_filter_path=${FRAME_FILTER_PATH}"
  "data.train.qwen_text_cache_dir=${QWEN_CACHE_DIR}"
)

imagewam_print_config TASK_NAME G1_ROOT G1_INIT_CKPT FLUX2_SRC FLUX2_MODEL_PATH \
  FLUX2_AE_MODEL_PATH QWEN_CACHE_DIR FRAME_FILTER_PATH ACTION_INIT GPU_PER_NODE ZERO_STAGE

# 1. Action expert initialisation. Built at action_dim=16 so the module shapes match
#    the G1 config; the weights are then overwritten by the resumed checkpoint.
if [ "${REBUILD_ACTION_INIT:-false}" = "true" ] || [ ! -f "${ACTION_INIT}" ]; then
  imagewam_run imagewam_python scripts/flux2/preprocess_action_dit_flux2.py \
    --model-config "${MODEL_CONFIG}" \
    --flux2-src-path "${FLUX2_SRC}" \
    --flux2-model-path "${FLUX2_MODEL_PATH}" \
    --variant klein-base-4b \
    --action-dim 16 \
    --output "${ACTION_INIT}" \
    --apply-alpha-scaling true
fi

# 2. Qwen3 text embedding cache. Required before any sample can be drawn, because
#    RobotVideoDataset reads the cache whenever qwen_text_cache_dir is set.
if [ "${PRECOMPUTE_QWEN3_CACHE}" = "true" ]; then
  imagewam_run torchrun --standalone --nproc_per_node="${QWEN_CACHE_NPROC:-1}" \
    scripts/flux2/precompute_flux2_qwen3_embeds.py \
    task="${TASK_NAME}" \
    qwen_cache_batch_size="${QWEN_CACHE_BATCH_SIZE:-8}" \
    qwen_cache_save_workers="${QWEN_CACHE_SAVE_WORKERS:-4}" \
    qwen_cache_overwrite="${QWEN_CACHE_OVERWRITE:-false}" \
    model.variant=klein-base-4b \
    model.qwen3_model_spec="${FLUX2_QWEN3_MODEL_SPEC}" \
    model.flux2_model_path="${FLUX2_MODEL_PATH}" \
    model.ae_model_path="${FLUX2_AE_MODEL_PATH}" \
    flux2_qwen3_model_spec="${FLUX2_QWEN3_MODEL_SPEC}" \
    "${DATASET_OVERRIDES[@]}"
fi

# 3. Training.
TASK="${TASK_NAME}" imagewam_run bash scripts/flux2/train_flux2_klein_imagewam.sh "${GPU_PER_NODE}" \
  "${DATASET_OVERRIDES[@]}" \
  "model.flux2_model_path=${FLUX2_MODEL_PATH}" \
  "model.ae_model_path=${FLUX2_AE_MODEL_PATH}" \
  "model.qwen3_model_spec=${FLUX2_QWEN3_MODEL_SPEC}" \
  "model.action_dit_pretrained_path=${ACTION_INIT}" \
  "resume=${G1_INIT_CKPT}" \
  "$@"
