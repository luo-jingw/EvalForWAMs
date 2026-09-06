#!/usr/bin/env bash
set -euo pipefail

# Single-task (Stack-the-cubes) G1 finetune of the RoboTwin FLUX.2-klein-4B checkpoint.
# Train uses gripper-transition loss weighting (configs/data/g1_stack_cubes.yaml).
#
# Differs from run_train_g1_flux2.sh: the task is g1_stack_cubes_flux2_klein_4b and
# there is no frame filter (this dataset has no outlier episodes), so the config's
# nonidle_filter_path=null is left untouched.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"
imagewam_init "${SCRIPT_DIR}/../.."

GPU_PER_NODE="${GPU_PER_NODE:-1}"
ZERO_STAGE="${ZERO_STAGE:-1}"
TASK_NAME="g1_stack_cubes_flux2_klein_4b"
MODEL_CONFIG="configs/model/imagewam_flux2_klein_4b_base.yaml"

imagewam_require_env FLUX2_SRC
imagewam_require_env FLUX2_MODEL_PATH
imagewam_require_env FLUX2_AE_MODEL_PATH
imagewam_require_env G1_INIT_CKPT

# Dataset root for this task; independent of the multi-task G1_ROOT in .env.local.
STACK_ROOT="${STACK_ROOT:-${REPO_ROOT}/../dataset/stack_the_cubes}"
FLUX2_QWEN3_MODEL_SPEC="${FLUX2_QWEN3_MODEL_SPEC:-Qwen/Qwen3-4B}"
QWEN_CACHE_DIR="${QWEN_CACHE_DIR:-${STACK_ROOT}/flux2_qwen3_cache_4b}"
# The dim-expanded RoboTwin init is task-agnostic; reuse it.
ACTION_INIT="${ACTION_INIT:-${MODEL_ROOT:-${REPO_ROOT}/checkpoints}/action_dit_flux2_4b_g1_init.pt}"

export PYTHONPATH="${REPO_ROOT}/src:${FLUX2_SRC}/src:${FLUX2_SRC}${PYTHONPATH:+:${PYTHONPATH}}"
export ZERO_STAGE

DATASET_OVERRIDES=(
  "data.train.dataset_dirs=[${STACK_ROOT}]"
  "data.train.qwen_text_cache_dir=${QWEN_CACHE_DIR}"
)

imagewam_print_config TASK_NAME STACK_ROOT G1_INIT_CKPT FLUX2_SRC FLUX2_MODEL_PATH \
  FLUX2_AE_MODEL_PATH QWEN_CACHE_DIR ACTION_INIT GPU_PER_NODE ZERO_STAGE

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

if [ "${PRECOMPUTE_QWEN3_CACHE:-false}" = "true" ]; then
  imagewam_run torchrun --standalone --nproc_per_node="${QWEN_CACHE_NPROC:-1}" \
    scripts/flux2/precompute_flux2_qwen3_embeds.py \
    task="${TASK_NAME}" \
    model.variant=klein-base-4b \
    model.qwen3_model_spec="${FLUX2_QWEN3_MODEL_SPEC}" \
    model.flux2_model_path="${FLUX2_MODEL_PATH}" \
    model.ae_model_path="${FLUX2_AE_MODEL_PATH}" \
    flux2_qwen3_model_spec="${FLUX2_QWEN3_MODEL_SPEC}" \
    "${DATASET_OVERRIDES[@]}"
fi

TASK="${TASK_NAME}" imagewam_run bash scripts/flux2/train_flux2_klein_imagewam.sh "${GPU_PER_NODE}" \
  "${DATASET_OVERRIDES[@]}" \
  "model.flux2_model_path=${FLUX2_MODEL_PATH}" \
  "model.ae_model_path=${FLUX2_AE_MODEL_PATH}" \
  "model.qwen3_model_spec=${FLUX2_QWEN3_MODEL_SPEC}" \
  "model.action_dit_pretrained_path=${ACTION_INIT}" \
  "resume=${G1_INIT_CKPT}" \
  "$@"
