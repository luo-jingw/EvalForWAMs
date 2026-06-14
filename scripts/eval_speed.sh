#!/usr/bin/env bash
# Speed-only eval for the three production variants (bf16 + viditq
# dynamic + viditq static).  Runs test_num=5 with --profile_ops so each
# server writes a per-task <task>_op_profile.json into its perf log dir.
#
# Results land in *_speed/ sibling directories so they do NOT clobber
# the authoritative 25-ep production runs.  After this script finishes,
# run aggregate_and_summary_speed.sh (or manually) to merge per-task
# op_profile JSONs and render cross_summary_speed/.
#
# Sequential, single GPU (mode=single) -- adjust GPU_ID below as needed.
# Profiler overhead is real (~5-10x for the instrumented calls only),
# so total wall clock ~= 30-60 min per variant.
set -euo pipefail

REPO=/home/arash/EvalForWAMs
PY=/home/arash/.conda/envs/lingbot-jw/bin/python
CFG_DIR=${REPO}/PTQEval/ptqeval/wam/lingbot_va/method/viditq/configs

TEST_NUM=5
PROFILE_N=5
TASK_CONFIG=demo_randomized
GPU_ID=${GPU_ID:-0}    # override via env

# --- bf16 ---------------------------------------------------------------
${PY} -m ptqeval.eval.run_eval --mode pool \
    --task_config ${TASK_CONFIG} --test_num ${TEST_NUM} \
    --save_root ${REPO}/results/bf16_speed \
    --profile_ops --profile_n_calls ${PROFILE_N}

# --- viditq dynamic (SmoothQuant + QuaRoT, no static act) --------------
${PY} -m ptqeval.eval.run_eval --mode pool --variant viditq \
    --variant_args ${CFG_DIR}/runtime_args_w8a8_viditq.yaml \
    --task_config ${TASK_CONFIG} --test_num ${TEST_NUM} \
    --save_root ${REPO}/results/viditq_w8a8_viditq_speed \
    --profile_ops --profile_n_calls ${PROFILE_N}

# --- viditq static (SmoothQuant + QuaRoT + static activation) ----------
${PY} -m ptqeval.eval.run_eval --mode pool --variant viditq \
    --variant_args ${CFG_DIR}/runtime_args_w8a8_viditq_static.yaml \
    --task_config ${TASK_CONFIG} --test_num ${TEST_NUM} \
    --save_root ${REPO}/results/viditq_w8a8_viditq_static_speed \
    --profile_ops --profile_n_calls ${PROFILE_N}

echo
echo "All three speed runs complete."
echo "Next: aggregator each + cross_summary_speed with --op_profile, e.g."
echo
cat <<'POST'
for d in bf16_speed viditq_w8a8_viditq_speed viditq_w8a8_viditq_static_speed; do
    python -m ptqeval.eval.aggregator \
        --save_root results/${d} \
        --perf_log_dir results/${d}/perf \
        --out_dir results/${d}/summary
done

python -m ptqeval.eval.calc_cross_ckpt \
    --variant bf16=results/bf16_speed/summary/summary.csv \
    --variant viditq_w8a8_dynamic=results/viditq_w8a8_viditq_speed/summary/summary.csv \
    --variant viditq_w8a8_static=results/viditq_w8a8_viditq_static_speed/summary/summary.csv \
    --op_profile bf16=results/bf16_speed/summary/op_profile.json \
    --op_profile viditq_w8a8_dynamic=results/viditq_w8a8_viditq_speed/summary/op_profile.json \
    --op_profile viditq_w8a8_static=results/viditq_w8a8_viditq_static_speed/summary/op_profile.json \
    --out_dir results/cross_summary_speed
POST
