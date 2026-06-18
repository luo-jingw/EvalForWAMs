#!/usr/bin/env bash
# Speed-only eval for the four production variants (bf16 + viditq W8A8
# dynamic + viditq W8A8 static + viditq W4A8 dynamic).  Runs test_num=5
# with --profile_ops so each server writes a per-task
# <task>_op_profile.json into its perf log dir.
#
# As of 2026-06-17, run_eval --profile_ops is default-on
# (BooleanOptionalAction, default=True) and --profile_n_calls defaults
# to 5; the explicit flags below pin the intent for this _speed eval and
# remain safe if the defaults change.
#
# Results land in *_speed/ sibling directories so they do NOT clobber
# the authoritative 25-ep production runs.  After this script finishes,
# run aggregator on each + cross_ckpt to render cross_summary_speed/.
#
# Pool mode -- all usable GPUs on the host are auto-detected by
# _pool_runner.py (free VRAM >= --min_free_mb).  Profiler overhead is
# real (~5-10x for the instrumented calls only), so total wall clock
# ~= 30-60 min per variant on a single A6000.
set -euo pipefail

REPO=/home/arash/EvalForWAMs
PY=/home/arash/.conda/envs/lingbot-jw/bin/python
CFG_DIR=${REPO}/PTQEval/ptqeval/wam/lingbot_va/method/viditq/configs

TEST_NUM=5
PROFILE_N=5
TASK_CONFIG=demo_randomized

# --- bf16 ---------------------------------------------------------------
${PY} -m ptqeval.eval.run_eval --mode pool \
    --task_config ${TASK_CONFIG} --test_num ${TEST_NUM} \
    --save_root ${REPO}/results/bf16_speed \
    --profile_ops --profile_n_calls ${PROFILE_N}

# --- viditq dynamic (SmoothQuant + QuaRoT, no static act) --------------
${PY} -m ptqeval.eval.run_eval --mode pool --variant viditq \
    --variant_args ${CFG_DIR}/runtime_args_w8a8_dynamic.yaml \
    --task_config ${TASK_CONFIG} --test_num ${TEST_NUM} \
    --save_root ${REPO}/results/viditq_w8a8_dynamic_speed \
    --profile_ops --profile_n_calls ${PROFILE_N}

# --- viditq W8A8 static (SmoothQuant + QuaRoT + static activation) -----
${PY} -m ptqeval.eval.run_eval --mode pool --variant viditq \
    --variant_args ${CFG_DIR}/runtime_args_w8a8_static.yaml \
    --task_config ${TASK_CONFIG} --test_num ${TEST_NUM} \
    --save_root ${REPO}/results/viditq_w8a8_static_speed \
    --profile_ops --profile_n_calls ${PROFILE_N}

# --- viditq W4A8 dynamic (SmoothQuant + QuaRoT, W4 + dynamic act) ------
${PY} -m ptqeval.eval.run_eval --mode pool --variant viditq \
    --variant_args ${CFG_DIR}/runtime_args_w4a8_dynamic.yaml \
    --task_config ${TASK_CONFIG} --test_num ${TEST_NUM} \
    --save_root ${REPO}/results/viditq_w4a8_dynamic_speed \
    --profile_ops --profile_n_calls ${PROFILE_N}

echo
echo "All four speed runs complete."
echo "Next: aggregator each + cross_summary_speed with --op_profile, e.g."
echo
cat <<'POST'
for d in bf16_speed viditq_w8a8_dynamic_speed viditq_w8a8_static_speed viditq_w4a8_dynamic_speed; do
    python -m ptqeval.eval.aggregator \
        --save_root results/${d} \
        --perf_log_dir results/${d}/perf \
        --out_dir results/${d}/summary
done

python -m ptqeval.eval.calc_cross_ckpt \
    --variant bf16=results/bf16_speed/summary/summary.csv \
    --variant viditq_w8a8_dynamic=results/viditq_w8a8_dynamic_speed/summary/summary.csv \
    --variant viditq_w8a8_static=results/viditq_w8a8_static_speed/summary/summary.csv \
    --variant viditq_w4a8_dynamic=results/viditq_w4a8_dynamic_speed/summary/summary.csv \
    --op_profile bf16=results/bf16_speed/summary/op_profile.json \
    --op_profile viditq_w8a8_dynamic=results/viditq_w8a8_dynamic_speed/summary/op_profile.json \
    --op_profile viditq_w8a8_static=results/viditq_w8a8_static_speed/summary/op_profile.json \
    --op_profile viditq_w4a8_dynamic=results/viditq_w4a8_dynamic_speed/summary/op_profile.json \
    --out_dir results/cross_summary_speed
POST
