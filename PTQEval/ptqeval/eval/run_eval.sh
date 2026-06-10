#!/bin/bash
# Orchestrator for WAM RoboTwin evaluation with perf logging.
# Modes:
#   smoke  : 1 GPU, 1 task, test_num=1. Sanity check.
#   single : 1 GPU, sequential loop over selected task list.
#   pool   : N GPUs, worker pool + shared queue. Each worker locks one GPU,
#            pops next task, restarts server per task. Handles resume and
#            VRAM checks. Preferred for full multi-GPU eval.
#
# Configuration via env vars (Section 25.5 of plan.txt):
#   WAM_NAME              default "lingbot_va"
#   WAM_MODEL_PATH        default /home/arash/EvalForWAMs/models/lingbot-va-posttrain-robotwin
#   ROBOTWIN_ROOT         default /home/arash/EvalForWAMs/RoboTwin
#   SAVE_ROOT             default /home/arash/EvalForWAMs/results/<variant_tag>
#                         (variant_tag = bf16 when VARIANT unset; else VARIANT
#                          or VARIANT_<variant_args basename>)
#   PERF_LOG_DIR          default ${SAVE_ROOT}/perf
#   VARIANT               optional ("viditq" or unset for bf16)
#   VARIANT_ARGS          optional, path to runtime_args_*.yaml
#   SERVER_ENV            default "lingbot-jw"
#   CLIENT_ENV            default "RoboTwin-jw"
#
# CLI args:
#   --mode <smoke|single|pool>   required
#   --task_name <name>           for smoke / single
#   --test_num <int>             default 25, smoke uses 1
#   --gpu_id <int>               default 0 for smoke / single
#   --seed <int>                 default 0
#   --min_free_mb <int>          default 40000
#   --gpu_wait_timeout <int>     default 0 (wait forever)
#   --gpu_poll_interval <int>    default 30
#   --rerun_all                  pool: include all SELECTED_TASKS regardless of prior state
set -euo pipefail

# ---- env-var configuration ----
WAM_NAME="${WAM_NAME:-lingbot_va}"
WAM_MODEL_PATH="${WAM_MODEL_PATH:-/home/arash/EvalForWAMs/models/lingbot-va-posttrain-robotwin}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/home/arash/EvalForWAMs/RoboTwin}"
VARIANT="${VARIANT:-}"
VARIANT_ARGS="${VARIANT_ARGS:-}"
CALIBRATE_OUT="${CALIBRATE_OUT:-}"
# Phase 31 v2: TASK_LIST_NAME picks which list in tasks.py to iterate.
# Default SELECTED_15_TASKS (production eval). CALIB_TASKS_ALL covers
# all 50 RoboTwin tasks for broader calibration coverage.
TASK_LIST_NAME="${TASK_LIST_NAME:-SELECTED_15_TASKS}"
SERVER_ENV="${SERVER_ENV:-lingbot-jw}"
CLIENT_ENV="${CLIENT_ENV:-RoboTwin-jw}"

if [ -z "${SAVE_ROOT:-}" ]; then
    if [ -n "$VARIANT" ] && [ -n "$VARIANT_ARGS" ]; then
        va_base="$(basename "$VARIANT_ARGS")"
        va_tag="${va_base%.yaml}"; va_tag="${va_tag%.yml}"
        SAVE_ROOT="/home/arash/EvalForWAMs/results/${VARIANT}_${va_tag}"
    elif [ -n "$VARIANT" ]; then
        SAVE_ROOT="/home/arash/EvalForWAMs/results/${VARIANT}"
    else
        SAVE_ROOT="/home/arash/EvalForWAMs/results/bf16"
    fi
fi
PERF_LOG_DIR="${PERF_LOG_DIR:-${SAVE_ROOT}/perf}"
mkdir -p "$SAVE_ROOT" "$PERF_LOG_DIR"

# ---- SELECTED_TASKS via tasks.py (TASK_LIST_NAME selects which list) ----
read -r -a SELECTED_TASKS <<< "$(
    python -c "from ptqeval.wam.${WAM_NAME}.tasks import ${TASK_LIST_NAME} as t; print(' '.join(t))"
)"
if [ "${#SELECTED_TASKS[@]}" -eq 0 ]; then
    echo "[run_eval.sh] failed to load ${TASK_LIST_NAME} from ptqeval.wam.${WAM_NAME}.tasks" >&2
    exit 1
fi

# ---- CLI parsing ----
MODE=""
TASK_NAME=""
TEST_NUM=""
GPU_ID="0"
SEED="0"
MIN_FREE_MB="${MIN_FREE_MB:-40000}"
GPU_WAIT_TIMEOUT="${GPU_WAIT_TIMEOUT:-0}"
GPU_POLL_INTERVAL="${GPU_POLL_INTERVAL:-30}"
RERUN_ALL="false"

print_usage() {
    cat <<EOF
Usage: run_eval.sh --mode {smoke,single,pool} [opts]
  --mode <smoke|single|pool>   required
  --task_name <name>           smoke / single. default adjust_bottle for smoke
  --test_num <int>             default 25, smoke uses 1
  --gpu_id <int>               default 0 for smoke / single
  --seed <int>                 default 0
  --min_free_mb <int>          default 40000
  --gpu_wait_timeout <int>     default 0 (wait forever)
  --gpu_poll_interval <int>    default 30
  --rerun_all                  pool: ignore prior res.json
Env vars: WAM_NAME, WAM_MODEL_PATH, ROBOTWIN_ROOT, SAVE_ROOT, PERF_LOG_DIR,
          VARIANT, VARIANT_ARGS, SERVER_ENV, CLIENT_ENV
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --mode) MODE="$2"; shift 2 ;;
        --task_name) TASK_NAME="$2"; shift 2 ;;
        --test_num) TEST_NUM="$2"; shift 2 ;;
        --gpu_id) GPU_ID="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --min_free_mb) MIN_FREE_MB="$2"; shift 2 ;;
        --gpu_wait_timeout) GPU_WAIT_TIMEOUT="$2"; shift 2 ;;
        --gpu_poll_interval) GPU_POLL_INTERVAL="$2"; shift 2 ;;
        --rerun_all) RERUN_ALL="true"; shift 1 ;;
        -h|--help) print_usage; exit 0 ;;
        *) echo "Unknown arg: $1"; print_usage; exit 1 ;;
    esac
done

if [ -z "$MODE" ]; then
    print_usage; exit 1
fi

# ---- Conda activation helper ----
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# ---- Cleanup: kill only servers this script launched ----
SERVER_PIDS=()
cleanup_all() {
    local pid
    for pid in "${SERVER_PIDS[@]:-}"; do
        [ -z "$pid" ] && continue
        kill "$pid" 2>/dev/null || true
    done
    sleep 1
    for pid in "${SERVER_PIDS[@]:-}"; do
        [ -z "$pid" ] && continue
        kill -9 "$pid" 2>/dev/null || true
    done
}
trap cleanup_all EXIT INT TERM

# ---- Common helpers ----
wait_for_port() {
    local port="$1"
    local timeout="${2:-300}"
    local elapsed=0
    while ! (echo > /dev/tcp/127.0.0.1/${port}) 2>/dev/null; do
        sleep 2
        elapsed=$((elapsed + 2))
        if [ "$elapsed" -ge "$timeout" ]; then
            echo "Timeout waiting for port ${port}" >&2
            return 1
        fi
    done
}

query_gpu_free_mb() {
    local gpu="$1"
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | tr -d ' '
}

wait_for_gpu_memory() {
    local gpu="$1"
    local elapsed=0
    while :; do
        local free_mb
        free_mb="$(query_gpu_free_mb "$gpu")"
        if ! [[ "$free_mb" =~ ^[0-9]+$ ]]; then
            echo "[wait_for_gpu_memory] nvidia-smi returned non-numeric '${free_mb}' for GPU ${gpu}" >&2
            return 1
        fi
        if [ "$free_mb" -ge "$MIN_FREE_MB" ]; then
            echo "[wait_for_gpu_memory] GPU ${gpu} has ${free_mb} MB free (>= ${MIN_FREE_MB} MB). Proceeding."
            return 0
        fi
        echo "[wait_for_gpu_memory] GPU ${gpu} has only ${free_mb} MB free, need ${MIN_FREE_MB} MB. Sleeping ${GPU_POLL_INTERVAL}s (elapsed ${elapsed}s)."
        sleep "$GPU_POLL_INTERVAL"
        elapsed=$((elapsed + GPU_POLL_INTERVAL))
        if [ "$GPU_WAIT_TIMEOUT" -gt 0 ] && [ "$elapsed" -ge "$GPU_WAIT_TIMEOUT" ]; then
            echo "[wait_for_gpu_memory] Timeout after ${elapsed}s waiting for GPU ${gpu}" >&2
            return 1
        fi
    done
}

start_server() {
    local gpu="$1"
    local port="$2"
    local master_port="$3"
    local task_name="$4"
    local server_log="$5"

    local variant_args_cli=""
    if [ -n "$VARIANT" ]; then
        variant_args_cli="--variant $VARIANT"
        if [ -n "$VARIANT_ARGS" ]; then
            variant_args_cli="$variant_args_cli --variant_args $VARIANT_ARGS"
        fi
    fi
    # Phase 31: forward CALIBRATE_OUT through to server.py if set.
    local calibrate_cli=""
    if [ -n "$CALIBRATE_OUT" ]; then
        calibrate_cli="--calibrate_out $CALIBRATE_OUT"
    fi

    conda activate "$SERVER_ENV"
    (
        # M5: forked server resolves via ptqeval pip-install; no cwd dependency.
        CUDA_VISIBLE_DEVICES="$gpu" \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        nohup python -m torch.distributed.run \
            --nproc_per_node 1 \
            --master_port "$master_port" \
            --module ptqeval.wam.lingbot_va.server \
            --config-name robotwin \
            --port "$port" \
            --save_root "${SAVE_ROOT}/visualization" \
            --perf_log_dir "$PERF_LOG_DIR" \
            --perf_task_name "$task_name" \
            --model_path "$WAM_MODEL_PATH" \
            $variant_args_cli \
            $calibrate_cli \
            > "$server_log" 2>&1 &
        echo $! > "${server_log}.pid"
    )
    conda deactivate
    SERVER_PIDS+=("$(cat "${server_log}.pid")")
}

start_client() {
    local task_name="$1"
    local port="$2"
    local test_num="$3"
    local client_log="$4"
    local gpu="$5"

    conda activate "$CLIENT_ENV"
    (
        # M5: forked client; ROBOTWIN_ROOT exported so eval_client.py finds it.
        export LD_LIBRARY_PATH="/usr/lib64:/usr/lib:${LD_LIBRARY_PATH:-}"
        export ROBOTWIN_ROOT
        CUDA_VISIBLE_DEVICES="$gpu" \
        PYTHONWARNINGS=ignore::UserWarning \
        XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
        python -m ptqeval.wam.lingbot_va.eval_client \
            --config "${ROBOTWIN_ROOT}/policy/ACT/deploy_policy.yml" \
            --overrides \
            --task_name "$task_name" \
            --task_config demo_clean \
            --train_config_name 0 \
            --model_name 0 \
            --ckpt_setting 0 \
            --seed "$SEED" \
            --policy_name ACT \
            --save_root "$SAVE_ROOT" \
            --video_guidance_scale 5 \
            --action_guidance_scale 1 \
            --test_num "$test_num" \
            --port "$port" \
            > "$client_log" 2>&1
    )
    conda deactivate
}

kill_pid_file() {
    local pidfile="$1"
    if [ -f "$pidfile" ]; then
        local pid
        pid="$(cat "$pidfile")"
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" || true
            sleep 2
            kill -9 "$pid" 2>/dev/null || true
        fi
        rm -f "$pidfile"
    fi
}

# ---- Mode implementations ----
run_one_task() {
    local task="$1"
    local gpu="$2"
    local port="$3"
    local master_port="$4"
    local test_num="$5"
    local tag="$6"

    local log_dir="${SAVE_ROOT}/logs/${tag}"
    mkdir -p "$log_dir"
    local server_log="${log_dir}/server_${task}.log"
    local client_log="${log_dir}/client_${task}.log"

    echo "[run_one_task] task=${task} gpu=${gpu} port=${port} test_num=${test_num}"
    wait_for_gpu_memory "$gpu"
    start_server "$gpu" "$port" "$master_port" "$task" "$server_log"
    wait_for_port "$port" 600
    start_client "$task" "$port" "$test_num" "$client_log" "$gpu"
    kill_pid_file "${server_log}.pid"
}

run_smoke() {
    local task="${TASK_NAME:-adjust_bottle}"
    local test_num="${TEST_NUM:-1}"
    run_one_task "$task" "$GPU_ID" 29056 29061 "$test_num" "smoke"
}

run_single() {
    local test_num="${TEST_NUM:-25}"
    if [ -n "$TASK_NAME" ]; then
        run_one_task "$TASK_NAME" "$GPU_ID" 29056 29061 "$test_num" "single"
        return
    fi
    for task in "${SELECTED_TASKS[@]}"; do
        run_one_task "$task" "$GPU_ID" 29056 29061 "$test_num" "single"
    done
}

# Returns 0 if the task still needs to run: no res.json, OR total_num < test_num.
task_needs_run() {
    local task="$1"
    local target="$2"
    local f="${SAVE_ROOT}/stseed-10000/metrics/${task}/res.json"
    [ -f "$f" ] || return 0
    local total
    total=$(python3 -c "import json,sys; print(int(json.load(open(sys.argv[1]))['total_num']))" "$f" 2>/dev/null || echo 0)
    [ "$total" -lt "$target" ] && return 0
    return 1
}

pop_task() {
    local queue_file="$1"
    local lock_file="$2"
    (
        flock -x 200
        local t
        t=$(head -1 "$queue_file" 2>/dev/null)
        if [ -n "$t" ]; then
            tail -n +2 "$queue_file" > "${queue_file}.tmp" && mv "${queue_file}.tmp" "$queue_file"
        fi
        echo "$t"
    ) 200>"$lock_file"
}

run_pool() {
    local test_num="${TEST_NUM:-25}"
    local -a queue=()
    local task
    for task in "${SELECTED_TASKS[@]}"; do
        if [ "$RERUN_ALL" = "true" ] || task_needs_run "$task" "$test_num"; then
            queue+=("$task")
        fi
    done
    if [ "${#queue[@]}" -eq 0 ]; then
        echo "[pool] all tasks already have >= ${test_num} episodes. Nothing to do."
        return
    fi

    local log_dir="${SAVE_ROOT}/logs/pool"
    mkdir -p "$log_dir"
    local queue_file="${log_dir}/queue.txt"
    local lock_file="${queue_file}.lock"
    printf "%s\n" "${queue[@]}" > "$queue_file"
    echo "[pool] queue (${#queue[@]} tasks):"
    cat "$queue_file" | sed 's/^/  - /'

    local start_port=29556
    local start_master_port=29661
    local -a usable_gpus=()
    local g free_mb
    for g in $(seq 0 7); do
        free_mb="$(query_gpu_free_mb "$g")"
        if [[ "$free_mb" =~ ^[0-9]+$ ]] && [ "$free_mb" -ge "$MIN_FREE_MB" ]; then
            usable_gpus+=("$g")
        else
            echo "[pool] skipping GPU ${g}: free=${free_mb}MB < ${MIN_FREE_MB}MB"
        fi
    done
    if [ "${#usable_gpus[@]}" -eq 0 ]; then
        echo "[pool] no GPU has >= ${MIN_FREE_MB} MB free. Lower --min_free_mb or wait." >&2
        return 1
    fi
    local n_workers="${#usable_gpus[@]}"
    [ "${#queue[@]}" -lt "$n_workers" ] && n_workers="${#queue[@]}"
    echo "[pool] using GPUs: ${usable_gpus[*]:0:$n_workers}"

    local -a worker_pids=()
    local wi
    for wi in $(seq 0 $((n_workers - 1))); do
        g="${usable_gpus[$wi]}"
        (
            local port=$((start_port + g))
            local mport=$((start_master_port + g))
            while :; do
                local t
                t=$(pop_task "$queue_file" "$lock_file")
                [ -z "$t" ] && break
                local server_log="${log_dir}/server_${g}_${t}.log"
                local client_log="${log_dir}/client_${g}_${t}.log"
                echo "[pool worker gpu=${g}] task=${t}"
                wait_for_gpu_memory "$g"
                start_server "$g" "$port" "$mport" "$t" "$server_log"
                wait_for_port "$port" 600 || true
                start_client "$t" "$port" "$test_num" "$client_log" "$g"
                kill_pid_file "${server_log}.pid"
            done
            echo "[pool worker gpu=${g}] queue empty, exit"
        ) &
        worker_pids+=($!)
    done

    local pid
    for pid in "${worker_pids[@]}"; do
        wait "$pid" || true
    done
}

case "$MODE" in
    smoke) run_smoke ;;
    single) run_single ;;
    pool) run_pool ;;
    *) print_usage; exit 1 ;;
esac

echo "[run_eval.sh] mode=${MODE} done. logs under ${SAVE_ROOT}/logs/, perf under ${PERF_LOG_DIR}/"
