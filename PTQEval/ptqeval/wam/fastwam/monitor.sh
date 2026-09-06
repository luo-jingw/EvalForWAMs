#!/bin/bash
# Read-only status dashboard for the fastwam eval pool.
# Designed for: watch -n 5 PTQEval/ptqeval/wam/fastwam/monitor.sh
#
# fastwam layout (differs from the lingbot_va server/client dashboard in
# ptqeval/eval/monitor.sh, which this does not touch):
#   <SAVE_ROOT>/pool.log                          pool-level start/done lines
#   <SAVE_ROOT>/logs/<task>.log                   one flat log per task
#   <SAVE_ROOT>/stseed-<seed>/metrics/<task>/res.json   final SR per task
#   <SAVE_ROOT>/perf/<task>_rank<gpu>_<ts>.jsonl  PerfProbe records
set -u

SAVE_ROOT="${SAVE_ROOT:-results/fastwam/fastwam_w4a4}"
PERF_LOG_DIR="${PERF_LOG_DIR:-${SAVE_ROOT}/perf}"
LOG_DIR="${LOG_DIR:-${SAVE_ROOT}/logs}"
POOL_LOG="${POOL_LOG:-${SAVE_ROOT}/pool.log}"

strip_ansi() { sed 's/\x1b\[[0-9;]*m//g'; }

# Latest "Success rate: N/M" in a task log -> "N/M" (M = episodes finished).
get_sr() {
    local log="$1" s
    [ -f "$log" ] || { echo "-"; return; }
    s="$(tail -c 16384 "$log" 2>/dev/null | strip_ansi \
        | grep -oE 'Success rate: [0-9]+/[0-9]+' | tail -1 | sed 's/Success rate: //')"
    echo "${s:--}"
}

latest_perf_for() { ls -t "$PERF_LOG_DIR/${1}_rank"*".jsonl" 2>/dev/null | head -1; }

get_perf_count() {
    local f="${1:-}"
    [ -n "$f" ] && [ -f "$f" ] || { echo 0; return; }
    wc -l < "$f" | tr -d ' '
}

get_last_total_ms() {
    local f="${1:-}" t
    [ -n "$f" ] && [ -f "$f" ] || { echo "-"; return; }
    t="$(tail -1 "$f" 2>/dev/null | grep -oE '"total_ms": [0-9.]+' | head -1 | awk '{printf "%.0f", $2}')"
    echo "${t:--}"
}

check_error() {
    local log="$1"
    [ -f "$log" ] || { echo ""; return; }
    grep -q "OutOfMemoryError" "$log" 2>/dev/null && { echo "OOM"; return; }
    grep -q "Traceback" "$log" 2>/dev/null && { echo "ERR"; return; }
    echo ""
}

# GPU a task was last dispatched to, from pool.log ("gpuN start <task>").
gpu_of_task() {
    local task="$1" g
    [ -f "$POOL_LOG" ] || { echo "-"; return; }
    g="$(grep -E "gpu[0-9]+ start ${task}\$" "$POOL_LOG" 2>/dev/null | tail -1 \
        | grep -oE 'gpu[0-9]+' | tail -1 | sed 's/gpu//')"
    echo "${g:--}"
}

printf "=== %s   SAVE_ROOT=%s ===\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$SAVE_ROOT"

# ---- GPU table ----
printf "\nGPU  Used/Total(MB)   Free(MB)   Util%%\n"
nvidia-smi --query-gpu=index,memory.used,memory.total,memory.free,utilization.gpu \
    --format=csv,noheader,nounits 2>/dev/null | \
    awk -F', *' '{printf "%-3s  %6s / %-6s   %-8s   %s\n", $1, $2, $3, $4, $5}'

# ---- pool progress ----
if [ -f "$POOL_LOG" ]; then
    # grep -c prints 0 and exits 1 when there is no match; do not add `|| echo 0`
    # or the substitution captures two lines and breaks the arithmetic below.
    n_started=$(grep -cE 'gpu[0-9]+ start ' "$POOL_LOG" 2>/dev/null)
    n_finished=$(grep -cE 'gpu[0-9]+ done ' "$POOL_LOG" 2>/dev/null)
    n_failed=$(grep -cE 'gpu[0-9]+ done .*FAILED' "$POOL_LOG" 2>/dev/null)
    head_line=$(grep -m1 'variant=' "$POOL_LOG" 2>/dev/null | sed 's/^\[[^]]*\] //')
    printf "\npool: %s\n" "${head_line:-(no header)}"
    printf "pool: started=%s finished=%s failed=%s running=%s\n" \
        "$n_started" "$n_finished" "$n_failed" "$((n_started - n_finished))"
    printf "last: %s\n" "$(tail -1 "$POOL_LOG" 2>/dev/null | sed 's/^\[[^]]*\] //')"
else
    printf "\n(no pool.log at %s)\n" "$POOL_LOG"
fi

# ---- DONE summary (res.json survives log cleanup) ----
n_done=0
done_rows=""
for f in "${SAVE_ROOT}"/stseed-*/metrics/*/res.json; do
    [ -f "$f" ] || continue
    n_done=$((n_done + 1))
    done_rows="${done_rows}$(python3 -c "
import json,sys
d=json.load(open(sys.argv[1]))
s=int(d.get('succ_num',0)); t=int(d.get('total_num',0))
print(f\"  {d.get('task_name',sys.argv[2]):<28} {s}/{t} ({s/t*100:.0f}%)\" if t else f\"  {sys.argv[2]:<28} -\")
" "$f" "$(basename "$(dirname "$f")")" 2>/dev/null)
"
done
printf "\nDONE (res.json): %s task(s)\n" "$n_done"
[ "$n_done" -gt 0 ] && printf "%s" "$(echo "$done_rows" | sort)" && printf "\n"

# ---- running / recent task table ----
printf "\n%-28s %-4s %-10s %-7s %-9s %-4s\n" "task" "GPU" "SR(ep)" "calls" "last_ms" "err"
printf -- "---------------------------------------------------------------------\n"
for log in "$LOG_DIR"/*.log; do
    [ -f "$log" ] || continue
    task="$(basename "$log" .log)"
    # skip tasks already finalized (res.json present) to keep the table short
    if ls "${SAVE_ROOT}"/stseed-*/metrics/"${task}"/res.json >/dev/null 2>&1; then
        continue
    fi
    perf_file="$(latest_perf_for "$task")"
    printf "%-28s %-4s %-10s %-7s %-9s %-4s\n" \
        "$task" "$(gpu_of_task "$task")" "$(get_sr "$log")" \
        "$(get_perf_count "$perf_file")" "$(get_last_total_ms "$perf_file")" \
        "$(check_error "$log")"
done

printf "\neval_policy procs: %s\n" "$(pgrep -fc 'script/eval_policy.py' 2>/dev/null)"
