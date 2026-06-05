#!/bin/bash
# Read-only status dashboard. Designed for: watch -n 1 monitor.sh
# Shows per-GPU VRAM and per-task progress (step, SR, perf-call count) for
# the most recently active log dir under SAVE_ROOT/logs/.
set -u

SAVE_ROOT="${SAVE_ROOT:-/home/arash/EvalForWAMs/results/bf16}"
PERF_LOG_DIR="${PERF_LOG_DIR:-${SAVE_ROOT}/perf}"

strip_ansi() {
    sed 's/\x1b\[[0-9;]*m//g'
}

# Latest step from a client log: "step: <i> / <total>".
get_step() {
    local log="$1"
    [ -f "$log" ] || { echo "-"; return; }
    local s
    s="$(tail -c 8192 "$log" 2>/dev/null | strip_ansi | grep -oE 'step: [0-9]+ / [0-9]+' | tail -1 | sed 's/step: //')"
    echo "${s:--}"
}

# Latest "Success rate: N/M".
get_sr() {
    local log="$1"
    [ -f "$log" ] || { echo "-"; return; }
    local s
    s="$(tail -c 8192 "$log" 2>/dev/null | strip_ansi | grep -oE 'Success rate: [0-9]+/[0-9]+' | tail -1 | sed 's/Success rate: //')"
    echo "${s:--}"
}

# Latest perf jsonl for a given task (newest mtime).
latest_perf_for() {
    local task="$1"
    ls -t "$PERF_LOG_DIR/${task}_rank"*".jsonl" 2>/dev/null | head -1
}

get_perf_count() {
    local f="$1"
    [ -f "$f" ] || { echo "0"; return; }
    wc -l < "$f" | tr -d ' '
}

get_last_total_ms() {
    local f="$1"
    [ -f "$f" ] || { echo "-"; return; }
    local t
    t="$(tail -1 "$f" 2>/dev/null | grep -oE '"total_ms": [0-9.]+' | head -1 | awk '{printf "%.0f", $2}')"
    echo "${t:--}"
}

# Flag only fatal errors. websockets handshake tracebacks (from TCP port
# probing in wait_for_port) are benign noise and ignored.
check_error() {
    local log="$1"
    [ -f "$log" ] || { echo ""; return; }
    if grep -q "OutOfMemoryError" "$log" 2>/dev/null; then echo "OOM"; return; fi
    echo ""
}

printf "=== %s   SAVE_ROOT=%s ===\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$SAVE_ROOT"

# ---- GPU table ----
printf "\nGPU  Used/Total(MB)   Free(MB)   Util%%\n"
nvidia-smi --query-gpu=index,memory.used,memory.total,memory.free,utilization.gpu \
    --format=csv,noheader,nounits 2>/dev/null | \
    awk -F', *' '{printf "%-3s  %6s / %-6s   %-8s   %s\n", $1, $2, $3, $4, $5}'

# ---- pick latest log dir ----
LATEST_DIR="$(ls -td "${SAVE_ROOT}"/logs/*/ 2>/dev/null | head -1)"
if [ -z "$LATEST_DIR" ]; then
    printf "\n(no log dir under %s/logs)\n" "$SAVE_ROOT"
    exit 0
fi
printf "\nlog_dir: %s\n" "$LATEST_DIR"

# ---- task table ----
printf "\n%-24s %-3s %-9s %-5s %-6s %-12s %-9s %-6s %-9s %-4s\n" \
    "task" "GPU" "srv_pid" "alive" "client" "step" "SR" "calls" "last_ms" "err"
printf -- "------------------------------------------------------------------------------------------------\n"
for client_log in "$LATEST_DIR"client_*.log; do
    [ -f "$client_log" ] || continue
    base="$(basename "$client_log" .log)"        # client_<i>_<task>
    rest="${base#client_}"                       # <i>_<task>
    gpu="${rest%%_*}"
    task="${rest#*_}"

    server_log="${LATEST_DIR}server_${gpu}_${task}.log"
    server_pidfile="${server_log}.pid"
    srv_pid="-"; alive="-"
    if [ -f "$server_pidfile" ]; then
        srv_pid="$(cat "$server_pidfile" 2>/dev/null)"
        if [ -n "$srv_pid" ] && kill -0 "$srv_pid" 2>/dev/null; then alive="Y"; else alive="N"; fi
    fi
    client_state="RUN"
    grep -q "Data has been saved to" "$client_log" 2>/dev/null && client_state="DONE"

    perf_file="$(latest_perf_for "$task")"
    err="$(check_error "$client_log")"
    [ -z "$err" ] && err="$(check_error "$server_log")"
    printf "%-24s %-3s %-9s %-5s %-6s %-12s %-9s %-6s %-9s %-4s\n" \
        "$task" "$gpu" "$srv_pid" "$alive" "$client_state" \
        "$(get_step "$client_log")" "$(get_sr "$client_log")" \
        "$(get_perf_count "$perf_file")" "$(get_last_total_ms "$perf_file")" "$err"
done

# ---- proc count ----
n_proc=$(ps -e -o comm= 2>/dev/null | grep -c -E "^python$" || true)
printf "\npython procs alive: %s\n" "$n_proc"
