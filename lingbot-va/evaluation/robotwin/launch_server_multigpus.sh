START_PORT=${START_PORT:-29556}
MASTER_PORT=${MASTER_PORT:-29661}
LOG_DIR='./logs'
mkdir -p $LOG_DIR

save_root='./visualization/'
mkdir -p $save_root

batch_time=$(date +%Y%m%d_%H%M%S)

# Optional perf logging via env vars.
# PERF_LOG_DIR: shared directory for JSONL.
# PERF_TASK_NAMES: colon-separated list of task names, one per rank (0..7).
#                  If unset but PERF_TASK_NAME is set, it is reused for all ranks.
if [ -n "$PERF_LOG_DIR" ]; then
    mkdir -p "$PERF_LOG_DIR"
fi
IFS=':' read -r -a perf_task_names_arr <<< "${PERF_TASK_NAMES:-}"


for i in {0..7}; do
    CURRENT_PORT=$((START_PORT + i))
    CURRENT_MASTER_PORT=$((MASTER_PORT + i))

    LOG_FILE="${LOG_DIR}/server_${i}_${batch_time}.log"
    echo "[Task ${j}] GPU: ${i} | PORT: ${CURRENT_PORT} | MASTER_PORT: ${CURRENT_MASTER_PORT} | Log: ${LOG_FILE}"

    perf_args=""
    if [ -n "$PERF_LOG_DIR" ]; then
        perf_args="--perf_log_dir $PERF_LOG_DIR"
        rank_task_name="${perf_task_names_arr[$i]:-$PERF_TASK_NAME}"
        if [ -n "$rank_task_name" ]; then
            perf_args="$perf_args --perf_task_name $rank_task_name"
        fi
    fi

    CUDA_VISIBLE_DEVICES=$i  \
    nohup python -m torch.distributed.run \
        --nproc_per_node 1 \
        --master_port $CURRENT_MASTER_PORT \
        wan_va/wan_va_server.py \
        --config-name robotwin \
        --save_root $save_root \
        --port $CURRENT_PORT  \
        $perf_args > $LOG_FILE 2>&1 &
    sleep 2;
done

echo "All 8 instances have been launched in the background."
wait
