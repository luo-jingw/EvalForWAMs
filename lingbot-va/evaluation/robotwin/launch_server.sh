START_PORT=${START_PORT:-29056}
MASTER_PORT=${MASTER_PORT:-29061}

save_root='visualization/'
mkdir -p $save_root

# Optional perf logging. Triggered by env vars PERF_LOG_DIR and PERF_TASK_NAME.
perf_args=""
if [ -n "$PERF_LOG_DIR" ]; then
    mkdir -p "$PERF_LOG_DIR"
    perf_args="--perf_log_dir $PERF_LOG_DIR"
    if [ -n "$PERF_TASK_NAME" ]; then
        perf_args="$perf_args --perf_task_name $PERF_TASK_NAME"
    fi
fi

python -m torch.distributed.run \
    --nproc_per_node 1 \
    --master_port $MASTER_PORT \
    wan_va/wan_va_server.py \
    --config-name robotwin \
    --port $START_PORT \
    --save_root $save_root \
    $perf_args


