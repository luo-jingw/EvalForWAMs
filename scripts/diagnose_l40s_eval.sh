#!/usr/bin/env bash
# L40s server eval failure diagnostic.
# Single-worker smoke run with unbuffered client stdout so we can see
# the actual sapien / RoboTwin failure that gets swallowed by
# subprocess stdout buffering in the normal pool path.
#
# Runs 6 stages, each in its own log so the report is partitionable:
#   1. env baseline      (GPU / vulkan / sapien / python deps)
#   2. sapien render step (Scene + camera take_picture, not just ctor)
#   3. RoboTwin task env reset (no policy server, isolates render path)
#   4. start lingbot-va server (GPU 0, port 29056) and wait for port
#   5. unbuffered client + sapien step (the real failure mode)
#   6. kernel logs after run (dmesg / journalctl segfault scan)
#
# Output goes under /tmp/l40s_diag_<ts>/; the final block prints a
# tarball path -- just `cat` or attach the contents.
#
# Usage:  bash scripts/diagnose_l40s_eval.sh
# Env:    REPO_ROOT (auto), CONDA_BASE (auto), GPU (default 0)
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null)}"
GPU="${GPU:-0}"

if [[ -z "$CONDA_BASE" ]]; then
    echo "FATAL: cannot find conda. Set CONDA_BASE=." >&2
    exit 1
fi

TS=$(date +%Y%m%d_%H%M%S)
OUT="/tmp/l40s_diag_${TS}"
mkdir -p "$OUT"
echo "=== output dir: $OUT ==="

source "$CONDA_BASE/etc/profile.d/conda.sh"

# ---------------------------------------------------------------------------
# Stage 1: environment baseline
# ---------------------------------------------------------------------------
{
    echo "### date";        date -u
    echo "### host";        hostname
    echo "### kernel";      uname -a
    echo "### nvidia-smi -L";    nvidia-smi -L
    echo "### nvidia driver";    nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1
    echo "### DISPLAY";          echo "DISPLAY=$DISPLAY"
    echo "### /dev/dri";         ls -la /dev/dri/ 2>/dev/null
    echo "### vulkan device count";
    vulkaninfo --summary 2>/dev/null | grep -c "deviceName" || true
    echo "### vulkan device names";
    vulkaninfo --summary 2>/dev/null | grep -E "deviceName|driverInfo" | head -20

    echo
    echo "### lingbot-jw python deps"
    conda activate lingbot-jw
    python - <<'PY'
import sys
print("python:", sys.version.split()[0])
for m in ["torch", "diffusers", "transformers", "websockets", "qwan_extension"]:
    try:
        x = __import__(m)
        v = getattr(x, "__version__", "?")
        print(f"  {m}: {v}")
    except Exception as e:
        print(f"  {m}: IMPORT FAIL -> {e!r}")
import torch
print("torch.cuda.is_available:", torch.cuda.is_available())
print("torch.version.cuda:", torch.version.cuda)
print("torch.cuda.device_count:", torch.cuda.device_count())
PY

    echo
    echo "### RoboTwin-jw python deps"
    conda activate RoboTwin-jw
    python - <<'PY'
import sys
print("python:", sys.version.split()[0])
for m in ["sapien", "torch", "websockets", "warp"]:
    try:
        x = __import__(m)
        v = getattr(x, "__version__", "?")
        print(f"  {m}: {v}")
    except Exception as e:
        print(f"  {m}: IMPORT FAIL -> {e!r}")
PY
} > "$OUT/01_env.log" 2>&1
echo "[1/6] env baseline -> 01_env.log"

# ---------------------------------------------------------------------------
# Stage 2: sapien actual render step (NOT just Scene ctor)
# ---------------------------------------------------------------------------
{
    conda activate RoboTwin-jw
    python -u - <<PY 2>&1
import sys, traceback
try:
    import numpy as np
    import sapien
    print("sapien version:", sapien.__version__)
    print("creating Scene ..."); sc = sapien.Scene(); print("OK")
    print("setting timestep ..."); sc.set_timestep(1.0 / 240.0); print("OK")
    print("adding ground ..."); sc.add_ground(0.0); print("OK")
    print("adding camera ...");
    cam = sc.add_camera(name="cam", width=224, height=224, fovy=1.4, near=0.1, far=10.0)
    cam.set_local_pose(sapien.Pose([1.0, 0.0, 1.0], [0.7, 0, 0.7, 0]))
    print("OK")
    print("update_render ..."); sc.update_render(); print("OK")
    print("take_picture (THIS is where vulkan actually runs) ...");
    cam.take_picture()
    print("OK")
    print("get_picture('Color') ..."); img = cam.get_picture("Color"); print("shape:", img.shape, "dtype:", img.dtype)
    print()
    print("ALL_GOOD sapien_full_render")
except Exception as e:
    print("FAILED:", repr(e))
    traceback.print_exc()
    sys.exit(2)
PY
    echo "exit_code: $?"
} > "$OUT/02_sapien_render.log" 2>&1
echo "[2/6] sapien render step -> 02_sapien_render.log"

# ---------------------------------------------------------------------------
# Stage 3: RoboTwin task env reset (no policy server, no model)
# ---------------------------------------------------------------------------
{
    conda activate RoboTwin-jw
    export ROBOTWIN_ROOT="$REPO_ROOT/RoboTwin"
    cd "$ROBOTWIN_ROOT" || { echo "RoboTwin dir missing"; exit 1; }
    python -u - <<'PY' 2>&1
import sys, traceback
sys.path.insert(0, ".")
try:
    print("import envs ..."); from envs import CONFIGS_REGISTRY; print("OK", len(CONFIGS_REGISTRY), "tasks registered")
    if "adjust_bottle" not in CONFIGS_REGISTRY:
        print("FAIL adjust_bottle not in registry"); sys.exit(3)
    print("instantiate adjust_bottle ...")
    cfg = CONFIGS_REGISTRY["adjust_bottle"]
    env = cfg(seed=0)
    print("OK env type:", type(env).__name__)
    print("env.reset() ..."); env.reset(); print("OK")
    print("env.step([0]*N) ...");
    # action space differs by task; try a no-op via env.action_space
    try:
        import numpy as np
        a = env.action_space.sample() * 0.0 if hasattr(env, "action_space") else None
        if a is not None:
            env.step(a); print("OK")
        else:
            print("(env has no action_space attr, skip step)")
    except Exception as e:
        print("step failed (non-fatal):", repr(e))
    print()
    print("ALL_GOOD robotwin_env_reset")
except Exception as e:
    print("FAILED:", repr(e))
    traceback.print_exc()
    sys.exit(3)
PY
    echo "exit_code: $?"
} > "$OUT/03_robotwin_env.log" 2>&1
echo "[3/6] robotwin env reset -> 03_robotwin_env.log"

# ---------------------------------------------------------------------------
# Stage 4: start lingbot-va server (bf16, GPU 0, port 29056)
# ---------------------------------------------------------------------------
mkdir -p "$OUT/results/visualization" "$OUT/results/perf"
SERVER_LOG="$OUT/04_server.log"
(
    conda activate lingbot-jw
    cd "$REPO_ROOT"
    CUDA_VISIBLE_DEVICES=$GPU \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -u -m torch.distributed.run \
        --nproc_per_node 1 --master_port 29661 \
        --module ptqeval.wam.lingbot_va.server \
        --config-name robotwin --port 29056 \
        --save_root "$OUT/results/visualization" \
        --perf_log_dir "$OUT/results/perf" \
        --perf_task_name adjust_bottle \
        --model_path "$REPO_ROOT/models/lingbot-va-posttrain-robotwin"
) > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "[4/6] server pid=$SERVER_PID, waiting for port 29056 (up to 120 s) ..."

# Wait up to 120 s for "listening on" line OR pid death
for i in $(seq 1 60); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "    server died early (pid gone) after ${i}*2 s"
        break
    fi
    if grep -q "listening on 0.0.0.0:29056" "$SERVER_LOG" 2>/dev/null; then
        echo "    server listening after ${i}*2 s"
        break
    fi
    sleep 2
done

# ---------------------------------------------------------------------------
# Stage 5: unbuffered client + ACTUAL sapien step (the real failure mode)
# ---------------------------------------------------------------------------
{
    conda activate RoboTwin-jw
    export ROBOTWIN_ROOT="$REPO_ROOT/RoboTwin"
    export LD_LIBRARY_PATH="/usr/lib64:/usr/lib:${LD_LIBRARY_PATH:-}"
    cd "$REPO_ROOT"

    echo "### running client with python -u (unbuffered)"
    echo "### sapien runs vulkan on whatever device 0 is (NOT CUDA_VISIBLE_DEVICES)"

    CUDA_VISIBLE_DEVICES=$GPU \
    PYTHONWARNINGS=ignore::UserWarning \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    timeout --signal=KILL 240 python -u -m ptqeval.wam.lingbot_va.eval_client \
        --config "$ROBOTWIN_ROOT/policy/ACT/deploy_policy.yml" \
        --overrides \
        --task_name adjust_bottle --task_config demo_randomized \
        --train_config_name 0 --model_name 0 --ckpt_setting 0 \
        --seed 0 --policy_name ACT \
        --save_root "$OUT/results/visualization" \
        --video_guidance_scale 5 --action_guidance_scale 1 \
        --test_num 1 --port 29056
    rc=$?
    echo
    echo "client exit_code: $rc"
    case $rc in
        0)   echo "(0 = success)";;
        1)   echo "(1 = Python uncaught exception; traceback above)";;
        124) echo "(124 = timeout 240 s, client never finished)";;
        137) echo "(137 = SIGKILL by timeout)";;
        139) echo "(139 = SIGSEGV, segfault -- vulkan/sapien crash)";;
        143) echo "(143 = SIGTERM external)";;
        *)   echo "(rc=$rc unknown)";;
    esac
} > "$OUT/05_client_unbuffered.log" 2>&1
echo "[5/6] client unbuffered -> 05_client_unbuffered.log"

# Tear down server
kill -TERM "$SERVER_PID" 2>/dev/null
sleep 3
pkill -9 -f "lingbot_va.server" 2>/dev/null
pkill -9 -f "torch.distributed.run" 2>/dev/null

# Capture last bit of server log (so we know server state when client died)
echo "### server log tail (after client teardown):" >> "$OUT/04_server.log"
echo

# ---------------------------------------------------------------------------
# Stage 6: kernel/system logs since run started
# ---------------------------------------------------------------------------
{
    echo "### dmesg last 5 min"
    sudo -n dmesg -T --since "5 min ago" 2>/dev/null | tail -100 \
        || dmesg -T 2>/dev/null | tail -100 \
        || echo "(no dmesg access)"
    echo
    echo "### segfault / oom / nvidia / vulkan greps"
    (sudo -n dmesg -T --since "10 min ago" 2>/dev/null \
        || dmesg -T 2>/dev/null) \
        | grep -iE "segfault|oom|nvidia|vulkan|sapien|python" | tail -50
    echo
    echo "### journalctl errs (if accessible)"
    journalctl --since "10 min ago" -p err --no-pager 2>/dev/null | tail -50 \
        || echo "(no journal access)"
    echo
    echo "### residual lingbot processes (should be empty)"
    ps -ef | grep -E "lingbot_va|torch\.distributed" | grep -v grep
    echo
    echo "### GPU memory now"
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
} > "$OUT/06_kernel_logs.log" 2>&1
echo "[6/6] kernel logs -> 06_kernel_logs.log"

# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------
TARBALL="/tmp/l40s_diag_${TS}.tar.gz"
tar -czf "$TARBALL" -C "$(dirname "$OUT")" "$(basename "$OUT")"
echo
echo "================================================================"
echo "ALL DONE."
echo "Output dir: $OUT"
echo "Tarball:    $TARBALL"
echo
echo "Quick summary:"
echo "----------------------------------------------------------------"
for f in 01_env 02_sapien_render 03_robotwin_env 05_client_unbuffered; do
    echo "[$f]"
    grep -E "ALL_GOOD|FAILED|exit_code|client exit_code|FAIL" "$OUT/${f}.log" 2>/dev/null | head -10
    echo
done
echo "Server log first 'listening' line + last 5 lines:"
grep -m1 "listening" "$OUT/04_server.log" 2>/dev/null
tail -5 "$OUT/04_server.log" 2>/dev/null
echo "----------------------------------------------------------------"
echo
echo "Paste the contents (or attach the tarball)."
