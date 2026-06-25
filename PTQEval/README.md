# PTQEval

PTQ evaluation harness for World-Action Models (WAMs) on RoboTwin.
Hosts WAM-agnostic eval pipeline, per-method quantization code, and
per-WAM server / client forks. Third-party subrepos under
`{lingbot-va, RoboTwin, ViDiT-Q}/` remain
upstream-equivalent.

Authoritative design doc: `plan.txt`
(Section 24 = layout, Section 25 = interfaces, Section 26 = migration phases).

## Layout

```
PTQEval/
  pyproject.toml                    pip install -e PTQEval/
  ptqeval/
    eval/                           WAM/method/benchmark-agnostic
      perf_probe.py                 PerfProbe class (CUDA Event + peak mem -> JSONL)
      aggregator.py                 perf JSONL + RoboTwin res.json -> summary.{csv,json}
      run_eval.py                   orchestrator (smoke / single / pool)
      monitor.sh                    watch-friendly dashboard
    wam/<wam>/                      one directory per WAM
      __init__.py                   sets LINGBOT_VA_PATH on sys.path (for lingbot_va)
      server.py                     FORK of upstream server, probe + variant dispatch added
      eval_client.py                FORK of upstream RoboTwin client
      tasks.py                      SELECTED_15_TASKS + EVAL_STEP_LIMIT
      launchers/                    standalone bash launchers (server / client x single / multi)
      method/<method>/              one directory per (WAM, method) pair
        ptq.py                      offline FP -> int weights + scales
        loader.py                   load_quant_model(...) -> nn.Module (server contract)
        block.py                    block-level wrapper subclass
        check_block.py              forward correctness vs FP reference
        configs/                    {w8a8.yaml, w4a8.yaml, runtime_args_*.yaml}
        kernel/                     method-only CUDA extension
          setup.py
          csrc/                     pybind + .cu + infra/ (verbatim ViDiT-Q headers)
          qwan_extension/           Python wrappers + bench + nn.Module
```

## Model download

The LingBot-VA bf16 checkpoint and the RoboTwin simulator assets must be
fetched once. Expected on-disk layout when finished:

```
models/lingbot-va-posttrain-robotwin/  (~25 GB, bf16)
├── transformer/              5.09 B params (30 WanTransformerBlock, dim=3072)
├── text_encoder/             UMT5-XXL (~11 GB)
├── vae/                      AutoencoderKLWan
├── tokenizer/
├── assets/
└── README.md
RoboTwin/aaa_assets/          RoboTwin 2.0 sim assets (URDF / SDF / meshes)
```

### LingBot-VA ckpt

```bash
# inside lingbot-jw env (uses huggingface_hub)
huggingface-cli download robbyant/lingbot-va-posttrain-robotwin \
    --local-dir models/lingbot-va-posttrain-robotwin

# attn_mode must be "torch" (or "flashattn"). Upstream ships "flex"
# which only works for training; inference crashes on it. The current
# release already ships "torch", verify with:
python -c "
import json
c = json.load(open('models/lingbot-va-posttrain-robotwin/transformer/config.json'))
assert c['attn_mode'] in ('torch','flashattn'), c['attn_mode']
print('attn_mode =', c['attn_mode'])
"
```

Available checkpoints (HF / ModelScope, pick the one you need):

| Variant | HF repo | Notes |
|---|---|---|
| **lingbot-va-posttrain-robotwin** | `robbyant/lingbot-va-posttrain-robotwin` | Used in this project; post-trained on RoboTwin |
| lingbot-va-base | `robbyant/lingbot-va-base` | Pre-trained backbone (no task post-train) |
| lingbot-va-posttrain-libero-long | `robbyant/lingbot-va-posttrain-libero-long` | LIBERO-LONG post-train |

### RoboTwin sim assets

```bash
# inside RoboTwin-jw env, follow upstream INSTALLATION.md, or:
cd RoboTwin && bash script/_download_assets.sh && cd ..
# Pulls embodiment URDFs + object meshes for the 50 tasks.
```

### Calibration data (optional, for skipping bf16 calib eval)

We publish derived calibration data so a downstream user can reproduce
quant variants without re-running the bf16 RoboTwin rollouts:

```bash
# (read-only — no token needed)
huggingface-cli download JingwuLuo/LingBot-VA_RoboTwin_clibration_data \
    --repo-type dataset --local-dir results/calib_capture
```

Contents:
- `calib_data.pth` (1.8 MB) — per-channel input absmax over 180 target
  Linears (30 block × 6), derived from 50 task × 5 ep bf16 rollouts. Feed
  directly into `ptq.py`.
- raw obs / latent / action chunks (~12 GB) — for replaying through a
  different WAM transformer to derive that WAM's own absmax.
- `configs/*.yaml` — quant configs used to produce each variant.

## Setup

Two separate conda envs are required: one for the WAM server (`lingbot-jw`),
one for the RoboTwin client (`RoboTwin-jw`). They MUST be separate because
RoboTwin pins `torch==2.4.1 + sapien==3.0.0b1` while LingBot-VA needs
`torch==2.9.0 + diffusers==0.36.0`. Server and client communicate over a
local websocket.

### 1. `lingbot-jw` (server, PTQ, kernel build)

```bash
conda create -n lingbot-jw python=3.10.16 -y
conda activate lingbot-jw

# Torch first (CUDA 12.6 wheels)
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
    --index-url https://download.pytorch.org/whl/cu126

# LingBot-VA upstream Python deps
pip install -r lingbot-va/requirements.txt
pip install flash-attn --no-build-isolation

# Editable install of our research package (auto-pulls omegaconf +
# websockets + msgpack which neither lingbot-va/requirements.txt nor
# RoboTwin-jw provisioning covers)
pip install -e PTQEval/

# fast_hadamard_transform: required for the QuaRoT runtime CUDA path
# (Python butterfly fallback works but is ~800 ms/call slower per Linear).
# pip's sdist is missing csrc/, so build from GitHub source:
git clone --depth 1 https://github.com/Dao-AILab/fast-hadamard-transform.git
cd fast-hadamard-transform
FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.6;8.9" \
    pip install --no-build-isolation .
cd ..
```

`TORCH_CUDA_ARCH_LIST="8.6;8.9"` emits a fatbinary covering both A6000
(sm_86) and L40 / L40S (sm_89) in one wheel. For other GPUs append /
replace: `"9.0"` for H100, `"12.0"` for RTX 5090, etc.

`attn_mode` must be `"torch"` (or `"flashattn"`) in
`models/lingbot-va-posttrain-robotwin/transformer/config.json`. The
upstream default `"flex"` only works for training and crashes at
inference; the bundled checkpoint already ships with the correct value.

### 2. `RoboTwin-jw` (client, sapien sim)

```bash
conda create -n RoboTwin-jw python=3.10 -y
conda activate RoboTwin-jw

# Run RoboTwin's own installer for python deps + sapien + mplib patches +
# Curobo. Per upstream README ~20 minutes. Takes care of sed-patching
# sapien's urdf_loader.py and mplib's planner.py.
cd RoboTwin && bash script/_install.sh && cd ..

# pytorch3d (RoboTwin _install.sh handles this, but if it fails:)
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" \
    --no-build-isolation

# Editable install of our research package (resolves eval_client +
# auto-pulls websockets + msgpack which RoboTwin's _install.sh doesn't)
pip install -e PTQEval/

# (one-time) download RoboTwin assets per upstream INSTALLATION.md
```

### Per-node provisioning (multi-host clusters)

On clusters where `/home` (and thus `~/miniconda3`) is **node-local** —
not NFS-shared — every node needs its own full env setup. Conda envs do
NOT follow you across nodes; `pip install -e PTQEval/` and the kernel
build (next section) must both rerun per node.

`pyproject.toml` now declares the dependencies that earlier slipped
through (`omegaconf` for server YAML loading, `websockets` + `msgpack`
for the client transport), so a fresh `pip install -e PTQEval/` in each
env pulls them automatically. If you upgrade from an older checkout that
skipped these, just rerun `pip install -e PTQEval/` to backfill.

**Symptom of a missed step.** Every pool worker dies instantly at
startup and `results/<run>/logs/pool/server_*.log` (or `client_*.log`)
shows `ModuleNotFoundError`. This is distinct from CUDA-OOM, which
shows a `Killed` line and the worker survives just past server init.

Smoke-test before any long run — surfaces every missing dep in ~5 min
instead of after a multi-hour pool failure:

```bash
python -m ptqeval.eval.run_eval --mode smoke --variant viditq \
    --variant_args PTQEval/ptqeval/wam/lingbot_va/method/viditq/configs/runtime_args_w8a8_dynamic.yaml \
    --gpu_id 0 --save_root results/_smoke_w8a8
```

If a node still misbehaves after the above, diff package names against
a known-good node (do NOT bulk-install the exact-version diff — it
would churn the pinned torch / sapien stack):

```bash
# on the working node:
pip freeze > /tmp/ref_freeze.txt   # then copy to the new node

# on the new node, in the env being debugged:
comm -23 <(sed 's/[ =@].*//' /tmp/ref_freeze.txt | sort -u) \
         <(pip freeze | sed 's/[ =@].*//' | sort -u)
```

### 3. CUDA kernel build (`qwan_extension._C`)

Required only for `viditq` quant variants (W8A8 / W4A8). Skip for bf16
baseline runs. Builds inside the `lingbot-jw` env.

```bash
conda activate lingbot-jw

# setup.py defaults to a multi-arch fatbinary covering sm_86 (A6000) and
# sm_89 (L40 / L40S), so the same build works on both. For any other GPU
# pass TORCH_CUDA_ARCH_LIST (e.g. "9.0" for H100, "12.0" for RTX 5090);
# when set it overrides the default list.
pip install --no-build-isolation -e \
    PTQEval/ptqeval/wam/lingbot_va/method/viditq/kernel/
```

Build inputs (`csrc/`):
- `pybind.cpp` — Python bindings (12+ launchers).
- `act_quant_bf16.cu` — per-token act quant: dynamic (`with_sum`) +
  static (`with_sum_static`) variants.
- `w8a8/w8a8_gemm.cu` — verbatim ViDiT-Q upstream + `typename OutT`
  templating to support `bf16` output (upstream is `fp16`-only).
- `w4a8/w4a8_gemm.cu` — verbatim QServe upstream + same templating.
- `toy_mma_int8.cu` — bench/sanity launchers.
- `infra/` — verbatim ViDiT-Q headers (mma/cp_async/permuted_smem etc.).

Verify the build:

```bash
python -c "import qwan_extension._C; print('ok')"
python PTQEval/ptqeval/wam/lingbot_va/method/viditq/kernel/qwan_extension/check_part6.py
```

`check_part6.py` runs the 6-variant numerical correctness suite
(baseline / smooth / quarot / viditq / static / viditq_static); all
should pass within `tol=5e-2`.

Rebuilding when the host's GPU is neither sm_86 nor sm_89: set
`TORCH_CUDA_ARCH_LIST` to your target list (e.g. `"8.6;8.9;9.0"`) and
re-run `pip install -e .` to re-emit `_C.cpython-*.so`. A stale `.so`
without the host's SM imports fine but every kernel launch returns
`CUDA error: no kernel image is available for execution on the device`.

## Run

```bash
# Smoke (bf16 baseline; 1 task, 1 episode, GPU 4)
python -m ptqeval.eval.run_eval \
    --mode smoke --task_name adjust_bottle --gpu_id 4 \
    --save_root results/smoke_bf16

# Pool (15 tasks, 25 episodes, all usable GPUs)
python -m ptqeval.eval.run_eval \
    --mode pool --task_config demo_randomized --test_num 25 \
    --save_root results/bf16

# Quant variant (viditq W8A8 dynamic example)
python -m ptqeval.eval.run_eval \
    --mode pool --variant viditq \
    --variant_args PTQEval/ptqeval/wam/lingbot_va/method/viditq/configs/runtime_args_w8a8_dynamic.yaml \
    --task_config demo_randomized --test_num 25 \
    --save_root results/viditq_w8a8_dynamic

# Calibration data collection (50 task x 5 ep on bf16, for viditq static
# activation quant)
python -m ptqeval.wam.lingbot_va.method.viditq.collect_calib_videos \
    --save_root results/calib_capture

# Aggregate (auto-merges per-task op_profile.json when --profile_ops was on)
python -m ptqeval.eval.aggregator \
    --save_root results/bf16 \
    --perf_log_dir results/bf16/perf \
    --out_dir results/bf16/summary

# Live dashboard
SAVE_ROOT=results/bf16 \
    watch -n 1 bash PTQEval/ptqeval/eval/monitor.sh
```

## Eval configurations

Two eval scales coexist in `ptqeval/wam/lingbot_va/tasks.py`. Pick by
`--task_list_name`:

| Tier | Attribute | Tasks | Typical `--test_num` | Use for |
|---|---|---|---|---|
| **Production SR** | `SELECTED_15_TASKS` | 15 | 25 | Cross-variant SR comparison (cross_summary). 1 long + 4 medium-long + 10 short, ~3-4 h on 8 GPU |
| **Full bench / calibration** | `CALIB_TASKS_ALL` | 50 | 5 (calib) or 100 (full sweep) | bf16 rollout collection for static-act calib; or a comprehensive variant sweep |

Task config (RoboTwin scene randomization) — `--task_config`:

| Value | Behavior | Use for |
|---|---|---|
| `demo_clean` | Fixed background / lighting / table height | legacy baseline only |
| **`demo_randomized`** | Per-episode randomized background / lighting / table | **production default** — all variants in this repo's cross_summary use it |

### Generating quantized weights for a variant

Each viditq variant needs its own `int_weights.pth`. The pipeline is:

```
calib_capture/                # raw bf16 rollouts (obs + latents)
    └── derive_calib_ptq.py   # replay through bf16 transformer + hook abs-max
        └── calib_data.pth    # 180 Linear x per-channel absmax (1.8 MB, variant-agnostic)
            └── ptq.py        # apply smooth + quarot + per-channel quant
                └── int_weights.pth  (per-variant; 2.3 - 4.6 GB)
                    └── run_eval --variant viditq --variant_args runtime_args_<variant>.yaml
```

`calib_data.pth` is **variant-agnostic** (raw absmax) — derive once, reuse
for every variant. `int_weights.pth` is **per-variant** (encodes bit
allocation + smooth/quarot transforms baked into weights).

Prereq: `results/calib_capture/` exists (either pulled from HF — see
"Calibration data" above — or generated locally via
`collect_calib_videos.py`).

```bash
conda activate lingbot-jw
CFG=PTQEval/ptqeval/wam/lingbot_va/method/viditq/configs

# Step 1: derive calib_data.pth (~30-60 min on 4-8 GPU; one-shot, all variants share it)
python -m ptqeval.wam.lingbot_va.method.viditq.derive_calib_ptq \
    --videos_root results/calib_capture \
    --all --skip_ptq --gpus auto
# -> results/calib_data/calib_data.pth

# Step 2: build int_weights.pth per variant (~5 min each on 1 GPU)
for v in w8a8_dynamic w8a8_static w4a8_dynamic w4a8_mixed; do
    python -m ptqeval.wam.lingbot_va.method.viditq.ptq \
        --layer_config ${CFG}/${v}.yaml \
        --output results/viditq_${v}/calib/int_weights.pth
done

# Step 3: eval (uses runtime_args_<v>.yaml, which points to int_weights.pth above)
for v in w8a8_dynamic w4a8_mixed; do
    python -m ptqeval.eval.run_eval --mode pool --variant viditq \
        --variant_args ${CFG}/runtime_args_${v}.yaml \
        --task_config demo_randomized --test_num 25 \
        --save_root results/viditq_${v}
done
```

The `<variant>.yaml` vs `runtime_args_<variant>.yaml` split:
- `<variant>.yaml` (e.g. `w4a8_mixed.yaml`) — PTQ-time config:
  `bit_alloc` + `smooth_alpha` + `quarot` switches + which Linears to
  skip (cross-attn `attn2` stays FP). Consumed by `ptq.py`.
- `runtime_args_<variant>.yaml` — server-side pointer: `layer_config:`
  (path to the above) + `int_weights_ckpt:` (path to the step-2 output).
  Consumed by `run_eval --variant_args`.

Skipping step 1 + 2 entirely: if you only need to evaluate the
LingBot-VA variants we already published, pull the prebuilt
`int_weights.pth` from the HF dataset
(`JingwuLuo/LingBot-VA_RoboTwin_clibration_data`, see "Calibration data"
section above) and jump straight to step 3.

### Standard 15-task SR sweep (~3-4 h on 8 GPU)

```bash
for save in \
    bf16 \
    "viditq_w8a8_dynamic --variant viditq --variant_args PTQEval/ptqeval/wam/lingbot_va/method/viditq/configs/runtime_args_w8a8_dynamic.yaml" \
    "viditq_w4a8_dynamic --variant viditq --variant_args PTQEval/ptqeval/wam/lingbot_va/method/viditq/configs/runtime_args_w4a8_dynamic.yaml" \
    "viditq_w4a8_mixed   --variant viditq --variant_args PTQEval/ptqeval/wam/lingbot_va/method/viditq/configs/runtime_args_w4a8_mixed.yaml"
do
    tag=${save%% *}; rest=${save#* }; [ "$rest" = "$tag" ] && rest=""
    python -m ptqeval.eval.run_eval --mode pool \
        --task_list_name SELECTED_15_TASKS --task_config demo_randomized \
        --test_num 25 --save_root results/${tag} $rest
done
```

### Full 50-task sweep (~12-20 h on 8 GPU per variant)

```bash
python -m ptqeval.eval.run_eval --mode pool \
    --task_list_name CALIB_TASKS_ALL --task_config demo_randomized \
    --test_num 100 --save_root results/bf16_full
```

Use the same command with `--variant viditq --variant_args ...` for each
quant variant. CALIB_TASKS_ALL covers all 50 RoboTwin 2.0 tasks; with
`--test_num 100` it produces 5000 episodes per variant — the most
statistically robust SR estimate but ~5× longer than the 15-task sweep.

### Cross-summary (after aggregator runs per variant)

```bash
python -m ptqeval.eval.calc_cross_ckpt \
    --variant bf16=results/bf16/summary/summary.csv \
    --variant viditq_w8a8_dynamic=results/viditq_w8a8_dynamic/summary/summary.csv \
    --variant viditq_w4a8_dynamic=results/viditq_w4a8_dynamic/summary/summary.csv \
    --variant viditq_w4a8_mixed=results/viditq_w4a8_mixed/summary/summary.csv \
    --op_profile bf16=results/bf16/summary/op_profile.json \
    --op_profile viditq_w8a8_dynamic=results/viditq_w8a8_dynamic/summary/op_profile.json \
    --op_profile viditq_w4a8_dynamic=results/viditq_w4a8_dynamic/summary/op_profile.json \
    --op_profile viditq_w4a8_mixed=results/viditq_w4a8_mixed/summary/op_profile.json \
    --measured_flops results/measured_flops.json \
    --measured_kv_cache results/measured_kv_cache.json \
    --int_weights_ckpt results/viditq_w4a8_dynamic/calib/int_weights.pth \
    --out_dir results/cross_summary
```

Produces `cross_summary.{csv,json}` + `report.md` + 7 plots:
SR per task, total_ms+speedup, speedup per task, latency distribution,
memory breakdown (uses measured KV cache), op breakdown (measured
profiler kernel time), roofline (FlopCounterMode FLOPs + memcpy bytes).
The first `--variant` is treated as baseline.

### GPU selection

```bash
# Limit to specific GPU ids
python -m ptqeval.eval.run_eval --mode pool --gpus 0,2,5 ...

# Cap worker count (after min_free_mb filter)
python -m ptqeval.eval.run_eval --mode pool --max_gpus 4 ...
```

## CLI flags consumed by run_eval.py

`run_eval.py` is fully CLI-driven (no env vars). See `--help` for the
complete list; the most common ones:

| Flag | Default | Purpose |
|---|---|---|
| `--mode` | required | `smoke` / `single` / `pool` |
| `--save_root` | required | Eval output root |
| `--wam_name` | `lingbot_va` | Picks `ptqeval.wam.<wam_name>.*` |
| `--wam_model_path` | `models/lingbot-va-posttrain-robotwin` | FP ckpt dir |
| `--robotwin_root` | `RoboTwin` | RoboTwin sim root |
| `--variant` | unset | Quant variant; resolves to `ptqeval.wam.<wam>.method.<variant>.loader` |
| `--variant_args` | unset | YAML with `layer_config` + `int_weights_ckpt` paths |
| `--task_config` | `demo_clean` | RoboTwin task config (`demo_clean` / `demo_randomized`); production eval uses `demo_randomized` |
| `--task_list_name` | `SELECTED_15_TASKS` | Task list attribute in `tasks.py` |
| `--test_num` | smoke=1, else=25 | Episodes per task |
| `--server_env` / `--client_env` | `lingbot-jw` / `RoboTwin-jw` | Conda env names |
| `--min_free_mb` | 33000 | GPU usable when free memory >= this |
| `--gpus` | unset | Comma-separated GPU ids to consider (e.g. `0,2,5`) |
| `--max_gpus` | unset | Cap to at most N GPUs after filtering |
| `--profile_ops` | on | Wrap first N infer calls in torch.profiler -> op_profile.json |

## Add a new quantization method (per existing WAM)

```
PTQEval/ptqeval/wam/<wam>/method/<new_method>/
  __init__.py
  ptq.py            offline FP -> int weights state_dict
  loader.py         load_quant_model(wam_model_path, variant_args, device, dtype)
  block.py          subclass of upstream block; swap target Linears for kernel modules
  configs/          {w8a8.yaml or whatever bitwidth, runtime_args_*.yaml}
  kernel/           optional CUDA extension; standalone setup.py
```

Loader contract (Section 25.1):
```python
def load_quant_model(wam_model_path: str, variant_args: dict,
                     device: torch.device, dtype: torch.dtype) -> nn.Module: ...
```

Server resolves the loader via
`importlib.import_module(f"ptqeval.wam.{wam_name}.method.{method_name}.loader")`.
No server-side code change is needed when adding a method.

## Add a new WAM

```
PTQEval/ptqeval/wam/<new_wam>/
  __init__.py       sets <NEW_WAM>_PATH on sys.path (mirror lingbot_va pattern)
  server.py         FORK of new WAM's upstream server; insert probe stages + variant dispatch
  eval_client.py    FORK of corresponding upstream client
  tasks.py          SELECTED_15_TASKS + EVAL_STEP_LIMIT
  launchers/        bash launchers (optional; run_eval.py subsumes them)
  method/<m>/...    per (new_wam, method) pair; mirror existing pattern
```

Set `WAM_NAME=<new_wam>` (and updated `WAM_MODEL_PATH`) when invoking `run_eval.py`.

## Principles (see plan.txt Section 0)

- P1 Research-oriented; faithful method reproduction over convenience.
- P2 No private algorithm simplification; equivalent engineering means (C++
  templates, verbatim kernel transcription) are permitted.
- P3 Copy + rename + modify is the default extension pattern. Avoid
  modifying third-party repos in place; fork inside our tree.
- P4 First-principles, straight-through first. Abstract only when a second
  concrete instance demands it.
