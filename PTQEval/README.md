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

# Editable install of our research package
pip install -e PTQEval/

# fast_hadamard_transform: required for the QuaRoT runtime CUDA path
# (Python butterfly fallback works but is ~800 ms/call slower per Linear).
# pip's sdist is missing csrc/, so build from GitHub source:
git clone --depth 1 https://github.com/Dao-AILab/fast-hadamard-transform.git
cd fast-hadamard-transform
FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.6" \
    pip install --no-build-isolation .
cd ..
```

Replace `TORCH_CUDA_ARCH_LIST="8.6"` with `"8.9"` on L40 / L40S, `"9.0"`
on H100, etc.

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

# Editable install of our research package (so eval_client can resolve
# `ptqeval.wam.lingbot_va.eval_client`)
pip install -e PTQEval/

# (one-time) download RoboTwin assets per upstream INSTALLATION.md
```

### 3. CUDA kernel build (`qwan_extension._C`)

Required only for `viditq` quant variants (W8A8 / W4A8). Skip for bf16
baseline runs. Builds inside the `lingbot-jw` env.

```bash
conda activate lingbot-jw

# setup.py hardcodes -gencode=arch=compute_86,code=sm_86. To target a
# different SM, either edit setup.py or pass TORCH_CUDA_ARCH_LIST and
# strip the hardcoded -gencode flag.
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

Rebuilding on a different SM (e.g. moving an existing tree from
A6000 sm_86 to L40S sm_89): edit `setup.py:51` to
`-gencode=arch=compute_89,code=sm_89`, then `pip install -e .` again to
re-emit `_C.cpython-*.so`. A stale `.so` from a different SM imports
fine but every kernel launch returns
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
