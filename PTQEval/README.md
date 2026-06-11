# PTQEval

PTQ evaluation harness for World-Action Models (WAMs) on RoboTwin.
Hosts WAM-agnostic eval pipeline, per-method quantization code, and
per-WAM server / client forks. Third-party subrepos under
`/home/arash/EvalForWAMs/{lingbot-va, RoboTwin, ViDiT-Q}/` remain
upstream-equivalent.

Authoritative design doc: `/home/arash/EvalForWAMs/plan.txt`
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

## Quick start

```bash
# One-time install (run in each conda env that imports ptqeval)
conda activate lingbot-jw
pip install -e /home/arash/EvalForWAMs/PTQEval/

conda activate RoboTwin-jw
pip install -e /home/arash/EvalForWAMs/PTQEval/

# Build CUDA kernel (qwan_extension._C). sm_86; needs torch + nvcc.
conda activate lingbot-jw
pip install --no-build-isolation -e \
    /home/arash/EvalForWAMs/PTQEval/ptqeval/wam/lingbot_va/method/viditq/kernel/

# Smoke (bf16 baseline; 1 task, 1 episode, GPU 4)
SAVE_ROOT=/home/arash/EvalForWAMs/results/smoke_bf16 \
ROBOTWIN_ROOT=/home/arash/EvalForWAMs/RoboTwin \
  python -m ptqeval.eval.run_eval \
    --mode smoke --task_name adjust_bottle --test_num 1 --gpu_id 4

# Pool (15 tasks, 25 episodes, all usable GPUs)
SAVE_ROOT=/home/arash/EvalForWAMs/results/bf16 \
ROBOTWIN_ROOT=/home/arash/EvalForWAMs/RoboTwin \
  python -m ptqeval.eval.run_eval \
    --mode pool --min_free_mb 40000

# Quant variant (viditq W8A8 example)
SAVE_ROOT=/home/arash/EvalForWAMs/results/viditq_w8a8_kernel \
ROBOTWIN_ROOT=/home/arash/EvalForWAMs/RoboTwin \
VARIANT=viditq \
VARIANT_ARGS=/home/arash/EvalForWAMs/PTQEval/ptqeval/wam/lingbot_va/method/viditq/configs/runtime_args_w8a8.yaml \
  python -m ptqeval.eval.run_eval \
    --mode pool

# Calibration data collection (Phase 31; 50 task x 5 ep on bf16)
TASK_LIST_NAME=CALIB_TASKS_ALL \
CALIBRATE_OUT=/home/arash/EvalForWAMs/results/calib_data/calib_data.pth \
SAVE_ROOT=/home/arash/EvalForWAMs/results/calib_capture \
  python -m ptqeval.eval.run_eval --mode pool --test_num 5

# Aggregate
python -m ptqeval.eval.aggregator \
  --save_root /home/arash/EvalForWAMs/results/bf16 \
  --perf_log_dir /home/arash/EvalForWAMs/results/bf16/perf \
  --out_dir /home/arash/EvalForWAMs/results/bf16/summary

# Live dashboard
SAVE_ROOT=/home/arash/EvalForWAMs/results/bf16 \
  watch -n 1 bash /home/arash/EvalForWAMs/PTQEval/ptqeval/eval/monitor.sh
```

## Env vars consumed by run_eval.py

| Var | Default | Purpose |
|---|---|---|
| `WAM_NAME` | `lingbot_va` | Picks `ptqeval.wam.<WAM_NAME>.*` |
| `WAM_MODEL_PATH` | `/home/arash/EvalForWAMs/models/lingbot-va-posttrain-robotwin` | FP checkpoint dir |
| `ROBOTWIN_ROOT` | `/home/arash/EvalForWAMs/RoboTwin` | RoboTwin sim root |
| `SAVE_ROOT` | `results/<variant_tag>` | Output root |
| `PERF_LOG_DIR` | `${SAVE_ROOT}/perf` | Per-call perf JSONL dir |
| `VARIANT` | unset | Quant variant; resolves to `ptqeval.wam.<WAM>.method.<VARIANT>.loader` |
| `VARIANT_ARGS` | unset | YAML with `layer_config` + `int_weights_ckpt` paths |
| `SERVER_ENV` | `lingbot-jw` | Conda env for server |
| `CLIENT_ENV` | `RoboTwin-jw` | Conda env for client |

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
