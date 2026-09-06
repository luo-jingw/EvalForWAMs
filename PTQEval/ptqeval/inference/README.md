# LingBot-VA w4a4 Inference (Unitree G1 deployment)

T5-free, w4a4-quantized LingBot-VA inference for real-robot deployment.
Serves the same websocket protocol as `lingbot-va/wan_va/wan_va_server.py`,
so an existing `g1-client` connects unchanged.

Target: **Unitree G1 EDU** (Jetson Orin, aarch64, `sm_87`, ~16 GB unified
memory; ~2 GB reserved for motion control → **~14 GB for the model**).

Everything here (assemble model → build w4a4 weights → precompute text →
serve) reuses existing repo code; this package only wires it for deploy.

---

## 0. What runs where

- **Server** (this package) runs on the G1 onboard compute (the 16 GB
  Jetson). Loads the w4a4 transformer + VAE, injects precomputed text
  embeds (no T5 on device), serves actions over websocket.
- **Client** (`g1-client`, a separate repo) runs on the robot control side,
  captures cameras, sends obs, drives arms/grippers. Connects to the server
  (localhost or the onboard IP).

w4a4 changes nothing on the client: quantization is internal to the
transformer; the obs/action wire protocol is identical to bf16.

---

## 1. Get the code

```bash
# Main repo (vendors lingbot-va/ + PTQEval/ — the SERVER side). One clone.
git clone <EvalForWAMs-repo-url> EvalForWAMs
cd EvalForWAMs

# Client (separate repo, robot control side)
git clone <g1-client-repo-url> g1-client
```

RoboTwin/ and ViDiT-Q/ in the main repo are eval/research only — not needed
to deploy (see §4).

---

## 2. Server environment (`lingbot-va` + `ptqeval` + w4a4 kernel)

On the G1 Jetson, or any CUDA box for testing. Two paths below: x86 dev box
(Python 3.10) and Jetson / JetPack 5.1.1 (Python 3.8).

### x86 test box (Python 3.10)
```bash
conda create -n lingbot-va python=3.10 -y
conda activate lingbot-va
pip install -r lingbot-va/requirements.txt      # torch==2.9.0 pin is fine here
pip install -e PTQEval/
```

### Jetson / JetPack 5.1.1 (Python 3.8, CUDA 11.4, Orin sm_87)
The x86 pins (torch==2.9.0, transformers==4.55.2) do NOT install on JP5.1.1
(Python 3.8). Use the Jetson requirements set instead — lingbot-va's only
version-sensitive deps are UMT5 (transformers <=4.46) + AutoencoderKLWan
(diffusers 0.32-0.34); the transformer is wan_va's own code. flash_attn is
NOT needed (the deploy forward uses attn_mode="torch"; model.py imports it
optionally).
```bash
# (a) conda via Miniforge — Anaconda's default installer is x86-only, so on
#     aarch64 (Jetson) use the conda-forge Miniforge aarch64 build.
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
bash Miniforge3-Linux-aarch64.sh -b -p $HOME/miniforge3
source $HOME/miniforge3/etc/profile.d/conda.sh
conda init bash   # (re-open the shell afterwards, or keep sourcing as above)

# (b) env MUST be Python 3.8 — the NVIDIA Jetson torch wheel is cp38. Do NOT
#     use 3.10 (no NVIDIA CUDA-11.4 torch wheel for it). Do NOT install a
#     conda cudatoolkit — the wheel links the SYSTEM JetPack CUDA 11.4.
conda create -n lingbot-va python=3.8 -y
conda activate lingbot-va
#     torch wheel build prerequisites (system libs):
sudo apt-get install -y libopenblas-dev libopenmpi-dev

# (c) torch — CUDA build FIRST. CRITICAL: `pip install torch` pulls the
#     CPU-only aarch64 wheel (torch.version.cuda == None -> the w4a4 kernel
#     build later fails with "CUDA_HOME ... None"). You MUST install the
#     NVIDIA JetPack 5.1.x wheel (cp38, CUDA 11.4). NOTE the JP version dir:
#     torch 2.1 is under v512 (NOT v511 -- v511 only has torch 2.0; a v511+
#     2.1 URL 404s). Both work on JP5.1.1 (same L4T r35 / CUDA 11.4). Verified
#     live 2026-07:
#       torch 2.1.0 (recommended):
pip install --no-cache \
  "https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl"
#       torch 2.0.0 (NVIDIA's official JP5.1.1 pick, fallback):
#         https://developer.download.nvidia.com/compute/redist/jp/v511/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
#     (Convenience alternative: the jetson-ai-lab index, e.g.
#      pip install torch --index-url https://pypi.jetson-ai-lab.dev/jp5/cu114 )
#
#     VERIFY it is a CUDA build before continuing — cuda must NOT be None:
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
#     e.g. -> 2.1.0  11.4  True     (if cuda is None, you got the CPU wheel)
#
#     torchvision is NOT needed by the lingbot-va server (only unrelated
#     policy adapters import it). Skip it; build from source (v0.16.x,
#     matching torch 2.1) only if some import actually demands it.

# (d) the rest (diffusers 0.34 + transformers 4.46 + ...), py3.8-verified
pip install -r PTQEval/ptqeval/inference/requirements-jetson.txt

# (e) ptqeval (editable): adds omegaconf/websockets/msgpack + the package
pip install -e PTQEval/

# (f) BUILD the w4a4 CUDA kernel (qwan_extension._C). REQUIRED for w4a4 —
#     it is NOT pip-installable from PTQEval's pyproject; it is a per-host
#     nvcc build. Set the arch for the target GPU: G1 EDU / Orin = sm_87.
#
#     Point CUDA_HOME at the REAL toolkit root, not the /usr/local/cuda
#     symlink (on JetPack it often points to /etc/alternatives/cuda, which
#     may lack bin/nvcc -> torch reports CUDA_HOME=None). Find it via:
#         export CUDA_HOME=$(dirname $(dirname $(readlink -f $(which nvcc))))
#     (typically /usr/local/cuda-11.4). Both of these must exist:
#         ls $CUDA_HOME/bin/nvcc $CUDA_HOME/lib64/libcudart.so
#     and torch must resolve it (prints a path, not None):
#         python -c "from torch.utils.cpp_extension import CUDA_HOME; print(CUDA_HOME)"
#
#     Build via setup.py (avoids pip build isolation, which spins up an env
#     WITHOUT torch -> "metadata generation failed" on `from torch...import`).
#     Do NOT use sudo (it strips CUDA_HOME/PATH; the conda env is yours).
cd PTQEval/ptqeval/wam/lingbot_va/method/viditq/kernel
export CUDA_HOME=$(dirname $(dirname $(readlink -f $(which nvcc))))
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
CUDA_HOME=$CUDA_HOME CUDA_PATH=$CUDA_HOME TORCH_CUDA_ARCH_LIST="8.7" \
  python setup.py develop
# (equivalent via pip: `pip install -e . --no-build-isolation` with the same
#  CUDA_HOME/CUDA_PATH/TORCH_CUDA_ARCH_LIST env in front.)
cd -
#     (Default build without the env var targets sm_86/sm_89 = A6000/L40S
#     dev boxes. Set TORCH_CUDA_ARCH_LIST to your GPU: 8.6 A6000, 8.9 L40S,
#     8.7 Orin, 9.0 H100.)

# sanity: kernel + package import
python -c "import qwan_extension._C; import ptqeval.inference; print('server env OK')"
```

---

## 3. Client environment (`g1-client`)

g1-client is pure py3.8-compatible (typing.Optional/Dict/Tuple, no py3.9+
syntax) and its deps (numpy/opencv/websockets/msgpack) are already in the
server's `lingbot-va` env — so on the robot you can REUSE that env instead
of a second one; just add the two non-pip SDKs:

```bash
conda activate lingbot-va                       # reuse the server env (py3.8)
pip install -r g1-client/requirements.txt       # numpy opencv websockets msgpack (already present)
# Plus two SDKs NOT on pip (see g1-client/requirements.txt header):
#   * unitree_sdk2py  (github.com/unitreerobotics/unitree_sdk2_python; pulls cyclonedds)
#   * teleimager      (G1 image-server client, from xr_teleoperate)
# After installing unitree_sdk2py, re-check numpy wasn't bumped:
python -c "import numpy; print(numpy.__version__)"   # keep 1.24.4 (py3.8 ceiling)
```
(A separate py3.8 env works too; reusing just avoids duplicate installs.)

---

## 4. Do you need the RoboTwin environment? — NO (for deployment)

RoboTwin (the `RoboTwin-jw` env / sapien simulator) is used ONLY by the
evaluation + calibration pipeline (`ptqeval.eval.*`, `collect_calib_videos`,
`eval_client`). The deployment server (`server_ws`) and `g1-client` never
import RoboTwin — the server only reads `env_type` as a config string for
the latent layout. **Skip the RoboTwin env on the robot.** (w4a4 weights are
also data-free, so no calibration is needed on the device either — build
them once on a workstation, §5b.)

---

## 5. Deploy pipeline

Paths below assume repo root `EvalForWAMs/` and the red-bottle model at
`models/lingbot-pick-red-bottle-3k`.

### (a) Download the self-contained model (once — that's it)
The HF repo `JingwuLuo/lingbot-pick-red-bottle-3k` is self-contained:
```
transformer/  int_weights_w4a4.pth  vae/  tokenizer/  text_cond_cache/
```
so one download gives a complete, ready-to-serve model_path — no assemble,
no weight_prep, no precompute on the device:
```bash
# Use huggingface-cli (NOT `hf`): on the edge, transformers 4.46 pins an old
# huggingface_hub (~0.26) and the `hf` CLI only exists in >=0.34.
huggingface-cli download JingwuLuo/lingbot-pick-red-bottle-3k \
  --local-dir models/lingbot-pick-red-bottle-3k
# version-agnostic Python fallback if the CLI misbehaves:
#   python -c "from huggingface_hub import snapshot_download; \
#     snapshot_download('JingwuLuo/lingbot-pick-red-bottle-3k', \
#     local_dir='models/lingbot-pick-red-bottle-3k')"
```
The server loads `vae/` + `tokenizer/` at startup and injects the cached
`text_cond_cache/` embed for "pick the red bottle" -> **T5-free, no
text_encoder shipped**. (The 11 GB `text_encoder/` is only needed on a cache
MISS; with the fixed prompt it always hits. To support arbitrary/new prompts
on-device, also download a `text_encoder/` into this dir and set
`serve_residency: true`.)

> **Building from parts (other models / rebuild).** If you have only a raw
> transformer checkpoint: `convert_to_lingbot_va.py --base <full-model>` adds
> vae/text_encoder/tokenizer; `ptqeval.inference.weight_prep` regenerates
> `int_weights_w4a4.pth` (data-free); `ptqeval.inference.precompute_text
> --prompt "..."` builds `text_cond_cache/`.

### (b) Write the deploy config
Copy `PTQEval/ptqeval/inference/configs/deploy.yaml` and set ABSOLUTE paths.
Key fields (see the sample for the full annotated set):
```yaml
base_config: g1_server        # G1 layout: env_type=none, joint actions, g1 cams
model_path:        /abs/.../models/lingbot-pick-red-bottle-3k
int_weights_ckpt:  /abs/.../models/lingbot-pick-red-bottle-3k/int_weights_w4a4.pth
layer_config:      /abs/.../PTQEval/ptqeval/inference/configs/w4a4.yaml
text_cond_cache:   /abs/.../models/lingbot-pick-red-bottle-3k/text_cond_cache
serve_residency: false        # no text_encoder shipped; fixed prompt always
                              # hits the cache. Set true ONLY if you added a
                              # text_encoder/ for arbitrary-prompt fallback.
attn_window: 48               # KV-memory knob for the 14 GB budget (see §6)
device: cuda:0
```

### (c) Launch the server (onboard)
```bash
python -m ptqeval.inference.server_ws \
  --config /abs/path/to/deploy.yaml \
  --host 0.0.0.0 --port 29536
```
Server listens on `ws://<host>:29536`. It prints the model/quant/cache
load, then blocks serving.

### (d) Launch the client (robot control side)
```bash
# from g1-client repo root, in the (reused) lingbot-va env
python lingbot_va/main.py \
  --iface <robot-net-iface> \
  --server-host <server-ip-or-127.0.0.1> \
  --server-port 29536 \
  --prompt "pick the red bottle"
```
Use the SAME prompt string baked into `text_cond_cache/` → 100% cache hit,
T5 never loads.

---

## 6. Memory on the 14 GB G1

Approximate resident footprint (w4a4, T5-free):
- transformer (w4a4 + FP attn2/blocks.0): a few GB
- KV cache: **scales with `attn_window`** — the dominant knob
- VAE / text_encoder: kept on CPU (`enable_offload` inherited true from
  g1_server) → ~0 on GPU
- T5: **never resident** (precomputed cache; serial swap on a miss)

Tuning to fit 14 GB:
- **`attn_window`** is the main lever. Default here is **48** (down from 72).
  If you OOM, drop toward the g1 training window (30); if you have headroom
  and want more temporal context, raise it.
- Keep `serve_residency: true` so a cache miss offloads the transformer to
  CPU before loading T5 (T5 and transformer never co-resident → no additive
  +T5 spike that would OOM). `offload_target: disk` frees host RAM too if
  RAM is also tight.
- Measure the real peak before trusting a setting: wrap the engine with
  `ptqeval.inference.metrics.MeasuredEngine` and read `peak_alloc_mb()` /
  `last_infer_ms()` on a few inference chunks.

---

## Ports (default)

| role | port | note |
|---|---|---|
| deploy server (`server_ws --port`) | 29536 | matches `g1_server` config; client `--server-port` must equal it |

## Files in this package

- `config.py` — `InferenceConfig` (deploy params; `base_config` picks robot layout)
- `engine.py` — `InferenceEngine` (wraps `VA_Server`; single-proc bootstrap)
- `server_ws.py` — websocket deploy server (this is the server entrypoint)
- `weight_prep.py` — bf16 → w4a4 int_weights (data-free)
- `precompute_text.py` — prompt(s) → text-cond cache (T5 once, offline)
- `metrics.py` — optional peak-VRAM / infer-time observation
- `configs/deploy.yaml` — annotated sample config
- `configs/w4a4.yaml` — packaged w4a4 layer config
- `infer_plan.txt` — design/plan

## Publishing the self-contained HF model (maintainers)

`JingwuLuo/lingbot-pick-red-bottle-3k` is made self-contained by uploading the
WAN base `vae/` + `tokenizer/` (from any complete lingbot-va model) and the
precomputed `text_cond_cache/` next to the transformer + int_weights. The
11 GB `text_encoder/` is intentionally NOT uploaded (see §5a). Run from repo
root (write-access HF token; entered at the prompt, not stored):
```bash
python -c "
import os
from getpass import getpass
from huggingface_hub import HfApi
tok = getpass('HF write token (input hidden), then Enter: ').strip()
api = HfApi(); repo = 'JingwuLuo/lingbot-pick-red-bottle-3k'
jobs = [
  ('models/lingbot-va-posttrain-robotwin/vae', 'vae'),
  ('models/lingbot-va-posttrain-robotwin/tokenizer', 'tokenizer'),
  ('PTQEval/ptqeval/inference/tmp/text_cond_cache', 'text_cond_cache'),
]
for local, dst in jobs: assert os.path.isdir(local), 'missing: ' + local
for local, dst in jobs:
    print('uploading', local, '->', dst, flush=True)
    api.upload_folder(folder_path=local, path_in_repo=dst, repo_id=repo, repo_type='model', token=tok)
print('DONE — repo is self-contained')
"
```
