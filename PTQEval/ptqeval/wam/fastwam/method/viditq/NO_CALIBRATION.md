# Why FastWAM W4A4 (and W4A8) need no calibration

The W4A4, W4A8, and stream_mixed configs are **data-free**: `ptq.py` produces
`int_weights.pth` from the released checkpoint alone, with no activation
statistics. Only the W8A8 paper-namesake config (SmoothQuant on) needs calib.

Every quantization component in the data-free path derives its scales from the
weights or computes them at runtime — none reads an activation dataset.

| Component | Where scales come from | Calib? |
| --- | --- | --- |
| Weight, W4A4 tier | per-group **symmetric** INT4 (group=128): `delta_g = w[c,g].abs().amax() / 7` — from the weight tensor | no |
| Weight, W8A8 tier | per-channel **asymmetric** INT8: `delta = (w.amax - w.amin)/255` — from the weight tensor | no |
| Activation | per-token **dynamic** INT4/INT8: amax computed **per row at runtime** in the kernel | no |
| QuaRoT | per-Linear random ±1 sign (seeded) + structured Hadamard rotation — deterministic, weight-only | no |
| SmoothQuant | per-channel `channel_mask` from **activation absmax** over a calib set | **yes** |

Code references (all in this method package unless noted):

- Weight quant, data-free: `ptq.py::_per_group_sym_quant_w4a4` (W4A4) and
  `_per_channel_asym_quant` / `_per_channel_asym_quant_unsigned` (W8A8/W4A8) —
  each reads only `linear.weight`.
- Runtime dynamic activation quant: `qwan_extension.nn.base.QuantWanLinearBase.forward`
  calls `act_quant_bf16_with_sum` (per-token amax at inference); no stored
  activation scale.
- QuaRoT: `...lingbot_va.method.viditq.quarot.random_sign_vector` (seeded, no data).
- SmoothQuant (the only calib consumer): `ptq.py::compute_int_state_dict`
  requires `calib_data` **only when** `smooth_quant_enabled`. The data-free
  configs set `smooth_quant: false`:
    - `configs/w4a4.yaml`    — `smooth_quant: false` (blocks.0 FP, quarot on)
    - `configs/w4a8.yaml`    — `smooth_quant: false`
    - `configs/stream_mixed.yaml` — `smooth_quant: false`
    - `configs/w8a8.yaml`    — `smooth_quant: true`  → **needs** calib_data.pth

## Upstream evidence (ViDiT-Q source, verified 2026-07-24)

This is ViDiT-Q's intentional design, not our simplification. Direct citations
(paths relative to the vendored `ViDiT-Q/`):

- `quant_utils/qdiff/base/quant_layer.py:11-12` — the base design in one line:
  *"adopt the **static weight** quantization, and the **dynamic activation**
  quantization."* Dynamic = computed online, no calibration.
- `quant_utils/qdiff/base/base_quantizer.py:110` and
  `quant_utils/qdiff/base/mixed_precision_quantizer.py:137` — activation quant
  params are *"get the quant_params **online**"*; `mixed_precision_quantizer.py:184`
  — *"for dynamic quantizer, **no delta list is initialized**"* (no precomputed,
  calib-derived scales).
- `README.md:123` — *"Some quantization techniques (**e.g., smooth quant**)
  requires calibration of activation distribution."* Calibration is scoped to
  SmoothQuant-like techniques only, never the base quantization.
- `examples/pixart/ptq.py:80` / `:115` — `calib_data` is loaded **only** inside
  `if quant_config.get("smooth_quant")` or `if quant_config.get("viditq")`
  (combined). The `quarot` branch (`:101`, `init_rotation_matrix_` at `:63`)
  takes **no** `calib_data`.
- `examples/pixart/configs/w4a4_mixed_precision.yaml` — the W4A4-MP config has a
  `quarot:` block and **no** `smooth_quant`/`viditq` block, so its
  `calib_data: save_path` line is **never read** (template residue). W4A4-MP =
  static weight quant + dynamic online activation quant + QuaRoT, all data-free.

## Is this "by design"? — precise answer (re-verified 2026-08-20)

Two separate facts, do not conflate them:

1. **Dynamic activation quant is ViDiT-Q's intentional core design** (always,
   every config): `quant_layer.py:11-12` — static weight + dynamic activation.
   The base quant never needs calibration. This is not recipe-specific.

2. **W4A4 being data-free is specific to the published W4A4-MP *recipe*, not a
   property of "mixed" or of 4-bit.** It is *not* the case that a mixed method
   skips SmoothQuant. Compare the released ViDiT-Q configs by their method block:

   | Config (vendored `ViDiT-Q/`) | method block | calib? |
   | --- | --- | --- |
   | `examples/pixart/configs/w4a4_mixed_precision.yaml` | **`quarot:` only** | **no (data-free)** |
   | `examples/pixart/configs/w4a8_mixed_precision.yaml` | **`viditq:`** (SmoothQuant α=0.99 + QuaRoT) | **yes** |
   | `examples/opensora1.2/configs/w4a8_mixed_precision.yaml` | **`viditq:`** | yes |
   | `examples/opensora1.2/configs/w8a8.yaml` | **`viditq:`** | yes |
   | `examples/dit/configs/quarot.yaml` | `quarot:` only | no |
   | `examples/dit/configs/sq.yaml` | `smooth_quant:` only | yes |

   So the *other* mixed config (W4A8-MP) DOES use SmoothQuant (via the combined
   `viditq` block) and needs calibration. Only **W4A4-MP** drops SmoothQuant and
   uses QuaRoT alone → data-free.

**Why does W4A4-MP drop SmoothQuant while W4A8-MP keeps it?** Verified against
the ViDiT-Q paper itself (arXiv 2406.02540v3), the answer is: **there is no
W4A4-specific technical reason given — W4A4-MP is simply an under-tuned appendix
config.** Paper evidence (verbatim):

- The full/standard ViDiT-Q method is the **combined** scaling + rotation:
  *"combining scaling and rotation-based channel balancing methods to leverage
  the strengths of both. The scaling-based method addresses the 'static' channel
  imbalance ... The rotation-based method is then utilized to address the
  'dynamic' varying distribution."* (scaling = SmoothQuant, needs calib; rotation
  = QuaRoT, data-free). This is what the headline W8A8 / W4A8 use.
- Activation quant is dynamic (data-free base): *"we propose using 'dynamic'
  quantization parameters, which are computed online."*
- W4A4 lives only in **Appendix D.4**: *"ViDiT-Q W4A4-MP plan employs mixed
  precision **without careful tuning**, assigning 66.7% of layers as W4A4 and
  remaining as W8A8."* (66.7 % matches our 232/348 target Linears.) The paper is
  **silent** on whether SmoothQuant/channel-balancing was applied to W4A4.

The released `w4a4_mixed_precision.yaml` resolves that silence: it ships
**QuaRoT-only** (no `smooth_quant`/`viditq` block), so it is data-free. This is
consistent with the paper's own "without careful tuning" label — the simpler
rotation-only + dynamic path, not the full combined method. So "W4A4 is
data-free" is a property of this **under-tuned appendix recipe**, not a
principled "W4A4 does not need SmoothQuant" design decision.

**Faithfulness note:** our QuaRoT-only, data-free W4A4 matches ViDiT-Q's
published `w4a4_mixed_precision.yaml`. Adding SmoothQuant to W4A4 would make it a
**different, non-published** W4A4 variant (and would then need calibration) —
closer in spirit to the `viditq` combined method used for W4A8/W8A8, but not
what ViDiT-Q released for W4A4.

## Consequence

- A W4A4 re-run is **not affected** by any change to the calibration dataset —
  the calib data is never read on this path. Re-running only re-samples the
  RoboTwin simulator (seed-controlled), so SR reproduces up to sim stochasticity.
- Producing `int_weights.pth` for W4A4 is a single `ptq.py` invocation with no
  prior calibration collection step.
- Only when the paper-faithful W8A8 (SmoothQuant) variant is evaluated does the
  calibration dataset matter.
