# Why FastWAM W4A4 uses SYMMETRIC weight+activation (not the yaml's asymmetric)

The ViDiT-Q W4A4-MP yaml declares `sym: false` (asymmetric). Our FastWAM W4A4
uses per-group **symmetric** INT4. This is a faithful, deliberate choice, not a
simplification — ViDiT-Q's own **real** W4A4 kernel is symmetric-only and was
never runnable. Same decision as LingBot-VA (plan.txt Phase 42 G5).

## ViDiT-Q is internally contradictory on W4A4 symmetry

| Representation | Symmetry | Actually runnable? | Source (vendored `ViDiT-Q/`) |
| --- | --- | --- | --- |
| yaml config | asymmetric | — | `examples/pixart/configs/w4a4_mixed_precision.yaml` (`weight.sym: false`, `act.sym: false`) |
| PyTorch fake-quant (software simulation — where any published W4A4 quality comes from) | asymmetric | yes, but **fake quant** (quantize→dequantize in fp; no real speedup) | `quant_utils/qdiff/base/quant_layer.py:83,88` (`'sym': False`) |
| real CUDA kernel `atom.cu` | **symmetric** | **no — never bound to Python** | see below |

Real-kernel evidence:

- `kernels/csrc/qgemm/w4a4/atom.cu:421-432` — the dequant epilogue is
  `accu += c_frag * (row_scale * col_scale)`, i.e. `int_result × scale_a ×
  scale_b` with **no zero-point term**. Purely symmetric.
- `kernels/csrc/qgemm/pybind.cpp` — binds only `w8a8_*` (×3) and
  `w4a8_of16_nobias_weight_asym_qserve`. There is **no `m.def` for any w4a4
  kernel**, so `atom.cu` compiles into the `.so` but Python can never call it.
  ViDiT-Q ships **no runnable real W4A4 kernel**.
- `atom.cu` also carries the Atom paper's outlier "keeper" path
  (`mma_calculate_keeper`, `A_keeper`/`B_keeper`), which ViDiT-Q's
  paper/yaml/quant_utils never reference (plan Phase 42 G2).

## Decision: keep symmetric — do NOT switch to asymmetric

1. ViDiT-Q's only real W4A4 kernel (`atom.cu`) is symmetric; our per-group
   symmetric INT4 matches its exact arithmetic (`int × scale_a × scale_b`).
2. That kernel is unbound → ViDiT-Q has no runnable real W4A4 to be "more
   faithful" to. Any real W4A4 (sym or asym) already goes beyond what ViDiT-Q
   shipped runnable.
3. Reproducing the asymmetric path as a *real* kernel would mean building a
   W4A4 asym GEMM that ViDiT-Q never provided — that is our kernel, not theirs,
   so it is *less* faithful, and it yields no real speedup (the asym path is
   fake-quant only).
4. This project measures **real-kernel** speedup + SR; a symmetric real kernel
   is the honest, runnable "ViDiT-Q-style W4A4".

## When asymmetric would matter

Only if the baseline's goal were to reproduce ViDiT-Q's *published W4A4 quality*
numbers (from its asymmetric fake-quant). But ViDiT-Q's W4A4-MP is an appendix
"future direction, without careful tuning" result, and matching it would need a
new asym kernel with no real speedup — not worth it for a speed + SR baseline.

## Honest baseline label

> ViDiT-Q W4A4 method (QuaRoT + dynamic per-token activation + mixed
> W4A4/W8A8), realized with a real **symmetric** per-group INT4 kernel matching
> ViDiT-Q `atom.cu`'s arithmetic. ViDiT-Q's yaml marks it asymmetric, but its
> real kernel is symmetric-only and unbound, so the asymmetric fields are
> non-functional residue.
