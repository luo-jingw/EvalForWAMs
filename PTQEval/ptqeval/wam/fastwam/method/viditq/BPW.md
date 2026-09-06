# FastWAM W4A4 — bits per weight (bpw)

Average weight precision of the W4A4 mixed-precision quantized FastWAM MoT
transformer. Computed from the released checkpoint's `mot` state_dict (total
weight count) and `results/fastwam/fastwam_w4a4/calib/int_weights.pth` (per-tier
quantized weight count + on-disk overhead tensors).

## Scope / denominator

bpw is over the **MoT transformer weights only** (video expert + action expert)
— the model that is quantized. VAE and the T5 text encoder are NOT part of this
count (they are never quantized). Total transformer weights:

    total = 6,020,688,078 params = 6.0207 B   (mot state_dict, float tensors)

Two bpw numbers are reported; they differ only in the **denominator** — which
weights the average runs over:

    whole transformer 6.02 B:
    +------------- quantized 4.26 B (70.7%) -------------+---- FP bf16 1.76 B (29.3%) ----+
    |  W4A4 2.49 B (4b)  |  W8A8 1.76 B (8b)             |  cross-attn / embed / norm /   |
    |                    |                              |  blocks.0 / head   (16b)       |
    +---------------------------------------------------+--------------------------------+
            ^ quantized-path bpw = 5.66  averages only the left block
            ^ whole-transformer bpw = 8.69  averages the whole bar (incl. the 16-bit right)

- **quantized-path bpw** — averages only the weights actually quantized (the
  348 target Linears). Reflects how aggressive the scheme is on the layers it
  touches.
- **whole-transformer bpw** — averages every transformer weight, including the
  29.3 % kept at bf16 (which pulls the average up). Reflects the model's real
  average precision / storage.

## Per-tier statistics

| Tier | Linears | Params | int bits/w | scale | zp/szeros | quarot_sign | overhead |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| W4A4 | 232 | 2.4935 B | 4 | 39.0 MB (bf16, per-group g=128) | — | 0.9 MB | 0.128 b/w |
| W8A8 | 116 | 1.7637 B | 8 | 1.3 MB (bf16, per-channel) | 1.3 MB (int16) | 0.1 MB | 0.012 b/w |
| **quantized** | **348** | **4.2572 B** | | | | | |
| FP (bf16) | — | 1.7635 B | 16 | — | — | — | — |

- **Quantized** = the 6 target Linears/block × 29 blocks × 2 experts (blocks.0 FP).
  W4A4 tier = self_attn.{q,k,v} + ffn.2 (per-group sym INT4); W8A8 tier =
  self_attn.o + ffn.0 (per-channel asym INT8).
- **FP (bf16)** = everything else in the transformer: cross-attn Linears,
  patch/text/time embeddings, head, blocks.0, and all non-Linear params
  (RMSNorm/LayerNorm weights, scale_shift_table). = total − quantized =
  6.0207 − 4.2572 = 1.7635 B (29.3 % of the model).
- **overhead** = scale + zp/szeros + quarot_sign storage, expressed as extra
  bits per weight of that tier. W4A4's per-group scale (bf16 per 128 weights)
  dominates: 16 / 128 = 0.125 b/w; W8A8's per-channel scale+zp is negligible.

## Calculation

### Nominal bpw (int weight bits only)

    whole-model = (Σ_tier params·bits + fp_params·16) / total
                = (2.4935e9·4 + 1.7637e9·8 + 1.7635e9·16) / 6.0207e9
                = (9.974 + 14.110 + 28.216) e9 / 6.0207e9
                = 52.300e9 / 6.0207e9
                = 8.687 bits/weight

    quantized-path = (2.4935e9·4 + 1.7637e9·8) / 4.2572e9
                   = 24.084e9 / 4.2572e9
                   = 5.657 bits/weight

### Effective bpw (int + scale + zp + quarot storage)

Adds the per-tier overhead tensors (bf16 scales, int16 zp, int8 quarot_sign):

    whole-model     = 8.743 bits/weight
    quantized-path  = 5.737 bits/weight

## Result

| bpw | whole transformer | quantized path only |
| --- | ---: | ---: |
| **nominal** (int weight) | **8.687** | **5.657** |
| **effective** (+ scale/zp/quarot) | **8.743** | **5.737** |

**Headline: the W4A4 mixed FastWAM transformer averages ≈ 8.7 bits/weight**
(≈ 5.7 bits/weight over the quantized 70.7 % of the model; the 29.3 % FP
remainder at 16-bit pulls the whole-model average up).

## Weight storage

    effective storage = 6.58 GB   (int 3.01 GB + FP 3.53 GB + scale/zp/quarot 0.04 GB)
    vs bf16           = 12.04 GB
    -> 1.83x smaller  (matches the measured transformer segment ~6.1-6.3 GB on GPU)

## Reproduce

    # loads mot state_dict (total params) + int_weights.pth (per-tier), CPU-only
    # see the inline computation in the BPW section of the session log, or
    # sum int_weight.numel() (x2 for the C_in-packed 4-bit tiers) per tier.
