# ImageWAM W4A4+SmoothQuant — bits per weight (bpw)

Average weight precision of the ViDiT-Q W4A4 mixed-precision + SmoothQuant
quantized ImageWAM MoT transformer (FLUX.2 Klein-4B video expert + SlimFlux2
action expert). Computed from the released checkpoint's `mot` state_dict (total
weight count) and
`results/imagewam/imagewam_w4a4_smooth/calib/int_weights_clean.pth` (per-tier
quantized weight count + on-disk overhead tensors). `int_weights_randomized.pth`
matches on every layer shape, so the numbers below are config-independent.

## Scope / denominator

bpw is over the **MoT transformer weights only** (video_expert.transformer +
action_expert) — the model that is quantized. The FLUX.2 autoencoder and the
Qwen3 text encoder are NOT part of this count (they are never quantized). Total
transformer weights:

    total = 4,517,571,328 params = 4.5176 B   (mot state_dict, dedup by storage)
          = video_expert.transformer 3.8755 B  (= FLUX.2 klein-base-4B backbone)
          + action_expert            0.6420 B

Two bpw numbers are reported; they differ only in the **denominator** — which
weights the average runs over:

    whole transformer 4.52 B:
    +--------------- quantized 4.04 B (89.4%) ---------------+ FP bf16 0.48 B (10.6%) +
    |  W4A4 1.41 B (4b)  |  W8A8 2.63 B (8b)                 |  double_blocks.0 /     |
    |                    |                                   |  modulation / img_in / |
    |                    |                                   |  final_layer / AE-adpt |
    +-------------------------------------------------------+------------ (16b) ------+
            ^ quantized-path bpw = 6.60  averages only the left block
            ^ whole-transformer bpw = 7.60  averages the whole bar (incl. the 16-bit right)

- **quantized-path bpw** — averages only the weights actually quantized (the
  128 target Linears swapped at eval). Reflects how aggressive the scheme is on
  the layers it touches; it is high (6.6) because ImageWAM's W8A8 tier carries
  more params than its W4A4 tier (the wide `mlp.0` / attention `proj` layers).
- **whole-transformer bpw** — averages every transformer weight, including the
  10.6 % kept at bf16. ImageWAM has the **smallest FP remainder** of the four
  WAMs (most aggressive relative quantization), so the whole-model average (7.6)
  sits well below the others.

## Per-tier statistics

| Tier | Linears | Params | int bits/w | scale | zp | quarot_sign | act_div | overhead |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| W4A4 | 64 | 1.4093 B | 4 | 22.0 MB (bf16, per-group g=128) | — | 0.4 MB | 1.0 MB | 0.133 b/w |
| W8A8 | 64 | 2.6298 B | 8 | 2.2 MB (bf16, per-channel) | 2.2 MB (int16) | 0.1 MB | 0.3 MB | 0.015 b/w |
| **quantized** | **128** | **4.0391 B** | | | | | | |
| FP (bf16) | — | 0.4785 B | 16 | — | — | — | — | — |

- **Quantized** = 64 W4A4 + 64 W8A8 (loader log: `{QuantWanLinearW4A4: 64,
  QuantWanLinearW8A8: 64}; 12 target Linears kept FP`). Split by expert:
  video 36+36, action 28+28.
  W4A4 tier (per-group sym INT4) = `{img,txt}_attn.qkv` + `{img,txt}_mlp.2` +
  `linear2`; W8A8 tier (per-channel asym INT8) = `{img,txt}_attn.proj` +
  `{img,txt}_mlp.0` + `linear1`.
- **FP (bf16)** = everything else in the transformer: both experts'
  `double_blocks.0` (12 Linears kept FP by `remain_fp_regex`), the modulation /
  `img_in` / `txt_in` / `time_in` / `final_layer` Linears, and all non-Linear
  params (norm scales). = total − quantized = 4.5176 − 4.0391 = 0.4785 B (10.6 %).
- **quarot exclusion note**: the 9216-dim `{img,txt}_mlp.2` (output not
  Hadamard-factorable) is quantized **without** QuaRoT but still lands in the
  W4A4 tier — it carries `scale_weight` (2-D per-group) but no `quarot_sign`.
- **overhead** = scale + zp + quarot_sign + act_channel_div (SmoothQuant divisor)
  storage, expressed as extra bits per weight of that tier. W4A4's per-group
  bf16 scale dominates: 16 / 128 = 0.125 b/w; the rest is negligible.

## Calculation

### Nominal bpw (int weight bits only)

    whole-model = (Σ_tier params·bits + fp_params·16) / total
                = (1.4093e9·4 + 2.6298e9·8 + 0.4785e9·16) / 4.5176e9
                = (5.637 + 21.039 + 7.655) e9 / 4.5176e9
                = 34.331e9 / 4.5176e9
                = 7.599 bits/weight

    quantized-path = (1.4093e9·4 + 2.6298e9·8) / 4.0391e9
                   = 26.676e9 / 4.0391e9
                   = 6.604 bits/weight

### Effective bpw (int + scale + zp + quarot + smooth-divisor storage)

Adds the per-tier overhead tensors (bf16 scales, int16 zp, int8 quarot_sign,
bf16 act_channel_div) — 28.3 MB total:

    whole-model     = 7.650 bits/weight
    quantized-path  = 6.660 bits/weight

## Result

| bpw | whole transformer | quantized path only |
| --- | ---: | ---: |
| **nominal** (int weight) | **7.599** | **6.604** |
| **effective** (+ scale/zp/quarot/smooth) | **7.650** | **6.660** |

**Headline: the W4A4+SmoothQuant ImageWAM transformer averages ≈ 7.6 bits/weight**
(≈ 6.6 bits/weight over the quantized 89.4 % of the model). The quantized-path
average is the highest of the four WAMs because ImageWAM's W8A8 tier (58.2 % of
weights) outweighs its W4A4 tier; the whole-model average is the lowest because
almost nothing is left at bf16.

## Weight storage

    effective storage = 4.32 GB   (int 3.33 GB + FP 0.96 GB + scale/zp/quarot/smooth 0.03 GB)
    vs bf16           = 9.04 GB
    -> 2.09x smaller

## Reproduce

    # per-tier: read int_weights_clean.pth on CPU, group keys by prefix.
    #   scale_weight 2-D -> W4A4 (params = int_weight[out, in/2].numel()*2)
    #   scale_weight 1-D + zp_weight -> W8A8 (params = int_weight[out, in].numel())
    #   filter to the loader's target suffixes (img_attn.qkv/proj, img_mlp.0/2,
    #   txt_attn.qkv/proj, txt_mlp.0/2, linear1/2); cross-checked exact against
    #   the FP mot weight shapes (mixtures.video.transformer.* / mixtures.action.*).
    # total: mot state_dict, dedup by tensor storage (mixtures.video.double_blocks
    #   aliases mixtures.video.transformer.double_blocks -> count once).
