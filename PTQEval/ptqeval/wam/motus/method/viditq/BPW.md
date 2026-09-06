# Motus W4A4+SmoothQuant — bits per weight (bpw)

Average weight precision of the ViDiT-Q W4A4 mixed-precision + SmoothQuant
quantized Motus transformer (WAN video expert + action expert + und expert).
Computed from the released DeepSpeed checkpoint's transformer state_dict (total
weight count) and
`results/motus/motus_w4a4_smooth/calib/int_weights_clean.pth` (per-tier
quantized weight count + on-disk overhead tensors). `int_weights_randomized.pth`
matches on every layer shape, so the numbers below are config-independent.

## Scope / denominator

bpw is over the **transformer weights only** — `video_model.wan_model` +
`action_expert` + `und_expert`, the three experts that are quantized. The Wan2.2
VAE, the umT5 text encoder, and the Qwen3-VL conditioning model are NOT part of
this count (they are never quantized). Total transformer weights (deduplicated
across the DeepSpeed `*_module` wrappers, which alias the same storage):

    total = 5,894,831,822 params = 5.8948 B
          = video_model.wan_model 4.9998 B
          + action_expert         0.6416 B
          + und_expert            0.2535 B

Two bpw numbers are reported; they differ only in the **denominator** — which
weights the average runs over:

    whole transformer 5.89 B:
    +------------- quantized 4.09 B (69.4%) -------------+----- FP bf16 1.80 B (30.6%) -----+
    |  W4A4 2.25 B (4b)  |  W8A8 1.84 B (8b)             |  action/und fused QKV / blocks.0 |
    |                    |                               |  / embeddings / cross-attn / head|
    +---------------------------------------------------+------------- (16b) --------------+
            ^ quantized-path bpw = 5.80  averages only the left block
            ^ whole-transformer bpw = 8.92  averages the whole bar (incl. the 16-bit right)

- **quantized-path bpw** — averages only the weights actually quantized (the
  348 target Linears). Reflects how aggressive the scheme is on the layers it
  touches.
- **whole-transformer bpw** — averages every transformer weight, including the
  30.6 % kept at bf16 (which pulls the average up). Reflects the model's real
  average precision / storage.

## Per-tier statistics

| Tier | Linears | Params | int bits/w | scale | zp | quarot_sign | act_div | overhead |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| W4A4 | 174 | 2.2502 B | 4 | 35.2 MB (bf16, per-group g=128) | — | 0.9 MB | 1.7 MB | 0.134 b/w |
| W8A8 | 174 | 1.8397 B | 8 | 1.5 MB (bf16, per-channel) | 1.5 MB (int16) | 0.1 MB | 0.8 MB | 0.017 b/w |
| **quantized** | **348** | **4.0900 B** | | | | | | |
| FP (bf16) | — | 1.8049 B | 16 | — | — | — | — | — |

- **Quantized** = 174 W4A4 + 174 W8A8 (loader log: `{QuantWanLinearW4A4: 174,
  QuantWanLinearW8A8: 174}; 12 target Linears kept FP`). Split by expert/tier:
  video 116 W4A4 + 58 W8A8, action 29 W4A4 + 58 W8A8, und 29 W4A4 + 58 W8A8.
  W4A4 tier (per-group sym INT4) = `self_attn.{q,k,v}` + `ffn.2`; W8A8 tier
  (per-channel asym INT8) = `self_attn.o` / `wan_{action,und}_o` + `ffn.0`.
- **Model-specific scope**: the action- and und-expert Q/K/V are a *fused
  `nn.Parameter`* (`wan_{action,und}_qkv`, applied by einsum), not `nn.Linear`,
  so they cannot be wrapped by the quantizer — only their output proj + FFN are
  quantized. Only the video expert receives full-attention (q/k/v/o) quant.
  Those fused QKV parameters therefore stay bf16 and are the bulk of the FP tail.
- **FP (bf16)** = everything else in the transformer: the fused action/und QKV
  parameters, all three experts' `blocks.0` (12 Linears kept FP by
  `remain_fp_regex`), cross-attn Linears, patch/text/time embeddings, heads, and
  all non-Linear params (LayerNorm/RMSNorm weights). = total − quantized =
  5.8948 − 4.0900 = 1.8049 B (30.6 %).
- **overhead** = scale + zp + quarot_sign + act_channel_div (SmoothQuant divisor)
  storage, expressed as extra bits per weight of that tier. W4A4's per-group
  bf16 scale dominates: 16 / 128 = 0.125 b/w; the rest is negligible.

## Calculation

### Nominal bpw (int weight bits only)

    whole-model = (Σ_tier params·bits + fp_params·16) / total
                = (2.2502e9·4 + 1.8397e9·8 + 1.8049e9·16) / 5.8948e9
                = (9.001 + 14.718 + 28.878) e9 / 5.8948e9
                = 52.597e9 / 5.8948e9
                = 8.922 bits/weight

    quantized-path = (2.2502e9·4 + 1.8397e9·8) / 4.0900e9
                   = 23.719e9 / 4.0900e9
                   = 5.799 bits/weight

### Effective bpw (int + scale + zp + quarot + smooth-divisor storage)

Adds the per-tier overhead tensors (bf16 scales, int16 zp, int8 quarot_sign,
bf16 act_channel_div) — 41.6 MB total:

    whole-model     = 8.979 bits/weight
    quantized-path  = 5.881 bits/weight

## Result

| bpw | whole transformer | quantized path only |
| --- | ---: | ---: |
| **nominal** (int weight) | **8.922** | **5.799** |
| **effective** (+ scale/zp/quarot/smooth) | **8.979** | **5.881** |

**Headline: the W4A4+SmoothQuant Motus transformer averages ≈ 8.9 bits/weight**
(≈ 5.8 bits/weight over the quantized 69.4 % of the model; the 30.6 % FP
remainder at 16-bit — inflated by the unquantizable fused action/und QKV — pulls
the whole-model average up).

## Weight storage

    effective storage = 6.62 GB   (int 2.96 GB + FP 3.61 GB + scale/zp/quarot/smooth 0.04 GB)
    vs bf16           = 11.79 GB
    -> 1.78x smaller

## Reproduce

    # per-tier: read int_weights_clean.pth on CPU, group keys by prefix.
    #   scale_weight 2-D -> W4A4 (params = int_weight[out, in/2].numel()*2)
    #   scale_weight 1-D + zp_weight -> W8A8 (params = int_weight[out, in].numel())
    #   filter to the loader's target suffixes (self_attn.{q,k,v,o},
    #   wan_{action,und}_o, ffn.0/2); cross-checked exact against the FP
    #   transformer weight shapes in the DeepSpeed 'module' state_dict.
    # total: sum video_model.wan_model + action_expert + und_expert params,
    #   dedup by tensor storage (the video_module/action_module/und_module
    #   wrappers alias the same tensors -> count once); exclude VAE/umT5/Qwen3-VL.
