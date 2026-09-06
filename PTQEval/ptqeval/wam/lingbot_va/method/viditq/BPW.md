# lingbot_va W4A4+SmoothQuant — bits per weight (bpw)

Average weight precision of the ViDiT-Q W4A4 mixed-precision + SmoothQuant
quantized lingbot_va WAN transformer (`blocks` DiT). Computed from the released
FP transformer (`models/lingbot-va-posttrain-robotwin/transformer/*.safetensors`,
total weight count) and
`results/lingbot_va/viditq_w4a4_smooth/calib/int_weights.pth` (per-tier
quantized weight count + on-disk overhead tensors).

## Scope / denominator

bpw is over the **WAN transformer weights only** — the `blocks` DiT that is
quantized. The VAE and the text encoder are NOT part of this count (they are
never quantized). Total transformer weights (all bf16):

    total = 5,089,465,566 params = 5.0895 B   (transformer state_dict, float tensors)

Two bpw numbers are reported; they differ only in the **denominator** — which
weights the average runs over:

    whole transformer 5.09 B:
    +------------- quantized 3.65 B (71.7%) -------------+----- FP bf16 1.44 B (28.3%) -----+
    |  W4A4 2.10 B (4b)  |  W8A8 1.55 B (8b)             |  cross-attn attn2.* / blocks.0 / |
    |                    |                               |  embeds / proj_out / norms (16b) |
    +---------------------------------------------------+----------------------------------+
            ^ quantized-path bpw = 5.70  averages only the left block
            ^ whole-transformer bpw = 8.62  averages the whole bar (incl. the 16-bit right)

- **quantized-path bpw** — averages only the weights actually quantized (the
  174 target Linears swapped at eval). Reflects how aggressive the scheme is on
  the layers it touches.
- **whole-transformer bpw** — averages every transformer weight, including the
  28.3 % kept at bf16 (which pulls the average up). Reflects the model's real
  average precision / storage.

## Per-tier statistics

| Tier | Linears | Params | int bits/w | scale | zp | quarot_sign | act_div | overhead |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| W4A4 | 116 | 2.0982 B | 4 | 32.8 MB (bf16, per-group g=128) | — | 0.7 MB | 1.4 MB | 0.133 b/w |
| W8A8 | 58 | 1.5508 B | 8 | 1.0 MB (bf16, per-channel) | 1.0 MB (int16) | 0.1 MB | 0.4 MB | 0.013 b/w |
| **quantized** | **174** | **3.6490 B** | | | | | | |
| FP (bf16) | — | 1.4404 B | 16 | — | — | — | — | — |

- **Quantized** = 116 W4A4 + 58 W8A8 (loader log: `{QuantWanLinearW4A4: 116,
  QuantWanLinearW8A8: 58}; 1 block kept fully FP`) = the 6 target Linears/block
  × 29 blocks (blocks.0 FP). W4A4 tier (per-group sym INT4) =
  `attn1.{to_q,to_k,to_v}` + `ffn.net.2`; W8A8 tier (per-channel asym INT8) =
  `attn1.to_out.0` + `ffn.net.0.proj`.
- **cross-attn kept bf16**: `ptq.py` also quantizes the 116 cross-attention
  `attn2.*` Linears (they appear in `int_weights.pth` as an extra W8A8 group of
  116 layers = 1.0947 B params), but the runtime loader's `_TARGET_SUFFIXES`
  cover only `attn1.*` + `ffn.*`, so **the `attn2.*` entries are never loaded and
  those layers run bf16**. The effective quantized scope is therefore 174 layers
  (116 W4A4 + 58 W8A8), and `attn2.*` counts toward the FP remainder here.
- **FP (bf16)** = everything else in the transformer: the cross-attn `attn2.*`
  Linears (kept bf16 as above), `blocks.0`, patch/action/condition/time
  embeddings, `proj_out` / `action_proj_out`, and all non-Linear params
  (norm weights, `scale_shift_table`). = total − quantized = 5.0895 − 3.6490 =
  1.4404 B (28.3 %).
- **overhead** = scale + zp + quarot_sign + act_channel_div (SmoothQuant divisor)
  storage, expressed as extra bits per weight of that tier. W4A4's per-group
  bf16 scale dominates: 16 / 128 = 0.125 b/w; the rest is negligible.

## Calculation

### Nominal bpw (int weight bits only)

    whole-model = (Σ_tier params·bits + fp_params·16) / total
                = (2.0982e9·4 + 1.5508e9·8 + 1.4404e9·16) / 5.0895e9
                = (8.393 + 12.407 + 23.047) e9 / 5.0895e9
                = 43.846e9 / 5.0895e9
                = 8.615 bits/weight

    quantized-path = (2.0982e9·4 + 1.5508e9·8) / 3.6490e9
                   = 20.800e9 / 3.6490e9
                   = 5.700 bits/weight

### Effective bpw (int + scale + zp + quarot + smooth-divisor storage)

Adds the per-tier overhead tensors (bf16 scales, int16 zp, int8 quarot_sign,
bf16 act_channel_div) — 37.3 MB total:

    whole-model     = 8.674 bits/weight
    quantized-path  = 5.782 bits/weight

## Result

| bpw | whole transformer | quantized path only |
| --- | ---: | ---: |
| **nominal** (int weight) | **8.615** | **5.700** |
| **effective** (+ scale/zp/quarot/smooth) | **8.674** | **5.782** |

**Headline: the W4A4+SmoothQuant lingbot_va transformer averages ≈ 8.6 bits/weight**
(≈ 5.7 bits/weight over the quantized 71.7 % of the model; the 28.3 % FP
remainder at 16-bit — cross-attn `attn2.*` + `blocks.0` + embeddings — pulls the
whole-model average up).

## Weight storage

    effective storage = 5.52 GB   (int 2.60 GB + FP 2.88 GB + scale/zp/quarot/smooth 0.04 GB)
    vs bf16           = 10.18 GB
    -> 1.84x smaller

## Reproduce

    # per-tier: read int_weights.pth on CPU, group keys by prefix.
    #   scale_weight 2-D -> W4A4 (params = int_weight[out, in/2].numel()*2)
    #   scale_weight 1-D + zp_weight -> W8A8 (params = int_weight[out, in].numel())
    #   filter to the loader's target suffixes (attn1.to_q/k/v, attn1.to_out.0,
    #   ffn.net.0.proj, ffn.net.2) -> 116 W4A4 + 58 W8A8; the 116 attn2.* W8A8
    #   entries in the ckpt are NOT loaded (not in _TARGET_SUFFIXES). Cross-checked
    #   exact against the FP transformer .safetensors weight shapes.
    # total: sum all tensors in models/lingbot-va-posttrain-robotwin/transformer/
    #   *.safetensors (the DiT); exclude VAE / text encoder.
