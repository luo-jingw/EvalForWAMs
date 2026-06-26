# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Phase 42 step 6c observational check: QuantWanLinearW4A4 wrapper.

Emits metrics only; no assert / no PASS (principle.txt L12).

Sections:
  [F]   from_fp_linear path — build wrapper from random nn.Linear, run
        forward(x_bf16), compare against fp32 reference
        y_ref = x @ dequant(int_weight, scale_weight_natural).T.
  [Fs]  state_dict round-trip — save wrapper.state_dict() then load into
        a fresh wrapper; verify _load_from_state_dict packs scale on
        load (observed via buffer shape switching from natural to packed)
        and forward output matches the source wrapper bit-for-bit.
  [B]   bias path — repeat F with bias=True; observe whether bias contribution
        shifts the diff metric.
  [Q]   QuaRoT integration — install quarot_sign buffer post-construction
        and observe forward result without crash; report shape + magnitude.

Run:
    cd PTQEval/ptqeval/wam/lingbot_va/method/viditq/kernel
    python -m qwan_extension.check_qlinear_w4a4
"""
from __future__ import annotations

import torch
import torch.nn as nn

from qwan_extension.nn.qlinear_w4a4 import QuantWanLinearW4A4


_GROUP_SIZE = 128


def _unpack_int4_signed(packed_u8: torch.Tensor) -> torch.Tensor:
    """uint8 [C_out, C_in/2] (low nibble = col 2c, high = col 2c+1)
    -> int8 [C_out, C_in] with signed [-8, 7] values."""
    C_out, half = packed_u8.shape
    u = packed_u8.to(torch.int32)
    low = u & 0xF
    high = (u >> 4) & 0xF
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    out = torch.empty(C_out, half * 2, dtype=torch.int8, device=packed_u8.device)
    out[:, 0::2] = low.to(torch.int8)
    out[:, 1::2] = high.to(torch.int8)
    return out


def _dequant_w_natural(int_w_packed: torch.Tensor, scale_natural: torch.Tensor) -> torch.Tensor:
    """int_w_packed uint8 [C_out, C_in/2] + scale_natural bf16 [C_out, C_in/128]
    -> bf16 [C_out, C_in] dequant float weight."""
    int_w = _unpack_int4_signed(int_w_packed).to(torch.float32)        # [C_out, C_in]
    C_out, C_in = int_w.shape
    n_groups = scale_natural.shape[1]
    scale_f32 = scale_natural.to(torch.float32)                        # [C_out, G]
    w_g = int_w.view(C_out, n_groups, _GROUP_SIZE) * scale_f32.unsqueeze(2)
    return w_g.view(C_out, C_in).to(torch.bfloat16)


def _summary(label: str, y_quant: torch.Tensor, y_ref: torch.Tensor) -> None:
    diff = (y_quant.float() - y_ref.float()).abs()
    mag = y_ref.float().abs().mean().item()
    print(f"  {label:30s} shape_match={tuple(y_quant.shape) == tuple(y_ref.shape)}  "
          f"max_abs={diff.max().item():.4e}  "
          f"mean_abs={diff.mean().item():.4e}  "
          f"ref_mag={mag:.4e}  "
          f"rel_max={diff.max().item() / (mag + 1e-12):.4e}")


def _case(label: str, in_features: int, out_features: int, M: int, has_bias: bool):
    print()
    print("-" * 88)
    print(f"[F] from_fp_linear  {label}  in={in_features} out={out_features} "
          f"M={M} bias={has_bias}")
    print("-" * 88)
    torch.manual_seed(0)
    fp = nn.Linear(in_features, out_features, bias=has_bias).to(torch.bfloat16).cuda()
    with torch.no_grad():
        fp.weight.uniform_(-0.05, 0.05)
        if has_bias:
            fp.bias.uniform_(-0.01, 0.01)

    mod = QuantWanLinearW4A4.from_fp_linear(fp)
    print(f"  buffers: int_weight {tuple(mod.int_weight.shape)} {mod.int_weight.dtype}, "
          f"scale_weight {tuple(mod.scale_weight.shape)} {mod.scale_weight.dtype}, "
          f"bias={'yes' if mod.bias is not None else 'no'}")

    x = (torch.randn(M, in_features, dtype=torch.bfloat16, device="cuda") * 0.05).contiguous()
    y_quant = mod(x)

    # fp32 reference. scale_weight buffer is now packed; recover natural via
    # the same quant function and recompute dequant for reference.
    from ptqeval.wam.lingbot_va.method.viditq.ptq import _per_group_sym_quant_w4a4
    w_f32 = fp.weight.detach().to(torch.float32)
    int_w_p, scale_w_nat = _per_group_sym_quant_w4a4(w_f32, group_size=_GROUP_SIZE)
    w_dequant = _dequant_w_natural(int_w_p.to("cuda"), scale_w_nat.to("cuda"))
    y_ref = x.float() @ w_dequant.float().T
    if has_bias:
        y_ref = y_ref + fp.bias.float()

    _summary("y_quant vs fp32 dequant ref", y_quant, y_ref.to(torch.bfloat16))


def _state_dict_roundtrip():
    print()
    print("-" * 88)
    print("[Fs] state_dict round-trip — scale_weight gets packed on load")
    print("-" * 88)
    torch.manual_seed(7)
    fp = nn.Linear(3072, 3072, bias=True).to(torch.bfloat16).cuda()
    with torch.no_grad():
        fp.weight.uniform_(-0.05, 0.05)
        fp.bias.uniform_(-0.01, 0.01)
    src = QuantWanLinearW4A4.from_fp_linear(fp)
    sd = src.state_dict()
    print(f"  on-disk shapes (state_dict): "
          f"scale_weight {tuple(sd['scale_weight'].shape)} "
          f"int_weight {tuple(sd['int_weight'].shape)} {sd['int_weight'].dtype}")
    # NOTE the on-disk state_dict carries scale_weight in PACKED layout because
    # from_fp_linear already packs at build time. To exercise the on-load pack
    # path, manually substitute a NATURAL-shaped tensor and reload.
    from ptqeval.wam.lingbot_va.method.viditq.ptq import _per_group_sym_quant_w4a4
    w_f32 = fp.weight.detach().to(torch.float32)
    _, scale_w_nat = _per_group_sym_quant_w4a4(w_f32, group_size=_GROUP_SIZE)
    sd_natural = dict(sd)
    sd_natural["scale_weight"] = scale_w_nat.cuda().contiguous()
    print(f"  injected NATURAL scale_weight {tuple(sd_natural['scale_weight'].shape)} "
          f"into a synthetic 'on-disk' state_dict")

    dst = QuantWanLinearW4A4(3072, 3072, has_bias=True).cuda()
    dst.load_state_dict(sd_natural)
    print(f"  after load: dst.scale_weight {tuple(dst.scale_weight.shape)} "
          f"(expect packed [{3072 // _GROUP_SIZE}, 3072])")

    x = torch.randn(128, 3072, dtype=torch.bfloat16, device="cuda") * 0.05
    y_src = src(x)
    y_dst = dst(x)
    _summary("y_src vs y_dst (round-trip)", y_dst, y_src)


def _quarot_smoke():
    print()
    print("-" * 88)
    print("[Q] QuaRoT input rotation present — shape + magnitude check only")
    print("-" * 88)
    torch.manual_seed(11)
    fp = nn.Linear(3072, 3072, bias=False).to(torch.bfloat16).cuda()
    with torch.no_grad():
        fp.weight.uniform_(-0.05, 0.05)
    mod = QuantWanLinearW4A4.from_fp_linear(fp)
    sign = torch.where(torch.randn(3072) > 0,
                       torch.tensor(1, dtype=torch.int8),
                       torch.tensor(-1, dtype=torch.int8)).cuda()
    mod.quarot_sign = sign
    x = torch.randn(128, 3072, dtype=torch.bfloat16, device="cuda") * 0.05
    y = mod(x)
    print(f"  shape={tuple(y.shape)}  dtype={y.dtype}  "
          f"max_abs={y.float().abs().max().item():.4e}  "
          f"mean_abs={y.float().abs().mean().item():.4e}  "
          f"(no fp ref; rotation flips both sides — observation only)")


def main():
    print("Phase 42 step 6c observational check — QuantWanLinearW4A4 wrapper")
    print(f"torch {torch.__version__}  device {torch.cuda.get_device_name(0)}")
    _case("attn1.to_q", in_features=3072, out_features=3072, M=128, has_bias=False)
    _case("ffn.net.2", in_features=14336, out_features=3072, M=128, has_bias=True)
    _case("M=256 multi-block", in_features=3072, out_features=3072, M=256, has_bias=False)
    _case("M=200 ragged (zero-pad)", in_features=3072, out_features=3072, M=200, has_bias=False)
    _state_dict_roundtrip()
    _quarot_smoke()


if __name__ == "__main__":
    main()
