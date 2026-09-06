# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Phase 37: QuaRoT data-independent Hadamard rotation.

Direct port of ViDiT-Q/quant_utils/qdiff/quarot/{quarot_quant_layer.py,
quarot_utils.py}:158-192. PTQ-time: for each target Linear, generate a
random Bernoulli sign vector s in {+-1}^C_in, rotate the weight
W_rot = (W * s) @ H where H is the Hadamard matrix of size C_in
applied via butterfly + a small head matrix (NOT materialized). At
runtime, forward applies the same rotation to the input before
quantization. Because H @ H.T / n = I, the un-quantized math is
preserved:
    y = (x @ R) @ (W @ R).T = x @ R @ R.T @ W.T = x @ W.T
where R = diag(s) @ H / sqrt(n) is the per-Linear orthogonal rotation.
Quantization happens on the rotated W and rotated x; the rotation
decorrelates outliers so per-channel scales fit tighter.

Storage per Linear: int8 [C_in] sign vector. The C_in x C_in Hadamard
matrix is NEVER materialized -- it would be 410 MB on a single
14336-wide Linear, which is untenable across 30 down_proj instances.
Recursive Kronecker factorization (a la upstream matmul_hadU) gives
the same numerical result via O(C_in log C_in) butterfly + a small
[K, K] @ [K, n/K] matmul, where K in {1, 12, 28} divides n into a
power-of-2 quotient. WAN dims:
    3072  = 12 * 2^8   -> K=12 had12
    14336 = 28 * 2^9   -> K=28 had28

had12 and had28 tables are verbatim from ViDiT-Q upstream
(quarot_utils.py:269-285 and :2422-...).
"""
from __future__ import annotations

import torch


# had12 / had28 are 12x12 / 28x28 Hadamard matrices in {+-1} (NOT normalized).
# Cached after first construction to avoid repeated tensor build cost.
_HAD12: torch.Tensor | None = None
_HAD28: torch.Tensor | None = None

# Lazy availability flag for fast_hadamard_transform (the same library
# ViDiT-Q upstream uses in matmul_hadU_cuda). When present, runtime
# butterfly collapses from ~9 PyTorch view/slice launches per layer to
# a single fused CUDA launch -- the engineering equivalent of upstream's
# CUDA path. When absent, we fall back to the verbatim Python butterfly
# below. Both paths are mathematically identical; the CUDA path is the
# faithful one (matches quarot_utils.py:195-208 verbatim) and the Python
# path is the fallback we shipped before the optional dep was wired up.
try:
    import fast_hadamard_transform as _fht  # type: ignore[import-not-found]
    _HAVE_FHT = True
except ImportError:
    _fht = None
    _HAVE_FHT = False


def _build_had12() -> torch.Tensor:
    # ViDiT-Q/quant_utils/qdiff/quarot/quarot_utils.py:269-285 verbatim.
    return torch.tensor(
        [
            [+1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [+1, +1, -1, +1, -1, -1, -1, +1, +1, +1, -1, +1],
            [+1, +1, +1, -1, +1, -1, -1, -1, +1, +1, +1, -1],
            [+1, -1, +1, +1, -1, +1, -1, -1, -1, +1, +1, +1],
            [+1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1, +1],
            [+1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1],
            [+1, +1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1],
            [+1, -1, +1, +1, +1, -1, +1, +1, -1, +1, -1, -1],
            [+1, -1, -1, +1, +1, +1, -1, +1, +1, -1, +1, -1],
            [+1, -1, -1, -1, +1, +1, +1, -1, +1, +1, -1, +1],
            [+1, +1, -1, -1, -1, +1, +1, +1, -1, +1, +1, -1],
            [+1, -1, +1, -1, -1, -1, +1, +1, +1, -1, +1, +1],
        ],
        dtype=torch.float32,
    )


def _build_had28() -> torch.Tensor:
    # ViDiT-Q/quant_utils/qdiff/quarot/quarot_utils.py:2422-... verbatim
    # (extracted to compact form; 28x28 = 784 +-1 entries preserved bit-exact).
    return torch.tensor(
        [
            [+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,-1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1],
            [+1,+1,+1,-1,+1,+1,-1,-1,-1,-1,+1,+1,-1,+1,+1,-1,+1,-1,+1,+1,-1,-1,-1,-1,+1,+1,-1,+1],
            [+1,+1,+1,+1,-1,+1,+1,-1,-1,-1,-1,+1,+1,-1,+1,+1,-1,+1,-1,+1,+1,-1,-1,-1,-1,+1,+1,-1],
            [+1,-1,+1,+1,+1,-1,+1,+1,-1,-1,-1,-1,+1,+1,+1,-1,+1,-1,+1,-1,+1,+1,-1,-1,-1,-1,+1,+1],
            [+1,+1,-1,+1,+1,+1,-1,+1,+1,-1,-1,-1,-1,+1,+1,+1,-1,+1,-1,+1,-1,+1,+1,-1,-1,-1,-1,+1],
            [+1,+1,+1,-1,+1,+1,+1,-1,+1,+1,-1,-1,-1,-1,+1,+1,+1,-1,+1,-1,+1,-1,+1,+1,-1,-1,-1,-1],
            [+1,-1,+1,+1,-1,+1,+1,+1,-1,+1,+1,-1,-1,-1,+1,-1,+1,+1,-1,+1,-1,+1,-1,+1,+1,-1,-1,-1],
            [+1,-1,-1,+1,+1,-1,+1,+1,+1,-1,+1,+1,-1,-1,+1,-1,-1,+1,+1,-1,+1,-1,+1,-1,+1,+1,-1,-1],
            [+1,-1,-1,-1,+1,+1,-1,+1,+1,+1,-1,+1,+1,-1,+1,-1,-1,-1,+1,+1,-1,+1,-1,+1,-1,+1,+1,-1],
            [+1,-1,-1,-1,-1,+1,+1,-1,+1,+1,+1,-1,+1,+1,+1,-1,-1,-1,-1,+1,+1,-1,+1,-1,+1,-1,+1,+1],
            [+1,+1,-1,-1,-1,-1,+1,+1,-1,+1,+1,+1,-1,+1,+1,+1,-1,-1,-1,-1,+1,+1,-1,+1,-1,+1,-1,+1],
            [+1,+1,+1,-1,-1,-1,-1,+1,+1,-1,+1,+1,+1,-1,+1,+1,+1,-1,-1,-1,-1,+1,+1,-1,+1,-1,+1,-1],
            [+1,-1,+1,+1,-1,-1,-1,-1,+1,+1,-1,+1,+1,+1,+1,-1,+1,+1,-1,-1,-1,-1,+1,+1,-1,+1,-1,+1],
            [+1,+1,-1,+1,+1,-1,-1,-1,-1,+1,+1,-1,+1,+1,+1,+1,-1,+1,+1,-1,-1,-1,-1,+1,+1,-1,+1,-1],
            [-1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
            [+1,-1,+1,-1,+1,+1,-1,-1,-1,-1,+1,+1,-1,+1,-1,-1,-1,+1,-1,-1,+1,+1,+1,+1,-1,-1,+1,-1],
            [+1,+1,-1,+1,-1,+1,+1,-1,-1,-1,-1,+1,+1,-1,-1,-1,-1,-1,+1,-1,-1,+1,+1,+1,+1,-1,-1,+1],
            [+1,-1,+1,-1,+1,-1,+1,+1,-1,-1,-1,-1,+1,+1,-1,+1,-1,-1,-1,+1,-1,-1,+1,+1,+1,+1,-1,-1],
            [+1,+1,-1,+1,-1,+1,-1,+1,+1,-1,-1,-1,-1,+1,-1,-1,+1,-1,-1,-1,+1,-1,-1,+1,+1,+1,+1,-1],
            [+1,+1,+1,-1,+1,-1,+1,-1,+1,+1,-1,-1,-1,-1,-1,-1,-1,+1,-1,-1,-1,+1,-1,-1,+1,+1,+1,+1],
            [+1,-1,+1,+1,-1,+1,-1,+1,-1,+1,+1,-1,-1,-1,-1,+1,-1,-1,+1,-1,-1,-1,+1,-1,-1,+1,+1,+1],
            [+1,-1,-1,+1,+1,-1,+1,-1,+1,-1,+1,+1,-1,-1,-1,+1,+1,-1,-1,+1,-1,-1,-1,+1,-1,-1,+1,+1],
            [+1,-1,-1,-1,+1,+1,-1,+1,-1,+1,-1,+1,+1,-1,-1,+1,+1,+1,-1,-1,+1,-1,-1,-1,+1,-1,-1,+1],
            [+1,-1,-1,-1,-1,+1,+1,-1,+1,-1,+1,-1,+1,+1,-1,+1,+1,+1,+1,-1,-1,+1,-1,-1,-1,+1,-1,-1],
            [+1,+1,-1,-1,-1,-1,+1,+1,-1,+1,-1,+1,-1,+1,-1,-1,+1,+1,+1,+1,-1,-1,+1,-1,-1,-1,+1,-1],
            [+1,+1,+1,-1,-1,-1,-1,+1,+1,-1,+1,-1,+1,-1,-1,-1,-1,+1,+1,+1,+1,-1,-1,+1,-1,-1,-1,+1],
            [+1,-1,+1,+1,-1,-1,-1,-1,+1,+1,-1,+1,-1,+1,-1,+1,-1,-1,+1,+1,+1,+1,-1,-1,+1,-1,-1,-1],
            [+1,+1,-1,+1,+1,-1,-1,-1,-1,+1,+1,-1,+1,-1,-1,-1,+1,-1,-1,+1,+1,+1,+1,-1,-1,+1,-1,-1],
        ],
        dtype=torch.float32,
    )


def _get_had12() -> torch.Tensor:
    global _HAD12
    if _HAD12 is None:
        _HAD12 = _build_had12()
    return _HAD12


def _get_had28() -> torch.Tensor:
    global _HAD28
    if _HAD28 is None:
        _HAD28 = _build_had28()
    return _HAD28


# Order matters: larger K is tried first because had28 covers 14336 but had12
# does not (14336 % 12 != 0 with pow-2 quotient). For 3072, had28 fails
# (3072 % 28 != 0), so K=12 is selected; for 14336, K=28; for pure pow-2
# dims (e.g. 1024 if ever encountered), K=1.
_SUPPORTED_BASES: list[tuple[int, callable]] = [
    (28, _get_had28),
    (12, _get_had12),
]


def _factor_hadamard_size(n: int) -> tuple[torch.Tensor | None, int]:
    """Return (hadK, K) such that n = K * 2^m. hadK is None if K == 1."""
    for K, getter in _SUPPORTED_BASES:
        if n % K != 0:
            continue
        q = n // K
        if q > 0 and (q & (q - 1)) == 0:
            return getter(), K
    if n > 0 and (n & (n - 1)) == 0:
        return None, 1
    raise ValueError(
        f"Hadamard size {n} not supported by bases {{28, 12, 1}}; expected "
        f"n = K * 2^m for some K in {{28, 12}} or n a power of 2."
    )


def _matmul_hadU(x: torch.Tensor) -> torch.Tensor:
    """Compute X @ H / sqrt(n) where H is the structured Hadamard matrix of
    size n = x.shape[-1]. Applies butterfly to the last dim then a small
    head matmul. Matches ViDiT-Q quarot_utils.py:195-208 numerically.

    The transform is its own inverse up to the 1/sqrt(n) scaling that this
    function applies on output (so H/sqrt(n) is orthogonal).

    Two implementations, picked at call time:
      - CUDA (default when fast_hadamard_transform is installed) follows
        upstream matmul_hadU_cuda verbatim: one fused FHT launch + a
        small K-Hadamard matmul. This is the runtime-hot path.
      - Python butterfly fallback (only when the dep is missing): the
        verbatim recursive Kronecker we shipped in Phase 37 prior to
        wiring the optional dep. Mathematically identical but ~9 launches
        per layer, which dominates QuaRoT overhead on real models.
    """
    n = x.shape[-1]
    hadK, K = _factor_hadamard_size(n)

    if _HAVE_FHT:
        # Upstream quarot_utils.py:195-208 exact form. fp64 inputs are
        # downcast to fp32 because fast_hadamard_transform's kernels do
        # not ship a fp64 instantiation; PTQ rotate_weight is the only
        # fp64 caller and stays on the Python path via _matmul_hadU_python.
        scale = 1.0 / (float(n) ** 0.5)
        if K == 1:
            return _fht.hadamard_transform(x.contiguous(), scale)
        # X @ (I_K kron H_(n/K)) then K-Hadamard on the K axis.
        x_view = x.contiguous().view(-1, K, n // K)
        x_view = _fht.hadamard_transform(x_view, scale)
        x_view = hadK.to(device=x_view.device, dtype=x_view.dtype) @ x_view
        return x_view.reshape(x.shape)

    return _matmul_hadU_python(x, hadK, K, n)


def _matmul_hadU_python(x: torch.Tensor, hadK, K: int, n: int) -> torch.Tensor:
    """Verbatim Python butterfly fallback (Phase 37 original implementation).
    Kept verbatim so the fp64 rotate_weight path -- which fast_hadamard
    transform does not support -- still has a correct implementation."""
    a = x.contiguous().view(-1, n, 1)
    out = a.clone()
    while a.shape[1] > K:
        a = a.view(a.shape[0], a.shape[1] // 2, 2, a.shape[2])
        out = out.view(a.shape)
        out[:, :, 0, :] = a[:, :, 0, :] + a[:, :, 1, :]
        out[:, :, 1, :] = a[:, :, 0, :] - a[:, :, 1, :]
        out = out.view(a.shape[0], a.shape[1], -1)
        a, out = out, a
    if K > 1:
        a = hadK.view(1, K, K).to(a) @ a
    return a.view(x.shape) / (n ** 0.5)


def random_sign_vector(in_features: int, seed: int) -> torch.Tensor:
    """Bernoulli sign vector for QuaRoT. Returns int8 [in_features] in
    {+1, -1}. Pinned to a per-Linear seed for reproducibility (so PTQ
    output is deterministic given the same layer order and config)."""
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    bits = torch.randint(0, 2, (in_features,), generator=g, dtype=torch.int8)
    return (bits * 2 - 1).contiguous()


def apply_input_rotation(x: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
    """Runtime rotation: x_rot = (x * sign) @ H / sqrt(n) along last dim.

    x:    [..., in_features], any float dtype
    sign: int8 [in_features], values in {+1, -1}

    Returns: same shape and dtype as x. Computed in x.dtype throughout
    (callers wanting fp32 promotion should cast x ahead of time).
    """
    assert sign.dim() == 1 and sign.shape[0] == x.shape[-1], (
        f"sign shape {tuple(sign.shape)} incompatible with x last dim {x.shape[-1]}"
    )
    s_eff = sign.to(x.dtype)
    # Broadcast over leading dims; multiplication is per-channel.
    return _matmul_hadU(x * s_eff)


def rotate_weight(w: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
    """PTQ-time weight rotation: W_rot = (W * sign[None, :]) @ H / sqrt(n).
    Matches viditq_quant_layer.py:48 (W @ rotation_matrix where
    rotation_matrix = diag(sign) @ H / sqrt(n) is upstream's materialized
    form). Computed in fp64 to mirror upstream's .double() promotion.

    w:    [out_features, in_features], any float dtype
    sign: int8 [in_features]

    Returns: rotated weight, same shape and dtype as w.
    """
    assert w.dim() == 2
    assert sign.dim() == 1 and sign.shape[0] == w.shape[1]
    w64 = w.to(torch.float64)
    s64 = sign.to(torch.float64)
    # Force Python butterfly: fast_hadamard_transform does not ship a
    # fp64 instantiation, and PTQ correctness is the priority here so we
    # mirror upstream's .double() promotion exactly.
    n = w64.shape[-1]
    hadK, K = _factor_hadamard_size(n)
    rotated = _matmul_hadU_python(w64 * s64.unsqueeze(0), hadK, K, n)
    return rotated.to(w.dtype)
