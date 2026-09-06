// Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
//
// Phase 42 step 6a: Atom W4A4 GEMM scale layout pack helpers.
//
// The atom.cu W4A4 kernel (csrc/w4a4/w4a4_gemm.cu) expects per-group scales in
// two on-device buffers, A_scale (per-token activation) and B_scale (per-
// channel weight), with kernel-internal byte layouts derived from the
// ldmatrix.m8n8.x4.shared.b16 load pattern in loadScale + loadScaleReg.
// act_quant_bf16_group128 (csrc/act_quant_bf16.cu) emits the natural layout
// [N_tokens, K/128] row-major; ptq.py emits weight scales in [C_out, K/128]
// row-major. The conversion to Atom-permuted layout is a separate concern
// (plan Phase 42 G5 — kernel is sym-only per-group, layout permutation is
// runtime/PTQ-time code, not a kernel branch).
//
// ---------------------------------------------------------------------------
// A_scale layout derivation
// ---------------------------------------------------------------------------
// In loadScale (w4a4_gemm.cu line ~287), the activation scale for one
// (BLOCK_M=128 row block) x (BLOCK_K=128 K-group) is 1024 bytes = 512 b16
// values = 4 b16 per natural M row (SCALE_PACKING_A(BLOCK_M) * sizeof(half2)
// = 256 * 4). The 4 b16 are *replicas* of the same scalar; the replication
// pattern is the inverse of ldmatrix.m8n8.x4 distribution per loadScaleReg
// (w4a4_gemm.cu line ~336):
//
//   smem_warp_base = wj * 512 bytes  (BLOCK_COL_WARPS = 2)
//   per warp: 4 matrices of 8x8 b16 = 128 bytes each, contiguous
//   matrix tile_i ∈ 0..3 lives at warp_base + tile_i*128 bytes
//   within matrix tile_i, row r ∈ 0..7 lives at +r*16 bytes
//   row r holds 8 b16 in column order:
//     col 0,2,4,6 = scale of "upper" mma row = wj*64 + tile_i*16 + r
//     col 1,3,5,7 = scale of "lower" mma row = wj*64 + tile_i*16 + r + 8
//   (mma fragment lane k holds upper row k/4 + lower row k/4+8, taken
//    from row_scale.x and row_scale.y of reg_a[tile_i].)
//
// Mapping a natural m_block ∈ 0..127 to its 4 b16 destinations:
//   wj      = m_block / 64
//   tile_i  = (m_block % 64) / 16
//   rir     = m_block % 16
//   is_low  = rir / 8
//   r       = rir % 8
//   base_b16_in_block = wj*256 + tile_i*64 + r*8
//   offsets = is_low ? {base+1, base+3, base+5, base+7}
//                    : {base+0, base+2, base+4, base+6}
//
// For global m = block_m * 128 + m_block (block_m = m / 128):
//   per-K-group row stride = 4 * M  (b16)
//   write index = block_m * 512 + offsets
//
// ---------------------------------------------------------------------------
// B_scale layout derivation
// ---------------------------------------------------------------------------
// In loadScale B side the gmem byte offset advances by 2 bytes per N column
// and `2 * N_GLOBAL` bytes per K-group (i.e. [K/128, N] row-major b16).
// No element permutation; the ldmatrix.x4 "replication" effect inside
// loadScaleReg (all 8 lanes per matrix providing the same source address)
// is by design — col-scale has no row dim, so the 8 identical rows are
// distributed as the 8 cols of the matrix.
//
// PTQ emits scale_w as [C_out, K/128] (per-channel per-group). The kernel
// wants [K/128, N] = [G, N] row-major in OutT. pack_atom_scale_b just does
// `transpose(0, 1).contiguous().to(out_dtype)`.
// ---------------------------------------------------------------------------

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>


namespace {

// Atom W4A4 kernel parameters (must match w4a4_gemm.cu).
constexpr int BLOCK_M    = 128;
constexpr int GROUP_SIZE = 128;

// One CUDA thread per (m_global, g_global) natural scalar; emits 4 writes.
template <typename OutT>
__global__ void pack_atom_scale_a_kernel(
    const OutT* __restrict__ src,    // [M, G] row-major
    OutT*       __restrict__ dst,    // [G, 4 * M] row-major
    const int M,
    const int G)
{
    const int total = M * G;
    const int stride = blockDim.x * gridDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += stride)
    {
        const int m = idx / G;
        const int g = idx % G;

        const OutT scale = src[m * G + g];

        const int block_m = m / BLOCK_M;
        const int local   = m & (BLOCK_M - 1);     // m % 128
        const int wj      = local >> 6;            // / 64
        const int tile_i  = (local & 63) >> 4;     // (% 64) / 16
        const int rir     = local & 15;            // % 16
        const int is_low  = rir >> 3;              // / 8
        const int r       = rir & 7;               // % 8

        const int base_b16 = block_m * 512
                           + wj * 256
                           + tile_i * 64
                           + r * 8;

        OutT* row_ptr = dst + g * (4 * M);

        // Disjoint writes per thread (is_low picks odd/even cols within row).
        if (is_low == 0) {
            row_ptr[base_b16 + 0] = scale;
            row_ptr[base_b16 + 2] = scale;
            row_ptr[base_b16 + 4] = scale;
            row_ptr[base_b16 + 6] = scale;
        } else {
            row_ptr[base_b16 + 1] = scale;
            row_ptr[base_b16 + 3] = scale;
            row_ptr[base_b16 + 5] = scale;
            row_ptr[base_b16 + 7] = scale;
        }
    }
}

template <typename OutT>
torch::Tensor _pack_atom_scale_a_impl(
    torch::Tensor natural,
    torch::ScalarType out_torch_dtype)
{
    TORCH_CHECK(natural.is_cuda(), "natural must be cuda");
    TORCH_CHECK(natural.dim() == 2, "natural must be 2-D [M, K/128]");
    TORCH_CHECK(natural.dtype() == out_torch_dtype,
                "natural dtype must match expected output dtype");
    TORCH_CHECK(natural.is_contiguous(), "natural must be contiguous");

    const int64_t M = natural.size(0);
    const int64_t G = natural.size(1);
    TORCH_CHECK(M % BLOCK_M == 0,
                "M must be a multiple of BLOCK_M (128); got ", M);

    auto opts = torch::TensorOptions().dtype(out_torch_dtype).device(natural.device());
    torch::Tensor packed = torch::empty({G, 4 * M}, opts);

    const int total = static_cast<int>(M * G);
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    pack_atom_scale_a_kernel<OutT><<<blocks, threads>>>(
        reinterpret_cast<const OutT*>(natural.data_ptr()),
        reinterpret_cast<OutT*>(packed.data_ptr()),
        static_cast<int>(M),
        static_cast<int>(G));
    return packed;
}

template <typename OutT>
torch::Tensor _pack_atom_scale_b_impl(
    torch::Tensor natural,
    torch::ScalarType out_torch_dtype)
{
    TORCH_CHECK(natural.is_cuda(), "natural must be cuda");
    TORCH_CHECK(natural.dim() == 2, "natural must be 2-D [N, K/128]");
    TORCH_CHECK(natural.dtype() == out_torch_dtype,
                "natural dtype must match expected output dtype");
    (void) out_torch_dtype;
    // [N, G] -> [G, N] row-major; the kernel reads gmem as flat b16 with
    // byte offset 2*(g*N + n). No element permutation needed.
    return natural.transpose(0, 1).contiguous();
}

}  // namespace


torch::Tensor pack_atom_scale_a_fp16(torch::Tensor natural) {
    return _pack_atom_scale_a_impl<__half>(natural, torch::kFloat16);
}

torch::Tensor pack_atom_scale_a_bf16(torch::Tensor natural) {
    return _pack_atom_scale_a_impl<__nv_bfloat16>(natural, torch::kBFloat16);
}

torch::Tensor pack_atom_scale_b_fp16(torch::Tensor natural) {
    return _pack_atom_scale_b_impl<__half>(natural, torch::kFloat16);
}

torch::Tensor pack_atom_scale_b_bf16(torch::Tensor natural) {
    return _pack_atom_scale_b_impl<__nv_bfloat16>(natural, torch::kBFloat16);
}
