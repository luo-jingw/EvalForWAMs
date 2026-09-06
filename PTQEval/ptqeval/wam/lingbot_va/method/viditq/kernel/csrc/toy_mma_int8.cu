// Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
//
// Phase 24a toy kernel: single-CTA, single-warp m16n8k32 s8s8s32 MMA.
//   A: int8  [M=16, K=32] row-major
//   B: int8  [N= 8, K=32] row-major  (MMA view: [K=32, N=8] col-major)
//   C: int32 [M=16, N= 8] row-major
//
// Pipeline:
//   1. cp_async stage gA, gB into smemA (512 B), smemB (256 B)
//   2. ldmatrix.x4 distributes A across 4 register fragments per lane
//   3. manual byte-pack of B per PTX m16n8k32 lane->data layout
//   4. mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
//   5. direct gmem writeback per PTX C output layout
//
// Correctness-only. Not a building block for the W8A8 GEMM (Phase 24b
// uses share_to_reg_B + permuted_smem from ViDiT-Q gemm_utils).

#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>

#include "infra/cp_async.cuh"
#include "infra/mma.cuh"


namespace {

__global__ void toy_mma_int8_kernel(
    const int8_t* __restrict__ gA,
    const int8_t* __restrict__ gB,
    int32_t* __restrict__ gC
) {
    __shared__ __align__(16) int8_t smemA[16 * 32];
    __shared__ __align__(16) int8_t smemB[ 8 * 32];

    const int lane = threadIdx.x;

    cp_async::load_128b<cp_async::PrefetchMode::kNoPrefetch>(
        reinterpret_cast<uint4*>(smemA) + lane,
        reinterpret_cast<const uint4*>(gA) + lane);
    if (lane < 16) {
        cp_async::load_128b<cp_async::PrefetchMode::kNoPrefetch>(
            reinterpret_cast<uint4*>(smemB) + lane,
            reinterpret_cast<const uint4*>(gB) + lane);
    }
    cp_async::commit_group();
    cp_async::wait_group<0>();
    __syncwarp();

    // A: ldmatrix.x4 address per lane.
    //   lanes 0-7   -> matrix 0 (rows 0-7,  cols 0-15)  -> RA[0]
    //   lanes 8-15  -> matrix 1 (rows 8-15, cols 0-15)  -> RA[1]
    //   lanes 16-23 -> matrix 2 (rows 0-7,  cols 16-31) -> RA[2]
    //   lanes 24-31 -> matrix 3 (rows 8-15, cols 16-31) -> RA[3]
    int8_t* a_ptr = &smemA[(lane % 16) * 32 + (lane / 16) * 16];
    uint32_t RA[4];
    mma::ldmatrix_m8n8x4(RA, a_ptr);

    // B: manual byte-pack per PTX m16n8k32 B layout.
    //   lane t holds 8 int8 in two uint32:
    //     RB[0]: N=t/4, K=(t%4)*4 + 0..3
    //     RB[1]: N=t/4, K=(t%4)*4 + 16..19
    //   smemB is [N=8, K=32] row-major -> smemB[n*32 + k].
    const int n_idx  = lane >> 2;
    const int k_base = (lane & 3) << 2;
    uint32_t RB[2];
    RB[0] = *reinterpret_cast<const uint32_t*>(&smemB[n_idx * 32 + k_base + 0]);
    RB[1] = *reinterpret_cast<const uint32_t*>(&smemB[n_idx * 32 + k_base + 16]);

    int32_t RC[4] = {0, 0, 0, 0};
    mma::mma_sync_m16n8k32_row_col_s8s8s32<mma::MMAMode::kInit>(RC, RA, RB);

    // C: PTX m16n8k32 output layout per lane t:
    //   RC[0]: C[t/4,     2*(t%4) + 0]
    //   RC[1]: C[t/4,     2*(t%4) + 1]
    //   RC[2]: C[t/4 + 8, 2*(t%4) + 0]
    //   RC[3]: C[t/4 + 8, 2*(t%4) + 1]
    const int c_row0 = lane >> 2;
    const int c_col0 = (lane & 3) << 1;
    gC[(c_row0    ) * 8 + c_col0 + 0] = RC[0];
    gC[(c_row0    ) * 8 + c_col0 + 1] = RC[1];
    gC[(c_row0 + 8) * 8 + c_col0 + 0] = RC[2];
    gC[(c_row0 + 8) * 8 + c_col0 + 1] = RC[3];
}

}  // anonymous namespace


void toy_mma_int8_gemm(
    torch::Tensor a,  // int8  [16, 32]
    torch::Tensor b,  // int8  [ 8, 32]
    torch::Tensor c   // int32 [16,  8]
) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && c.is_cuda(),
                "toy_mma_int8_gemm: all tensors must be CUDA");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && c.is_contiguous(),
                "toy_mma_int8_gemm: all tensors must be contiguous");
    TORCH_CHECK(a.scalar_type() == torch::kInt8,  "a must be int8");
    TORCH_CHECK(b.scalar_type() == torch::kInt8,  "b must be int8");
    TORCH_CHECK(c.scalar_type() == torch::kInt32, "c must be int32");
    TORCH_CHECK(a.dim() == 2 && a.size(0) == 16 && a.size(1) == 32,
                "a must be shape [16, 32]");
    TORCH_CHECK(b.dim() == 2 && b.size(0) ==  8 && b.size(1) == 32,
                "b must be shape [ 8, 32]");
    TORCH_CHECK(c.dim() == 2 && c.size(0) == 16 && c.size(1) ==  8,
                "c must be shape [16,  8]");

    auto stream = at::cuda::getCurrentCUDAStream();
    toy_mma_int8_kernel<<<1, 32, 0, stream>>>(
        a.data_ptr<int8_t>(),
        b.data_ptr<int8_t>(),
        c.data_ptr<int32_t>());
}
