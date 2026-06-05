// Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
//
// Templated quantized GEMM kernel:
//   y_bf16[N, M] = ( (x_int8[N, K] @ w_int_t.T) * scale_x[N] * scale_w[M] )
//                  + (bias_bf16[M] if present)
//
// WBITS = 8: w stored as int8 [M, K]
// WBITS = 4: w stored as packed int8 [M, K/2]; each byte holds two signed
//            int4 values. Low nibble -> column index 2*c+0, high nibble ->
//            column index 2*c+1. Both nibbles are sign-extended at load.
//
// Tile layout (block-level):
//   one CUDA block computes a TILE_N x TILE_M output tile.
//   thread layout inside the block: (TILE_M, TILE_N) with one thread per
//   output element. K dimension is processed in TILE_K-sized chunks with
//   per-chunk shared-memory staging.

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>


template <int WBITS, int TILE_M, int TILE_N, int TILE_K>
__global__ void quant_gemm_bf16_kernel(
    const int8_t* __restrict__ x,           // [N, K] int8
    const int8_t* __restrict__ w,           // [M, K] (WBITS==8) or [M, K/2] (WBITS==4)
    const __nv_bfloat16* __restrict__ scale_x,   // [N] bf16
    const __nv_bfloat16* __restrict__ scale_w,   // [M] bf16
    const __nv_bfloat16* __restrict__ bias,      // [M] bf16 or nullptr
    __nv_bfloat16* __restrict__ y,          // [N, M] bf16
    int N, int M, int K
) {
    static_assert(WBITS == 8 || WBITS == 4, "WBITS must be 8 or 4");
    static_assert(TILE_K % 2 == 0, "TILE_K must be even for W4 packing");

    __shared__ int8_t x_smem[TILE_N * TILE_K];   // [TILE_N, TILE_K]
    __shared__ int8_t w_smem[TILE_M * TILE_K];   // [TILE_M, TILE_K] (unpacked)

    const int block_m = blockIdx.x * TILE_M;
    const int block_n = blockIdx.y * TILE_N;
    const int tid = threadIdx.x;
    const int tid_m = tid % TILE_M;
    const int tid_n = tid / TILE_M;

    const int global_m = block_m + tid_m;
    const int global_n = block_n + tid_n;

    int32_t acc = 0;

    const int K_aligned = ((K + TILE_K - 1) / TILE_K) * TILE_K;

    for (int k0 = 0; k0 < K_aligned; k0 += TILE_K) {
        // ---- Load X tile : [TILE_N, TILE_K] int8 elements ----
        #pragma unroll
        for (int i = tid; i < TILE_N * TILE_K; i += TILE_M * TILE_N) {
            int row = i / TILE_K;
            int col = i % TILE_K;
            int gx_n = block_n + row;
            int gx_k = k0 + col;
            x_smem[i] = (gx_n < N && gx_k < K)
                            ? x[gx_n * K + gx_k]
                            : (int8_t)0;
        }

        // ---- Load W tile : [TILE_M, TILE_K] int8 elements (unpacked) ----
        if constexpr (WBITS == 8) {
            #pragma unroll
            for (int i = tid; i < TILE_M * TILE_K; i += TILE_M * TILE_N) {
                int row = i / TILE_K;
                int col = i % TILE_K;
                int gw_m = block_m + row;
                int gw_k = k0 + col;
                w_smem[i] = (gw_m < M && gw_k < K)
                                ? w[gw_m * K + gw_k]
                                : (int8_t)0;
            }
        } else {
            // WBITS == 4: each global byte yields two int4 values.
            // Iterate over packed bytes in this K-tile.
            constexpr int PACK_PER_ROW = TILE_K / 2;
            const int K_pack = K / 2;
            #pragma unroll
            for (int i = tid; i < TILE_M * PACK_PER_ROW; i += TILE_M * TILE_N) {
                int row = i / PACK_PER_ROW;
                int col_byte = i % PACK_PER_ROW;
                int gw_m = block_m + row;
                int gw_kpack = (k0 / 2) + col_byte;
                int8_t byte = (gw_m < M && gw_kpack < K_pack)
                                  ? w[gw_m * K_pack + gw_kpack]
                                  : (int8_t)0;
                // Extract signed int4 from low nibble (bits 0..3).
                // Cannot use ((byte << 4) >> 4) on int8 because int
                // promotion sign-extends byte before shifting, so the
                // high bits of the promoted int are not cleared.
                // Use XOR-subtract sign-extend: (x ^ 0x8) - 0x8 maps
                // {0..7}->{0..7} and {8..15}->{-8..-1}.
                int lo_u = static_cast<int>(byte) & 0xF;
                int8_t low  = static_cast<int8_t>((lo_u ^ 0x8) - 0x8);
                // High nibble: arithmetic right shift of the promoted
                // int already sign-extends bit 7 into bits 4..7.
                int8_t high = static_cast<int8_t>(static_cast<int>(byte) >> 4);
                int base_col = col_byte * 2;
                w_smem[row * TILE_K + base_col + 0] = low;
                w_smem[row * TILE_K + base_col + 1] = high;
            }
        }

        __syncthreads();

        if (global_m < M && global_n < N) {
            const int8_t* x_ptr = &x_smem[tid_n * TILE_K];
            const int8_t* w_ptr = &w_smem[tid_m * TILE_K];
            #pragma unroll
            for (int kk = 0; kk < TILE_K; ++kk) {
                acc += (int32_t)x_ptr[kk] * (int32_t)w_ptr[kk];
            }
        }

        __syncthreads();
    }

    if (global_m < M && global_n < N) {
        float sx = __bfloat162float(scale_x[global_n]);
        float sw = __bfloat162float(scale_w[global_m]);
        float result = static_cast<float>(acc) * sx * sw;
        if (bias != nullptr) {
            result += __bfloat162float(bias[global_m]);
        }
        y[global_n * M + global_m] = __float2bfloat16(result);
    }
}
