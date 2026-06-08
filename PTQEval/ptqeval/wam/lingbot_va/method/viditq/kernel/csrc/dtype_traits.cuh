// Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
//
// Q11 DtypeTraits: type-aware abstraction over __half and __nv_bfloat16
// intrinsics. Lets a single templated CUDA kernel target both dtypes
// without semantic change (P2: type-aware refactor without semantic
// change is permitted).
//
// Usage in kernels:
//   template <typename DType, ...>
//   __global__ void Kernel(const DType* input, ...) {
//     using Traits  = DtypeTraits<DType>;
//     using Packed2 = typename Traits::Packed2;          // half2 or bf162
//     Packed2 x;
//     float f = Traits::to_float(x.x);
//     DType  h = Traits::from_float_rn(f);
//     DType  a = Traits::habs(h);
//     ...
//   }
//
// Phase 24c adds: Packed2, to_float, from_float_rn, habs (act_quant path).
// Phase 25  adds: from_float2_rn for the GEMM epilogue.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>


template <typename DType>
struct DtypeTraits;


template <>
struct DtypeTraits<__half> {
    using Packed2 = __half2;

    static __device__ __forceinline__ float to_float(__half v) {
        return __half2float(v);
    }

    static __device__ __forceinline__ __half from_float_rn(float v) {
        return __float2half_rn(v);
    }

    static __device__ __forceinline__ __half habs(__half v) {
        return __habs(v);
    }
};


template <>
struct DtypeTraits<__nv_bfloat16> {
    using Packed2 = __nv_bfloat162;

    static __device__ __forceinline__ float to_float(__nv_bfloat16 v) {
        return __bfloat162float(v);
    }

    static __device__ __forceinline__ __nv_bfloat16 from_float_rn(float v) {
        return __float2bfloat16_rn(v);
    }

    static __device__ __forceinline__ __nv_bfloat16 habs(__nv_bfloat16 v) {
        return __habs(v);
    }
};
