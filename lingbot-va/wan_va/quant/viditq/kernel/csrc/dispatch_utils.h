// Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
#pragma once

#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#define QWAN_CUDA_CHECK(stmt)                                            \
    do {                                                                 \
        cudaError_t _err = (stmt);                                       \
        TORCH_CHECK(_err == cudaSuccess,                                 \
                    "CUDA error: ", cudaGetErrorString(_err));           \
    } while (0)
