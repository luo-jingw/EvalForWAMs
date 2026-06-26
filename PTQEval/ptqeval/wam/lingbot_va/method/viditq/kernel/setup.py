# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Build script for qwan_extension: BF16-native int8/int4 GEMM kernels.

Build in-place:
    cd lingbot-va/wan_va/quant/viditq/kernel
    pip install -e .
"""
import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CSRC = os.path.join(THIS_DIR, "csrc")
CSRC_INFRA = os.path.join(CSRC, "infra")


def _sources() -> list[str]:
    return [
        os.path.join(CSRC, "pybind.cpp"),
        os.path.join(CSRC, "act_quant_bf16.cu"),
        os.path.join(CSRC, "toy_mma_int8.cu"),
        os.path.join(CSRC, "w8a8", "w8a8_gemm.cu"),
        os.path.join(CSRC, "w4a8", "w4a8_gemm.cu"),
        os.path.join(CSRC, "w4a4", "w4a4_gemm.cu"),
        os.path.join(CSRC, "w4a4", "scale_layout.cu"),
    ]


# Default = multi-arch fatbinary covering the two GPUs we run on (A6000
# sm_86, L40 / L40S sm_89). One .so works on either host without per-
# machine edits; build is ~2x slower but only happens once per machine.
# Honor TORCH_CUDA_ARCH_LIST when explicitly set (e.g. "9.0" for H100):
# torch.utils.cpp_extension's CUDAExtension reads that env var and emits
# -gencode for each entry automatically, so we skip our own list.
_TARGET_SMS_DEFAULT = ("86", "89")


def _nvcc_arch_flags() -> list[str]:
    if os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return []
    return [f"-gencode=arch=compute_{sm},code=sm_{sm}"
            for sm in _TARGET_SMS_DEFAULT]


setup(
    name="qwan_extension",
    version="0.1.0",
    description="BF16-native int8/int4 GEMM kernels for LingBot-VA.",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="qwan_extension._C",
            sources=_sources(),
            include_dirs=[CSRC, CSRC_INFRA],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "--use_fast_math",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                    *_nvcc_arch_flags(),
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
