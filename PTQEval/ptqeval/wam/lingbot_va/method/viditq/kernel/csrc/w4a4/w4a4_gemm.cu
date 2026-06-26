// Phase 42 commit 1: W4A4 sym per-group INT4xINT4 GEMM, fp16 output.
//
// Ported from ViDiT-Q/kernels/csrc/qgemm/w4a4/atom.cu (which itself
// carries a "// copied from Atom" header). Keeper outlier path stripped
// at port time (G2 in plan.txt Phase 42): KEEPER macros,
// mma_calculate_keeper, A_keeper/B_keeper params, and the entire second-
// pass keeper compute block are removed. The remaining main path is the
// per-group symmetric INT4xINT4 mma kernel only.
//
// Per-group symmetric quantization: group=128 along K-dim.
// Dequant: accu += c_frag * scale_a * scale_b (no zp / sum terms).
// See plan.txt Phase 42 G5 for the yaml-vs-kernel quant-scheme decision.

#include <assert.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdio.h>
#include <random>
#include <torch/extension.h>
#include "dtype_traits.cuh"

// Accumulator: 128 * 128 * sizeof(int32_t) = 64KB
// Block A + B: 128 * 128 * sizeof(int8_t) * 0.5 * 2 = 16KB
#define BLOCK_M 128
#define BLOCK_N 128
#define BLOCK_K 128

#define BLOCK_WARPS 8
#define BLOCK_ROW_WARPS 4
#define BLOCK_COL_WARPS 2

#define WARP_ROW_TILES 4
#define WARP_COL_TILES 4

#define M 16
#define N 8
#define K 64

#define STAGE 4

#define WARP_SIZE 32

// Quantization configuration
#define GROUP_SIZE 128
// Note: Packed into half2 and copy 4 times -> (M_GLOBAL / 2) * 4
#define SCALE_PACKING_A(x) ((x) * 2)
#define SCALE_PACKING_B(x) ((x) / 2)

// This is only for calculating A's scales number of half2 in unit test
// Calculated by the layout of A Scale matrix
#define SCALE_SIZE_A(x) ((x) / 16 * 32 + 32 - (1 - (x % 16) / 8) * (8 - (x % 8)) * 4)

// 16 Bytes = 128 bits = 32 * sizeof(u4) -> actually per row
// Chunk means per row loading
typedef int4 copy_t;
#define CHUNK_LOAD_BYTES (BLOCK_K * sizeof(int8_t) / 2)
#define CHUNK_LOAD_LANES_PER (CHUNK_LOAD_BYTES / sizeof(copy_t))
#define CHUNK_LOAD_PER_WARP (WARP_SIZE / CHUNK_LOAD_LANES_PER)

#define E2S(x) ((x) >> 1)

// Load BLOCK_M * BLOCK_K elements from global memory to shared memory
__device__ __forceinline__ void loadASMem(
  uint8_t *smem,
  const uint8_t *gmem,
  const int max_m_dimension,
  const int gmem_ldm,
  const int k,
  bool predGuard
){
  const int warpId = threadIdx.y + threadIdx.z * blockDim.y;
  const int laneId = threadIdx.x;
  int gmem_row = warpId * BLOCK_M / BLOCK_WARPS + laneId / CHUNK_LOAD_LANES_PER;
  int gmem_col = laneId % CHUNK_LOAD_LANES_PER;
  int smem_row = gmem_row;
  int smem_col = gmem_col ^ ((smem_row / 2) & 3);

  predGuard = predGuard && ((gmem_row + blockIdx.y * BLOCK_M) < max_m_dimension);
  predGuard = predGuard && ((k + gmem_col * 2 * sizeof(copy_t)) < gmem_ldm);

#pragma unroll
  for(int i = 0; i < BLOCK_M / BLOCK_WARPS / CHUNK_LOAD_PER_WARP; ++i){
    asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %0, 0;\n"
      "@!p st.shared.v4.u32 [%1], {0, 0, 0, 0};\n"
      "@p cp.async.cg.shared.global [%1], [%2], 16;\n"
      "}\n"
      ::
        "r"((int) predGuard),
        "l"(__cvta_generic_to_shared((void*)smem) + E2S(smem_row * BLOCK_K) + sizeof(copy_t) * smem_col),
        "l"((copy_t*)(&gmem[E2S(gmem_row * gmem_ldm)]) + gmem_col)
    );
    gmem_row += CHUNK_LOAD_PER_WARP;
    smem_row += CHUNK_LOAD_PER_WARP;
    predGuard = predGuard && ((gmem_row + blockIdx.y * BLOCK_M) < max_m_dimension);
  }
}

__device__ __forceinline__ void loadBSMem(
  uint8_t *smem,
  const uint8_t *gmem,
  const int gmem_ldm,
  const int k,
  bool predGuard
){
  const int warpId = threadIdx.y + threadIdx.z * blockDim.y;
  const int laneId = threadIdx.x;
  int gmem_row = warpId * BLOCK_N / BLOCK_WARPS + laneId / CHUNK_LOAD_LANES_PER;
  int gmem_col = laneId % CHUNK_LOAD_LANES_PER;
  int smem_row = gmem_row;
  int smem_col = gmem_col ^ ((smem_row / 2) & 3);

  predGuard = predGuard && ((k + gmem_col * 2 * sizeof(copy_t)) < gmem_ldm);

#pragma unroll
  for(int i = 0; i < BLOCK_N / BLOCK_WARPS / CHUNK_LOAD_PER_WARP; ++i){
    asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %0, 0;\n"
      "@!p st.shared.v4.u32 [%1], {0, 0, 0, 0};\n"
      "@p cp.async.cg.shared.global [%1], [%2], 16;\n"
      "}\n"
      ::
        "r"((int) predGuard),
        "l"(__cvta_generic_to_shared((void*)smem) + E2S(smem_row * BLOCK_K) + sizeof(copy_t) * smem_col),
        "l"((copy_t*)(&gmem[E2S(gmem_row * gmem_ldm)]) + gmem_col)
    );
    gmem_row += CHUNK_LOAD_PER_WARP;
    smem_row += CHUNK_LOAD_PER_WARP;
  }
}

// Templated on output dtype (half or __nv_bfloat16). copy_t = int4 = 16 bytes
// = 8 elements for either dtype (both are 2-byte), so the byte-level chunk
// copy is dtype-agnostic; only the gmem/smem typed pointer changes.
template <typename OutT>
__device__ __forceinline__ void storeSMem(
  const OutT *smem,
  OutT *gmem,
  const int smem_ldm,
  const int max_m_dimension,
  const int gmem_ldm
){
  const int warpId = threadIdx.y + threadIdx.z * blockDim.y;
  const int laneId = threadIdx.x;
  int gmem_row = warpId * BLOCK_M / BLOCK_WARPS + laneId / (CHUNK_LOAD_LANES_PER * 4);
  int gmem_col = laneId % (CHUNK_LOAD_LANES_PER * 4);
  int smem_row = gmem_row;
  int smem_col = gmem_col;

#pragma unroll
  for(int i = 0; i < BLOCK_M / BLOCK_WARPS / (CHUNK_LOAD_PER_WARP / 4); ++i){
    if(gmem_row + blockIdx.y * BLOCK_M < max_m_dimension){
      *((copy_t*)(&gmem[gmem_row * gmem_ldm]) + gmem_col) =
        *((copy_t*)(smem + smem_row * smem_ldm) + smem_col);

      gmem_row += (CHUNK_LOAD_PER_WARP / 4);
      smem_row += (CHUNK_LOAD_PER_WARP / 4);
    }
  }
}

__device__ __forceinline__ void loadAFrag(
  int32_t *a_frag,
  const uint8_t *smem,
  const int smem_ldm,
  const int k
){
  const int tid = threadIdx.x;
#pragma unroll
  for(int i = 0;i < WARP_COL_TILES; i += 1){
    int smem_row = i * M + tid % 16;
    int smem_col = (k * 2 + tid / 16) ^ ((smem_row / 2) & 3);
    copy_t *ptr = (copy_t*)(&smem[E2S(smem_row * smem_ldm)]) + smem_col;
    asm volatile(
      "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      :  "=r"(a_frag[i * 4]), "=r"(a_frag[i * 4 + 1]), "=r"(a_frag[i * 4 + 2]), "=r"(a_frag[i * 4 + 3])
      :  "l"(__cvta_generic_to_shared(ptr))
    );
  }
}

__device__ __forceinline__ void loadBFrag(
  int32_t *b_frag,
  const uint8_t *smem,
  const int smem_ldm,
  const int k
){
  const int tid = threadIdx.x;
#pragma unroll
  for(int i = 0;i < WARP_ROW_TILES; i += 2){
    int smem_row = i * N + tid % 16;
    int smem_col = (k * 2 + tid / 16) ^ ((smem_row / 2) & 3);
    copy_t *ptr = (copy_t*)(&smem[E2S(smem_row * smem_ldm)]) + smem_col;
    asm volatile(
      "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      :  "=r"(b_frag[i * 2 + 0]), "=r"(b_frag[i * 2 + 2]), "=r"(b_frag[i * 2 + 1]), "=r"(b_frag[i * 2 + 3])
      :  "l"(__cvta_generic_to_shared(ptr))
    );
  }
}

// Templated on output dtype. Per-fragment float -> OutT cast uses
// DtypeTraits::from_float_rn (half: __float2half_rn; bf16: __float2bfloat16_rn).
template <typename OutT>
__device__ __forceinline__ void storeAccumulator(
  float *c_frag,
  OutT *smem,
  const int smem_ldm
){
  using Traits = DtypeTraits<OutT>;
  const int ti = threadIdx.x % 4;
  const int tj = threadIdx.x / 4;
#pragma unroll
  for(int i = 0;i < WARP_COL_TILES; ++i){
#pragma unroll
    for(int j = 0;j < WARP_ROW_TILES; ++j){
      OutT *ptr = &smem[i * smem_ldm * M + j * N];
      ptr[tj * smem_ldm + ti * 2 + 0] = Traits::from_float_rn(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 0]);
      ptr[tj * smem_ldm + ti * 2 + 1] = Traits::from_float_rn(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 1]);
      ptr[(tj+8) * smem_ldm + ti * 2 + 0] = Traits::from_float_rn(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 2]);
      ptr[(tj+8) * smem_ldm + ti * 2 + 1] = Traits::from_float_rn(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 3]);
    }
  }
}

template <bool initZero>
__device__ __forceinline__ void mma_calculate(
  int32_t* __restrict__ c_frag,
  int32_t* __restrict__ a_frag,
  int32_t* __restrict__ b_frag
){
#pragma unroll
  for(int i = 0;i < WARP_COL_TILES; ++i){
#pragma unroll
    for(int j = 0;j < WARP_ROW_TILES; ++j){
      if constexpr (initZero){
        asm volatile(
          "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10,  %11,  %12,  %13};\n"
          : "=r"(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 0]), "=r"(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 1]),
            "=r"(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 2]), "=r"(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 3])
          : "r"(a_frag[i * 4]), "r"(a_frag[i * 4 + 1]), "r"(a_frag[i * 4 + 2]), "r"(a_frag[i * 4 + 3]),
            "r"(b_frag[j * 2]), "r"(b_frag[j * 2 + 1]),
            "r"(0), "r"(0),
            "r"(0), "r"(0)
        );
      }else{
        asm volatile(
          "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10,  %11,  %12,  %13};\n"
          : "=r"(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 0]), "=r"(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 1]),
            "=r"(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 2]), "=r"(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 3])
          : "r"(a_frag[i * 4]), "r"(a_frag[i * 4 + 1]), "r"(a_frag[i * 4 + 2]), "r"(a_frag[i * 4 + 3]),
            "r"(b_frag[j * 2]), "r"(b_frag[j * 2 + 1]),
            "r"(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 0]), "r"(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 1]),
            "r"(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 2]), "r"(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 3])
        );
      }
    }
  }
}

__device__ __forceinline__ void loadScale(
  uint8_t *smem_A_scale,
  uint8_t *smem_B_scale,
  const uint8_t *gmem_A_scale,
  const uint8_t *gmem_B_scale,
  const int max_m_dimension,
  bool predGuard
){
  const int laneId = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
  constexpr int copySingleSizeA = SCALE_PACKING_A(BLOCK_M) * sizeof(half2) * BLOCK_K / GROUP_SIZE;
  constexpr int neededLanesA = copySingleSizeA / sizeof(copy_t);
  constexpr int copySingleSizeB = SCALE_PACKING_B(BLOCK_N) * sizeof(half2) * BLOCK_K / GROUP_SIZE;
  constexpr int neededLanesB = copySingleSizeB / sizeof(copy_t);

  predGuard = predGuard && (laneId < neededLanesA + neededLanesB);
  bool M_tail_check = ((laneId / 8) * 16 + (laneId % 8) + blockIdx.y * BLOCK_M < max_m_dimension);
  predGuard = predGuard && (M_tail_check || laneId >= neededLanesA);

  if(laneId < neededLanesA){
    copy_t *dst_ptr = (copy_t*) smem_A_scale + laneId;
    copy_t *src_ptr = (copy_t*) gmem_A_scale + laneId;
    asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %0, 0;\n"
      "@p cp.async.cg.shared.global [%1], [%2], 16;\n"
      "}\n"
      ::
        "r"((int) predGuard),
        "l"(__cvta_generic_to_shared(dst_ptr)),
        "l"(src_ptr)
    );
  }else{
    copy_t *dst_ptr = (copy_t*) smem_B_scale + (laneId - neededLanesA);
    copy_t *src_ptr = (copy_t*) gmem_B_scale + (laneId - neededLanesA);
    asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %0, 0;\n"
      "@p cp.async.cg.shared.global [%1], [%2], 16;\n"
      "}\n"
      ::
        "r"((int) predGuard),
        "l"(__cvta_generic_to_shared(dst_ptr)),
        "l"(src_ptr)
    );
  }
}

__device__ __forceinline__ void loadScaleReg(
  int32_t *reg_a,
  int32_t *reg_b,
  const uint8_t *smem_A_scale,
  const uint8_t *smem_B_scale
){
  const int tid = threadIdx.x;
  copy_t *ptr = (copy_t *)smem_A_scale + tid;
  asm volatile(
    "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
    :  "=r"(reg_a[0]), "=r"(reg_a[1]), "=r"(reg_a[2]), "=r"(reg_a[3])
    :  "l"(__cvta_generic_to_shared(ptr))
  );
  ptr = (copy_t *)smem_B_scale + tid / 8;
  asm volatile(
    "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
    :  "=r"(reg_b[0]), "=r"(reg_b[1]), "=r"(reg_b[2]), "=r"(reg_b[3])
    :  "l"(__cvta_generic_to_shared(ptr))
  );
}

// Templated on scale dtype (= output dtype). Scales are stored as raw bytes
// in gmem/smem and reinterpreted here as Packed2 (half2 / __nv_bfloat162).
// CUDA's __hmul2 has overloads for both __half2 and __nv_bfloat162 since
// CUDA 11; DtypeTraits::to_float handles per-component cast.
template <typename OutT>
__device__ __forceinline__ void dequant(
  int32_t *c_frag,
  int32_t *reg_a,
  int32_t *reg_b,
  float *accu
){
  using Traits  = DtypeTraits<OutT>;
  using Packed2 = typename Traits::Packed2;
#pragma unroll
  for(int i = 0;i < WARP_COL_TILES; ++i){
    Packed2 row_scale = *(Packed2*)(&reg_a[i]);
#pragma unroll
    for(int j = 0;j < WARP_ROW_TILES; ++j){
      Packed2 col_scale = *(Packed2*)(&reg_b[j]);
      Packed2 rs_scale  = __hmul2(row_scale, col_scale);
      float rs_scale_u = Traits::to_float(rs_scale.x);
      float rs_scale_d = Traits::to_float(rs_scale.y);
      accu[i * WARP_ROW_TILES * 4 + j * 4 + 0] +=
        (c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 0]) * rs_scale_u;
      accu[i * WARP_ROW_TILES * 4 + j * 4 + 1] +=
        (c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 1]) * rs_scale_u;
      accu[i * WARP_ROW_TILES * 4 + j * 4 + 2] +=
        (c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 2]) * rs_scale_d;
      accu[i * WARP_ROW_TILES * 4 + j * 4 + 3] +=
        (c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 3]) * rs_scale_d;
    }
  }
}

template <typename OutT>
__global__ void __launch_bounds__(256) w4a4_compute_gemm_imma(
  const uint8_t *A,
  const uint8_t *B,
  OutT *D,
  const int M_GLOBAL,
  const int N_GLOBAL,
  const int K_GLOBAL,
  const uint8_t *A_scale,
  const uint8_t *B_scale
){
  extern __shared__ uint8_t shmem[];

  const size_t shmem_B_offset = E2S(BLOCK_M * BLOCK_K * sizeof(int8_t));
  const size_t shmem_stage_offset = E2S(BLOCK_K * (BLOCK_M + BLOCK_N) * sizeof(int8_t));
  const size_t shmem_scale_offset = STAGE * shmem_stage_offset;
  const size_t shmem_scale_B_offset = SCALE_PACKING_A(BLOCK_M) * BLOCK_K / GROUP_SIZE * sizeof(half2);
  const size_t shmem_scale_stage_offset = (SCALE_PACKING_A(BLOCK_M) + SCALE_PACKING_B(BLOCK_N)) * BLOCK_K / GROUP_SIZE * sizeof(half2);
  const int A_scale_stride = SCALE_SIZE_A(M_GLOBAL);

  const int bi = blockIdx.x;
  const int bj = blockIdx.y;
  const int wi = threadIdx.y;
  const int wj = threadIdx.z;

  int32_t c[WARP_COL_TILES * WARP_ROW_TILES * 4] = {0};
  int32_t a[2][WARP_COL_TILES * 4] = {0};
  int32_t b[2][WARP_ROW_TILES * 2] = {0};
  float c_fp[WARP_COL_TILES * WARP_ROW_TILES * 4] = {0.0f};
  int32_t a_s[WARP_COL_TILES] = {0};
  int32_t b_s[WARP_ROW_TILES] = {0};

  size_t writePtr = STAGE - 1;
#pragma unroll
  for(int i = 0; i < STAGE - 1;++i){
    loadASMem(
      shmem + i * shmem_stage_offset,
      A + E2S(bj * BLOCK_M * K_GLOBAL + i * BLOCK_K),
      M_GLOBAL,
      K_GLOBAL,
      (i * BLOCK_K),
      true
    );
    loadBSMem(
      shmem + shmem_B_offset + i * shmem_stage_offset,
      B + E2S(bi * BLOCK_N * K_GLOBAL + i * BLOCK_K),
      K_GLOBAL,
      (i * BLOCK_K),
      true
    );
    loadScale(
      shmem + shmem_scale_offset + i * shmem_scale_stage_offset,
      shmem + shmem_scale_offset + shmem_scale_B_offset + i * shmem_scale_stage_offset,
      A_scale + sizeof(half2) * ((i * BLOCK_K) / GROUP_SIZE * A_scale_stride + SCALE_PACKING_A(bj * BLOCK_M)),
      B_scale + sizeof(half2) * SCALE_PACKING_B((i * BLOCK_K) / GROUP_SIZE * N_GLOBAL + bi * BLOCK_N),
      M_GLOBAL,
      true
    );
    asm volatile("cp.async.commit_group;\n" ::);
  }
  asm volatile("cp.async.wait_group %0;\n" ::"n"(STAGE - 2));
  __syncthreads();

  loadAFrag(
    a[0],
    shmem + E2S(wj * WARP_COL_TILES * M * BLOCK_K) + (writePtr + 1) % STAGE * shmem_stage_offset,
    BLOCK_K,
    0
  );
  loadBFrag(
    b[0],
    shmem + shmem_B_offset + E2S(wi * WARP_ROW_TILES * N * BLOCK_K) + (writePtr + 1) % STAGE * shmem_stage_offset,
    BLOCK_K,
    0
  );
  for(int k = 0; k < K_GLOBAL; k += BLOCK_K){
    loadAFrag(
      a[1],
      shmem + E2S(wj * WARP_COL_TILES * M * BLOCK_K) + (writePtr + 1) % STAGE * shmem_stage_offset,
      BLOCK_K,
      1
    );
    loadBFrag(
      b[1],
      shmem + shmem_B_offset + E2S(wi * WARP_ROW_TILES * N * BLOCK_K) + (writePtr + 1) % STAGE * shmem_stage_offset,
      BLOCK_K,
      1
    );
    loadScaleReg(
      a_s,
      b_s,
      shmem + shmem_scale_offset + (writePtr + 1) % STAGE * shmem_scale_stage_offset + sizeof(half2) * SCALE_PACKING_A(wj * WARP_COL_TILES * M),
      shmem + shmem_scale_offset + shmem_scale_B_offset + (writePtr + 1) % STAGE * shmem_scale_stage_offset + sizeof(half2) * SCALE_PACKING_B(wi * WARP_ROW_TILES * N)
    );
    mma_calculate<true>(c, a[0], b[0]);
    bool predGuard = (k + (STAGE - 1) * BLOCK_K) < K_GLOBAL;
    loadASMem(
      shmem + writePtr * shmem_stage_offset,
      A + E2S(bj * BLOCK_M * K_GLOBAL + k + (STAGE - 1) * BLOCK_K),
      M_GLOBAL,
      K_GLOBAL,
      (k + (STAGE - 1) * BLOCK_K),
      predGuard
    );
    loadBSMem(
      shmem + shmem_B_offset + writePtr * shmem_stage_offset,
      B + E2S(bi * BLOCK_N * K_GLOBAL + k + (STAGE - 1) * BLOCK_K),
      K_GLOBAL,
      (k + (STAGE - 1) * BLOCK_K),
      predGuard
    );
    loadScale(
      shmem + shmem_scale_offset + writePtr * shmem_scale_stage_offset,
      shmem + shmem_scale_offset + shmem_scale_B_offset + writePtr * shmem_scale_stage_offset,
      A_scale + sizeof(half2) * ((k + (STAGE - 1) * BLOCK_K) / GROUP_SIZE * A_scale_stride + SCALE_PACKING_A(bj * BLOCK_M)),
      B_scale + sizeof(half2) * SCALE_PACKING_B((k + (STAGE - 1) * BLOCK_K) / GROUP_SIZE * N_GLOBAL + bi * BLOCK_N),
      M_GLOBAL,
      predGuard
    );
    asm volatile("cp.async.commit_group;\n" ::);
    mma_calculate<false>(c, a[1], b[1]);
    asm volatile("cp.async.wait_group %0;\n" ::"n"(STAGE - 2));
    writePtr = (writePtr + 1) % STAGE;
    __syncthreads();
    loadAFrag(
      a[0],
      shmem + E2S(wj * WARP_COL_TILES * M * BLOCK_K) + (writePtr + 1) % STAGE * shmem_stage_offset,
      BLOCK_K,
      0
    );
    loadBFrag(
      b[0],
      shmem + shmem_B_offset + E2S(wi * WARP_ROW_TILES * N * BLOCK_K) + (writePtr + 1) % STAGE * shmem_stage_offset,
      BLOCK_K,
      0
    );
    dequant<OutT>(
      c,
      a_s,
      b_s,
      c_fp
    );
  }

  storeAccumulator<OutT>(
    c_fp,
    (OutT *)shmem + wj * WARP_COL_TILES * M * BLOCK_N + wi * WARP_ROW_TILES * N,
    BLOCK_N
  );
  __syncthreads();

  storeSMem<OutT>(
    (OutT *)shmem,
    D + bj * BLOCK_M * N_GLOBAL + bi * BLOCK_N,
    BLOCK_N,
    M_GLOBAL,
    N_GLOBAL
  );
}

/*!
 * \brief Dense W4A4 per-group symmetric GEMM (keeper stripped, sym).
 *   Templated on output dtype OutT in {half, __nv_bfloat16}.
 *   Scales are stored as raw bytes (2 bytes per element) in the same dtype
 *   as OutT — caller passes the .data_ptr() reinterpreted as uint8.
 * \param A INT4 matrix in global memory. Packed in uint8_t. [M, K/2] row-major.
 * \param B INT4 matrix in global memory. Packed in uint8_t. [K, N] column-major
 *          (equiv [N, K/2] row-major in PyTorch storage).
 * \param A_scale per-group scale for A. [M, K/128] with Atom-permuted layout.
 * \param B_scale per-group scale for B. [K/128, N] with Atom-permuted layout.
 * \param D Output matrix in global memory. OutT [M, N] row-major.
 */
template <typename OutT>
static void w4a4_launch_dense_layer_gemm(
  const uint8_t *A,
  const uint8_t *B,
  const uint8_t *A_scale,
  const uint8_t *B_scale,
  OutT *D,
  const size_t M_GLOBAL,
  const size_t N_GLOBAL,
  const size_t K_GLOBAL
){
  dim3 gridDim(
    (N_GLOBAL + BLOCK_N - 1) / BLOCK_N,
    (M_GLOBAL + BLOCK_M - 1) / BLOCK_M
  );
  dim3 blockDim(
    WARP_SIZE,
    BLOCK_ROW_WARPS,
    BLOCK_COL_WARPS
  );

  // The scale-block half2 sizing was derived for fp16; the same byte budget
  // covers bf16 since __nv_bfloat162 and __half2 are both 4 bytes.
  constexpr size_t shmem_size1 = sizeof(uint8_t) * BLOCK_K * (BLOCK_M + BLOCK_N) / 2 * STAGE +
      sizeof(half2) * (SCALE_PACKING_A(BLOCK_M) + SCALE_PACKING_B(BLOCK_N)) * STAGE;
  constexpr size_t shmem_size2 = BLOCK_M * BLOCK_N * sizeof(OutT);
  constexpr size_t SHMEM_SZ = shmem_size1 > shmem_size2 ? shmem_size1 : shmem_size2;

  cudaFuncSetAttribute(
    w4a4_compute_gemm_imma<OutT>,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    SHMEM_SZ
  );

  w4a4_compute_gemm_imma<OutT><<<gridDim, blockDim, SHMEM_SZ>>>(
    A, B, D,
    M_GLOBAL, N_GLOBAL, K_GLOBAL,
    A_scale, B_scale
  );
}


// ============================================================================
// Torch entry point: w4a4_of16_nobias_weight_sym
// ============================================================================

// Shape + alignment checks shared between fp16 + bf16 entry points.
// out_torch_dtype is torch::kFloat16 or torch::kBFloat16.
template <typename OutT>
static torch::Tensor _w4a4_nobias_weight_sym_impl(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor scale_input,
    torch::Tensor scale_weight,
    torch::ScalarType out_torch_dtype
){
    TORCH_CHECK(input.is_cuda(),  "input must be cuda");
    TORCH_CHECK(weight.is_cuda(), "weight must be cuda");
    TORCH_CHECK(scale_input.is_cuda(),  "scale_input must be cuda");
    TORCH_CHECK(scale_weight.is_cuda(), "scale_weight must be cuda");
    TORCH_CHECK(input.dtype()  == torch::kUInt8, "input must be uint8 (packed int4)");
    TORCH_CHECK(weight.dtype() == torch::kUInt8, "weight must be uint8 (packed int4)");
    TORCH_CHECK(scale_input.dtype()  == out_torch_dtype,  "scale_input dtype must match output dtype");
    TORCH_CHECK(scale_weight.dtype() == out_torch_dtype,  "scale_weight dtype must match output dtype");
    TORCH_CHECK(input.dim()  == 2, "input must be 2-D [M, K/2]");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2-D [N, K/2]");

    // NB: M/N/K are macro-defined as mma tile constants above (#define M 16
    // etc.), so the torch entry must use different identifiers.
    const int64_t M_global = input.size(0);
    const int64_t K_half   = input.size(1);
    const int64_t K_global = K_half * 2;
    const int64_t N_global = weight.size(0);

    TORCH_CHECK(weight.size(1) == K_half, "input/weight K mismatch");
    TORCH_CHECK(M_global % BLOCK_M == 0, "M must be multiple of BLOCK_M (128)");
    TORCH_CHECK(N_global % BLOCK_N == 0, "N must be multiple of BLOCK_N (128)");
    TORCH_CHECK(K_global % BLOCK_K == 0, "K must be multiple of BLOCK_K (128)");

    auto opts = torch::TensorOptions()
                    .dtype(out_torch_dtype)
                    .device(input.device());
    torch::Tensor output = torch::empty({M_global, N_global}, opts);

    w4a4_launch_dense_layer_gemm<OutT>(
        reinterpret_cast<const uint8_t*>(input.data_ptr()),
        reinterpret_cast<const uint8_t*>(weight.data_ptr()),
        reinterpret_cast<const uint8_t*>(scale_input.data_ptr()),
        reinterpret_cast<const uint8_t*>(scale_weight.data_ptr()),
        reinterpret_cast<OutT*>(output.data_ptr()),
        static_cast<size_t>(M_global),
        static_cast<size_t>(N_global),
        static_cast<size_t>(K_global)
    );
    return output;
}


torch::Tensor w4a4_of16_nobias_weight_sym(
    torch::Tensor input,        // uint8 [M, K/2]  packed int4 row-major
    torch::Tensor weight,       // uint8 [N, K/2]  packed int4 row-major
    torch::Tensor scale_input,  // fp16  [M, K/128] Atom-layout
    torch::Tensor scale_weight  // fp16  [K/128, N] Atom-layout
) {
    return _w4a4_nobias_weight_sym_impl<half>(
        input, weight, scale_input, scale_weight, torch::kFloat16);
}


torch::Tensor w4a4_obf16_nobias_weight_sym(
    torch::Tensor input,        // uint8 [M, K/2]  packed int4 row-major
    torch::Tensor weight,       // uint8 [N, K/2]  packed int4 row-major
    torch::Tensor scale_input,  // bf16  [M, K/128] Atom-layout
    torch::Tensor scale_weight  // bf16  [K/128, N] Atom-layout
) {
    return _w4a4_nobias_weight_sym_impl<__nv_bfloat16>(
        input, weight, scale_input, scale_weight, torch::kBFloat16);
}
