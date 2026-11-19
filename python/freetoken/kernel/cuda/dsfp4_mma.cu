// Grouped DS-FP4 MoE GEMM on sm_120a native block-scaled tensor cores.
//
//   out[route, n] = sum_k A[row(route), k] * W[expert, n, k]      (x tw[route] for the down proj)
//
// A: FP8 e4m3 codes [Ra, K] with one e8m0 exponent per 128 values (act_quant_fp8); W: DS-FP4 =
// MXFP4 (e2m1 packed 2/byte, low nibble = even k) with one e8m0 per 32 values. The MMA is
// mma.sync.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32 e4m3 x e2m1 -> f32: the products are
// the same exact values the bf16-dequant kernels multiply (q * 2^e is exact either way); only the
// fp32 accumulation order differs. The block-scaled MMA's internal accumulation is short of full
// fp32 (over K=4096 ~4x cuBLAS fp32's error), so each 128-K stage accumulates into a fresh
// fragment that is then added to an fp32 register accumulator (more accurate than cuBLAS fp32).
// Routes come pre-sorted by expert in 64-row blocks
// (moe_align_block_size); padding ids are >= R. A row = a_map[route] (token-level FP8 input,
// gate_up) or route itself (per-route input, down).
//
// Tile: CTA BM (routes, 64 or 128: -DBM) x 128 (n), BM/32 x 2 warps of 32x64, K in 128-steps,
// STAGES-deep cp.async; A exponents staged once per CTA. BM=128 (11-18): one M block per expert at
// prefill sizes, so each weight tile is read once (1.16x on PRO 4000, bit-identical per route).
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

#ifndef BM
#define BM 64
#endif
#define NTH (BM * 2)        // 2 x (BM / 32) warps of 32 x 64
#define BN 128
#define BK 128
#ifndef STAGES
#define STAGES 3
#endif
#define A_LD 144  // bytes per A smem row: 128 + 16 pad (conflict-free fragment loads)
#define B_LD 80   // bytes per B smem row: 64 packed + 16 pad
#define A_BYTES (BM * A_LD)
#define B_BYTES (BN * B_LD)
#define STAGE_BYTES (A_BYTES + B_BYTES + BM + BN * 4)

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp16(void* dst, const void* src, int src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)), "l"(src), "r"(src_bytes));
}
__device__ __forceinline__ void cp4(void* dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(smem_u32(dst)), "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// 4 packed e2m1 nibbles (low nibble = lower k) -> 4 bytes with the code in bits [5:2]
__device__ __forceinline__ uint32_t expand_fp4x4(uint32_t v) {
  uint32_t y = (v & 0xFu) | ((v & 0xF0u) << 4) | ((v & 0xF00u) << 8) | ((v & 0xF000u) << 12);
  return y << 2;
}

__device__ __forceinline__ void mma_fp8xfp4(float* d, const uint32_t* a, uint32_t b0, uint32_t b1,
                                            uint32_t sfa, uint32_t sfb) {
  const uint16_t z = 0;
  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e2m1.f32.ue8m0 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{%11,%12},{%13},{%14,%15};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1),
        "r"(sfa), "h"(z), "h"(z), "r"(sfb), "h"(z), "h"(z));
}

__global__ void __launch_bounds__(NTH) dsfp4_mma_grouped_kernel(
    const uint8_t* __restrict__ aq, const uint8_t* __restrict__ asc, const int32_t* __restrict__ a_map,
    const uint8_t* __restrict__ w, const uint8_t* __restrict__ wsc,
    const int32_t* __restrict__ sorted_ids, const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ ntpp, const float* __restrict__ tw, __nv_bfloat16* __restrict__ out,
    int R, int N, int K, int ldo) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int mb = blockIdx.x, nb = blockIdx.y;
  if (mb * BM >= ntpp[0]) return;
  const int e = expert_ids[mb];
  if (e < 0) return;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, tig = lane & 3;
  const int wm = warp >> 1, wn = warp & 1;   // BM/32 x 2 warps
  const int n0 = nb * BN;
  const int KB = K / BK, KS = K / 128, KW = K / 2, KSC = K / 32;

  constexpr int AI = BM * 8 / NTH;    // = 4
  const uint8_t* a_src[AI];
  int a_bytes[AI];
#pragma unroll
  for (int i = 0; i < AI; ++i) {
    const int row = (tid >> 3) + (NTH / 8) * i;
    const int r = sorted_ids[mb * BM + row];
    const bool ok = r < R;
    const int arow = ok ? (a_map ? a_map[r] : r) : 0;
    a_src[i] = aq + (size_t)arow * K + (tid & 7) * 16;
    a_bytes[i] = ok ? 16 : 0;  // padding rows: zero-filled
  }
  // A scales: every row's K/128 exponents -> smem once (cp.async 16 B pieces; KS is 16 or 32)
  uint8_t* Asc_all = smem + STAGES * STAGE_BYTES;
  for (int c = tid; c < BM * (KS / 16); c += NTH) {
    const int row = c / (KS / 16), part = c % (KS / 16);
    const int r = sorted_ids[mb * BM + row];
    const bool ok = r < R;
    const uint8_t* src = asc + (size_t)(ok ? (a_map ? a_map[r] : r) : 0) * KS + part * 16;
    cp16(Asc_all + row * KS + part * 16, src, ok ? 16 : 0);
  }
  // B loads: chunk c = tid + 128 i -> n (tid >> 2) + 32 i, 16-byte column (tid & 3)
  constexpr int BI = BN * 4 / NTH;
  const uint8_t* b_src[BI];
#pragma unroll
  for (int i = 0; i < BI; ++i) {
    const int n = (tid >> 2) + (NTH / 4) * i;
    b_src[i] = w + ((size_t)e * N + n0 + n) * KW + (tid & 3) * 16;
  }
  const uint8_t* bs_src = wsc + ((size_t)e * N + n0 + (tid % BN)) * KSC;  // B scale: thread owns n = tid

  auto load_stage = [&](int s, int kb) {
    uint8_t* st = smem + s * STAGE_BYTES;
    uint8_t* As = st;
    uint8_t* Bs = st + A_BYTES;
    uint8_t* Asc = Bs + B_BYTES;
    uint8_t* Bsc = Asc + BM;
    const int k0 = kb * BK;
#pragma unroll
    for (int i = 0; i < AI; ++i)
      cp16(As + ((tid >> 3) + (NTH / 8) * i) * A_LD + (tid & 7) * 16, a_src[i] + k0, a_bytes[i]);
#pragma unroll
    for (int i = 0; i < BI; ++i)
      cp16(Bs + ((tid >> 2) + (NTH / 4) * i) * B_LD + (tid & 3) * 16, b_src[i] + k0 / 2, 16);
    if (tid < BN) cp4(Bsc + tid * 4, bs_src + kb * 4);

  };

  float acc[2][8][4];
  float part[2][8][4];
#pragma unroll
  for (int mi = 0; mi < 2; ++mi)
#pragma unroll
    for (int ni = 0; ni < 8; ++ni)
#pragma unroll
      for (int j = 0; j < 4; ++j) acc[mi][ni][j] = 0.f;

#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < KB) load_stage(s, s);
    cp_commit();
  }
  for (int kb = 0; kb < KB; ++kb) {
    cp_wait<STAGES - 2>();
    __syncthreads();
    const uint8_t* st = smem + (kb % STAGES) * STAGE_BYTES;
    const uint8_t* As = st;
    const uint8_t* Bs = st + A_BYTES;
    const uint8_t* Asc = Bs + B_BYTES;
    const uint8_t* Bsc = Asc + BM;
    uint32_t sfa[2];
#pragma unroll
    for (int mi = 0; mi < 2; ++mi) sfa[mi] = Asc_all[(wm * 32 + mi * 16 + g + 8 * (lane & 1)) * KS + kb];
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
      for (int ni = 0; ni < 8; ++ni)
#pragma unroll
        for (int j = 0; j < 4; ++j) part[mi][ni][j] = 0.f;
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      uint32_t af[2][4];
#pragma unroll
      for (int mi = 0; mi < 2; ++mi) {
        const uint8_t* r0 = As + (wm * 32 + mi * 16 + g) * A_LD + ks * 32 + tig * 4;
        af[mi][0] = *(const uint32_t*)(r0);
        af[mi][1] = *(const uint32_t*)(r0 + 8 * A_LD);
        af[mi][2] = *(const uint32_t*)(r0 + 16);
        af[mi][3] = *(const uint32_t*)(r0 + 8 * A_LD + 16);
      }
#pragma unroll
      for (int ni = 0; ni < 8; ++ni) {
        const int n = wn * 64 + ni * 8 + g;
        const uint8_t* bp = Bs + n * B_LD + ks * 16 + tig * 2;
        const uint32_t b0 = expand_fp4x4(*(const uint16_t*)(bp));
        const uint32_t b1 = expand_fp4x4(*(const uint16_t*)(bp + 8));
        const uint32_t sfb = Bsc[(wn * 64 + ni * 8 + g) * 4 + ks];
#pragma unroll
        for (int mi = 0; mi < 2; ++mi) mma_fp8xfp4(part[mi][ni], af[mi], b0, b1, sfa[mi], sfb);
      }
    }
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
      for (int ni = 0; ni < 8; ++ni)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[mi][ni][j] += part[mi][ni][j];
    const int nk = kb + STAGES - 1;
    if (nk < KB) load_stage(nk % STAGES, nk);
    cp_commit();
  }
  cp_wait<0>();

  // epilogue: bf16 pairs; padding rows (route >= R) dropped
#pragma unroll
  for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int route = sorted_ids[mb * BM + wm * 32 + mi * 16 + g + 8 * h];
      if (route >= R) continue;
      const float sc = tw ? tw[route] : 1.f;
      __nv_bfloat16* orow = out + (size_t)route * ldo + n0 + wn * 64 + tig * 2;
#pragma unroll
      for (int ni = 0; ni < 8; ++ni) {
        float v0 = acc[mi][ni][2 * h], v1 = acc[mi][ni][2 * h + 1];
        if (tw) { v0 *= sc; v1 *= sc; }
        *(__nv_bfloat162*)(orow + ni * 8) = __floats2bfloat162_rn(v0, v1);
      }
    }
  }
}

extern "C" int dsfp4_mma_grouped(const void* aq, const void* asc, const void* a_map, const void* w,
                                 const void* wsc, const void* sorted_ids, const void* expert_ids,
                                 const void* ntpp, const void* tw, void* out, int EM, int R, int N,
                                 int K, int ldo, void* stream) {
  static int configured[64] = {0};  // the smem opt-in is per device
  const int smem = STAGES * STAGE_BYTES + BM * 32;   // + A scales (K/128 <= 32 per row)
  int dev = 0;
  cudaGetDevice(&dev);
  if (dev < 0 || dev >= 64) return -2;
  if (!configured[dev]) {
    cudaError_t err = cudaFuncSetAttribute(dsfp4_mma_grouped_kernel,
                                           cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (err != cudaSuccess) return (int)err;
    configured[dev] = 1;
  }
  if (N % BN || K % BK || K / 128 > 32 || (K / 128) % 16) return -1;
  dim3 grid((EM + BM - 1) / BM, N / BN);
  dsfp4_mma_grouped_kernel<<<grid, NTH, smem, (cudaStream_t)stream>>>(
      (const uint8_t*)aq, (const uint8_t*)asc, (const int32_t*)a_map, (const uint8_t*)w,
      (const uint8_t*)wsc, (const int32_t*)sorted_ids, (const int32_t*)expert_ids,
      (const int32_t*)ntpp, (const float*)tw, (__nv_bfloat16*)out, R, N, K, ldo);
  return (int)cudaGetLastError();
}
