// DSV4 Lightning-Indexer prefill logits on sm_120a native FP4 tensor cores.
//
//   out[s, t] = sum_h w[s, h] * relu(q[s, h, :] . k[t, :])      (-inf where t >= live(s))
//
// q / k are FP4 round-trips (e2m1 x 2^e per 32 dims): stored packed (2 codes per byte, low nibble
// = even dim) + one ue8m0 exponent byte per 32 dims, so mma kind::mxf4 block_scale scale_vec::2X
// (m16n8k64) consumes them directly: no dequant, no CUDA-core scaling, 1/4 of bf16's bytes.
// Products are exact; the tensor core's internal accumulation and the head-sum order differ
// from the Triton bf16 kernel (owner 11-18: low-precision differences acceptable).
//
// CTA: QS queries x a range of keys, 4 warps. q group lives in smem (rows padded to 80 B: the
// 8 rows a fragment load touches hit distinct banks). Keys stream in 128-key tiles (cp.async,
// double-buffered); warp w owns keys [32w, 32w+32) of a tile: its B fragments are loaded once
// per tile and reused for every query of the group. Per query: 4 m16 head tiles x 4 n8 key tiles
// x 2 k64 steps = 32 MMAs, then relu*w FMA over the thread's 8 heads and 3 xor-shuffles.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>

#define H 64
#define ROWB 64      // packed bytes per 128-dim row
#define LDB 80       // padded smem row

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp16(void* dst, const void* src, bool pred) {
  asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %2, 0;\n @p cp.async.cg.shared.global [%0], [%1], 16;\n @!p st.shared.v4.u32 [%0], {0,0,0,0};\n}\n"
               :: "r"(smem_u32(dst)), "l"(src), "r"((int)pred));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

__device__ __forceinline__ void mma_fp4(float* d, const uint32_t* a, uint32_t b0, uint32_t b1,
                                        uint32_t sfa, uint32_t sfb) {
  const uint16_t z = 0;
  asm volatile(
      "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{%11,%12},{%13},{%14,%15};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1),
        "r"(sfa), "h"(z), "h"(z), "r"(sfb), "h"(z), "h"(z));
}

template <int QS, int NW, typename OutT>
__global__ void __launch_bounds__(NW * 32) idx_fp4_kernel(
    const uint8_t* __restrict__ qp, const uint8_t* __restrict__ qe,   // [S, H, 64], [S, H, 4]
    const uint8_t* __restrict__ kp, const uint8_t* __restrict__ ke,   // [T, 64], [T, 4]
    const __nv_bfloat16* __restrict__ w, int stride_ws,               // [S, H]
    OutT* __restrict__ out, long stride_os,                           // [S, T] fp32 or bf16
    int S, int T, int keys_per_cta, int start_pos, int ratio, int do_mask) {
  constexpr int BT = NW * 32;
  extern __shared__ __align__(16) uint8_t smem[];
  uint8_t* sq = smem;                                   // QS*H rows x LDB
  uint8_t* sqe = sq + QS * H * LDB;                     // QS*H x 4
  uint8_t* sk = sqe + QS * H * 4;                       // 2 x BT rows x LDB
  uint8_t* ske = sk + 2 * BT * LDB;                     // 2 x BT x 4

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int s0 = blockIdx.x * QS;
  const int t_begin = blockIdx.y * keys_per_cta;
  const int t_end = min(T, t_begin + keys_per_cta);
  if (t_begin >= t_end) return;

  // q group -> smem (cp.async 16 B chunks), exponents
  for (int c = tid; c < QS * H * 4; c += NW * 32) {          // 4 x 16 B per row
    int row = c >> 2, part = c & 3;
    int s = s0 + row / H;
    cp16(sq + row * LDB + part * 16, qp + ((long)min(s, S - 1) * H + row % H) * ROWB + part * 16, s < S);
  }
  for (int c = tid; c < QS * H; c += NW * 32) {
    int s = s0 + c / H;
    *(uint32_t*)(sqe + c * 4) = s < S ? *(const uint32_t*)(qe + ((long)s * H + c % H) * 4) : 0x7f7f7f7fu;
  }
  auto load_k = [&](int buf, int tb) {
    for (int c = tid; c < BT * 4; c += NW * 32) {
      int row = c >> 2, part = c & 3, key = tb + row;
      cp16(sk + (buf * BT + row) * LDB + part * 16, kp + (long)min(key, T - 1) * ROWB + part * 16, key < T);
    }
    for (int c = tid; c < BT / 4; c += NW * 32) {          // 4 keys' exponents per 16 B
      int key = tb + c * 4;
      if (key + 3 < T || key >= T) {
        cp16(ske + (buf * BT + c * 4) * 4, ke + (long)min(key, T - 4) * 4, key + 3 < T);
      } else {   // ragged tail (T % 4 != 0): per key (a 16 B copy would zero the last keys' scales)
        for (int j = 0; j < 4; ++j)
          *(uint32_t*)(ske + (buf * BT + c * 4 + j) * 4) = key + j < T ? *(const uint32_t*)(ke + (long)(key + j) * 4) : 0u;
      }
    }
    cp_commit();
  };
  load_k(0, t_begin);

  int buf = 0;
  for (int tb = t_begin; tb < t_end; tb += BT, buf ^= 1) {
    if (tb + BT < t_end) { load_k(buf ^ 1, tb + BT); cp_wait<1>(); } else { cp_wait<0>(); }
    __syncthreads();
    // B fragments for this warp's 32 keys: 4 n8 tiles x 2 k64 steps
    uint32_t b[4][2][2], sfb[4][2];
    const uint8_t* kb = sk + (buf * BT + warp * 32) * LDB;
    const uint8_t* kbe = ske + (buf * BT + warp * 32) * 4;
#pragma unroll
    for (int n = 0; n < 4; ++n) {
#pragma unroll
      for (int ks = 0; ks < 2; ++ks) {
        const uint8_t* r = kb + (n * 8 + g) * LDB + ks * 32 + t * 4;
        b[n][ks][0] = *(const uint32_t*)r;
        b[n][ks][1] = *(const uint32_t*)(r + 16);
        const uint8_t* e = kbe + (n * 8 + g) * 4 + ks * 2;
        sfb[n][ks] = (uint32_t)e[0] | ((uint32_t)e[1] << 8);
      }
    }
    const int ra = (lane >> 2) + 8 * (lane & 1);         // scale-factor row a thread supplies
    for (int qi = 0; qi < QS; ++qi) {
      const int s = s0 + qi;
      if (s >= S) break;
      float part[4][2];
#pragma unroll
      for (int n = 0; n < 4; ++n) part[n][0] = part[n][1] = 0.f;
#pragma unroll
      for (int m = 0; m < 4; ++m) {
        float acc[4][4];
#pragma unroll
        for (int n = 0; n < 4; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f;
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
          const uint8_t* r0 = sq + (qi * H + m * 16 + g) * LDB + ks * 32 + t * 4;
          const uint8_t* r1 = r0 + 8 * LDB;
          uint32_t a[4] = {*(const uint32_t*)r0, *(const uint32_t*)r1,
                           *(const uint32_t*)(r0 + 16), *(const uint32_t*)(r1 + 16)};
          const uint8_t* e = sqe + (qi * H + m * 16 + ra) * 4 + ks * 2;
          uint32_t sfa = (uint32_t)e[0] | ((uint32_t)e[1] << 8);
#pragma unroll
          for (int n = 0; n < 4; ++n) mma_fp4(acc[n], a, b[n][ks][0], b[n][ks][1], sfa, sfb[n][ks]);
        }
        const float w0 = __bfloat162float(w[(long)s * stride_ws + m * 16 + g]);
        const float w1 = __bfloat162float(w[(long)s * stride_ws + m * 16 + g + 8]);
#pragma unroll
        for (int n = 0; n < 4; ++n) {
          part[n][0] = fmaf(fmaxf(acc[n][0], 0.f), w0, part[n][0]);
          part[n][1] = fmaf(fmaxf(acc[n][1], 0.f), w0, part[n][1]);
          part[n][0] = fmaf(fmaxf(acc[n][2], 0.f), w1, part[n][0]);
          part[n][1] = fmaf(fmaxf(acc[n][3], 0.f), w1, part[n][1]);
        }
      }
      // reduce over the 8 groups (lanes differing in g)
#pragma unroll
      for (int n = 0; n < 4; ++n) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          float v = part[n][j];
          v += __shfl_xor_sync(0xffffffffu, v, 4);
          v += __shfl_xor_sync(0xffffffffu, v, 8);
          v += __shfl_xor_sync(0xffffffffu, v, 16);
          part[n][j] = v;
        }
      }
      if (g == 0) {
        const int live = do_mask ? (start_pos + s + 1) / ratio : 0x7fffffff;
#pragma unroll
        for (int n = 0; n < 4; ++n) {
          const int key = tb + warp * 32 + n * 8 + 2 * t;
          float v0 = key >= live ? -INFINITY : part[n][0];
          float v1 = key + 1 >= live ? -INFINITY : part[n][1];
          OutT* o = out + (long)s * stride_os + key;
          if constexpr (sizeof(OutT) == 4) {
            if (key + 1 < t_end) { *(float2*)o = make_float2(v0, v1); }
            else if (key < t_end) { o[0] = v0; }
          } else {
            if (key + 1 < t_end) { *(__nv_bfloat162*)o = __floats2bfloat162_rn(v0, v1); }
            else if (key < t_end) { o[0] = __float2bfloat16(v0); }
          }
        }
      }
    }
    __syncthreads();
  }
}

template <int QS, int NW, typename OutT>
static int launch(const void* qp, const void* qe, const void* kp, const void* ke, const void* w, int stride_ws,
                  void* out, long stride_os, int S, int T, int start_pos, int ratio, int do_mask,
                  int keys_per_cta, cudaStream_t st) {
  constexpr int BT = NW * 32;
  dim3 grid((S + QS - 1) / QS, (T + keys_per_cta - 1) / keys_per_cta);
  size_t sm = QS * H * LDB + QS * H * 4 + 2 * BT * LDB + 2 * BT * 4;
  cudaFuncSetAttribute(idx_fp4_kernel<QS, NW, OutT>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
  idx_fp4_kernel<QS, NW, OutT><<<grid, NW * 32, sm, st>>>((const uint8_t*)qp, (const uint8_t*)qe, (const uint8_t*)kp,
      (const uint8_t*)ke, (const __nv_bfloat16*)w, stride_ws, (OutT*)out, stride_os, S, T,
      keys_per_cta, start_pos, ratio, do_mask);
  return (int)cudaGetLastError();
}


// ---- transposed variant (11-18): keys = A rows (16 per m-tile, in registers per key tile),
// q heads = B columns (8 per n-tile, from smem per query). C rows = keys, cols = heads, so the
// head sum runs mostly in-thread (16 heads per lane) + 2 xor-shuffles (was 3 over 8 values each),
// and every lane writes one key (one coalesced 64 B store per warp and query).
template <int QS, int NW, typename OutT>
__global__ void __launch_bounds__(NW * 32) idx_fp4t_kernel(
    const uint8_t* __restrict__ qp, const uint8_t* __restrict__ qe,
    const uint8_t* __restrict__ kp, const uint8_t* __restrict__ ke,
    const __nv_bfloat16* __restrict__ w, int stride_ws,
    OutT* __restrict__ out, long stride_os,
    int S, int T, int keys_per_cta, int start_pos, int ratio, int do_mask) {
  constexpr int BT = NW * 32;
  extern __shared__ __align__(16) uint8_t smem[];
  uint8_t* sq = smem;                                   // QS*H rows x LDB
  uint8_t* sqe = sq + QS * H * LDB;                     // QS*H x 4
  float* sw = (float*)(sqe + QS * H * 4);               // QS*H fp32
  uint8_t* sk = (uint8_t*)(sw + QS * H);                // 2 x BT rows x LDB
  uint8_t* ske = sk + 2 * BT * LDB;                     // 2 x BT x 4

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int s0 = blockIdx.x * QS;
  const int t_begin = blockIdx.y * keys_per_cta;
  const int t_end = min(T, t_begin + keys_per_cta);
  if (t_begin >= t_end) return;

  for (int c = tid; c < QS * H * 4; c += NW * 32) {
    int row = c >> 2, part = c & 3;
    int s = s0 + row / H;
    cp16(sq + row * LDB + part * 16, qp + ((long)min(s, S - 1) * H + row % H) * ROWB + part * 16, s < S);
  }
  for (int c = tid; c < QS * H; c += NW * 32) {
    int s = s0 + c / H;
    *(uint32_t*)(sqe + c * 4) = s < S ? *(const uint32_t*)(qe + ((long)s * H + c % H) * 4) : 0x7f7f7f7fu;
    sw[c] = s < S ? __bfloat162float(w[(long)s * stride_ws + c % H]) : 0.f;
  }
  auto load_k = [&](int buf, int tb) {
    for (int c = tid; c < BT * 4; c += NW * 32) {
      int row = c >> 2, part = c & 3, key = tb + row;
      cp16(sk + (buf * BT + row) * LDB + part * 16, kp + (long)min(key, T - 1) * ROWB + part * 16, key < T);
    }
    for (int c = tid; c < BT / 4; c += NW * 32) {
      int key = tb + c * 4;
      if (key + 3 < T || key >= T) {
        cp16(ske + (buf * BT + c * 4) * 4, ke + (long)min(key, T - 4) * 4, key + 3 < T);
      } else {   // ragged tail (T % 4 != 0): per key (a 16 B copy would zero the last keys' scales)
        for (int j = 0; j < 4; ++j)
          *(uint32_t*)(ske + (buf * BT + c * 4 + j) * 4) = key + j < T ? *(const uint32_t*)(ke + (long)(key + j) * 4) : 0u;
      }
    }
    cp_commit();
  };
  load_k(0, t_begin);

  const int ra = (lane >> 2) + 8 * (lane & 1);         // scale-factor row a thread supplies
  // the key this lane writes after the quad reduction: (m, r) = (t >> 1, t & 1)
  const int my_key_off = (t >> 1) * 16 + g + 8 * (t & 1);
  int buf = 0;
  for (int tb = t_begin; tb < t_end; tb += BT, buf ^= 1) {
    if (tb + BT < t_end) { load_k(buf ^ 1, tb + BT); cp_wait<1>(); } else { cp_wait<0>(); }
    __syncthreads();
    uint32_t a[2][2][4], sfa[2][2];
    const uint8_t* kb = sk + (buf * BT + warp * 32) * LDB;
    const uint8_t* kbe = ske + (buf * BT + warp * 32) * 4;
#pragma unroll
    for (int m = 0; m < 2; ++m) {
      const uint32_t e4 = *(const uint32_t*)(kbe + (m * 16 + ra) * 4);
#pragma unroll
      for (int ks = 0; ks < 2; ++ks) {
        const uint8_t* r0 = kb + (m * 16 + g) * LDB + ks * 32 + t * 4;
        const uint8_t* r1 = r0 + 8 * LDB;
        a[m][ks][0] = *(const uint32_t*)r0;
        a[m][ks][1] = *(const uint32_t*)r1;
        a[m][ks][2] = *(const uint32_t*)(r0 + 16);
        a[m][ks][3] = *(const uint32_t*)(r1 + 16);
        sfa[m][ks] = (e4 >> (16 * ks)) & 0xffffu;
      }
    }
    const int key = tb + warp * 32 + my_key_off;
    for (int qi = 0; qi < QS; ++qi) {
      const int s = s0 + qi;
      float part[2][2] = {{0.f, 0.f}, {0.f, 0.f}};
#pragma unroll
      for (int n = 0; n < 8; ++n) {
        const uint8_t* rb = sq + (qi * H + n * 8 + g) * LDB + t * 4;
        const uint32_t e4 = *(const uint32_t*)(sqe + (qi * H + n * 8 + g) * 4);
        const float2 wv = *(const float2*)(sw + qi * H + n * 8 + 2 * t);
        float acc[2][4];
#pragma unroll
        for (int m = 0; m < 2; ++m) acc[m][0] = acc[m][1] = acc[m][2] = acc[m][3] = 0.f;
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
          const uint32_t b0 = *(const uint32_t*)(rb + ks * 32), b1 = *(const uint32_t*)(rb + ks * 32 + 16);
          const uint32_t sfb = (e4 >> (16 * ks)) & 0xffffu;
#pragma unroll
          for (int m = 0; m < 2; ++m) mma_fp4(acc[m], a[m][ks], b0, b1, sfa[m][ks], sfb);
        }
#pragma unroll
        for (int m = 0; m < 2; ++m) {
          part[m][0] = fmaf(fmaxf(acc[m][0], 0.f), wv.x, part[m][0]);
          part[m][0] = fmaf(fmaxf(acc[m][1], 0.f), wv.y, part[m][0]);
          part[m][1] = fmaf(fmaxf(acc[m][2], 0.f), wv.x, part[m][1]);
          part[m][1] = fmaf(fmaxf(acc[m][3], 0.f), wv.y, part[m][1]);
        }
      }
#pragma unroll
      for (int m = 0; m < 2; ++m)
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          float v = part[m][r];
          v += __shfl_xor_sync(0xffffffffu, v, 1);
          v += __shfl_xor_sync(0xffffffffu, v, 2);
          part[m][r] = v;
        }
      float v = (t & 2) ? ((t & 1) ? part[1][1] : part[1][0]) : ((t & 1) ? part[0][1] : part[0][0]);
      if (s < S && key < t_end) {
        const int live = do_mask ? (start_pos + s + 1) / ratio : 0x7fffffff;
        if (key >= live) v = -INFINITY;
        out[(long)s * stride_os + key] = (OutT)v;
      }
    }
    __syncthreads();
  }
}

template <int QS, int NW, typename OutT>
static int launch_t(const void* qp, const void* qe, const void* kp, const void* ke, const void* w, int stride_ws,
                    void* out, long stride_os, int S, int T, int start_pos, int ratio, int do_mask,
                    int keys_per_cta, cudaStream_t st) {
  constexpr int BT = NW * 32;
  dim3 grid((S + QS - 1) / QS, (T + keys_per_cta - 1) / keys_per_cta);
  size_t sm = QS * H * LDB + QS * H * 4 + QS * H * 4 + 2 * BT * LDB + 2 * BT * 4;
  cudaFuncSetAttribute(idx_fp4t_kernel<QS, NW, OutT>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
  idx_fp4t_kernel<QS, NW, OutT><<<grid, NW * 32, sm, st>>>((const uint8_t*)qp, (const uint8_t*)qe, (const uint8_t*)kp,
      (const uint8_t*)ke, (const __nv_bfloat16*)w, stride_ws, (OutT*)out, stride_os, S, T,
      keys_per_cta, start_pos, ratio, do_mask);
  return (int)cudaGetLastError();
}

extern "C" int dsv4_idx_fp4(const void* qp, const void* qe, const void* kp, const void* ke,
                            const void* w, int stride_ws, void* out, long stride_os,
                            int S, int T, int start_pos, int ratio, int do_mask, int qs,
                            int keys_per_cta, void* stream) {
  cudaStream_t st = (cudaStream_t)stream;
  int bf = qs / 10000; qs %= 10000;           // qs argument encodes BF16*10000 + TR*1000 + NW*100 + QS
  int tr = qs / 1000; qs %= 1000;
  int nw = qs / 100; qs = qs % 100;
  if (nw == 0) nw = 4;
#define L(Q, N) if (qs == Q && nw == N) return bf ? launch<Q, N, __nv_bfloat16>(qp, qe, kp, ke, w, stride_ws, out, stride_os, S, T, start_pos, ratio, do_mask, keys_per_cta, st) : launch<Q, N, float>(qp, qe, kp, ke, w, stride_ws, out, stride_os, S, T, start_pos, ratio, do_mask, keys_per_cta, st);
#define LT(Q, N) if (qs == Q && nw == N) return bf ? launch_t<Q, N, __nv_bfloat16>(qp, qe, kp, ke, w, stride_ws, out, stride_os, S, T, start_pos, ratio, do_mask, keys_per_cta, st) : launch_t<Q, N, float>(qp, qe, kp, ke, w, stride_ws, out, stride_os, S, T, start_pos, ratio, do_mask, keys_per_cta, st);
  if (tr) {
    LT(4, 4) LT(8, 4) LT(4, 8) LT(8, 8) LT(16, 4) LT(16, 8)
    return -1;
  }
  L(4, 4) L(8, 4) L(4, 8) L(8, 8) L(2, 8) L(4, 16) L(2, 16)
#undef L
#undef LT
  return -1;
}

// ---- decode (11-18): one query row per CTA row, keys gathered off the decode full-loc snapshot
// (block b -> compressed row snap[row, b * ratio] / ratio) from the FP4-packed pool (codes [R, 64],
// exps [R, 4]); q's 64 heads stay in registers as B fragments, keys are the A rows (32 per warp,
// 2 m-tiles). Columns >= valid[row] get -inf; a CTA past the live count only writes -inf
// (graph-safe: the staged width is static, the live count is read from device memory).
#define DEC_NW 8
#define DEC_BT (DEC_NW * 32)
__global__ void __launch_bounds__(DEC_NW * 32) idx_fp4_decode_kernel(
    const uint8_t* __restrict__ qp, const uint8_t* __restrict__ qe,      // [B, H, 64], [B, H, 4]
    const uint8_t* __restrict__ kc, const uint8_t* __restrict__ ke,      // pool codes / exps
    const __nv_bfloat16* __restrict__ w, int stride_wb,                  // [B, H]
    const int64_t* __restrict__ snap, long stride_sb,                     // [B, W] full locs
    const int* __restrict__ valid,                                        // [B]
    __nv_bfloat16* __restrict__ out, long stride_ob,                      // [B, n_stage]
    int n_stage, int ratio, int keys_per_cta) {
  __shared__ __align__(16) uint8_t sk[2][DEC_BT * LDB];
  __shared__ __align__(16) uint32_t ske[2][DEC_BT];
  const int b = blockIdx.x;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int t_begin = blockIdx.y * keys_per_cta;
  const int t_end = min(n_stage, t_begin + keys_per_cta);
  const int nv = min(valid[b], n_stage);
  __nv_bfloat16* orow = out + (long)b * stride_ob;
  if (t_begin >= nv) {
    for (int c = t_begin + tid; c < t_end; c += DEC_NW * 32) orow[c] = __float2bfloat16(-INFINITY);
    return;
  }
  const int t_live = min(t_end, nv);
  // q B fragments: head n*8+g, bytes ks*32 + t*4 (+16); scales head n*8+g
  uint32_t qb[8][2][2], qs4[8];
  float wv[8][2];
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    const uint8_t* r = qp + ((long)b * H + n * 8 + g) * ROWB + t * 4;
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      qb[n][ks][0] = *(const uint32_t*)(r + ks * 32);
      qb[n][ks][1] = *(const uint32_t*)(r + ks * 32 + 16);
    }
    qs4[n] = *(const uint32_t*)(qe + ((long)b * H + n * 8 + g) * 4);
    wv[n][0] = __bfloat162float(w[(long)b * stride_wb + n * 8 + 2 * t]);
    wv[n][1] = __bfloat162float(w[(long)b * stride_wb + n * 8 + 2 * t + 1]);
  }
  const int64_t* srow = snap + (long)b * stride_sb;
  auto load_k = [&](int buf, int tb) {
    // thread -> key row (tid), its 4 x 16 B code chunks + 4 B exponents
    const int key = tb + tid;
    const bool ok = key < t_live;
    long r = 0;
    if (ok) { long f = srow[(long)key * ratio]; r = (f < 0 ? 0 : f) / ratio; }
#pragma unroll
    for (int part = 0; part < 4; ++part)
      cp16(sk[buf] + tid * LDB + part * 16, kc + r * ROWB + part * 16, ok);
    ske[buf][tid] = ok ? *(const uint32_t*)(ke + r * 4) : 0x7f7f7f7fu;
    cp_commit();
  };
  const int ra = (lane >> 2) + 8 * (lane & 1);
  const int my_key_off = (t >> 1) * 16 + g + 8 * (t & 1);
  load_k(0, t_begin);
  int buf = 0;
  for (int tb = t_begin; tb < t_end; tb += DEC_BT, buf ^= 1) {
    if (tb + DEC_BT < t_end) { load_k(buf ^ 1, tb + DEC_BT); cp_wait<1>(); } else { cp_wait<0>(); }
    __syncthreads();
    const uint8_t* kb = sk[buf] + warp * 32 * LDB;
    float part[2][2] = {{0.f, 0.f}, {0.f, 0.f}};
#pragma unroll
    for (int m = 0; m < 2; ++m) {
      uint32_t a[2][4];
      const uint32_t e4 = ske[buf][warp * 32 + m * 16 + ra];
#pragma unroll
      for (int ks = 0; ks < 2; ++ks) {
        const uint8_t* r0 = kb + (m * 16 + g) * LDB + ks * 32 + t * 4;
        a[ks][0] = *(const uint32_t*)r0;
        a[ks][1] = *(const uint32_t*)(r0 + 8 * LDB);
        a[ks][2] = *(const uint32_t*)(r0 + 16);
        a[ks][3] = *(const uint32_t*)(r0 + 8 * LDB + 16);
      }
#pragma unroll
      for (int n = 0; n < 8; ++n) {
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int ks = 0; ks < 2; ++ks)
          mma_fp4(acc, a[ks], qb[n][ks][0], qb[n][ks][1], (e4 >> (16 * ks)) & 0xffffu, (qs4[n] >> (16 * ks)) & 0xffffu);
        part[m][0] = fmaf(fmaxf(acc[0], 0.f), wv[n][0], part[m][0]);
        part[m][0] = fmaf(fmaxf(acc[1], 0.f), wv[n][1], part[m][0]);
        part[m][1] = fmaf(fmaxf(acc[2], 0.f), wv[n][0], part[m][1]);
        part[m][1] = fmaf(fmaxf(acc[3], 0.f), wv[n][1], part[m][1]);
      }
    }
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        float v = part[m][r];
        v += __shfl_xor_sync(0xffffffffu, v, 1);
        v += __shfl_xor_sync(0xffffffffu, v, 2);
        part[m][r] = v;
      }
    float v = (t & 2) ? ((t & 1) ? part[1][1] : part[1][0]) : ((t & 1) ? part[0][1] : part[0][0]);
    const int key = tb + warp * 32 + my_key_off;
    if (key < t_end) orow[key] = __float2bfloat16(key < t_live ? v : -INFINITY);
    __syncthreads();
  }
}

extern "C" int dsv4_idx_fp4_decode(const void* qp, const void* qe, const void* kc, const void* ke,
                                   const void* w, int stride_wb, const void* snap, long stride_sb,
                                   const void* valid, void* out, long stride_ob, int B, int n_stage,
                                   int ratio, int keys_per_cta, void* stream) {
  if (B == 0 || n_stage == 0) return 0;
  dim3 grid(B, (n_stage + keys_per_cta - 1) / keys_per_cta);
  idx_fp4_decode_kernel<<<grid, DEC_NW * 32, 0, (cudaStream_t)stream>>>(
      (const uint8_t*)qp, (const uint8_t*)qe, (const uint8_t*)kc, (const uint8_t*)ke,
      (const __nv_bfloat16*)w, stride_wb, (const int64_t*)snap, stride_sb, (const int*)valid,
      (__nv_bfloat16*)out, stride_ob, n_stage, ratio, keys_per_cta);
  return (int)cudaGetLastError();
}
