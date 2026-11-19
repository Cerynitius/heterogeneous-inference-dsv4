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
#define BT 128       // keys per tile

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

template <int QS>
__global__ void __launch_bounds__(128) idx_fp4_kernel(
    const uint8_t* __restrict__ qp, const uint8_t* __restrict__ qe,   // [S, H, 64], [S, H, 4]
    const uint8_t* __restrict__ kp, const uint8_t* __restrict__ ke,   // [T, 64], [T, 4]
    const __nv_bfloat16* __restrict__ w, int stride_ws,               // [S, H]
    float* __restrict__ out, long stride_os,                          // [S, T]
    int S, int T, int keys_per_cta, int start_pos, int ratio, int do_mask) {
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
  for (int c = tid; c < QS * H * 4; c += 128) {          // 4 x 16 B per row
    int row = c >> 2, part = c & 3;
    int s = s0 + row / H;
    cp16(sq + row * LDB + part * 16, qp + ((long)min(s, S - 1) * H + row % H) * ROWB + part * 16, s < S);
  }
  for (int c = tid; c < QS * H; c += 128) {
    int s = s0 + c / H;
    *(uint32_t*)(sqe + c * 4) = s < S ? *(const uint32_t*)(qe + ((long)s * H + c % H) * 4) : 0x7f7f7f7fu;
  }
  auto load_k = [&](int buf, int tb) {
    for (int c = tid; c < BT * 4; c += 128) {
      int row = c >> 2, part = c & 3, key = tb + row;
      cp16(sk + (buf * BT + row) * LDB + part * 16, kp + (long)min(key, T - 1) * ROWB + part * 16, key < T);
    }
    for (int c = tid; c < BT / 4; c += 128) {          // 4 keys' exponents per 16 B
      int key = tb + c * 4;
      cp16(ske + (buf * BT + c * 4) * 4, ke + (long)min(key, T - 4) * 4, key + 3 < T);
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
          float* o = out + (long)s * stride_os + key;
          if (key + 1 < t_end) { *(float2*)o = make_float2(v0, v1); }
          else if (key < t_end) { o[0] = v0; }
        }
      }
    }
    __syncthreads();
  }
}

extern "C" int dsv4_idx_fp4(const void* qp, const void* qe, const void* kp, const void* ke,
                            const void* w, int stride_ws, void* out, long stride_os,
                            int S, int T, int start_pos, int ratio, int do_mask, int qs,
                            int keys_per_cta, void* stream) {
  dim3 grid((S + qs - 1) / qs, (T + keys_per_cta - 1) / keys_per_cta);
  cudaStream_t st = (cudaStream_t)stream;
  if (qs == 8) {
    size_t sm = 8 * H * LDB + 8 * H * 4 + 2 * BT * LDB + 2 * BT * 4;
    cudaFuncSetAttribute(idx_fp4_kernel<8>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    idx_fp4_kernel<8><<<grid, 128, sm, st>>>((const uint8_t*)qp, (const uint8_t*)qe, (const uint8_t*)kp,
        (const uint8_t*)ke, (const __nv_bfloat16*)w, stride_ws, (float*)out, stride_os, S, T,
        keys_per_cta, start_pos, ratio, do_mask);
  } else if (qs == 16) {
    size_t sm = 16 * H * LDB + 16 * H * 4 + 2 * BT * LDB + 2 * BT * 4;
    cudaFuncSetAttribute(idx_fp4_kernel<16>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    idx_fp4_kernel<16><<<grid, 128, sm, st>>>((const uint8_t*)qp, (const uint8_t*)qe, (const uint8_t*)kp,
        (const uint8_t*)ke, (const __nv_bfloat16*)w, stride_ws, (float*)out, stride_os, S, T,
        keys_per_cta, start_pos, ratio, do_mask);
  } else {
    return -1;
  }
  return (int)cudaGetLastError();
}
