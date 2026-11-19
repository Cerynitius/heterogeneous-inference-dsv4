// DSV4 sparse MLA prefill attention for sm_120a (bf16 mma.sync, fp32 accumulate).
// One CTA = one query x all 64 heads (the Triton kernel takes 32 heads per program, so each
// gathered KV tile is reused by 2x the rows). 8 warps: warp w -> head group rg = w & 3
// (16 heads), DV half ch = w >> 2 (256 dims). S = Q K^T: each warp of a head-group pair computes
// the partial dot over its own 256 dims, the pair swaps partials through smem and both add
// them (a + b == b + a, so both warps hold identical S / softmax state). Q fragments live in
// registers, KV tiles of BT keys are gathered with cp.async into a swizzled double buffer.
//   out[q, h, :] = softmax_j(scale * q_h . kv_j  (+ sink_h)) . kv_j  over the valid idx[q, j]
// idx layout [window | compressed]: column j < n_window reads win_pool, else cmp_pool;
// idx < 0 (or j >= n_window + cnt[q]) is masked.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define D 512
#define H 64
#define BT 32
#define NWARP 8
#define ROWB (D * 2)              // bytes per KV row
#define TILEB (BT * ROWB)         // 32 KB

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool pred) {
  int sz = pred ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(sz));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void mma_bf16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
// swizzled byte offset of (row, 16-byte chunk c) inside a [BT, 512] bf16 tile
__device__ __forceinline__ uint32_t swz(int row, int c) {
  return row * ROWB + ((c ^ (row & 7)) << 4);
}

struct Smem {
  uint8_t kv[2][TILEB];
  float xs[4][2][32][16];       // partial S exchange [head group][dv half][lane][16]
  int valid[2][BT];
};

__global__ void __launch_bounds__(NWARP * 32, 1)
sparse_mla_kernel(const __nv_bfloat16* __restrict__ q, long sqm, long sqh,
                  const __nv_bfloat16* __restrict__ win, const __nv_bfloat16* __restrict__ cmp, long skv,
                  const float* __restrict__ sink,
                  const int* __restrict__ idx, long sim, int topk, int n_window,
                  const int* __restrict__ cnt,
                  __nv_bfloat16* __restrict__ out, long som, long soh,
                  float scale_log2) {
  extern __shared__ __align__(128) uint8_t smem_raw[];
  Smem& sm = *reinterpret_cast<Smem*>(smem_raw);
  const int qi = blockIdx.x;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int rg = warp & 3, ch = warp >> 2;
  const int g = lane >> 2, t = lane & 3;

  int n_active = topk;
  if (cnt != nullptr) n_active = min(topk, n_window + cnt[qi]);
  const int n_tiles = (n_active + BT - 1) / BT;
  const int* irow = idx + (long)qi * sim;

  // gather one tile: thread -> row r = tid >> 3, chunks (tid & 7) + 8 i
  auto load_tile = [&](int tile, int buf) {
    const int r = tid >> 3;
    const int col = tile * BT + r;
    int id = col < n_active ? irow[col] : -1;
    const bool ok = id >= 0;
    const __nv_bfloat16* base = (col < n_window ? win : cmp) + (long)(ok ? id : 0) * skv;
    const uint32_t dst = smem_u32(sm.kv[buf]);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int c = (tid & 7) + 8 * i;
      cp_async16(dst + swz(r, c), base + c * 8, ok);
    }
    if ((tid & 7) == 0) sm.valid[buf][r] = ok;
  };

  if (n_tiles > 0) load_tile(0, 0);
  cp_commit();

  // Q fragments: rows rg*16 + g (+8), dims ch*256 + ks*16 + 2t (+1, +8, +9)
  uint32_t qa[16][4];
  {
    const __nv_bfloat16* q0 = q + (long)qi * sqm + (long)(rg * 16 + g) * sqh + ch * 256 + 2 * t;
    const __nv_bfloat16* q1 = q0 + 8 * sqh;
#pragma unroll
    for (int ks = 0; ks < 16; ++ks) {
      qa[ks][0] = *reinterpret_cast<const uint32_t*>(q0 + ks * 16);
      qa[ks][1] = *reinterpret_cast<const uint32_t*>(q1 + ks * 16);
      qa[ks][2] = *reinterpret_cast<const uint32_t*>(q0 + ks * 16 + 8);
      qa[ks][3] = *reinterpret_cast<const uint32_t*>(q1 + ks * 16 + 8);
    }
  }

  float acc[32][4];
#pragma unroll
  for (int j = 0; j < 32; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;

  for (int tile = 0; tile < n_tiles; ++tile) {
    const int buf = tile & 1;
    if (tile + 1 < n_tiles) load_tile(tile + 1, buf ^ 1);
    cp_commit();
    cp_wait<1>();
    __syncthreads();
    const uint32_t kvb = smem_u32(sm.kv[buf]);

    // ---- partial S over this warp's 256 dims: 4 n-tiles (8 keys each)
    float s[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j) s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.f;
#pragma unroll
    for (int ks = 0; ks < 16; ++ks) {
      const int c0 = ch * 32 + ks * 2;   // 16-byte chunk of dims ks*16 within this half
#pragma unroll
      for (int np = 0; np < 2; ++np) {   // n-tile pair (keys np*16 .. +16)
        // matrices: (keys +0..7, chunk c0), (keys +0..7, chunk c0+1), (keys +8..15, c0), (keys +8..15, c0+1)
        const int mi = lane >> 3, rr = lane & 7;
        const int key = np * 16 + (mi >> 1) * 8 + rr;
        const int c = c0 + (mi & 1);
        uint32_t b0, b1, b2, b3;
        ldsm_x4(kvb + swz(key, c), b0, b1, b2, b3);
        mma_bf16(s[np * 2], qa[ks][0], qa[ks][1], qa[ks][2], qa[ks][3], b0, b1);
        mma_bf16(s[np * 2 + 1], qa[ks][0], qa[ks][1], qa[ks][2], qa[ks][3], b2, b3);
      }
    }
    // ---- swap partials with the other DV half of this head group
    float* mine = sm.xs[rg][ch][lane];
    float* other = sm.xs[rg][ch ^ 1][lane];
#pragma unroll
    for (int j = 0; j < 4; ++j)
      *reinterpret_cast<float4*>(mine + 4 * j) = make_float4(s[j][0], s[j][1], s[j][2], s[j][3]);
    asm volatile("bar.sync %0, 64;\n" ::"r"(1 + rg));
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float4 o4 = *reinterpret_cast<const float4*>(other + 4 * j);
      s[j][0] += o4.x; s[j][1] += o4.y; s[j][2] += o4.z; s[j][3] += o4.w;
    }
    // ---- mask + online softmax (rows g -> m0/l0, g+8 -> m1/l1)
    float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int k0 = j * 8 + 2 * t;
      const bool v0 = sm.valid[buf][k0], v1 = sm.valid[buf][k0 + 1];
      s[j][0] = v0 ? s[j][0] * scale_log2 : -INFINITY;
      s[j][1] = v1 ? s[j][1] * scale_log2 : -INFINITY;
      s[j][2] = v0 ? s[j][2] * scale_log2 : -INFINITY;
      s[j][3] = v1 ? s[j][3] * scale_log2 : -INFINITY;
      mx0 = fmaxf(mx0, fmaxf(s[j][0], s[j][1]));
      mx1 = fmaxf(mx1, fmaxf(s[j][2], s[j][3]));
    }
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
    const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
    const float al0 = mn0 == -INFINITY ? 1.f : exp2f(m0 - mn0);
    const float al1 = mn1 == -INFINITY ? 1.f : exp2f(m1 - mn1);
    float ps0 = 0.f, ps1 = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      s[j][0] = mn0 == -INFINITY ? 0.f : exp2f(s[j][0] - mn0);
      s[j][1] = mn0 == -INFINITY ? 0.f : exp2f(s[j][1] - mn0);
      s[j][2] = mn1 == -INFINITY ? 0.f : exp2f(s[j][2] - mn1);
      s[j][3] = mn1 == -INFINITY ? 0.f : exp2f(s[j][3] - mn1);
      ps0 += s[j][0] + s[j][1];
      ps1 += s[j][2] + s[j][3];
    }
    l0 = l0 * al0 + ps0;
    l1 = l1 * al1 + ps1;
    m0 = mn0; m1 = mn1;
#pragma unroll
    for (int j = 0; j < 32; ++j) {
      acc[j][0] *= al0; acc[j][1] *= al0; acc[j][2] *= al1; acc[j][3] *= al1;
    }
    // ---- O += P V over this warp's 256 dims: 2 k-steps of 16 keys, 32 n-tiles of 8 dims
#pragma unroll
    for (int kk = 0; kk < 2; ++kk) {
      const uint32_t a0 = pack_bf16(s[2 * kk][0], s[2 * kk][1]);
      const uint32_t a1 = pack_bf16(s[2 * kk][2], s[2 * kk][3]);
      const uint32_t a2 = pack_bf16(s[2 * kk + 1][0], s[2 * kk + 1][1]);
      const uint32_t a3 = pack_bf16(s[2 * kk + 1][2], s[2 * kk + 1][3]);
#pragma unroll
      for (int jp = 0; jp < 16; ++jp) {  // n-tile pair: dims chunk ch*32 + 2 jp (+1)
        // matrices: (keys kk*16 +0..7, chunk), (keys +8..15, chunk), (keys +0..7, chunk+1), (keys +8..15, chunk+1)
        const int mi = lane >> 3, rr = lane & 7;
        const int key = kk * 16 + (mi & 1) * 8 + rr;
        const int c = ch * 32 + 2 * jp + (mi >> 1);
        uint32_t b0, b1, b2, b3;
        ldsm_x4_t(kvb + swz(key, c), b0, b1, b2, b3);
        mma_bf16(acc[2 * jp], a0, a1, a2, a3, b0, b1);
        mma_bf16(acc[2 * jp + 1], a0, a1, a2, a3, b2, b3);
      }
    }
    asm volatile("bar.sync %0, 64;\n" ::"r"(1 + rg));   // partner done reading xs
    __syncthreads();                                     // buffer `buf` free for tile + 2
  }
  cp_wait<0>();

  // ---- finalize: l over the quad, + sink, store
  l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
  l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
  l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
  l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
  const int h0 = rg * 16 + g, h1 = h0 + 8;
  const float LOG2E = 1.4426950408889634f;
  const float inv0 = m0 == -INFINITY ? 0.f : 1.f / (l0 + exp2f(sink[h0] * LOG2E - m0));
  const float inv1 = m1 == -INFINITY ? 0.f : 1.f / (l1 + exp2f(sink[h1] * LOG2E - m1));
  __nv_bfloat16* o0 = out + (long)qi * som + (long)h0 * soh + ch * 256 + 2 * t;
  __nv_bfloat16* o1 = out + (long)qi * som + (long)h1 * soh + ch * 256 + 2 * t;
#pragma unroll
  for (int j = 0; j < 32; ++j) {
    *reinterpret_cast<__nv_bfloat162*>(o0 + 8 * j) = __floats2bfloat162_rn(acc[j][0] * inv0, acc[j][1] * inv0);
    *reinterpret_cast<__nv_bfloat162*>(o1 + 8 * j) = __floats2bfloat162_rn(acc[j][2] * inv1, acc[j][3] * inv1);
  }
}

extern "C" int dsv4_sparse_mla(const void* q, long sqm, long sqh, const void* win, const void* cmp, long skv,
                               const void* sink, const void* idx, long sim, int topk, int n_window,
                               const void* cnt, void* out, long som, long soh, int m, float scale,
                               void* stream) {
  const int smem = sizeof(Smem);
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(sparse_mla_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    attr = true;
  }
  if (m == 0) return 0;
  sparse_mla_kernel<<<m, NWARP * 32, smem, (cudaStream_t)stream>>>(
      (const __nv_bfloat16*)q, sqm, sqh, (const __nv_bfloat16*)win, (const __nv_bfloat16*)cmp, skv,
      (const float*)sink, (const int*)idx, sim, topk, n_window, (const int*)cnt,
      (__nv_bfloat16*)out, som, soh, scale * 1.4426950408889634f);
  return (int)cudaGetLastError();
}
