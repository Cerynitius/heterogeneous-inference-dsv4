// Row-wise top-k over bf16 scores for the DSV4 indexer prefill (sm_120a).
// One CTA per row: 16-bit radix select in two histogram passes (high byte, then low byte among the
// high-byte bin that holds the k-th largest), then a third pass writes the selected column indices
// in index order (elements > threshold, then the first `need` elements == threshold), so the
// result is deterministic. Output is the indexer's final form: col >= live -> -1, else col + offset.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define NTH 1024
#define NWP (NTH / 32)

__device__ __forceinline__ uint32_t ord16(uint16_t b) {      // bf16 bits -> order-preserving uint
  return (b & 0x8000u) ? (uint32_t)(~b & 0xFFFFu) : (uint32_t)(b | 0x8000u);
}

// per-warp private histograms (scores cluster in few bins: shared atomics would serialize)
__device__ __forceinline__ void hist_add(int* wh, uint32_t bin) {
  const uint32_t peers = __match_any_sync(__activemask(), bin);
  const int leader = __ffs(peers) - 1;
  if ((threadIdx.x & 31) == leader) atomicAdd(&wh[bin], __popc(peers));
}

// suffix search over hist[256]: largest bin b with count(>= b) >= need; returns b, sets *above = count(> b)
__device__ int find_bin(const int* hist, int need, int* above, int* sh_b, int* sh_above) {
  if (threadIdx.x < 32) {
    const int lane = threadIdx.x;
    // lane owns bins [lane*8, lane*8+8)
    int c[8], tot = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) { c[i] = hist[lane * 8 + i]; tot += c[i]; }
    // inclusive suffix sum over lanes (lanes with higher bins first)
    int suf = tot;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int v = __shfl_down_sync(0xffffffffu, suf, o);
      if (lane + o < 32) suf += v;
    }
    const int above_lane = suf - tot;          // count in bins of higher lanes
    // the owning lane: above_lane < need <= suf
    if (above_lane < need && need <= suf) {
      int acc = above_lane;
#pragma unroll
      for (int i = 7; i >= 0; --i) {
        if (acc + c[i] >= need) { *sh_b = lane * 8 + i; *sh_above = acc; break; }
        acc += c[i];
      }
    }
  }
  __syncthreads();
  *above = *sh_above;
  return *sh_b;
}

__global__ void __launch_bounds__(NTH, 1)
topk_bf16_kernel(const __nv_bfloat16* __restrict__ x, long stride_x, int T, int K,
                 int64_t* __restrict__ out, long stride_o, int start_pos, int ratio, long offset,
                 const int* __restrict__ valid, int raw) {
  __shared__ int wh[NWP][256];
  __shared__ int hist[256];
  __shared__ int sh_b, sh_above;
  __shared__ int wsum[2][NWP];
  const int row = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const uint16_t* xr = reinterpret_cast<const uint16_t*>(x + (long)row * stride_x);
  // decode: per-row live count in device memory (graph replay tracks position); only the first
  // max(valid, K) columns are scanned -- the rest are -inf by construction and never needed
  int live = (start_pos + row + 1) / ratio;
  if (valid != nullptr) {
    live = valid[row];
    T = min(T, max(live, K));
  }
  const bool vec = ((reinterpret_cast<uintptr_t>(xr) & 15) == 0);

  auto load8 = [&](int base, uint16_t* v) {        // elements [base, base+8); missing -> 0 (unused)
    if (vec && base + 8 <= T) {
      const uint4 q = *reinterpret_cast<const uint4*>(xr + base);
      const uint16_t* p = reinterpret_cast<const uint16_t*>(&q);
#pragma unroll
      for (int i = 0; i < 8; ++i) v[i] = p[i];
    } else {
#pragma unroll
      for (int i = 0; i < 8; ++i) v[i] = base + i < T ? xr[base + i] : 0;
    }
  };

  // ---- pass 1: high byte
  for (int i = tid; i < NWP * 256; i += NTH) (&wh[0][0])[i] = 0;
  __syncthreads();
  for (int base = tid * 8; base < T; base += NTH * 8) {
    uint16_t v[8];
    load8(base, v);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      if (base + i < T) hist_add(wh[warp], ord16(v[i]) >> 8);
    }
  }
  __syncthreads();
  for (int b = tid; b < 256; b += NTH) {
    int s = 0;
    for (int w = 0; w < NWP; ++w) s += wh[w][b];
    hist[b] = s;
  }
  __syncthreads();
  int above1;
  const int b1 = find_bin(hist, K, &above1, &sh_b, &sh_above);
  const int need1 = K - above1;                    // still to take from bin b1

  // ---- pass 2: low byte inside bin b1
  for (int i = tid; i < NWP * 256; i += NTH) (&wh[0][0])[i] = 0;
  __syncthreads();
  for (int base = tid * 8; base < T; base += NTH * 8) {
    uint16_t v[8];
    load8(base, v);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint32_t u = ord16(v[i]);
      if (base + i < T && (int)(u >> 8) == b1) hist_add(wh[warp], u & 0xFFu);
    }
  }
  __syncthreads();
  for (int b = tid; b < 256; b += NTH) {
    int s = 0;
    for (int w = 0; w < NWP; ++w) s += wh[w][b];
    hist[b] = s;
  }
  __syncthreads();
  int above2;
  const int b2 = find_bin(hist, need1, &above2, &sh_b, &sh_above);
  const uint32_t thr = ((uint32_t)b1 << 8) | (uint32_t)b2;
  const int n_gt = above1 + above2;                // elements strictly above thr
  const int need_eq = K - n_gt;                    // ties at thr to take (in index order)

  // ---- pass 3: ordered write
  int64_t* orow = out + (long)row * stride_o;
  int off_gt = 0, off_eq = 0;
  for (int base0 = 0; base0 < T; base0 += NTH * 8) {
    const int base = base0 + tid * 8;
    uint16_t v[8];
    load8(base, v);
    int cg = 0, ce = 0;
    uint32_t u[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      u[i] = base + i < T ? ord16(v[i]) : 0u;
      const bool valid = base + i < T;
      cg += valid && u[i] > thr;
      ce += valid && u[i] == thr;
    }
    // block exclusive scan of (cg, ce)
    int sg = cg, se = ce;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int a = __shfl_up_sync(0xffffffffu, sg, o), b = __shfl_up_sync(0xffffffffu, se, o);
      if (lane >= o) { sg += a; se += b; }
    }
    if (lane == 31) { wsum[0][warp] = sg; wsum[1][warp] = se; }
    __syncthreads();
    if (warp == 0) {
      int a = wsum[0][lane], b = wsum[1][lane];
      int ia = a, ib = b;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int x1 = __shfl_up_sync(0xffffffffu, ia, o), x2 = __shfl_up_sync(0xffffffffu, ib, o);
        if (lane >= o) { ia += x1; ib += x2; }
      }
      wsum[0][lane] = ia - a; wsum[1][lane] = ib - b;     // exclusive warp offsets
    }
    __syncthreads();
    int pg = off_gt + wsum[0][warp] + sg - cg;
    int pe = off_eq + wsum[1][warp] + se - ce;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int col = base + i;
      if (col < T) {
        int slot = -1;
        if (u[i] > thr) slot = pg++;
        else if (u[i] == thr) { if (pe < need_eq) slot = n_gt + pe; ++pe; }
        if (slot >= 0) orow[slot] = raw ? (int64_t)col : (col >= live ? (int64_t)-1 : (int64_t)col + offset);
      }
    }
    // tile totals for the next tile
    __syncthreads();
    if (tid == NTH - 1) { wsum[0][0] = pg - off_gt; wsum[1][0] = pe - off_eq; }
    __syncthreads();
    off_gt += wsum[0][0]; off_eq += wsum[1][0];
    __syncthreads();
  }
}

extern "C" int dsv4_topk_bf16(const void* x, long stride_x, int rows, int T, int K, void* out, long stride_o,
                              int start_pos, int ratio, long offset, void* stream, const void* valid, int raw) {
  if (rows == 0 || K == 0) return 0;
  topk_bf16_kernel<<<rows, NTH, 0, (cudaStream_t)stream>>>((const __nv_bfloat16*)x, stride_x, T, K,
                                                           (int64_t*)out, stride_o, start_pos, ratio, offset,
                                                           (const int*)valid, raw);
  return (int)cudaGetLastError();
}

// ---- decode, multi-CTA (11-19): stage 1 = one CTA per (row, 4096-column chunk) selects its local
// top-K (radix select in shared memory, column order) into a candidate list; stage 2 = one CTA
// per row selects the global top-K among the <= nchunk*K candidates (same radix code over values),
// writing columns in candidate (= column) order. Same selected SET as a single global top-K
// (a global top-K element is in its chunk's local top-K). Only chunks below max(valid, K) work.
#define DCH 4096
__global__ void __launch_bounds__(NTH, 1)
topk_dec_stage1(const __nv_bfloat16* __restrict__ x, long stride_x, int T, int K, const int* __restrict__ valid,
                uint16_t* __restrict__ cand_v, int* __restrict__ cand_c, int* __restrict__ cand_n, int nchunk) {
  __shared__ uint16_t sv[DCH];
  __shared__ int wh[NWP][256];
  __shared__ int hist[256];
  __shared__ int sh_b, sh_above;
  __shared__ int wsum[2][NWP];
  const int row = blockIdx.x, ch = blockIdx.y, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int live = valid[row];
  const int Tn = min(T, max(live, K));
  const int c0 = ch * DCH;
  const int n = max(0, min(DCH, Tn - c0));
  int* cn = cand_n + (long)row * nchunk + ch;
  if (n == 0) { if (tid == 0) *cn = 0; return; }
  const uint16_t* xr = reinterpret_cast<const uint16_t*>(x + (long)row * stride_x) + c0;
  for (int i = tid; i < n; i += NTH) sv[i] = (uint16_t)ord16(xr[i]);
  __syncthreads();
  const int Kc = min(K, n);
  // pass 1/2 histograms over shared memory
  for (int i = tid; i < NWP * 256; i += NTH) (&wh[0][0])[i] = 0;
  __syncthreads();
  for (int i = tid; i < n; i += NTH) hist_add(wh[warp], sv[i] >> 8);
  __syncthreads();
  for (int b = tid; b < 256; b += NTH) { int s = 0; for (int w = 0; w < NWP; ++w) s += wh[w][b]; hist[b] = s; }
  __syncthreads();
  int above1; const int b1 = find_bin(hist, Kc, &above1, &sh_b, &sh_above);
  const int need1 = Kc - above1;
  for (int i = tid; i < NWP * 256; i += NTH) (&wh[0][0])[i] = 0;
  __syncthreads();
  for (int i = tid; i < n; i += NTH) if ((int)(sv[i] >> 8) == b1) hist_add(wh[warp], sv[i] & 0xFFu);
  __syncthreads();
  for (int b = tid; b < 256; b += NTH) { int s = 0; for (int w = 0; w < NWP; ++w) s += wh[w][b]; hist[b] = s; }
  __syncthreads();
  int above2; const int b2 = find_bin(hist, need1, &above2, &sh_b, &sh_above);
  const uint32_t thr = ((uint32_t)b1 << 8) | (uint32_t)b2;
  const int n_gt = above1 + above2, need_eq = Kc - n_gt;
  // ordered compaction: thread owns DCH/NTH = 4 consecutive columns
  uint16_t* ov = cand_v + ((long)row * nchunk + ch) * K;
  int* oc = cand_c + ((long)row * nchunk + ch) * K;
  constexpr int PER = DCH / NTH;
  int cg = 0, ce = 0;
#pragma unroll
  for (int i = 0; i < PER; ++i) {
    const int j = tid * PER + i;
    if (j < n) { cg += sv[j] > thr; ce += sv[j] == thr; }
  }
  int sg = cg, se = ce;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int a = __shfl_up_sync(0xffffffffu, sg, o), b = __shfl_up_sync(0xffffffffu, se, o);
    if (lane >= o) { sg += a; se += b; }
  }
  if (lane == 31) { wsum[0][warp] = sg; wsum[1][warp] = se; }
  __syncthreads();
  if (warp == 0) {
    int a = wsum[0][lane], b = wsum[1][lane], ia = a, ib = b;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int x1 = __shfl_up_sync(0xffffffffu, ia, o), x2 = __shfl_up_sync(0xffffffffu, ib, o);
      if (lane >= o) { ia += x1; ib += x2; }
    }
    wsum[0][lane] = ia - a; wsum[1][lane] = ib - b;
  }
  __syncthreads();
  int pg = wsum[0][warp] + sg - cg, pe = wsum[1][warp] + se - ce;
  // merge gt and eq in column order: final slot = (#gt before) + (#eq-taken before)
#pragma unroll
  for (int i = 0; i < PER; ++i) {
    const int j = tid * PER + i;
    if (j < n) {
      const uint16_t u = sv[j];
      if (u > thr) { const int slot = pg + min(pe, need_eq); ov[slot] = u; oc[slot] = c0 + j; ++pg; }
      else if (u == thr) { if (pe < need_eq) { const int slot = pg + pe; ov[slot] = u; oc[slot] = c0 + j; } ++pe; }
    }
  }
  for (int i = Kc + tid; i < K; i += NTH) { ov[i] = 0; oc[i] = -1; }
  if (tid == 0) *cn = Kc;
}

__global__ void __launch_bounds__(NTH, 1)
topk_dec_stage2(const uint16_t* __restrict__ cand_v, const int* __restrict__ cand_c, const int* __restrict__ cand_n,
                int nchunk, int K, int64_t* __restrict__ out, long stride_o, const int* __restrict__ valid, int raw,
                long offset, int group, uint16_t* __restrict__ mv, int* __restrict__ mc, int* __restrict__ mn) {
  // group > 0: CTA (row, g) merges chunks [g*group, (g+1)*group) into candidate slot g of (mv, mc, mn)
  // (layout [rows, gridDim.y, K]); group == 0: one CTA per row over all chunks -> final picks
  __shared__ int wh[NWP][256];
  __shared__ int hist[256];
  __shared__ int sh_b, sh_above;
  __shared__ int wsum[2][NWP];
  __shared__ int coff[65];
  const int row = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int live = valid[row];
  const int cb = group > 0 ? blockIdx.y * group : 0;   // first chunk of this CTA
  const uint16_t* cv = cand_v + ((long)row * nchunk + cb) * K;
  const int* cc = cand_c + ((long)row * nchunk + cb) * K;
  if (tid == 0) {
    int acc = 0;
    // group > 0: merge stage-1 chunks [cb, cb+group) (bounded by the live chunks, dead ones hold no
    // values); group == 0: final over stage-1 chunks (bounded); group < 0: final over merge slots
    // (every slot written, padded -> unbounded)
    int nl = group > 0 ? min(group, nchunk - cb) : nchunk;
    if (group >= 0) nl = min(nl, (max(live, K) + DCH - 1) / DCH - cb);
    nl = max(nl, 0);
    for (int c = 0; c < nl; ++c) acc += cand_n[(long)row * nchunk + cb + c];
    coff[0] = nl; coff[1] = acc;
  }
  __syncthreads();
  const int total = coff[0] * K;      // fixed K slots per live chunk, pads hold value 0
  const int real = coff[1];
  auto at = [&](int i, uint16_t& v, int& c) { v = cv[i]; c = cc[i]; };
  const int Kc = min(K, real);
  for (int i = tid; i < NWP * 256; i += NTH) (&wh[0][0])[i] = 0;
  __syncthreads();
  for (int i = tid; i < total; i += NTH) { uint16_t v; int c; at(i, v, c); hist_add(wh[warp], v >> 8); }
  __syncthreads();
  for (int b = tid; b < 256; b += NTH) { int s = 0; for (int w = 0; w < NWP; ++w) s += wh[w][b]; hist[b] = s; }
  __syncthreads();
  int above1; const int b1 = find_bin(hist, Kc, &above1, &sh_b, &sh_above);
  const int need1 = Kc - above1;
  for (int i = tid; i < NWP * 256; i += NTH) (&wh[0][0])[i] = 0;
  __syncthreads();
  for (int i = tid; i < total; i += NTH) { uint16_t v; int c; at(i, v, c); if ((int)(v >> 8) == b1) hist_add(wh[warp], v & 0xFFu); }
  __syncthreads();
  for (int b = tid; b < 256; b += NTH) { int s = 0; for (int w = 0; w < NWP; ++w) s += wh[w][b]; hist[b] = s; }
  __syncthreads();
  int above2; const int b2 = find_bin(hist, need1, &above2, &sh_b, &sh_above);
  const uint32_t thr = ((uint32_t)b1 << 8) | (uint32_t)b2;
  const int n_gt = above1 + above2, need_eq = Kc - n_gt;
  int64_t* orow = out + (long)row * stride_o;
  const long mbase = ((long)row * gridDim.y + blockIdx.y) * K;
  if (mv != nullptr) {   // merge mode: pad unused slots (value 0 < any real), record the count
    for (int i = Kc + tid; i < K; i += NTH) { mv[mbase + i] = 0; mc[mbase + i] = -1; }
    if (tid == 0) mn[(long)row * gridDim.y + blockIdx.y] = Kc;
  }
  int off_gt = 0, off_eq = 0;
  for (int base0 = 0; base0 < total; base0 += NTH) {
    const int i = base0 + tid;
    uint16_t v = 0; int c = 0;
    const bool ok = i < total;
    if (ok) at(i, v, c);
    const int cg = ok && v > thr, ce = ok && v == thr;
    int sg = cg, se = ce;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int a = __shfl_up_sync(0xffffffffu, sg, o), b = __shfl_up_sync(0xffffffffu, se, o);
      if (lane >= o) { sg += a; se += b; }
    }
    if (lane == 31) { wsum[0][warp] = sg; wsum[1][warp] = se; }
    __syncthreads();
    if (warp == 0) {
      int a = wsum[0][lane], b = wsum[1][lane], ia = a, ib = b;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int x1 = __shfl_up_sync(0xffffffffu, ia, o), x2 = __shfl_up_sync(0xffffffffu, ib, o);
        if (lane >= o) { ia += x1; ib += x2; }
      }
      wsum[0][lane] = ia - a; wsum[1][lane] = ib - b;
    }
    __syncthreads();
    const int pg = off_gt + wsum[0][warp] + sg - cg, pe = off_eq + wsum[1][warp] + se - ce;
    int slot = -1;
    if (cg) slot = pg + min(pe, need_eq);
    else if (ce && pe < need_eq) slot = pg + pe;
    if (slot >= 0) {
      if (mv != nullptr) { mv[mbase + slot] = v; mc[mbase + slot] = c; }
      else orow[slot] = raw ? (int64_t)c : (c >= live ? (int64_t)-1 : (int64_t)c + offset);
    }
    __syncthreads();
    if (tid == NTH - 1) { wsum[0][0] = pg + cg - off_gt; wsum[1][0] = pe + ce - off_eq; }
    __syncthreads();
    off_gt += wsum[0][0]; off_eq += wsum[1][0];
    __syncthreads();
  }
}

extern "C" int dsv4_topk_dec2(const void* x, long stride_x, int rows, int T, int K, const void* valid,
                              void* cand_v, void* cand_c, void* cand_n, void* out, long stride_o, int raw,
                              long offset, void* stream, void* merge_v, void* merge_c, void* merge_n) {
  if (rows == 0 || K == 0) return 0;
  const int nchunk = (T + DCH - 1) / DCH;
  if (nchunk > 64) return -1;
  cudaStream_t st = (cudaStream_t)stream;
  topk_dec_stage1<<<dim3(rows, nchunk), NTH, 0, st>>>((const __nv_bfloat16*)x, stride_x, T, K, (const int*)valid,
                                                      (uint16_t*)cand_v, (int*)cand_c, (int*)cand_n, nchunk);
  if (nchunk > 8 && merge_v != nullptr) {
    const int G = 8, ng = (nchunk + G - 1) / G;
    topk_dec_stage2<<<dim3(rows, ng), NTH, 0, st>>>((const uint16_t*)cand_v, (const int*)cand_c, (const int*)cand_n,
                                                   nchunk, K, (int64_t*)out, stride_o, (const int*)valid, raw, offset,
                                                   G, (uint16_t*)merge_v, (int*)merge_c, (int*)merge_n);
    topk_dec_stage2<<<rows, NTH, 0, st>>>((const uint16_t*)merge_v, (const int*)merge_c, (const int*)merge_n, ng, K,
                                         (int64_t*)out, stride_o, (const int*)valid, raw, offset, -1, nullptr, nullptr,
                                         nullptr);
  } else {
    topk_dec_stage2<<<rows, NTH, 0, st>>>((const uint16_t*)cand_v, (const int*)cand_c, (const int*)cand_n, nchunk, K,
                                         (int64_t*)out, stride_o, (const int*)valid, raw, offset, 0, nullptr, nullptr,
                                         nullptr);
  }
  return (int)cudaGetLastError();
}
