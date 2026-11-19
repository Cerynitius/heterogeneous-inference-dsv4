"""Exact replacement for the decode indexer's ``scores.topk(k, dim=-1)[1]`` (bf16 scores).

What torch's CUDA top-k returns (k = 512, sorted): it radix-SELECTS the k largest by the
bf16 radix key (negatives bit-flipped, positives sign-flipped -> +0 ranks above -0, NaN
largest), gathering ``> kth`` then ``== kth`` in ascending column order, and then STABLY
radix-sorts the k winners by value, where the sort treats -0 == +0. So the output order is
(value desc with -0 == +0, column asc), and the selected set is (select-key desc, column asc).
Both are strict total orders once the column is appended, so we reproduce them exactly with
32-bit composites ``key16 << 16 | (0xFFFF - col)``:

  K1  per-chunk high-byte histograms of the select key (global atomics)
  K2  per-chunk low-byte histograms among hi == H (H derived from K1 in every program)
  K3  per-chunk counts of key == kth
  K4  per-chunk gather of the winners -- key > kth, or key == kth with global rank <= need
      (rank = prefix of K3 over earlier chunks + local cumsum) -- as SORT composites
  K5  one program per row: sort the k composites descending, emit columns, reset buffers.

Only the first ``valid`` columns are read (``valid`` from device memory: work tracks the live
context under CUDA-graph replay). With ``valid <= k`` K1-K3 exit at once and K4 takes every
valid column; the tail is filled with ``valid, valid+1, ...`` (the columns torch picks among
the -inf entries; the caller maps every pick >= valid to -1 anyway).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _load_bits(srow, offs, m):
    return tl.load(srow + offs, mask=m, other=0).to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF


@triton.jit
def _sel_key(bits):
    """torch TopKTypeConfig<BFloat16>::convert: monotone uint16, +0 > -0, NaN -> 0xFFFF."""
    neg = (bits & 0x8000) != 0
    k = tl.where(neg, bits ^ 0xFFFF, bits ^ 0x8000)
    isnan = ((bits & 0x7F80) == 0x7F80) & ((bits & 0x007F) != 0)
    return tl.where(isnan, 0xFFFF, k)


@triton.jit
def _sort_key(bits):
    """Order used by the final sort: like ``_sel_key`` but -0 == +0."""
    return _sel_key(tl.where(bits == 0x8000, 0, bits))


@triton.jit
def _hist256(v, m, CH: tl.constexpr):
    h = tl.histogram(tl.where(m, v, 0), 256)
    nmask = CH - tl.sum(m.to(tl.int32), axis=0)
    return h - tl.where(tl.arange(0, 256) == 0, nmask, 0)


@triton.jit
def _pick_bin(hist, need):
    """Largest bin b with #(entries in bins >= b) >= need -> (b, #entries in bins > b)."""
    bins = tl.arange(0, 256)
    cum = tl.cumsum(hist, axis=0)
    total = tl.sum(hist, axis=0)
    ge = total - cum + hist
    b = tl.sum((ge >= need).to(tl.int32), axis=0) - 1
    gt = tl.sum(tl.where(bins == b, ge - hist, 0), axis=0)
    return b, gt


@triton.jit
def _k1_hist_hi(score_ptr, valid_ptr, h1_ptr, kmin_ptr, n_stage, stride_sb, K: tl.constexpr,
                CH: tl.constexpr):
    b = tl.program_id(0)
    c0 = tl.program_id(1) * CH
    valid = tl.minimum(tl.load(valid_ptr + b).to(tl.int32), n_stage)
    if c0 < valid:
        offs = c0 + tl.arange(0, CH)
        m = offs < valid
        key = _sel_key(_load_bits(score_ptr + b * stride_sb, offs, m))
        if valid > K:
            h = _hist256(key >> 8, m, CH)
            tl.atomic_add(h1_ptr + b * 256 + tl.arange(0, 256), h, mask=h != 0)
        elif valid == K:  # every valid column wins; torch's kth is their minimum key
            tl.atomic_min(kmin_ptr + b, tl.min(tl.where(m, key, 0xFFFF), axis=0))


@triton.jit
def _k2_hist_lo(score_ptr, valid_ptr, h1_ptr, h2_ptr, n_stage, stride_sb, K: tl.constexpr, CH: tl.constexpr):
    b = tl.program_id(0)
    c0 = tl.program_id(1) * CH
    valid = tl.minimum(tl.load(valid_ptr + b).to(tl.int32), n_stage)
    if (valid > K) & (c0 < valid):
        H, gt_hi = _pick_bin(tl.load(h1_ptr + b * 256 + tl.arange(0, 256)), K)
        offs = c0 + tl.arange(0, CH)
        m = offs < valid
        key = _sel_key(_load_bits(score_ptr + b * stride_sb, offs, m))
        h = _hist256(key & 0xFF, m & ((key >> 8) == H), CH)
        tl.atomic_add(h2_ptr + b * 256 + tl.arange(0, 256), h, mask=h != 0)


@triton.jit
def _kth(h1_ptr, h2_ptr, b, K: tl.constexpr):
    H, gt_hi = _pick_bin(tl.load(h1_ptr + b * 256 + tl.arange(0, 256)), K)
    L, gt_lo = _pick_bin(tl.load(h2_ptr + b * 256 + tl.arange(0, 256)), K - gt_hi)
    return H * 256 + L, K - gt_hi - gt_lo   # kth select key, #winners with key == kth


@triton.jit
def _k3_eqcount(score_ptr, valid_ptr, h1_ptr, h2_ptr, eqc_ptr, n_stage, stride_sb, NCH,
                K: tl.constexpr, CH: tl.constexpr):
    b = tl.program_id(0)
    ci = tl.program_id(1)
    c0 = ci * CH
    valid = tl.minimum(tl.load(valid_ptr + b).to(tl.int32), n_stage)
    if (valid > K) & (c0 < valid):
        kth, _need = _kth(h1_ptr, h2_ptr, b, K)
        offs = c0 + tl.arange(0, CH)
        m = offs < valid
        key = _sel_key(_load_bits(score_ptr + b * stride_sb, offs, m))
        tl.store(eqc_ptr + b * NCH + ci, tl.sum((m & (key == kth)).to(tl.int32), axis=0))


@triton.jit
def _k4_gather(score_ptr, valid_ptr, h1_ptr, h2_ptr, eqc_ptr, cnt_ptr, buf_ptr, kmin_ptr, n_stage, stride_sb,
               NCH, K: tl.constexpr, CH: tl.constexpr, NCHP: tl.constexpr):
    b = tl.program_id(0)
    ci = tl.program_id(1)
    c0 = ci * CH
    valid = tl.minimum(tl.load(valid_ptr + b).to(tl.int32), n_stage)
    if c0 < valid:
        offs = c0 + tl.arange(0, CH)
        m = offs < valid
        bits = _load_bits(score_ptr + b * stride_sb, offs, m)
        key = _sel_key(bits)
        if valid > K:
            kth, need = _kth(h1_ptr, h2_ptr, b, K)
            eq = m & (key == kth)
            cj = tl.arange(0, NCHP)
            before = tl.sum(tl.load(eqc_ptr + b * NCH + cj, mask=cj < ci, other=0), axis=0)
            rank = before + tl.cumsum(eq.to(tl.int32), axis=0)
            sel = m & ((key > kth) | (eq & (rank <= need)))
        else:
            # torch's kth: the -inf key when fewer than K columns are finite-valid, else
            # (valid == K) the minimum key among the valid columns
            kth = tl.where(valid == K, tl.load(kmin_ptr + b), 0x007F)
            sel = m
        # gather order of torch (`> kth` before `== kth`, column asc inside each) is what the
        # stable value sort keeps among values it treats as equal (-0 == +0)
        gt = (key > kth).to(tl.int64)
        comp = (_sort_key(bits).to(tl.int64) << 17) | (gt << 16) | (0xFFFF - offs).to(tl.int64)
        n = tl.sum(sel.to(tl.int32), axis=0)
        base = tl.atomic_add(cnt_ptr + b, n)
        pos = base + tl.cumsum(sel.to(tl.int32), axis=0) - 1
        tl.store(buf_ptr + b * K + pos, comp, mask=sel)


@triton.jit
def _k5_sort(valid_ptr, cnt_ptr, buf_ptr, out_ptr, h1_ptr, h2_ptr, kmin_ptr, n_stage, stride_ob, K: tl.constexpr):
    b = tl.program_id(0)
    valid = tl.minimum(tl.load(valid_ptr + b).to(tl.int32), n_stage)
    cnt = tl.load(cnt_ptr + b)
    kk = tl.arange(0, K)
    c = tl.load(buf_ptr + b * K + kk, mask=kk < cnt, other=0)
    c = tl.sort(c, descending=True)
    col = (0xFFFF - (c & 0xFFFF)).to(tl.int64)  # low 16 bits: 0xFFFF - column
    col = tl.where(kk < cnt, col, (valid + (kk - cnt)).to(tl.int64))
    tl.store(out_ptr + b * stride_ob + kk, col)
    # reset the per-call accumulators for the next layer / step (same stream, in order)
    zb = tl.arange(0, 256)
    tl.store(h1_ptr + b * 256 + zb, tl.zeros((256,), tl.int32))
    tl.store(h2_ptr + b * 256 + zb, tl.zeros((256,), tl.int32))
    tl.store(cnt_ptr + b, 0)
    tl.store(kmin_ptr + b, 0xFFFF)


_WS: dict = {}


_WS_ROWS = 16      # rows (requests) pre-allocated; views [:B] are handed out
_WS_NCH = 16       # max chunks: n_stage <= 65536 at CH 4096


def _alloc_ws(dev, rows, k, nch):
    return (
        torch.zeros((rows, 256), dtype=torch.int32, device=dev),
        torch.zeros((rows, 256), dtype=torch.int32, device=dev),
        torch.zeros((rows * nch,), dtype=torch.int32, device=dev),
        torch.zeros((rows,), dtype=torch.int32, device=dev),
        torch.zeros((rows, k), dtype=torch.int64, device=dev),
        torch.full((rows,), 0xFFFF, dtype=torch.int32, device=dev),
    )


def topk_workspace_prewarm(dev, k, rows: int = _WS_ROWS) -> None:
    """Allocate (zero-init) the workspace for top-k width ``k`` EAGERLY. Called from model bind so
    the zero-fills never land inside a captured decode graph (where they would replay every step:
    the eager warmup uses a smaller staging width than the capture, so a lazily keyed workspace was
    first created during capture)."""
    key = (torch.device(dev).index, k)
    if key not in _WS or _WS[key][0].shape[0] < rows:
        _WS[key] = _alloc_ws(dev, rows, k, _WS_NCH)


def _workspace(dev, B, k, nch):
    # zero-initialized once; K5 leaves rows [:B] zeroed after every call (eqc is rewritten by K3)
    assert nch <= _WS_NCH, nch
    key = (dev.index, k)
    if key not in _WS or _WS[key][0].shape[0] < B:
        topk_workspace_prewarm(dev, k, max(B, _WS_ROWS))
    h1, h2, eqc, cnt, buf, kmin = _WS[key]
    return h1[:B], h2[:B], eqc, cnt[:B], buf[:B], kmin[:B]


def topk_exact(scores: torch.Tensor, valid: torch.Tensor, k: int, CH: int = 4096) -> torch.Tensor:
    """``scores`` [B, 1, n_stage] bf16 (columns >= valid hold -inf), ``valid`` [B] ->
    [B, 1, k] int64, equal to ``scores.topk(k, -1)[1]`` on every pick < valid."""
    B = scores.shape[0]
    n_stage = scores.shape[-1]
    assert scores.dtype == torch.bfloat16 and scores.stride(-1) == 1
    assert n_stage <= 65536 and k & (k - 1) == 0 and k <= n_stage
    s2 = scores.reshape(B, n_stage)
    nch = triton.cdiv(n_stage, CH)
    h1, h2, eqc, cnt, buf, kmin = _workspace(scores.device, B, k, nch)
    out = torch.empty((B, 1, k), dtype=torch.int64, device=scores.device)
    grid = (B, nch)
    _k1_hist_hi[grid](s2, valid, h1, kmin, n_stage, s2.stride(0), K=k, CH=CH, num_warps=8)
    _k2_hist_lo[grid](s2, valid, h1, h2, n_stage, s2.stride(0), K=k, CH=CH, num_warps=8)
    _k3_eqcount[grid](s2, valid, h1, h2, eqc, n_stage, s2.stride(0), nch, K=k, CH=CH, num_warps=8)
    _k4_gather[grid](s2, valid, h1, h2, eqc, cnt, buf, kmin, n_stage, s2.stride(0), nch, K=k, CH=CH,
                     NCHP=max(2, triton.next_power_of_2(nch)), num_warps=8)
    _k5_sort[(B,)](valid, cnt, buf, out, h1, h2, kmin, n_stage, out.stride(0), K=k, num_warps=8)
    return out


__all__ = ["topk_exact", "topk_workspace_prewarm"]
