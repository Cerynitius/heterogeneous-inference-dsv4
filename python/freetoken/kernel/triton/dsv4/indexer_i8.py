"""Lightning-Indexer prefill logits on INT8 tensor cores -- same values as the bf16 kernel.

Index queries and keys are FP4 round-trips: every value is e2m1 x 2^e with one exponent per
32 dims (``fp4_act_quant_inplace(x, 32)``). Written as ``int8(2 * v / 2^e)`` (codes in
{0, +-1, +-2, +-3, +-4, +-6, +-8, +-12}) plus the exponent, each 32-dim block dot product is an
exact int32 (|sum| <= 32 * 144), and

    q . k = sum_b  idot_b * 2^(eq_b + ek_b - 2)

is the exact real dot product. The bf16 kernel's fp32 tensor-core accumulation of the same
products is exact too whenever every partial sum fits fp32's 24-bit significand (all of them
here unless one dot's 32-dim blocks span > ~11 binary orders of magnitude), so both give the
same fp32 score; ReLU, the head weights and the head reduction are the bf16 kernel's own code
and layout. INT8 MMA runs at 2x the bf16 rate and K/Q bytes halve. The causal mask is written
in-kernel (-inf), the same values ``masked_fill`` produced.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _log2_ceil(v):
    bits = v.to(tl.uint32, bitcast=True)
    e = ((bits >> 23) & 0xFF).to(tl.int32)
    man = (bits & 0x7FFFFF).to(tl.int32)
    return e - 127 + tl.where(man != 0, 1, 0)


@triton.jit
def _to_i8_kernel(x_ptr, c_ptr, e_ptr, n_rows, stride_x, BLOCK_R: tl.constexpr):
    """[n, 128] FP4-round-trip bf16 -> int8 codes [n, 128] + exponents [n, 4] (int8)."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    rm = rows < n_rows
    offs = tl.arange(0, 32)
    for blk in tl.static_range(4):
        x = tl.load(x_ptr + rows[:, None].to(tl.int64) * stride_x + blk * 32 + offs[None, :],
                    mask=rm[:, None], other=0.0).to(tl.float32)
        amax = tl.maximum(tl.max(tl.abs(x), axis=1), 6.0 * (2.0 ** -126))
        e = _log2_ceil(amax * (1.0 / 6.0))                       # [BLOCK_R]
        inv2 = ((128 - e) << 23).to(tl.float32, bitcast=True)     # 2^(1-e): exact
        c = (x * inv2[:, None]).to(tl.int8)                      # exact small integers
        tl.store(c_ptr + rows[:, None].to(tl.int64) * 128 + blk * 32 + offs[None, :], c, mask=rm[:, None])
        tl.store(e_ptr + rows.to(tl.int64) * 4 + blk, e.to(tl.int8), mask=rm)


def to_i8(x: torch.Tensor):
    """``x`` [..., 128] bf16 (FP4 round-trip values) -> (codes int8 [..., 128], exps int8 [..., 4])."""
    shp = x.shape
    x2 = x.reshape(-1, 128)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    n = x2.shape[0]
    c = torch.empty((n, 128), dtype=torch.int8, device=x.device)
    e = torch.empty((n, 4), dtype=torch.int8, device=x.device)
    if n:
        BR = 64
        _to_i8_kernel[(triton.cdiv(n, BR),)](x2, c, e, n, x2.stride(0), BLOCK_R=BR, num_warps=4)
    return c.view(*shp[:-1], 128), e.view(*shp[:-1], 4)


@triton.jit
def _blk_dot(qc_ptr, kc_ptr, offs_h, offs_t, h_mask, t_mask, BLK: tl.constexpr):
    o32 = tl.arange(0, 32)
    qb = tl.load(qc_ptr + offs_h[:, None] * 128 + BLK * 32 + o32[None, :], mask=h_mask[:, None], other=0)
    kb = tl.load(kc_ptr + offs_t[:, None].to(tl.int64) * 128 + BLK * 32 + o32[None, :],
                 mask=t_mask[:, None], other=0)
    return tl.dot(qb, tl.trans(kb))                           # [BLOCK_H, BLOCK_T] int32, exact


@triton.jit
def _blk_scale(qe_ptr, ke_ptr, offs_h, offs_t, h_mask, t_mask, BLK: tl.constexpr):
    eq = tl.load(qe_ptr + offs_h * 4 + BLK, mask=h_mask, other=0).to(tl.int32)
    ek = tl.load(ke_ptr + offs_t.to(tl.int64) * 4 + BLK, mask=t_mask, other=0).to(tl.int32)
    sq = ((eq + 126) << 23).to(tl.float32, bitcast=True)    # 2^(eq-1)
    sk = ((ek + 126) << 23).to(tl.float32, bitcast=True)    # 2^(ek-1)
    return sq[:, None] * sk[None, :]                         # 2^(eq+ek-2), exact


@triton.jit
def _indexer_logits_i8_kernel(
    qc_ptr, qe_ptr, kc_ptr, ke_ptr, w_ptr, out_ptr,
    S, T, start_pos, RATIO,
    stride_ws, stride_wh, stride_os,
    H: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_T: tl.constexpr, MASK: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_t = tl.program_id(2)
    offs_h = tl.arange(0, BLOCK_H)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    h_mask = offs_h < H
    t_mask = offs_t < T
    qc = qc_ptr + pid_s.to(tl.int64) * (H * 128)
    qe = qe_ptr + pid_s.to(tl.int64) * (H * 4)
    w = tl.load(w_ptr + pid_s * stride_ws + offs_h * stride_wh, mask=h_mask, other=0.0).to(tl.float32)

    score = _blk_dot(qc, kc_ptr, offs_h, offs_t, h_mask, t_mask, 0).to(tl.float32) * \
        _blk_scale(qe, ke_ptr, offs_h, offs_t, h_mask, t_mask, 0)
    score += _blk_dot(qc, kc_ptr, offs_h, offs_t, h_mask, t_mask, 1).to(tl.float32) * \
        _blk_scale(qe, ke_ptr, offs_h, offs_t, h_mask, t_mask, 1)
    score += _blk_dot(qc, kc_ptr, offs_h, offs_t, h_mask, t_mask, 2).to(tl.float32) * \
        _blk_scale(qe, ke_ptr, offs_h, offs_t, h_mask, t_mask, 2)
    score += _blk_dot(qc, kc_ptr, offs_h, offs_t, h_mask, t_mask, 3).to(tl.float32) * \
        _blk_scale(qe, ke_ptr, offs_h, offs_t, h_mask, t_mask, 3)
    # identical epilogue to _indexer_logits_kernel
    score = tl.maximum(score, 0.0)
    score = score * w[:, None]
    logits = tl.sum(score, axis=0)
    if MASK:   # causal: query s at start_pos + s sees blocks [0, (start_pos + s + 1) // RATIO)
        live = (start_pos + pid_s + 1) // RATIO
        logits = tl.where(offs_t >= live, float("-inf"), logits)
    tl.store(out_ptr + pid_s.to(tl.int64) * stride_os + offs_t, logits, mask=t_mask)


def indexer_logits_i8(qc, qe, kc, ke, weights, start_pos: int | None = None, ratio: int = 4):
    """B == 1. ``qc/qe`` [S, H, 128]/[S, H, 4], ``kc/ke`` [T, 128]/[T, 4], weights [S, H].
    Returns fp32 [S, T]; with ``start_pos`` the causal mask is applied (-inf)."""
    S, H, _ = qc.shape
    T = kc.shape[0]
    out = torch.empty((S, T), dtype=torch.float32, device=qc.device)
    if T == 0 or S == 0:
        return out
    BLOCK_T = 128
    _indexer_logits_i8_kernel[(S, 1, triton.cdiv(T, BLOCK_T))](
        qc, qe, kc, ke, weights, out, S, T, 0 if start_pos is None else start_pos, ratio,
        weights.stride(0), weights.stride(1), out.stride(0),
        H=H, BLOCK_H=triton.next_power_of_2(H), BLOCK_T=BLOCK_T, MASK=start_pos is not None,
        num_warps=4, num_stages=2,
    )
    return out


@triton.jit
def _indexer_logits_i8_ms_kernel(
    qc_ptr, qe_ptr, kc_ptr, ke_ptr, w_ptr, out_ptr,
    S, T, start_pos, RATIO,
    stride_ws, stride_wh, stride_os,
    H: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_S: tl.constexpr,
    MASK: tl.constexpr,
):
    """Same per-query math as ``_indexer_logits_i8_kernel`` (same dot shapes, same epilogue);
    one program walks BLOCK_S queries and keeps the key tile + key exponents in registers."""
    pid_sb = tl.program_id(0)
    pid_t = tl.program_id(1)
    offs_h = tl.arange(0, BLOCK_H)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    h_mask = offs_h < H
    t_mask = offs_t < T
    o32 = tl.arange(0, 32)
    kb0 = tl.load(kc_ptr + offs_t[:, None].to(tl.int64) * 128 + 0 + o32[None, :], mask=t_mask[:, None], other=0)
    kb1 = tl.load(kc_ptr + offs_t[:, None].to(tl.int64) * 128 + 32 + o32[None, :], mask=t_mask[:, None], other=0)
    kb2 = tl.load(kc_ptr + offs_t[:, None].to(tl.int64) * 128 + 64 + o32[None, :], mask=t_mask[:, None], other=0)
    kb3 = tl.load(kc_ptr + offs_t[:, None].to(tl.int64) * 128 + 96 + o32[None, :], mask=t_mask[:, None], other=0)
    sk0 = ((tl.load(ke_ptr + offs_t.to(tl.int64) * 4 + 0, mask=t_mask, other=0).to(tl.int32) + 126) << 23).to(tl.float32, bitcast=True)
    sk1 = ((tl.load(ke_ptr + offs_t.to(tl.int64) * 4 + 1, mask=t_mask, other=0).to(tl.int32) + 126) << 23).to(tl.float32, bitcast=True)
    sk2 = ((tl.load(ke_ptr + offs_t.to(tl.int64) * 4 + 2, mask=t_mask, other=0).to(tl.int32) + 126) << 23).to(tl.float32, bitcast=True)
    sk3 = ((tl.load(ke_ptr + offs_t.to(tl.int64) * 4 + 3, mask=t_mask, other=0).to(tl.int32) + 126) << 23).to(tl.float32, bitcast=True)
    for i in range(BLOCK_S):
        s = pid_sb * BLOCK_S + i
        if s < S:
            qc = qc_ptr + s.to(tl.int64) * (H * 128)
            qe = qe_ptr + s.to(tl.int64) * (H * 4)
            w = tl.load(w_ptr + s * stride_ws + offs_h * stride_wh, mask=h_mask, other=0.0).to(tl.float32)
            q0 = tl.load(qc + offs_h[:, None] * 128 + 0 + o32[None, :], mask=h_mask[:, None], other=0)
            q1 = tl.load(qc + offs_h[:, None] * 128 + 32 + o32[None, :], mask=h_mask[:, None], other=0)
            q2 = tl.load(qc + offs_h[:, None] * 128 + 64 + o32[None, :], mask=h_mask[:, None], other=0)
            q3 = tl.load(qc + offs_h[:, None] * 128 + 96 + o32[None, :], mask=h_mask[:, None], other=0)
            sq0 = ((tl.load(qe + offs_h * 4 + 0, mask=h_mask, other=0).to(tl.int32) + 126) << 23).to(tl.float32, bitcast=True)
            sq1 = ((tl.load(qe + offs_h * 4 + 1, mask=h_mask, other=0).to(tl.int32) + 126) << 23).to(tl.float32, bitcast=True)
            sq2 = ((tl.load(qe + offs_h * 4 + 2, mask=h_mask, other=0).to(tl.int32) + 126) << 23).to(tl.float32, bitcast=True)
            sq3 = ((tl.load(qe + offs_h * 4 + 3, mask=h_mask, other=0).to(tl.int32) + 126) << 23).to(tl.float32, bitcast=True)
            score = tl.dot(q0, tl.trans(kb0)).to(tl.float32) * (sq0[:, None] * sk0[None, :])
            score += tl.dot(q1, tl.trans(kb1)).to(tl.float32) * (sq1[:, None] * sk1[None, :])
            score += tl.dot(q2, tl.trans(kb2)).to(tl.float32) * (sq2[:, None] * sk2[None, :])
            score += tl.dot(q3, tl.trans(kb3)).to(tl.float32) * (sq3[:, None] * sk3[None, :])
            score = tl.maximum(score, 0.0)
            score = score * w[:, None]
            logits = tl.sum(score, axis=0)
            if MASK:
                live = (start_pos + s + 1) // RATIO
                logits = tl.where(offs_t >= live, float("-inf"), logits)
            tl.store(out_ptr + s.to(tl.int64) * stride_os + offs_t, logits, mask=t_mask)


def indexer_logits_i8_ms(qc, qe, kc, ke, weights, start_pos: int | None = None, ratio: int = 4,
                         block_s: int = 16, block_t: int = 128, num_warps: int = 4):
    S, H, _ = qc.shape
    T = kc.shape[0]
    out = torch.empty((S, T), dtype=torch.float32, device=qc.device)
    if T == 0 or S == 0:
        return out
    _indexer_logits_i8_ms_kernel[(triton.cdiv(S, block_s), triton.cdiv(T, block_t))](
        qc, qe, kc, ke, weights, out, S, T, 0 if start_pos is None else start_pos, ratio,
        weights.stride(0), weights.stride(1), out.stride(0),
        H=H, BLOCK_H=triton.next_power_of_2(H), BLOCK_T=block_t, BLOCK_S=block_s,
        MASK=start_pos is not None, num_warps=num_warps, num_stages=1,
    )
    return out
