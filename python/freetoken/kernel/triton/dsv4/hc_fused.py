"""mHC pre-mix decode fusion, take 2 (take 1 -- everything in one CTA -- destroyed
parallelism and measured 39 tok/s; reverted).

This version keeps the parallel grid shape of ``hc_mix_gemv`` ((m, r) programs) and
folds the rms INTO it: each program redundantly accumulates the row's sum of squares
alongside its dot (16K extra FMAs per program, free next to the loads) and reads the
bf16 stream DIRECTLY -- eliminating the ``hc_rms_cast`` launch and the [M, HC*D]
fp32 intermediate (64 KB written+read per call, ~87 calls/step). Sinkhorn and
pre_combine stay as their existing single launches.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _hc_mix_rms_kernel(
    x_ptr,                    # [M, KD] bf16
    w_ptr,                    # [MIX, KD] fp32
    out_ptr,                  # [M, MIX] fp32
    KD: tl.constexpr, norm_eps,
    stride_xm, stride_wr, stride_om,
    MIX: tl.constexpr, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    m = pid // MIX
    r = pid - m * MIX
    dot = tl.zeros((BLOCK,), tl.float32)
    ssq = tl.zeros((BLOCK,), tl.float32)
    # KD constexpr + static_range: the 8 load pairs issue up front (latency-bound at 24
    # CTAs); per-lane accumulation order unchanged -> bit-identical to the rolled loop.
    for t in tl.static_range(0, (KD + BLOCK - 1) // BLOCK):
        offs = t * BLOCK + tl.arange(0, BLOCK)
        mask = offs < KD
        xv = tl.load(x_ptr + m * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
        wv = tl.load(w_ptr + r * stride_wr + offs, mask=mask, other=0.0)
        dot += xv * wv
        ssq += xv * xv
    rs = 1.0 / tl.sqrt(tl.sum(ssq, axis=0) / KD + norm_eps)
    tl.store(out_ptr + m * stride_om + r, tl.sum(dot, axis=0) * rs)


def hc_pre_fused(x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor,
                 hc_base: torch.Tensor, norm_eps: float, hc_eps: float,
                 iters: int, out_dtype: torch.dtype):
    """``x`` [M, HC, D] (bf16); returns (y [M, D] out_dtype, post [M, HC] fp32,
    comb [M, HC, HC] fp32). Three launches (was four + an fp32 intermediate)."""
    from freetoken.kernel.triton.dsv4.hc import hc_pre_combine
    from freetoken.kernel.triton.dsv4.sinkhorn import hc_split_sinkhorn

    M, HC, D = x.shape
    KD = HC * D
    MIX = hc_fn.shape[0]
    xf = x.reshape(M, KD)
    mixes = torch.empty(M, MIX, dtype=torch.float32, device=x.device)
    _hc_mix_rms_kernel[(M * MIX,)](
        xf, hc_fn, mixes, KD, norm_eps,
        xf.stride(0), hc_fn.stride(0), mixes.stride(0),
        MIX=MIX, BLOCK=2048, num_warps=8,
    )
    pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, HC, iters, hc_eps)
    y = hc_pre_combine(x, pre, out_dtype)
    return y, post, comb


# ---------------------------------------------------------------------------------------------
# hc_post (sublayer L) + hc_pre (sublayer L+1) at a decode boundary in two launches instead of
# four (post_combine, mix_rms, sinkhorn, pre_combine), bit-identical to that chain:
#  * _hc_post_mix_kernel: the mix_rms grid (M*MIX programs, same BLOCK / num_warps / chunk order)
#    recomputes the post-combine of its chunk in registers (elementwise, same per-element ops as
#    _hc_post_combine_kernel), rounds it to the stream dtype exactly where the eager path stored it,
#    and feeds the same value into the MIX dot; program r == 0 also writes the new residual.
#  * _hc_sinkhorn_pre_kernel: the sinkhorn program (same code, num_warps=1 -> same [HC, HC] layout)
#    per (token, column block) followed by pre_combine of that block (pre kept in registers;
#    extracted with where+sum over exact zeros).
# ---------------------------------------------------------------------------------------------
@triton.jit
def _hc_post_mix_kernel(
    a_ptr, res_ptr, post_ptr, comb_ptr, xnew_ptr, w_ptr, out_ptr, norm_eps,
    D: tl.constexpr, HC: tl.constexpr, MIX: tl.constexpr, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    m = pid // MIX
    r = pid - m * MIX
    KD: tl.constexpr = HC * D
    dot = tl.zeros((BLOCK,), tl.float32)
    ssq = tl.zeros((BLOCK,), tl.float32)
    for t in tl.static_range(0, KD // BLOCK):
        q = (t * BLOCK) // D
        offs_d = (t * BLOCK) % D + tl.arange(0, BLOCK)
        av = tl.load(a_ptr + m * D + offs_d).to(tl.float32)
        pq = tl.load(post_ptr + m * HC + q).to(tl.float32)
        c0 = tl.load(comb_ptr + m * HC * HC + q).to(tl.float32)
        r0 = tl.load(res_ptr + m * KD + offs_d).to(tl.float32)
        # _hc_post_combine_kernel's compiled contraction of post*a + c0*r0 (packed f32x2) differs
        # by lane: even d -> fma(c0, r0, post*a), odd d -> fma(post, a, c0*r0); then c1..c3 fma'd
        # in order. Explicit tl.fma pins that per element.
        acc = tl.where(offs_d % 2 == 0, tl.fma(c0, r0, pq * av), tl.fma(pq, av, c0 * r0))
        for p in tl.static_range(1, HC):
            c = tl.load(comb_ptr + m * HC * HC + p * HC + q).to(tl.float32)
            rv = tl.load(res_ptr + m * KD + p * D + offs_d).to(tl.float32)
            acc = tl.fma(c, rv, acc)
        xb = acc.to(xnew_ptr.dtype.element_ty)
        if r == 0:
            tl.store(xnew_ptr + m * KD + q * D + offs_d, xb)
        xv = xb.to(tl.float32)
        wv = tl.load(w_ptr + r * KD + t * BLOCK + tl.arange(0, BLOCK))
        dot += xv * wv
        ssq += xv * xv
    rs = 1.0 / tl.sqrt(tl.sum(ssq, axis=0) / KD + norm_eps)
    tl.store(out_ptr + m * MIX + r, tl.sum(dot, axis=0) * rs)


@triton.jit
def _hc_sinkhorn_pre_kernel(
    mixes_ptr, scale_ptr, base_ptr, x_ptr, y_ptr, post_ptr, comb_ptr,
    D: tl.constexpr, HC: tl.constexpr, ITERS: tl.constexpr, EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cb = tl.program_id(1)
    h = tl.arange(0, HC)
    m = mixes_ptr + row * ((2 + HC) * HC)
    sc0 = tl.load(scale_ptr + 0)
    sc1 = tl.load(scale_ptr + 1)
    sc2 = tl.load(scale_ptr + 2)
    pre = tl.sigmoid(tl.load(m + h) * sc0 + tl.load(base_ptr + h)) + EPS
    post = 2.0 * tl.sigmoid(tl.load(m + HC + h) * sc1 + tl.load(base_ptr + HC + h))
    idx = h[:, None] * HC + h[None, :]
    c = tl.load(m + 2 * HC + idx) * sc2 + tl.load(base_ptr + 2 * HC + idx)
    c = c - tl.max(c, axis=1)[:, None]
    c = tl.exp(c)
    c = c / tl.sum(c, axis=1)[:, None]
    c = c + EPS
    c = c / (tl.sum(c, axis=0)[None, :] + EPS)
    for _ in range(ITERS - 1):
        c = c / (tl.sum(c, axis=1)[:, None] + EPS)
        c = c / (tl.sum(c, axis=0)[None, :] + EPS)
    if cb == 0:
        tl.store(post_ptr + row * HC + h, post)
        tl.store(comb_ptr + row * HC * HC + idx, c)
    offs = cb * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for hh in tl.static_range(HC):
        xh = tl.load(x_ptr + row * HC * D + hh * D + offs).to(tl.float32)
        p = tl.sum(tl.where(h == hh, pre, 0.0), axis=0)
        acc += p * xh
    tl.store(y_ptr + row * D + offs, acc.to(y_ptr.dtype.element_ty))


def hc_post_pre(a: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor,
                hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor,
                norm_eps: float, hc_eps: float, iters: int):
    """``hc_post(a, residual, post, comb)`` then ``hc_pre_fused`` of the next sublayer, in two
    launches. ``a`` [M, D], ``residual`` [M, HC, D] (stream dtype), ``post`` [M, HC],
    ``comb`` [M, HC, HC] fp32. Returns (new residual [M, HC, D], y [M, D], post', comb')."""
    M, HC, D = residual.shape
    MIX = hc_fn.shape[0]
    a = a.reshape(M, D).contiguous()
    residual = residual.contiguous()
    xnew = torch.empty_like(residual)
    mixes = torch.empty(M, MIX, dtype=torch.float32, device=a.device)
    _hc_post_mix_kernel[(M * MIX,)](
        a, residual, post.contiguous(), comb.contiguous(), xnew, hc_fn, mixes, norm_eps,
        D=D, HC=HC, MIX=MIX, BLOCK=2048, num_warps=8,
    )
    y = torch.empty(M, D, dtype=a.dtype, device=a.device)
    post2 = torch.empty(M, HC, device=a.device, dtype=torch.float32)
    comb2 = torch.empty(M, HC, HC, device=a.device, dtype=torch.float32)
    BLOCK = 1024
    _hc_sinkhorn_pre_kernel[(M, D // BLOCK)](
        mixes, hc_scale.float().contiguous(), hc_base.float().contiguous(), xnew, y, post2, comb2,
        D=D, HC=HC, ITERS=iters, EPS=hc_eps, BLOCK=BLOCK, num_warps=1,
    )
    return xnew, y, post2, comb2


__all__ = ["hc_pre_fused", "hc_post_pre"]
