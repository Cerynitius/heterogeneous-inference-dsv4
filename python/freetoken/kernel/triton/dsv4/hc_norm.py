"""Fused mHC pre-norm helpers (2026-08-28 tail-fusion pass).

Per hc_pre call the eager chain was: big fp32 cast -> square -> mean-reduce ->
rsqrt -> cublas gemv (dot_kernel + reduce_1Block) -> broadcast mul, i.e. ~7
launches x 90 calls/step. Here:

  * ``hc_rms_cast``  -- one kernel: bf16->fp32 copy (the gemv still needs fp32
    input) + sum-of-squares + rsqrt(mean+eps), one CTA per token (deterministic).
  * ``hc_mix_gemv``  -- [MIX, KDIM] fp32 gemv with the rsqrt scale folded into the
    epilogue; one CTA per (token, row), sequential k-loop (no atomics, so
    run-to-run deterministic like the cublas dot+reduce pair it replaces).

Numerics: identical inputs/ops in fp32; only reduction order differs from ATen /
cublas (epsilon-level). CUDA-graph safe: fixed shapes, no host sync."""
import torch
import triton
import triton.language as tl


@triton.jit
def _hc_rms_cast_kernel(
    x_ptr, xf_ptr, rs_ptr, D, eps,
    stride_xm, stride_fm,
    BLOCK: tl.constexpr,
):
    m = tl.program_id(0)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for start in range(0, tl.cdiv(D, BLOCK)):
        offs = start * BLOCK + tl.arange(0, BLOCK)
        mask = offs < D
        v = tl.load(x_ptr + m * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(xf_ptr + m * stride_fm + offs, v, mask=mask)
        acc += v * v
    ssq = tl.sum(acc, axis=0)
    tl.store(rs_ptr + m, 1.0 / tl.sqrt(ssq / D + eps))


def hc_rms_cast(x: torch.Tensor, eps: float):
    """``x`` [M, D] (bf16/fp16/fp32) -> (``xf`` [M, D] fp32, ``rs`` [M] fp32)."""
    M, D = x.shape
    xf = torch.empty(M, D, dtype=torch.float32, device=x.device)
    rs = torch.empty(M, dtype=torch.float32, device=x.device)
    _hc_rms_cast_kernel[(M,)](
        x, xf, rs, D, eps, x.stride(0), xf.stride(0),
        BLOCK=1024, num_warps=4,
    )
    return xf, rs


@triton.jit
def _hc_mix_gemv_kernel(
    xf_ptr, w_ptr, rs_ptr, out_ptr, D,
    stride_fm, stride_wr, stride_om,
    MIX: tl.constexpr, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    m = pid // MIX
    r = pid - m * MIX
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for start in range(0, tl.cdiv(D, BLOCK)):
        offs = start * BLOCK + tl.arange(0, BLOCK)
        mask = offs < D
        xv = tl.load(xf_ptr + m * stride_fm + offs, mask=mask, other=0.0)
        wv = tl.load(w_ptr + r * stride_wr + offs, mask=mask, other=0.0)
        acc += xv * wv
    dot = tl.sum(acc, axis=0)
    rs = tl.load(rs_ptr + m)
    tl.store(out_ptr + m * stride_om + r, dot * rs)


def hc_mix_gemv(xf: torch.Tensor, w: torch.Tensor, rs: torch.Tensor) -> torch.Tensor:
    """``out[m, r] = rs[m] * dot(w[r], xf[m])``; ``w`` [MIX, D] fp32."""
    M, D = xf.shape
    MIX = w.shape[0]
    out = torch.empty(M, MIX, dtype=torch.float32, device=xf.device)
    _hc_mix_gemv_kernel[(M * MIX,)](
        xf, w, rs, out, D, xf.stride(0), w.stride(0), out.stride(0),
        MIX=MIX, BLOCK=2048, num_warps=8,
    )
    return out


# ---- prefill (large M) path, 2026-11-18 ------------------------------------------------------
# hc_rms_cast + hc_mix_gemv re-read each fp32 row MIX (=24) times through one CTA per (m, r):
# ~1.1 ms per call at M=3443, D=16384 on the PRO 6000 vs a ~70 us bf16 stream bound. Here one CTA
# owns BM rows, streams the bf16 row block once, and keeps the sum of squares and all MIX dots in
# registers (fp32 FMA dot, w [MIX, D] fp32 stays L2-resident). Same fp32 math, different
# reduction order (epsilon level).
@triton.jit
def _hc_mix_rms_rows_kernel(
    x_ptr, w_ptr, part_ptr, M, D, KS,
    stride_xm, stride_wr,
    MIX: tl.constexpr, MIXP: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr, PREC: tl.constexpr,
):
    # grid (row blocks, K splits); partials [SPLIT, M, MIXP + 1] (last column = sum of squares)
    pid = tl.program_id(0)
    sp = tl.program_id(1)
    rows = pid * BM + tl.arange(0, BM)
    rm = rows < M
    rr = tl.arange(0, MIXP)
    wm = rr < MIX
    acc = tl.zeros((BM, MIXP), dtype=tl.float32)
    ssq = tl.zeros((BM, BK), dtype=tl.float32)
    for k0 in range(sp * KS, sp * KS + KS, BK):
        kk = k0 + tl.arange(0, BK)
        x = tl.load(x_ptr + rows[:, None].to(tl.int64) * stride_xm + kk[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
        w = tl.load(w_ptr + rr[None, :] * stride_wr + kk[:, None], mask=wm[None, :], other=0.0)
        ssq += x * x
        acc = tl.dot(x, w, acc, input_precision=PREC)
    base = part_ptr + (sp * M + rows[:, None]).to(tl.int64) * (MIXP + 1)
    tl.store(base + rr[None, :], acc, mask=rm[:, None])
    tl.store(base + MIXP, tl.sum(ssq, axis=1)[:, None], mask=rm[:, None])


@triton.jit
def _hc_mix_rms_fin_kernel(part_ptr, out_ptr, M, D, eps, stride_om,
                           SPLIT: tl.constexpr, MIX: tl.constexpr, MIXP: tl.constexpr, BM: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    rm = rows < M
    rr = tl.arange(0, MIXP)
    acc = tl.zeros((BM, MIXP), dtype=tl.float32)
    ssq = tl.zeros((BM,), dtype=tl.float32)
    for sp in tl.static_range(SPLIT):
        base = part_ptr + (sp * M + rows).to(tl.int64) * (MIXP + 1)
        acc += tl.load(base[:, None] + rr[None, :], mask=rm[:, None], other=0.0)
        ssq += tl.load(base + MIXP, mask=rm, other=0.0)
    rs = 1.0 / tl.sqrt(ssq / D + eps)
    tl.store(out_ptr + rows[:, None] * stride_om + rr[None, :], acc * rs[:, None], mask=rm[:, None] & (rr[None, :] < MIX))


_N_SM = {}


def hc_mix_rms_rows(x: torch.Tensor, w: torch.Tensor, eps: float, BM: int = 64, BK: int = 64,
                    num_warps: int = 4, prec: str = "tf32x3", split: int | None = None) -> torch.Tensor:
    """``out[m, r] = rsqrt(mean(x[m]^2) + eps) * dot(w[r], x[m])``, x [M, D] bf16/fp32, w [MIX, D] fp32.
    Prefill path (large M): tf32x3 tensor-core dot (~fp32 accuracy), split-K to fill the SMs."""
    M, D = x.shape
    MIX = w.shape[0]
    assert x.stride(1) == 1 and w.stride(1) == 1
    MIXP = max(16, triton.next_power_of_2(MIX))
    out = torch.empty(M, MIX, dtype=torch.float32, device=x.device)
    if not M:
        return out
    nb = triton.cdiv(M, BM)
    if split is None:
        dev = x.device.index
        if dev not in _N_SM:
            _N_SM[dev] = torch.cuda.get_device_properties(x.device).multi_processor_count
        split = 1
        while nb * split < 2 * _N_SM[dev] and D % (BK * split * 2) == 0 and split < 16:
            split *= 2
    assert D % (BK * split) == 0
    part = torch.empty(split, M, MIXP + 1, dtype=torch.float32, device=x.device)
    _hc_mix_rms_rows_kernel[(nb, split)](
        x, w, part, M, D, D // split, x.stride(0), w.stride(0),
        MIX=MIX, MIXP=MIXP, BM=BM, BK=BK, PREC=prec, num_warps=num_warps)
    _hc_mix_rms_fin_kernel[(triton.cdiv(M, 64),)](part, out, M, D, eps, out.stride(0),
                                                  SPLIT=split, MIX=MIX, MIXP=MIXP, BM=64, num_warps=4)
    return out
