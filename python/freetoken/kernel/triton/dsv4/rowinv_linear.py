"""Row-invariant BF16 GEMM: ``out[M, N] = x[M, K] @ w[N, K]^T`` with FP32 accumulation.

Every output row is computed by the same instruction sequence whatever M is and wherever the
row sits in its tile: one fixed tile config, no split-K, M not specialized (``do_not_specialize``),
tensor-core ``tl.dot`` over K in a fixed BLOCK_K order. So a token's projection does not depend on
how many tokens share the launch -- unlike cuBLAS, whose algorithm (and fp32 summation order)
changes with M, which made prefill logits depend on the chunk size and a prefix-cache hit (short
extend) differ from a cold prefill of the same prompt.

Used for the prefill-path projections that went through cuBLAS: the compressor wkv|wgate (fp32
out; the inputs are BF16 values, so bf16 x bf16 products are exact in fp32), the indexer
weights_proj (bf16 out) and the MoE router (fp32 out, M > 1).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["M"])
def _rowinv_gemm_kernel(
    x_ptr, w_ptr, o_ptr, M, N, K,
    stride_xm, stride_xk, stride_wn, stride_wk, stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, KS: tl.constexpr,
):
    """K is cut into KS contiguous slices, each reduced by tensor-core dots into its own fp32
    accumulator; the slice partials are then added in order. A single running accumulator over
    all of K=4096 was ~17x less accurate than cuBLAS fp32 (sequential summation); 16 slices
    match cuBLAS fp32 accuracy (max |err| 2.5e-6 vs 2.3e-6 against fp64) at 2x its speed."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    n_mask = offs_n < N
    xp = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    wp = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
    SL = K // KS
    tot = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for s in tl.static_range(KS):
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(s * SL, (s + 1) * SL, BLOCK_K):
            a = tl.load(xp + k0 * stride_xk, mask=m_mask[:, None], other=0.0)
            b = tl.load(wp + k0 * stride_wk, mask=n_mask[:, None], other=0.0)
            acc = tl.dot(a, tl.trans(b), acc)
        tot = tot + acc
    tl.store(o_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
             tot.to(o_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :])


_CFG = dict(BLOCK_M=64, BLOCK_N=64, BLOCK_K=64, KS=16, num_warps=4, num_stages=3)


def rowinv_linear(x: torch.Tensor, w: torch.Tensor, out_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """``x [..., K]`` (bf16, or fp32 holding bf16 values) @ ``w [N, K]`` bf16 ^T -> ``[..., N]``
    in ``out_dtype``, row-invariant (see module doc)."""
    *lead, K = x.shape
    N = w.shape[0]
    assert w.dtype == torch.bfloat16 and w.shape[1] == K, (w.shape, K)
    assert K % (_CFG["KS"] * _CFG["BLOCK_K"]) == 0, K
    x2 = x.reshape(-1, K)
    if x2.dtype != torch.bfloat16:
        x2 = x2.to(torch.bfloat16)  # exact: callers pass BF16 values held in fp32
    M = x2.shape[0]
    out = torch.empty((M, N), dtype=out_dtype, device=x.device)
    if M == 0:
        return out.reshape(*lead, N)
    grid = (triton.cdiv(M, _CFG["BLOCK_M"]), triton.cdiv(N, _CFG["BLOCK_N"]))
    _rowinv_gemm_kernel[grid](
        x2, w, out, M, N, K,
        x2.stride(0), x2.stride(1), w.stride(0), w.stride(1), out.stride(0), out.stride(1),
        **_CFG,
    )
    return out.reshape(*lead, N)


__all__ = ["rowinv_linear"]
