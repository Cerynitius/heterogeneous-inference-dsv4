"""Decode fusion of ``hadamard_transform`` + ``fp4_act_quant_inplace(x, 32)`` (DSV4 indexer q and
indexer-compressor key): one launch reading the bf16 rows directly, instead of the eager chain
fp32 cast -> cuBLAS ``x @ H`` -> scale -> bf16 cast -> FP4 round-trip (5 launches).

Bit-identical to the eager chain. Every product ``x[k] * H[k, j]`` is exact (H is +-1), so the
result only depends on the fp32 summation order, which is reproduced per cuBLAS kernel (measured
on sm_120, ``tests/dsv4_ep/test_hadamard_fp4.py``):
  * M % 64 == 0 (the q rows, 64 heads per token): SIMT sgemm, sequential k = 0..127.
  * M == 1 (the compressor key at B=1): gemv, 4 interleaved partials (k % 4) summed
    (p0 + p2) + (p1 + p3).
Other M (2..63: the compressor key at B >= 2) uses yet another kernel -> eager fallback.
The H sign is computed in-register: H[k, j] = (-1)^popcount(k & j) (Sylvester construction).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from freetoken.kernel.triton.dsv4.fp8_linear import _log2_ceil, _round_fp4, fp4_act_quant_inplace
from freetoken.kernel.triton.dsv4.hadamard import hadamard_transform


@triton.jit
def _parity7(v):
    v = v ^ (v >> 4)
    v = v ^ (v >> 2)
    v = v ^ (v >> 1)
    return v & 1


@triton.jit
def _hadamard_fp4_kernel(x_ptr, o_ptr, f_ptr, rows_per_b, scale, D: tl.constexpr, ORDER4: tl.constexpr,
                         RD: tl.constexpr):
    m = tl.program_id(0)
    offs = tl.arange(0, D)
    row = x_ptr + m * D
    if RD > 0:
        # interleaved decode RoPE on the last RD dims (= _rope_decode_kernel math), written back
        # in place as bf16 like the eager op, then re-read below after a CTA barrier
        b = m // rows_per_b
        pr = tl.arange(0, RD // 2)
        xr = tl.load(row + (D - RD) + 2 * pr).to(tl.float32)
        xi = tl.load(row + (D - RD) + 2 * pr + 1).to(tl.float32)
        fr = tl.load(f_ptr + b * RD + 2 * pr)
        fi = tl.load(f_ptr + b * RD + 2 * pr + 1)
        out_real = xr * fr - xi * fi
        out_imag = xr * fi + xi * fr
        tl.store(row + (D - RD) + 2 * pr, out_real.to(tl.bfloat16))
        tl.store(row + (D - RD) + 2 * pr + 1, out_imag.to(tl.bfloat16))
        tl.debug_barrier()
    if ORDER4:
        p0 = tl.zeros((D,), tl.float32)
        p1 = tl.zeros((D,), tl.float32)
        p2 = tl.zeros((D,), tl.float32)
        p3 = tl.zeros((D,), tl.float32)
        for j in tl.static_range(D // 4):
            x0 = tl.load(row + 4 * j).to(tl.float32)
            x1 = tl.load(row + 4 * j + 1).to(tl.float32)
            x2 = tl.load(row + 4 * j + 2).to(tl.float32)
            x3 = tl.load(row + 4 * j + 3).to(tl.float32)
            p0 += tl.where(_parity7((4 * j) & offs) == 1, -x0, x0)
            p1 += tl.where(_parity7((4 * j + 1) & offs) == 1, -x1, x1)
            p2 += tl.where(_parity7((4 * j + 2) & offs) == 1, -x2, x2)
            p3 += tl.where(_parity7((4 * j + 3) & offs) == 1, -x3, x3)
        y = (p0 + p2) + (p1 + p3)
    else:
        y = tl.zeros((D,), tl.float32)
        for k in tl.static_range(D):
            xk = tl.load(row + k).to(tl.float32)
            y += tl.where(_parity7(k & offs) == 1, -xk, xk)
    y = (y * scale).to(tl.bfloat16).to(tl.float32)
    # FP4 e2m1 quant+dequant round-trip per 32-wide block (= _act_quant_inplace_kernel, FP4)
    x = tl.reshape(y, (D // 32, 32))
    amax = tl.maximum(tl.max(tl.abs(x), axis=1), 6.0 * (2.0 ** -126))
    e = _log2_ceil(amax * (1.0 / 6.0))
    s = tl.exp2(e.to(tl.float32))
    q = _round_fp4(tl.clamp(x / s[:, None], -6.0, 6.0))
    out = tl.reshape((q * s[:, None]).to(tl.bfloat16), (D,))
    tl.store(o_ptr + m * D + offs, out)


def hadamard_fp4_decode(x: torch.Tensor, freqs_cis: torch.Tensor | None = None) -> torch.Tensor:
    """``fp4_act_quant_inplace(hadamard_transform(x), 32)`` on the last dim; returns a new tensor.
    With ``freqs_cis`` ([B, rd//2] complex, per decode row) the interleaved RoPE on the last rd
    dims (``apply_rotary_emb_decode``, in place on ``x``) runs first in the same launch."""
    d = x.shape[-1]
    M = x.numel() // d
    order4 = M == 1
    if d != 128 or x.dtype != torch.bfloat16 or not x.is_contiguous() or not (
            order4 or (M % 64 == 0 and M <= 256)):
        if freqs_cis is not None:
            from freetoken.models.deepseek_v4.ops import apply_rotary_emb_decode
            apply_rotary_emb_decode(x[..., -2 * freqs_cis.shape[-1]:], freqs_cis)
        return fp4_act_quant_inplace(hadamard_transform(x), 32)
    out = torch.empty_like(x)
    if freqs_cis is None:
        f, rd, rows_per_b = x, 0, 1
    else:
        f = torch.view_as_real(freqs_cis).flatten(-2).contiguous()   # [B, rd] (cos, sin)
        rd, rows_per_b = f.shape[-1], M // f.shape[0]
    _hadamard_fp4_kernel[(M,)](x, out, f, rows_per_b, d ** -0.5, D=d, ORDER4=order4, RD=rd,
                               num_warps=1)
    return out


__all__ = ["hadamard_fp4_decode"]
