"""Decode fusion of the per-row attention input chains (DSV4):
  kv: rms_norm(kv_norm) -> rope(last rd) -> FP8 round-trip on [:-rd] (block 64) -> window-pool write
  q:  rms_norm(no weight, per head) -> rope(last rd)
one launch each instead of 4 / 2. Bit-identical: each stage is the original kernel's code on the
same tile shapes and num_warps (rms_norm [BLOCK_D] / 4 warps, rope [128 pairs] masked, act quant
[32, 64] tiles), stages chained through the output row in global memory exactly where the eager
ops stored bf16, with a CTA barrier in between (``tests/dsv4_ep/test_norm_rope.py``).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from freetoken.kernel.triton.dsv4.fp8_linear import _log2_ceil
from freetoken.kernel.triton.e4m3_compat import e4m3_native_cx, round_e4m3


@triton.jit
def _norm_rope_kernel(
    x_ptr, w_ptr, out_ptr, f_ptr, pool_ptr, slot_ptr,
    rows_per_b, eps, stride_pool,
    D: tl.constexpr, RD: tl.constexpr, HAS_W: tl.constexpr, QUANT: tl.constexpr,
    STORE: tl.constexpr,
):
    row = tl.program_id(0)
    # --- rms_norm (= norm._rmsnorm_kernel, BLOCK_D == D) ---
    offs = tl.arange(0, D)
    x = tl.load(x_ptr + row * D + offs).to(tl.float32)
    var = tl.sum(x * x, axis=0) / D
    y = x * tl.rsqrt(var + eps)
    if HAS_W:
        y = y * tl.load(w_ptr + offs).to(tl.float32)
    o_row = out_ptr + row * D
    tl.store(o_row + offs, y.to(tl.bfloat16))
    tl.debug_barrier()
    # --- interleaved decode rope on the last RD dims (= rope._rope_decode_kernel, BLOCK_SIZE 128) ---
    b = row // rows_per_b
    offs_pair = tl.arange(0, 128)
    pmask = offs_pair < (RD // 2)
    r_ptr = o_row + (D - RD) + offs_pair * 2
    x_real = tl.load(r_ptr, mask=pmask, other=0.0).to(tl.float32)
    x_imag = tl.load(r_ptr + 1, mask=pmask, other=0.0).to(tl.float32)
    f_real = tl.load(f_ptr + b * RD + offs_pair * 2, mask=pmask, other=0.0)
    f_imag = tl.load(f_ptr + b * RD + offs_pair * 2 + 1, mask=pmask, other=0.0)
    out_real = x_real * f_real - x_imag * f_imag
    out_imag = x_real * f_imag + x_imag * f_real
    tl.store(r_ptr, out_real, mask=pmask)
    tl.store(r_ptr + 1, out_imag, mask=pmask)
    if QUANT:
        tl.debug_barrier()
        # --- FP8 e4m3 round-trip on [:D-RD], block 64 (= fp8_linear._act_quant_inplace_kernel,
        #     [BLOCK_M=32, 64] tile, this row in tile row 0) ---
        offs_m = tl.arange(0, 32)
        m_mask = offs_m < 1
        for blk in tl.static_range((D - RD) // 64):
            offs_k = blk * 64 + tl.arange(0, 64)
            ptrs = o_row + offs_m[:, None] * 0 + offs_k[None, :]
            xq = tl.load(ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)
            amax = tl.max(tl.abs(xq), axis=1)
            amax = tl.maximum(amax, 1e-4)
            e = _log2_ceil(amax * (1.0 / 448.0))
            s = tl.exp2(e.to(tl.float32))
            q = tl.clamp(xq / s[:, None], -448.0, 448.0)
            if e4m3_native_cx():
                q = q.to(tl.float8e4nv).to(tl.float32)
            else:
                q = round_e4m3(q)
            tl.store(ptrs, (q * s[:, None]).to(tl.bfloat16), mask=m_mask[:, None])
    if STORE:
        tl.debug_barrier()
        slot = tl.load(slot_ptr + row)
        tl.store(pool_ptr + slot * stride_pool + offs, tl.load(o_row + offs))


def _freqs_real(freqs_cis):
    return torch.view_as_real(freqs_cis).flatten(-2).contiguous()   # [B, rd] (cos, sin)


def kv_norm_rope_quant_store(kv: torch.Tensor, weight: torch.Tensor, eps: float, freqs_cis: torch.Tensor,
                             pool: torch.Tensor, window_slot: torch.Tensor) -> torch.Tensor:
    """``kv`` [B, 1, D] bf16 -> normalized / roped / FP8-round-tripped kv (new tensor), also
    written to ``pool[window_slot[b]]`` (= kv_norm + apply_rotary_emb_decode +
    act_quant_fp8_inplace(kv[..., :-rd], 64) + store_window)."""
    B = kv.shape[0]
    D = kv.shape[-1]
    f = _freqs_real(freqs_cis)
    out = torch.empty(B, D, dtype=torch.bfloat16, device=kv.device)
    _norm_rope_kernel[(B,)](
        kv.reshape(B, D).contiguous(), weight, out, f, pool, window_slot.contiguous(),
        1, eps, pool.stride(0),
        D=D, RD=f.shape[-1], HAS_W=True, QUANT=True, STORE=True, num_warps=4,
    )
    return out.view(kv.shape)


def q_norm_rope(q: torch.Tensor, eps: float, freqs_cis: torch.Tensor) -> torch.Tensor:
    """``q`` [B, 1, H, D] bf16 -> rms_norm(q, None, eps) then decode rope on the last rd (new tensor)."""
    B, _, H, D = q.shape
    f = _freqs_real(freqs_cis)
    out = torch.empty(B * H, D, dtype=torch.bfloat16, device=q.device)
    _norm_rope_kernel[(B * H,)](
        q.reshape(B * H, D).contiguous(), q, out, f, out, out,
        H, eps, 0,
        D=D, RD=f.shape[-1], HAS_W=False, QUANT=False, STORE=False, num_warps=4,
    )
    return out.view(q.shape)


__all__ = ["kv_norm_rope_quant_store", "q_norm_rope"]
