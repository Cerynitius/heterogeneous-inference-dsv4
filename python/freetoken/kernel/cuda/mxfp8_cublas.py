"""Prefill FP8 linear on cuBLASLt MXFP8 (sm_120 supports 1x32 e8m0 block scaling; the DeepSeek
1x128 / 128x128 recipe is SM90-only in cuBLAS). DSV4 weights carry one e8m0 per 128x128 block and
act_quant_fp8 one per 1x128: both are replicated onto the 1x32 MX grid, so every product is the
same exact value; only the fp32 accumulation order is cuBLAS's (11-18: 1.04-1.5x vs the Triton
GEMM on PRO 4000 at M=3443). Weight tensors are used as they are (column-major view), only a
swizzled scale copy is cached per weight (N*K/32 bytes)."""

from __future__ import annotations

import os

import torch

ENABLED = os.environ.get("FREETOKEN_DSV4_MXFP8_CUBLAS", "0") == "1"
MIN_M = int(os.environ.get("FREETOKEN_DSV4_MXFP8_MIN_M", "256"))
_WCACHE: dict = {}


def _to_blocked(m: torch.Tensor) -> torch.Tensor:
    """[rows, cols] e8m0 scales -> cuBLAS 128x4 tile-swizzled flat layout (SWIZZLE_32_4_4)."""
    rows, cols = m.shape
    nr, nc = (rows + 127) // 128, (cols + 3) // 4
    if rows != nr * 128 or cols != nc * 4:
        p = torch.zeros(nr * 128, nc * 4, dtype=m.dtype, device=m.device)
        p[:rows, :cols] = m
    else:
        p = m
    return p.view(nr, 128, nc, 4).permute(0, 2, 1, 3).reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


def _weight_scale(w: torch.Tensor, sb: torch.Tensor) -> torch.Tensor:
    key = (w.data_ptr(), sb.data_ptr(), tuple(w.shape))
    t = _WCACHE.get(key)
    if t is None:
        codes = sb.view(torch.uint8) if sb.dtype != torch.uint8 else sb
        N, K = w.shape
        full = codes.repeat_interleave(128, 0)[:N].repeat_interleave(4, 1)[:, : K // 32].contiguous()
        t = _to_blocked(full.view(torch.float8_e8m0fnu))
        _WCACHE[key] = t
    return t


def usable(M: int, w: torch.Tensor, out: torch.Tensor | None = None) -> bool:
    if not ENABLED or M < MIN_M or w.dtype != torch.float8_e4m3fn:
        return False
    N, K = w.shape
    if K % 128 or N % 16:
        return False
    if out is not None and (out.stride(1) != 1 or out.stride(0) != N or out.dtype != torch.bfloat16):
        return False
    return True


def mx_linear(a8: torch.Tensor, sa_codes: torch.Tensor, w: torch.Tensor, sb: torch.Tensor,
              out: torch.Tensor) -> torch.Tensor:
    """out[M, N] (bf16) = (a8 x 2^sa) @ (w x 2^sb)^T; a8 [M, K] fp8 codes, sa_codes [M, K/128] uint8
    (act_quant_fp8), w [N, K] fp8, sb [N/128, K/128] e8m0/uint8."""
    M = a8.shape[0]
    sa = sa_codes.view(torch.uint8) if sa_codes.dtype != torch.uint8 else sa_codes
    sa32 = _to_blocked(sa[:M].repeat_interleave(4, dim=1).view(torch.float8_e8m0fnu))
    a = a8[:M] if a8.dtype == torch.float8_e4m3fn else a8[:M].view(torch.float8_e4m3fn)
    return torch._scaled_mm(a, w.t(), scale_a=sa32, scale_b=_weight_scale(w, sb), out_dtype=torch.bfloat16, out=out)
