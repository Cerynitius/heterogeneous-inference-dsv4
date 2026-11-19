"""Decode GEMV v2 for DeepSeek-FP4 routed experts (per-route ``c[t, r, :] = a[row] @ W[slot]^T``).

Same contract and layout as ``_decode_dsfp4_moe_kernel`` (packed e2m1 ``[S, N, K//2]``, e8m0
per-32 scales ``[S, N, K//32]``, ``slots [T, top_k]`` with -1 = skip), restructured for
memory-level parallelism at the 1-6 active routes per token a decode step has:

  * each thread loads 16 contiguous weight bytes (= one e8m0 scale group of 32 values)
    per row and iteration, so the 32-value group dot product is a thread-local sum;
  * the group's scale multiplies the group SUM (one multiply per 32 values instead of per
    value) and products accumulate in per-thread registers -- one cross-thread reduction
    per row at the very end instead of two per K step;
  * E2M1 -> fp32 either via the 16-entry LUT (``DEQ=0``) or arithmetically (``DEQ=1``):
    the 3 magnitude bits placed as fp16 exponent/mantissa bits give ``value * 2^-14``
    exactly (the e=0 code lands on the fp16 subnormal 2^-15 = 0.5 * 2^-14), and the
    2^14 folds into the group scale.

Numerics: every product is exact in fp32; only the fp32 summation order differs from the v1
kernel (group sums before scaling, register accumulation) -> noise-level, not bitwise.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _e2m1_f32_scaled(nib):
    """E2M1 code -> fp32 value * 2^-14 (exact), via fp16 bits."""
    h = ((nib & 0x7) << 9) | ((nib & 0x8) << 12)
    return h.to(tl.int16).to(tl.float16, bitcast=True).to(tl.float32)


@triton.jit
def _dsfp4_gemv_v2_kernel(
    a_ptr, packed_ptr, scale_ptr, c_ptr, topk_weights_ptr, topk_ids_ptr, lut_ptr,
    total_routes, N, K,
    stride_am, stride_ak,
    stride_pe, stride_pn, stride_pkb,
    stride_se, stride_sn, stride_sblk,
    stride_cm, stride_ck, stride_cn,
    stride_tw_m, stride_tw_k,
    stride_tid_m, stride_tid_k,
    ROWS: tl.constexpr,        # output rows per program
    BKB: tl.constexpr,         # weight bytes per row per K step (multiple of 16)
    TOP_K: tl.constexpr,
    A_ROW_IS_ROUTE: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    DEQ: tl.constexpr,         # 0 = LUT gather, 1 = fp16-bit arithmetic
    compute_type: tl.constexpr,
):
    route_id = tl.program_id(0)
    n_block_id = tl.program_id(1)
    token_id = route_id // TOP_K
    route_k = route_id - token_id * TOP_K
    slot = tl.load(topk_ids_ptr + token_id * stride_tid_m + route_k * stride_tid_k).to(tl.int64)
    if slot < 0:
        return
    a_row = route_id if A_ROW_IS_ROUTE else token_id
    a_base = a_ptr + a_row * stride_am

    offs_n = n_block_id * ROWS + tl.arange(0, ROWS)
    n_mask = offs_n < N
    NG: tl.constexpr = BKB // 16                       # scale groups per K step
    offs_g = tl.arange(0, NG)
    offs_j = tl.arange(0, 16)
    K_BYTES = K // 2
    p_rows = packed_ptr + slot * stride_pe + offs_n[:, None, None] * stride_pn
    s_rows = scale_ptr + slot * stride_se + offs_n[:, None] * stride_sn
    acc = tl.zeros((ROWS, NG), dtype=tl.float32)
    for kb0 in range(0, K_BYTES, BKB):
        byte_idx = kb0 + offs_g[:, None] * 16 + offs_j[None, :]               # [NG, 16]
        b = tl.load(p_rows + byte_idx[None, :, :] * stride_pkb,
                    mask=n_mask[:, None, None], other=0).to(tl.int32)        # [ROWS, NG, 16]
        a_lo = tl.load(a_base + (2 * byte_idx) * stride_ak).to(tl.float32)    # [NG, 16]
        a_hi = tl.load(a_base + (2 * byte_idx + 1) * stride_ak).to(tl.float32)
        if DEQ == 1:
            w_lo = _e2m1_f32_scaled(b & 0xF)
            w_hi = _e2m1_f32_scaled((b >> 4) & 0xF)
        else:
            w_lo = tl.load(lut_ptr + (b & 0xF))
            w_hi = tl.load(lut_ptr + ((b >> 4) & 0xF))
        gsum = tl.sum(w_lo * a_lo[None, :, :] + w_hi * a_hi[None, :, :], axis=2)  # [ROWS, NG]
        codes = tl.load(s_rows + (kb0 // 16 + offs_g)[None, :] * stride_sblk,
                        mask=n_mask[:, None], other=0).to(tl.float32)
        if DEQ == 1:
            sc = tl.exp2(codes - 113.0)                  # 2^(s-127) * 2^14
        else:
            sc = tl.exp2(codes - 127.0)
        acc += gsum * sc
    out = tl.sum(acc, axis=1)                                                  # [ROWS]
    if MUL_ROUTED_WEIGHT:
        weight = tl.load(topk_weights_ptr + token_id * stride_tw_m + route_k * stride_tw_k)
        out = out * weight
    c_ptrs = c_ptr + token_id * stride_cm + route_k * stride_ck + offs_n * stride_cn
    tl.store(c_ptrs, out.to(compute_type), mask=(route_id < total_routes) & n_mask)


# (rows, bytes per step, warps, deq) per projection; env FREETOKEN_DSFP4_V2_{GU,DOWN}=r,b,w,d
# swept on both cards (tests/dsv4_ep/bench_gemv_v2.py): within ~5% of the per-case best for
# 1-6 active routes, B = 1/2, gate_up and down, on the RTX PRO 6000 and the RTX PRO 5000
_DEFAULT_CFG = {"gu": (8, 256, 4, 1), "down": (8, 256, 4, 1)}


def _cfg(a_row_is_route: bool):
    key = "down" if a_row_is_route else "gu"
    env = os.environ.get(f"FREETOKEN_DSFP4_V2_{key.upper()}", "")
    if env:
        r, b, w, d = (int(v) for v in env.split(","))
        return r, b, w, d
    return _DEFAULT_CFG[key]


def gemv_v2_enabled() -> bool:
    return os.environ.get("FREETOKEN_DSFP4_GEMV", "v1") == "v2"


def dsfp4_gemv_v2(a, packed, scale, slots, topk_weights, out, *, a_row_is_route: bool,
                  mul_routed_weight: bool, cfg=None):
    """``out[t, r, :] = a[row] @ dequant(W[slots[t, r]])^T`` (x topk weight for the down
    projection); slot -1 routes are skipped (rows of ``out`` untouched)."""
    from freetoken.kernel.triton.dsv4.fused_moe import _e2m1_lut
    from freetoken.moe.fused_ds_fp4 import _compute_type

    T, top_k = slots.shape
    N = packed.shape[1]
    K = packed.shape[2] * 2
    total_routes = T * top_k
    if topk_weights is None:  # never read (MUL_ROUTED_WEIGHT=False); persistent, capture-safe
        topk_weights = _dummy_w(a.device)
    ROWS, BKB, NW, DEQ = cfg or _cfg(a_row_is_route)
    assert (K // 2) % BKB == 0, (K, BKB)
    scale_u8 = scale.view(torch.uint8)
    grid = (total_routes, triton.cdiv(N, ROWS))
    _dsfp4_gemv_v2_kernel[grid](
        a, packed, scale_u8, out, topk_weights, slots, _e2m1_lut(a.device.index),
        total_routes, N, K,
        a.stride(0), a.stride(1),
        packed.stride(0), packed.stride(1), packed.stride(2),
        scale_u8.stride(0), scale_u8.stride(1), scale_u8.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        topk_weights.stride(0) if topk_weights.ndim == 2 else 0,
        topk_weights.stride(1) if topk_weights.ndim == 2 else 0,
        slots.stride(0), slots.stride(1),
        ROWS=ROWS, BKB=BKB, TOP_K=top_k, A_ROW_IS_ROUTE=a_row_is_route,
        MUL_ROUTED_WEIGHT=mul_routed_weight, DEQ=DEQ,
        compute_type=_compute_type(a.dtype), num_warps=NW,
    )
    return out


_DUMMY: dict = {}


def _dummy_w(dev):
    t = _DUMMY.get(dev.index)
    if t is None:
        t = _DUMMY[dev.index] = torch.zeros((1, 1), dtype=torch.float32, device=dev)
    return t


__all__ = ["dsfp4_gemv_v2", "gemv_v2_enabled"]
