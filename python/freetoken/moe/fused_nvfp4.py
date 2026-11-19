"""Host orchestration for the inline-dequant NVFP4 fused-MoE path.

Mirrors :mod:`freetoken.moe.fused` (gemm1 -> act -> gemm2 -> sum-reduce) but the two
grouped GEMMs read the NVFP4 expert cache directly and dequantize inside the K-loop,
so no BF16 copy of the experts is ever materialized.
"""

from __future__ import annotations

from typing import Any, Dict

import os
import torch
import triton
import triton.language as tl

from freetoken.kernel import moe_sum_reduce_triton
from freetoken.kernel.triton.e4m3_compat import e4m3_kernel_view
from freetoken.kernel.triton.nvfp4_fused_moe import (
    _decode_nvfp4_marlin_kernel,
    _decode_nvfp4_moe_kernel,
    _e2m1_lut,
    _prefill_nvfp4_moe_kernel,
)
from freetoken.layers import (
    gelu_and_mul,
    gelu_tanh_and_mul,
    silu_and_mul,
    swigluoai_and_mul,
)
from freetoken.moe.fused import moe_align_block_size

_ACT = {"silu": silu_and_mul, "gelu": gelu_and_mul, "gelu_tanh": gelu_tanh_and_mul}


def _run_act(
    activation: str,
    gate_up: torch.Tensor,
    out: torch.Tensor,
    act_alpha: float,
    act_limit: float,
) -> None:
    """gemm1 -> gemm2 activation dispatch. ``swigluoai`` (MiniMax-M3, clamped
    gpt-oss swiglu over the banks' uninterleaved [gate; up] halves) carries the
    per-model ``act_alpha``/``act_limit`` scalars; the plain *_and_mul kinds
    ignore them."""
    if activation == "swigluoai":
        swigluoai_and_mul(gate_up, out, alpha=act_alpha, limit=act_limit)
        return
    if activation == "swiglu_clamp":
        # GLM-5.3: silu(min(gate, limit)) * clamp(up, -limit, limit) -- the dsv4 fused
        # kernel implements exactly this (verified bit-exact vs the HF formula).
        from freetoken.kernel.triton.dsv4.fused_moe import fused_swiglu

        out.copy_(fused_swiglu(gate_up, act_limit))
        return
    _ACT[activation](gate_up, out)

# Decode is captured into a CUDA graph, so the config must be fixed (no triton.autotune,
# which benchmarks at run time). Tuned offline against the NVFP4 decode kernels.
# These drive the original LUT-gather decode (_decode_gemm), kept only for A/B.
_DECODE_BLOCK_N = 64
_DECODE_BLOCK_KB = 128
_DECODE_WARPS = 4

# Marlin-style decode config (int32 wide loads + deferred reduction). Offline sweep over
# the qwen35/qwen3moe (I=512/768) decode shapes picked BLOCK_N=16, BLOCK_KW=16 (== 128
# k-values/iter), 4 warps -- the wide load lifts the gate/up GEMM ~43%->~51% of peak BW.
# 2026-08-28 microbench sweep (bench_marlin.py): gate_up 42.4us->35.9us (-15%), down par.
# Reduction-order change only (max rel 0.6% vs 16/16/4, within NVFP4 noise).
_DECODE_MARLIN_BLOCK_N = int(os.environ.get("FREETOKEN_MARLIN_BN", "16"))
_DECODE_MARLIN_BLOCK_KW = int(os.environ.get("FREETOKEN_MARLIN_BKW", "32"))
_DECODE_MARLIN_WARPS = int(os.environ.get("FREETOKEN_MARLIN_WARPS", "2"))
# down GEMV (a = per-route rows, K = I): 1 warp -- Naive H4096/I2048 microbench (64 slots,
# L2-proof, 2026-09-29): down+fused-silu 29-30 us vs 44-45 us at 2 warps; gate_up stays
# fastest at 2 warps (52 vs 62 us at 1).
_DECODE_MARLIN_WARPS_DOWN = int(os.environ.get("FREETOKEN_MARLIN_WARPS_DOWN", "1"))
# FREETOKEN_NVFP4_SWIGLU_FUSED=1: fold 'swiglu_clamp' (GLM-5.3 routed experts) into the
# marlin down GEMV like FUSE_SILU does for silu -- drops the fused_swiglu launch + the
# copy_ into ic2 (both per phase in the split path). Bit-identical (same fp32 math and
# bf16 rounding; see kernel/triton/nvfp4_marlin_swiglu.py). Read once; default off.
_SWIGLU_FUSED = os.environ.get("FREETOKEN_NVFP4_SWIGLU_FUSED", "0") == "1"


def _tl_dtype(dt: torch.dtype):
    if dt == torch.bfloat16:
        return tl.bfloat16
    if dt == torch.float16:
        return tl.float16
    return tl.float32


def _decode_gemm(
    a: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    glob: torch.Tensor,
    c: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    mul_routed_weight: bool,
    a_row_is_route: bool,
) -> None:
    M, top_k = topk_ids.shape
    N = packed.shape[1]
    K = packed.shape[2] * 2
    scale = e4m3_kernel_view(scale)
    total_routes = M * top_k
    grid = (total_routes, triton.cdiv(N, _DECODE_BLOCK_N))
    _decode_nvfp4_moe_kernel[grid](
        a, packed, scale, glob, c, topk_weights, topk_ids,
        _e2m1_lut(a.device.index),
        total_routes, N, K,
        a.stride(0), a.stride(1),
        packed.stride(0), packed.stride(1), packed.stride(2),
        scale.stride(0), scale.stride(1), scale.stride(2),
        glob.stride(0), glob.stride(1),
        c.stride(0), c.stride(1), c.stride(2),
        topk_weights.stride(0), topk_weights.stride(1),
        topk_ids.stride(0), topk_ids.stride(1),
        BLOCK_SIZE_N=_DECODE_BLOCK_N,
        BLOCK_SIZE_KB=_DECODE_BLOCK_KB,
        TOP_K=top_k,
        A_ROW_IS_ROUTE=a_row_is_route,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        compute_type=_tl_dtype(c.dtype),
        num_warps=_DECODE_WARPS,
    )


def _decode_gemm_marlin(
    a: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    glob: torch.Tensor,
    c: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    mul_routed_weight: bool,
    a_row_is_route: bool,
    route_mask: torch.Tensor | None = None,
    fuse_silu: bool = False,
) -> None:
    """Marlin-style decode GEMV: int32 wide loads + deferred reduction
    (:func:`_decode_nvfp4_marlin_kernel`). ``packed`` is the uint8 ``[S, N, K//2]`` bank;
    it is reinterpreted as int32 ``[S, N, K//8]`` (contiguous, K%8==0 for NVFP4)."""
    M, top_k = topk_ids.shape
    N = packed.shape[1]
    K = packed.shape[2] * 2
    packed_i32 = packed.view(torch.int32)  # [S, N, K // 8]
    # fuse_silu: ``a`` is the [routes, 2K] gate_up buffer; the kernel builds the K-wide
    # activation silu(gate)*up on the fly (act_up_off = K).
    scale = e4m3_kernel_view(scale)
    total_routes = M * top_k
    grid = (total_routes, triton.cdiv(N, _DECODE_MARLIN_BLOCK_N))
    _decode_nvfp4_marlin_kernel[grid](
        a, packed_i32, scale, glob, c, topk_weights, topk_ids,
        _e2m1_lut(a.device.index),
        route_mask if route_mask is not None else topk_ids,
        K,
        total_routes, N, K,
        a.stride(0), a.stride(1),
        packed_i32.stride(0), packed_i32.stride(1), packed_i32.stride(2),
        scale.stride(0), scale.stride(1), scale.stride(2),
        glob.stride(0), glob.stride(1),
        c.stride(0), c.stride(1), c.stride(2),
        topk_weights.stride(0), topk_weights.stride(1),
        topk_ids.stride(0), topk_ids.stride(1),
        BLOCK_SIZE_N=_DECODE_MARLIN_BLOCK_N,
        BLOCK_SIZE_KW=_DECODE_MARLIN_BLOCK_KW,
        TOP_K=top_k,
        A_ROW_IS_ROUTE=a_row_is_route,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        compute_type=_tl_dtype(c.dtype),
        HAS_MASK=route_mask is not None,
        FUSE_SILU=fuse_silu,
        num_warps=_DECODE_MARLIN_WARPS_DOWN if a_row_is_route else _DECODE_MARLIN_WARPS,
    )


def _decode_gemm_marlin_swiglu(
    gate_up: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    glob: torch.Tensor,
    c: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    mul_routed_weight: bool,
    act_limit: float,
    route_mask: torch.Tensor | None = None,
) -> None:
    """Down GEMV with swiglu_clamp fused: ``gate_up`` is the [routes, 2K] gate_up buffer;
    the K-wide activation is rebuilt per K-iter from its [gate | up] halves. Same launch
    config as the unfused ``_decode_gemm_marlin(..., a_row_is_route=True)``."""
    from freetoken.kernel.triton.nvfp4_marlin_swiglu import _decode_nvfp4_marlin_swiglu_kernel

    M, top_k = topk_ids.shape
    N = packed.shape[1]
    K = packed.shape[2] * 2
    packed_i32 = packed.view(torch.int32)  # [S, N, K // 8]
    scale = e4m3_kernel_view(scale)
    total_routes = M * top_k
    limit = float(act_limit)
    grid = (total_routes, triton.cdiv(N, _DECODE_MARLIN_BLOCK_N))
    _decode_nvfp4_marlin_swiglu_kernel[grid](
        gate_up, packed_i32, scale, glob, c, topk_weights, topk_ids,
        _e2m1_lut(gate_up.device.index),
        route_mask if route_mask is not None else topk_ids,
        K, limit,
        total_routes, N, K,
        gate_up.stride(0), gate_up.stride(1),
        packed_i32.stride(0), packed_i32.stride(1), packed_i32.stride(2),
        scale.stride(0), scale.stride(1), scale.stride(2),
        glob.stride(0), glob.stride(1),
        c.stride(0), c.stride(1), c.stride(2),
        topk_weights.stride(0), topk_weights.stride(1),
        topk_ids.stride(0), topk_ids.stride(1),
        BLOCK_SIZE_N=_DECODE_MARLIN_BLOCK_N,
        BLOCK_SIZE_KW=_DECODE_MARLIN_BLOCK_KW,
        TOP_K=top_k,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        compute_type=_tl_dtype(c.dtype),
        act_type=_tl_dtype(gate_up.dtype),
        HAS_MASK=route_mask is not None,
        HAS_LIMIT=limit > 0,  # == fused_swiglu's HAS_LIMIT
        num_warps=_DECODE_MARLIN_WARPS_DOWN,
    )


def _fused_experts_decode_nvfp4(
    gemm_fn,
    hidden_states: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    apply_router_weight_on_input: bool,
    act_alpha: float = 1.702,
    act_limit: float = 7.0,
) -> torch.Tensor:
    """Shared decode body (gemm1 -> act -> gemm2 -> sum-reduce); ``gemm_fn`` is either
    the marlin-style int32 GEMV (:func:`_decode_gemm_marlin`) or the original LUT-gather
    GEMV (:func:`_decode_gemm`), both with the same calling convention."""
    M, H = hidden_states.shape
    top_k = topk_ids.shape[1]
    two_i = gate_up_packed.shape[1]
    inter = two_i // 2
    dev, dt = hidden_states.device, hidden_states.dtype

    ic1 = torch.empty((M, top_k, two_i), device=dev, dtype=dt)
    gemm_fn(
        hidden_states, gate_up_packed, gate_up_scale, gate_up_global,
        ic1, topk_weights, topk_ids, apply_router_weight_on_input, False,
    )
    if _SWIGLU_FUSED and activation == "swiglu_clamp" and gemm_fn is _decode_gemm_marlin:
        # act folded into the down GEMV (no ic2, no fused_swiglu, no copy_)
        ic3 = torch.empty((M, top_k, H), device=dev, dtype=dt)
        _decode_gemm_marlin_swiglu(
            ic1.view(-1, two_i), down_packed, down_scale, down_global,
            ic3, topk_weights, topk_ids, not apply_router_weight_on_input, act_limit,
        )
        out = torch.empty_like(hidden_states)
        moe_sum_reduce_triton(ic3, out)
        return out
    ic2 = torch.empty((M * top_k, inter), device=dev, dtype=dt)
    _run_act(activation, ic1.view(-1, two_i), ic2, act_alpha, act_limit)
    ic3 = torch.empty((M, top_k, H), device=dev, dtype=dt)
    gemm_fn(
        ic2, down_packed, down_scale, down_global,
        ic3, topk_weights, topk_ids, not apply_router_weight_on_input, True,
    )
    out = torch.empty_like(hidden_states)
    moe_sum_reduce_triton(ic3, out)
    return out


def fused_experts_decode_nvfp4_marlin(
    hidden_states: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    act_alpha: float = 1.702,
    act_limit: float = 7.0,
) -> torch.Tensor:
    """Decode inline-NVFP4 MoE using the Marlin-style int32 wide-load GEMV."""
    return _fused_experts_decode_nvfp4(
        _decode_gemm_marlin,
        hidden_states, gate_up_packed, gate_up_scale, gate_up_global,
        down_packed, down_scale, down_global,
        topk_weights, topk_ids, activation, apply_router_weight_on_input,
        act_alpha, act_limit,
    )


def fused_experts_decode_nvfp4_serial(
    hidden_states: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    act_alpha: float = 1.702,
    act_limit: float = 7.0,
) -> torch.Tensor:
    """Original LUT-gather decode (one program per route, full K reduction). Retained for
    A/B benchmarking against the marlin decode path; not on the production decode path."""
    return _fused_experts_decode_nvfp4(
        _decode_gemm,
        hidden_states, gate_up_packed, gate_up_scale, gate_up_global,
        down_packed, down_scale, down_global,
        topk_weights, topk_ids, activation, apply_router_weight_on_input,
        act_alpha, act_limit,
    )


def _prefill_config(M: int) -> Dict[str, int]:
    # ``BLOCK_SIZE_M`` is coupled to host-side ``moe_align_block_size`` (token padding),
    # so it cannot be picked by triton.autotune; these were chosen by an offline sweep
    # over (BLOCK_M, BLOCK_N, BLOCK_KB, num_warps, num_stages). Re-swept 2026-09-29 for
    # Naive (H4096/I2048, top-8/256, full-layer bank) with the arithmetic e2m1 decode
    # (ARITH_DECODE, no LUT gather): per layer 8192 tok 125 -> 49 ms, 1024 tok 19.3 ->
    # 10.6, 256 tok 10.6 -> 6.0. BLOCK_KB stays 32: bit-identical output to the old
    # kernel (KB=64 is ~5% faster but reorders the fp32 sums, rel diff 1.6e-4).
    if M <= 64:
        return dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_KB=32,
                    GROUP_SIZE_M=1, num_warps=8, num_stages=4, ARITH_DECODE=True)
    if M <= 512:
        return dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_KB=32,
                    GROUP_SIZE_M=8, num_warps=8, num_stages=4, ARITH_DECODE=True)
    if M < 4096:
        return dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, BLOCK_SIZE_KB=32,
                    GROUP_SIZE_M=8, num_warps=4, num_stages=4, ARITH_DECODE=True,
                    BF16_DECODE=True)
    # BF16_DECODE (exact, bit-identical): 8192 tok 49.1 -> 45.5 ms/layer
    return dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_KB=32,
                GROUP_SIZE_M=8, num_warps=4, num_stages=4, ARITH_DECODE=True,
                BF16_DECODE=True)


def _prefill_gemm(
    a: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    glob: torch.Tensor,
    c: torch.Tensor,
    topk_weights_flat: torch.Tensor,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    num_valid_tokens: int,
    kernel_top_k: int,
    mul_routed_weight: bool,
    cfg: Dict[str, Any],
) -> None:
    N = packed.shape[1]
    K = packed.shape[2] * 2
    EM = sorted_ids.shape[0]
    scale = e4m3_kernel_view(scale)
    grid = lambda META: (  # noqa: E731
        triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    _prefill_nvfp4_moe_kernel[grid](
        a, packed, scale, glob, c, topk_weights_flat, sorted_ids, expert_ids,
        num_tokens_post_padded,
        _e2m1_lut(a.device.index),
        N, K, EM, num_valid_tokens,
        a.stride(0), a.stride(1),
        packed.stride(0), packed.stride(1), packed.stride(2),
        scale.stride(0), scale.stride(1), scale.stride(2),
        glob.stride(0), glob.stride(1),
        c.stride(1), c.stride(2),
        topk_weights_flat.stride(0),
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=kernel_top_k,
        compute_type=_tl_dtype(c.dtype),
        **cfg,
    )


def fused_experts_nvfp4(
    hidden_states: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    act_alpha: float = 1.702,
    act_limit: float = 7.0,
) -> torch.Tensor:
    """Prefill inline-NVFP4 MoE. ``topk_ids`` index rows of the bank tensors in
    ``[0, num_experts)``: full-layer banks with position == expert id (the
    materialized ``[:E]`` slot view or the overlap double buffer), raw ids."""
    M, H = hidden_states.shape
    top_k = topk_ids.shape[1]
    two_i = gate_up_packed.shape[1]
    inter = two_i // 2
    dev, dt = hidden_states.device, hidden_states.dtype
    cfg = _prefill_config(M)

    sorted_ids, expert_ids, ntpp = moe_align_block_size(topk_ids, cfg["BLOCK_SIZE_M"], num_experts)
    tw = topk_weights.reshape(-1).contiguous()
    num_valid = topk_ids.numel()

    ic1 = torch.empty((M, top_k, two_i), device=dev, dtype=dt)
    _prefill_gemm(
        hidden_states, gate_up_packed, gate_up_scale, gate_up_global, ic1,
        tw, sorted_ids, expert_ids, ntpp, num_valid, top_k,
        apply_router_weight_on_input, cfg,
    )
    ic2 = torch.empty((M * top_k, inter), device=dev, dtype=dt)
    _run_act(activation, ic1.view(-1, two_i), ic2, act_alpha, act_limit)
    ic3 = torch.empty((M, top_k, H), device=dev, dtype=dt)
    _prefill_gemm(
        ic2, down_packed, down_scale, down_global, ic3,
        tw, sorted_ids, expert_ids, ntpp, num_valid, 1,
        not apply_router_weight_on_input, cfg,
    )
    out = torch.empty_like(hidden_states)
    moe_sum_reduce_triton(ic3, out)
    return out


__all__ = [
    "fused_experts_decode_nvfp4_marlin",
    "fused_experts_decode_nvfp4_serial",
    "fused_experts_nvfp4",
]


def fused_experts_decode_nvfp4_split(
    hidden_states: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    miss_mask: torch.Tensor,
    copy_event: torch.cuda.Event,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    act_alpha: float = 1.702,
    act_limit: float = 7.0,
) -> torch.Tensor:
    """Decode MoE in two phases around an in-flight miss copy.

    Phase A computes every route whose expert was already resident (``~miss_mask``)
    while the missing experts are still being copied on a side stream; phase B waits
    ``copy_event`` and computes the missed routes. The phases write disjoint route rows
    of the same per-route buffers and one sum-reduce closes them, so the result is
    bit-identical to the single-phase path (same kernels, same per-route math)."""
    M, H = hidden_states.shape
    top_k = topk_ids.shape[1]
    two_i = gate_up_packed.shape[1]
    inter = two_i // 2
    dev, dt = hidden_states.device, hidden_states.dtype
    from freetoken.utils.regions import region

    if isinstance(miss_mask, tuple):  # (hit, miss) int8 masks from the ensure kernel
        mask_a, mask_b = miss_mask
    else:
        with region("moe.mask_prep"):
            mask_b = miss_mask.reshape(-1).to(torch.int8)
            mask_a = 1 - mask_b
    fuse = activation == "silu" and os.environ.get("FREETOKEN_MOE_FUSE_ACT", "1") != "0"
    fuse_sw = _SWIGLU_FUSED and activation == "swiglu_clamp"
    ic1 = torch.empty((M, top_k, two_i), device=dev, dtype=dt)
    ic2 = None if (fuse or fuse_sw) else torch.empty((M * top_k, inter), device=dev, dtype=dt)
    ic3 = torch.empty((M, top_k, H), device=dev, dtype=dt)
    for phase, mask in ((0, mask_a), (1, mask_b)):
        tag = "hit" if phase == 0 else "miss"
        if phase == 1:
            with region("moe.wait_copy"):
                torch.cuda.current_stream().wait_event(copy_event)
        with region(f"moe.{tag}.gate_up"):
            _decode_gemm_marlin(
                hidden_states, gate_up_packed, gate_up_scale, gate_up_global,
                ic1, topk_weights, topk_ids, apply_router_weight_on_input, False,
                route_mask=mask,
            )
        # act over all rows: rows of the other phase hold stale/garbage values here,
        # but each phase's down GEMM only reads its own rows, and the later phase
        # rewrites its rows' activations before reading them.
        if fuse:
            with region(f"moe.{tag}.down+act"):
                _decode_gemm_marlin(
                    ic1.view(-1, two_i), down_packed, down_scale, down_global,
                    ic3, topk_weights, topk_ids, not apply_router_weight_on_input, True,
                    route_mask=mask, fuse_silu=True,
                )
            continue
        if fuse_sw:
            with region(f"moe.{tag}.down+act"):
                _decode_gemm_marlin_swiglu(
                    ic1.view(-1, two_i), down_packed, down_scale, down_global,
                    ic3, topk_weights, topk_ids, not apply_router_weight_on_input, act_limit,
                    route_mask=mask,
                )
            continue
        with region(f"moe.{tag}.act"):
            _run_act(activation, ic1.view(-1, two_i), ic2, act_alpha, act_limit)
        with region(f"moe.{tag}.down"):
            _decode_gemm_marlin(
                ic2, down_packed, down_scale, down_global,
                ic3, topk_weights, topk_ids, not apply_router_weight_on_input, True,
                route_mask=mask,
            )
    with region("moe.sum_reduce"):
        out = torch.empty_like(hidden_states)
        moe_sum_reduce_triton(ic3, out)
    return out
