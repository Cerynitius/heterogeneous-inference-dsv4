"""Two-GPU expert parallelism (EP) for DeepSeek-V4-Flash on heterogeneous cards.

dev0 (the engine device) keeps everything it had -- attention, router, shared expert,
KV, head -- plus its share of every layer's routed FP4 experts; dev1 holds the rest of
the routed experts. Every expert is resident on one of the two cards, so there is no
host offload, no PCIe expert traffic, and no pinned host banks.

Per MoE layer:

  decode (CUDA-graph captured, B <= max_bs):
    dev0: FP8 round-trip x straight into a staging buffer + one split kernel that writes
          dev1's slots / weights next to it and dev0's local slots + the per-route owner
          mask (2 launches), then record ``e_in``.
    dev1: wait ``e_in`` -> pull the staging buffer (one P2P DMA) -> gate_up GEMV -> swiglu
          -> down GEMV on its local banks -> push its per-route outputs [B, K, H] back
          (one P2P DMA) -> record ``e_out``.
    dev0: meanwhile the shared expert and the same three kernels on its own banks; then
          wait ``e_out``, pick each route's row from its owner and run the identical
          ``.sum(dim=1)``.
  prefill (eager): per token chunk, both cards compute only their own routes (compacted,
          host-synced -- prefill is never captured); dev0 scatters both results back into
          [T, K, H] and runs the same ``.sum(dim=1)``.

Exactness: each route's output comes from the same kernel (the decode GEMV gains only an
early return for ``slot < 0``; prefill uses the unmodified kernels) on bit-identical
inputs, and the GEMV-vs-grouped choice is still made on the FULL ``T * top_k`` -- so the
layer output is bit-identical to the single-GPU offload path (validated by
tests/dsv4_ep/test_ep_exact.py).

P2P copies go through ``cuMemcpyAsync`` on an explicit stream: torch's cross-device
``copy_`` adds a two-way barrier against the other device's current stream, which would
make dev1's push wait for all of dev0's queued work.
"""

from __future__ import annotations

import ctypes
import os
import re
import time

import torch
import triton
import triton.language as tl

from freetoken.utils import init_logger

logger = init_logger(__name__)

_EXPERT_RE = re.compile(
    r"^layers\.(?P<layer>\d+)\.ffn\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>w1|w2|w3)\.(?P<kind>weight|scale)$"
)


# FP8 x FP4 grouped MoE GEMM with 128-route blocks when routes/expert is large (11-18: 1.16x)
# per device: PRO 5000 (dev1) 152 -> 147 ms/chunk at 127K; PRO 6000 slower (100 -> 129 ms: the
# 2x padding on experts with <= 64 routes costs more than the weight re-read it saves)
_ONE_SYNC = os.environ.get("FREETOKEN_DSV4_EP_ONE_SYNC", "1") == "1"
_MOE_BM128_DEVS = {int(d) for d in os.environ.get("FREETOKEN_DSV4_MOE_BM128_DEVS", "1").split(",") if d.strip()}

def ep_enabled() -> bool:
    return os.environ.get("FREETOKEN_DSV4_EP", "0") == "1"


# ----------------------------------------------------------------------------------------
# P2P copy on an explicit stream (driver API; graph-capturable as a memcpy node).
# ----------------------------------------------------------------------------------------
_libcuda = None


def _cuda():
    global _libcuda
    if _libcuda is None:
        lib = ctypes.CDLL("libcuda.so.1")
        lib.cuMemcpyAsync.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_size_t, ctypes.c_void_p]
        lib.cuMemcpyAsync.restype = ctypes.c_int
        _libcuda = lib
    return _libcuda


def p2p_copy(dst: torch.Tensor, src: torch.Tensor, nbytes: int, stream: torch.cuda.Stream) -> None:
    """``nbytes`` from ``src`` to ``dst`` (any devices, UVA) on ``stream``. The caller must
    have ``stream``'s device current (driver API uses the thread's current context)."""
    if nbytes <= 0:
        return
    rc = _cuda().cuMemcpyAsync(dst.data_ptr(), src.data_ptr(), nbytes, stream.cuda_stream)
    if rc != 0:
        raise RuntimeError(f"cuMemcpyAsync failed: CUresult {rc} ({nbytes} B, "
                           f"{src.device} -> {dst.device})")


# ----------------------------------------------------------------------------------------
# Kernels
# ----------------------------------------------------------------------------------------
@triton.jit
def _decode_dsfp4_moe_kernel_skip(
    a_ptr, packed_ptr, scale_ptr, c_ptr, topk_weights_ptr, topk_ids_ptr, lut_ptr,
    total_routes, N, K,
    stride_am, stride_ak,
    stride_pe, stride_pn, stride_pkb,
    stride_se, stride_sn, stride_sblk,
    stride_cm, stride_ck, stride_cn,
    stride_tw_m, stride_tw_k,
    stride_tid_m, stride_tid_k,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_KB: tl.constexpr,
    TOP_K: tl.constexpr,
    A_ROW_IS_ROUTE: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    compute_type: tl.constexpr,
):
    """``_decode_dsfp4_moe_kernel`` (kernel/triton/dsv4/fused_moe.py) verbatim plus an
    early return for routes whose slot is -1 (owned by the other card). The arithmetic
    of every computed route is unchanged."""
    route_id = tl.program_id(0)
    n_block_id = tl.program_id(1)
    token_id = route_id // TOP_K
    route_k = route_id - token_id * TOP_K

    offs_n = n_block_id * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    n_mask = offs_n < N

    slot = tl.load(topk_ids_ptr + token_id * stride_tid_m + route_k * stride_tid_k).to(tl.int64)
    if slot < 0:
        return
    a_row = route_id if A_ROW_IS_ROUTE else token_id
    a_base = a_ptr + a_row * stride_am

    NB: tl.constexpr = BLOCK_SIZE_KB // 16
    offs_kb = tl.arange(0, BLOCK_SIZE_KB)
    offs_nb = tl.arange(0, NB)
    K_BYTES = K // 2
    accumulator = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

    packed_slot = packed_ptr + slot * stride_pe
    scale_slot = scale_ptr + slot * stride_se
    for kb_start in range(0, K_BYTES // BLOCK_SIZE_KB):
        byte_idx = kb_start * BLOCK_SIZE_KB + offs_kb
        p_ptrs = packed_slot + offs_n[:, None] * stride_pn + byte_idx[None, :] * stride_pkb
        bytes_ = tl.load(p_ptrs, mask=n_mask[:, None], other=0).to(tl.int32)
        b_lo = tl.load(lut_ptr + (bytes_ & 0xF))
        b_hi = tl.load(lut_ptr + ((bytes_ >> 4) & 0xF))

        sblk = kb_start * NB + offs_nb
        codes = tl.load(scale_slot + offs_n[:, None] * stride_sn + sblk[None, :] * stride_sblk,
                        mask=n_mask[:, None], other=0).to(tl.float32)
        sc = tl.exp2(codes - 127.0)
        b_lo = tl.reshape(b_lo, (BLOCK_SIZE_N, NB, 16)) * sc[:, :, None]
        b_hi = tl.reshape(b_hi, (BLOCK_SIZE_N, NB, 16)) * sc[:, :, None]
        b_lo = tl.reshape(b_lo, (BLOCK_SIZE_N, BLOCK_SIZE_KB))
        b_hi = tl.reshape(b_hi, (BLOCK_SIZE_N, BLOCK_SIZE_KB))

        a_lo = tl.load(a_base + (2 * byte_idx) * stride_ak).to(tl.float32)
        a_hi = tl.load(a_base + (2 * byte_idx + 1) * stride_ak).to(tl.float32)
        accumulator += tl.sum(b_lo * a_lo[None, :], axis=1)
        accumulator += tl.sum(b_hi * a_hi[None, :], axis=1)

    if MUL_ROUTED_WEIGHT:
        weight = tl.load(topk_weights_ptr + token_id * stride_tw_m + route_k * stride_tw_k)
        accumulator = accumulator * weight

    c_ptrs = c_ptr + token_id * stride_cm + route_k * stride_ck + offs_n * stride_cn
    tl.store(c_ptrs, accumulator.to(compute_type), mask=(route_id < total_routes) & n_mask)


@triton.jit
def _ep_split_kernel(
    ids_ptr, w_ptr, tab0_ptr, tab1_ptr,
    s0_ptr, s1_ptr, w1_ptr, m1_ptr, n,
    BLOCK: tl.constexpr,
):
    """Per route: dev0 slot, dev1 slot, dev1's copy of the weight, owner mask (1 = dev1)."""
    offs = tl.arange(0, BLOCK)
    m = offs < n
    e = tl.load(ids_ptr + offs, mask=m, other=0)
    s0 = tl.load(tab0_ptr + e, mask=m, other=-1)
    s1 = tl.load(tab1_ptr + e, mask=m, other=-1)
    w = tl.load(w_ptr + offs, mask=m, other=0.0)
    tl.store(s0_ptr + offs, s0, mask=m)
    tl.store(s1_ptr + offs, s1, mask=m)
    tl.store(w1_ptr + offs, w, mask=m)
    tl.store(m1_ptr + offs, (s1 >= 0).to(tl.int8), mask=m)


def _gemv_cfg(a_row_is_route: bool):
    # identical env/defaults to fused_ds_fp4._grouped_decode (same compiled shapes)
    if a_row_is_route:
        return (int(os.environ.get("FREETOKEN_DSFP4_DOWN_BN", "32")),
                int(os.environ.get("FREETOKEN_DSFP4_DOWN_BKB", "128")),
                int(os.environ.get("FREETOKEN_DSFP4_DOWN_WARPS", "4")))
    return (int(os.environ.get("FREETOKEN_DSFP4_GU_BN", "8")),
            int(os.environ.get("FREETOKEN_DSFP4_GU_BKB", "128")),
            int(os.environ.get("FREETOKEN_DSFP4_GU_WARPS", "1")))


_DUMMY_W: dict = {}


def _dummy_w(dev: torch.device) -> torch.Tensor:
    t = _DUMMY_W.get(dev.index)
    if t is None:
        t = _DUMMY_W[dev.index] = torch.zeros((1, 1), dtype=torch.float32, device=dev)
    return t


def _gemv_skip(a, packed, scale, slots, topk_weights, out, *, a_row_is_route, mul_routed_weight):
    """``fused_ds_fp4._grouped_decode`` into a caller-owned ``out`` [T, top_k, N], skipping
    slot -1 routes (their rows of ``out`` are left untouched). FREETOKEN_DSFP4_GEMV=v2 runs
    the restructured kernel (fp4_gemv_v2.py; noise-level, not bitwise)."""
    from freetoken.kernel.triton.dsv4.fp4_gemv_v2 import dsfp4_gemv_v2, gemv_v2_enabled

    if gemv_v2_enabled():
        return dsfp4_gemv_v2(a, packed, scale, slots, topk_weights, out,
                             a_row_is_route=a_row_is_route, mul_routed_weight=mul_routed_weight)
    from freetoken.kernel.triton.dsv4.fused_moe import _e2m1_lut
    from freetoken.moe.fused_ds_fp4 import _compute_type

    T, top_k = slots.shape
    N = packed.shape[1]
    K = packed.shape[2] * 2
    total_routes = T * top_k
    if topk_weights is None:  # never read (MUL_ROUTED_WEIGHT=False); a persistent per-device
        topk_weights = _dummy_w(a.device)  # placeholder, so nothing is allocated inside a capture
    scale_u8 = scale.view(torch.uint8)
    BN, BKB, NW = _gemv_cfg(a_row_is_route)
    assert (K // 2) % BKB == 0, (K, BKB)
    grid = (total_routes, triton.cdiv(N, BN))
    _decode_dsfp4_moe_kernel_skip[grid](
        a, packed, scale_u8, out, topk_weights, slots,
        _e2m1_lut(a.device.index),
        total_routes, N, K,
        a.stride(0), a.stride(1),
        packed.stride(0), packed.stride(1), packed.stride(2),
        scale_u8.stride(0), scale_u8.stride(1), scale_u8.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        topk_weights.stride(0) if topk_weights.ndim == 2 else 0,
        topk_weights.stride(1) if topk_weights.ndim == 2 else 0,
        slots.stride(0), slots.stride(1),
        BLOCK_SIZE_N=BN, BLOCK_SIZE_KB=BKB, TOP_K=top_k,
        A_ROW_IS_ROUTE=a_row_is_route, MUL_ROUTED_WEIGHT=mul_routed_weight,
        compute_type=_compute_type(a.dtype), num_warps=NW,
    )
    return out


def _swiglu_into(gate_up: torch.Tensor, out: torch.Tensor, limit: float, quant: bool) -> None:
    """``fused_moe.fused_swiglu`` into a caller-owned ``out`` [R, I] (same kernel/launch)."""
    from freetoken.kernel.triton.dsv4.fused_moe import _compute_type, _swiglu_kernel

    two_I = gate_up.shape[-1]
    I = two_I // 2
    gu = gate_up.reshape(-1, two_I)
    R = gu.shape[0]
    BLOCK = 1024
    grid = (R, triton.cdiv(I, BLOCK))
    _swiglu_kernel[grid](
        gu, out, R, I, float(limit),
        gu.stride(0), gu.stride(1), out.stride(0), out.stride(1),
        BLOCK=BLOCK, HAS_LIMIT=limit > 0, QUANT=quant,
        compute_type=_compute_type(gate_up.dtype), num_warps=4,
    )


def _roundtrip_into(x: torch.Tensor, out: torch.Tensor, block: int = 128) -> None:
    """``act_quant_fp8_roundtrip(x)`` written into ``out`` (same kernel, same launch)."""
    from freetoken.kernel.triton.dsv4.fp8_linear import _act_quant_inplace_kernel

    N = x.shape[-1]
    x2d = x.reshape(-1, N)
    o2d = out.view(-1, N)
    M = x2d.shape[0]
    BLOCK_M = 32
    grid = (triton.cdiv(M, BLOCK_M), N // block)
    _act_quant_inplace_kernel[grid](
        x2d, o2d, M, N, x2d.stride(0), x2d.stride(1), o2d.stride(0), o2d.stride(1),
        -448.0, 448.0, 1.0 / 448.0, False, BLOCK_M=BLOCK_M, BLOCK=block,
    )


# FREETOKEN_DSFP4_PREFILL_MMA=1 (default): the grouped prefill MoE runs on sm_120a native block-
# scaled tensor cores (kernel/cuda/dsfp4_mma.cu: FP8 e4m3 activations x DS-FP4 = MXFP4 weights)
# instead of dequantizing the FP4 weights to BF16 for a BF16 MMA. Same exact products, different
# fp32 accumulation order (noise level, owner-approved 10-08); ~2x the grouped GEMM throughput.
_PREFILL_MMA = os.environ.get("FREETOKEN_DSFP4_PREFILL_MMA", "1") != "0"


def routes_fp4(a_rows, slots, weights, banks, swiglu_limit, grouped: bool,
               aq=None, ae=None, a_map=None):
    """Per-route routed-expert outputs [R, H] for compacted routes (eager / prefill).

    ``a_rows`` [R, H] are the FP8 round-tripped token rows of each route, ``slots`` [R]
    int32 local bank rows, ``weights`` [R] fp32. ``grouped`` is decided by the caller
    on the FULL route count (``T * top_k >= _GROUPED_MIN_ROUTES``), exactly like the
    single-GPU path; each route's value does not depend on which other routes share
    the launch, so this matches ``routed_experts_fp4[_prefill]`` route by route.
    With the native-FP4 path (``_PREFILL_MMA``, grouped) the input instead comes as the
    token-level FP8 codes ``aq`` [T, H] + e8m0 exponents ``ae`` [T, H/128] and the route ->
    token map ``a_map`` [R] (``a_rows`` may then be None)."""
    from freetoken.kernel.triton.dsv4.fp8_linear import act_quant_fp8, act_quant_fp8_inplace
    from freetoken.kernel.triton.dsv4.fused_moe import fused_swiglu
    from freetoken.moe.fused import moe_align_block_size
    from freetoken.moe.fused_ds_fp4 import _grouped_decode, _grouped_prefill

    gu_p, gu_s, dn_p, dn_s = banks
    R = slots.numel()
    two_I = gu_p.shape[1]
    I = two_I // 2
    sl = slots.view(R, 1)
    tw = weights.view(R, 1)
    if grouped and _PREFILL_MMA and aq is not None:
        from freetoken.kernel.cuda.dsfp4_mma import dsfp4_mma_grouped

        H = aq.shape[1]
        # 128-route blocks at prefill sizes (one M block per expert: each weight tile read once)
        bm = 128 if (aq.device.index in _MOE_BM128_DEVS and R >= 48 * gu_p.shape[0]) else 64
        sorted_ids, expert_ids, ntpp = moe_align_block_size(sl, bm, gu_p.shape[0])
        gate_up = torch.empty((R, two_I), dtype=torch.bfloat16, device=aq.device)
        dsfp4_mma_grouped(aq, ae, a_map, gu_p, gu_s, sorted_ids, expert_ids, ntpp, None,
                          gate_up, R, bm=bm)
        act = fused_swiglu(gate_up, swiglu_limit).reshape(R, I)
        act_q, act_e = act_quant_fp8(act, 128)  # the values act_quant_fp8_inplace rounds to
        down = torch.empty((R, H), dtype=torch.bfloat16, device=aq.device)
        dsfp4_mma_grouped(act_q, act_e, None, dn_p, dn_s, sorted_ids, expert_ids, ntpp,
                          weights.reshape(R).contiguous(), down, R, bm=bm)
        return down
    if a_rows is None:
        a_rows = (aq.float().view(aq.shape[0], -1, 128) * torch.exp2(ae.float() - 127)[:, :, None]
                  ).view(aq.shape[0], -1).to(torch.bfloat16).index_select(0, a_map.long())
    H = a_rows.shape[1]
    if not grouped:
        gate_up = _grouped_decode(a_rows, gu_p, gu_s, sl, None,
                                  a_row_is_route=False, mul_routed_weight=False)
        act = fused_swiglu(gate_up, swiglu_limit, quant=True).reshape(R, I)
        down = _grouped_decode(act, dn_p, dn_s, sl, tw,
                               a_row_is_route=True, mul_routed_weight=True)
        return down.view(R, H)
    # grouped: routed_experts_fp4_prefill's body with top_k=1 over the compacted routes
    if (os.environ.get("FREETOKEN_DSFP4_TILE", "").strip().lower() != "legacy"
            and torch.cuda.get_device_capability(a_rows.device)[0] >= 12):
        cfg = dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, BLOCK_SIZE_K=128, GROUP_SIZE_M=8,
                   num_warps=4, num_stages=1)
    else:
        cfg = dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64, GROUP_SIZE_M=8,
                   num_warps=8, num_stages=1)
    sorted_ids, expert_ids, ntpp = moe_align_block_size(sl, cfg["BLOCK_SIZE_M"], gu_p.shape[0])
    twf = tw.reshape(-1).contiguous()
    gate_up = torch.empty((R, 1, two_I), dtype=a_rows.dtype, device=a_rows.device)
    _grouped_prefill(a_rows, gu_p, gu_s, gate_up, twf, sorted_ids, expert_ids, ntpp,
                     R, 1, False, cfg)
    act = fused_swiglu(gate_up, swiglu_limit).reshape(R, I)
    act_quant_fp8_inplace(act, 128)
    down = torch.empty((R, 1, H), dtype=a_rows.dtype, device=a_rows.device)
    _grouped_prefill(act, dn_p, dn_s, down, twf, sorted_ids, expert_ids, ntpp,
                     R, 1, True, cfg)
    return down.view(R, H)


# ----------------------------------------------------------------------------------------
# Placement
# ----------------------------------------------------------------------------------------
def expert_bytes(H: int, I: int) -> int:
    return 2 * I * (H // 2 + H // 32) + H * (I // 2 + I // 32)


def default_placement(L: int, E: int, n1: int) -> torch.Tensor:
    """owner [L, E] int8 (1 = dev1): odd expert ids first, then even, n1 per layer.
    Overridden by FREETOKEN_DSV4_EP_PLACEMENT (an .npz/.pt with ``owner`` [L, E])."""
    order = list(range(1, E, 2)) + list(range(0, E, 2))
    owner = torch.zeros(L, E, dtype=torch.int8)
    owner[:, order[:n1]] = 1
    return owner


class DSV4ExpertParallel:
    """Resident two-device expert banks + per-layer route split for DeepSeek-V4."""

    def __init__(self, args, dev0: torch.device, dev1: torch.device, *, max_bs: int):
        self.args = args
        self.dev0, self.dev1 = dev0, dev1
        self.L, self.E = args.n_layers, args.n_routed_experts
        self.H, self.I = args.dim, args.moe_inter_dim
        self.K = args.n_activated_experts
        self.swiglu_limit = args.swiglu_limit
        self.max_bs = max(1, int(max_bs))
        self.prefill_chunk = int(os.environ.get("FREETOKEN_DSV4_EP_PREFILL_CHUNK", "16384"))
        # decode transport variants (both keep every route's arithmetic unchanged):
        #  PUSH: dev1's down GEMV stores its routes' rows straight into dev0's merge buffer
        #        (peer stores) -> no return DMA, no where(); dev0 sums the buffer.
        #  PULL: dev0's round-trip/split kernels store dev1's inputs straight into dev1's
        #        staging buffer (peer stores) -> no outbound DMA.
        self.push = os.environ.get("FREETOKEN_DSV4_EP_PUSH", "1") == "1"
        self.pull = os.environ.get("FREETOKEN_DSV4_EP_PULL", "1") == "1"
        # dev0's own routed GEMVs on a side stream, concurrent with the shared expert the
        # caller runs on the main stream (pure scheduling; same kernels, same values)
        self.side = os.environ.get("FREETOKEN_DSV4_EP_SIDE", "1") == "1"
        self.banks0: list[tuple] = []
        self.banks1: list[tuple] = []

    # ---------------------------------------------------------------- placement / alloc
    def plan(self, check_fit: bool = True, dev1_extra_bytes: int = 0) -> None:
        L, E = self.L, self.E
        per = expert_bytes(self.H, self.I)
        path = os.environ.get("FREETOKEN_DSV4_EP_PLACEMENT", "")
        if path:
            if path.endswith(".npz"):
                import numpy as np

                owner = torch.from_numpy(np.load(path)["owner"]).to(torch.int8)
            else:
                owner = torch.load(path)["owner"].to(torch.int8)
            assert owner.shape == (L, E), owner.shape
            src = path
        else:
            n1 = os.environ.get("FREETOKEN_DSV4_EP_N1", "")
            if n1:
                n1 = int(n1)
            else:
                free1, _ = torch.cuda.mem_get_info(self.dev1)
                reserve = int(float(os.environ.get("FREETOKEN_DSV4_EP_DEV1_RESERVE_GB", "2.5")) * (1 << 30))
                n1 = min(E, max(0, (free1 - reserve - dev1_extra_bytes) // (L * per)))
            owner = default_placement(L, E, n1)
            src = f"default n1={n1}"
        self.owner = owner
        n1_per = owner.sum(dim=1).tolist()
        need1 = sum(n1_per) * per + dev1_extra_bytes
        free1, _ = torch.cuda.mem_get_info(self.dev1)
        if check_fit and need1 > free1:
            raise RuntimeError(
                f"EP placement needs {need1 / 2**30:.2f} GiB on {self.dev1} but only "
                f"{free1 / 2**30:.2f} GiB is free")
        slot0 = torch.full((L, E), -1, dtype=torch.int32)
        slot1 = torch.full((L, E), -1, dtype=torch.int32)
        for li in range(L):
            on1 = owner[li].bool()
            i1 = on1.nonzero().squeeze(1)
            i0 = (~on1).nonzero().squeeze(1)
            slot1[li, i1] = torch.arange(i1.numel(), dtype=torch.int32)
            slot0[li, i0] = torch.arange(i0.numel(), dtype=torch.int32)
        self.slot0_cpu, self.slot1_cpu = slot0, slot1
        self.slot0 = slot0.to(self.dev0)
        self.slot1 = slot1.to(self.dev0)
        n0_tot = int((owner == 0).sum())
        n1_tot = int(owner.sum())
        logger.info_rank0(
            f"DSV4 EP placement ({src}): dev0 {self.dev0} {n0_tot} experts "
            f"({n0_tot * per / 2**30:.2f} GiB), dev1 {self.dev1} {n1_tot} experts "
            f"({n1_tot * per / 2**30:.2f} GiB); per-layer dev1 {min(n1_per)}..{max(n1_per)}")

    def _alloc_banks(self, dev, n):
        H, I = self.H, self.I
        e8m0 = torch.float8_e8m0fnu
        return (
            torch.empty((n, 2 * I, H // 2), dtype=torch.uint8, device=dev),
            torch.empty((n, 2 * I, H // 32), dtype=e8m0, device=dev),
            torch.empty((n, H, I // 2), dtype=torch.uint8, device=dev),
            torch.empty((n, H, I // 32), dtype=e8m0, device=dev),
        )

    def load(self, model_path: str) -> None:
        """Read every routed expert once (parallel O_DIRECT reader) straight into the
        owning card's bank."""
        from freetoken.models.weight import iter_expert_tensors_parallel

        L, I = self.L, self.I
        for li in range(L):
            n1 = int(self.owner[li].sum())
            self.banks0.append(self._alloc_banks(self.dev0, self.E - n1))
            self.banks1.append(self._alloc_banks(self.dev1, n1))

        def is_expert(name):
            m = _EXPERT_RE.match(name)
            return m is not None and int(m.group("layer")) < L

        t0 = time.time()
        placed = 0
        nbytes = 0
        for name, t in iter_expert_tensors_parallel(model_path, is_expert, workers=8):
            m = _EXPERT_RE.match(name)
            li, e = int(m.group("layer")), int(m.group("expert"))
            proj, kind = m.group("proj"), m.group("kind")
            if self.owner[li, e]:
                banks, s = self.banks1[li], int(self.slot1_cpu[li, e])
            else:
                banks, s = self.banks0[li], int(self.slot0_cpu[li, e])
            gu_p, gu_s, dn_p, dn_s = banks
            if kind == "weight":
                t = t.view(torch.uint8)
                dst = gu_p[s, :I] if proj == "w1" else gu_p[s, I:] if proj == "w3" else dn_p[s]
            else:
                dst = gu_s[s, :I] if proj == "w1" else gu_s[s, I:] if proj == "w3" else dn_s[s]
            dst.copy_(t)
            placed += 1
            nbytes += t.numel() * t.element_size()
        expected = L * self.E * 6
        assert placed == expected, f"EP loaded {placed} expert tensors, expected {expected}"
        torch.cuda.synchronize(self.dev0)
        torch.cuda.synchronize(self.dev1)
        dt = time.time() - t0
        logger.info_rank0(f"DSV4 EP experts loaded: {nbytes / 2**30:.2f} GiB in {dt:.1f} s "
                          f"({nbytes / dt / 1e9:.2f} GB/s)")

    def init_runtime(self) -> None:
        """Static decode buffers (graph-safe: nothing is allocated on dev1 while a dev0
        capture runs), the dev1 stream and the two fork/join events."""
        H, I, K, B = self.H, self.I, self.K, self.max_bs
        stage_bytes = B * (H * 2 + K * 8)
        self.stage0 = torch.empty(stage_bytes, dtype=torch.uint8, device=self.dev0)
        self.recv0 = torch.empty((B, K, H), dtype=torch.bfloat16, device=self.dev0)
        self.s0buf = torch.empty((B, K), dtype=torch.int32, device=self.dev0)
        self.m1buf = torch.empty((B, K), dtype=torch.int8, device=self.dev0)
        with torch.cuda.device(self.dev1):
            self.s1 = torch.cuda.Stream(self.dev1)
            self.stage1 = torch.empty(stage_bytes, dtype=torch.uint8, device=self.dev1)
            self.gu1 = torch.empty((B, K, 2 * I), dtype=torch.bfloat16, device=self.dev1)
            self.act1 = torch.empty((B * K, I), dtype=torch.bfloat16, device=self.dev1)
            self.dn1 = torch.empty((B, K, H), dtype=torch.bfloat16, device=self.dev1)
        self.merge0 = torch.empty((B, K, H), dtype=torch.bfloat16, device=self.dev0)
        self.x0buf = torch.empty((B, H), dtype=torch.bfloat16, device=self.dev0)
        logger.info_rank0(f"DSV4 EP decode transport: push={self.push} pull={self.pull} side={self.side} max_bs={B}")
        _dummy_w(self.dev0)
        _dummy_w(self.dev1)
        self.e_in = torch.cuda.Event()
        self.e_out = torch.cuda.Event()
        self.s0 = torch.cuda.Stream(self.dev0)
        self._stats_dir = os.environ.get("FREETOKEN_DSV4_EP_STATS", "")
        if self._stats_dir:
            os.makedirs(self._stats_dir, exist_ok=True)
            self.cnt_dec = torch.zeros((self.L, self.E), dtype=torch.int64, device=self.dev0)
            self.cnt_pre = torch.zeros((self.L, self.E), dtype=torch.int64, device=self.dev0)
        self.e_side = torch.cuda.Event()
        # Peer mappings, both directions, BEFORE any raw P2P traffic: with
        # expandable_segments (the engine default) every block lives in a cuMemMap'ed
        # segment that only the caching allocator's enablePeerAccess hook maps for the
        # other device -- and only torch's own cross-device copy path calls that hook
        # (at::cuda::get_p2p_access). cuMemcpyAsync / peer loads+stores into unmapped
        # VMM memory fail with CUDA_ERROR_INVALID_VALUE. The hook also maps segments
        # created later.
        t0 = torch.zeros(16, device=self.dev0)
        t1 = torch.zeros(16, device=self.dev1)
        t1.copy_(t0)
        t0.copy_(t1)
        torch.cuda.synchronize(self.dev0)
        torch.cuda.synchronize(self.dev1)
        with torch.cuda.device(self.dev1), torch.cuda.stream(self.s1):
            p2p_copy(self.stage1, self.stage0, 64, self.s1)
            p2p_copy(self.recv0, self.dn1, 64, self.s1)
        torch.cuda.synchronize(self.dev1)

    def _stage_views(self, buf: torch.Tensor, B: int):
        H, K = self.H, self.K
        ox = B * H * 2
        os_ = ox + B * K * 4
        x = buf[:ox].view(torch.bfloat16).view(B, H)
        s = buf[ox:os_].view(torch.int32).view(B, K)
        w = buf[os_:os_ + B * K * 4].view(torch.float32).view(B, K)
        return x, s, w, os_ + B * K * 4

    # ---------------------------------------------------------------- forward
    def _stats(self, layer_id, ids, is_prefill):
        """FREETOKEN_DSV4_EP_STATS=<dir>: per-(layer, expert) route counts for decode and
        prefill, accumulated on the GPU (graph-safe index_add_). Touching <dir>/dump makes the
        next prefill write <dir>/route_counts.pt (decode replays never run Python)."""
        d = self._stats_dir
        if layer_id == 0 and is_prefill and os.path.exists(os.path.join(d, "dump")):
            torch.save({"decode": self.cnt_dec.cpu(), "prefill": self.cnt_pre.cpu()},
                       os.path.join(d, "route_counts.pt"))
            os.remove(os.path.join(d, "dump"))
        cnt = self.cnt_pre if is_prefill else self.cnt_dec
        flat = ids.reshape(-1).long()
        cnt[layer_id].index_add_(0, flat, torch.ones_like(flat))

    def prefill_launch(self, layer_id: int, x, topk_weights, topk_ids, overlap_fn=None):
        """Split-phase prefill (micro-batch pipeline): :meth:`forward` == finish(launch)."""
        if self._stats_dir:
            self._stats(layer_id, topk_ids, True)
        return self._prefill_launch(layer_id, x, topk_weights, topk_ids, overlap_fn)

    def prefill_finish(self, handle):
        return self._prefill_finish(handle)

    def forward(self, layer_id: int, x, topk_weights, topk_ids, overlap_fn=None, is_prefill=False):
        if self._stats_dir:
            self._stats(layer_id, topk_ids, is_prefill or x.shape[0] > self.max_bs)
        if is_prefill or x.shape[0] > self.max_bs:
            return self._prefill(layer_id, x, topk_weights, topk_ids, overlap_fn)
        return self._decode(layer_id, x, topk_weights, topk_ids, overlap_fn)

    def _decode(self, li, x, w, ids, overlap_fn):
        B, H, K, I = x.shape[0], self.H, self.K, self.I
        main = torch.cuda.current_stream(self.dev0)
        n = B * K
        s0 = self.s0buf[:B]
        m1 = self.m1buf[:B]
        if self.pull:
            # dev1's inputs are stored into its staging buffer by dev0's kernels
            x1, s1_st, w1_st, _ = self._stage_views(self.stage1, B)
            x0 = self.x0buf[:B]
            _roundtrip_into(x, x0)
            _roundtrip_into(x, x1)
            _ep_split_kernel[(1,)](ids, w, self.slot0[li], self.slot1[li], s0, s1_st, w1_st, m1, n,
                                   BLOCK=triton.next_power_of_2(n))
            nbytes = 0
        else:
            x0, s1_st, w1_st, nbytes = self._stage_views(self.stage0, B)
            _roundtrip_into(x, x0)
            _ep_split_kernel[(1,)](ids, w, self.slot0[li], self.slot1[li], s0, s1_st, w1_st, m1, n,
                                   BLOCK=triton.next_power_of_2(n))
        self.e_in.record(main)
        b1 = self.banks1[li]
        mg = self.merge0[:B]
        with torch.cuda.device(self.dev1), torch.cuda.stream(self.s1):
            self.s1.wait_event(self.e_in)
            if nbytes:
                p2p_copy(self.stage1, self.stage0, nbytes, self.s1)
            x1, s1, w1, _ = self._stage_views(self.stage1, B)
            gu1 = self.gu1[:B]
            act1 = self.act1[:n]
            dn1 = mg if self.push else self.dn1[:B]
            _gemv_skip(x1, b1[0], b1[1], s1, None, gu1, a_row_is_route=False, mul_routed_weight=False)
            _swiglu_into(gu1, act1, self.swiglu_limit, quant=True)
            _gemv_skip(act1, b1[2], b1[3], s1, w1, dn1, a_row_is_route=True, mul_routed_weight=True)
            if not self.push:
                p2p_copy(self.recv0, dn1, n * H * 2, self.s1)
            self.e_out.record(self.s1)
        b0 = self.banks0[li]
        side = self.side and overlap_fn is not None
        if side:
            self.e_side.record(main)
            self.s0.wait_event(self.e_side)
        with torch.cuda.stream(self.s0 if side else main):
            gu0 = torch.empty((B, K, 2 * I), dtype=x.dtype, device=self.dev0)
            act0 = torch.empty((n, I), dtype=x.dtype, device=self.dev0)
            dn0 = mg if self.push else torch.empty((B, K, H), dtype=x.dtype, device=self.dev0)
            _gemv_skip(x0, b0[0], b0[1], s0, None, gu0, a_row_is_route=False, mul_routed_weight=False)
            _swiglu_into(gu0, act0, self.swiglu_limit, quant=True)
            _gemv_skip(act0, b0[2], b0[3], s0, w, dn0, a_row_is_route=True, mul_routed_weight=True)
        if overlap_fn is not None:
            overlap_fn()  # shared expert on main, concurrent with the side-stream GEMVs
        if side:
            main.wait_stream(self.s0)
        main.wait_event(self.e_out)
        if self.push:
            return mg.sum(dim=1)
        sel = m1.view(torch.bool).view(B, K, 1)
        return torch.where(sel, self.recv0[:B], dn0).sum(dim=1)

    def _prefill(self, li, x, w, ids, overlap_fn=None):
        """Eager prefill: per token chunk, both cards compute only their own routes; dev1's
        routes run in ``prefill_split`` sub-chunks whose outputs stream back on a separate
        dev1 copy stream while the next sub-chunk computes, and dev0 runs its own routes and
        the shared expert (``overlap_fn``) before waiting for dev1. Each route's value is
        independent of how routes are grouped into launches (and the grouped kernel is fixed
        for all prefill sizes), so the result is unchanged."""
        T = x.shape[0]
        TC = self.prefill_chunk
        if T <= TC:
            return self._prefill_finish(self._prefill_launch(li, x, w, ids, overlap_fn))
        out = torch.empty((T, self.H), dtype=x.dtype, device=self.dev0)
        for t0 in range(0, T, TC):
            t1 = min(T, t0 + TC)
            h = self._prefill_launch(li, x[t0:t1], w[t0:t1], ids[t0:t1],
                                     overlap_fn if t0 == 0 else None)
            out[t0:t1] = self._prefill_finish(h)
        return out

    def _prefill_launch(self, li, x, w, ids, overlap_fn=None):
        """Issue one prefill MoE layer for ``x`` [T <= prefill_chunk, H]: dev1's routes are
        dispatched (input copy, expert-run sub-chunks, streamed copy-back on s1 / s1c), dev0's
        own routes and ``overlap_fn`` (the shared expert) are enqueued on the current stream.
        Returns a handle for :meth:`_prefill_finish`, which waits for dev1 and combines. Several
        handles may be in flight (the in-chunk micro-batch pipeline, model.py): each owns its
        events and every buffer a pending copy touches, so nothing is recycled under a copy."""
        from freetoken.kernel.triton.dsv4.fp8_linear import act_quant_fp8, act_quant_fp8_roundtrip
        from freetoken.moe.fused_ds_fp4 import _GROUPED_MIN_ROUTES, _PREFILL_ALWAYS_GROUPED

        Tc, H, K = x.shape[0], self.H, self.K
        grouped = _PREFILL_ALWAYS_GROUPED or Tc * K >= _GROUPED_MIN_ROUTES
        main = torch.cuda.current_stream(self.dev0)
        b0, b1 = self.banks0[li], self.banks1[li]
        nsplit = max(1, int(os.environ.get("FREETOKEN_DSV4_EP_PREFILL_SPLIT", "4")))
        # sub-chunk sizes shrink geometrically (ratio r): each copy-back hides under the next
        # (smaller) sub-chunk's compute and the exposed last copy is short
        ratio = float(os.environ.get("FREETOKEN_DSV4_EP_PREFILL_SPLIT_RATIO", "0.5"))
        fp8_xfer = os.environ.get("FREETOKEN_DSV4_EP_FP8_XFER", "1") != "0"
        mma = grouped and _PREFILL_MMA
        if not hasattr(self, "s1c"):
            with torch.cuda.device(self.dev1):
                self.s1c = torch.cuda.Stream(self.dev1)
        # FP8 input: BF16 round-trip rows (BF16-dequant kernels) and/or the codes + exponents
        # (native FP4 MMA, FP8 transfer to dev1)
        xs = None if (mma and fp8_xfer) else act_quant_fp8_roundtrip(x, 128)  # [Tc, H]
        xq = xe = None
        if mma or fp8_xfer:
            xq, xe = act_quant_fp8(x, 128)
        idl = ids.reshape(-1).long()
        wf = w.reshape(-1)
        s0 = self.slot0[li][idl]
        s1 = self.slot1[li][idl]
        on1 = s1 >= 0
        cnt_host = None
        if _ONE_SYNC:
            # ONE host sync per layer: dev1's per-slot route counts (R1 and the sub-chunk cuts
            # follow on the host); r0 / r1 via nonzero_static with the now-known sizes. Was 3
            # syncs (2 nonzero + searchsorted.tolist()), each draining dev0 and exposing the
            # CPU's launch latency for the dev1 dispatch.
            E1 = b1[0].shape[0]
            cnt_host = torch.bincount(torch.where(on1, s1, E1), minlength=E1 + 1)[:E1].tolist()
            R1 = sum(cnt_host)
            r1 = torch.nonzero_static(on1, size=R1).squeeze(1)
            r0 = torch.nonzero_static(~on1, size=idl.numel() - R1).squeeze(1)
        else:
            r1 = on1.nonzero().squeeze(1)
            r0 = (~on1).nonzero().squeeze(1)
            R1 = r1.numel()
        full = torch.empty((Tc * K, H), dtype=x.dtype, device=self.dev0)
        hd = {"Tc": Tc, "full": full, "R1": R1, "keep": [xs, xq, xe]}
        if R1:
            tok1 = (r1 // K).to(torch.int32)
            sl1 = s1[r1].contiguous()
            ww1 = wf[r1].contiguous()
            if nsplit > 1:
                # Sub-chunks = runs of whole experts (routes sorted by bank slot), so every
                # expert's weights are read once per layer however many sub-chunks stream
                # back. (Token-order sub-chunks each touched ~all of dev1's experts: the
                # weight re-reads ate the copy overlap.) Cut at the expert run containing
                # each target of the geometric share.
                sl1, order = torch.sort(sl1, stable=True)
                r1, tok1, ww1 = r1[order], tok1[order], ww1[order]
                fr = [ratio ** j for j in range(nsplit)]
                cum = [sum(fr[:j]) / sum(fr) for j in range(1, nsplit)]
                tg = [min(int(R1 * c), R1 - 1) for c in cum]
                if cnt_host is not None:
                    # == searchsorted(sl1, sl1[tgt]): start of the slot run holding route tgt
                    import bisect
                    starts, acc = [], 0
                    for c_ in cnt_host:
                        starts.append(acc)
                        acc += c_
                    ends = [st + c_ for st, c_ in zip(starts, cnt_host)]
                    cuts = [starts[bisect.bisect_right(ends, t)] for t in tg]
                else:
                    tgt = torch.tensor(tg, device=sl1.device)
                    cuts = torch.searchsorted(sl1, sl1[tgt]).tolist()
                bounds = [0] + cuts + [R1]
            else:
                bounds = [0, R1]
            recv = torch.empty((R1, H), dtype=x.dtype, device=self.dev0)
            e_in, e_out = torch.cuda.Event(), torch.cuda.Event()
            # (fp8_xfer: the FP8 codes + e8m0 block exponents go to dev1 -- half the bytes of
            # the BF16 round-trip; dev1 rebuilds xs from them, exact in BF16, or feeds them to
            # the native-FP4 MMA as they are)
            e_in.record(main)
            with torch.cuda.device(self.dev1), torch.cuda.stream(self.s1):
                self.s1.wait_event(e_in)
                t_1 = torch.empty_like(tok1, device=self.dev1)
                sl_1 = torch.empty_like(sl1, device=self.dev1)
                w_1 = torch.empty_like(ww1, device=self.dev1)
                if fp8_xfer:
                    xq1 = torch.empty_like(xq, device=self.dev1)
                    xe1 = torch.empty_like(xe, device=self.dev1)
                    p2p_copy(xq1, xq, xq.numel() * xq.element_size(), self.s1)
                    p2p_copy(xe1, xe, xe.numel(), self.s1)
                    xs1 = None
                    if not mma:
                        scale = (xe1.to(torch.int32) << 23).view(torch.float32)  # 2^(e-127)
                        xs1 = (xq1.view(Tc, H // 128, 128).float() * scale[:, :, None]).view(
                            Tc, H).to(x.dtype)
                    for t in (xq1, xe1):
                        t.record_stream(self.s1c)
                else:
                    xs1 = torch.empty_like(xs, device=self.dev1)
                    p2p_copy(xs1, xs, xs.numel() * 2, self.s1)
                p2p_copy(t_1, tok1, R1 * 4, self.s1)
                p2p_copy(sl_1, sl1, R1 * 4, self.s1)
                p2p_copy(w_1, ww1, R1 * 4, self.s1)
                self.s1c.wait_stream(self.s1)
                for j in range(nsplit):
                    a, b = bounds[j], bounds[j + 1]
                    if a == b:
                        continue
                    if mma and fp8_xfer:
                        o1 = routes_fp4(None, sl_1[a:b], w_1[a:b], b1, self.swiglu_limit, grouped,
                                        aq=xq1, ae=xe1, a_map=t_1[a:b])
                    else:
                        o1 = routes_fp4(xs1.index_select(0, t_1[a:b]), sl_1[a:b], w_1[a:b], b1,
                                        self.swiglu_limit, grouped)
                    ev = torch.cuda.Event()
                    ev.record(self.s1)
                    self.s1c.wait_event(ev)
                    o1.record_stream(self.s1c)
                    p2p_copy(recv[a:b], o1, (b - a) * H * 2, self.s1c)
                e_out.record(self.s1c)
                for t in (xs1, t_1, sl_1, w_1):
                    if t is not None:  # xs1: None on the native-FP4 path (codes go in as is)
                        t.record_stream(self.s1c)
            hd.update(r1=r1, recv=recv, e_out=e_out)
            hd["keep"] += [tok1, sl1, ww1]
        if r0.numel():
            if mma:
                o0 = routes_fp4(None, s0[r0].contiguous(), wf[r0].contiguous(), b0,
                                self.swiglu_limit, grouped, aq=xq, ae=xe,
                                a_map=(r0 // K).to(torch.int32))
            else:
                o0 = routes_fp4(xs.index_select(0, r0 // K), s0[r0].contiguous(),
                                wf[r0].contiguous(), b0, self.swiglu_limit, grouped)
            full.index_copy_(0, r0, o0)
        if overlap_fn is not None:
            overlap_fn()  # shared expert on dev0 while dev1 is still computing
        return hd

    def _prefill_finish(self, hd):
        """Wait for a :meth:`_prefill_launch` handle's dev1 routes and combine all routes."""
        full = hd["full"]
        if hd["R1"]:
            torch.cuda.current_stream(self.dev0).wait_event(hd["e_out"])
            full.index_copy_(0, hd["r1"], hd["recv"])
        return full.view(hd["Tc"], self.K, self.H).sum(dim=1)
