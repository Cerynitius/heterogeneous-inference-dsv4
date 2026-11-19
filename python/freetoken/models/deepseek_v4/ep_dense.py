"""Heterogeneous split of DSV4's dense decode work across the two EP cards.

Ratios come from a startup calibration of each card's memory bandwidth (decode GEMVs are
bandwidth-bound) and dense compute (prefill), see :func:`calibrate`.

``EPDenseWo`` -- exact proportional split of the B=1 output projection
    wo_a (grouped FP8 GEMV, 8 groups) -> wo_b (FP8 split-K GEMV, K=8192 in 8 chunks of 1024).
    wo_b's split-K chunk c contracts exactly o_lora[1024c : 1024(c+1)], i.e. wo_a group c, so
    dev1 takes the last ``g1`` groups of wo_a AND the matching ``g1`` split-K chunks of wo_b
    and writes those fp32 partials straight into dev0's partial buffer (peer stores); dev0
    does the first ``8 - g1`` groups/chunks and runs the unchanged split-K reduce over all 8.
    Every partial is produced by the same kernel over the same K range with the same
    in-register activation quant, and the reduce order is unchanged -> bit-identical to
    ``Attention._wo`` (tests/dsv4_ep/test_ep_dense.py). One fork/join per layer.
"""

from __future__ import annotations

import os
import time

import torch
import triton

from freetoken.utils import init_logger

from .ep import p2p_copy

logger = init_logger(__name__)


def calibrate(dev0: torch.device, dev1: torch.device) -> dict:
    """Measure each card's effective memory bandwidth (large D2D copy: read + write) and
    dense bf16 matmul throughput. Returns the raw numbers and dev1's share for a
    bandwidth-bound and a compute-bound split."""
    res = {}
    for d in (dev0, dev1):
        with torch.cuda.device(d):
            a = torch.empty(512 << 20, dtype=torch.uint8, device=d)
            b = torch.empty_like(a)
            for _ in range(3):
                b.copy_(a)
            torch.cuda.synchronize(d)
            t = time.perf_counter()
            n = 20
            for _ in range(n):
                b.copy_(a)
            torch.cuda.synchronize(d)
            bw = 2 * a.numel() * n / (time.perf_counter() - t) / 1e9
            del a, b
            x = torch.randn(8192, 8192, dtype=torch.bfloat16, device=d)
            for _ in range(2):
                x @ x
            torch.cuda.synchronize(d)
            t = time.perf_counter()
            n = 10
            for _ in range(n):
                x @ x
            torch.cuda.synchronize(d)
            tf = 2 * 8192 ** 3 * n / (time.perf_counter() - t) / 1e12
            del x
            torch.cuda.empty_cache()
        res[d.index] = {"bw_gbs": bw, "bf16_tflops": tf}
    r0, r1 = res[dev0.index], res[dev1.index]
    res["bw_share1"] = r1["bw_gbs"] / (r0["bw_gbs"] + r1["bw_gbs"])
    res["compute_share1"] = r1["bf16_tflops"] / (r0["bf16_tflops"] + r1["bf16_tflops"])
    logger.info_rank0(
        f"DSV4 EP calibration: {dev0} {r0['bw_gbs']:.0f} GB/s {r0['bf16_tflops']:.0f} TFLOPS | "
        f"{dev1} {r1['bw_gbs']:.0f} GB/s {r1['bf16_tflops']:.0f} TFLOPS | dev1 share: "
        f"bandwidth {res['bw_share1']:.3f}, compute {res['compute_share1']:.3f}")
    return res


def wo_groups_for_dev1(bw_share1: float, groups: int = 8) -> int:
    """Groups (= wo_b split-K chunks) for dev1: its bandwidth share of the 8, rounded DOWN
    (the fork/join over PCIe is paid on dev1's side). FREETOKEN_DSV4_EP_WO_G1 overrides
    (0 disables the split)."""
    env = os.environ.get("FREETOKEN_DSV4_EP_WO_G1", "")
    if env:
        return max(0, min(groups - 1, int(env)))
    return max(1, min(groups - 1, int(bw_share1 * groups)))


def wo_dev1_bytes(n_layers: int, g1: int, o_lora: int = 1024, kg: int = 4096, dim: int = 4096) -> int:
    rows = g1 * o_lora
    per = rows * kg + (rows // 128) * (kg // 128) + dim * rows + (dim // 128) * (rows // 128)
    return n_layers * per


class EPDenseWo:
    """Exact cross-card split of the B=1 decode wo_a -> wo_b (see module docstring)."""

    def __init__(self, ep, attns: list, g1: int):
        from freetoken.kernel.triton.dsv4.fp8_linear import e4m3_act_dtype

        self.ep = ep
        self.g1 = g1
        a0 = attns[0]
        self.G = a0.n_groups
        self.R = a0.o_lora_rank
        self.KG = a0.n_heads * a0.head_dim // self.G
        self.dim = a0.dim
        self.g0 = self.G - g1
        self.r0 = self.g0 * self.R          # o_lora rows (= wo_b K) on dev0
        self.k0 = self.g0 * self.KG         # o input columns on dev0
        d0, d1 = ep.dev0, ep.dev1
        self.w1 = []
        for at in attns:
            assert at._woa_fp8, "EPDenseWo needs FREETOKEN_DSV4_WOA_FP8=1 (grouped FP8 wo_a)"
            wqa = at.wo_a_q[self.r0:].contiguous().to(d1)
            wsa = at.wo_a_s[self.r0 // 128:].contiguous().to(d1)
            sbw = _sb_u8(at.wo_b.scale)
            wb = at.wo_b.weight[:, self.r0:].contiguous().to(d1)
            # scale buffer keeps the full row stride (64) so the launch below specializes
            # exactly like the single-card one; columns [0, g1*8) hold chunks g0..7's codes
            sbb = torch.zeros_like(sbw, device=d1)
            sbb[:, : sbw.shape[1] - self.r0 // 128] = sbw[:, self.r0 // 128:].to(d1)
            self.w1.append((wqa, wsa, wb, sbb))
        k1, r1 = g1 * self.KG, g1 * self.R
        with torch.cuda.device(d1):
            self.o1 = torch.empty((1, k1), dtype=torch.bfloat16, device=d1)
            self.a8 = torch.empty((1, k1), dtype=e4m3_act_dtype(), device=d1)
            self.sa = torch.empty((1, k1 // 128), dtype=torch.uint8, device=d1)
            self.parta1 = torch.empty((8, 1, self.G * self.R), dtype=torch.float32, device=d1)
            self.olora1 = torch.empty((1, r1), dtype=torch.bfloat16, device=d1)
        self.partb = torch.empty((8, self.dim), dtype=torch.float32, device=d0)
        # dev0's o_lora lives in a FULL-length buffer (first r0 entries used) so its wo_b
        # launch passes the same pointers/strides as the single-card one
        self.olora0 = torch.empty((1, self.G * self.R), dtype=torch.bfloat16, device=d0)
        self.e_in = torch.cuda.Event()
        self.e_out = torch.cuda.Event()
        self.peer_read = os.environ.get("FREETOKEN_DSV4_EP_PEER_READ", "1") == "1"
        torch.cuda.synchronize(d1)
        logger.info_rank0(f"DSV4 EP dense split: wo_a groups / wo_b chunks dev0:dev1 = {self.g0}:{g1}")

    # -- kernels (same launches as fp8_linear.grouped_fp8_gemv / _fp8_act_gemv, into buffers)
    @staticmethod
    def _act_quant_into(x2d, y, s):
        from freetoken.kernel.triton.dsv4.fp8_linear import _act_quant_fp8_kernel

        M, K = x2d.shape
        _act_quant_fp8_kernel[(triton.cdiv(M, 32), K // 128)](
            x2d, y, s, M, K,
            x2d.stride(0), x2d.stride(1), y.stride(0), y.stride(1), s.stride(0), s.stride(1),
            BLOCK_M=32, BLOCK=128,
        )

    def _grouped_into(self, a8, sa, w, sb, part, out, rows):
        """wo_a grouped GEMV + split-K reduce for the first ``rows`` output rows of the
        (pointer-advanced) ``w``/``sb``/``a8``/``sa``. Same exactness rule as ``_splitk_into``:
        the launch passes the full N (= G*R) and a full-width ``part`` so the compiled
        binary is the single-card one; only the grid's row-tile dimension shrinks."""
        from freetoken.kernel.triton.dsv4.fp8_linear import (
            _TL_DTYPE, _fp8_act_gemv_grouped_splitk_kernel, _splitk_reduce_kernel, e4m3_kernel_view)

        M = a8.shape[0]
        Nfull = self.G * self.R
        KG = w.shape[1]
        split_k = max(1, min(8, KG // 128))
        assert part.shape == (split_k, M, Nfull), part.shape
        for t in (a8, sa, w, sb, part, out):
            assert t.data_ptr() % 16 == 0, "pointer alignment changes the Triton specialization"
        _fp8_act_gemv_grouped_splitk_kernel[(rows // 16, split_k)](
            e4m3_kernel_view(a8), sa, e4m3_kernel_view(w), sb, part, Nfull, KG, M,
            a8.stride(0), a8.stride(1), sa.stride(0),
            w.stride(0), w.stride(1), sb.stride(0), sb.stride(1),
            part.stride(0), part.stride(1), part.stride(2),
            M_POW2=1, R_PER_G=self.R, BLOCK_N=16, SPLIT_K=split_k, num_warps=1,
        )
        assert M == 1
        p2 = part.view(split_k, Nfull)
        _splitk_reduce_kernel[(triton.cdiv(rows, 256),)](
            p2, out, rows, split_k, p2.stride(0), p2.stride(1),
            BLOCK=256, OUT=_TL_DTYPE[out.dtype], num_warps=2,
        )

    def _splitk_into(self, a, w, sb, part, chunks):
        """wo_b split-K partials for the first ``chunks`` chunks of ``a``/``w``/``sb``/``part``.

        EXACTNESS: the launch is the single-card one -- K = full 8192, SPLIT_K = 8, all
        pointers 16-byte aligned, every stride in the same Triton specialization class -- and
        only the grid's chunk dimension shrinks. Triton compiles one binary per (constexprs,
        specialization) key, and a different SPLIT_K / K compiles a different binary whose
        FMA contraction rounds fp32 partials differently (measured: 1 ulp). Callers pass
        pointers already advanced to their first chunk."""
        from freetoken.kernel.triton.dsv4.fp8_linear import (
            _decode_cfg, _fp8_act_gemv_splitk_kernel, e4m3_kernel_view)

        N = w.shape[0]
        Kfull = self.G * self.R
        BLOCK_N, split_k, num_warps = _decode_cfg(N, Kfull)  # (16, 8, 1)
        assert split_k == self.G
        for t in (a, w, sb, part):
            assert t.data_ptr() % 16 == 0, "pointer alignment changes the Triton specialization"
        _fp8_act_gemv_splitk_kernel[(triton.cdiv(N, BLOCK_N), chunks)](
            a, e4m3_kernel_view(w), sb, part, N, Kfull,
            a.stride(0), w.stride(0), w.stride(1),
            sb.stride(0), sb.stride(1), part.stride(0), part.stride(1),
            BLOCK_N=BLOCK_N, SPLIT_K=split_k, num_warps=num_warps,
        )

    def forward(self, li: int, at, o: torch.Tensor) -> torch.Tensor:
        """``o`` [1, 1, G, KG] (or any view with 1 * G * KG elements), after the inverse rope.
        Returns the wo_b output [1, 1, dim] (bf16) on dev0."""
        from freetoken.kernel.triton.dsv4.fp8_linear import (
            _TL_DTYPE, _splitk_reduce_kernel, e4m3_act_dtype)

        ep = self.ep
        d0 = ep.dev0
        main = torch.cuda.current_stream(d0)
        o2 = o.reshape(1, -1)
        if not o2.is_contiguous():
            o2 = o2.contiguous()
        g0, g1 = self.g0, self.g1
        wqa1, wsa1, wb1, sbb1 = self.w1[li]
        self.e_in.record(main)
        with torch.cuda.device(ep.dev1), torch.cuda.stream(ep.s1):
            ep.s1.wait_event(self.e_in)
            if self.peer_read:
                # the act-quant kernel reads dev1's slice of o straight from dev0 (each element
                # once) -> no DMA hop before the first dev1 kernel
                self._act_quant_into(o2[:, self.k0:], self.a8, self.sa)
            else:
                p2p_copy(self.o1, o2[:, self.k0:], self.o1.numel() * 2, ep.s1)
                self._act_quant_into(self.o1, self.a8, self.sa)
            self._grouped_into(self.a8, self.sa, wqa1, wsa1, self.parta1, self.olora1, g1 * self.R)
            self._splitk_into(self.olora1[0], wb1, sbb1, self.partb[g0:], g1)  # chunks g0..7
            self.e_out.record(ep.s1)
        # dev0: first g0 groups / chunks (views of the full weights, no copies)
        x0 = o2[:, : self.k0]
        a8 = torch.empty((1, self.k0), dtype=e4m3_act_dtype(), device=d0)
        sa = torch.empty((1, self.k0 // 128), dtype=torch.uint8, device=d0)
        self._act_quant_into(x0, a8, sa)
        parta = torch.empty((8, 1, self.G * self.R), dtype=torch.float32, device=d0)
        self._grouped_into(a8, sa, at.wo_a_q, at.wo_a_s, parta, self.olora0, self.r0)
        self._splitk_into(self.olora0[0], at.wo_b.weight, _sb_u8(at.wo_b.scale), self.partb, g0)
        main.wait_event(self.e_out)
        out = torch.empty(self.dim, dtype=torch.bfloat16, device=d0)
        _splitk_reduce_kernel[(triton.cdiv(self.dim, 256),)](
            self.partb, out, self.dim, 8, self.partb.stride(0), self.partb.stride(1),
            BLOCK=256, OUT=_TL_DTYPE[torch.bfloat16], num_warps=2,
        )
        return out.view(1, 1, self.dim)


# ----------------------------------------------------------------------------------------
# q projection on dev1, overlapped with dev0's kv / compressor / indexer branch
# ----------------------------------------------------------------------------------------
import triton.language as tl  # noqa: E402


@triton.jit
def _peer_copy_kernel(src_ptr, dst_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(dst_ptr + offs, tl.load(src_ptr + offs, mask=m), mask=m)


def _sb_u8(scale: torch.Tensor) -> torch.Tensor:
    return (scale.view(torch.uint8) if scale.dtype == torch.float8_e8m0fnu else scale).contiguous()


def _gemv_into(a, w, sb, part, out):
    """``fp8_linear._fp8_act_gemv`` (M=1 decode GEMV + split-K reduce) into caller buffers;
    ``out`` may live on the other card (the reduce then stores across PCIe)."""
    from freetoken.kernel.triton.dsv4.fp8_linear import (
        _TL_DTYPE, _decode_cfg, _fp8_act_gemv_splitk_kernel, _splitk_reduce_kernel, e4m3_kernel_view)

    N, K = w.shape
    BLOCK_N, split_k, num_warps = _decode_cfg(N, K)
    assert part.shape == (split_k, N), (part.shape, split_k, N)
    _fp8_act_gemv_splitk_kernel[(triton.cdiv(N, BLOCK_N), split_k)](
        a, e4m3_kernel_view(w), sb, part, N, K,
        a.stride(0), w.stride(0), w.stride(1),
        sb.stride(0), sb.stride(1), part.stride(0), part.stride(1),
        BLOCK_N=BLOCK_N, SPLIT_K=split_k, num_warps=num_warps,
    )
    _splitk_reduce_kernel[(triton.cdiv(N, 256),)](
        part, out, N, split_k, part.stride(0), part.stride(1),
        BLOCK=256, OUT=_TL_DTYPE[out.dtype], num_warps=2,
    )


def _gemm_into(x2d, w, sb, a8, sa, out):
    """``fp8_linear.block_fp8_linear`` for M > 1 (act_quant_fp8 + tensor-core FP8 GEMM) into
    caller buffers, same launches. Each output column is computed by the same 128-wide tile
    program whatever N is (tiles start at multiples of 128), so a weight split along N at a
    128 boundary (wq_a out of the fused wq_a|wkv) reproduces the fused columns exactly."""
    from freetoken.kernel.triton.dsv4.fp8_linear import (
        _TL_DTYPE, _act_quant_fp8_kernel, _fp8_act_gemm_kernel, e4m3_kernel_view)

    M, K = x2d.shape
    N = w.shape[0]
    _act_quant_fp8_kernel[(triton.cdiv(M, 32), K // 128)](
        x2d, a8, sa, M, K,
        x2d.stride(0), x2d.stride(1), a8.stride(0), a8.stride(1), sa.stride(0), sa.stride(1),
        BLOCK_M=32, BLOCK=128,
    )
    from freetoken.kernel.cuda import mxfp8_cublas as _mx
    if out.dtype == torch.bfloat16 and _mx.usable(M, w, out) and a8.stride(0) == K:
        _mx.mx_linear(a8, sa, w, sb, out)   # cuBLASLt MXFP8 (prefill sizes)
        return
    bm, nw = (64, 8) if M >= 256 else (32, 4)   # prefill tiles: bit-identical, 1.25-1.35x (11-18)
    _fp8_act_gemm_kernel[(triton.cdiv(M, bm), triton.cdiv(N, 128))](
        a8, e4m3_kernel_view(w), sa, sb, out,
        M, N, K,
        a8.stride(0), a8.stride(1), w.stride(0), w.stride(1),
        sa.stride(0), sa.stride(1), sb.stride(0), sb.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=bm, BLOCK_N=128, BLOCK_K=128,
        compute_type=_TL_DTYPE[out.dtype], num_warps=nw, num_stages=3,
    )


def _rmsnorm_into(x2d, weight, out, eps):
    """``norm.rms_norm`` into ``out`` (same kernel / launch)."""
    from freetoken.kernel.triton.dsv4.norm import _TL, _rmsnorm_kernel

    M, D = x2d.shape
    BLOCK_D = triton.next_power_of_2(D)
    num_warps = 4 if BLOCK_D <= 1024 else (8 if BLOCK_D <= 4096 else 16)
    _rmsnorm_kernel[(M,)](
        x2d, weight, out, M, D, eps, x2d.stride(0), out.stride(0),
        BLOCK_D=BLOCK_D, HAS_W=weight is not None, compute_type=_TL[out.dtype], num_warps=num_warps,
    )


def _bf16_gemv_into(x1d, w, out, rows=None, n_full=None):
    """``bf16_linear.bf16_linear_fp32`` (M=1 Triton GEMV, fp32 out) into ``out`` -- same
    launch, so the same compiled binary; ``out`` may be on the other card. ``rows`` computes
    only the first ``rows`` rows of the (pointer-advanced) ``w``/``out`` with the full-size
    ``n_full`` passed, so a row split keeps the single-card binary."""
    from freetoken.kernel.triton.dsv4.bf16_linear import _bf16_gemv_fp32_kernel

    N, K = w.shape
    rows = N if rows is None else rows
    BLOCK_K = min(triton.next_power_of_2(K), 4096)
    _bf16_gemv_fp32_kernel[(triton.cdiv(rows, 2),)](
        x1d, w, out, n_full or N, K, w.stride(0), w.stride(1),
        BLOCK_N=2, BLOCK_K=BLOCK_K, num_warps=4,
    )


def _kvgate_w(comp) -> torch.Tensor:
    """The compressor's fused wkv|wgate weight exactly as Compressor.decode_step builds it."""
    w2 = getattr(comp, "_kvgate_w", None)
    if w2 is None:
        w2 = comp._kvgate_w = torch.cat([comp.wkv.weight, comp.wgate.weight], 0).contiguous()
    return w2


def comp_dev1_bytes(attns) -> int:
    split = float(os.environ.get("FREETOKEN_DSV4_EP_COMP_SPLIT", "1.0"))
    n = 0
    for at in attns:
        for comp in ((at.compressor,) if at.compressor is not None else ()) + (
                (at.indexer.compressor,) if at.indexer is not None else ()):
            rows = comp.wkv.weight.shape[0] + comp.wgate.weight.shape[0]
            r0 = min(rows, max(0, int(round(split * rows / 16)) * 16))
            n += (rows - r0) * comp.wkv.weight.shape[1] * 2
    return n


def q_dev1_bytes(attns) -> int:
    n = 0
    for at in attns:
        for lin in (at.wq_a, at.wq_b) + ((at.indexer.wq_b,) if at.indexer is not None else ()):
            n += lin.weight.numel() + lin.scale.numel()
        n += at.q_norm.weight.numel() * 4
    return n


class EPDenseQ:
    """B=1 decode: dev1 runs wq_a -> q_norm -> (indexer wq_b) -> wq_b and stores the raw
    (pre-norm/rope) q and indexer q straight into dev0 buffers, while dev0 runs wkv, the
    kv path, the compressor(s) and the indexer's own compressor. dev0 waits only right
    before the first consumer (indexer scoring / attention). wq_a alone uses the same
    GEMV config as the fused wq_a|wkv projection (16/16/1), and wkv alone too, so every
    value is bit-identical to ``Attention.decode_step`` (tests/dsv4_ep/test_ep_dense.py)."""

    def __init__(self, ep, attns: list):
        from freetoken.kernel.triton.dsv4.fp8_linear import _decode_cfg

        self.ep = ep
        d0, d1 = ep.dev0, ep.dev1
        self.w1 = []
        self.wc1 = []
        self.comp = os.environ.get("FREETOKEN_DSV4_EP_DENSE_COMP", "1") == "1"
        # fraction of each compressor projection's rows computed on dev0 (the rest on dev1)
        self.comp_split = float(os.environ.get("FREETOKEN_DSV4_EP_COMP_SPLIT", "1.0"))
        for at in attns:
            idx = at.indexer
            self.w1.append((
                at.wq_a.weight.to(d1), _sb_u8(at.wq_a.scale).to(d1),
                at.q_norm.weight.to(d1), at.q_norm.eps,
                at.wq_b.weight.to(d1), _sb_u8(at.wq_b.scale).to(d1),
                (idx.wq_b.weight.to(d1), _sb_u8(idx.wq_b.scale).to(d1)) if idx is not None else None,
            ))
            # compressor input projections (wkv|wgate, bf16) of the attention / indexer compressors:
            # rows [0, r0) stay on dev0, rows [r0, N) run on dev1 (both halves land in one dev0
            # buffer), r0 = comp_split * N rounded to 16 rows
            def _split(comp):
                if not (self.comp and comp is not None):
                    return None
                w2 = _kvgate_w(comp)
                N = w2.shape[0]
                r0 = min(N, max(0, int(round(self.comp_split * N / 16)) * 16))
                return (w2, r0, w2[r0:].contiguous().to(d1) if r0 < N else None)
            ca = _split(at.compressor)
            ci = _split(idx.compressor) if idx is not None else None
            self.wc1.append((ca, ci))
            if self.comp:
                if at.compressor is not None:
                    at.compressor._ep_kvs = (self, "attn")
                if idx is not None:
                    idx.compressor._ep_kvs = (self, "idx")
        a0 = attns[0]
        D, QL = a0.dim, a0.q_lora_rank
        NQ = a0.n_heads * a0.head_dim
        ib = next((at.indexer for at in attns if at.indexer is not None), None)
        NI = ib.wq_b.weight.shape[0] if ib is not None else 0
        with torch.cuda.device(d1):
            self.x1 = torch.empty(D, dtype=torch.bfloat16, device=d1)
            self.part_a = torch.empty((_decode_cfg(QL, D)[1], QL), dtype=torch.float32, device=d1)
            self.qa1 = torch.empty(QL, dtype=torch.bfloat16, device=d1)
            self.qr1 = torch.empty(QL, dtype=torch.bfloat16, device=d1)
            self.part_b = torch.empty((_decode_cfg(NQ, QL)[1], NQ), dtype=torch.float32, device=d1)
            self.part_i = (torch.empty((_decode_cfg(NI, QL)[1], NI), dtype=torch.float32, device=d1)
                           if NI else None)
        self.q_raw = torch.empty(NQ, dtype=torch.bfloat16, device=d0)
        self.iq_raw = torch.empty(max(NI, 1), dtype=torch.bfloat16, device=d0)
        # B >= 2 decode (tensor-core GEMM path), up to max_bs rows
        from freetoken.kernel.triton.dsv4.fp8_linear import e4m3_act_dtype

        MB = self.max_bs = ep.max_bs
        f8 = e4m3_act_dtype()
        with torch.cuda.device(d1):
            self.xb = torch.empty((MB, D), dtype=torch.bfloat16, device=d1)
            self.a8x = torch.empty((MB, D), dtype=f8, device=d1)
            self.sax = torch.empty((MB, D // 128), dtype=torch.uint8, device=d1)
            self.qab = torch.empty((MB, QL), dtype=torch.bfloat16, device=d1)
            self.qrb = torch.empty((MB, QL), dtype=torch.bfloat16, device=d1)
            self.a8q = torch.empty((MB, QL), dtype=f8, device=d1)
            self.saq = torch.empty((MB, QL // 128), dtype=torch.uint8, device=d1)
        self.q_raw_b = torch.empty((MB, NQ), dtype=torch.bfloat16, device=d0)
        self.iq_raw_b = torch.empty((MB, max(NI, 1)), dtype=torch.bfloat16, device=d0)
        nca = max([w[0][0].shape[0] for w in self.wc1 if w[0] is not None] or [1])
        nci = max([w[1][0].shape[0] for w in self.wc1 if w[1] is not None] or [1])
        self.kv_attn = torch.empty(nca, dtype=torch.float32, device=d0)
        self.kv_idx = torch.empty(nci, dtype=torch.float32, device=d0)
        self.e_in = torch.cuda.Event()
        self.ev = {k: torch.cuda.Event() for k in ("attn", "idxq", "idx", "q")}
        self.pending = set()
        torch.cuda.synchronize(d1)
        logger.info_rank0(f"DSV4 EP dense split: q projection (wq_a/q_norm/wq_b{'/indexer wq_b' if NI else ''})"
                          f"{' + compressor wkv|wgate projections' if self.comp else ''} on {d1}")

    def launch(self, li: int, x: torch.Tensor) -> None:
        """dev1 order = order of first use on dev0: attention-compressor projection, q_lora,
        indexer q, indexer-compressor projection, q. One event per result."""
        if x.shape[0] > 1:
            return self._launch_batched(li, x)
        self.B = 1
        ep = self.ep
        main = torch.cuda.current_stream(ep.dev0)
        wqa, sqa, qnw, eps, wqb, sqb, iw = self.w1[li]
        ca, ci = self.wc1[li]
        x1d = x.reshape(-1)
        assert x1d.is_contiguous() and x1d.numel() == self.x1.numel()
        self.e_in.record(main)
        with torch.cuda.device(ep.dev1), torch.cuda.stream(ep.s1):
            ep.s1.wait_event(self.e_in)
            n = self.x1.numel()
            _peer_copy_kernel[(triton.cdiv(n, 1024),)](x1d, self.x1, n, BLOCK=1024)
            if ca is not None and ca[2] is not None:
                _bf16_gemv_into(self.x1, ca[2], self.kv_attn[ca[1]:], rows=ca[0].shape[0] - ca[1],
                                n_full=ca[0].shape[0])
                self.ev["attn"].record(ep.s1)
            _gemv_into(self.x1, wqa, sqa, self.part_a, self.qa1)
            _rmsnorm_into(self.qa1.view(1, -1), qnw, self.qr1.view(1, -1), eps)
            if iw is not None:
                _gemv_into(self.qr1, iw[0], iw[1], self.part_i, self.iq_raw[: iw[0].shape[0]])
                self.ev["idxq"].record(ep.s1)
            if ci is not None and ci[2] is not None:
                _bf16_gemv_into(self.x1, ci[2], self.kv_idx[ci[1]:], rows=ci[0].shape[0] - ci[1],
                                n_full=ci[0].shape[0])
                self.ev["idx"].record(ep.s1)
            _gemv_into(self.qr1, wqb, sqb, self.part_b, self.q_raw)
            self.ev["q"].record(ep.s1)
        # dev0's share of the compressor projections, on the main stream (it overlaps dev1's
        # half; dev0 still runs wkv / kv path first, ordered by the caller)
        self._x0 = x1d
        self.pending = {"q"} | ({"attn"} if ca is not None and ca[2] is not None else set()) | (
            {"idxq"} if iw is not None else set()) | ({"idx"} if ci is not None and ci[2] is not None else set())
        self._comp = {"attn": ca, "idx": ci}
        self._comp_done = set()

    def _launch_batched(self, li: int, x: torch.Tensor) -> None:
        """B >= 2: q path only, through the M > 1 tensor-core GEMMs (the compressor projections
        stay on dev0 there: bf16_linear_fp32 is cuBLAS for M > 1, whose algorithm choice is
        per device and therefore not bit-reproducible across the two cards)."""
        ep = self.ep
        B = x.shape[0]
        assert B <= self.max_bs
        self.B = B
        main = torch.cuda.current_stream(ep.dev0)
        wqa, sqa, qnw, eps, wqb, sqb, iw = self.w1[li]
        x2 = x.reshape(B, -1)
        assert x2.is_contiguous()
        self.e_in.record(main)
        with torch.cuda.device(ep.dev1), torch.cuda.stream(ep.s1):
            ep.s1.wait_event(self.e_in)
            n = x2.numel()
            _peer_copy_kernel[(triton.cdiv(n, 1024),)](x2, self.xb, n, BLOCK=1024)
            xb = self.xb[:B]
            _gemm_into(xb, wqa, sqa, self.a8x[:B], self.sax[:B], self.qab[:B])
            _rmsnorm_into(self.qab[:B], qnw, self.qrb[:B], eps)
            if iw is not None:
                _gemm_into(self.qrb[:B], iw[0], iw[1], self.a8q[:B], self.saq[:B],
                           self.iq_raw_b[:B, : iw[0].shape[0]])
                self.ev["idxq"].record(ep.s1)
            _gemm_into(self.qrb[:B], wqb, sqb, self.a8q[:B], self.saq[:B], self.q_raw_b[:B])
            self.ev["q"].record(ep.s1)
        self.pending = {"q"} | ({"idxq"} if iw is not None else set())
        self._ni = iw[0].shape[0] if iw is not None else 0

    def _wait(self, key: str) -> None:
        if key in self.pending:
            torch.cuda.current_stream(self.ep.dev0).wait_event(self.ev[key])
            self.pending.discard(key)

    def q(self) -> torch.Tensor:
        """Raw wq_b output [1, 1, n_heads*head_dim] (== Attention.wq_b(qr))."""
        self._wait("q")
        if self.B > 1:
            return self.q_raw_b[: self.B].view(self.B, 1, -1)
        return self.q_raw.view(1, 1, -1)

    def idx_q(self) -> torch.Tensor:
        """Raw indexer wq_b output [1, 1, idx_heads*idx_dim] (== Indexer.wq_b(qr))."""
        self._wait("idxq")
        if self.B > 1:
            return self.iq_raw_b[: self.B, : self._ni].view(self.B, 1, -1)
        return self.iq_raw.view(1, 1, -1)

    def comp_kvs(self, which: str) -> torch.Tensor:
        """Compressor wkv|wgate projection [1, N] fp32 (== bf16_linear_fp32(x, _kvgate_w)):
        dev0 computes rows [0, r0) here, then waits for dev1's rows [r0, N)."""
        w2, r0, _w1 = self._comp[which]
        buf = self.kv_attn if which == "attn" else self.kv_idx
        N = w2.shape[0]
        if which not in self._comp_done:
            if r0 > 0:
                _bf16_gemv_into(self._x0, w2, buf, rows=r0, n_full=N)
            self._comp_done.add(which)
        self._wait(which)
        return buf[:N].view(1, N)
