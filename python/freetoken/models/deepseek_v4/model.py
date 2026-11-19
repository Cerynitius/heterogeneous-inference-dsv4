"""DeepSeek-V4-Flash model (engine-native FreeToken port of inference/model.py).

A faithful single-stream port of the reference (MLA attention with a sliding window
+ stateful KV compressors / Lightning Indexer, manifold-constrained Hyper-Connections,
sqrtsoftplus / hash MoE), wired onto FreeToken's shared paged engine:

  - KV lives in DSV4-owned paged pools (:class:`~freetoken.kvcache.dsv4_paged_pool.DSV4PagedKVCache`):
    each layer's window-ring + compressed KV is a region of a shared global pool, addressed by
    per-layer slot maps. Sparse attention is a PAGED physical-slot gather:
    ``sparse_attn_paged`` reads KV directly from the two global pools (window / compressed) at
    the GLOBAL top-k slots inside the kernel, with no per-forward staging ``index_select``.
  - The model is a registered :class:`BaseLLMModel` built by ``create_model``; it is driven
    by ``ctx.batch`` (positions / input_ids), captured by the engine ``GraphRunner`` for
    decode, and freed by dropping page-table indices (no separate runner / python cursor).
  - Routed FP4 experts are served from :class:`OffloadMoeCache` (on-demand) -- the
    framework's core acceleration.

Heavy ops are FreeToken Triton kernels. Precision matches the reference (FP8/FP4
activation quant + Hadamard rotation re-introduced; see ``ops.py`` / the dsv4 kernels).
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch import nn

from freetoken.core import get_global_ctx
from freetoken.kernel.triton.dsv4.hc import hc_post_combine, hc_pre_combine
from freetoken.kernel.triton.dsv4.sinkhorn import hc_split_sinkhorn
from freetoken.kernel.triton.dsv4.hc_norm import hc_mix_gemv, hc_mix_rms_rows, hc_rms_cast

_HC_ROWS = os.environ.get("FREETOKEN_DSV4_HC_ROWS", "1") == "1"
from freetoken.models.blocks import BaseLLMModel

from .args import DeepseekV4Args
from .attention import Attention
from .moe import MoE

# Re-exports: keep every class/helper previously defined here importable from .model
# (external import stability; the moved definitions live in their own modules).
from .compress import Compressor, Indexer  # noqa: F401
from .layers import (  # noqa: F401
    FP8,
    Linear,
    RMSNorm,
    get_compress_topk_idxs,
    get_window_topk_idxs,
)
from .moe import Expert, Gate  # noqa: F401

# In-chunk 2-micro-batch prefill pipeline under expert parallelism (Transformer.
# _prefill_pipelined): a segment longer than FREETOKEN_DSV4_PREFILL_MB_UNIT tokens is cut at that
# many tokens (FREETOKEN_DSV4_PREFILL_MB=0: never). The unit must not be small: the FP4 grouped
# MoE GEMM's cost is mostly per-expert weight dequant, so halving its tokens barely halves its
# time (2048-token micro-batches made the MoE +33..41% dearer, eating the overlap). Serve with
# FREETOKEN_PREFILL_CHUNK_CAP = 2 x unit so a chunk splits into two full-size micro-batches.
# Decode hc boundaries fused into 2 launches (hc_fused.hc_post_pre, bit-identical); 0 = per-op path
_HC_POSTPRE = os.environ.get("FREETOKEN_DSV4_HC_POSTPRE", "1") == "1"
_PREFILL_MB = os.environ.get("FREETOKEN_DSV4_PREFILL_MB", "1") != "0"
_PREFILL_MB_UNIT = int(os.environ.get("FREETOKEN_DSV4_PREFILL_MB_UNIT", "4096"))


class Block(nn.Module):
    """Decoder block with manifold-constrained Hyper-Connections (4 residual streams)."""

    def __init__(self, layer_id: int, args: DeepseekV4Args):
        super().__init__()
        self.layer_id = layer_id
        self.norm_eps = args.norm_eps
        self.dim = args.dim
        self.attn = Attention(layer_id, args)
        self.ffn = MoE(layer_id, args)
        self.attn_norm = RMSNorm(args.dim, self.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, self.norm_eps)
        self.hc_mult = hc_mult = args.hc_mult
        self.hc_sinkhorn_iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * args.dim
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32), requires_grad=False)
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32), requires_grad=False)
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32), requires_grad=False)
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32), requires_grad=False)
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32), requires_grad=False)
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32), requires_grad=False)

    def hc_pre(self, x, hc_fn, hc_scale, hc_base):
        shape0 = x.shape
        M0 = shape0[0] * shape0[1]
        if M0 <= 16 and os.environ.get("FREETOKEN_DSV4_HC_FUSED", "1") == "1":
            # decode: the whole pre-mix (rms + MIX dots + sinkhorn + combine) in ONE launch
            from freetoken.kernel.triton.dsv4.hc_fused import hc_pre_fused

            y, post, comb = hc_pre_fused(
                x.reshape(M0, self.hc_mult, self.dim), hc_fn, hc_scale, hc_base,
                self.norm_eps, self.hc_eps, self.hc_sinkhorn_iters, x.dtype,
            )
            return (y.view(*shape0[:2], self.dim), post,
                    comb.view(M0, self.hc_mult, self.hc_mult))
        # tail-fusion (ported from glm5_next 2026-08-29): cast+square+mean+rsqrt -> one
        # kernel; gemv with rsqrt epilogue -> one kernel; pre_combine upcasts bf16
        # in-register (the eager chain was ~7 launches + two fp32 casts per call).
        shape, dtype = x.size(), x.dtype
        x2 = x.flatten(2)
        M = shape[0] * shape[1]
        if _HC_ROWS:
            # prefill: row-blocked rms+mix, one bf16 pass (9-10x vs rms_cast+gemv at M=3443)
            mixes = hc_mix_rms_rows(x2.reshape(M, -1), hc_fn, self.norm_eps)
        else:
            xf, rs = hc_rms_cast(x2.reshape(M, -1), self.norm_eps)
            mixes = hc_mix_gemv(xf, hc_fn, rs)
        pre, post, comb = hc_split_sinkhorn(
            mixes, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps
        )
        y = hc_pre_combine(x2.reshape(M, self.hc_mult, self.dim), pre, dtype).view(*shape[:2], self.dim)
        return y, post.view(M, self.hc_mult), comb.view(M, self.hc_mult, self.hc_mult)

    def hc_post(self, x, residual, post, comb):
        shape = residual.size()
        M = shape[0] * shape[1]
        y = hc_post_combine(
            x.reshape(M, self.dim), residual.reshape(M, self.hc_mult, self.dim), post, comb
        )
        return y.view(shape)

    def prefill_attn_half(self, x, segments, flat_positions):
        """First half of :meth:`prefill_batched` (split phase for the micro-batch pipeline):
        attention + its HC, then the FFN's HC pre-mix and norm. Returns the FFN input and the
        HC state :meth:`prefill_ffn_finish` needs."""
        residual = x
        x, post, comb = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = self.attn_norm(x)
        x = self.attn.forward_ragged(x, segments, flat_positions)
        x = self.hc_post(x, residual, post, comb)
        residual = x
        x, post, comb = self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = self.ffn_norm(x)
        return x, (residual, post, comb)

    def prefill_ffn_finish(self, ffn_state, hc_state):
        residual, post, comb = hc_state
        return self.hc_post(self.ffn.forward_finish(ffn_state), residual, post, comb)

    def prefill_batched(self, x, input_ids, segments, flat_positions):
        # Ragged batched prefill (cu_seqlens, no padding; bs >= 1, cold and radix-hit segments
        # mixed freely). ``x`` is [1, T, hc_mult, dim] -- the requests' token streams
        # concatenated. Per-token ops (HC / norm / MoE) run batched over ALL T tokens (the
        # MoE-offload amortization win); attention runs as ONE flat cu_seqlens launch over all
        # T queries (Attention.forward_ragged), with the stateful compressor/indexer looped per
        # request. ``segments`` = [(offset, extend_len, table_idx, start_pos)] off the attention
        # metadata; ``flat_positions`` [T] = per-token ABSOLUTE position (batch.positions).
        residual = x
        x, post, comb = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = self.attn_norm(x)
        x = self.attn.forward_ragged(x, segments, flat_positions)
        x = self.hc_post(x, residual, post, comb)

        residual = x
        x, post, comb = self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = self.ffn_norm(x)
        x = self.ffn(x, input_ids)
        x = self.hc_post(x, residual, post, comb)
        return x

    def decode_attn_ffn(self, y, residual, post, comb, pos, rows, cmp_stage_cap, input_ids, wctx, dctx):
        """Decode with the hc boundaries fused (hc_fused.hc_post_pre): takes this block's attention
        input ``y`` (already hc_pre-mixed) + HC state, returns the ffn output ``a`` and the HC state
        for the caller to fuse into the next block's attention hc_pre (or a final hc_post)."""
        from freetoken.kernel.triton.dsv4.hc_fused import hc_post_pre

        a = self.attn.decode_step(self.attn_norm(y), pos, rows, cmp_stage_cap, wctx, dctx)
        residual, y, post, comb = hc_post_pre(
            a, residual, post, comb, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base,
            self.norm_eps, self.hc_eps, self.hc_sinkhorn_iters)
        a = self.ffn(self.ffn_norm(y.view(-1, 1, self.dim)), input_ids)
        return a, residual, post, comb

    def decode_step(self, x, pos, rows, cmp_stage_cap, input_ids, wctx=None, dctx=None):
        residual = x
        x, post, comb = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = self.attn_norm(x)
        x = self.attn.decode_step(x, pos, rows, cmp_stage_cap, wctx, dctx)
        x = self.hc_post(x, residual, post, comb)

        residual = x
        x, post, comb = self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = self.ffn_norm(x)
        x = self.ffn(x, input_ids)
        x = self.hc_post(x, residual, post, comb)
        return x


class Transformer(nn.Module):
    def __init__(self, args: DeepseekV4Args):
        super().__init__()
        self.args = args
        self.norm_eps = args.norm_eps
        self.hc_eps = args.hc_eps
        self.hc_mult = hc_mult = args.hc_mult
        self.embed = nn.Embedding(args.vocab_size, args.dim)
        self.embed.weight.requires_grad_(False)
        self.layers = nn.ModuleList([Block(i, args) for i in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, self.norm_eps)
        # FREETOKEN_DSV4_HEAD_FP8=1: W8A16 lm_head (fp8-e4m3 + per-row scale, quantized
        # at load) -- the full-vocab logits GEMV reads ~1 GB of bf16 every decode step
        # (measured 670 us, 3.7% of the step); fp8 halves it. Same scheme as GlmFp8LMHead.
        self._head_fp8 = os.environ.get("FREETOKEN_DSV4_HEAD_FP8", "0") == "1"
        if self._head_fp8:
            self.head = nn.Parameter(
                torch.empty(args.vocab_size, args.dim, dtype=torch.float8_e4m3fn),
                requires_grad=False)
            self.head_scale = nn.Parameter(
                torch.empty(args.vocab_size, dtype=torch.float32), requires_grad=False)
        else:
            self.head = nn.Parameter(torch.empty(args.vocab_size, args.dim, dtype=torch.bfloat16), requires_grad=False)
        hc_dim = hc_mult * args.dim
        self.hc_head_fn = nn.Parameter(torch.empty(hc_mult, hc_dim, dtype=torch.float32), requires_grad=False)
        self.hc_head_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32), requires_grad=False)
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32), requires_grad=False)

    def bind(self, pool, device: torch.device) -> None:
        for layer in self.layers:
            layer.attn.bind(pool, device)
        # eager (first forward, before any graph capture): keeps the indexer top-k workspace
        # zero-fills out of the captured decode graphs
        from freetoken.kernel.triton.dsv4.topk_exact import topk_workspace_prewarm

        topk_workspace_prewarm(device, self.args.index_topk)

    def hc_head(self, x):
        shape, dtype = x.size(), x.dtype
        dim = self.args.dim
        x2 = x.flatten(2)
        M = shape[0] * shape[1]
        xf, rs = hc_rms_cast(x2.reshape(M, -1), self.norm_eps)
        mixes = hc_mix_gemv(xf, self.hc_head_fn, rs)
        pre = torch.sigmoid(mixes * self.hc_head_scale + self.hc_head_base) + self.hc_eps
        return hc_pre_combine(x2.reshape(M, self.hc_mult, dim), pre.view(M, self.hc_mult), dtype).view(*shape[:2], dim)

    def prefill_batched(
        self, input_ids: torch.Tensor, segments, flat_positions: torch.Tensor,
        last_indices: torch.Tensor,
    ) -> torch.Tensor:
        # Ragged batched prefill (bs >= 1). ``input_ids`` is [1, T] -- the requests' NEW tokens
        # concatenated (cu_seqlens, no padding); each request starts at its own cached_len
        # (cold == 0, radix hit / chunk continuation > 0). Per-token ops run batched over all T
        # tokens; attention runs per request on its [offset, offset+n) segment (see Block). Each
        # request's window/cmp/idx slots + ring blocks were allocated disjointly by
        # allocate_paged, so the per-request attention never reads another request's KV/carry.
        # ``segments`` [(offset, extend_len, table_idx, start_pos)] comes off the attention
        # metadata; ``flat_positions`` [T] is the scheduler-staged batch.positions (per-token
        # ABSOLUTE position); ``last_indices`` [B] is the flattened index of each request's
        # final token -> its next-token logits row.
        h = self.embed(input_ids)
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        tg = getattr(self, "_spec_targets", None)
        parts = [] if tg is not None else None
        cut = self._micro_batch_cut(segments) if tg is None else 0
        if cut:
            h = self._prefill_pipelined(h, input_ids, segments, flat_positions, cut)
        else:
            for li, layer in enumerate(self.layers):
                h = layer.prefill_batched(h, input_ids, segments, flat_positions)
                if parts is not None and li in tg:
                    parts.append(h[0].mean(dim=1))  # [T, dim] hc-mean at a target layer
        if parts is not None:
            _b = get_global_ctx().batch
            _flat = torch.cat(parts, dim=-1)  # [T, n_targets * dim]
            _hd = {}
            for _i, (_off, _n, _ti, _start) in enumerate(segments):
                _lo = max(0, _n - 128)  # the rings hold at most the trailing window
                _hd[_i] = (flat_positions[_off + _lo: _off + _n],
                           _flat[_off + _lo: _off + _n].clone())
            _b.spec_hiddens = _hd
        h = self.hc_head(h)
        h = self.norm(h)
        if self._head_fp8:
            from freetoken.kernel.triton.fp8_pertensor_linear import fp8_pertensor_linear

            return fp8_pertensor_linear(h[0, last_indices].contiguous(), self.head, self.head_scale)
        return F.linear(h[0, last_indices], self.head)  # [B, vocab]

    def _micro_batch_cut(self, segments) -> int:
        """Token split of the in-chunk 2-micro-batch prefill pipeline (0 = run unsplit): one
        request's segment longer than FREETOKEN_DSV4_PREFILL_MB_UNIT tokens under expert
        parallelism, cut at the unit (a window-page multiple) so the second micro-batch is an
        ordinary chunk continuation."""
        if not _PREFILL_MB or len(segments) != 1:
            return 0
        if getattr(self.layers[0].ffn.experts, "ep", None) is None:
            return 0
        _off, n, _ti, start = segments[0]
        P = self.args.window_size
        cut = _PREFILL_MB_UNIT
        if start % P or cut % P or n <= cut:
            return 0
        return cut

    def _prefill_pipelined(self, h, input_ids, segments, flat_positions, cut):
        """In-chunk 2-micro-batch prefill (expert parallelism). Tokens [0, cut) and [cut, n) run
        as two consecutive segments of the same request -- exactly a chunk boundary at ``cut``,
        which the row-invariant prefill makes bit-identical to the unsplit chunk -- software-
        pipelined so the PRO 5000 computes one micro-batch's routed experts while the PRO 6000
        runs the other's attention: per layer attn(a) precedes attn(b) (b reads a's KV / carry
        of that layer), and a's MoE is finished only after b's attention and dispatch are
        enqueued."""
        _off, n, ti, start = segments[0]
        ha, hb = h[:, :cut], h[:, cut:]
        ia, ib = input_ids[:, :cut], input_ids[:, cut:]
        sa, sb = [(0, cut, ti, start)], [(0, n - cut, ti, start + cut)]
        pa, pb = flat_positions[:cut], flat_positions[cut:]
        layers = self.layers
        xa, hca = layers[0].prefill_attn_half(ha, sa, pa)
        fa = layers[0].ffn.forward_launch(xa, ia)
        for li, layer in enumerate(layers):
            xb, hcb = layer.prefill_attn_half(hb, sb, pb)
            fb = layer.ffn.forward_launch(xb, ib)
            ha = layer.prefill_ffn_finish(fa, hca)
            if li + 1 < len(layers):
                xa, hca = layers[li + 1].prefill_attn_half(ha, sa, pa)
                fa = layers[li + 1].ffn.forward_launch(xa, ia)
            hb = layer.prefill_ffn_finish(fb, hcb)
        return torch.cat([ha, hb], dim=1)

    def _decode_layers_postpre(self, h, pos, rows, cmp_stage_cap, input_ids, wctx, dctx):
        # h [B, 1, HC, dim]. Every ffn hc_post is fused with the next block's attention hc_pre
        # (and each block's attention hc_post with its ffn hc_pre): 2 launches per boundary.
        from freetoken.kernel.triton.dsv4.hc_fused import hc_post_pre

        shape = h.shape
        l0 = self.layers[0]
        y, post, comb = l0.hc_pre(h, l0.hc_attn_fn, l0.hc_attn_scale, l0.hc_attn_base)
        residual = h.view(-1, l0.hc_mult, l0.dim)
        last = len(self.layers) - 1
        for li, layer in enumerate(self.layers):
            a, residual, post, comb = layer.decode_attn_ffn(
                y, residual, post, comb, pos, rows, cmp_stage_cap, input_ids, wctx, dctx)
            if li < last:
                nxt = self.layers[li + 1]
                residual, y, post, comb = hc_post_pre(
                    a, residual, post, comb, nxt.hc_attn_fn, nxt.hc_attn_scale, nxt.hc_attn_base,
                    nxt.norm_eps, nxt.hc_eps, nxt.hc_sinkhorn_iters)
                y = y.view(*shape[:2], -1)
            else:
                h = layer.hc_post(a, residual.view(shape), post, comb)
        return h

    def decode(
        self, input_ids: torch.Tensor, pos: torch.Tensor, cmp_stage_cap: int
    ) -> torch.Tensor:
        # input_ids [B,1], pos [B] (GPU int). cmp_stage_cap = max position any row reaches; each
        # layer derives its compressed staging width = (cmp_stage_cap+1)//ratio (max valid count
        # over rows in eager; a static capture width = max_seq-1 under graph).
        #
        # Overlap safety: the global page-table rows are NOT read inside the graph. The attention
        # metadata carries a snapshot of the active rows' whole-history full locs (staged before a
        # replay); the decode derives every window/cmp/idx read slot from that snapshot IN-GRAPH by
        # LOCAL row ``rows`` = arange(B). So the next batch's allocate_paged cannot corrupt this
        # in-flight replay (it mutates only the live map).
        B = input_ids.size(0)
        rows = torch.arange(B, device=input_ids.device)
        h = self.embed(input_ids)
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        # Hoist the layer-invariant per-step decode tensors (shared window-ring global slots):
        # resolved ONCE off the attention metadata (recomputed per call, so a capture records
        # the gathers -- never cache it on the metadata) and threaded
        # into every layer. They read only the shared snapshot / positions, so they are identical
        # across layers.
        wctx = get_global_ctx().batch.attn_metadata.window_ctx(pos, rows)
        # Layer-invariant decode memo (freqs gathers, pos arithmetic): fresh per step,
        # first layer fills, the other 42 reuse. See ops.dctx_get.
        dctx = {}
        tg = getattr(self, "_spec_targets", None)
        if tg is None and B <= 16 and _HC_POSTPRE:
            h = self._decode_layers_postpre(h, pos, rows, cmp_stage_cap, input_ids, wctx, dctx)
        else:
            for li, layer in enumerate(self.layers):
                h = layer.decode_step(h, pos, rows, cmp_stage_cap, input_ids, wctx, dctx)
                if tg is not None and li in tg:
                    j = tg.index(li)
                    self.spec_hidden_decode[:B, j * self.args.dim:(j + 1) * self.args.dim].copy_(
                        h.mean(dim=2)[:, 0])
        h = self.hc_head(h)
        h = self.norm(h)
        if self._head_fp8:
            from freetoken.kernel.triton.fp8_pertensor_linear import fp8_pertensor_linear

            return fp8_pertensor_linear(h[:, -1].contiguous(), self.head, self.head_scale)
        return F.linear(h[:, -1], self.head)


class DeepseekV4ForCausalLM(BaseLLMModel):
    """Engine adapter: a registered :class:`BaseLLMModel` wrapping the DSV4 transformer.

    Built on the engine ``meta`` device then loaded via :meth:`load_state_dict` (which
    casts each incoming tensor to its parameter dtype and assigns it). KV pools, rope
    constants and compressor state are bound on the first forward.
    """

    def __init__(self, config):
        self._config = config
        self._args: DeepseekV4Args = config.dsv4_args
        self._transformer = Transformer(self._args)
        self._bound = False

    def _ensure_bound(self) -> None:
        if self._bound:
            return
        pool = get_global_ctx().kv_cache
        self._transformer.bind(pool, pool.device)
        self._bound = True

    def mark_for_rebind(self) -> None:
        """Force a re-bind on the next forward. The model holds NO pool reference -- buffers are read
        off ctx.kv_cache via @property -- so a runtime rebuild needs no unbind; the old pool frees
        when the engine drops ctx.kv_cache. But the per-bind scratch (the indexer's arange over the
        block count, freqs) depends on the new pool's geometry, so re-derive it via _ensure_bound."""
        self._bound = False

    def _iter_offload_moe_layers(self):
        # Hook for moe.offload_cache.iter_offload_moe_layers: DSV4's MoE block (block.ffn)
        # is a bespoke nn.Module (not OffloadMoELayer), so yield its inner routed-expert
        # layer instead, whose .offload_cache the generic attach sets.
        for block in self._transformer.layers:
            yield block.ffn.experts

    def init_expert_parallel(self, config) -> None:
        """FREETOKEN_DSV4_EP=1: load every routed expert resident on two cards (this
        process's device + FREETOKEN_DSV4_EP_DEV1, default visible index 1) and switch
        each MoE layer to the EP path (ep.py). Replaces the offload cache entirely."""
        from .ep import DSV4ExpertParallel

        dev0 = torch.device("cuda", torch.cuda.current_device())
        dev1 = torch.device("cuda", int(os.environ.get("FREETOKEN_DSV4_EP_DEV1", "1")))
        assert dev0 != dev1, f"EP needs two devices, got {dev0} twice"
        from .ep_dense import (EPDenseQ, EPDenseWo, calibrate, comp_dev1_bytes, q_dev1_bytes,
                               wo_dev1_bytes, wo_groups_for_dev1)

        max_bs = max(config.max_running_req + 1, config.cuda_graph_max_bs or 0)
        ep = DSV4ExpertParallel(self._args, dev0, dev1, max_bs=max_bs)
        # Split ratios from measured card speeds: decode work is bandwidth-bound.
        cal = calibrate(dev0, dev1)
        attns = [blk.attn for blk in self._transformer.layers]
        g1 = wo_groups_for_dev1(cal["bw_share1"]) if all(a._woa_fp8 for a in attns) else 0
        dense_q = os.environ.get("FREETOKEN_DSV4_EP_DENSE_Q", "1") == "1"
        dense_comp = dense_q and os.environ.get("FREETOKEN_DSV4_EP_DENSE_COMP", "1") == "1"
        extra1 = ((wo_dev1_bytes(len(attns), g1) if g1 else 0) + (q_dev1_bytes(attns) if dense_q else 0)
                  + (comp_dev1_bytes(attns) if dense_comp else 0))
        ep.plan(dev1_extra_bytes=extra1)
        ep.load(config.model_path)
        ep.init_runtime()
        for block in self._transformer.layers:
            block.ffn.experts.ep = ep
        if g1:
            wo = EPDenseWo(ep, attns, g1)
            for a in attns:
                a._ep_wo = wo
        if dense_q:
            qp = EPDenseQ(ep, attns)
            for a in attns:
                a._ep_q = qp
        self._ep = ep

    def state_dict(self, *, prefix: str = "", result=None):
        result = {} if result is None else result
        for name, p in self._transformer.named_parameters():
            result[f"{prefix}.{name}" if prefix else name] = p
        return result

    def load_state_dict(self, state_dict, *, prefix: str = "", _internal: bool = False) -> None:
        casted = {}
        for name, p in self._transformer.named_parameters():
            key = f"{prefix}.{name}" if prefix else name
            if key not in state_dict:
                raise RuntimeError(f"Missing weight for DeepSeek-V4 parameter: {key}")
            casted[name] = state_dict.pop(key).to(p.dtype)
        if state_dict and not _internal:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")
        self._transformer.load_state_dict(casted, assign=True, strict=False)
        import os as _os
        if _os.environ.get("FREETOKEN_DSV4_SPEC", "0") == "1":
            from .dspark_load import load_dspark
            load_dspark(self._transformer, _os.environ["FREETOKEN_DSV4_SPEC_CKPT"])

    def forward(self) -> torch.Tensor:
        self._ensure_bound()
        batch = get_global_ctx().batch
        input_ids = batch.input_ids.long()
        md = batch.attn_metadata
        if batch.is_prefill:
            # Ragged batched prefill (bs >= 1): each request starts from its own cached_len.
            # A cold segment (start_pos == 0) re-seeds the compressor carry register inside its
            # own attention segment; a radix hit / chunk continuation (start_pos > 0) resumes it
            # FROM THE RING. Per-token ops (embed / HC / norm / MoE) run batched over the
            # concatenated tokens; attention runs per segment so the carry / slot maps never
            # cross requests.
            return self._transformer.prefill_batched(
                input_ids.view(1, -1), md.segments, batch.positions.long(),
                md.last_indices.long(),
            )
        # DECODE (bs>=1): per-row position (GPU int tensor -> no host syncs / graph safe). The
        # compressed staging cap is the max position any row reaches (eager); a static max_seq-1
        # under graph capture (so the captured static-shape graph serves any real replay position).
        B = batch.padded_size
        pos = batch.positions.long().view(-1)[:B]
        if torch.cuda.is_current_stream_capturing():
            # Stage exactly as wide as the snapshot those columns are gathered FROM. The backend
            # sizes it to the live ceiling min(model max, KV token budget), which the scheduler
            # also admits against, so no replay can reach a column past it -- and the two stay in
            # lockstep by construction rather than by convention.
            cmp_stage_cap = md.stage_width - 1
        else:
            cmp_stage_cap = int(pos.max().item())
        return self._transformer.decode(input_ids.view(B, 1), pos, cmp_stage_cap)


__all__ = ["Transformer", "DeepseekV4ForCausalLM"]
