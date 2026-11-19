"""DSV4 MoE: sqrtsoftplus/hash router, shared SwiGLU expert, offloaded FP4 routed
experts (GPU slot-cache / cpu / hybrid decode paths)."""

from __future__ import annotations

import os as _os

import torch
import torch.nn.functional as F
from torch import nn

from freetoken.kernel.triton.dsv4.bf16_linear import bf16_linear_fp32
from freetoken.kernel.triton.dsv4.swiglu import fused_swiglu
from freetoken.layers import OffloadMoELayer

from .args import DeepseekV4Args
from .layers import Linear


def _in_decode() -> bool:
    from freetoken.core import get_global_ctx

    try:
        return get_global_ctx().batch.is_decode
    except AssertionError:  # no active batch (offline / unit tests)
        return False


class Gate(nn.Module):
    """MoE router: sqrtsoftplus scoring + hash routing (first ``n_hash_layers``)."""

    def __init__(self, layer_id: int, args: DeepseekV4Args):
        super().__init__()
        self.topk = args.n_activated_experts
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.hash = layer_id < args.n_hash_layers
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim, dtype=torch.bfloat16), requires_grad=False)
        if self.hash:
            self.tid2eid = nn.Parameter(
                torch.empty(args.vocab_size, args.n_activated_experts, dtype=torch.int64), requires_grad=False
            )
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(torch.empty(args.n_routed_experts, dtype=torch.float32), requires_grad=False)

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor):
        # prefill: row-invariant router GEMM (prefix hit == cold); decode: the fast cuBLAS path
        scores = bf16_linear_fp32(x, self.weight, row_invariant=not _in_decode())
        if (
            self.score_func == "sqrtsoftplus" and not self.hash and self.bias is not None
            and _os.environ.get("FREETOKEN_FUSED_ROUTE", "1") == "1"
        ):
            # tail-fusion, sqrtsoftplus flavor: softplus/sqrt/+bias/topk/gather/renorm
            # in one kernel. Deltas vs eager: <=1ulp log1p rounding, +1e-20 renorm.
            from freetoken.kernel.triton.fused_route import fused_route

            return fused_route(scores, self.bias, self.topk, True, self.route_scale,
                               act="sqrtsoftplus")
        if (
            self.score_func == "sigmoid" and not self.hash and self.bias is not None
            and _os.environ.get("FREETOKEN_FUSED_ROUTE", "1") == "1"
        ):
            # tail-fusion (ported from glm_moe_dsa): sigmoid/+bias/topk/gather/renorm
            # chain -> one kernel. Same selection semantics; the only deltas are
            # <=1ulp sigmoid rounding and a +1e-20 in the renorm denominator.
            from freetoken.kernel.triton.fused_route import fused_route

            return fused_route(scores, self.bias, self.topk, True, self.route_scale)
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.hash:
            indices = self.tid2eid[input_ids]
        else:
            indices = scores.topk(self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func != "softmax":
            weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights * self.route_scale
        return weights, indices


class Expert(nn.Module):
    """Dense SwiGLU expert (the shared expert; routed experts are offloaded FP4)."""

    def __init__(self, dim: int, inter_dim: int, swiglu_limit: float):
        super().__init__()
        self.w1 = Linear(dim, inter_dim, kind="fp8")
        self.w2 = Linear(inter_dim, dim, kind="fp8")
        self.w3 = Linear(dim, inter_dim, kind="fp8")
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # w1|w3 share input x: one block-fp8 GEMV, output [..., 2I] fed straight to
        # the gate_up-halves swiglu kernel (same math as the two-arg variant).
        # Valid when w1's out rows land on a 128-scale-block boundary; else fallback.
        w13 = getattr(self, "_w13", None)
        if w13 is None and self.w1.kind == "fp8" and self.w3.kind == "fp8" and self.w1.weight.shape[0] % 128 == 0:
            w = torch.cat([self.w1.weight.view(torch.uint8), self.w3.weight.view(torch.uint8)], 0)
            self._w13 = w13 = w.view(self.w1.weight.dtype)
            self._w13_s = torch.cat([self.w1.scale, self.w3.scale], 0).contiguous()
        if w13 is not None:
            from freetoken.kernel.triton.dsv4.fp8_linear import block_fp8_linear
            from freetoken.kernel.triton.dsv4.fused_moe import fused_swiglu as _gu_swiglu

            gu = block_fp8_linear(x, w13, self._w13_s, None)
            h = _gu_swiglu(gu, self.swiglu_limit).to(x.dtype)
        else:
            h = fused_swiglu(self.w1(x), self.w3(x), self.swiglu_limit, x.dtype)
        return self.w2(h)


class DSV4OffloadMoELayer(OffloadMoELayer):
    """Routed FP4 experts on the shared offload cache: the base whole-layer
    streaming prefill (grouped inline-dequant GEMM for dense chunks, GEMV
    below the route crossover) and slot-cache / cpu / hybrid decode paths
    (per-route dequant GEMV)."""

    def __init__(self, layer_id: int, args: DeepseekV4Args):
        super().__init__(
            layer_id=layer_id,
            num_experts=args.n_routed_experts,
            top_k=args.n_activated_experts,
            hidden_size=args.dim,
            intermediate_size=args.moe_inter_dim,
            renormalize=True,
            activation="silu",
        )
        self.swiglu_limit = args.swiglu_limit
        # Two-GPU expert parallelism (FREETOKEN_DSV4_EP=1, see ep.py): set by
        # DeepseekV4ForCausalLM.init_expert_parallel; None = single-GPU offload path.
        self.ep = None

    def routed_forward(self, hidden_states, topk_weights, topk_ids, overlap_fn=None,
                       split_overlap_fn=None):
        ep = self.ep
        if ep is not None:
            from freetoken.core import get_global_ctx

            return ep.forward(self.layer_id, hidden_states, topk_weights, topk_ids,
                              overlap_fn=overlap_fn,
                              is_prefill=get_global_ctx().batch.is_prefill)
        return super().routed_forward(hidden_states, topk_weights, topk_ids,
                                      overlap_fn=overlap_fn, split_overlap_fn=split_overlap_fn)

    def _prefill_routed(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        # Whole-layer streaming moves all num_experts rows per layer; a small
        # chunk touches at most T*top_k of them, so below that crossover the
        # decode-style on-demand slot path strictly moves fewer bytes (and
        # keeps short-prompt slot residency -- hence hybrid decode's GPU/CPU
        # route split -- unchanged). Mixing modes across chunks is safe: the
        # streaming buffers disown their borrowed slots on invalidation.
        cache = self.offload_cache
        assert cache is not None
        # unpinned (LOCKED) layers must take the base materialize path: their copy_missing is the whole-layer pageable branch with position == expert id, which ensure_experts's LRU slot remap would contradict (the GEMM would gather other experts' weights)
        if (
            hidden_states.shape[0] * self.top_k >= self.num_experts
            or cache.is_unpinned_layer(self.layer_id)
        ):
            return super()._prefill_routed(hidden_states, topk_weights, topk_ids)
        cache.ensure_experts(self.layer_id, topk_ids)  # in-place expert-id -> slot
        cache.copy_missing()
        if cache.collect_stats:
            cache.record_decode_stats(self.layer_id)
        return self._expert_gemm(
            cache,
            hidden_states,
            topk_weights,
            topk_ids,
            views=cache.bank_views(),
            n=None,
            alphas=cache.alphas_for_slots(self.layer_id),
            is_prefill=True,
        )


class MoE(nn.Module):
    """Sparse MoE: hash/score router -> offloaded FP4 routed experts + shared expert."""

    def __init__(self, layer_id: int, args: DeepseekV4Args):
        super().__init__()
        self.dim = args.dim
        self.gate = Gate(layer_id, args)
        self.shared_experts = Expert(args.dim, args.moe_inter_dim, args.swiglu_limit)
        self.experts = DSV4OffloadMoELayer(layer_id, args)
        # Speculative expert prefetch: the offload layer predicts the NEXT layer's
        # routing with that layer's own gate (spec_predict_scores below); register
        # by reference (survives weight load). Inert unless FREETOKEN_MOE_SPEC_PREFETCH>0.
        if hasattr(self.experts, "spec_set_block"):
            self.experts.spec_set_block(self)

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x, input_ids.flatten())
        # Shared expert placement: under FREETOKEN_MOE_COPY_OVERLAP the routed layer
        # invokes it via overlap_fn WHILE the miss-fetch gather runs on a side stream
        # (compute hidden behind PCIe); otherwise enqueue it up front as before (the
        # hybrid-CPU overlap rationale).
        shared_box: list = []
        routed = self.experts.routed_forward(
            x, weights.float().contiguous(), indices.to(torch.int32).contiguous(),
            overlap_fn=lambda: shared_box.append(self.shared_experts(x)),
        )
        shared = shared_box[0] if shared_box else self.shared_experts(x)
        return (routed + shared).view(shape)

    def forward_launch(self, x: torch.Tensor, input_ids: torch.Tensor):
        """EP prefill, split phase (the in-chunk micro-batch pipeline, model.py): router + the
        expert-parallel dispatch (dev1's routes start; dev0's own routes and the shared expert
        are enqueued). :meth:`forward_finish` waits for dev1 and combines -- the same ops in
        the same order as :meth:`forward`, so the result is identical."""
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x, input_ids.flatten())
        shared_box: list = []
        handle = self.experts.ep.prefill_launch(
            self.experts.layer_id, x, weights.float().contiguous(), indices.to(torch.int32).contiguous(),
            overlap_fn=lambda: shared_box.append(self.shared_experts(x)),
        )
        return shape, x, handle, shared_box

    def forward_finish(self, state) -> torch.Tensor:
        shape, x, handle, shared_box = state
        routed = self.experts.ep.prefill_finish(handle)
        shared = shared_box[0] if shared_box else self.shared_experts(x)
        return (routed + shared).view(shape)

    def spec_predict_scores(self, hidden: torch.Tensor):
        """Selection scores for the speculative prefetch (same transform chain as
        Gate.forward minus the gather/renorm). ``None`` for hash-routed layers --
        their routing depends on input_ids, which the previous layer cannot know."""
        g = self.gate
        if g.hash:
            return None
        scores = bf16_linear_fp32(hidden, g.weight)
        if g.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif g.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        if g.bias is not None:
            scores = scores + g.bias
        return scores
