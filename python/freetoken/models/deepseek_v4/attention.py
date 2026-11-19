"""Multi-head Latent Attention over the paged DSV4 pools (sliding window + compressed
top-k); owns the per-layer Compressor/Indexer instances."""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch import nn

from freetoken.core import get_global_ctx
from freetoken.kernel.triton.dsv4.fp8_linear import act_quant_fp8_inplace
from freetoken.kernel.triton.dsv4.norm import rms_norm
from freetoken.kernel.triton.dsv4.norm_rope import kv_norm_rope_quant_store, q_norm_rope

from .args import DeepseekV4Args
from .compress import _NOSYNC_SLOTS, Compressor, Indexer
from .layers import Linear, RMSNorm, get_compress_topk_idxs, get_window_topk_idxs
from .ops import apply_rotary_emb, apply_rotary_emb_decode, dctx_get, get_freqs_cis

# decode kv/q norm+rope(+quant+window write) fusions (norm_rope.py, bit-identical); 0 = per-op path
_NORM_ROPE = os.environ.get("FREETOKEN_DSV4_NORM_ROPE", "1") == "1"


# one-kernel index-list assembly for indexer layers at decode (exact; =0 restores the torch chain)
_IDX_ASSEMBLE = os.environ.get("FREETOKEN_DSV4_IDX_ASSEMBLE", "1") == "1"


# ratio-128 prefill: row-shared compressed list instead of [n, n_blocks] picks (CUDA attention only)
_SHARED_CMP = os.environ.get("FREETOKEN_DSV4_SHARED_CMP", "0") == "1"

class Attention(nn.Module):
    """Multi-head Latent Attention with sliding window + optional KV compression.

    KV lives in the paged DSV4 pools (window / compressed / indexer), resolved per token via
    the manager's slot maps. Attention is a PAGED gather: the top-k is built as GLOBAL pool
    slots (window-first, then compressed) and the backend's paged kernel gathers KV straight from
    ``window_pool[L]`` / ``cmp_pool[L]`` inside the kernel -- no per-forward staging slab. Both
    prefill and decode use the same paged kernel."""

    def __init__(self, layer_id: int, args: DeepseekV4Args):
        super().__init__()
        self.layer_id = layer_id
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.q_lora_rank = args.q_lora_rank
        self.o_lora_rank = args.o_lora_rank
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.n_groups = args.o_groups
        self.window_size = args.window_size
        self.compress_ratio = args.compress_ratios[layer_id]
        self.eps = args.norm_eps

        self.attn_sink = nn.Parameter(torch.empty(self.n_heads, dtype=torch.float32), requires_grad=False)
        self.wq_a = Linear(self.dim, self.q_lora_rank, kind="fp8")
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = Linear(self.q_lora_rank, self.n_heads * self.head_dim, kind="fp8")
        self.wkv = Linear(self.dim, self.head_dim, kind="fp8")
        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        # wo_a: dequantized to bf16 (reference runs a bf16 grouped-output einsum).
        wo_a_rows = self.n_groups * args.o_lora_rank
        wo_a_k = self.n_heads * self.head_dim // self.n_groups
        # FREETOKEN_DSV4_WOA_FP8=1: keep wo_a in its checkpoint-native FP8 (128x128
        # e8m0 blocks) and run decode through the grouped FP8 GEMV -- wo_a is the
        # largest remaining BF16 read of the decode step (67 MB/layer). The reference
        # runs this einsum in BF16, so the FP8 activation quant is a numerics change;
        # gate behind the quality battery before defaulting on.
        self._woa_fp8 = os.environ.get("FREETOKEN_DSV4_WOA_FP8", "0") == "1"
        if self._woa_fp8:
            self.wo_a_q = nn.Parameter(
                torch.empty(wo_a_rows, wo_a_k, dtype=torch.float8_e4m3fn), requires_grad=False)
            self.wo_a_s = nn.Parameter(
                torch.empty(wo_a_rows // 128, wo_a_k // 128, dtype=torch.uint8), requires_grad=False)
        else:
            self.wo_a = nn.Parameter(torch.empty(wo_a_rows, wo_a_k, dtype=torch.bfloat16), requires_grad=False)
        self.wo_b = Linear(self.n_groups * args.o_lora_rank, self.dim, kind="fp8")
        self.softmax_scale = self.head_dim ** -0.5

        if self.compress_ratio:
            self.compressor = Compressor(args, self.compress_ratio, self.head_dim)
            self.indexer = Indexer(args, self.compress_ratio) if self.compress_ratio == 4 else None
        else:
            self.compressor = None
            self.indexer = None

        if self.compress_ratio:
            original_seq_len, rope_theta = args.original_seq_len, args.compress_rope_theta
        else:
            original_seq_len, rope_theta = 0, args.rope_theta
        self._freqs_params = (
            self.rope_head_dim, args.max_seq_len, original_seq_len,
            rope_theta, args.rope_factor, args.beta_fast, args.beta_slow,
        )
        # Bound on first forward (freqs/arange only; pool buffers are read off the LIVE pool).
        self.freqs_cis: torch.Tensor | None = None

    # KV addressing lives in the attention backend (ctx.attn_backend), pool buffers on the live
    # pool (ctx.kv_cache) -- both read per access, so the model holds NO state a runtime rebuild
    # would have to unbind. Capture bakes whatever the property returns at capture time (stable).
    @property
    def attn(self):
        return get_global_ctx().attn_backend

    def bind(self, pool, device: torch.device) -> None:
        # pool is the DSV4PagedKVCache carrying the slot maps (set by the manager). The model holds
        # only the pool HANDLE; buffers + slot maps are read off it per access via @property, so a
        # runtime pool rebuild needs no per-buffer unbind.
        L = self.layer_id
        win = self.window_size
        self.P = pool.P
        self.freqs_cis = get_freqs_cis(*self._freqs_params, device)
        if self.compress_ratio:
            self.compressor.bind_paged(pool, L, self.freqs_cis, device, tier="attn")
            if self.indexer is not None:
                self.indexer.bind_paged(pool, L, self.freqs_cis, device)

    def reset(self) -> None:
        if self.compressor is not None:
            self.compressor.reset()
        if self.indexer is not None:
            self.indexer.reset()


    def _woa_dequant(self) -> torch.Tensor:
        """Per-call BF16 scratch for the (rare) non-GEMV paths under WOA_FP8: block
        128x128 dequant of the native FP8 weight (~67 MB write, ~70 us -- negligible
        against a prefill chunk; avoids keeping a persistent 2.7 GB BF16 copy)."""
        rows, k = self.wo_a_q.shape
        sc = torch.exp2(self.wo_a_s.float() - 127.0)
        w = self.wo_a_q.float().view(rows // 128, 128, k // 128, 128)
        return (w * sc[:, None, :, None]).view(rows, k).to(torch.bfloat16)

    def _wo(self, o: torch.Tensor, bsz: int, seqlen: int) -> torch.Tensor:
        o = o.reshape(bsz, seqlen, self.n_groups, -1)
        if self._woa_fp8:
            T = bsz * seqlen
            ep_wo = getattr(self, "_ep_wo", None)
            if T == 1 and ep_wo is not None:
                # two-card EP: proportional exact split of wo_a groups / wo_b split-K chunks
                res = ep_wo.forward(self.layer_id, self, o)
                if os.environ.get("FREETOKEN_DSV4_EP_WO_CHECK", "0") == "1":
                    from freetoken.kernel.triton.dsv4.fp8_linear import grouped_fp8_gemv

                    r = grouped_fp8_gemv(o.reshape(T, -1), self.wo_a_q, self.wo_a_s, self.n_groups)
                    ref = self.wo_b(r.view(bsz, seqlen, -1))
                    if not torch.equal(ref, res):
                        n = getattr(Attention, "_wo_bad", 0)
                        Attention._wo_bad = n + 1
                        if n < 5:
                            print(f"[wo-check] layer {self.layer_id} MISMATCH max|d|="
                                  f"{(ref.float() - res.float()).abs().max().item():.3e} o.shape={tuple(o.shape)} "
                                  f"stride={o.stride()} contig={o.is_contiguous()}", flush=True)
                            torch.save({"o": o.detach().cpu(), "ref": ref.cpu(), "res": res.cpu(),
                                        "layer": self.layer_id},
                                       os.path.join(os.environ.get("FREETOKEN_DEBUG_DIR", "/tmp"), f"wo_bad_{self.layer_id}_{n}.pt"))
                return res
            if T <= 16:  # decode batches (single + concurrent): single-launch grouped FP8
                from freetoken.kernel.triton.dsv4.fp8_linear import grouped_fp8_gemv

                r = grouped_fp8_gemv(
                    o.reshape(T, -1), self.wo_a_q, self.wo_a_s, self.n_groups)
                return self.wo_b(r.view(bsz, seqlen, -1))
            wo_a = self._woa_dequant().view(self.n_groups, self.o_lora_rank, -1)
        else:
            wo_a = self.wo_a.view(self.n_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a).flatten(2)
        return self.wo_b(o)

    def _prefill_segment(self, x_seg, qr_seg, kv_seg, ti: int, start_pos: int, n: int):
        """One request's prefill work inside a (possibly ragged) batch: persist its window KV,
        advance its compressor/indexer carry, and return its per-query candidate lists.

        ``start_pos == 0`` is a cold prompt (carry re-seeded from scratch); ``start_pos > 0`` is a
        radix hit, whose carry is seeded BY VALUE from the matched tail page's ring block. Returns
        ``(win_global [1, n, w], cmp_global [1, n, c] | None)`` at natural widths (cold: w =
        min(n, win); extend: w = win); the caller pads BOTH halves to the batch-uniform widths
        and concatenates.
        """
        win, ratio, device = self.window_size, self.compress_ratio, x_seg.device
        end = start_pos + n
        slots = self.attn.window_slots_of(ti, start_pos, end)
        self.attn.store_window(kv_seg, self.layer_id, slots)

        if start_pos == 0:
            win_cols = get_window_topk_idxs(win, 1, n, 0).to(device)
            # natural width min(n, win); the caller pads to the batch-uniform width
            win_global = self.attn.win_cols_to_global(win_cols, slots)
        else:
            # Candidates span [w_lo, end): each new query's 128-sliding window over the retained
            # prefix plus the new tokens, causal-masked to -1 past its own position.
            w_lo = max(0, start_pos - win + 1)
            ws_pool = self.attn.window_slots_of(ti, w_lo, end)
            abs_p = start_pos + torch.arange(n, device=device).unsqueeze(1)
            cand = (abs_p - win + 1).clamp(min=w_lo) + torch.arange(win, device=device)
            win_cols = torch.where(cand > abs_p, -1, cand - w_lo).unsqueeze(0)
            win_global = self.attn.win_cols_to_global(win_cols, ws_pool)

        if not ratio:
            return win_global, None
        _shared = self.indexer is None and _SHARED_CMP
        # Only the compressor/indexer read the matched tail page's slot; resolving it costs a
        # host sync (.item()), so do it after the ratio-0 early-out.
        _stw = getattr(get_global_ctx().batch, "spec_tail_ws", None)
        tail_ws = (
            _stw if _stw is not None else (
                (self.attn.window_slots_of(ti, start_pos - 1, start_pos).reshape(())
                 if _NOSYNC_SLOTS else int(self.attn.window_slots_of(ti, start_pos - 1, start_pos).item()))
                if start_pos > 0 else None
            )
        )
        if start_pos == 0:
            self.reset()  # re-seed the compressor/indexer carry from scratch
            blocks = (
                self.indexer(x_seg, qr_seg, 0, 0, slots, ti) if self.indexer is not None
                else None if _shared else get_compress_topk_idxs(ratio, 1, n, 0, 0).to(device)
            )
            self.compressor(x_seg, 0, slots, ti=ti)
        else:
            _spec = getattr(get_global_ctx().batch, "spec_verify", False)
            blocks = (
                self.indexer.extend(x_seg, qr_seg, start_pos, 0, slots, tail_ws, ti,
                                    spec=_spec)
                if self.indexer is not None
                else None if _shared else self._compress_topk_extend(n, start_pos, end, 0, device, 1)
            )
            if _spec:
                self.compressor.spec_advance(x_seg, start_pos, ti)
            else:
                self.compressor(x_seg, start_pos, slots, tail_window_slot=tail_ws, ti=ti)
        if _shared:
            # ratio-128: every query sees blocks [0, (p+1)//ratio) of the same list -> one shared
            # global-slot list + per-query live count (no [n, n_blocks] int64 transients)
            nb = end // ratio
            shared = self.attn.blocks_to_global(
                torch.arange(nb, device=device).view(1, 1, nb), ratio, ti=ti).view(-1).int()
            cnt = ((start_pos + torch.arange(n, device=device) + 1) // ratio).int()
            return win_global, ("shared", shared, cnt)
        return win_global, self.attn.blocks_to_global(blocks, ratio, ti=ti)

    def forward_ragged(self, x, segments, flat_positions):
        """Ragged batched prefill (cu_seqlens). ``x`` is [1, T, dim] -- the requests' NEW token
        streams concatenated (NO padding); ``segments`` is [(offset, n, table_idx, start_pos)]
        tiling [0, T); ``flat_positions`` [T] is each token's ABSOLUTE position, so a radix-hit
        request continues from its cached length.

        q / kv / o run FLAT over all T tokens (one GEMM, one rope each). The compressor + indexer
        carry is stateful (a bs=1 register + per-page ring), so they run PER REQUEST on the
        request's [offset, offset+n) slice with its own slot-map row -- cold segments re-seed the
        carry, radix-hit segments resume it from the matched tail page. The per-request top-k lists
        are padded to a UNIFORM width (window -> ``win`` cols at bs>1, the natural width at
        bs==1; compressed -> the batch max) with
        ``-1`` (the kernel masks ``-1`` to a no-op, so padding is bit-identical), then concatenated
        into ONE flat ``topk_idxs[1, T, win+max_c]`` for a SINGLE paged-attention launch
        (grid (T, head)). Each query gathers only its own request's slots, so requests are isolated.
        """
        win, ratio, rd = self.window_size, self.compress_ratio, self.rope_head_dim
        device = x.device
        _, T, _ = x.size()
        if len(segments) == 1:
            # single contiguous segment: a free slice view instead of a per-layer gather
            freqs = self.freqs_cis[segments[0][3]:segments[0][3] + T]
        else:
            freqs = self.freqs_cis.index_select(0, flat_positions)  # [T, rd//2] per-token rope

        # wq_a|wkv share input x: one block-fp8 GEMV (and one act_quant) instead of
        # two. Valid because q_lora_rank % 128 == 0, so wkv's 128x128 scale blocks
        # start on a fresh block row after the cat; guarded otherwise. Slices are
        # made contiguous (decode B is small) since downstream kernels index flat.
        _mw = getattr(self, "_qakv_w", None)
        if _mw is None and self.wq_a.weight.shape[0] % 128 == 0 and self.wq_a.kind == "fp8" and self.wkv.kind == "fp8":
            import torch as _t
            _w = _t.cat([self.wq_a.weight.view(_t.uint8), self.wkv.weight.view(_t.uint8)], 0)
            self._qakv_w = _w.view(self.wq_a.weight.dtype)
            self._qakv_s = _t.cat([self.wq_a.scale, self.wkv.scale], 0).contiguous()
            self._qakv_split = self.wq_a.weight.shape[0]
            _mw = self._qakv_w
        if _mw is not None:
            from freetoken.kernel.triton.dsv4.fp8_linear import block_fp8_linear
            _qakv = block_fp8_linear(x, _mw, self._qakv_s, None)
            _qa = _qakv[..., : self._qakv_split].contiguous()
            _kv_pre = _qakv[..., self._qakv_split :].contiguous()
        else:
            _qa = self.wq_a(x)
            _kv_pre = None
        qr = q = self.q_norm(_qa)
        q = self.wq_b(q).unflatten(-1, (self.n_heads, self.head_dim))
        q = rms_norm(q, None, self.eps)
        apply_rotary_emb(q[..., -rd:], freqs)

        kv = self.wkv(x)
        kv = self.kv_norm(kv)
        apply_rotary_emb(kv[..., -rd:], freqs)
        act_quant_fp8_inplace(kv[..., :-rd], 64)

        # Speculative verify: stash this layer's attention inputs so the carry fix can
        # re-advance the compressor/indexer over just the accepted prefix (spec.py).
        _st = getattr(get_global_ctx().batch, "spec_stash", None)
        if _st is not None and self.compress_ratio:
            _st[self.layer_id] = (x.clone(), qr.clone())

        # Per-request: persistent window write + window/compressed top-k + stateful compressor/
        # indexer carry. Each request owns disjoint window/cmp/idx slots + ring blocks.
        win_parts: list[torch.Tensor] = []
        cmp_parts: list[torch.Tensor | None] = []
        max_c = 0
        for off, n, ti, start_pos in segments:
            win_global, cmp_global = self._prefill_segment(
                x[:, off:off + n], qr[:, off:off + n], kv[0, off:off + n], ti, start_pos, n
            )
            win_parts.append(win_global)
            cmp_parts.append(cmp_global)
            if cmp_global is not None and not isinstance(cmp_global, tuple):
                max_c = max(max_c, cmp_global.shape[-1])

        # Uniform window width for the single flat launch. bs==1 keeps the segment's natural
        # width (a cold prompt shorter than the window yields < win columns) -- bit-identical
        # to the historical single-request path: pad columns are masked no-ops value-wise, but
        # they shift the kernel's window/compressed tile split, and that rounding-order change
        # is enough to flip a greedy near-tie. bs>1 pads every part to win (the historical
        # batched behavior).
        if ratio and len(segments) > 1:   # shared lists only serve a single segment: materialize
            for i, part in enumerate(cmp_parts):
                if isinstance(part, tuple):
                    _, sh, cn = part
                    nb = sh.numel()
                    cmp_parts[i] = torch.where(torch.arange(nb, device=sh.device)[None, :] < cn[:, None].long(),
                                               sh[None, :].long(), -1).unsqueeze(0)
                    max_c = max(max_c, nb)
        if ratio and isinstance(cmp_parts[0], tuple):
            _, shared, cnt = cmp_parts[0]
            o = self.attn.attend_shared(q, self.layer_id, win_parts[0].int(), shared, cnt,
                                        self.attn_sink, self.softmax_scale)
            apply_rotary_emb(o[..., -rd:], freqs, True)
            return self._wo(o, 1, T)
        n_window = win if len(segments) > 1 else win_parts[0].shape[-1]
        flat = []
        for i, (off, n, ti, _start) in enumerate(segments):
            wg = win_parts[i]
            if wg.shape[-1] < n_window:  # pad window picks to the uniform width
                wg = F.pad(wg, (0, n_window - wg.shape[-1]), value=-1)
            if ratio:
                cg = cmp_parts[i]
                if cg.shape[-1] < max_c:  # pad compressed picks to the uniform width
                    cg = F.pad(cg, (0, max_c - cg.shape[-1]), value=-1)
                flat.append(torch.cat([wg, cg], dim=-1))
            else:
                flat.append(wg)
        # [1, T, win(+max_c)]; a 1-element cat would copy the whole list per layer for nothing
        topk_idxs = (flat[0] if len(flat) == 1 else torch.cat(flat, dim=1)).int()

        o = self.attn.attend(
            q, self.layer_id, topk_idxs, n_window, self.attn_sink, self.softmax_scale,
            has_compression=bool(ratio),
        )
        apply_rotary_emb(o[..., -rd:], freqs, True)
        return self._wo(o, 1, T)

    def _compress_topk_extend(self, seqlen, start_pos, end, offset, device, bsz):
        # ratio-128 layers (no indexer): each new query at abs pos p attends compressed blocks
        # [0, (p+1)//ratio); pool index = block + offset, causal-masked to -1 beyond (p+1)//ratio.
        ratio = self.compress_ratio
        n_blocks = end // ratio
        blk = torch.arange(n_blocks, device=device).unsqueeze(0)         # [1, n_blocks]
        abs_p1 = (start_pos + torch.arange(seqlen, device=device) + 1).unsqueeze(1)  # [seqlen,1]
        valid = blk < (abs_p1 // ratio)
        out = torch.where(valid, blk + offset, -1)
        return out.unsqueeze(0).expand(bsz, -1, -1)

    def decode_step(
        self, x: torch.Tensor, pos: torch.Tensor, rows: torch.Tensor, cmp_stage_cap: int,
        wctx=None, dctx=None,
    ) -> torch.Tensor:
        """Batched single-token attention (EAGER/graph). Per row, builds the per-query top-k as
        GLOBAL pool slots -- the live 128-window ring slots (window-first) then the first
        ``n_cmp_stage`` compressed-block slots -- and hands them to the backend, whose kernel
        gathers KV straight from ``window_pool`` / ``cmp_pool`` (no staging slab).

        ``pos`` is ``[B]`` GPU int (per-row position). ``rows`` is ``[B]`` LOCAL row indices
        (arange B) into the backend's decode SNAPSHOT -- the captured graph reads
        the snapshot, never the live slot maps, so a concurrent next-batch allocate_paged cannot
        corrupt this replay (overlap safety). ``cmp_stage_cap`` is the max position any row reaches;
        the layer's fixed compressed top-k width is ``(cmp_stage_cap+1)//ratio`` (= max valid count
        over rows in eager; a static capture width = ``(max_seq-1+1)//ratio`` under graph). Picks
        past each row's valid count are masked to -1 (the kernel ignores -1), so co-tenant rows
        stay isolated."""
        B = x.size(0)
        win, ratio, rd = self.window_size, self.compress_ratio, self.rope_head_dim
        device = x.device
        n_cmp_stage = (cmp_stage_cap + 1) // ratio if ratio else 0
        # Layer-invariant window-ring slots are computed ONCE in the model decode loop and threaded
        # in via wctx (fall back to a fresh metadata compute when called standalone). freqs_t
        # stays per-layer (freqs_cis differs between compressed and non-compressed layers).
        if wctx is None:
            wctx = get_global_ctx().batch.attn_metadata.window_ctx(pos, rows)
        window_slots, prev_window_slots, window_slots_topk = wctx
        # freqs_cis is get_freqs_cis-memoized and SHARED across same-param layers, so the
        # id() key dedups the gather to one per rope variant per step (was one per layer).
        freqs_t = dctx_get(dctx, ("freqs", id(self.freqs_cis)),
                           lambda: self.freqs_cis.index_select(0, pos))

        # two-card EP (B=1): the q projection runs on the second card concurrently with the
        # kv / compressor / indexer branch below (ep_dense.EPDenseQ); q is picked up later.
        epq = getattr(self, "_ep_q", None)
        if epq is not None and B > epq.max_bs:
            epq = None
        _mw = None if epq is not None else getattr(self, "_qakv_w", None)
        if epq is None and _mw is None and self.wq_a.weight.shape[0] % 128 == 0 and self.wq_a.kind == "fp8" and self.wkv.kind == "fp8":
            import torch as _t
            _w = _t.cat([self.wq_a.weight.view(_t.uint8), self.wkv.weight.view(_t.uint8)], 0)
            self._qakv_w = _w.view(self.wq_a.weight.dtype)
            self._qakv_s = _t.cat([self.wq_a.scale, self.wkv.scale], 0).contiguous()
            self._qakv_split = self.wq_a.weight.shape[0]
            _mw = self._qakv_w
        if epq is not None:
            epq.launch(self.layer_id, x)
            qr = q = None
            kv = self.wkv(x)  # alone: same GEMV config as the fused wq_a|wkv rows
        else:
            if _mw is not None:
                from freetoken.kernel.triton.dsv4.fp8_linear import block_fp8_linear
                _qakv = block_fp8_linear(x, _mw, self._qakv_s, None)
                _qa = _qakv[..., : self._qakv_split].contiguous()
                _kv_pre = _qakv[..., self._qakv_split :].contiguous()
            else:
                _qa = self.wq_a(x)
                _kv_pre = None
            qr = q = self.q_norm(_qa)
            q = self.wq_b(q).unflatten(-1, (self.n_heads, self.head_dim))
            if _NORM_ROPE:   # rms_norm + per-row rope, one launch
                q = q_norm_rope(q, self.eps, freqs_t)
            else:
                q = rms_norm(q, None, self.eps)
                apply_rotary_emb_decode(q[..., -rd:], freqs_t)  # per-row position freqs

            kv = _kv_pre if _kv_pre is not None else self.wkv(x)
        wpool = self.attn.pool.window_pool[self.layer_id]
        fuse_kv = (_NORM_ROPE and kv.dtype == torch.bfloat16 and wpool.dtype == torch.bfloat16
                   and wpool.stride(-1) == 1 and kv.shape[-1] == wpool.shape[-1])
        if fuse_kv:   # kv_norm + rope + FP8 round-trip + window write, one launch (bit-identical)
            kv = kv_norm_rope_quant_store(kv, self.kv_norm.weight, self.kv_norm.eps, freqs_t,
                                          wpool, window_slots)
        else:
            kv = self.kv_norm(kv)
            apply_rotary_emb_decode(kv[..., -rd:], freqs_t)
            act_quant_fp8_inplace(kv[..., :-rd], 64)
        # Spec verify: stash this layer's attention input so the post-accept carry fix
        # can re-advance the compressors over just the accepted prefix (spec.py).
        _st = getattr(get_global_ctx().batch, "spec_stash", None)
        if _st is not None and self.compress_ratio:
            _st[self.layer_id] = x.detach().clone()
        # Persistent window write: each row's new token to its own window slot (swa_ratio=1).
        if not fuse_kv:
            self.attn.store_window(kv.view(B, -1), self.layer_id, window_slots)
        # prev_window_slots (compressor rolling carry, read from the previous token's window page)
        # and window_slots_topk (the GLOBAL window-ring slots, window-first column order) are
        # layer-invariant -> taken from the hoisted wctx instead of recomputed here.
        n_window = win
        cmp_counts = None
        if self.compress_ratio:
            self.compressor.decode_step(x, pos, prev_window_slots, window_slots, rows, dctx)
            valid = dctx_get(dctx, ("valid", ratio), lambda: (pos + 1) // ratio)
            if self.indexer is not None and _IDX_ASSEMBLE:
                # Raw top-k columns -> final [window | compressed] int32 list in ONE kernel
                # (bitwise the chain below: -1 past valid, snapshot gather, floor-div, cat, int).
                from freetoken.kernel.triton.dsv4.idx_assemble import idx_assemble

                picks = self.indexer.decode_step(
                    x, qr, pos, 0, prev_window_slots, window_slots, rows, n_cmp_stage, dctx,
                    q_raw_fn=epq.idx_q if epq is not None else None, raw_picks=True,
                )  # [B, 1, k] raw top-k columns
                topk_idxs = idx_assemble(window_slots_topk, picks, valid, self.attn.snapshot(),
                                         rows, ratio)
                compress_idxs = picks  # only its width is read below
            elif self.indexer is not None:
                # Indexer returns the selected BLOCK indices (offset 0) into [0, n_cmp_stage) or -1.
                blocks = self.indexer.decode_step(
                    x, qr, pos, 0, prev_window_slots, window_slots, rows, n_cmp_stage, dctx,
                    q_raw_fn=epq.idx_q if epq is not None else None,
                )  # [B, 1, k] block index or -1
                # Block index -> GLOBAL cmp row, off the decode SNAPSHOT (rows are local).
                compress_idxs = self.attn.blocks_to_global(blocks, ratio, rows=rows)
                topk_idxs = torch.cat([window_slots_topk, compress_idxs], dim=-1)
            else:
                # Non-indexer compressed layers: each valid block b is its own pick (column
                # order = block order; invalid -> -1). The whole index list depends only on
                # (pos, rows, ratio, staged width) -- identical for every such layer of the step,
                # so it is built once per step (integer ops: exact).
                def _nc_topk():
                    blk = torch.arange(n_cmp_stage, device=device)
                    col_valid = blk[None, :] < valid[:, None]  # [B, n_cmp_stage]
                    blks = torch.where(col_valid, blk[None, :], -1).view(B, 1, n_cmp_stage)
                    ci = self.attn.blocks_to_global(blks, ratio, rows=rows)
                    return ci, torch.cat([window_slots_topk, ci], dim=-1).int()
                compress_idxs, topk_idxs = dctx_get(dctx, ("nc_topk", ratio, n_cmp_stage), _nc_topk)
            # Valid compressed picks are a PREFIX of the list: the ratio-128 layer emits blocks in
            # order, and the indexer's top-k sorts its -1 picks (score -inf) last. So one count per
            # row bounds the kernel's loop -- read from DEVICE memory, so the captured graph's work
            # tracks the live position instead of the staged width.
            cmp_counts = dctx_get(
                dctx, ("cmpc", ratio, compress_idxs.shape[-1]),
                lambda: valid.clamp(max=compress_idxs.shape[-1]).to(torch.int32).view(B, 1))
        else:
            topk_idxs = dctx_get(dctx, ("w_topk",), lambda: window_slots_topk.int())
        if topk_idxs.dtype != torch.int32:
            topk_idxs = topk_idxs.int()
        if epq is not None:
            q = epq.q().unflatten(-1, (self.n_heads, self.head_dim))
            if _NORM_ROPE:   # rms_norm + per-row rope, one launch
                q = q_norm_rope(q, self.eps, freqs_t)
            else:
                q = rms_norm(q, None, self.eps)
                apply_rotary_emb_decode(q[..., -rd:], freqs_t)  # per-row position freqs

        o = self.attn.attend(
            q, self.layer_id, topk_idxs, n_window, self.attn_sink, self.softmax_scale,
            cmp_counts=cmp_counts, has_compression=bool(ratio),
            rope_inv=freqs_t if _NORM_ROPE else None,   # inverse rope in the merge epilogue
        )
        if not _NORM_ROPE:
            apply_rotary_emb_decode(o[..., -rd:], freqs_t, True)  # per-row position freqs (inverse)
        return self._wo(o, B, 1)
