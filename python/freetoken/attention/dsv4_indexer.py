"""Lightning-Indexer addressing and block selection, mixed into the sparse-attention backend.

The indexer scores every compressed block a query may attend and returns the ``index_topk``
best BLOCK indices; the attention layer then maps those to global compressed rows. Both the
gather (which paged row holds block ``b``) and the selection (which blocks are causally live,
and what a pick past the live count means) are addressing, so they live here -- the module
keeps ``wq_b`` / ``weights_proj`` and hands the projections in, mirroring sglang's
``C4IndexerBackendMixin.forward_c4_indexer(..., c4_indexer=self, ...)``.

The ``-1`` a selection emits for a pick past the live count is a gather-only sentinel: the
sparse-attention kernel masks it, and it never reaches a store.
"""

from __future__ import annotations

import torch

# 1808 = transposed FP4 kernel (keys as MMA rows: in-thread head sum, 1.15x on PRO 4000), 808 = v1
_TOPK_CUDA = __import__("os").environ.get("FREETOKEN_DSV4_TOPK_CUDA", "0") == "1"   # 1.9x vs torch.topk
_FP4_QS = 1808 if __import__("os").environ.get("FREETOKEN_DSV4_IDX_FP4T", "0") == "1" else 808


class IndexerBackendMixin:
    _dump_count: dict = {}

    def indexer_keys(
        self, ti: int, n_blocks: int, ratio: int, layer_id: int, bsz: int
    ) -> torch.Tensor:
        """Blocks ``[0, n_blocks)`` of this request's indexer keys as a dense
        ``[bsz, n_blocks, index_head_dim]`` slab, in ascending block order.

        Byte-exact ``index_select`` of the bf16 pool: block ``b`` lives at the arithmetic row
        ``full_loc(b * ratio) // ratio``, read off the request's LIVE full locs (prefill/extend
        only -- decode reads the snapshot inside the fused kernel instead).
        """
        block_starts = torch.arange(0, n_blocks * ratio, ratio, device=self.device)
        rows = self.compress_rows_of(ti, block_starts, ratio)
        pool = self.compress_pool(layer_id, "idx")
        if not isinstance(pool, torch.Tensor):   # Fp4Rows: gather the packed rows (68 B each)
            from freetoken.kernel.triton.dsv4.fp4_rows import Fp4Keys
            c, e = pool.gather(rows)
            return Fp4Keys(c.unsqueeze(0).expand(bsz, -1, -1), e.unsqueeze(0).expand(bsz, -1, -1))
        return pool.index_select(0, rows).unsqueeze(0).expand(
            bsz, -1, -1
        )

    def indexer_prefill_logits(
        self, q: torch.Tensor, keys: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """Head-reduced scores ``[bsz, seqlen, n_blocks]`` over a dense key slab."""
        from freetoken.kernel.triton.dsv4.indexer import indexer_logits

        return indexer_logits(q, keys, weights)

    def indexer_decode_scores(
        self, q: torch.Tensor, weights: torch.Tensor, valid: torch.Tensor, n_stage: int,
        ratio: int, layer_id: int,
    ) -> torch.Tensor:
        """Head-reduced scores ``[B, n_stage]`` for a decode step, gathering each block's key off
        the decode SNAPSHOT and bounding the work by the live block count read from device
        memory (so a captured graph tracks the position, not the staged width)."""
        from freetoken.kernel.triton.dsv4.indexer import indexer_decode_logits

        return indexer_decode_logits(
            q, weights, self.compress_pool(layer_id, "idx"), self.snapshot(),
            valid, n_stage, ratio,
        )

    def indexer_select_prefill(
        self, scores: torch.Tensor, *, start_pos: int, seqlen: int, ratio: int, topk: int,
        offset: int,
    ) -> torch.Tensor:
        """Causal top-k over compressed blocks for a prefill/extend range.

        Query ``s`` sits at absolute position ``p = start_pos + s`` and may attend blocks
        ``[0, (p + 1) // ratio)``. Blocks past that score ``-inf`` so they lose the top-k; any
        that still get picked (a short live count) come back as ``-1``.
        """
        device = scores.device
        n_blocks = scores.shape[-1]
        live = (start_pos + torch.arange(1, seqlen + 1, device=device)) // ratio
        blk = torch.arange(n_blocks, device=device)
        # Row-sliced causal mask + top-k: the old path materialized a [seqlen, n_blocks]
        # int64 arange (2 GiB at an 8K chunk against a 131K-deep prefix) plus two fp32
        # copies of the same shape, which OOMed deep prefills. Rows are independent, so
        # slice; peak transient is one [slice, n_blocks] fp32.
        SLICE = 1024
        outs = []
        for s0 in range(0, seqlen, SLICE):
            sl = slice(s0, min(s0 + SLICE, seqlen))
            live_s = live[sl].unsqueeze(1)
            sc = scores[..., sl, :].masked_fill(blk >= live_s, float("-inf"))
            picks = sc.topk(min(topk, n_blocks), dim=-1)[1]
            outs.append(torch.where(picks >= live_s, -1, picks + offset))
        return torch.cat(outs, dim=-2)

    def indexer_prefill_select(
        self, q: torch.Tensor, keys: torch.Tensor, weights: torch.Tensor, *, start_pos: int,
        seqlen: int, ratio: int, topk: int, offset: int,
    ) -> torch.Tensor:
        """``indexer_select_prefill(indexer_prefill_logits(q, keys, weights), ...)`` without the
        full ``[seqlen, n_blocks]`` fp32 score matrix (4.3 GB for a 4096 chunk at 1M context):
        scores are computed per 1024-row slice (each (query, block) score is independent) and
        masked in place, then the same per-slice top-k runs. Bit-identical picks."""
        from freetoken.kernel.triton.dsv4.indexer import indexer_logits

        import os

        device = q.device
        n_blocks = keys.shape[1]
        packed = None
        if not isinstance(keys, torch.Tensor):   # Fp4Keys from the FP4 indexer pool
            if os.environ.get("FREETOKEN_DSV4_IDX_KERNEL", "ref") == "fp4" and q.shape[0] == 1:
                packed = (keys.codes[0], keys.exps[0])
            else:
                keys = keys.dequant()
        live = (start_pos + torch.arange(1, seqlen + 1, device=device)) // ratio
        blk = torch.arange(n_blocks, device=device)
        SLICE = 1024
        # INT8 tensor-core scores (kernel/triton/dsv4/indexer_i8.py): exact integer block dots,
        # same fp32 scores bit for bit, causal mask written in-kernel. B == 1 only.
        dump_dir = os.environ.get("FREETOKEN_DEBUG_IDX_DUMP")
        if dump_dir and str(start_pos) in os.environ.get("FREETOKEN_DEBUG_IDX_DUMP_POS", "").split(","):
            # debug: real indexer inputs for offline studies (first 3 ratio-4 layers per position)
            n = IndexerBackendMixin._dump_count.get(start_pos, 0)
            if n < 3:
                IndexerBackendMixin._dump_count[start_pos] = n + 1
                os.makedirs(dump_dir, exist_ok=True)
                torch.save({"q": q[0].cpu(), "keys": keys[0].cpu(), "w": weights[0].cpu(),
                            "start_pos": start_pos, "ratio": ratio, "topk": topk},
                           os.path.join(dump_dir, f"idx_{start_pos}_{n}.pt"))
        # kernel: "ref" (per-query bf16 kernel + masked_fill), "ms" (bf16, BLOCK_S queries per
        # program, key tile reused, in-kernel mask: bit-identical), "i8" (INT8 block dots)
        kern = os.environ.get("FREETOKEN_DSV4_IDX_KERNEL", "i8" if os.environ.get("FREETOKEN_DSV4_IDX_I8", "0") == "1" else "ref")
        if q.shape[0] != 1:
            kern = "ref"
        i8 = kern == "i8"
        ms = kern == "ms"
        fp4 = kern == "fp4"
        if fp4:   # native FP4 tensor cores (kernel/cuda/dsv4_idx_fp4.cu)
            from freetoken.kernel.cuda.dsv4_idx_fp4 import idx_logits_fp4, pack_fp4

            qp4, qe4 = pack_fp4(q[0])
            kp4, ke4 = packed if packed is not None else pack_fp4(keys[0])
            w4 = weights[0] if weights.dtype == torch.bfloat16 else weights[0].to(torch.bfloat16)
        if ms:
            from freetoken.kernel.triton.dsv4.indexer import indexer_logits_ms
        if i8:
            from freetoken.kernel.triton.dsv4.indexer_i8 import indexer_logits_i8_ms, to_i8

            qc, qe = to_i8(q[0])
            kc, ke = to_i8(keys[0])
        if fp4 and _TOPK_CUDA:
            # query split across the two EP cards (attention/dsv4_idx_split.py; off unless
            # FREETOKEN_DSV4_IDX_SPLIT > 0): same kernels and arguments per query -> bitwise picks
            from freetoken.attention import dsv4_idx_split as _split
            if _split.wanted(n_blocks, seqlen):
                picks = _split.split_select(qp4, qe4, kp4, ke4, w4, start_pos=start_pos, seqlen=seqlen,
                                            ratio=ratio, topk=topk, offset=offset, qs=_FP4_QS,
                                            local_slice=SLICE)
                if picks is not None:
                    return picks.unsqueeze(0)
        outs = []
        for s0 in range(0, seqlen, SLICE):
            sl = slice(s0, min(s0 + SLICE, seqlen))
            live_s = live[sl].unsqueeze(1)
            if fp4:
                sc = idx_logits_fp4(qp4[sl].contiguous(), qe4[sl].contiguous(), kp4, ke4, w4[sl],
                                    start_pos + s0, ratio, _FP4_QS, 4096,
                                    out_dtype=torch.bfloat16).unsqueeze(0)   # bf16 scores, like decode
            elif ms:
                sc = indexer_logits_ms(q[0, sl], keys[0], weights[0, sl], start_pos + s0, ratio,
                                       32, 4).unsqueeze(0)
            elif i8:
                sc = indexer_logits_i8_ms(qc[sl], qe[sl], kc, ke, weights[0, sl], start_pos + s0,
                                          ratio, 32, 128, 4).unsqueeze(0)
            else:
                sc = indexer_logits(q[:, sl], keys, weights[:, sl])
                sc.masked_fill_(blk >= live_s, float("-inf"))
            if fp4 and _TOPK_CUDA:
                # radix-select CUDA top-k (deterministic, column order) emitting the final picks
                from freetoken.kernel.cuda.dsv4_topk import topk_picks
                outs.append(topk_picks(sc[0], topk, start_pos + s0, ratio, offset).unsqueeze(0))
                del sc
                continue
            picks = sc.topk(min(topk, n_blocks), dim=-1)[1]
            del sc
            outs.append(torch.where(picks >= live_s, -1, picks + offset))
        return torch.cat(outs, dim=-2)

    def indexer_select_decode(
        self, scores: torch.Tensor, *, valid: torch.Tensor, topk: int, offset: int
    ) -> torch.Tensor:
        """Top-k over a decode step's staged scores. Columns past each row's live count already
        score ``-inf`` (the scoring kernel writes them), so they sort last -- which is what lets
        the attention layer bound the sparse kernel with a single per-row count."""
        n_stage = scores.shape[-1]
        picks = scores.topk(min(topk, n_stage), dim=-1)[1]
        return torch.where(picks >= valid[:, None, None], -1, picks + offset)


__all__ = ["IndexerBackendMixin"]
