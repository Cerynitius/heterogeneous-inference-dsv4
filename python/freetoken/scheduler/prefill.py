from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Tuple

import torch
from freetoken.core import Batch, Req
from freetoken.utils import align_down, div_ceil, init_logger

_logger_mm = init_logger(__name__)

# Prefill the LAST prompt token as its own 1-token step. A full prefix-cache hit matches
# input_ids[:input_len-1] and recomputes that token alone (M=1: split-K GEMV, 1-row cuBLAS
# router, 1-query extend), while a fresh prefill computed it inside the M=N chunk (tiled GEMM),
# a different reduction order -> ~1 ulp -> bf16 logit ties break differently and identical
# T=0 requests diverged (fresh vs hit, 2026-09-29). With the split both take the M=1 path over
# the same cached KV, so a repeat is bit-identical. Costs one decode-sized step per prefill.
# page_size 1 only: chunk ends must be page-aligned for the swa/compressor continuation.
_PREFILL_SPLIT_LAST = os.environ.get("FREETOKEN_PREFILL_SPLIT_LAST", "1") != "0"

# FREETOKEN_MM_CHUNKED_PREFILL=1 (default off): a multimodal prompt whose uncached extend can
# NEVER fit one prefill chunk (> --max-extend-tokens, or > the swa pool) is prefilled over
# several chunks like text; each chunk carries only the image rows whose placeholders fall in
# its [cached_len, cached_len + chunk) range. Prompts that can fit one chunk always run as ONE
# chunk (deferred to a later pass rather than split), flag on or off -- the pre-existing path.
# Off: such a prompt is answered with an error instead (it used to kill the scheduler).
_MM_CHUNKED_PREFILL = os.environ.get("FREETOKEN_MM_CHUNKED_PREFILL", "0") == "1"
# ErrorReplyMsg.code of a multimodal prompt the scheduler cannot prefill.
MM_REJECT_CODE = "multimodal_prompt_too_long"

# FREETOKEN_TEXT_PREFIX_TRACK=1 (default OFF until verified on GPU): a continuation chunk
# inherits its ChunkedReq's hybrid-radix track boundary (mamba_last_track_seqlen: the x64
# KDA/conv snapshot the chunk's forward froze into a ping-pong slot, attention/linear.py
# _build_track_metadata). ChunkedReqs are never committed (scheduler._process_last_data), so
# without the carry that snapshot is dropped -- and under split-last EVERY text prompt is an
# (N-1)-token ChunkedReq + a 1-token Req, so text prompts never insert a reusable node and never
# hit. With it the final Req's prefill commit (cache._cache_req_hybrid)
# donates the frozen snapshot at the boundary (or the deeper one its own forward froze: the
# ping-pong flip keeps the inherited slot intact until a newer track supersedes it).
_TEXT_PREFIX_TRACK = os.environ.get("FREETOKEN_TEXT_PREFIX_TRACK", "0") == "1"

# FREETOKEN_SCHED_ALIGNED_CHUNKS=1 (default on): a fresh prompt that does not fit the budget
# LEFT OVER by earlier requests of the same pass waits for the next pass instead of starting
# with a short first chunk. Every prompt is then chunked at cached_len + k * budget whatever
# else is queued, so its prefill (and logits) do not depend on what it was co-scheduled with
# (prefill chunking changes the logits). A prompt that fits the leftover whole still joins.
_ALIGNED_CHUNKS = os.environ.get("FREETOKEN_SCHED_ALIGNED_CHUNKS", "1") != "0"

# FREETOKEN_SCHED_SHORT_FIRST=1 (default on): while a long prompt is being prefilled chunk by
# chunk, a queued prompt that fits ONE chunk is prefilled first (the long prompt keeps its
# whole-chunk boundaries -- it resumes on the next pass -- and its remaining prompt stays
# reserved), so a short request no longer waits for someone else's whole long prefill.
_SHORT_FIRST = os.environ.get("FREETOKEN_SCHED_SHORT_FIRST", "1") != "0"


def _hot(key: str, boot):
    """Boot value, overridden by the hot runtime.json key (runtime_conf.py) when set."""
    from freetoken.runtime_conf import runtime_overrides
    v = runtime_overrides().get(key)
    return boot if v is None else v

from .utils import PendingReq

if TYPE_CHECKING:
    from freetoken.kvcache import BaseCacheHandle
    from freetoken.message import UserMsg

    from .cache import CacheManager
    from .decode import DecodeManager
    from .table import TableManager

logger = init_logger(__name__)


def _maybe_pinned(t: torch.Tensor) -> torch.Tensor:
    """Pinning only buys the async H2D copy below; without a device it just raises."""
    return t.pin_memory() if torch.cuda.is_available() else t


def _mm_rows_before(pending_req: PendingReq, image_token_id: int | None) -> torch.Tensor | None:
    """``cum[p]`` = number of image-placeholder positions in ``prompt[:p]`` (len input_len + 1,
    CPU int64), built once per request and cached on it. ``mm_embeds`` rows are in placeholder
    order, so the rows of the extend ``[s, e)`` are exactly ``mm_embeds[cum[s]:cum[e]]``.
    Placeholders are the negative content-hash key ids when present (the same test the
    single-chunk prefix-hit trim below uses), else ``input_ids == image_token_id`` (offline
    path). None if they cannot be located."""
    cum = pending_req.mm_rows_before
    if cum is None:
        if pending_req.cache_key_ids is not None:
            mask = pending_req.cache_key_ids[: pending_req.input_len] < 0
        elif image_token_id is not None:
            mask = pending_req.input_ids == image_token_id
        else:
            return None
        cum = torch.zeros(pending_req.input_len + 1, dtype=torch.int64)
        torch.cumsum(mask.to(torch.int64), dim=0, out=cum[1:])
        pending_req.mm_rows_before = cum
    return cum


class ChunkedReq(Req):
    def _alloc_ids_buf(self) -> None:
        pass  # never sampled; keep input_ids a view of the pending prompt

    def append_host(self, next_token: torch.Tensor) -> None:
        raise NotImplementedError("ChunkedReq should not be sampled")

    @property
    def can_decode(self) -> bool:
        return False  # avoid being added to decode manager


@dataclass
class PrefillAdder:
    token_budget: int
    reserved_size: int
    cache_manager: CacheManager
    table_manager: TableManager
    # SWA-pool tokens charged to reqs admitted so far this pass. Mirrors reserved_size: swa is
    # allocated only in allocate_paged (after the pass), so swa_available_size does not decrement
    # across the admission loop -- without this, successive admits all see the full pool.
    reserved_swa: int = 0
    # Multimodal: placeholder id for locating image rows when a request has no content-hash
    # keys (offline path), and the (uid, error, code) of requests refused this pass (the
    # scheduler replies; refusal happens before any budget/pool mutation of this pass).
    image_token_id: int | None = None
    rejected: List[Tuple[int, str, str]] = field(default_factory=list)
    # Future (output) tokens reserved per request: min(max_tokens, reserve_cap); 0 = all of
    # max_tokens (see DecodeManager.reserve_cap).
    reserve_cap: int = 0
    # A continuation chunk whose pages the pool cannot back right now waits (the scheduler
    # retracts younger decodes for it, or decodes run until one is retracted) instead of
    # failing in allocate_paged. Only while decodes run: with nothing else holding pages
    # its admission reservation guarantees the fit.
    defer_short_continuation: bool = False

    def _out_reserve(self, output_len: int) -> int:
        return min(output_len, self.reserve_cap) if self.reserve_cap > 0 else output_len

    def __post_init__(self) -> None:
        # The whole pass's budget = the largest chunk any request can ever be given (the head
        # of the queue on a pass no continuation shares).
        self.full_budget = self.token_budget

    def _mm_fits_one_chunk(self, cached_len: int, remain_len: int) -> bool:
        """Could this multimodal extend ``[cached_len, cached_len + remain_len)`` ever run as
        ONE chunk -- on a pass where it gets the full token budget and the whole swa pool?"""
        if remain_len > self.full_budget:
            return False
        cm = self.cache_manager
        cap = getattr(getattr(cm, "swa_pool", None), "swa_num_tokens", None) if cm.swa_paged else None
        if cap is not None:
            # _add_one_req's swa cap with swa_budget = the whole pool (slot 0 = sentinel)
            ps = cm.page_size
            need = (div_ceil(cached_len + remain_len, ps) - div_ceil(cached_len, ps)) * ps
            if need > ((cap - 1) // ps) * ps:
                return False
        return True

    def _reject_mm(self, pending_req: PendingReq, error: str) -> None:
        """Refuse a FRESH multimodal admission this pass can never seat. Returns None like any
        non-admission; schedule_next_batch drops the request and the scheduler replies."""
        assert pending_req.chunked_req is None  # a continuation owns live chunk state
        self.rejected.append((pending_req.uid, error, MM_REJECT_CODE))
        return None

    def _try_allocate_one(self, req: PendingReq):
        if self.table_manager.available_size == 0:
            return None

        # host L2 (dsv4_l2.py): bring a RAM-resident snapshot of this prefix back first
        if getattr(self.cache_manager, "l2", None) is not None and getattr(req, "chunked_req", None) is None:
            try:
                self.cache_manager.l2_restore(req)
            except Exception as e:  # noqa: BLE001 -- never take the server down for the L2
                import logging as _lg
                _lg.getLogger(__name__).error(f"L2 restore failed, disabling L2: {type(e).__name__}: {e}")
                self.cache_manager.prefix_cache.on_evict_leaf = None
                self.cache_manager.l2 = None
        mr = self.cache_manager.match_req(req)
        handle = mr.cuda_handle
        cached_len = handle.cached_len
        # TODO: better estimate policy
        extend_len = req.input_len - cached_len
        estimated_len = extend_len + self._out_reserve(req.output_len)

        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            return None
        self.cache_manager.lock(handle)
        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            return self.cache_manager.unlock(handle)

        # Second currency (hybrid GDN): reserve 1 live + 2 ping-pong state slots; evict tree
        # snapshots if the pool is short, fail admission if still short (mirrors the KV gate).
        if self.cache_manager.is_hybrid:
            pool = self.cache_manager.linear_state_pool
            if pool.num_free_slots < 3:
                self.cache_manager.ensure_mamba_slots(3)
            if pool.num_free_slots < 3:
                return self.cache_manager.unlock(handle)

        # Third currency (SWA): refuse admission unless the swa pool can seat this request's first
        # chunk / one window (the per-chunk charge is in _add_one_req; the reclaim -- radix
        # evict_swa -- happens in allocate_paged, so no ensure here; swa_available_size already
        # folds the evictable tree). For naive (no tree) this can only refuse, which is correct.
        if self.cache_manager.swa_paged:
            ps = self.cache_manager.page_size
            # swa is charged per WHOLE page (allocate_paged -> alloc_swa), so the seat check is
            # in page units too; identical at page_size==1.
            need_swa = div_ceil(
                min(max(extend_len, 1), self.cache_manager.sliding_window_size) + 1, ps
            ) * ps
            if self.cache_manager.swa_available_size - self.reserved_swa < need_swa:
                return self.cache_manager.unlock(handle)

        table_idx = self.table_manager.allocate()
        if cached_len > 0:  # NOTE: set the cached part
            device_ids = self.table_manager.token_pool[table_idx][:cached_len]
            device_ids.copy_(_maybe_pinned(req.input_ids[:cached_len]), non_blocking=True)
            # Write the matched indices into the TAIL of the page_entry: a cache may return
            # fewer matched indices than cached_len, in which case only the trailing n slots are
            # known-live. Today both the generic radix and the SWA radix match a prefix whose
            # full-loc row is entirely live (n == cached_len), so the tail IS the whole prefix.
            # (DSV4 reads this table too: its pool's full_loc_map is attached to it.)
            matched = handle.get_matched_indices()
            n = int(matched.numel())
            self.table_manager.page_table[table_idx][cached_len - n : cached_len].copy_(matched)

        linear_slot_idx = ping_pong = None
        if self.cache_manager.is_hybrid:
            pool = self.cache_manager.linear_state_pool
            linear_slot_idx = pool.alloc(1)[0]
            if getattr(self.cache_manager, "gdn_track_snapshots", True):
                ping_pong = tuple(pool.alloc(2))

        return handle, table_idx, linear_slot_idx, ping_pong, mr.mamba_value

    def _add_one_req(
        self,
        pending_req: PendingReq,
        cache_handle: BaseCacheHandle,
        table_idx: int,
        cached_len: int,
        linear_slot_idx: int | None = None,
        ping_pong: tuple | None = None,
        next_track_idx: int = 0,
        restore_src: int | None = None,
        swa_evicted_seqlen: int = 0,
        last_track_seqlen: int | None = None,
    ) -> Req | None:
        remain_len = pending_req.input_len - cached_len
        chunk_size = min(self.token_budget, remain_len)
        if (_PREFILL_SPLIT_LAST and chunk_size == remain_len > 1 and pending_req.mm_embeds is None
                and self.cache_manager.page_size == 1):
            chunk_size = remain_len - 1  # the last token runs alone next pass (see top of file)
        if self.cache_manager.swa_paged:
            # Cap this chunk by the swa the pool can back this pass. swa is allocated per token in
            # allocate_paged, and token_budget (max_extend_tokens, default 8192) won't chunk a
            # shorter prompt -- so this cap is what forces a prompt whose swa footprint exceeds the
            # pool to chunk. Credit the slots THIS request's own extend-free (in _prepare_batch,
            # which runs AFTER this sizing) will release this batch, else a continuation sees a
            # drained pool and stalls at chunk_size 0.
            cm = self.cache_manager
            window, ps = cm.sliding_window_size, cm.page_size
            floor = cache_handle.cached_len
            new_evicted = align_down(cached_len - window - ps, ps)
            self_reclaim = max(0, new_evicted - max(swa_evicted_seqlen, floor))
            swa_budget = cm.swa_available_size + self_reclaim - self.reserved_swa
            # swa is charged per WHOLE page: cap the chunk so its PAGE-SPAN cost fits the budget
            # (the extend [cached_len, cached_len+chunk) pulls div_ceil(end,ps)-div_ceil(start,ps)
            # fresh pages -- the partial head page was charged by the previous chunk), and reserve
            # that cost, not the raw token count. Degenerates to the token math at page_size==1.
            max_end = (div_ceil(cached_len, ps) + max(swa_budget, 0) // ps) * ps
            chunk_size = min(chunk_size, max(max_end - cached_len, 0))
            # A continuation resumes the compressor carry at its boundary, which must be
            # page-aligned; the token_budget leftover (unlike max_end) is not. Align the end
            # down when the chunk mints a continuation; no whole page -> retry next pass.
            # 0 <: a chunk the swa cap collapsed to 0 must NOT bail (undersized pool --
            # bailing would livelock; the floor tests pin the loud failure).
            if 0 < chunk_size < remain_len:
                aligned = align_down(cached_len + chunk_size, ps) - cached_len
                if aligned <= 0:
                    return None
                chunk_size = aligned
            # reserved below, after the multimodal gate (which must not mutate the pass)
            swa_charge = (div_ceil(cached_len + chunk_size, ps) - div_ceil(cached_len, ps)) * ps
        is_chunked = chunk_size < remain_len
        if _hot("sched_aligned_chunks", _ALIGNED_CHUNKS) and self.token_budget < min(self.full_budget, remain_len):
            return None  # the pass's LEFTOVER budget would cut it: whole chunk next pass instead
        if pending_req.chunked_req is not None and self.defer_short_continuation:
            ps = self.cache_manager.page_size
            need = (div_ceil(cached_len + chunk_size, ps) - div_ceil(cached_len, ps)) * ps
            if need > self.cache_manager.available_size:
                return None  # pool short: wait (scheduler._make_room_for_continuation)
        # Multimodal gate -- decided BEFORE any budget/pool/token_pool mutation of this pass, so
        # a deferral or refusal leaves the pass exactly as if this request were not queued
        # (try_add_one undoes _try_allocate_one's lock/table/GDN slots, as for a no-chunk retry).
        mm_split = False  # this request's prefill spans several chunks (chunked-mm path)
        if pending_req.mm_embeds is not None:
            if pending_req.chunked_req is not None:
                mm_split = True  # continuation: only the flag-on path ever splits a mm prompt
            elif is_chunked:
                if self._mm_fits_one_chunk(cached_len, remain_len):
                    # Only this pass's leftover budget / swa is short: wait for a pass that
                    # seats the prompt whole (the unchanged single-chunk path, flag on or off).
                    return None
                if not _MM_CHUNKED_PREFILL:
                    return self._reject_mm(
                        pending_req,
                        f"multimodal prompt needs {remain_len} uncached prefill tokens in one "
                        f"chunk but the prefill chunk limit is {self.full_budget} tokens "
                        f"(--max-extend-tokens / swa pool); send a smaller image or fewer "
                        f"images (FREETOKEN_GLM5_MAX_IMAGE_TOKENS), raise --max-extend-tokens, "
                        f"or enable FREETOKEN_MM_CHUNKED_PREFILL=1",
                    )
                cum = _mm_rows_before(pending_req, self.image_token_id)
                n_rows = int(pending_req.mm_embeds.shape[0])
                if cum is None or int(cum[-1]) != n_rows:
                    # (the forward's splice would assert on this; refuse the request instead)
                    return self._reject_mm(
                        pending_req,
                        "cannot split this multimodal prompt across prefill chunks: "
                        + ("image placeholders cannot be located" if cum is None else
                           f"{int(cum[-1])} image placeholders vs {n_rows} image embeddings"),
                    )
                mm_split = True
        if self.cache_manager.swa_paged:
            self.reserved_swa += swa_charge
        CLS = ChunkedReq if is_chunked else Req
        self.token_budget -= chunk_size
        self.reserved_size += remain_len + self._out_reserve(pending_req.output_len)
        # NOTE: update the tokens ids only; new pages will be allocated in the scheduler
        _slice = slice(cached_len, cached_len + chunk_size)
        device_ids = self.table_manager.token_pool[table_idx, _slice]
        device_ids.copy_(_maybe_pinned(pending_req.input_ids[_slice]), non_blocking=True)
        mm_embeds = pending_req.mm_embeds
        if mm_split:
            # This chunk scatters exactly the rows whose placeholders lie in
            # [cached_len, cached_len + chunk_size): rows before cum[cached_len] belong to the
            # prefix-cache hit or to earlier chunks (same count as the hit trim below), rows
            # from cum[end] on to later chunks; a span cut by a boundary splits between the two
            # chunks at the same row. Positions stay arange(cached_len, end) (1-D), so every
            # placeholder gets the same row at the same position as in a one-chunk prefill.
            # Never None for a chunk without images: an empty slice keeps the request marked
            # multimodal (cache.py's no-content-key guard) and the splice skips it.
            cum = pending_req.mm_rows_before
            mm_embeds = mm_embeds[int(cum[cached_len]) : int(cum[cached_len + chunk_size])]
        if mm_embeds is not None and pending_req.cache_key_ids is not None and (
            pending_req.chunked_req is None
        ):
            # mm-forensic: match depth for every media request (spans start where the
            # first negative key id sits, so depth vs span position is decodable).
            _span0 = int((pending_req.cache_key_ids < 0).nonzero()[0, 0]) if bool(
                (pending_req.cache_key_ids < 0).any()) else -1
            _logger_mm.info(
                "mm-match uid=%d cached_len=%d prompt_len=%d span0=%d",
                pending_req.uid, cached_len, len(pending_req.input_ids), _span0,
            )
        if (not mm_split and mm_embeds is not None and cached_len > 0
                and pending_req.cache_key_ids is not None):
            # A cache hit may cover leading images (whole or partial spans -- content
            # hashes guarantee a matched span is the SAME image). The extend only
            # scatters the remaining placeholder rows; drop the covered ones. Image
            # positions are exactly the negative key ids.
            covered = int((pending_req.cache_key_ids[:cached_len] < 0).sum())
            mm_embeds = mm_embeds[covered:]
        req = CLS(
            input_ids=pending_req.input_ids[: cached_len + chunk_size],
            table_idx=table_idx,
            cached_len=cached_len,
            output_len=pending_req.output_len,
            uid=pending_req.uid,
            cache_handle=cache_handle,
            sampling_params=pending_req.sampling_params,
            mm_embeds=mm_embeds,
        )
        # Hybrid GDN per-request state slots (None for non-hybrid). On a fresh admit these are
        # freshly allocated; on a chunked continuation they are inherited from the prior chunk.
        req.cache_key_ids = pending_req.cache_key_ids
        req.stop_prompt_len = pending_req.stop_prompt_len
        req.linear_slot_idx = linear_slot_idx
        req.mamba_ping_pong = ping_pong
        req.mamba_next_track_idx = next_track_idx
        req.mamba_restore_src = restore_src
        req.swa_evicted_seqlen = swa_evicted_seqlen  # carry the extend-free watermark across chunks
        # pending (uncommitted) track snapshot of the prior chunk; None on a fresh admit
        req.mamba_last_track_seqlen = last_track_seqlen
        return req

    def try_add_one(self, pending_req: PendingReq) -> Req | None:
        if self.token_budget <= 0:
            return None

        if chunked_req := pending_req.chunked_req:
            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=chunked_req.cache_handle,
                table_idx=chunked_req.table_idx,
                cached_len=chunked_req.cached_len,
                linear_slot_idx=chunked_req.linear_slot_idx,
                ping_pong=chunked_req.mamba_ping_pong,
                next_track_idx=chunked_req.mamba_next_track_idx,
                restore_src=None,  # continuation chunk already has live state
                swa_evicted_seqlen=chunked_req.swa_evicted_seqlen,  # extend-free watermark so far
                last_track_seqlen=(
                    chunked_req.mamba_last_track_seqlen if _TEXT_PREFIX_TRACK else None
                ),
            )

        if resource := self._try_allocate_one(pending_req):
            cache_handle, table_idx, linear_slot_idx, ping_pong, restore_src = resource
            req = self._add_one_req(
                pending_req=pending_req,
                cache_handle=cache_handle,
                table_idx=table_idx,
                cached_len=cache_handle.cached_len,
                linear_slot_idx=linear_slot_idx,
                ping_pong=ping_pong,
                next_track_idx=0,
                restore_src=restore_src,
            )
            if req is None:
                # no aligned chunk / multimodal deferral or refusal this pass: undo the
                # admission (a continuation keeps its resources -- they belong to the prior
                # chunk's Req)
                self.cache_manager.unlock(cache_handle)
                self.table_manager.free(table_idx)
                if linear_slot_idx is not None:
                    # ping_pong is None when the model has no track snapshots (glm5_next KDA)
                    self.cache_manager.linear_state_pool.free(
                        [linear_slot_idx, *(ping_pong or ())]
                    )
            return req

        return None


@dataclass
class PrefillManager:
    cache_manager: CacheManager
    table_manager: TableManager
    decode_manager: DecodeManager
    pending_list: List[PendingReq] = field(default_factory=list)
    # Locates image rows for a multimodal request without content-hash keys (offline path).
    image_token_id: int | None = None
    # (uid, error, code) of requests dropped from pending_list because they can never be
    # prefilled; the scheduler drains this after each schedule_next_batch and replies.
    rejected: List[Tuple[int, str, str]] = field(default_factory=list)
    reserve_cap: int = 0  # see PrefillAdder.reserve_cap

    def add_one_req(self, req: UserMsg) -> None:
        self.pending_list.append(
            PendingReq(
                req.uid,
                req.input_ids,
                req.sampling_params,
                mm_embeds=req.mm_embeds,
                cache_key_ids=getattr(req, "cache_key_ids", None),
            )
        )

    def schedule_next_batch(self, prefill_budget: int) -> Batch | None:
        if len(self.pending_list) == 0:
            return None

        # estimated offset due to in-flight decode
        adder = PrefillAdder(
            token_budget=prefill_budget,
            reserved_size=self.decode_manager.inflight_tokens,
            cache_manager=self.cache_manager,
            table_manager=self.table_manager,
            image_token_id=self.image_token_id,
            reserve_cap=self.reserve_cap,
            defer_short_continuation=self.decode_manager.runnable,
        )
        order = self.pending_list
        head = order[0]
        n_short = 0
        if (_hot("sched_short_first", _SHORT_FIRST) and head.chunked_req is not None and len(order) > 1
                and head.overtaken_at != head.chunked_req.cached_len):
            def _uncached(p):
                # a prompt whose prefix is cached only prefills its tail: judge THAT length
                # (a 1M-token follow-up with a 3K new tail is short; it used to wait for a whole
                # long prefill, 105 s on 11-18)
                if p.input_len <= prefill_budget:
                    return p.input_len
                return p.input_len - self.cache_manager.match_req(p).cuda_handle.cached_len

            shorts = [p for p in order[1:] if p.chunked_req is None and p.mm_embeds is None
                      and _uncached(p) <= prefill_budget]
            if shorts:
                # at most one overtaking pass per chunk of the long prompt: it always advances
                head.overtaken_at = head.chunked_req.cached_len
                ids = {id(p) for p in shorts}
                order = shorts + [p for p in order if id(p) not in ids]
                n_short = len(shorts)
                # the overtaken prompt's remaining extend (+ output reserve) stays reserved
                adder.reserved_size += (head.input_len - head.chunked_req.cached_len
                                        + adder._out_reserve(head.output_len))
        reqs: List[Req] = []
        chunked_list: List[PendingReq] = []
        prompt_admissions: List[Tuple[int, int, int]] = []
        # Snapshot here, before the forward's complete_one() advances cached_len: the tokens
        # forwarded this batch (extend_len) and the prefix-cache hit. SGLang counts the hit
        # once at admission, so continuation chunks (already-chunked reqs) contribute 0.
        log_new_tokens = 0
        log_cached_tokens = 0
        done: set = set()  # ids of the entries (leading `order`) admitted or refused this pass
        for i, pending_req in enumerate(order):
            is_continuation = pending_req.chunked_req is not None
            if req := adder.try_add_one(pending_req):
                done.add(id(pending_req))
                pending_req.chunked_req = None
                if isinstance(req, ChunkedReq):
                    pending_req.chunked_req = req
                    chunked_list.append(pending_req)
                reqs.append(req)
                if not is_continuation and not pending_req.resumed:
                    # Record the COMPLETE prompt length and the prefix-cache hit on the
                    # first chunk. The scheduler publishes them only after _prepare_batch
                    # succeeds; continuation chunks (and a request resumed after a
                    # retraction) must never publish them again.
                    prompt_admissions.append(
                        (req.uid, pending_req.input_len, req.cache_handle.cached_len)
                    )
                log_new_tokens += req.extend_len
                if not is_continuation:
                    log_cached_tokens += req.cache_handle.cached_len
            elif adder.rejected and adder.rejected[-1][0] == pending_req.uid:
                # Refused before touching this pass's budget/pools: drop it from the queue
                # (the scheduler answers it with an error) and keep admitting behind it.
                done.add(id(pending_req))
            elif i < n_short:
                continue  # an overtaking short that does not fit must not block the long prompt
            else:
                break  # We cannot add more requests
        if adder.rejected:
            self.rejected.extend(adder.rejected)
        remaining = [p for p in self.pending_list if id(p) not in done]
        if len(reqs) == 0:
            self.pending_list = remaining
            return None
        self.pending_list = chunked_list + remaining
        batch = Batch(reqs=reqs, phase="prefill")
        batch.log_new_tokens = log_new_tokens
        batch.log_cached_tokens = log_cached_tokens
        batch.prompt_admissions = prompt_admissions
        return batch

    def abort_req(self, uid: int) -> Req | None:
        for i, req in enumerate(self.pending_list):
            if req.uid == uid:
                self.pending_list.pop(i)
                return req.chunked_req
        return None

    @property
    def runnable(self) -> bool:
        return len(self.pending_list) > 0
