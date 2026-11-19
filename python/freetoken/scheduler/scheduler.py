from __future__ import annotations

from freetoken.runtime_conf import runtime_overrides
import dataclasses
import os
from typing import TYPE_CHECKING, List, NamedTuple, NoReturn, Set, Tuple, TypeAlias

import torch
from freetoken.attention.linear import build_fla_metadata
from freetoken.core import Batch, Req
from freetoken.env import ENV
from freetoken.gpu_select import gpu_identity
from freetoken.message import (
    AbortBackendMsg,
    BaseBackendMsg,
    BatchBackendMsg,
    CacheRebuildBackendMsg,
    CacheRebuildResultMsg,
    DetokenizeMsg,
    ErrorReplyMsg,
    ExitMsg,
    PromptAdmittedMsg,
    UserMsg,
)
from freetoken.utils import (
    init_logger,
    load_eos_token_ids,
    load_tokenizer,
    load_toolcall_anchor_id,
)

from .cache import CacheManager
from .config import SchedulerConfig
from .decode import DecodeManager
from .io import SchedulerIOMixin
from .prefill import ChunkedReq, PrefillManager
from .status import SchedulerStatusReporter
from .table import TableManager
from .utils import PendingReq

if TYPE_CHECKING:
    from freetoken.engine import BatchSamplingArgs, ForwardOutput


logger = init_logger(__name__)

Indice2D: TypeAlias = Tuple[torch.Tensor, torch.Tensor]

# ---- context pool: admission, retraction, prefill/decode time share ----
# FREETOKEN_SCHED_RESERVE_TOKENS (default 16384): future tokens a request reserves in the KV
# pool, min(max_tokens, this). The old rule reserved all of max_tokens, so two requests sent
# with max_tokens = the context length could never run together. 0 restores it. When the pool
# really runs short (a decode step or the next prefill chunk cannot get its pages), the
# YOUNGEST request holding pages is retracted: its KV prefix is committed to the prefix cache,
# its pages freed, and it is requeued under the same uid with prompt = prompt + output so far
# and max_tokens = what is left -- its client stream pauses, then continues.
_RESERVE_TOKENS = int(os.environ.get("FREETOKEN_SCHED_RESERVE_TOKENS", "16384"))
# FREETOKEN_SCHED_DECODE_SHARE (default 0.5): fraction of GPU time running decodes get while
# another request's prefill is pending. After a prefill chunk of t seconds, decode steps run
# for t * share / (1 - share) seconds before the next chunk. 0 = strict prefill priority (old:
# a long prompt stalled every running stream until its whole prefill was done).
_DECODE_SHARE = min(max(float(os.environ.get("FREETOKEN_SCHED_DECODE_SHARE", "0.5")), 0.0), 0.95)


def _decode_share() -> float:
    """_DECODE_SHARE, overridden by the hot runtime.json `sched_decode_share` when set."""
    v = runtime_overrides().get("sched_decode_share")
    return _DECODE_SHARE if v is None else float(v)


def _reserve_tokens() -> int:
    v = runtime_overrides().get("sched_reserve_tokens")
    return _RESERVE_TOKENS if v is None else v


def _gib(n_bytes: int) -> str:
    return f"{n_bytes / (1 << 30):.2f} GiB"


# For overlap scheduling, we also need to cache some other data to avoid IMA
class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs
    input_tuple: Indice2D  # (token_mapping, positions)
    write_tuple: Indice2D  # (req_mapping, seq_lens or -1)


ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"


class Scheduler(SchedulerIOMixin):
    @property
    def prefill_budget(self) -> int:
        """Prefill chunk tokens: the boot budget, lowered by the hot `prefill_chunk_tokens`
        (runtime_conf.py) when set."""
        hot = runtime_overrides().get("prefill_chunk_tokens")
        return self._prefill_budget_boot if hot is None else min(self._prefill_budget_boot, hot)

    def __init__(self, config: SchedulerConfig):
        from freetoken.engine import Engine

        self.engine = Engine(config)

        # use another stream to overlap metadata processing with computation
        self.device = self.engine.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        torch.cuda.set_stream(self.stream)
        # sent on the readiness ack for /v1/stats gpus; a list so TP can add one entry per rank
        self.gpus = [gpu_identity(self.device.index)] if self.device.type == "cuda" else []

        # initialize other managers
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        # ONE cache manager for every model (ShadowRadix layering): the shared page table is the
        # virtual full-token coordinate; model-specific tiers ride the plug-ins -- DSV4's
        # window/cmp/idx shadows via swa_pool, Gemma's swa via swa_pool, GDN state via
        # linear_state_pool. No model supplies its own manager.
        self.cache_manager = CacheManager(
            self.engine.num_pages, config.page_size, self.engine.page_table, config.cache_type,
            linear_state_pool=self.engine.linear_state_pool,
            swa_pool=self.engine.kv_cache,
            sliding_window_size=next(
                (g.sliding_window for g in config.model_config.kv_cache_group_specs() if g.is_swa),
                None,
            ) or getattr(self.engine.kv_cache, "sliding_window_size", None),
        )
        # KDA (glm5_next) has no track-snapshot writer yet: without this gate its frozen
        # ping-pong donates would put UNWRITTEN all-zero states into the radix tree and
        # every prefix hit would restore zero recurrent state (full-attention layers then
        # carry the prefix alone -- usually survivable, intermittently wrong).
        self.cache_manager.gdn_track_snapshots = getattr(
            config.model_config, "gdn_track_snapshots", True
        )
        if getattr(self.engine, "elastic", None) is not None:
            self.cache_manager.attach_elastic(self.engine.elastic)
        self.decode_manager = DecodeManager(config.page_size, reserve_cap=_RESERVE_TOKENS)
        self.prefill_manager = PrefillManager(
            self.cache_manager, self.table_manager, self.decode_manager,
            image_token_id=getattr(
                getattr(self.engine.model, "config", None), "image_token_id", None
            ),
            reserve_cap=_RESERVE_TOKENS,
        )
        # Retracted requests whose last forward is still in flight (uid -> Req): the drain
        # requeues them (or drops them if an abort arrived meanwhile).
        self._retracting: dict[int, Req] = {}
        self._n_retract = 0
        # Prefill/decode time share (_DECODE_SHARE): seconds of decode owed to running
        # requests, earned by prefill chunks; plus "one decode step is due" after every chunk
        # (the chunk's own time is only known when it drains, a batch later under overlap).
        self._decode_credit = 0.0
        self._decode_due = False

        # some alias for easy access
        self.finished_reqs: Set[Req] = set()
        # Abort acknowledgements are a terminal accounting barrier. Queue them while processing
        # inbound control messages, then flush only AFTER _process_last_data publishes any
        # sampled replies from the prior overlapped forward.
        self._pending_abort_acks: Set[int] = set()
        # With multiple tokenizer workers, an AbortBackendMsg and its earlier UserMsg can arrive
        # through different PUSH producers and be observed out of order. Preserve a bounded
        # tombstone so an abort-before-admission request can never be resurrected after its
        # terminal accounting acknowledgement has already been published.
        self._abort_tombstones: dict[int, None] = {}
        self._forward_iter = 0  # global forward counter; drives the SWA proactive-eviction cadence
        # The launched-but-not-yet-drained batch (overlap): set at the top of each overlap_loop
        # iteration so the abort handler can tell whether a request's forward is still in flight
        # (mark it, defer the free to _process_last_data) or not (free immediately). Stays None
        # in normal_loop, where a batch launches and drains within one iteration.
        self._last_data: ForwardData | None = None
        # A received-but-not-yet-executed runtime cache rebuild (CacheRebuildBackendMsg),
        # run at the next idle safe point in overlap_loop. None when no rebuild is pending.
        self._pending_rebuild: CacheRebuildBackendMsg | None = None
        self.tokenizer = load_tokenizer(config.model_path)
        self.eos_token_ids = load_eos_token_ids(config.model_path, self.tokenizer)
        self.toolcall_anchor_id = None
        if config.special_token_ckpt and (
            self.cache_manager.is_hybrid or self.cache_manager.is_swa
        ):
            from freetoken.server.function_call_parser import toolcall_opener_for

            self.toolcall_anchor_id = load_toolcall_anchor_id(
                self.tokenizer,
                toolcall_opener_for(getattr(config, "tool_call_parser", "")),
            )
        self.token_pool = self.table_manager.token_pool
        # Floor the prefill chunk by the cache manager's cap (DSV4: ~half the window pool) so a
        # sliding-window cache chunks long prompts and frees out-of-window pages between chunks
        # instead of OOMing _alloc_window on a prompt longer than the window pool.
        _chunk_cap = self.cache_manager.prefill_chunk_budget
        self._prefill_budget_boot = (
            min(config.max_extend_tokens, _chunk_cap) if _chunk_cap else config.max_extend_tokens
        )
        self.config = config
        self.status_reporter = SchedulerStatusReporter(
            log=logger.info_rank0,
            decode_log_interval=config.decode_log_interval,
        )

        # Initialize the I/O mixin
        super().__init__(config, self.engine.tp_cpu_group)

    def run_when_idle(self) -> None:
        """Called when the scheduler is idle to perform background tasks."""
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()
        el = getattr(self.engine, "elastic", None)
        if el is not None and el.steps > 1 and self.tp_cpu_group.size() == 1:
            # grown KV: give the bytes back to the expert cache. Capacity no live request
            # has needed for evict_after_s is reclaimed at once (evicting prefix cache LRU);
            # otherwise wait idle_s and shrink only as far as compaction allows.
            import time as _t

            q = self._recv_from_tokenizer
            evict = el.top_step_unneeded()
            if not evict:
                t0 = _t.monotonic()
                while _t.monotonic() - t0 < el.idle_s and q.empty():
                    _t.sleep(0.05)
            if q.empty():
                with torch.inference_mode(), torch.cuda.stream(self.engine.stream):
                    while el.steps > 1 and el.shrink(self.cache_manager, allow_evict=evict):
                        pass

    def _elastic_busy_shrink(self, ongoing_data: "ForwardData | None" = None) -> None:
        """Elastic KV under continuous load: once the grown capacity has gone unneeded by
        any live request for evict_after_s, shrink at this safe point (between batches)
        unless a LIVE request's page-table row references a slot of the top step (running
        decodes + partially prefilled chunked requests pin their rows [:device_len],
        including the tree prefix they matched). Tree-only KV there is compacted down
        (evicting LRU prefix cache only if it does not fit)."""
        el = getattr(self.engine, "elastic", None)
        if el is None or el.steps <= 1 or not el.top_step_unneeded():
            return
        import time as _t

        now = _t.monotonic()
        if now - getattr(el, "_busy_try", 0.0) < 15.0:
            return
        el._busy_try = now
        lo = (el.steps - 1) * el.step
        hi = lo + el.step
        live = list(self.decode_manager.running_reqs) + [
            p.chunked_req for p in self.prefill_manager.pending_list if p.chunked_req is not None
        ]
        # Overlap mode: _forward's filter_reqs already dropped requests whose LAST step is in
        # the batch just launched (length-capped decodes, max_tokens=1 prefills); they are
        # only cached/freed by the next _process_last_data, so their rows still hold slots.
        if ongoing_data is not None:
            live += [r for r in ongoing_data[0].batch.reqs if r.table_idx >= 0]
        for r in live:
            # spec decoding pre-allocates [device_len, spec_alloc_len)
            s = self.engine.page_table[r.table_idx, : max(r.device_len, r.spec_alloc_len)]
            if bool(((s >= lo) & (s < hi)).any()):
                return
        torch.cuda.synchronize(self.engine.device)
        with torch.inference_mode(), torch.cuda.stream(self.engine.stream):
            if el.shrink(self.cache_manager, allow_evict=True):
                # the new top step starts its own evict_after_s window (no 15 s cascade)
                el.top_needed_ts = now

    @torch.inference_mode()
    def rebuild_cache(
        self,
        *,
        moe_cache_size: int | None = None,
        num_pages: int | None = None,
        num_mamba_slots: int | None = None,
        num_swa_pages: int | None = None,
    ) -> None:
        """Idle-only runtime cache rebuild: resize the MoE slot cache, KV pages, GDN (mamba) state
        pool, and/or the window pool (num_swa_pages), re-capture CUDA graphs, and re-thread the
        page managers (clearing the prefix cache on a KV/mamba/window resize). The caller MUST
        guarantee the scheduler is idle — no pending prefill, no running decode, no in-flight
        finished requests. All TP ranks must call this with identical arguments.
        """
        assert not self.prefill_manager.runnable, "rebuild requires no pending prefill"
        assert not self.decode_manager.runnable, "rebuild requires no running decode"
        torch.cuda.synchronize(self.device)
        if self.config.tp_info.size > 1:
            self.sync_all_ranks()
        self.engine.rebuild_runtime_cache(
            moe_cache_size=moe_cache_size, num_pages=num_pages, num_mamba_slots=num_mamba_slots,
            num_swa_pages=num_swa_pages,
        )
        if num_pages is not None or num_mamba_slots is not None or num_swa_pages is not None:
            # Any of these resizes invalidates the prefix cache: a KV resize leaves stale page
            # indices, a mamba resize leaves stale GDN-snapshot slot ids, and a window-pool resize
            # (num_swa_pages) reallocates the SWA/window token pool, leaving stale slot ids in the
            # radix tree. Rebuild the prefix cache + reclaim the resized free-lists.
            self.cache_manager.rebuild(self.engine.num_pages, self.engine.page_table)
            if num_pages is not None:
                # token_pool is sized to the page table; only a KV-page resize reallocates it.
                # A mamba-only rebuild leaves the page table untouched, so skip this (else it
                # needlessly reallocates + zeros the whole GPU token_pool every mamba resize).
                self.table_manager.rebuild(self.engine.page_table)
                self.token_pool = self.table_manager.token_pool
            self.cache_manager.check_integrity()
        # The prefill chunk cap tracks the CURRENT window-pool size (DSV4); a rebuild that
        # shrank the pool must shrink the cap too, or the next long prompt is chunked against
        # the stale budget and crashes _alloc_window.
        _chunk_cap = self.cache_manager.prefill_chunk_budget
        self._prefill_budget_boot = (
            min(self.config.max_extend_tokens, _chunk_cap)
            if _chunk_cap else self.config.max_extend_tokens
        )
        if self.config.tp_info.size > 1:
            self.sync_all_ranks()

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        The main loop of overlapping scheduling and execution.

        It will overlap the execution of current batch and processing of last batch's results,
        which can effectively hide CPU latency and improve GPU utilization.
        """
        # Expose the un-drained batch to _process_one_msg (abort in-flight check). Assigning
        # before the message loop is what makes the check airtight: the batch launched later
        # this iteration can only be probed by messages of the NEXT iteration, which sees it here.
        self._last_data = last_data
        blocking = not (
            last_data is not None  # don't block if we have a batch to be processed
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
            or self._pending_rebuild is not None  # a queued rebuild to drain toward + execute
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        # Execute a queued cache rebuild once the scheduler is fully idle (the safe point):
        # no last batch to process, no pending prefill, no running decode. finished_reqs is
        # NOT a gate — those requests are already freed (no live GPU/page resources).
        if self._pending_rebuild is not None and last_data is None and not (
            self.prefill_manager.runnable or self.decode_manager.runnable
        ):
            self._execute_pending_rebuild()

        # Order this iteration's host->device token_pool copies (issued on ``self.stream``
        # during scheduling) after the previous batch's sampled-token writes (issued on the
        # engine stream in ``_forward``). Without this, a request that reuses a just-freed
        # table_idx can have its freshly copied prompt clobbered by the prior occupant's
        # still-pending output write -- corrupting tokens (e.g. dropping an image
        # placeholder, which the multimodal merge then rejects).
        self.stream.wait_stream(self.engine.stream)
        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            with self.engine_stream_ctx:  # run the batch in the engine's stream
                self.engine.stream.wait_stream(self.stream)
                # COW-restore GDN snapshots for prefix hits ON THE ENGINE STREAM, after the
                # cross-stream wait and before the forward reads the live slot (program order
                # vs the prior batch's snapshot writes). Doing this on self.stream would race.
                self._restore_linear_states(forward_input.batch)
                ongoing_data = (forward_input, self._forward(forward_input))

        # The drain issues GPU-visible writes to state the batch just launched still reads: the
        # page-table re-point and, for the paged-SWA pools, the full->swa (DSV4: full->window)
        # sentinel scatter. DSV4 stages the page table at replay time and translates
        # full_to_window INSIDE the captured graph, so an unordered drain can redirect an
        # in-flight forward. copy_done only covers batch N; order against N+1 explicitly.
        self.stream.wait_stream(self.engine.stream)
        self._process_last_data(last_data)
        self._flush_abort_acks()
        self._elastic_busy_shrink(ongoing_data)
        return ongoing_data

    def normal_loop(self) -> None:
        blocking = not (
            self.prefill_manager.runnable
            or self.decode_manager.runnable
            or self._pending_rebuild is not None  # a queued rebuild to execute at idle
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        # Non-overlap mode has no last_data to drain; execute a queued rebuild as soon as
        # the scheduler is idle (no pending prefill / running decode). Without this, a
        # rebuild in DISABLE_OVERLAP_SCHEDULING mode stays pending until the HTTP timeout.
        if self._pending_rebuild is not None and not (
            self.prefill_manager.runnable or self.decode_manager.runnable
        ):
            self._execute_pending_rebuild()

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            # already inside engine_stream_ctx (run_forever); restore on the engine stream
            self._restore_linear_states(forward_input.batch)
            ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(ongoing_data)
        self._flush_abort_acks()
        self._elastic_busy_shrink()

    @torch.inference_mode()
    def run_forever(self) -> NoReturn:
        # DSV4 (owned-KV) decode reads its per-token window/cmp/idx slot maps off the attention
        # backend's per-batch SNAPSHOT (staged in prepare_for_replay right before the replay, on
        # the same stream, like the generic out_loc copy_from), not the live slot maps -- so the
        # next batch's allocate_paged cannot corrupt the in-flight graph replay. DSV4 overlaps.
        if ENV.DISABLE_OVERLAP_SCHEDULING:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self.normal_loop()
        else:
            assert torch.cuda.current_stream() == self.stream
            data = None
            while True:
                data = self.overlap_loop(data)

    def shutdown(self) -> None:
        torch.cuda.synchronize(self.device)
        self.sync_all_ranks()
        self.engine.shutdown()

    def _process_last_data(self, last_data: ForwardData | None) -> None:
        if last_data is None:
            return

        batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
        copy_done.synchronize()
        getattr(self, "_account_time_share", lambda _b: None)(batch)
        reply: List[DetokenizeMsg] = []
        new_finished_reqs: Set[Req] = set()
        with self.cache_manager.lazy_free_region():
            for i, req in enumerate(batch.reqs):
                if isinstance(req, ChunkedReq):
                    # Don't cache intermediate chunks; the full prompt is cached once when the
                    # final chunk is processed. Caching here snapshots a handle the next chunk
                    # already copied (overlap), so cache_req double-frees the prior chunk.
                    if req.aborted:
                        # Aborted mid-chunked-prefill while this chunk was in flight: the abort
                        # popped the pending continuation (no next chunk launches), and this
                        # drain point frees the chunk's pages/slots exactly once.
                        self._free_req_resources(req)
                    continue
                if req.retracted:
                    getattr(self, "_retracting", {}).pop(req.uid, None)
                if req.aborted:
                    # Aborted while this final-chunk prefill / decode step was in flight: free
                    # here (the forward is drained) and finish the request. No DetokenizeMsg --
                    # the abort ack flushed after this method stays the uid's terminal reply.
                    self.decode_manager.remove_req(req)
                    self._free_req_resources(req)
                    new_finished_reqs.add(req)
                    continue
                if req in self.finished_reqs:
                    # Overlap scheduling launched one more decode step for a request that
                    # already terminated (filter_reqs keeps it while output budget remains,
                    # and the next batch is scheduled before this drain runs). Its resources
                    # are freed below/already; shipping this token would append past the
                    # client's terminal reply.
                    continue
                next_token = next_tokens_cpu[i]
                req.append_host(next_token.unsqueeze(0))
                next_token = int(next_token.item())
                # EOS / stop-string -> "stop", output budget exhausted -> "length";
                # EOS and stop strings win over length. Length counts the tokens delivered so
                # far, not device_len: under overlap the next step (already launched) has
                # advanced device_len, which finished every length-capped request one token
                # early (max_tokens=N streamed N-1 and dropped the last forward's token).
                hit_length = len(req.input_ids) >= req.max_device_len
                hit_eos = (
                    not req.sampling_params.ignore_eos and next_token in self.eos_token_ids
                )
                matched_stop = (
                    self._match_stop_str(req)
                    if not hit_eos and req.sampling_params.stop_strs
                    else None
                )
                finished = hit_length or hit_eos or matched_stop is not None
                finish_reason = (
                    ("stop" if (hit_eos or matched_stop is not None) else "length")
                    if finished
                    else None
                )
                if (
                    next_token == self.toolcall_anchor_id
                    and req.toolcall_anchor_len is None
                    and not finished
                ):
                    req.toolcall_anchor_len = req.input_ids.numel()
                reply.append(
                    DetokenizeMsg(
                        uid=req.uid,
                        next_token=next_token,
                        finished=finished,
                        finish_reason=finish_reason,
                        matched_stop=matched_stop,
                        stop_strs=req.sampling_params.stop_strs or None,
                    )
                )

                # NOTE: overlap scheduling may make the request freed twice, skip second free
                if finished and req not in self.finished_reqs:
                    self.decode_manager.remove_req(req)
                    self._free_req_resources(req)
                    new_finished_reqs.add(req)
                elif req.retracted:
                    # retracted under this in-flight forward: its pages are already back in
                    # the pool / prefix cache; resume it from the token just sampled
                    self._requeue(req)
                elif batch.is_prefill and req.table_idx != -1:
                    # for prefill, non-chunk req, cache the prefix.
                    # Polymorphic: the DSV4 naive manager keeps the request's slots (no-op);
                    # the generic manager inserts the prefix into its radix/naive cache.
                    # table_idx == -1 is defense-in-depth: aborts mark in-flight requests
                    # instead of freeing them (handled above), so a freed request should
                    # never reach this commit -- but if a future path frees one early, skip
                    # rather than re-read the freed page-table row (and on hybrid, deref the
                    # None'd GDN ping-pong slots).
                    self.cache_manager.cache_req(req, finished=False)

        self.finished_reqs = new_finished_reqs
        # Stamp each reply with the post-batch KV page occupancy so the frontend (shell
        # status bar) can show live KV usage without a separate query.
        used, total = self._kv_usage_pages()
        mamba_slots = self._mamba_slot_usage()
        swa_tokens = self._swa_token_usage()
        if reply:
            mem = self._gpu_mem_bytes()
            mamba_used, mamba_total = mamba_slots or (0, 0)
            swa_used, swa_total = swa_tokens or (0, 0)
            for m in reply:
                m.kv_used_pages = used
                m.kv_total_pages = total
                m.mamba_used_slots = mamba_used
                m.mamba_total_slots = mamba_total
                m.swa_used_tokens = swa_used
                m.swa_total_tokens = swa_total
                m.gpu_mem_bytes = mem
        self.status_reporter.report_batch(
            batch,
            running_reqs=len(self.decode_manager.running_reqs),
            queue_reqs=len(self.prefill_manager.pending_list),
            kv_used_pages=used,
            kv_total_pages=total,
            page_size=self.config.page_size,
            mamba_slots=mamba_slots,
            swa_tokens=swa_tokens,
        )
        self.send_result(reply)

    def _match_stop_str(self, req: Req) -> str | None:
        """First stop string present in this request's generated tail, else None. Decodes
        only a short suffix (bounded by the longest stop string's char length, so a stop of
        N chars spans at most N tokens) to keep the per-step cost small."""
        stop_strs = req.sampling_params.stop_strs
        prompt_len = req.max_device_len - req.output_len
        if req.stop_prompt_len is not None:
            prompt_len = req.stop_prompt_len
        if len(req.input_ids) <= prompt_len:
            return None
        max_chars = max(len(s) for s in stop_strs)
        tail_start = max(prompt_len, len(req.input_ids) - (max_chars + 1))
        tail = self.tokenizer.decode(req.input_ids[tail_start:].tolist())
        for s in stop_strs:
            if s in tail:
                return s
        return None

    def _kv_usage_pages(self) -> Tuple[int, int]:
        """(used_pages, total_pages) of the KV page pool.

        ``used`` follows SGLang's logging semantics: allocated pages that are not
        evictable (active requests + protected prefix cache). Evictable prefix-cache
        pages are available to future requests, so they are excluded from usage.
        Always the manager's own primary pool (for DSV4 the FULL cmp/idx tier); the
        window (swa) tier is reported separately by ``_swa_token_usage``.
        """
        return self.cache_manager.page_usage()

    def _mamba_slot_usage(self) -> Tuple[int, int] | None:
        """(used_slots, total_slots) of the GDN-state (mamba) pool for hybrid models, else None.

        Mirrors SGLang's mamba-pool semantics: ``total`` excludes the reserved padding
        sink (slot 0); ``used`` excludes free slots and evictable tree snapshots.
        """
        if not self.cache_manager.is_hybrid:
            return None
        total = self.cache_manager.linear_state_pool.num_slots - 1
        return total - self.cache_manager.mamba_available_size, total

    def _swa_token_usage(self) -> Tuple[int, int] | None:
        """(used_tokens, total_tokens) of the window (swa) pool for SWA models, else None.

        Mirrors the mamba accounting: ``total`` excludes the pool's reserved sentinel
        unit; ``used`` excludes free slots and evictable (unlocked) tree tokens.
        """
        cm = self.cache_manager
        if not cm.swa_paged:
            return None
        total = cm.swa_pool.swa_num_tokens - 1
        return total - cm.swa_available_size, total

    def _gpu_mem_bytes(self) -> int:
        """Bytes this engine process holds on the GPU (torch's reserved caching-allocator
        pool: weights + KV + MoE cache + graphs). 0 on CPU. Cheap, no device sync."""
        if self.device.type != "cuda":
            return 0
        return torch.cuda.memory_reserved(self.device)

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        if isinstance(msg, BatchBackendMsg):
            for msg in msg.data:
                self._process_one_msg(msg)
        elif isinstance(msg, ExitMsg):
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):
            logger.debug_rank0("Received user msg: %s", msg)
            tombstones = getattr(self, "_abort_tombstones", None)
            if tombstones is not None and msg.uid in tombstones:
                tombstones.pop(msg.uid, None)
                logger.debug_rank0(
                    "Dropping request %d because its abort arrived before admission", msg.uid
                )
                return
            _img_tid = getattr(getattr(self.engine.model, "config", None), "image_token_id", None)
            if _img_tid is not None and not getattr(msg, "mm_images", None):
                # mm-forensic: a prompt carrying image placeholders MUST arrive with
                # pixel payloads; if not, its span would enter the prefix cache as raw
                # placeholder ids and false-hit any other image (and the model never
                # sees the pixels). Refuse loudly instead of serving garbage.
                if msg.mm_embeds is None and bool((msg.input_ids == _img_tid).any()):
                    logger.warning_rank0(
                        f"mm-forensic uid={msg.uid}: image placeholders present but "
                        f"mm payload missing (len={len(msg.input_ids)}); rejecting"
                    )
                    self.send_result(
                        [
                            ErrorReplyMsg(
                                uid=msg.uid,
                                error="image placeholders present but pixel payload missing",
                                code="invalid_image",
                            )
                        ]
                    )
                    return
            if getattr(msg, "mm_images", None):
                # Vision: encode pixels into soft-token embeddings here -- this process
                # owns the model/GPU; the tower is a ~0.6B BF16 ViT, a few ms per image.
                try:
                    import numpy as np

                    model = self.engine.model
                    assert hasattr(model, "encode_images"), (
                        "model is text-only (boot with FREETOKEN_GLM5_VISION=1)"
                    )
                    pix_parts, grids = [], []
                    for d in msg.mm_images:
                        t, gh, gw = d["grid"]
                        arr = np.frombuffer(d["pix"], dtype=np.float32).copy()
                        pix_parts.append(arr.reshape(t * gh * gw, -1))
                        grids.append([t, gh, gw])
                    _cuda = torch.cuda.is_available()
                    if _cuda:  # host-side allocator stats only (no device sync)
                        _alloc0 = torch.cuda.memory_allocated(self.device)
                        torch.cuda.reset_peak_memory_stats(self.device)
                    msg.mm_embeds = model.encode_images(
                        torch.from_numpy(np.concatenate(pix_parts, axis=0)).to(
                            self.device, torch.bfloat16
                        ),
                        torch.tensor(grids, dtype=torch.long, device=self.device),
                    )
                    if _cuda:
                        # vision-encode transient high-water mark (VRAM headroom check)
                        logger.info_rank0(
                            "vision encode uid=%d patches=%d peak +%d MiB",
                            msg.uid, sum(len(p) for p in pix_parts),
                            (torch.cuda.max_memory_allocated(self.device) - _alloc0) >> 20,
                        )
                    # Content-hash cache keys: fill each image's placeholder span with a
                    # per-image pixel hash (negative id range). Identical prefixes with the
                    # SAME image become prefix-cache hits; different images diverge at the
                    # span's first token, so cross-image reuse is impossible.
                    image_token_id = getattr(
                        getattr(self.engine.model, "config", None), "image_token_id", None
                    )
                    if image_token_id is not None:
                        import hashlib

                        key_ids = msg.input_ids.clone()
                        pos = (key_ids == image_token_id).nonzero(as_tuple=True)[0]
                        runs = []
                        if pos.numel():
                            brk = (pos[1:] - pos[:-1] > 1).nonzero(as_tuple=True)[0] + 1
                            runs = [
                                t
                                for t in torch.tensor_split(pos, brk.tolist())
                                if t.numel()
                            ]
                        expected = sum(d["grid"][0] for d in msg.mm_images)
                        assert len(runs) == expected, (
                            f"{len(runs)} image-token spans vs {expected} expected "
                            "(one per image, grid_t per video; literal placeholder "
                            "text in the prompt?)"
                        )
                        ri = 0
                        for d in msg.mm_images:
                            # Key = (grid, pixels): the patch rows alone are ambiguous -- the
                            # same bytes under a transposed / re-factored grid (e.g. a solid-color
                            # 112x56 vs 56x112 image) are a different image to the tower (rope
                            # positions, spatial merge), so they must not share a cache key.
                            hb = hashlib.blake2b(digest_size=8)
                            hb.update(np.asarray(d["grid"], dtype=np.int64).tobytes())
                            hb.update(d["pix"])
                            h = int.from_bytes(hb.digest(), "big")
                            hid = -(h % (2**31 - 2)) - 2
                            for _ in range(d["grid"][0]):
                                key_ids[runs[ri]] = hid
                                ri += 1
                        msg.cache_key_ids = key_ids
                        logger.info_rank0(
                            "mm-forensic uid=%d images=%d hashes=%s",
                            msg.uid,
                            len(msg.mm_images),
                            [hex(-int(key_ids[r[0]]) & 0xFFFFFFFF) for r in runs[:4]],
                        )
                    msg.mm_images = None
                except Exception as exc:  # noqa: BLE001 -- reply, never crash the loop
                    logger.warning_rank0(
                        f"vision encode failed for request {msg.uid}: {exc!r}"
                    )
                    self.send_result(
                        [
                            ErrorReplyMsg(
                                uid=msg.uid,
                                error=f"could not encode request images: {exc}",
                                code="invalid_image",
                            )
                        ]
                    )
                    return
            input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
            max_output_len = max_seq_len - input_len
            if max_output_len <= 0:
                logger.warning_rank0(
                    f"Input sequence length {input_len} exceeds {max_seq_len}, "
                    f"request {msg.uid} is dropped."
                )
                # Tell the client instead of dropping silently — otherwise its wait_for_ack
                # never sees a `finished` reply and hangs until the request times out.
                self.send_result(
                    [
                        ErrorReplyMsg(
                            uid=msg.uid,
                            # "prompt is too long: N tokens > M" is the phrasing Claude Code and
                            # OpenClaw match on; the Anthropic wire has no error code to read.
                            error=(
                                f"prompt is too long: {input_len} tokens > {max_seq_len} maximum "
                                f"(prompt + generation); shorten the prompt or increase the KV "
                                f"cache budget"
                            ),
                            # OpenAI's standard class for this, for clients that read a code.
                            code="context_length_exceeded",
                        )
                    ]
                )
                return
            if msg.sampling_params.max_tokens > max_output_len:
                msg.sampling_params.max_tokens = max_output_len
                logger.warning_rank0(
                    f"Adjust max_tokens to {max_output_len} for request {msg.uid}."
                )
            self.prefill_manager.add_one_req(msg)
        elif isinstance(msg, AbortBackendMsg):
            logger.debug_rank0("Aborting request %d", msg.uid)
            tombstones = getattr(self, "_abort_tombstones", None)
            if tombstones is None:
                tombstones = self._abort_tombstones = {}
            tombstones[msg.uid] = None
            # Unknown aborts normally consume their tombstone when the cross-worker UserMsg
            # catches up. Bound hostile/no-followup abort traffic without affecting realistic
            # in-flight concurrency.
            while len(tombstones) > 65_536:
                tombstones.pop(next(iter(tombstones)))
            req_to_free = self.prefill_manager.abort_req(msg.uid)
            req_to_free = req_to_free or self.decode_manager.abort_req(msg.uid)
            retracting = getattr(self, "_retracting", None)
            if req_to_free is None and retracting and msg.uid in retracting:
                # retracted (resources already freed) with its last forward in flight: the
                # drain must drop it instead of requeueing it
                retracting[msg.uid].aborted = True
            if req_to_free is not None:
                # SGLang-style abort: never free resources under an in-flight forward. If the
                # request is in the launched-but-not-drained batch (overlap), only mark it;
                # _process_last_data frees it this same iteration, after copy_done.synchronize()
                # -- so its KV pages / GDN slots are never recycled mid-write, and the
                # finished=False prefix-commit can't run on a freed request. A request with no
                # forward in flight (e.g. a decode req starved behind a long chunked prefill)
                # is freed immediately -- deferring would leak until its next batch, which
                # strict prefill-priority puts arbitrarily far away.
                inflight = (
                    self._last_data is not None
                    and req_to_free in self._last_data[0].batch.reqs
                )
                if inflight:
                    req_to_free.aborted = True
                else:
                    self._free_req_resources(req_to_free)
            # Always acknowledge the abort, even when the request already left the manager,
            # but NOT yet: overlap_loop still has to publish the prior forward's sampled reply.
            # _flush_abort_acks runs after _process_last_data, making this a true terminal
            # accounting barrier for FrontendManager/prepare-stop.
            self._pending_abort_acks.add(msg.uid)
        elif isinstance(msg, CacheRebuildBackendMsg):
            # v1 scope: only if_idle, single-rank, non-owned-KV. drain mode and TP rebuild
            # need the drain-gate / all-rank failure-agreement machinery (deferred), so we
            # reject them cleanly rather than ship hang-prone half-wired paths.
            if not self.cache_manager.supports_runtime_rebuild:
                self._reply_rebuild(
                    msg.request_id, "unsupported", "this model's cache does not support runtime rebuild"
                )
            elif msg.mode != "if_idle":
                self._reply_rebuild(
                    msg.request_id, "unsupported", f"mode {msg.mode!r} unsupported (use if_idle)"
                )
            elif self.config.tp_info.size > 1:
                self._reply_rebuild(
                    msg.request_id, "unsupported", "runtime rebuild unsupported under TP > 1"
                )
            elif self.prefill_manager.runnable or self.decode_manager.runnable:
                # if_idle: refuse rather than wait. (finished_reqs hold no resources — they
                # are already freed — so they do not block a rebuild.)
                self._reply_rebuild(msg.request_id, "busy")
            else:
                self._pending_rebuild = msg
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    def _restore_linear_states(self, batch) -> None:
        """COW-restore a hybrid prefix hit's GDN snapshot into its freshly-allocated live slot
        (first chunk only). MUST run on the ENGINE stream so it is program-ordered after the
        prior batch's snapshot writes and before this forward reads the live slot."""
        pool = self.engine.linear_state_pool
        if pool is None or not batch.is_prefill:
            return
        if not any(r.mamba_restore_src is not None for r in batch.reqs):
            return
        # GDN hit/donate barrier (empirical fix, 2026-08-30): without a full device sync
        # here, prefix-HIT requests co-scheduled with in-flight decode intermittently
        # restore/donate corrupted GDN state (~10% wrong answers under saturation:
        # blank visual grounding, empty/garbled text -- reproduced by the abort/text/
        # saturation hammers, eliminated 0/30+ by this barrier pair). Stream-level
        # wait_stream ordering alone does not close the window; the racing party is
        # host-side (suspected pinned-staging or free-list lifetime), root cause still
        # open. Cost: one sync per HIT admission -- request-level, sub-millisecond.
        torch.cuda.synchronize(self.device)
        import os as _os

        _dbg = _os.getenv("FREETOKEN_GDN_DEBUG", "0") != "0" or _os.getenv("FREETOKEN_GDN_DEBUG_R", "0") != "0"
        for req in batch.reqs:
            if req.mamba_restore_src is not None:
                if _dbg:
                    logger.info_rank0(
                        "gdn-restore uid=%d src=%d dst=%d src_cksum=%.3f",
                        req.uid, req.mamba_restore_src, req.linear_slot_idx,
                        pool.debug_checksum(req.mamba_restore_src),
                    )
                pool.copy_from(req.mamba_restore_src, req.linear_slot_idx)
                req.mamba_restore_src = None  # consumed: restore exactly once

    def _free_req_resources(self, req: Req) -> None:
        # Idempotent: an EOS-finished request can stay in running_reqs (output budget left), so an
        # abort in the same overlap iteration races _process_last_data and would free it twice --
        # double-freeing its table_idx and (hybrid) GDN slots onto the free-list, handing the same
        # slots to two later requests. table_idx == -1 marks an already-freed request.
        if req.table_idx == -1:
            return
        # Polymorphic free: the DSV4 manager returns the request's window pages + cmp/idx blocks
        # to their tier free-lists; the generic manager frees its KV pages (it reads
        # page_table[req.table_idx], so free the table entry after).
        self.cache_manager.cache_req(req, finished=True)
        self.table_manager.free(req.table_idx)
        req.table_idx = -1

    def _reply_rebuild(self, request_id: str, status: str, error: str | None = None) -> None:
        # Single source of truth with the rollback snapshot (_current_cache_geometry): mamba is
        # usable slots (padding sink excluded, matching the status-bar gauge), and num_swa_pages
        # reports 0 unless the model actually has a window pool.
        geo = self._current_cache_geometry()
        self.send_result(
            [
                CacheRebuildResultMsg(
                    request_id=request_id,
                    status=status,
                    moe_cache_size=geo["moe_cache_size"] or 0,
                    num_pages=geo["num_pages"],
                    mamba_slots=geo["num_mamba_slots"] or 0,
                    num_swa_pages=geo["num_swa_pages"] or 0,
                    error=error,
                )
            ]
        )

    def _execute_pending_rebuild(self) -> None:
        from freetoken.engine.engine import CacheRebuildRejected

        msg = self._pending_rebuild
        assert msg is not None
        self._pending_rebuild = None
        requested = {
            "moe_cache_size": msg.moe_cache_size,
            "num_pages": msg.num_pages,
            "num_mamba_slots": msg.num_mamba_slots,
            "num_swa_pages": msg.num_swa_pages,
        }
        # Rollback target: the CURRENT (serving) sizes of ONLY the pools this request touches.
        # Passing the untouched pools too would trip rebuild_cache's KV/mamba/SWA gate and wipe
        # the prefix cache that a successful resize of just the requested pool preserves.
        snapshot = self._current_cache_geometry()
        prior = {k: snapshot[k] for k, v in requested.items() if v is not None}
        # Cleared here, set by engine.rebuild_runtime_cache at its point of no return — lets the
        # except below tell a pre-teardown failure (engine untouched) from a mid-teardown one.
        self.engine.rebuild_teardown_started = False
        try:
            self.rebuild_cache(**requested)
        except CacheRebuildRejected as e:
            # Rejected before any destructive free — old cache intact, keep serving.
            logger.warning(f"cache rebuild rejected: {e}")
            self._reply_rebuild(msg.request_id, "rejected", error=str(e))
            return
        except Exception as e:  # noqa: BLE001
            if not getattr(self.engine, "rebuild_teardown_started", True):
                # Failed before the destructive phase began: graphs and pools are untouched and
                # the engine is still serving. A destructive rollback would only add risk.
                logger.error(f"cache rebuild failed before teardown: {e!r} — old cache intact")
                self._reply_rebuild(msg.request_id, "rejected", error=repr(e))
                return
            if self.config.tp_info.size > 1:
                # A lone-rank failure cannot be rolled back symmetrically: rebuild_cache runs TP
                # barriers, and ranks that succeeded will not re-enter them — a solo rollback
                # would desync the group. Keep the latch-failed behavior for tp>1.
                logger.error(f"cache rebuild failed: {e!r} — tp>1, latching failed")
                self._reply_rebuild(msg.request_id, "failed", error=repr(e))
                return
            # The destructive phase failed — typically a CUDA OOM while reallocating a pool or
            # recapturing graphs. The graphs/pools are already torn down, so the engine cannot
            # serve as-is. Rather than latch "failed" (which forces a full process restart),
            # rebuild the touched pools back to the sizes that were serving a moment ago: they
            # fit before, so shrinking back frees the just-attempted allocation and restores
            # service. Only if the rollback ALSO fails is the engine genuinely wedged. (Post-OOM
            # CUDA state is not guaranteed sane — a rollback that succeeds here may still surface
            # a deferred fault on a later request; that residual risk is accepted over always
            # forcing a restart.)
            logger.error(f"cache rebuild failed: {e!r} — rolling back to the previous geometry")
            try:
                self.rebuild_cache(**prior)
            except Exception as e2:  # noqa: BLE001 — rollback failed too; genuinely unrecoverable
                logger.error(f"cache rebuild rollback failed: {e2!r} — server latched failed")
                self._reply_rebuild(
                    msg.request_id,
                    "failed",
                    error=f"{e!r}; rollback to the prior geometry also failed: {e2!r}",
                )
                return
            logger.warning("cache rebuild rolled back to the previous geometry — still serving")
            self._log_cache_geometry("Cache rolled back")
            self._reply_rebuild(
                msg.request_id, "rejected", error=f"rebuild failed and was rolled back: {e!r}"
            )
            return
        # Outside the try: an ack/send failure after a fully-applied rebuild must not be
        # mistaken for a rebuild failure and roll back the geometry the engine now serves.
        self._log_cache_geometry("Cache rebuilt")
        self._reply_rebuild(msg.request_id, "ok")

    def _current_cache_geometry(self) -> dict:
        """The pools' current (serving) sizes as rebuild_cache kwargs — the rollback snapshot and
        the single source for _reply_rebuild's readout. None for a pool this model lacks
        (rebuild_cache skips those; the reply maps them to the wire format's 0). num_swa_pages is
        the CONCRETE current window (usable pages) so a rollback restores it byte-for-byte,
        whether it was pinned or ratio-derived."""
        eng = self.engine
        config = self.config
        mc = config.model_config
        num_swa_pages = None
        if getattr(mc, "dsv4_args", None) is not None:
            sizes = getattr(eng.kv_cache, "sizes", None)
            if sizes is not None:  # usable window pages = physical n_win_pages minus the dummy page
                num_swa_pages = max(0, sizes.n_win_pages - 1)
        elif getattr(mc, "has_swa_attention", False) and (
            getattr(config, "cache_type", None) == "swa_radix"
        ):  # usable window tokens = pool tokens minus the slot-0 sentinel
            num_swa_pages = max(0, int(getattr(eng.kv_cache, "swa_num_tokens", 0) or 0) - 1)
        return dict(
            num_pages=eng.num_pages,
            moe_cache_size=eng.moe_offload_cache.cache_size if eng.moe_offload_cache is not None else None,
            num_mamba_slots=(eng.linear_state_pool.num_slots - 1) if eng.linear_state_pool is not None else None,
            num_swa_pages=num_swa_pages,
        )

    def _log_cache_geometry(self, event: str) -> None:
        """One-line readout of every pool's new size + VRAM after a rebuild changed them:
        full KV always; swa/mamba/MoE only for models with the pool. Byte figures are
        best-effort (0 when a unit cost cannot be measured) and must never block the reply."""
        from freetoken.kvcache.cache_status import compute_cache_pools, compute_cache_unit_bytes

        try:
            pools = compute_cache_pools(self.engine)
            unit = compute_cache_unit_bytes(self.engine)
            kv_tokens = pools["num_pages"] * pools["page_size"]
            parts = [
                f"KV {pools['num_pages']} pages"
                f" ({kv_tokens} tokens, {_gib(kv_tokens * unit['kv_bytes_per_token'])})"
            ]
            if pools["num_swa_pages"]:
                swa_tokens = pools["num_swa_pages"] * pools["swa_page_size"]
                parts.append(
                    f"swa {pools['num_swa_pages']} pages"
                    f" ({swa_tokens} tokens, {_gib(swa_tokens * unit['swa_bytes_per_token'])})"
                )
            if pools["num_mamba_slots"]:
                parts.append(
                    f"mamba {pools['num_mamba_slots']} slots"
                    f" ({_gib(pools['num_mamba_slots'] * unit['mamba_bytes_per_slot'])})"
                )
            moe = self.engine.moe_offload_cache
            if moe is not None:
                parts.append(
                    f"MoE cache {moe.cache_size}/{moe.num_layers * moe.num_experts}"
                    f" ({_gib(moe.cache_size * unit['moe_bytes_per_expert'])})"
                )
            logger.info_rank0(f"{event}: " + ", ".join(parts))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"could not log cache geometry: {e!r}")

    def _prepare_batch(self, batch: Batch) -> ForwardInput:
        self.engine.graph_runner.pad_batch(batch)
        self._forward_iter += 1
        if batch.is_decode:
            # Free each decoding request's now-out-of-window SWA slots BEFORE the alloc below,
            # so they can back the new token -- this is what bounds the per-request swa
            # footprint during decode. (no-op unless the model is SWA / paged swa pool.)
            self.cache_manager.maybe_free_swa_out_of_window(
                batch.reqs, forward_iter=self._forward_iter)
            for req in batch.reqs:
                req.decode_batch_idx += 1
        else:
            # Prefill sibling of the decode driver: free out-of-window swa BEFORE allocating
            # this chunk, so a chunked prompt longer than the swa pool never accumulates its
            # whole swa footprint (which would exhaust alloc_swa). No-op unless SWA/paged.
            self.cache_manager.free_swa_out_of_window_extend(batch.reqs)
        # Polymorphic page allocation: DSV4 allocates window pages + cmp/idx blocks into its
        # slot maps; the generic manager allocates KV pages into the page table.
        self.cache_manager.allocate_paged(batch.reqs)
        if batch.is_prefill:
            self._gather_multimodal(batch)
        batch.positions = _make_positions(batch, self.device)
        input_mapping = _make_input_tuple(batch, self.device)
        write_mapping = _make_write_tuple(batch, self.device)
        batch.out_loc = self.engine.page_table[input_mapping]
        if self.engine.linear_state_pool is not None:
            if batch.is_decode:
                # GPU GDN-state slot (one per padded request) for the decode gather/scatter;
                # lands in the CUDA-graph input buffer via copy_from. Gate on the cache mode,
                # NOT on whether any padded req has a linear_slot_idx -- the persistent dummy
                # req always carries one (= padding_slot), so that test is True even for naive
                # and would collapse all real naive reqs onto the padding slot. Hybrid: build
                # per padded req from Req.linear_slot_idx (dummy -> padding_slot). Naive: keep
                # the old keying = input_mapping's table_idx column (already staged, no H2D).
                if self.cache_manager.is_hybrid:
                    pool = self.engine.linear_state_pool
                    slots = [r.linear_slot_idx if r.linear_slot_idx is not None
                             else pool.padding_slot for r in batch.padded_reqs]
                    batch.linear_table_idx = torch.tensor(
                        slots, dtype=torch.int32, device="cpu", pin_memory=True
                    ).to(self.device, non_blocking=True)
                else:
                    batch.linear_table_idx = input_mapping[0].to(torch.int32)
            # Per-forward GDN metadata (cu_seqlens / cache_indices / continuation flags),
            # built once here instead of rebuilt in each of the 30 GDN layers. For decode
            # under CUDA graph the persistent cu_seqlens buffer is supplied by set_batch.
            batch.fla_metadata = build_fla_metadata(batch, self.device)
        if batch.is_decode:
            # This batch's padded per-row page-table rows. Backends that snapshot the table for
            # a captured replay (DSV4) read them in prepare_metadata / prepare_for_replay.
            batch.active_table_idx = input_mapping[0].view(-1)
        self.engine.attn_backend.prepare_metadata(batch)
        return ForwardInput(
            batch=batch,
            sample_args=self.engine.sampler.prepare(batch),
            input_tuple=input_mapping,
            write_tuple=write_mapping,
        )

    def _gather_multimodal(self, batch: Batch) -> None:
        """Concatenate per-request vision soft tokens (in request order) for a prefill
        batch so the model can scatter them at image-token positions. ``req.mm_embeds``
        is kept (not cleared) so the cache manager can recognize multimodal requests and
        keep them out of the shared prefix cache (image placeholders share a token id but
        carry per-image content)."""
        parts = [req.mm_embeds for req in batch.reqs if req.mm_embeds is not None]
        if parts:
            batch.mm_embeds = torch.cat(parts, dim=0)

    def _schedule_next_batch(self) -> ForwardInput | None:
        decode_first = (
            _decode_share() > 0
            and (getattr(self, "_decode_due", False) or getattr(self, "_decode_credit", 0.0) > 0)
            and self.decode_manager.runnable
            and self.prefill_manager.runnable
        )
        batch = None if decode_first else self._schedule_prefill()
        if batch is None:
            batch = self.decode_manager.schedule_next_batch()
            if batch is not None:
                batch = self._make_room_for_decode(batch)
            if batch is None and decode_first:
                batch = self._schedule_prefill()  # decode retracted itself away
        if batch is None:
            return None
        batch.ts_decode_first = decode_first and batch.is_decode
        if batch.is_prefill:
            self._decode_due = self.decode_manager.runnable and _decode_share() > 0
        elif decode_first:
            self._decode_due = False
        forward_input = self._prepare_batch(batch)
        self._report_prompt_admissions(batch)
        return forward_input

    def _schedule_prefill(self) -> Batch | None:
        self._make_room_for_continuation()
        self.decode_manager.reserve_cap = self.prefill_manager.reserve_cap = _reserve_tokens()
        batch = self.prefill_manager.schedule_next_batch(self.prefill_budget)
        if getattr(self.prefill_manager, "rejected", None):
            self._reply_prefill_rejections()
        return batch

    def _account_time_share(self, batch: Batch) -> None:
        """Drain-time bookkeeping of the decode share: prefill chunks earn decode credit while
        decodes wait, decode steps run for credit spend it. A batch's time is its span on the
        engine stream (CUDA events around _forward) -- host clocks around the launch miss it,
        since the EP prefill launch itself blocks on the device for most of the chunk."""
        ev = getattr(batch, "ts_events", None)
        if ev is None:
            return
        if not (self.prefill_manager.runnable and self.decode_manager.runnable):
            self._decode_credit = 0.0  # nobody waits for anybody
            return
        ev[1].synchronize()  # just past this batch's copy_done
        dt = ev[0].elapsed_time(ev[1]) / 1000.0
        if batch.is_prefill:
            _share = _decode_share()
            self._decode_credit += dt * _share / (1.0 - _share)
        elif getattr(batch, "ts_decode_first", False):
            self._decode_credit = max(self._decode_credit - dt, -1.0)

    # ---- context pool: retraction (see _RESERVE_TOKENS) ----
    def _pool_holders(self) -> List[Tuple[Req, PendingReq | None]]:
        """Requests holding pool pages that may be retracted: running decodes and an
        in-progress chunked prefill (with its pending entry). Multimodal requests are not
        retractable (their image rows cannot be re-derived from the token ids)."""
        out: List[Tuple[Req, PendingReq | None]] = [
            (r, None) for r in self.decode_manager.running_reqs if r.mm_embeds is None
        ]
        out += [
            (p.chunked_req, p) for p in self.prefill_manager.pending_list
            if p.chunked_req is not None and p.mm_embeds is None
        ]
        return out

    def _make_room_for_decode(self, batch: Batch) -> Batch | None:
        """Retract the youngest holders until this decode step's pages (and window pages) fit."""
        cm = self.cache_manager
        while batch.reqs and not cm.can_allocate(batch.reqs):
            holders = self._pool_holders()
            if not holders:
                break  # nothing retractable: let allocate_paged report the shortage
            victim, pending = max(holders, key=lambda h: h[0].uid)
            self._retract(victim, pending, why=f"decode step needs {cm.tokens_needed(batch.reqs)}")
            if victim in batch.reqs:
                batch.reqs.remove(victim)
        return batch if batch.reqs else None

    def _make_room_for_continuation(self) -> None:
        """The head chunked prefill was admitted with its whole prompt reserved, but running
        decodes may since have grown past their reservations. Retract decodes YOUNGER than it
        until its next chunk fits; an older decode keeps running (the chunk then waits, see
        PrefillAdder.defer_short_continuation)."""
        pl = self.prefill_manager.pending_list
        if not pl or pl[0].chunked_req is None or not self.decode_manager.runnable:
            return
        head, cr = pl[0], pl[0].chunked_req
        ps = self.cache_manager.page_size
        start = cr.cached_len
        end = min(start + self.prefill_budget, head.input_len)
        need = (-(-end // ps) - (-(-start // ps))) * ps
        while need > self.cache_manager.available_size:
            younger = [r for r in self.decode_manager.running_reqs
                       if r.uid > head.uid and r.mm_embeds is None]
            if not younger:
                return
            self._retract(max(younger, key=lambda r: r.uid), None,
                          why=f"prefill chunk of uid {head.uid} needs {need}")

    def _retract(self, req: Req, pending: PendingReq | None, *, why: str) -> None:
        cm = self.cache_manager
        self._n_retract += 1
        logger.info_rank0(
            f"context pool short ({why} tokens, available {cm.available_size}): retracting "
            f"uid {req.uid} at {req.cached_len} tokens "
            f"({'chunked prefill' if pending is not None else 'decode'}), "
            f"#retractions {self._n_retract}"
        )
        if pending is not None:
            # In-progress chunked prefill: commit what is prefilled to the prefix cache and
            # restart it from there (its pending entry stays at the head of the queue). A chunk
            # still in flight drains as a plain ChunkedReq (no commit), so freeing now is safe:
            # the GPU work is stream-ordered before any reuse of the pages.
            self._free_req_resources(req)
            pending.chunked_req = None
            return
        self.decode_manager.remove_req(req)
        inflight = self._last_data is not None and req in self._last_data[0].batch.reqs
        self._free_req_resources(req)
        if inflight:
            req.retracted = True  # the drain appends its token, then requeues it
            self._retracting[req.uid] = req
        else:
            self._requeue(req)

    def _requeue(self, req: Req) -> None:
        """Queue a retracted request again: prompt = everything it has so far, max_tokens = its
        remaining budget, same uid (the client stream continues). Ahead of every younger
        waiting request, behind the in-progress chunked prefill."""
        remain = req.max_device_len - len(req.input_ids)
        if remain <= 0:
            return
        sp = dataclasses.replace(req.sampling_params, max_tokens=remain)
        stop_prompt_len = (req.stop_prompt_len if req.stop_prompt_len is not None
                           else req.max_device_len - req.output_len)
        p = PendingReq(req.uid, req.input_ids.clone(), sp, resumed=True,
                       stop_prompt_len=stop_prompt_len)
        pl = self.prefill_manager.pending_list
        i = 0
        while i < len(pl) and (pl[i].chunked_req is not None or pl[i].uid < p.uid):
            i += 1
        pl.insert(i, p)

    def _reply_prefill_rejections(self) -> None:
        """Finish the requests the prefill pass refused (a multimodal prompt that can never
        be prefilled: see PrefillAdder._add_one_req) with a per-request error, the same
        terminal ErrorReplyMsg path as an over-long prompt. They were dropped from the queue
        before any budget/pool mutation, hold no resources, and were never admitted."""
        rejected, self.prefill_manager.rejected = self.prefill_manager.rejected, []
        for uid, error, _code in rejected:
            logger.warning_rank0(f"request {uid} rejected at prefill admission: {error}")
        self.send_result(
            [ErrorReplyMsg(uid=uid, error=error, code=code) for uid, error, code in rejected]
        )

    def _report_prompt_admissions(self, batch: Batch) -> None:
        """Publish first-prefill accounting only after batch preparation succeeded.

        ``send_result`` is rank-aware: TP rank 0 forwards the signal, other ranks are
        no-ops. The offline handler explicitly ignores this online-accounting message.
        """
        if not batch.is_prefill or not batch.prompt_admissions:
            return
        self.send_result(
            [
                PromptAdmittedMsg(uid=uid, prompt_tokens=prompt_tokens, cached_tokens=cached_tokens)
                for uid, prompt_tokens, cached_tokens in batch.prompt_admissions
            ]
        )

    def _flush_abort_acks(self) -> None:
        pending = getattr(self, "_pending_abort_acks", None)
        if not pending:
            return
        uids = sorted(pending)
        pending.clear()
        self.send_result([ErrorReplyMsg(uid=uid, error="request aborted") for uid in uids])

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        batch, sample_args, input_mapping, output_mapping = forward_input
        if _decode_share() > 0:  # time-share accounting (_account_time_share)
            ev0 = torch.cuda.Event(enable_timing=True)
            ev0.record()
        batch.input_ids = self.token_pool[input_mapping]
        if self.toolcall_anchor_id is not None and not batch.is_prefill:
            self.cache_manager.snapshot_toolcall_anchor(batch.reqs)
        forward_output = self.engine.forward_batch(batch, sample_args)
        self.token_pool[output_mapping] = forward_output.next_tokens_gpu
        if _decode_share() > 0:
            ev1 = torch.cuda.Event(enable_timing=True)
            ev1.record()
            batch.ts_events = (ev0, ev1)
        self.decode_manager.filter_reqs(forward_input.batch.reqs)
        return forward_output


def _make_positions(batch: Batch, device: torch.device) -> torch.Tensor:
    needed_size = sum(r.extend_len for r in batch.padded_reqs)
    indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        torch.arange(
            req.cached_len,
            req.device_len,
            dtype=torch.int32,
            out=indices_host[offset : offset + length],
        )
        offset += length
    return indices_host.to(device, non_blocking=True)


def _make_input_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_host = torch.empty(len(batch.positions), dtype=torch.int64, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        mapping_host[offset : offset + length].fill_(req.table_idx)
        offset += length
    return mapping_host.to(device, non_blocking=True), batch.positions.to(torch.int64)


def _make_write_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_list = [req.table_idx for req in batch.reqs]
    mapping_host = torch.tensor(mapping_list, dtype=torch.int64, pin_memory=True)
    write_list = [(req.device_len if req.can_decode else -1) for req in batch.reqs]
    write_host = torch.tensor(write_list, dtype=torch.int64, pin_memory=True)
    return mapping_host.to(device, non_blocking=True), write_host.to(device, non_blocking=True)
