"""Query split of the FP4 indexer prefill top-k across the two EP cards (FREETOKEN_DSV4_IDX_SPLIT).

The last ``n_remote`` queries of an indexer_prefill_select call are scored and top-k'd on the second
card (the PRO 5000) while the first ``seqlen - n_remote`` run on the calling card as before. Every
query goes through the same kernels with the same arguments and the same key bytes, so the picks are
bitwise those of the unsplit loop (each query's scores and top-k are independent of the others).

Memory on the remote card is FIXED: one workspace allocated at first use (keys, one score slice, the
inputs and picks of up to RQMAX remote queries; ~179 MiB at the defaults), nothing allocated per call. The workspace is only
allocated if the remote card keeps >= FREETOKEN_DSV4_IDX_SPLIT_MARGIN_MB free afterwards; otherwise
the split stays off for the process (logged). A call that does not fit runs entirely locally.

Ordering is explicit (events), never left to torch.copy_'s cross-device barrier (absent on a
same-device run). Copies use torch.copy_ with the CURRENT stream of BOTH devices set to private side
streams, so the caller's stream is never blocked by a barrier; it waits once, at the end. All remote
inputs go over in one round before scoring and all picks come back in one copy (no per-group
barriers: torch's two-way copy barrier would serialise any per-group prefetch anyway).

Env (read at import):
  FREETOKEN_DSV4_IDX_SPLIT            share of each call's queries computed remotely, 0 = off (default)
  FREETOKEN_DSV4_IDX_SPLIT_DEV        remote device (default cuda:1)
  FREETOKEN_DSV4_IDX_SPLIT_MIN_WORK   split only when queries x key blocks of the call >= this (default 1.1e8)
  FREETOKEN_DSV4_IDX_SPLIT_MIN_BLOCKS and only when the call scores >= this many key blocks (default 512 = topk)
  FREETOKEN_DSV4_IDX_SPLIT_MAX_BLOCKS workspace key capacity (default DSV4_MAX_SEQ // 4, else 262144)
  FREETOKEN_DSV4_IDX_SPLIT_SLICE      remote score-slice rows (default 256)
  FREETOKEN_DSV4_IDX_SPLIT_RQMAX      remote query capacity per call; extra queries stay local (default 4096)
  FREETOKEN_DSV4_IDX_SPLIT_MARGIN_MB  free MiB the remote card must keep after the workspace (default 768)
"""
from __future__ import annotations

import os

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

FRAC = float(os.environ.get("FREETOKEN_DSV4_IDX_SPLIT", "0") or 0)
DEV = os.environ.get("FREETOKEN_DSV4_IDX_SPLIT_DEV", "cuda:1")
MIN_BLOCKS = int(os.environ.get("FREETOKEN_DSV4_IDX_SPLIT_MIN_BLOCKS", "512"))
# Work gate (measured 11-19 on PRO 6000 + PRO 5000, FREETOKEN_DSV4_EP_N1=124): indexer + top-k on the 6000
# costs c = 22.6 ms per layer for 8192 queries x 122.5K blocks, i.e. c = 2.25e-8 ms per (query x block);
# split at f = 0.4 with remote speed s = 0.674 turns c into ~0.593 c + overhead, the overhead being
# ~0.05 ms sync + keys (68 B/block) + remote q in / picks out (~8.5 KB/remote query) at ~28 GB/s P2P.
# Net gain 0.407 c - overhead > 0 needs c > ~2.5 ms for a full 8K chunk -> queries x blocks >= ~1.1e8
# (a full chunk deeper than ~54K tokens; few new tokens need a deeper context; checked against the
# explicit gain inequality over a grid of query counts and depths).
MIN_WORK = float(os.environ.get("FREETOKEN_DSV4_IDX_SPLIT_MIN_WORK", "1.1e8"))
MAX_BLOCKS = int(os.environ.get("FREETOKEN_DSV4_IDX_SPLIT_MAX_BLOCKS",
                                str(int(os.environ.get("DSV4_MAX_SEQ", "1048576")) // 4)))
RSLICE = int(os.environ.get("FREETOKEN_DSV4_IDX_SPLIT_SLICE", "256"))
RQMAX = int(os.environ.get("FREETOKEN_DSV4_IDX_SPLIT_RQMAX", "4096"))
MARGIN_MB = int(os.environ.get("FREETOKEN_DSV4_IDX_SPLIT_MARGIN_MB", "768"))
_ALLOW_SAME = os.environ.get("FREETOKEN_DSV4_IDX_SPLIT_ALLOW_SAME_DEVICE", "0") == "1"   # one-card tests only

IH, QB, QE = 64, 64, 4          # index heads, packed FP4 bytes per head row, exponent bytes per head row
KB, KE = 64, 4                  # packed key bytes, key exponent bytes
TOPK_W = 512                    # picks width the workspace holds (index_topk)


def n_remote_of(seqlen: int, frac: float = None, cap: int = None) -> int:
    """How many of a call's queries go to the remote card (the LAST ones), at most the capacity."""
    f = FRAC if frac is None else frac
    return max(0, min(seqlen, int(round(f * seqlen)), RQMAX if cap is None else cap))


def plan(seqlen: int, n_remote: int, group: int, rslice: int):
    """Pure index plan. Local queries [0, k) keep the caller's slicing; remote queries [k, seqlen)
    are processed in groups (one copy round each) and score slices inside a group.
    Returns (k, [(g0, g1, [(s0, s1), ...]), ...]) with absolute query indices."""
    k = seqlen - n_remote
    groups = []
    for g0 in range(k, seqlen, group):
        g1 = min(g0 + group, seqlen)
        groups.append((g0, g1, [(s0, min(s0 + rslice, g1)) for s0 in range(g0, g1, rslice)]))
    return k, groups


class _Workspace:
    def __init__(self, dev1: torch.device):
        self.dev = dev1
        self.tmax = MAX_BLOCKS
        tp = self.tmax + (self.tmax & 1)                       # even row stride, as idx_logits_fp4 pads
        self.kp = torch.empty((self.tmax, KB), dtype=torch.uint8, device=dev1)
        self.ke = torch.empty((self.tmax, KE), dtype=torch.uint8, device=dev1)
        self.qp = torch.empty((RQMAX, IH, QB), dtype=torch.uint8, device=dev1)
        self.qe = torch.empty((RQMAX, IH, QE), dtype=torch.uint8, device=dev1)
        self.w = torch.empty((RQMAX, IH), dtype=torch.bfloat16, device=dev1)
        self.picks = torch.empty((RQMAX, TOPK_W), dtype=torch.int64, device=dev1)
        self.scores = torch.empty((RSLICE, tp), dtype=torch.bfloat16, device=dev1)
        ts = [self.kp, self.ke, self.scores, self.qp, self.qe, self.w, self.picks]
        self.nbytes = sum(t.numel() * t.element_size() for t in ts)
        self.rs = torch.cuda.Stream(dev1)


_state = {"ws": None, "side0": {}, "off": None}


def _release(dev1: torch.device) -> None:
    """Hand a refused workspace's cached blocks back to the driver (one-time, disable path only)."""
    with torch.cuda.device(dev1):
        torch.cuda.empty_cache()


def _workspace(dev0: torch.device):
    """Lazily allocate the fixed workspace; None (and split off for good) if it does not fit safely."""
    if _state["off"] is not None:
        return None
    if _state["ws"] is not None:
        return _state["ws"]
    dev1 = torch.device(DEV)
    if dev1.index is None:
        dev1 = torch.device(dev1.type, 0)
    if dev1.type != "cuda" or dev1.index >= torch.cuda.device_count() or (dev1 == dev0 and not _ALLOW_SAME):
        _state["off"] = f"remote device {DEV} unavailable (calling device {dev0}, {torch.cuda.device_count()} visible)"
        logger.warning("indexer split disabled: %s", _state["off"])
        return None
    try:
        ws = _Workspace(dev1)
    except torch.cuda.OutOfMemoryError as exc:
        _state["off"] = f"workspace allocation failed on {dev1} (OutOfMemoryError: {exc})"
        logger.warning("indexer split disabled: %s", _state["off"])
        _release(dev1)
        return None
    free, total = torch.cuda.mem_get_info(dev1)
    if free < MARGIN_MB * 2**20:
        _state["off"] = (f"{dev1} would keep {free / 2**20:.0f} MiB free after the {ws.nbytes / 2**20:.1f} MiB "
                         f"workspace, below the {MARGIN_MB} MiB margin")
        del ws
        _release(dev1)
        logger.warning("indexer split disabled: %s", _state["off"])
        return None
    _state["ws"] = ws
    logger.info("indexer split on: %.2f of each call's queries on %s (workspace %.1f MiB fixed, %.0f MiB free "
                "after; slice %d, remote capacity %d, min work %.3g, min blocks %d, max blocks %d)", FRAC, dev1,
                ws.nbytes / 2**20, free / 2**20, RSLICE, RQMAX, MIN_WORK, MIN_BLOCKS, ws.tmax)
    return ws


def wanted(n_blocks: int, seqlen: int) -> bool:
    return (FRAC > 0 and n_blocks >= max(MIN_BLOCKS, TOPK_W) and seqlen * n_blocks >= MIN_WORK
            and n_remote_of(seqlen) > 0)


def split_select(qp4, qe4, kp4, ke4, w4, *, start_pos: int, seqlen: int, ratio: int, topk: int, offset: int,
                 qs: int, local_slice: int):
    """== the production FP4 + CUDA top-k slice loop over all ``seqlen`` queries ([seqlen, min(topk, T)]
    int64 on the calling device), with the last n_remote queries computed on the remote card.
    Returns None when the split cannot run (caller falls back to its own loop)."""
    from freetoken.kernel.cuda import dsv4_idx_fp4 as _idx
    from freetoken.kernel.cuda import dsv4_topk as _tk

    dev0 = qp4.device
    T = kp4.shape[0]
    ws = _workspace(dev0)
    if ws is None or T > ws.tmax or T == 0:
        return None
    K = min(topk, T)
    if K != TOPK_W:                     # narrower picks would make the copy-back non-contiguous (a temp
        return None                     # allocation on the remote card); MIN_BLOCKS >= topk avoids this
    k, groups = plan(seqlen, n_remote_of(seqlen), RQMAX, RSLICE)   # one group: all remote queries
    if not groups:
        return None
    for t in (qp4, qe4, kp4, ke4):
        assert t.is_contiguous()
    assert w4.dtype == torch.bfloat16 and w4.stride(-1) == 1
    out = torch.empty((seqlen, K), dtype=torch.int64, device=dev0)
    main = torch.cuda.current_stream(dev0)
    side0 = _state["side0"].get(dev0)
    if side0 is None:
        side0 = _state["side0"][dev0] = torch.cuda.Stream(dev0)
    rs = ws.rs
    qs_bf16 = qs + 10000                              # idx_logits_fp4(out_dtype=bf16) encoding
    tp = ws.scores.stride(0)

    assert len(groups) == 1
    g0, g1, slices = groups[0]
    n = g1 - g0
    ready = torch.cuda.Event()                        # inputs exist on the caller's stream
    ready.record(main)
    side0.wait_event(ready)
    rs.wait_event(ready)
    side0.wait_stream(rs)                             # the previous call no longer reads the workspace
    # current stream on BOTH devices (+ explicit current device for the raw ctypes launches)
    with torch.cuda.stream(side0), torch.cuda.stream(rs), torch.cuda.device(ws.dev):
        ws.kp[:T].copy_(kp4)                          # cross-device: runs on side0; same device: on rs
        ws.ke[:T].copy_(ke4)
        ws.qp[:n].copy_(qp4[g0:g1])
        ws.qe[:n].copy_(qe4[g0:g1])
        ws.w[:n].copy_(w4[g0:g1])
        landed = torch.cuda.Event()
        landed.record(side0)
        rs.wait_event(landed)                         # all inputs on the remote card before scoring
        for s0, s1 in slices:
            m, r0, r1 = s1 - s0, s0 - g0, s1 - g0
            sc = ws.scores[:m]
            rc = _idx._lib().dsv4_idx_fp4(
                ws.qp[r0:r1].data_ptr(), ws.qe[r0:r1].data_ptr(), ws.kp.data_ptr(), ws.ke.data_ptr(),
                ws.w[r0:r1].data_ptr(), ws.w.stride(0), sc.data_ptr(), tp, m, T,
                start_pos + s0, ratio, 1, qs_bf16, 4096, rs.cuda_stream)
            assert rc == 0, f"dsv4_idx_fp4 failed: {rc}"
            rc = _tk._fn()(sc.data_ptr(), tp, m, T, K, ws.picks[r0:r1].data_ptr(), ws.picks.stride(0),
                           int(start_pos + s0), int(ratio), int(offset), rs.cuda_stream, None, 0)
            assert rc == 0, f"dsv4_topk_bf16 failed: {rc}"
        out[g0:g1].copy_(ws.picks[:n])                # runs on rs (the source device's current stream)
    for s0 in range(0, k, local_slice):               # local queries: the production loop, unchanged
        e = min(s0 + local_slice, k)
        sc = _idx.idx_logits_fp4(qp4[s0:e].contiguous(), qe4[s0:e].contiguous(), kp4, ke4, w4[s0:e],
                                 start_pos + s0, ratio, qs, 4096, out_dtype=torch.bfloat16)
        out[s0:e] = _tk.topk_picks(sc, topk, start_pos + s0, ratio, offset)
        del sc
    # the caller's stream continues only after every remote kernel and copy has finished
    # (keeps the inputs alive until read, the workspace free for the next call, the picks complete)
    done_rs, done_s0 = torch.cuda.Event(), torch.cuda.Event()
    done_rs.record(rs)
    done_s0.record(side0)
    main.wait_event(done_rs)
    main.wait_event(done_s0)
    return out
