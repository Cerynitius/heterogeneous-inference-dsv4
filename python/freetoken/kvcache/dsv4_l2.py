"""Host-RAM L2 prefix cache for the DSV4 paged pool (11-19, owner: 「L2 给 64 GB」).

L1 is the VRAM radix cache (SWARadixCache). When an unlocked LIVE leaf -- a finished session's
resumable end -- is dropped from L1, its whole root->leaf path is snapshotted here instead of being
lost: every full page's compressed state (ratio-4 FP8 cmp rows, FP4 indexer rows, ratio-128 cmp
rows) plus the tail's window pages (window KV of every layer + the compressor / indexer state-ring
blocks that carry the resume state). A later request whose prefix contains a whole snapshot gets it
restored into fresh L1 pages and re-inserted into the radix tree (tombstone body + live tail, the
same shape a finished request leaves), so it resumes with a cache hit instead of re-prefilling.

Copies are zero-copy block copies over UVA (kernel/cuda/dsv4_blockcopy.cu) on the current stream,
so stream order keeps them ahead of any reuse of the freed device pages and of later restores.
Storage: one pinned buffer cut into per-array page records ("full" records per 128-token page,
"win" records per window page); snapshots are LRU, a snapshot that is a prefix of a newer one of
the same conversation is dropped. Lookup = hash of the token prefix at each page boundary, then an
exact token compare. Restored KV is the same bytes as the evicted KV (FREETOKEN_DSV4_L2_CHECK=1
verifies per-array checksums on restore).
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernel", "cuda", "dsv4_blockcopy.cu")
_FN = None


def _copy_fn():
    global _FN
    if _FN is None:
        src = open(_SRC, "rb").read()
        tag = hashlib.sha1(src).hexdigest()[:12]
        out_dir = os.path.expanduser("~/.cache/freetoken/cuda")
        os.makedirs(out_dir, exist_ok=True)
        so = os.path.join(out_dir, f"dsv4_blockcopy_{tag}.so")
        if not os.path.exists(so):
            nvcc = os.path.join(os.environ.get("CUDA_HOME", "/usr/local/cuda"), "bin", "nvcc")
            tmp = f"{so}.{os.getpid()}.tmp"
            subprocess.check_call([nvcc, "-O3", "-gencode=arch=compute_120a,code=sm_120a", "-cudart", "shared",
                                   "-shared", "-Xcompiler", "-fPIC", "-o", tmp, _SRC])
            os.replace(tmp, so)
        fn = ctypes.CDLL(so).dsv4_block_copy
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_long, ctypes.c_long, ctypes.c_void_p]
        _FN = fn
    return _FN


def block_copy(src_ptr: int, dst_ptr: int, src_off: torch.Tensor, dst_off: torch.Tensor, block_bytes: int) -> None:
    n = int(src_off.numel())
    if n == 0:
        return
    rc = _copy_fn()(src_ptr, dst_ptr, src_off.data_ptr(), dst_off.data_ptr(), n, int(block_bytes),
                    torch.cuda.current_stream().cuda_stream)
    assert rc == 0, f"dsv4_block_copy failed: {rc}"


@dataclass
class _Array:
    name: str
    dev: torch.Tensor          # contiguous device tensor, row-major
    rows_per_page: int         # device rows per (full or window) page
    row_bytes: int
    host: torch.Tensor = None  # pinned uint8 [n_pages, rec_bytes] (view into the arena)

    @property
    def rec_bytes(self) -> int:
        return self.rows_per_page * self.row_bytes


@dataclass
class _Entry:
    tokens: torch.Tensor       # CPU int64 [L]
    L: int
    full_pages: list           # host full-page ids, one per 128 tokens of [0, L)
    bounds: list               # resume boundaries B (multiples of P, ascending, last == L)
    win_pages: list            # host window-page id of each boundary's last page
    hashes: list               # prefix hash at each boundary
    checks: dict = field(default_factory=dict)
    pinned: bool = False


class DSV4HostL2:
    def __init__(self, pool, budget_gb: float, win_fraction: float = 0.12):
        self.pool = pool
        self.P = int(pool.P)
        self.device = pool._device if hasattr(pool, "_device") else torch.device("cuda")
        P = self.P
        full, win = [], []
        for L, ratio in enumerate(pool.compress_ratios):
            if ratio == 0:
                pass
            elif ratio == 4:
                split = pool.cmp_split[L] if getattr(pool, "cmp_split", None) else None
                if split is not None:
                    assert split[0] == 0, "L2 supports the all-FP8 cmp tier only (CMP_BF16_TOKENS=0)"
                    t = pool.cmp_packed[L]
                else:
                    t = pool.cmp_pool[L]
                full.append(_Array(f"cmp{L}", t, P // 4, t.shape[1] * t.element_size()))
                ip = pool.idx_pool[L]
                if isinstance(ip, torch.Tensor):
                    full.append(_Array(f"idx{L}", ip, P // 4, ip.shape[1] * ip.element_size()))
                else:   # Fp4Rows
                    full.append(_Array(f"idxc{L}", ip.codes, P // 4, 64))
                    full.append(_Array(f"idxe{L}", ip.exps, P // 4, 4))
            else:
                t = pool.cmp_pool[L]
                assert P % ratio == 0
                full.append(_Array(f"cmp{L}", t, P // ratio, t.shape[1] * t.element_size()))
        for L, t in enumerate(pool.window_pool):
            win.append(_Array(f"win{L}", t, P, t.shape[1] * t.element_size()))
        for L in range(len(pool.compress_ratios)):
            for kind, rings in (("ring", pool.state_ring), ("iring", pool.indexer_state_ring)):
                r = rings[L] if rings is not None else None
                if r is not None:
                    b = r.buffer
                    win.append(_Array(f"{kind}{L}", b, r.ring_size, b.shape[1] * b.element_size()))
        for a in full + win:
            assert a.dev.is_contiguous() and a.dev.is_cuda, a.name
            assert a.rec_bytes % 16 == 0 and a.row_bytes % 16 == 0 or (a.rec_bytes % 16 == 0), a.name
        self.full, self.win = full, win
        self.full_rec = sum(a.rec_bytes for a in full)
        self.win_rec = sum(a.rec_bytes for a in win)
        budget = int(budget_gb * (1 << 30))
        self.n_win = max(4, int(budget * win_fraction) // self.win_rec)
        self.n_full = max(16, (budget - self.n_win * self.win_rec) // self.full_rec)
        t0 = time.time()
        total = self.n_full * self.full_rec + self.n_win * self.win_rec
        devs = [d for d in range(torch.cuda.device_count()) if torch.cuda.memory_reserved(d) > 0]
        free0 = [torch.cuda.mem_get_info(d)[0] for d in devs]
        dev = pool.window_pool[0].device.index
        # pinned + UVA-mapped into EVERY GPU's page tables: 64 GiB costs ~128 MiB of device memory
        # per GPU (4 KiB PTEs) -- logged below; one-device cudaHostRegister does not avoid it.
        self.arena = torch.empty(total, dtype=torch.uint8, pin_memory=True)
        torch.cuda.synchronize(dev)
        free1 = [torch.cuda.mem_get_info(d)[0] for d in devs]
        logger.info("DSV4 host L2 device-memory cost: " + ", ".join(
            f"cuda:{d} {(a - b) / 2**20:.0f} MiB" for d, a, b in zip(devs, free0, free1)))
        off = 0
        for a, n in [(a, self.n_full) for a in full] + [(a, self.n_win) for a in win]:
            a.host = self.arena[off:off + n * a.rec_bytes].view(n, a.rec_bytes)
            off += n * a.rec_bytes
        self.free_full = list(range(self.n_full))
        self.free_win = list(range(self.n_win))
        self.entries: "OrderedDict[int, _Entry]" = OrderedDict()   # LRU order (oldest first)
        self.by_end: dict = {}                                     # (L, end_hash) -> entry id
        self._next_id = 0
        self.check = os.environ.get("FREETOKEN_DSV4_L2_CHECK", "0") == "1"
        self.stats = {"saved": 0, "restored": 0, "dropped": 0, "saved_tokens": 0, "restored_tokens": 0}
        logger.info(f"DSV4 host L2: {total / 2**30:.1f} GiB pinned in {time.time() - t0:.1f} s -- "
                    f"{self.n_full} full pages ({self.n_full * P / 1e6:.2f} M tokens, {self.full_rec} B/page), "
                    f"{self.n_win} window pages ({self.win_rec / 2**20:.1f} MiB each)")

    # ------------------------------------------------------------------ hashing
    def _page_hashes(self, tokens: torch.Tensor, n_pages: int) -> list:
        """Chained hash of the token prefix at every page boundary 1..n_pages."""
        P = self.P
        arr = tokens[: n_pages * P].to(torch.int64).numpy()
        out, h = [], 0
        for k in range(n_pages):
            h = hash((h, arr[k * P:(k + 1) * P].tobytes()))
            out.append(h)
        return out

    # ------------------------------------------------------------------ space
    def _drop(self, eid: int) -> None:
        e = self.entries.pop(eid)
        for B, h in zip(e.bounds, e.hashes):
            self.by_end.pop((B, h), None)
        self.free_full.extend(e.full_pages)
        self.free_win.extend(e.win_pages)
        self.stats["dropped"] += 1

    def _reserve(self, n_full: int, n_win: int) -> bool:
        if n_full > self.n_full or n_win > self.n_win:
            return False
        for eid in list(self.entries.keys()):
            if len(self.free_full) >= n_full and len(self.free_win) >= n_win:
                break
            if not self.entries[eid].pinned:
                self._drop(eid)
        return len(self.free_full) >= n_full and len(self.free_win) >= n_win

    # ------------------------------------------------------------------ copies
    def _offsets(self, dev_pages: torch.Tensor, host_pages: list, a: _Array):
        so = dev_pages.to(torch.int64) * a.rec_bytes
        do = torch.tensor(host_pages, dtype=torch.int64, device=dev_pages.device) * a.rec_bytes
        return so, do

    def _d2h(self, arrays, dev_pages: torch.Tensor, host_pages: list, checks: dict) -> None:
        for a in arrays:
            so, do = self._offsets(dev_pages, host_pages, a)
            block_copy(a.dev.data_ptr(), a.host.data_ptr(), so, do, a.rec_bytes)
            if self.check:
                checks[a.name] = self._page_sums(a, dev_pages)

    def _h2d(self, arrays, dev_pages: torch.Tensor, host_pages: list) -> None:
        for a in arrays:
            so, do = self._offsets(dev_pages, host_pages, a)   # so = device offsets, do = host offsets
            block_copy(a.host.data_ptr(), a.dev.data_ptr(), do, so, a.rec_bytes)

    @staticmethod
    def _page_sums(a: _Array, dev_pages: torch.Tensor) -> torch.Tensor:
        """Per-page (plain + position-weighted) byte sums of the device records (check mode)."""
        rows = (dev_pages.to(torch.int64)[:, None] * a.rows_per_page
                + torch.arange(a.rows_per_page, device=dev_pages.device)).view(-1)
        v = a.dev.view(torch.uint8).view(a.dev.shape[0], -1).index_select(0, rows)
        v = v.view(dev_pages.numel(), -1).to(torch.int64)
        w = torch.arange(v.shape[1], device=v.device) % 251 + 1
        return (v.sum(1) * 1000003 + (v * w).sum(1)).cpu()

    # ------------------------------------------------------------------ save
    def save(self, tokens: torch.Tensor, slots: torch.Tensor, bounds: list) -> bool:
        """Snapshot a root->leaf path: tokens / full slots [L] (page aligned) and the resume
        boundaries ``bounds`` (ends of live radix nodes; each boundary's last page holds a live
        window page + compressor carry). Called before the leaf's pages are freed."""
        P = self.P
        L = int(slots.numel())
        bounds = sorted({int(b) for b in bounds if 0 < b <= L and b % P == 0})
        if L == 0 or L % P or not bounds or bounds[-1] != L:
            return False
        n_full = L // P
        tok = tokens[:L].to("cpu", torch.int64)
        hashes = self._page_hashes(tok, n_full)
        hit = self.by_end.get((L, hashes[-1]))
        if hit is not None and torch.equal(self.entries[hit[0]].tokens[:L], tok):
            # already a resume point of a longer snapshot (cascade-evicted parent of a saved leaf)
            self.entries.move_to_end(hit[0])
            return False
        # An older snapshot with a resume point on this path is an earlier state of the same
        # conversation: this one covers it. Its later resume points are a dead branch (the next
        # turn re-renders generated tokens through the chat template, so they never match).
        for k, h in enumerate(hashes):
            hit = self.by_end.get(((k + 1) * P, h))
            if hit is not None and hit[0] in self.entries:
                old = self.entries[hit[0]]
                if not old.pinned and torch.equal(old.tokens[:(k + 1) * P], tok[:(k + 1) * P]):
                    self._drop(hit[0])
        wl = self.pool.full_to_window[slots[torch.tensor([b - P for b in bounds], device=slots.device)].to(torch.int64)]
        keep = (wl >= 0).tolist()
        bounds = [b for b, k in zip(bounds, keep) if k]
        if not bounds or bounds[-1] != L:
            return False
        wl = wl[wl >= 0]
        if not self._reserve(n_full, len(bounds)):
            return False
        dev_full = torch.div(slots[::P].to(torch.int64), P, rounding_mode="floor")
        dev_win = torch.div(wl, P, rounding_mode="floor")
        e = _Entry(tok, L, [self.free_full.pop() for _ in range(n_full)], bounds,
                   [self.free_win.pop() for _ in range(len(bounds))], [hashes[b // P - 1] for b in bounds])
        self._d2h(self.full, dev_full, e.full_pages, e.checks)
        self._d2h(self.win, dev_win, e.win_pages, e.checks)
        eid = self._next_id
        self._next_id += 1
        self.entries[eid] = e
        for jx, (B, h) in enumerate(zip(e.bounds, e.hashes)):
            self.by_end[(B, h)] = (eid, jx)
        self.stats["saved"] += 1
        self.stats["saved_tokens"] += L
        logger.info(f"L2 save: {L} tokens ({n_full} pages, resume points {bounds[-3:]}) -> host; "
                    f"{len(self.entries)} entries, {self.n_full - len(self.free_full)}/{self.n_full} pages used")
        return True

    # ------------------------------------------------------------------ lookup / restore
    def lookup(self, tokens: torch.Tensor, min_len: int):
        """Deepest snapshot resume boundary B with tokens[:B] == snapshot[:B] and B > min_len.
        Returns (entry id, boundary index) or None."""
        if not self.entries:
            return None
        P = self.P
        n = int(tokens.numel()) // P
        if n == 0:
            return None
        tok = tokens.to("cpu", torch.int64)
        hashes = self._page_hashes(tok, n)
        for k in range(n - 1, -1, -1):
            B = (k + 1) * P
            if B <= min_len:
                break
            hit = self.by_end.get((B, hashes[k]))
            if hit is None:
                continue
            e = self.entries[hit[0]]
            if torch.equal(e.tokens[:B], tok[:B]):
                self.entries.move_to_end(hit[0])
                return hit
        return None

    def restore_pages(self, eid: int, jx: int, page_from: int, dev_full: torch.Tensor, dev_win: torch.Tensor) -> None:
        """Copy the snapshot's full pages [page_from, B/P) into device pages ``dev_full`` and the
        window page of boundary ``jx`` into ``dev_win`` (one page)."""
        e = self.entries[eid]
        B = e.bounds[jx]
        self._h2d(self.full, dev_full, e.full_pages[page_from:B // self.P])
        self._h2d(self.win, dev_win, [e.win_pages[jx]])
        if self.check:
            bad = [a.name for a in self.full
                   if not torch.equal(self._page_sums(a, dev_full), e.checks[a.name][page_from:B // self.P])]
            bad += [a.name for a in self.win
                    if not torch.equal(self._page_sums(a, dev_win), e.checks[a.name][jx:jx + 1])]
            assert not bad, f"L2 restore checksum mismatch: {bad[:6]}"
            logger.info(f"L2 restore check OK ({len(self.full)} full + {len(self.win)} window arrays)")
        self.stats["restored"] += 1
        self.stats["restored_tokens"] += B - page_from * self.P
