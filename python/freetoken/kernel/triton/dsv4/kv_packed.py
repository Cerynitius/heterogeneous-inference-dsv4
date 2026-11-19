"""Lossless FP8 storage tier for the DSV4 ratio-4 compressed-KV pools.

Every compressed-KV row is already an FP8 round-trip on its first 448 dims
(``act_quant_fp8_inplace(kv[..., :-64], 64)``: e4m3 x 2^e per 64-block, ue8m0 exponent) plus 64
bf16 rope dims, so bf16 storage carries redundant bytes. A packed row stores the same values:

    [0,448) e4m3 codes | [448,576) rope bf16 | [576,583) exponent bytes (e+127) | pad   = 592 B

The exponent is recomputed from the stored bf16 values (e' = ceil(log2(amax/448)) <= the
quantizer's e), and q*2^(e-e') <= 448 is again an e4m3 value, so unpack(pack(v)) == v bit for
bit (FREETOKEN_DSV4_PACK_CHECK=1 verifies every write).

Layout per ratio-4 layer (kvcache/dsv4_paged_pool.py): logical rows [0, G) live in the plain bf16
tensor the attention kernel reads; rows >= G live packed (VRAM first, then host RAM). The bf16
tensor carries ``stage`` extra rows after G: decode copies the few packed rows a step's top-k
selected into them (``decode_stage``) and remaps those columns, so the unchanged attention kernel
sees identical values. Prefill gathers through ``gather_rows``.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

KV_D = 512
KV_ROW = 592          # 37 x 16 B

_CHECK = os.environ.get("FREETOKEN_DSV4_PACK_CHECK", "0") == "1"
_bad: torch.Tensor | None = None


def fp8_tier_enabled() -> bool:
    return os.environ.get("FREETOKEN_DSV4_CMP_FP8", "0") == "1"


def _bad_counter(device) -> torch.Tensor:
    global _bad
    if _bad is None or _bad.device != device:
        _bad = torch.zeros(1, dtype=torch.int32, device=device)
    return _bad


def pack_mismatches() -> int:
    return 0 if _bad is None else int(_bad.item())


@triton.jit
def _log2_ceil(v):
    bits = v.to(tl.uint32, bitcast=True)
    e = ((bits >> 23) & 0xFF).to(tl.int32)
    man = (bits & 0x7FFFFF).to(tl.int32)
    return e - 127 + tl.where(man != 0, 1, 0)


@triton.jit
def _pow2(e):
    return ((e + 127).to(tl.int32) << 23).to(tl.float32, bitcast=True)


@triton.jit
def _pack_row(src, dst, bad_ptr, CHECK: tl.constexpr):
    """bf16 row (512) at ``src`` -> packed row (592 B) at ``dst``."""
    offs = tl.arange(0, 64)
    for blk in tl.static_range(7):
        x = tl.load(src + blk * 64 + offs).to(tl.float32)
        amax = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-4)
        e = _log2_ceil(amax * (1.0 / 448.0))
        q = (x * _pow2(-e)).to(tl.float8e4nv)
        tl.store(dst + blk * 64 + offs, q.to(tl.uint8, bitcast=True))
        tl.store(dst + 576 + blk, (e + 127).to(tl.uint8))
        if CHECK:
            back = q.to(tl.float32) * _pow2(e)
            nbad = tl.sum((back.to(tl.uint32, bitcast=True) != x.to(tl.uint32, bitcast=True)
                           ).to(tl.int32), axis=0)
            tl.atomic_add(bad_ptr, nbad)
    r = tl.load(src + 448 + offs)
    tl.store((dst + 448).to(tl.pointer_type(tl.bfloat16), bitcast=True) + offs, r)


@triton.jit
def _copy_unpacked_vram(src_u8, dst_bf16):
    """packed row (VRAM) -> the identical bf16 row: per-element loads, L1-served."""
    d = tl.arange(0, 512)
    nope = d < 448
    q = tl.load(src_u8 + d, mask=nope, other=0)
    code = tl.load(src_u8 + 576 + d // 64, mask=nope, other=127)
    v = q.to(tl.float8e4nv, bitcast=True).to(tl.float32) * (code.to(tl.int32) << 23).to(
        tl.float32, bitcast=True)
    rp = (src_u8 + 448).to(tl.pointer_type(tl.bfloat16), bitcast=True)
    r = tl.load(rp + (d - 448), mask=d >= 448, other=0.0)
    tl.store(dst_bf16 + d, tl.where(nope, v.to(tl.bfloat16), r))


@triton.jit
def _copy_unpacked_burst(src_u8, dst_bf16):
    """packed row (host RAM) -> the identical bf16 row."""
    # ONE contiguous 592-byte read (a host-RAM row is one PCIe burst, not 512 scattered byte
    # reads); the exponent bytes and rope halves are picked out in registers.
    b = tl.arange(0, 1024)
    raw = tl.load(src_u8 + b, mask=b < 592, other=0).to(tl.int32)        # [1024]
    d = tl.arange(0, 512)
    nope = d < 448
    q = tl.gather(raw, d, 0)                                              # e4m3 code (d < 448)
    code = tl.gather(raw, tl.where(nope, 576 + d // 64, 0), 0)
    v = q.to(tl.uint8).to(tl.float8e4nv, bitcast=True).to(tl.float32) * (code << 23).to(
        tl.float32, bitcast=True)
    rb = tl.where(nope, 0, 448 + 2 * (d - 448))
    rbits = tl.gather(raw, rb, 0) | (tl.gather(raw, rb + 1, 0) << 8)
    r = rbits.to(tl.int16).to(tl.bfloat16, bitcast=True)
    tl.store(dst_bf16 + d, tl.where(nope, v.to(tl.bfloat16), r))


@triton.jit
def _copy_unpacked(packed_ptr, prow, dst_bf16, N_VRAM):
    src = packed_ptr + prow.to(tl.int64) * 592
    if prow < N_VRAM:
        _copy_unpacked_vram(src, dst_bf16)
    else:
        _copy_unpacked_burst(src, dst_bf16)


@triton.jit
def _store_rows_kernel(src_ptr, stride_s, rows_ptr, dense_ptr, packed_ptr, bad_ptr, G,
                       CHECK: tl.constexpr):
    pid = tl.program_id(0)
    row = tl.load(rows_ptr + pid).to(tl.int64)
    src = src_ptr + pid.to(tl.int64) * stride_s
    if row < G:
        d = tl.arange(0, 512)
        tl.store(dense_ptr + row * 512 + d, tl.load(src + d))
    else:
        _pack_row(src, packed_ptr + (row - G) * 592, bad_ptr, CHECK)


def store_rows(dense: torch.Tensor, packed: torch.Tensor, G: int, rows: torch.Tensor,
               src: torch.Tensor) -> None:
    """Logical ``pool[rows] = src``: rows < G into ``dense`` (bf16), others packed."""
    n = rows.numel()
    if n == 0:
        return
    src = src.reshape(n, -1)
    if src.dtype != torch.bfloat16:
        src = src.to(torch.bfloat16)
    if src.stride(-1) != 1:
        src = src.contiguous()
    assert src.shape[1] == KV_D and dense.stride(0) == KV_D, (src.shape, dense.stride())
    _store_rows_kernel[(n,)](src, src.stride(0), rows.reshape(-1), dense, packed,
                             _bad_counter(dense.device), G, CHECK=_CHECK, num_warps=2)


@triton.jit
def _gather_rows_kernel(rows_ptr, dense_ptr, packed_ptr, out_ptr, G, N_VRAM):
    pid = tl.program_id(0)
    row = tl.load(rows_ptr + pid).to(tl.int64)
    dst = out_ptr + pid.to(tl.int64) * 512
    if row < G:
        d = tl.arange(0, 512)
        tl.store(dst + d, tl.load(dense_ptr + row * 512 + d))
    else:
        _copy_unpacked(packed_ptr, row - G, dst, N_VRAM)


def gather_rows(dense: torch.Tensor, packed: torch.Tensor, G: int, rows: torch.Tensor,
                n_vram: int | None = None) -> torch.Tensor:
    """bf16 ``[n, 512]`` equal to the logical ``pool[rows]``."""
    rows = rows.reshape(-1)
    out = torch.empty(rows.numel(), KV_D, dtype=torch.bfloat16, device=dense.device)
    if rows.numel():
        nv = packed.shape[0] if n_vram is None else n_vram
        _gather_rows_kernel[(rows.numel(),)](rows, dense, packed, out, G, nv, num_warps=2)
    return out


@triton.jit
def _decode_stage_kernel(idx_ptr, out_ptr, stride_r, dense_ptr, packed_ptr, G, N_PACKED, S0,
                         N_WINDOW, TOPK_C, N_VRAM):
    r = tl.program_id(0)
    col = tl.program_id(1)
    v = tl.load(idx_ptr + r * stride_r + col)
    res = v
    if col >= N_WINDOW:
        if (v >= G) & (v < G + N_PACKED):
            slot = S0 + r * TOPK_C + (col - N_WINDOW)
            _copy_unpacked(packed_ptr, v - G, dense_ptr + slot.to(tl.int64) * 512, N_VRAM)
            res = slot
    tl.store(out_ptr + r * stride_r + col, res)


def decode_stage(topk_idxs: torch.Tensor, n_window: int, dense: torch.Tensor,
                 packed: torch.Tensor, G: int, stage_rows: int, n_vram: int | None = None) -> torch.Tensor:
    """Graph-safe: copies each selected packed row into ``dense[G + r*TOPK_C + j]`` and returns
    the remapped indices (dense rows, window slots and -1 pass through unchanged)."""
    shp = topk_idxs.shape
    idx = topk_idxs.reshape(-1, shp[-1]).to(torch.int32).contiguous()
    R, topk = idx.shape
    topk_c = topk - n_window
    assert R * topk_c <= stage_rows, (R, topk_c, stage_rows)
    out = torch.empty_like(idx)
    nv = packed.shape[0] if n_vram is None else n_vram
    _decode_stage_kernel[(R, topk)](idx, out, idx.stride(0), dense, packed, G, packed.shape[0], G,
                                    n_window, topk_c, nv, num_warps=2)
    return out.view(shp)


# ---- prefill staging without a sort: bitmap -> cumsum -> compact -> gather, one remap pass ----
@triton.jit
def _mark_kernel(c_ptr, n, stride_row, NW, TOPK, map_ptr, N_ROWS, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    TC = TOPK - NW
    r = offs // TC
    j = offs % TC
    m = offs < n
    v = tl.load(c_ptr + r.to(tl.int64) * stride_row + NW + j, mask=m, other=-1)
    ok = m & (v >= 0) & (v < N_ROWS)
    tl.store(map_ptr + v, tl.full((BLOCK,), 1, tl.int32), mask=ok)


@triton.jit
def _compact_kernel(map_ptr, pos_ptr, rows_ptr, N_ROWS, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < N_ROWS
    hit = tl.load(map_ptr + offs, mask=m, other=0)
    p = tl.load(pos_ptr + offs, mask=m, other=0)
    tl.store(rows_ptr + (p - 1), offs, mask=m & (hit != 0))


@triton.jit
def _remap_kernel(idx_ptr, out_ptr, n, stride_row, NW, TOPK, pos_ptr, N_ROWS, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    r = offs // TOPK
    j = offs % TOPK
    v = tl.load(idx_ptr + r.to(tl.int64) * stride_row + j, mask=m, other=-1)
    is_c = j >= NW
    ok = is_c & (v >= 0) & (v < N_ROWS)
    p = tl.load(pos_ptr + v, mask=m & ok, other=0)
    res = tl.where(is_c, tl.where(ok, p - 1, -1), v)
    tl.store(out_ptr + offs, res, mask=m)


_map_bufs: dict = {}


def prefill_stage(topk_idxs: torch.Tensor, n_window: int, dense: torch.Tensor, packed: torch.Tensor,
                  G: int, n_vram: int):
    """Stage every logical compressed row the prefill top-k references into a bf16 slab (dense
    rows copied, packed rows decoded) and remap. Returns ``(slab, idx2)``; values identical."""
    shp = topk_idxs.shape
    idx = topk_idxs.reshape(-1, shp[-1]).to(torch.int32).contiguous()
    R, topk = idx.shape
    n_rows = G + packed.shape[0]
    key = (dense.device, n_rows)
    mp = _map_bufs.get(key)
    if mp is None:
        _map_bufs.clear()
        mp = _map_bufs[key] = torch.empty(n_rows, dtype=torch.int32, device=dense.device)
    mp.zero_()
    BLOCK = 1024
    nc = R * (topk - n_window)
    if nc:
        _mark_kernel[(triton.cdiv(nc, BLOCK),)](idx, nc, idx.stride(0), n_window, topk, mp, n_rows,
                                                BLOCK=BLOCK, num_warps=4)
    pos = torch.cumsum(mp, 0, dtype=torch.int32)
    nnz = int(pos[-1].item())
    rows = torch.empty(nnz, dtype=torch.int32, device=dense.device)
    if nnz:
        _compact_kernel[(triton.cdiv(n_rows, BLOCK),)](mp, pos, rows, n_rows, BLOCK=BLOCK, num_warps=4)
    slab = gather_rows(dense, packed, G, rows, n_vram)
    out = torch.empty_like(idx)
    n = R * topk
    _remap_kernel[(triton.cdiv(n, BLOCK),)](idx, out, n, idx.stride(0), n_window, topk, pos, n_rows,
                                            BLOCK=BLOCK, num_warps=4)
    return slab, out.view(shp)
