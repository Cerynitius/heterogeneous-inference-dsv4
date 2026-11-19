"""One-launch assembly of a decode step's sparse-attention index list for indexer layers.

Replaces, bit for bit (integer ops only), the chain
    blocks = where(picks >= valid, -1, picks + offset)                       (indexer_select_decode)
    full_at = snapshot[rows, clamp_min(blocks, 0) * ratio]                   (blocks_to_global)
    g = where(blocks < 0, -1, floor_div(full_at, ratio))
    topk_idxs = cat([window_slots_topk, g], -1).int()
which ran as ~12 tiny torch kernels per indexer layer per decode step.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _idx_assemble_kernel(
    win_ptr, picks_ptr, valid_ptr, snap_ptr, rows_ptr, out_ptr,
    W, KP, offset, ratio,
    stride_win_b, stride_pk_b, stride_snap_r, stride_out_b,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    j = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = j < W + KP
    is_win = j < W
    wv = tl.load(win_ptr + b * stride_win_b + j, mask=m & is_win, other=0).to(tl.int64)
    p = tl.load(picks_ptr + b * stride_pk_b + (j - W), mask=m & (j >= W), other=0).to(tl.int64)
    valid = tl.load(valid_ptr + b).to(tl.int64)
    blk = tl.where(p >= valid, -1, p + offset)
    safe = tl.maximum(blk, 0)
    row = tl.load(rows_ptr + b).to(tl.int64)
    fa = tl.load(snap_ptr + row * stride_snap_r + safe * ratio, mask=m & (j >= W), other=0).to(tl.int64)
    q = fa // ratio                                   # truncates toward zero ...
    q = tl.where((fa % ratio != 0) & (fa < 0), q - 1, q)  # ... -> floor (torch rounding_mode="floor")
    g = tl.where(blk < 0, -1, q)
    v = tl.where(is_win, wv, g)
    tl.store(out_ptr + b * stride_out_b + j, v.to(tl.int32), mask=m)


def idx_assemble(window_slots_topk, picks, valid, snapshot, rows, ratio: int, offset: int = 0):
    """``window_slots_topk`` [B, 1, W], ``picks`` [B, 1, k] raw top-k column indices,
    ``valid`` [B], ``snapshot`` [R, L] int64 full locs, ``rows`` [B] -> [B, 1, W + k] int32."""
    B = picks.shape[0]
    W = window_slots_topk.shape[-1]
    KP = picks.shape[-1]
    assert window_slots_topk.stride(-1) == 1 and picks.stride(-1) == 1 and snapshot.stride(-1) == 1
    out = torch.empty((B, 1, W + KP), dtype=torch.int32, device=picks.device)
    BLOCK = 256
    _idx_assemble_kernel[(B, triton.cdiv(W + KP, BLOCK))](
        window_slots_topk, picks, valid, snapshot, rows, out,
        W, KP, offset, ratio,
        window_slots_topk.stride(0), picks.stride(0), snapshot.stride(0), out.stride(0),
        BLOCK=BLOCK,
    )
    return out
