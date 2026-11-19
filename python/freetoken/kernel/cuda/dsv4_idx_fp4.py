"""ctypes binding of dsv4_idx_fp4.cu (sm_120a native FP4 Lightning-Indexer prefill logits) and
the Triton packer that turns FP4-round-trip bf16 rows into (packed e2m1 bytes, exponent bytes)."""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess

import torch
import triton
import triton.language as tl

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dsv4_idx_fp4.cu")
_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        src = open(_SRC, "rb").read()
        tag = hashlib.sha1(src).hexdigest()[:12]
        out_dir = os.path.expanduser("~/.cache/freetoken/cuda")
        os.makedirs(out_dir, exist_ok=True)
        so = os.path.join(out_dir, f"dsv4_idx_fp4_{tag}.so")
        if not os.path.exists(so):
            nvcc = os.path.join(os.environ.get("CUDA_HOME", "/usr/local/cuda"), "bin", "nvcc")
            tmp = f"{so}.{os.getpid()}.tmp"
            subprocess.check_call([nvcc, "-O3", "-gencode=arch=compute_120a,code=sm_120a", "-cudart", "shared",
                                   "-shared", "-Xcompiler", "-fPIC", "-o", tmp, _SRC])
            os.replace(tmp, so)
        lib = ctypes.CDLL(so)
        fn = lib.dsv4_idx_fp4
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int, ctypes.c_void_p, ctypes.c_long] + \
            [ctypes.c_int] * 7 + [ctypes.c_void_p]
        dfn = lib.dsv4_idx_fp4_decode
        dfn.restype = ctypes.c_int
        dfn.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int, ctypes.c_void_p, ctypes.c_long, ctypes.c_void_p,
                                                ctypes.c_void_p, ctypes.c_long] + [ctypes.c_int] * 4 + [ctypes.c_void_p]
        _LIB = lib
    return _LIB


@triton.jit
def _log2_ceil(v):
    bits = v.to(tl.uint32, bitcast=True)
    e = ((bits >> 23) & 0xFF).to(tl.int32)
    man = (bits & 0x7FFFFF).to(tl.int32)
    return e - 127 + tl.where(man != 0, 1, 0)


@triton.jit
def _e2m1_code(a):
    return tl.where(a < 0.25, 0, tl.where(a < 0.75, 1, tl.where(a < 1.25, 2, tl.where(
        a < 1.75, 3, tl.where(a < 2.5, 4, tl.where(a < 3.5, 5, tl.where(a < 5.0, 6, 7)))))))


@triton.jit
def _pack_kernel(x_ptr, p_ptr, e_ptr, n, stride_x, BLOCK_R: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    rm = rows < n
    offs = tl.arange(0, 16)
    for blk in tl.static_range(4):
        base = x_ptr + rows[:, None].to(tl.int64) * stride_x + blk * 32 + offs[None, :] * 2
        ev = tl.load(base, mask=rm[:, None], other=0.0).to(tl.float32)
        od = tl.load(base + 1, mask=rm[:, None], other=0.0).to(tl.float32)
        amax = tl.maximum(tl.maximum(tl.max(tl.abs(ev), axis=1), tl.max(tl.abs(od), axis=1)),
                          6.0 * (2.0 ** -126))
        e = _log2_ceil(amax * (1.0 / 6.0))
        inv = ((127 - e) << 23).to(tl.float32, bitcast=True)
        ce = _e2m1_code(tl.abs(ev * inv[:, None])) | ((ev.to(tl.uint32, bitcast=True) >> 31) << 3).to(tl.int32)
        co = _e2m1_code(tl.abs(od * inv[:, None])) | ((od.to(tl.uint32, bitcast=True) >> 31) << 3).to(tl.int32)
        tl.store(p_ptr + rows[:, None].to(tl.int64) * 64 + blk * 16 + offs[None, :], (ce | (co << 4)).to(tl.uint8),
                 mask=rm[:, None])
        tl.store(e_ptr + rows.to(tl.int64) * 4 + blk, (e + 127).to(tl.uint8), mask=rm)


def pack_fp4(x: torch.Tensor):
    """[..., 128] FP4-round-trip bf16 -> (packed uint8 [..., 64], ue8m0 exponents uint8 [..., 4])."""
    shp = x.shape
    x2 = x.reshape(-1, 128)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    n = x2.shape[0]
    p = torch.empty((n, 64), dtype=torch.uint8, device=x.device)
    e = torch.empty((n, 4), dtype=torch.uint8, device=x.device)
    if n:
        BR = 64
        _pack_kernel[(triton.cdiv(n, BR),)](x2, p, e, n, x2.stride(0), BLOCK_R=BR, num_warps=4)
    return p.view(*shp[:-1], 64), e.view(*shp[:-1], 4)


def idx_logits_fp4(qp, qe, kp, ke, weights, start_pos: int | None = None, ratio: int = 4,
                   qs: int = 808, keys_per_cta: int = 4096, out_dtype=torch.float32) -> torch.Tensor:
    """qp/qe [S, 64, 64]/[S, 64, 4], kp/ke [T, 64]/[T, 4], weights [S, 64] bf16 -> fp32 [S, T]."""
    S = qp.shape[0]
    T = kp.shape[0]
    # the kernel stores key pairs (float2 / bfloat162): the row stride must be even or odd-T rows
    # are misaligned (11-18 crash: CUDA misaligned address on a short prompt) -> pad the row stride
    Tp = T + (T & 1)
    out = torch.empty((S, Tp), dtype=out_dtype, device=qp.device)[:, :T]
    if S == 0 or T == 0:
        return out
    if out_dtype == torch.bfloat16:
        qs += 10000
    assert weights.dtype == torch.bfloat16 and weights.stride(-1) == 1
    for tt in (qp, qe, kp, ke):
        assert tt.is_contiguous()
    rc = _lib().dsv4_idx_fp4(qp.data_ptr(), qe.data_ptr(), kp.data_ptr(), ke.data_ptr(), weights.data_ptr(),
                             weights.stride(0), out.data_ptr(), out.stride(0), S, T,
                             0 if start_pos is None else start_pos, ratio, int(start_pos is not None),
                             qs, keys_per_cta, torch.cuda.current_stream().cuda_stream)
    assert rc == 0, f"dsv4_idx_fp4 failed: {rc}"
    return out


def idx_decode_fp4(q: torch.Tensor, weights: torch.Tensor, pool, full_snap: torch.Tensor, valid: torch.Tensor,
                   n_stage: int, ratio: int, out: torch.Tensor | None = None, keys_per_cta: int = 512) -> torch.Tensor:
    """Decode Lightning-Indexer scores on FP4 tensor cores: q [B, H, 128] (FP4 round-trip bf16),
    weights [B, H] bf16, pool = fp4_rows.Fp4Rows, full_snap [B, W] int64 decode snapshot,
    valid [B] live block count (device) -> bf16 [B, n_stage] (-inf past valid). Graph-safe."""
    B = q.shape[0]
    qp, qe = pack_fp4(q)
    w = weights if weights.dtype == torch.bfloat16 else weights.to(torch.bfloat16)
    if w.stride(-1) != 1:
        w = w.contiguous()
    v = valid.reshape(B)
    v = v if v.dtype == torch.int32 else v.to(torch.int32)
    snap = full_snap if full_snap.stride(-1) == 1 else full_snap.contiguous()
    if out is None:
        out = torch.empty((B, n_stage), dtype=torch.bfloat16, device=q.device)
    rc = _lib().dsv4_idx_fp4_decode(qp.data_ptr(), qe.data_ptr(), pool.codes.data_ptr(), pool.exps.data_ptr(),
                                    w.data_ptr(), w.stride(0), snap.data_ptr(), snap.stride(0), v.contiguous().data_ptr(),
                                    out.data_ptr(), out.stride(0), B, n_stage, ratio, keys_per_cta,
                                    torch.cuda.current_stream().cuda_stream)
    assert rc == 0, f"dsv4_idx_fp4_decode failed: {rc}"
    return out
