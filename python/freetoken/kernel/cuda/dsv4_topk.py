"""ctypes binding of dsv4_topk.cu: deterministic row-wise top-k over bf16 indexer scores (radix select),
emitting the indexer's final picks (col >= live -> -1, else col + offset), in column order."""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess

import torch

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dsv4_topk.cu")
_FN = None


def _fn():
    global _FN
    if _FN is None:
        src = open(_SRC, "rb").read()
        tag = hashlib.sha1(src).hexdigest()[:12]
        out_dir = os.path.expanduser("~/.cache/freetoken/cuda")
        os.makedirs(out_dir, exist_ok=True)
        so = os.path.join(out_dir, f"dsv4_topk_{tag}.so")
        if not os.path.exists(so):
            nvcc = os.path.join(os.environ.get("CUDA_HOME", "/usr/local/cuda"), "bin", "nvcc")
            tmp = f"{so}.{os.getpid()}.tmp"
            subprocess.check_call([nvcc, "-O3", "-gencode=arch=compute_120a,code=sm_120a", "-cudart", "shared",
                                   "-shared", "-Xcompiler", "-fPIC", "-o", tmp, _SRC])
            os.replace(tmp, so)
        fn = ctypes.CDLL(so).dsv4_topk_bf16
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.c_void_p, ctypes.c_long, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                       ctypes.c_long, ctypes.c_int, ctypes.c_int, ctypes.c_long, ctypes.c_void_p,
                       ctypes.c_void_p, ctypes.c_int]
        lib = ctypes.CDLL(so)
        f2 = lib.dsv4_topk_dec2
        f2.restype = ctypes.c_int
        f2.argtypes = [ctypes.c_void_p, ctypes.c_long, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                       ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long, ctypes.c_int,
                       ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        fn.dec2 = f2
        _FN = fn
    return _FN


def topk_picks(scores: torch.Tensor, k: int, start_pos: int, ratio: int, offset: int) -> torch.Tensor:
    """scores [S, T] bf16 (row stride free, unit column stride) -> int64 [S, min(k, T)]:
    the top-k columns of each row in column order; col >= (start_pos + row + 1) // ratio -> -1."""
    S, T = scores.shape
    assert scores.dtype == torch.bfloat16 and scores.stride(1) == 1
    K = min(k, T)
    out = torch.empty((S, K), dtype=torch.int64, device=scores.device)
    rc = _fn()(scores.data_ptr(), scores.stride(0), S, T, K, out.data_ptr(), out.stride(0), int(start_pos),
               int(ratio), int(offset), torch.cuda.current_stream().cuda_stream, None, 0)
    assert rc == 0, f"dsv4_topk_bf16 failed: {rc}"
    return out


def topk_decode(scores: torch.Tensor, valid: torch.Tensor, k: int, raw: bool, offset: int = 0) -> torch.Tensor:
    """Decode-step top-k: scores [B, n_stage] bf16, valid [B] int32 live column count (device);
    only max(valid, k) columns are scanned (graph-safe). raw: the column ids as they are (the
    caller masks past ``valid``); else the indexer's final form (-1 past valid, + offset).
    Column order, not score order."""
    B, T = scores.shape
    assert scores.dtype == torch.bfloat16 and scores.stride(1) == 1
    K = min(k, T)
    v = valid if valid.dtype == torch.int32 else valid.to(torch.int32)
    out = torch.empty((B, K), dtype=torch.int64, device=scores.device)
    rc = _fn()(scores.data_ptr(), scores.stride(0), B, T, K, out.data_ptr(), out.stride(0), 0, 1,
               int(offset), torch.cuda.current_stream().cuda_stream, v.contiguous().data_ptr(), int(bool(raw)))
    assert rc == 0, f"dsv4_topk_bf16 failed: {rc}"
    return out


def topk_decode2(scores: torch.Tensor, valid: torch.Tensor, k: int, raw: bool, offset: int = 0) -> torch.Tensor:
    """Multi-CTA decode top-k (4096-column chunks -> local top-k -> global top-k), live columns only.
    Same selected set as torch.topk; column order. Graph-safe. Requires n_stage <= 64 * 4096."""
    B, T = scores.shape
    assert scores.dtype == torch.bfloat16 and scores.stride(1) == 1 and T <= 64 * 4096
    K = min(k, T)
    nch = (T + 4095) // 4096
    v = valid.reshape(B)
    v = (v if v.dtype == torch.int32 else v.to(torch.int32)).contiguous()
    cv = torch.empty((B, nch, K), dtype=torch.int16, device=scores.device)
    cc = torch.empty((B, nch, K), dtype=torch.int32, device=scores.device)
    cn = torch.empty((B, nch), dtype=torch.int32, device=scores.device)
    out = torch.empty((B, K), dtype=torch.int64, device=scores.device)
    ng = (nch + 7) // 8
    mv = torch.empty((B, ng, K), dtype=torch.int16, device=scores.device)
    mc = torch.empty((B, ng, K), dtype=torch.int32, device=scores.device)
    mn = torch.empty((B, ng), dtype=torch.int32, device=scores.device)
    rc = _fn().dec2(scores.data_ptr(), scores.stride(0), B, T, K, v.data_ptr(), cv.data_ptr(), cc.data_ptr(),
                    cn.data_ptr(), out.data_ptr(), out.stride(0), int(bool(raw)), int(offset),
                    torch.cuda.current_stream().cuda_stream, mv.data_ptr(), mc.data_ptr(), mn.data_ptr())
    assert rc == 0, f"dsv4_topk_dec2 failed: {rc}"
    return out
