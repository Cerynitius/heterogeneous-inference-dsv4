"""ctypes binding of dsv4_sparse_mla.cu: sm_120a sparse MLA prefill attention, 64 heads per CTA
(bf16 mma.sync, fp32 accumulate, P in bf16 like the Triton BF16 path)."""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess

import torch

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dsv4_sparse_mla.cu")
_FN = None


def _fn():
    global _FN
    if _FN is None:
        src = open(_SRC, "rb").read()
        tag = hashlib.sha1(src).hexdigest()[:12]
        out_dir = os.path.expanduser("~/.cache/freetoken/cuda")
        os.makedirs(out_dir, exist_ok=True)
        so = os.path.join(out_dir, f"dsv4_sparse_mla_{tag}.so")
        if not os.path.exists(so):
            nvcc = os.path.join(os.environ.get("CUDA_HOME", "/usr/local/cuda"), "bin", "nvcc")
            tmp = f"{so}.{os.getpid()}.tmp"
            subprocess.check_call([nvcc, "-O3", "-gencode=arch=compute_120a,code=sm_120a", "-cudart", "shared",
                                   "-shared", "-Xcompiler", "-fPIC", "-o", tmp, _SRC])
            os.replace(tmp, so)
        fn = ctypes.CDLL(so).dsv4_sparse_mla
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.c_void_p, ctypes.c_long, ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long, ctypes.c_int, ctypes.c_int,
                       ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long, ctypes.c_long, ctypes.c_int, ctypes.c_float,
                       ctypes.c_void_p, ctypes.c_int]
        lib = ctypes.CDLL(so)
        sp = lib.dsv4_sparse_mla_splitk
        sp.restype = ctypes.c_int
        sp.argtypes = [ctypes.c_void_p, ctypes.c_long, ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long, ctypes.c_int, ctypes.c_int,
                       ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_float,
                       ctypes.c_void_p]
        assert lib.dsv4_sparse_mla_init() == 0   # smem opt-in now, never during a graph capture
        fn.splitk = sp
        sh = lib.dsv4_sparse_mla_shared
        sh.restype = ctypes.c_int
        sh.argtypes = [ctypes.c_void_p, ctypes.c_long, ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long, ctypes.c_int, ctypes.c_int,
                       ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long, ctypes.c_long, ctypes.c_int,
                       ctypes.c_float, ctypes.c_void_p]
        fn.shared = sh
        _FN = fn
    return _FN


def supported(q: torch.Tensor) -> bool:
    return q.shape[-1] == 512 and q.shape[-2] == 64 and q.dtype == torch.bfloat16


def sparse_mla_prefill(q, window_pool, cmp_pool, sink, idx, n_window: int, scale: float, cnt=None, out=None):
    """q [b, m, 64, 512] bf16 contiguous; pools [*, 512] bf16 sharing strides; sink [64] fp32;
    idx [b, m, topk] int32 contiguous ([window | compressed] global slots, -1 = masked);
    cnt [b, m] int32 live compressed count or None. Returns [b, m, 64, 512] bf16."""
    b, m, h, d = q.shape
    assert q.is_contiguous() and idx.is_contiguous() and window_pool.stride() == cmp_pool.stride()
    assert sink.dtype == torch.float32 and idx.dtype == torch.int32
    if out is None:
        out = torch.empty_like(q)
    rc = _fn()(q.data_ptr(), q.stride(1), q.stride(2), window_pool.data_ptr(), cmp_pool.data_ptr(),
               window_pool.stride(0), sink.data_ptr(), idx.data_ptr(), idx.stride(1), idx.shape[-1], int(n_window),
               0 if cnt is None else cnt.data_ptr(), out.data_ptr(), out.stride(1), out.stride(2), b * m,
               float(scale), torch.cuda.current_stream().cuda_stream, 0)
    assert rc == 0, f"dsv4_sparse_mla failed: {rc}"
    return out


def sparse_mla_splitk(q, window_pool, cmp_pool, sink, idx, n_window: int, scale: float, cnt, mid_o, mid_lse,
                      n_splits: int) -> None:
    """Decode split-K stage 1 into the Triton split-K layout (mid_o [b, m, h, S, d] fp32 normalized
    partials, mid_lse [b, m, h, S] natural-log lse); _sparse_attn_splitk_merge_kernel finishes."""
    b, m, h, d = q.shape
    assert q.is_contiguous() and idx.is_contiguous() and mid_o.is_contiguous() and mid_lse.is_contiguous()
    rc = _fn().splitk(q.data_ptr(), q.stride(1), q.stride(2), window_pool.data_ptr(), cmp_pool.data_ptr(),
                      window_pool.stride(0), sink.data_ptr(), idx.data_ptr(), idx.stride(1), idx.shape[-1],
                      int(n_window), 0 if cnt is None else cnt.data_ptr(), mid_o.data_ptr(), mid_lse.data_ptr(),
                      int(n_splits), b * m, float(scale), torch.cuda.current_stream().cuda_stream)
    assert rc == 0, f"dsv4_sparse_mla_splitk failed: {rc}"


def sparse_mla_prefill_shared(q, window_pool, cmp_pool, sink, win_idx, cmp_shared, cnt, scale: float):
    """Prefill attention with a row-shared compressed list: q [1, m, 64, 512]; win_idx [1, m, w] int32
    (window columns, -1 masked); cmp_shared [n] int32 global compressed slots shared by every query;
    cnt [m] int32 live compressed count per query. == the materialized [window | compressed] path."""
    b, m, h, d = q.shape
    assert b == 1 and q.is_contiguous() and win_idx.is_contiguous() and cmp_shared.is_contiguous()
    assert window_pool.stride() == cmp_pool.stride()
    out = torch.empty_like(q)
    w = win_idx.shape[-1]
    rc = _fn().shared(q.data_ptr(), q.stride(1), q.stride(2), window_pool.data_ptr(), cmp_pool.data_ptr(),
                      window_pool.stride(0), sink.data_ptr(), win_idx.data_ptr(), win_idx.stride(1),
                      w + cmp_shared.numel(), w, cnt.data_ptr(), cmp_shared.data_ptr(), out.data_ptr(),
                      out.stride(1), out.stride(2), m, float(scale), torch.cuda.current_stream().cuda_stream)
    assert rc == 0, f"dsv4_sparse_mla_shared failed: {rc}"
    return out
