"""ctypes binding of dsfp4_mma.cu (sm_120a native block-scaled FP8 x FP4 grouped MoE GEMM).

Built with nvcc on first use into ~/.cache/freetoken/cuda/dsfp4_mma_<source hash>.so, linked
against the shared cudart so it runs on the CUDA runtime/context torch already uses."""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess

import torch

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dsfp4_mma.cu")
_LIBS: dict = {}


def _load(bm: int = 64):
    if bm not in _LIBS:
        src = open(_SRC, "rb").read()
        tag = hashlib.sha1(src).hexdigest()[:12] + f"_bm{bm}"
        out_dir = os.path.expanduser("~/.cache/freetoken/cuda")
        os.makedirs(out_dir, exist_ok=True)
        so = os.path.join(out_dir, f"dsfp4_mma_{tag}.so")
        if not os.path.exists(so):
            nvcc = os.path.join(os.environ.get("CUDA_HOME", "/usr/local/cuda"), "bin", "nvcc")
            tmp = f"{so}.{os.getpid()}.tmp"
            subprocess.check_call([nvcc, "-O3", "-gencode=arch=compute_120a,code=sm_120a", f"-DBM={bm}",
                                   "-cudart", "shared", "-shared", "-Xcompiler", "-fPIC",
                                   "-o", tmp, _SRC])
            os.replace(tmp, so)
        lib = ctypes.CDLL(so)
        fn = lib.dsfp4_mma_grouped
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.c_void_p] * 10 + [ctypes.c_int] * 5 + [ctypes.c_void_p]
        _LIBS[bm] = lib
    return _LIBS[bm]


def _ptr(t: torch.Tensor | None):
    return ctypes.c_void_p(0 if t is None else t.data_ptr())


def dsfp4_mma_grouped(aq: torch.Tensor, a_exp: torch.Tensor, a_map: torch.Tensor | None,
                      w: torch.Tensor, w_scale: torch.Tensor, sorted_ids: torch.Tensor,
                      expert_ids: torch.Tensor, ntpp: torch.Tensor, tw: torch.Tensor | None,
                      out: torch.Tensor, R: int, bm: int = 64) -> torch.Tensor:
    """``out[r] = A[a_map[r] if a_map is not None else r] @ W[expert(r)]^T`` (x ``tw[r]``).

    aq [Ra, K] FP8 e4m3 codes, a_exp [Ra, K/128] uint8 e8m0 (act_quant_fp8); w [S, N, K/2] packed
    e2m1, w_scale [S, N, K/32] e8m0; sorted_ids / expert_ids / ntpp from moe_align_block_size
    (block ``bm``: 64 or 128); out [R, N] bf16 (rows of padding ids untouched)."""
    S, N, KW = w.shape
    K = KW * 2
    assert aq.shape[1] == K and a_exp.shape[1] == K // 128, (aq.shape, a_exp.shape, K)
    assert out.dtype == torch.bfloat16 and out.shape[1] == N and out.stride(1) == 1
    for t in (aq, a_exp, w, w_scale, sorted_ids, expert_ids):
        assert t.is_contiguous()
    assert sorted_ids.dtype == torch.int32 and expert_ids.dtype == torch.int32
    lib = _load(bm)
    with torch.cuda.device(aq.device):
        rc = lib.dsfp4_mma_grouped(
            _ptr(aq), _ptr(a_exp), _ptr(a_map), _ptr(w), _ptr(w_scale.view(torch.uint8)),
            _ptr(sorted_ids), _ptr(expert_ids), _ptr(ntpp), _ptr(tw), _ptr(out),
            sorted_ids.numel(), R, N, K, out.stride(0),
            ctypes.c_void_p(torch.cuda.current_stream(aq.device).cuda_stream))
    if rc != 0:
        raise RuntimeError(f"dsfp4_mma_grouped failed: CUDA error {rc}")
    return out


__all__ = ["dsfp4_mma_grouped"]
