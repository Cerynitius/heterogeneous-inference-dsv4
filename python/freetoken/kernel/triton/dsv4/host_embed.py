"""Embedding table kept in pinned host memory, gathered by the GPU through its UVA address
(zero-copy, graph-safe). A decode step reads one 8 KB row per token, so the ~1 GiB bf16 table
does not need to sit in VRAM. Bit-identical to ``nn.Embedding`` (a pure copy).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import nn


class _HostPtr:
    """Triton pointer argument for a pinned host tensor (the UVA address is device-accessible)."""

    def __init__(self, t: torch.Tensor):
        self.t = t
        self.dtype = t.dtype

    def data_ptr(self) -> int:
        return self.t.data_ptr()


@triton.jit
def _host_gather_kernel(ids_ptr, w_ptr, out_ptr, D: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)
    c = tl.program_id(1)
    row = tl.load(ids_ptr + i).to(tl.int64)
    offs = c * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + i * D + offs, tl.load(w_ptr + row * D + offs))


class HostEmbedding(nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.host_weight = weight.detach().to("cpu").pin_memory()
        self.device = weight.device
        self.dim = weight.shape[1]
        assert self.dim % 1024 == 0, self.dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        flat = ids.reshape(-1).contiguous()
        out = torch.empty(flat.numel(), self.dim, dtype=self.host_weight.dtype, device=ids.device)
        if flat.numel():
            _host_gather_kernel[(flat.numel(), self.dim // 1024)](
                flat, _HostPtr(self.host_weight), out, D=self.dim, BLOCK=1024, num_warps=4)
        return out.view(*ids.shape, self.dim)


__all__ = ["HostEmbedding"]
