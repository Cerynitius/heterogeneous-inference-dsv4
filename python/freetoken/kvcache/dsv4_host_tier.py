"""Two-tier (VRAM + host RAM) backing for the DSV4 ratio-4 compressed-KV pools.

KVMem idea, made lossless by DSV4's own sparsity: a ratio-4 layer's compressed KV
(``cmp_pool``, 1 KiB/row, 78% of the per-token KV bytes) is only ever READ at the rows the
Lightning Indexer picked (top-512 per query), while the indexer keys / ratio-128 KV / window
stay fully resident in VRAM. So the full compressed history may live in host RAM and the GPU
only touches the rows that the model's own indexer selects -- no approximation.

Physical layout: ONE contiguous CUDA virtual range per layer (cuMemAddressReserve), rows
``[0, gpu_rows)`` mapped to device memory and the rest to pinned host memory
(cuMemCreate ``CU_MEM_LOCATION_TYPE_HOST_NUMA``). Every existing kernel / torch op keeps using
a single ordinary CUDA tensor; host rows are read zero-copy over PCIe. The page allocator
hands out the LOWEST free pages first (CacheManager sorts the free list when this tier is on),
so the host tier is only touched once live + cached KV exceeds the VRAM tier.

Enabled by ``FREETOKEN_DSV4_CMP_GPU_TOKENS=<full tokens kept in VRAM>`` (unset = off, every
pool all-VRAM exactly as before).
"""

from __future__ import annotations

import ctypes
import os

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

_CU_MEM_ALLOCATION_TYPE_PINNED = 1
_CU_MEM_LOCATION_TYPE_DEVICE = 1
_CU_MEM_LOCATION_TYPE_HOST_NUMA = 3
_CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3
_CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = 1


class _Loc(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _AllocFlags(ctypes.Structure):
    _fields_ = [
        ("compressionType", ctypes.c_ubyte),
        ("gpuDirectRDMACapable", ctypes.c_ubyte),
        ("usage", ctypes.c_ushort),
        ("reserved", ctypes.c_ubyte * 4),
    ]


class _Prop(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleTypes", ctypes.c_int),
        ("location", _Loc),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", _AllocFlags),
    ]


class _Access(ctypes.Structure):
    _fields_ = [("location", _Loc), ("flags", ctypes.c_int)]


_cu = None


def _lib():
    global _cu
    if _cu is None:
        _cu = ctypes.CDLL("libcuda.so.1")
        u64, sz = ctypes.c_uint64, ctypes.c_size_t
        _cu.cuMemAddressReserve.argtypes = [ctypes.POINTER(u64), sz, sz, u64, u64]
        _cu.cuMemAddressFree.argtypes = [u64, sz]
        _cu.cuMemCreate.argtypes = [ctypes.POINTER(u64), sz, ctypes.POINTER(_Prop), u64]
        _cu.cuMemRelease.argtypes = [u64]
        _cu.cuMemMap.argtypes = [u64, sz, sz, u64, u64]
        _cu.cuMemUnmap.argtypes = [u64, sz]
        _cu.cuMemSetAccess.argtypes = [u64, sz, ctypes.POINTER(_Access), sz]
        _cu.cuMemGetAllocationGranularity.argtypes = [ctypes.POINTER(sz), ctypes.POINTER(_Prop), ctypes.c_int]
        _cu.cuGetErrorName.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
    return _cu


def _check(rc: int, what: str) -> None:
    if rc != 0:
        name = ctypes.c_char_p()
        _lib().cuGetErrorName(rc, ctypes.byref(name))
        raise RuntimeError(f"{what} failed: CUresult {rc} ({(name.value or b'?').decode()})")


def _numa_node(device: torch.device) -> int:
    p = torch.cuda.get_device_properties(device)
    bdf = f"{p.pci_domain_id:04x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}.0"
    try:
        with open(f"/sys/bus/pci/devices/{bdf.lower()}/numa_node") as f:
            return max(0, int(f.read().strip()))
    except OSError:
        return 0


def _prop(loc_type: int, loc_id: int) -> _Prop:
    p = _Prop()
    p.type = _CU_MEM_ALLOCATION_TYPE_PINNED
    p.location.type = loc_type
    p.location.id = loc_id
    return p


def _granularity(prop: _Prop) -> int:
    g = ctypes.c_size_t()
    _check(_lib().cuMemGetAllocationGranularity(ctypes.byref(g), ctypes.byref(prop),
                                               _CU_MEM_ALLOC_GRANULARITY_RECOMMENDED),
           "cuMemGetAllocationGranularity")
    return int(g.value)


def _round_up(n: int, g: int) -> int:
    return -(-n // g) * g


class _TieredRange:
    """Owns one reserved VA range + its two physical handles; released on GC."""

    def __init__(self, device: torch.device, gpu_bytes: int, host_bytes: int) -> None:
        cu = _lib()
        torch.cuda.init()
        with torch.cuda.device(device):
            torch.empty(1, device=device)  # make the primary context current
            dev_idx = device.index if device.index is not None else torch.cuda.current_device()
            pd = _prop(_CU_MEM_LOCATION_TYPE_DEVICE, dev_idx)
            ph = _prop(_CU_MEM_LOCATION_TYPE_HOST_NUMA, _numa_node(device))
            g = max(_granularity(pd), _granularity(ph))
            self.gpu_bytes = _round_up(gpu_bytes, g) if gpu_bytes > 0 else 0
            self.host_bytes = _round_up(host_bytes, g) if host_bytes > 0 else 0
            self.size = self.gpu_bytes + self.host_bytes
            self.handles: list[int] = []
            self.mapped: list[tuple[int, int]] = []
            ptr = ctypes.c_uint64()
            _check(cu.cuMemAddressReserve(ctypes.byref(ptr), self.size, g, 0, 0), "cuMemAddressReserve")
            self.ptr = int(ptr.value)
            try:
                off = 0
                for nbytes, prop, what in ((self.gpu_bytes, pd, "device"), (self.host_bytes, ph, "host")):
                    if nbytes == 0:
                        continue
                    h = ctypes.c_uint64()
                    _check(cu.cuMemCreate(ctypes.byref(h), nbytes, ctypes.byref(prop), 0),
                           f"cuMemCreate({what}, {nbytes} B)")
                    self.handles.append(int(h.value))
                    _check(cu.cuMemMap(self.ptr + off, nbytes, 0, int(h.value), 0), f"cuMemMap({what})")
                    self.mapped.append((self.ptr + off, nbytes))
                    off += nbytes
                acc = _Access()
                acc.location.type = _CU_MEM_LOCATION_TYPE_DEVICE
                acc.location.id = dev_idx
                acc.flags = _CU_MEM_ACCESS_FLAGS_PROT_READWRITE
                _check(cu.cuMemSetAccess(self.ptr, self.size, ctypes.byref(acc), 1), "cuMemSetAccess")
            except Exception:
                self.release()
                raise

    def release(self) -> None:
        cu = _lib()
        for p, n in getattr(self, "mapped", []):
            cu.cuMemUnmap(p, n)
        self.mapped = []
        for h in getattr(self, "handles", []):
            cu.cuMemRelease(h)
        self.handles = []
        if getattr(self, "ptr", 0):
            cu.cuMemAddressFree(self.ptr, self.size)
            self.ptr = 0

    def __del__(self) -> None:
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            self.release()
        except Exception:
            pass


class _CAI:
    def __init__(self, ptr: int, shape: tuple, typestr: str, owner) -> None:
        self.owner = owner  # keeps the mapping alive as long as any tensor view exists
        self.__cuda_array_interface__ = {
            "shape": shape, "typestr": typestr, "data": (ptr, False), "version": 3, "strides": None,
        }


def tiered_rows(rows: int, gpu_rows: int, row_dim: int, dtype: torch.dtype,
                device: torch.device) -> torch.Tensor:
    """A zeroed ``[rows, row_dim]`` CUDA tensor whose first ``gpu_rows`` rows are VRAM and the
    rest pinned host RAM (one contiguous device address range)."""
    assert dtype in (torch.bfloat16, torch.uint8), dtype
    gpu_rows = max(0, min(int(gpu_rows), int(rows)))
    esz = 2 if dtype == torch.bfloat16 else 1
    row_bytes = row_dim * esz
    rng = _TieredRange(device, gpu_rows * row_bytes, (rows - gpu_rows) * row_bytes)
    # The device part is rounded UP to the granularity, so a few extra rows past gpu_rows may
    # also be VRAM -- never fewer.
    t = torch.as_tensor(_CAI(rng.ptr, (rows, row_dim), "<i2" if esz == 2 else "|u1", rng),
                        device=device)
    if dtype == torch.bfloat16:
        t = t.view(torch.bfloat16)
    t.zero_()
    return t


def gpu_tier_tokens() -> int | None:
    v = os.environ.get("FREETOKEN_DSV4_CMP_GPU_TOKENS")
    return None if v in (None, "") else int(v)
