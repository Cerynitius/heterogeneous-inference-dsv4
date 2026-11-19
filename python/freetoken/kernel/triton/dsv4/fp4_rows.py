"""FP4-packed Lightning-Indexer key pool (11-18).

The indexer keys are written as FP4 round-trips (hadamard + fp4_act_quant_inplace, e2m1 x 2^e per
32 dims), so (packed e2m1 codes [64 B], ue8m0 exponents [4 B]) per 128-dim row is lossless: 68 B
instead of 256 B of bf16. ``Fp4Rows`` stands in for the bf16 ``idx_pool[L]`` tensor; writers pack
(``index_copy_``), the FP4 prefill indexer reads the codes as they are, the decode kernel decodes
them in registers to the identical bf16 values, and ``unpack`` rebuilds bf16 for any other user.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

ENABLED = os.environ.get("FREETOKEN_DSV4_IDX_FP4POOL", "0") == "1"
_CHECK = int(os.environ.get("FREETOKEN_DSV4_IDX_FP4POOL_CHECK", "0") or 0)   # verify N writes


class Fp4Rows:
    """Packed rows: codes [n, 64] uint8 (low nibble = even dim), exps [n, 4] uint8 (ue8m0)."""

    def __init__(self, n: int, device):
        self.codes = torch.zeros((n, 64), dtype=torch.uint8, device=device)
        self.exps = torch.full((n, 4), 127, dtype=torch.uint8, device=device)
        self.shape = (n, 128)
        self.dtype = torch.bfloat16      # logical dtype of the rows it stands for
        self.device = self.codes.device

    def numel(self) -> int:
        return self.codes.numel() + self.exps.numel()

    def element_size(self) -> int:
        return 1

    def index_copy_(self, dim: int, rows: torch.Tensor, kv: torch.Tensor) -> "Fp4Rows":
        assert dim == 0
        from freetoken.kernel.cuda.dsv4_idx_fp4 import pack_fp4

        p, e = pack_fp4(kv.reshape(-1, 128).to(torch.bfloat16))
        global _CHECK
        if _CHECK > 0 and not torch.cuda.is_current_stream_capturing():
            _CHECK -= 1
            back = unpack(p, e)
            assert torch.equal(back, kv.reshape(-1, 128).to(torch.bfloat16)), \
                "FP4 idx pool: key not an exact FP4 round-trip"
        self.codes.index_copy_(0, rows, p)
        self.exps.index_copy_(0, rows, e)
        return self

    def gather(self, rows: torch.Tensor):
        return self.codes.index_select(0, rows), self.exps.index_select(0, rows)


class Fp4Keys:
    """Dense gathered keys for one prefill: codes [bsz, n, 64], exps [bsz, n, 4]."""

    def __init__(self, codes: torch.Tensor, exps: torch.Tensor):
        self.codes, self.exps = codes, exps
        self.shape = (codes.shape[0], codes.shape[1], 128)
        self.dtype = torch.bfloat16
        self.device = codes.device

    def __getitem__(self, i):
        assert i == 0
        return unpack(self.codes[0], self.exps[0])

    def dequant(self) -> torch.Tensor:
        return unpack(self.codes.reshape(-1, 64), self.exps.reshape(-1, 4)).view(self.shape)


@triton.jit
def _e2m1(c):
    """4-bit e2m1 code -> fp32 value (sign bit 3), exact."""
    m = c & 7
    mag = tl.where(m < 4, m.to(tl.float32) * 0.5,
                   tl.where(m < 6, (m - 2).to(tl.float32), tl.where(m == 6, 4.0, 6.0)))
    return tl.where((c & 8) != 0, -mag, mag)


@triton.jit
def _e2m1_bf16_bits(c, eadj):
    """4-bit e2m1 code + (exponent - 127) -> bf16 bit pattern (int32), exact, no float math.
    E = code bits [2:1], m = bit 0: E > 0 -> 2^(E-1) (1 + m/2) = bf16 exp E+126, mantissa m<<6;
    E = 0 -> 0 or 0.5 (bf16 exp 126). The 2^(e-127) scale adds to the exponent field."""
    E = (c >> 1) & 3
    m = c & 1
    mag = tl.where(E == 0, m * (126 << 7), ((E + 126) << 7) | (m << 6))
    mag = tl.where(mag != 0, mag + (eadj << 7), 0)
    return ((c & 8) << 12) | mag


@triton.jit
def load_fp4_rows(codes_ptr, exps_ptr, rows, mask, stride_c, stride_e):
    """rows [BT] int64 -> [BT, 128] bf16 decoded keys (exact FP4 round-trip values)."""
    offs_j = tl.arange(0, 64)
    c = tl.load(codes_ptr + rows[:, None] * stride_c + offs_j[None, :], mask=mask[:, None], other=0).to(tl.int32)
    e4 = tl.load(exps_ptr + rows[:, None] * stride_e + tl.arange(0, 4)[None, :], mask=mask[:, None], other=127).to(tl.int32) - 127
    eadj = tl.reshape(tl.broadcast_to(e4[:, :, None], (e4.shape[0], 4, 16)), (e4.shape[0], 64))
    lo = _e2m1_bf16_bits(c & 0xF, eadj)
    hi = _e2m1_bf16_bits(c >> 4, eadj)
    bits = tl.reshape(tl.join(lo, hi), (lo.shape[0], 128)).to(tl.int16)
    return bits.to(tl.bfloat16, bitcast=True)


@triton.jit
def _unpack_kernel(codes_ptr, exps_ptr, out_ptr, n, BT: tl.constexpr):
    rows = tl.program_id(0) * BT + tl.arange(0, BT)
    m = rows < n
    v = load_fp4_rows(codes_ptr, exps_ptr, rows.to(tl.int64), m, 64, 4)
    offs = tl.arange(0, 128)
    tl.store(out_ptr + rows.to(tl.int64)[:, None] * 128 + offs[None, :], v, mask=m[:, None])


def unpack(codes: torch.Tensor, exps: torch.Tensor) -> torch.Tensor:
    """codes [n, 64], exps [n, 4] (contiguous) -> bf16 [n, 128]."""
    codes, exps = codes.contiguous(), exps.contiguous()
    n = codes.shape[0]
    out = torch.empty((n, 128), dtype=torch.bfloat16, device=codes.device)
    if n:
        _unpack_kernel[(triton.cdiv(n, 32),)](codes, exps, out, n, BT=32, num_warps=4)
    return out
