"""Speculative SAMPLING for DSpark (temperature / top-k / top-p requests).

Standard speculative sampling (Leviathan et al. / Chen et al.): the draft samples d_i ~ q_i,
the target computes p_i on the verify pass; d_i is accepted with probability
min(1, p_i(d_i) / q_i(d_i)); at the first rejection a token is drawn from
norm(max(0, p_i - q_i)); if all k drafts pass, a bonus token is drawn from p_k. The emitted
sequence is distributed exactly as sampling the target one token at a time.

``p`` is the engine sampler's distribution (engine/sample.py): softmax(logits / T) truncated to
the top-k tokens and to the top-p nucleus (smallest prefix of the sorted probabilities whose mass
reaches top_p, the crossing token included), renormalized. ``q`` uses the same processing on the
draft logits (any q is correct as long as d ~ q; matching the target's processing maximizes
acceptance).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

MIN_T = MIN_P = 1e-6   # = engine/sample.py


@dataclass(frozen=True)
class SampleParams:
    temperature: float
    top_k: int          # 0 = off
    top_p: float        # 1.0 = off

    @classmethod
    def from_request(cls, sp, vocab_size: int) -> "SampleParams":
        t = max(float(sp.temperature), MIN_T)
        k = int(sp.top_k) if sp.top_k >= 1 else 0
        if k >= vocab_size:
            k = 0
        p = min(max(float(sp.top_p), MIN_P), 1.0)
        return cls(t, k, p)


def probs_from_logits(logits: torch.Tensor, sp: SampleParams) -> torch.Tensor:
    """[N, V] logits -> [N, V] fp32 sampling distribution (temperature, top-k, top-p)."""
    probs = torch.softmax(logits.float() / sp.temperature, dim=-1)
    if sp.top_k == 0 and sp.top_p >= 1.0:
        return probs
    sorted_p, idx = probs.sort(dim=-1, descending=True)
    keep = torch.ones_like(sorted_p, dtype=torch.bool)
    if sp.top_k:
        keep[:, sp.top_k:] = False
    if sp.top_p < 1.0:
        before = sorted_p.cumsum(dim=-1) - sorted_p      # mass strictly before each token
        keep &= before < sp.top_p
    sorted_p = torch.where(keep, sorted_p, torch.zeros_like(sorted_p))
    out = torch.zeros_like(probs).scatter_(-1, idx, sorted_p)
    return out / out.sum(dim=-1, keepdim=True)


def accept(p: torch.Tensor, q: torch.Tensor, d: torch.Tensor, max_accept: int):
    """p [k+1, V] target, q [k, V] draft, d [k] draft tokens -> (a, emitted [a+1] int64).
    ``max_accept`` caps a (request length budget); a capped round emits d[:a+1]."""
    k = d.numel()
    rows = torch.arange(k, device=d.device)
    pd = p[rows, d]
    qd = q[rows, d]
    u = torch.rand(k, device=d.device)
    ok = u * qd < pd                                     # u < p/q without dividing by q
    a = int(torch.cumprod(ok.int(), 0).sum().item())
    if a > max_accept:
        a = max_accept
        return a, d[: a + 1].clone()
    if a < k:
        resid = torch.clamp(p[a] - q[a], min=0.0)
        s = resid.sum()
        dist = resid / s if float(s) > 0 else p[a]
    else:
        dist = p[k]
    tok = torch.multinomial(dist, 1)
    return a, torch.cat([d[:a], tok])


__all__ = ["SampleParams", "probs_from_logits", "accept"]
