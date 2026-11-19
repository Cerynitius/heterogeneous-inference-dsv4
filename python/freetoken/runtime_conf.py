"""Hot-reloadable request defaults (owner 2026-11-19: change values without a restart).

FREETOKEN_RUNTIME_CONF names a JSON file, re-read when its mtime/size changes (one stat per call).
"""
from __future__ import annotations

import json
import os as _os
from typing import Any

# Hot-reloadable request defaults (owner 2026-11-19: change values without a restart).
# FREETOKEN_RUNTIME_CONF names a JSON file; it is re-read whenever its mtime/size changes
# (one stat per request). Keys present override the startup values; a missing file or an
# absent key falls back to them. A malformed file is rejected with its real error logged and
# the last good values stay in force.
#   default_max_output_tokens  int >= 1     budget when the request omits max_tokens
#   temperature / top_p / top_k             defaults for unspecified request fields
#   force_temperature          float|null   every request samples at this (null = honour request)
#   default_reasoning_effort   str|null     effort when the request sends none (encoder validates)
#   default_thinking           bool         thinking on when the request sends no thinking switch
#   max_running_requests       int >= 1     requests admitted at once; capped at the boot
#                                           --max-running-requests (slots are sized at boot);
#                                           lowering it never stops running requests
#   prefill_chunk_tokens       int >= 1     prefill chunk size; capped at the boot chunk
#                                           (FREETOKEN_PREFILL_CHUNK_CAP / window budget)
#   idx_split                  float 0..1   share of deep prefill indexer queries on the
#                                           second card (0 = off; FREETOKEN_DSV4_IDX_SPLIT)
#   idx_split_min_work         float >= 0   split only when queries x key blocks reach this
#   sched_decode_share         float 0..0.95 GPU time share running decodes get while prompts
#                                           prefill (FREETOKEN_SCHED_DECODE_SHARE)
#   sched_reserve_tokens       int >= 0     future tokens reserved per running request
#                                           (0 = whole max_tokens; FREETOKEN_SCHED_RESERVE_TOKENS)
#   sched_short_first          bool         short prompts overtake a long chunked prefill
#   sched_aligned_chunks       bool         a fresh prompt waits for a whole chunk
#   swa_eviction_interval      int >= 1     forwards between out-of-window SWA frees
#   ep_prefill_split           int >= 1     EP prefill MoE sub-chunks per chunk
#   ep_prefill_split_ratio     float 0..1   geometric shrink of those sub-chunks
#   forward_unknown_tools      bool         pass through tool calls whose name the request
#                                           did not declare
_RUNTIME_CONF = _os.getenv("FREETOKEN_RUNTIME_CONF", "").strip()
_RT_TYPES: dict[str, tuple[type, ...]] = {
    "default_max_output_tokens": (int,),
    "temperature": (int, float),
    "top_p": (int, float),
    "top_k": (int,),
    "force_temperature": (int, float, type(None)),
    "default_reasoning_effort": (str, type(None)),
    "default_thinking": (bool,),
    "max_running_requests": (int,),
    "prefill_chunk_tokens": (int,),
    "idx_split": (int, float),
    "idx_split_min_work": (int, float),
    "sched_decode_share": (int, float),
    "sched_reserve_tokens": (int,),
    "sched_short_first": (bool,),
    "sched_aligned_chunks": (bool,),
    "swa_eviction_interval": (int,),
    "ep_prefill_split": (int,),
    "ep_prefill_split_ratio": (int, float),
    "forward_unknown_tools": (bool,),
}
_rt_state: dict[str, Any] = {"key": None, "values": {}}


def _validate_runtime(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"top level must be a JSON object, got {type(raw).__name__}")
    unknown = sorted(set(raw) - set(_RT_TYPES))
    if unknown:
        raise ValueError(f"unknown keys {unknown}; allowed {sorted(_RT_TYPES)}")
    for k, v in raw.items():
        if (isinstance(v, bool) and bool not in _RT_TYPES[k]) or not isinstance(v, _RT_TYPES[k]):
            raise ValueError(f"{k}={v!r}: wrong type")
    if raw.get("default_max_output_tokens", 1) < 1:
        raise ValueError("default_max_output_tokens must be >= 1")
    for k, lo, hi in (("max_running_requests", 1, None), ("prefill_chunk_tokens", 1, None),
                      ("idx_split", 0, 1), ("idx_split_min_work", 0, None), ("sched_decode_share", 0, 0.95),
                      ("sched_reserve_tokens", 0, None), ("swa_eviction_interval", 1, None),
                      ("ep_prefill_split", 1, None), ("ep_prefill_split_ratio", 0.01, 1)):
        if k in raw and (raw[k] < lo or (hi is not None and raw[k] > hi)):
            raise ValueError(f"{k}={raw[k]!r} out of range [{lo}, {hi if hi is not None else 'inf'}]")
    return dict(raw)


def runtime_overrides() -> dict[str, Any]:
    """Current hot overrides ({} when FREETOKEN_RUNTIME_CONF is unset)."""
    if not _RUNTIME_CONF:
        return {}
    try:
        st = _os.stat(_RUNTIME_CONF)
        key = (st.st_mtime_ns, st.st_size)
    except FileNotFoundError:
        key = None
    if key == _rt_state["key"]:
        return _rt_state["values"]
    if key is None:
        values: dict[str, Any] = {}
    else:
        try:
            with open(_RUNTIME_CONF, encoding="utf-8") as f:
                values = _validate_runtime(json.load(f))
        except Exception as exc:  # keep the last good values; report the real cause
            _rt_state["key"] = key
            _rt_logger().error("runtime conf %s rejected (%s: %s); keeping %s",
                               _RUNTIME_CONF, type(exc).__name__, exc, _rt_state["values"])
            return _rt_state["values"]
    _rt_state["key"], _rt_state["values"] = key, values
    _rt_logger().info("runtime conf %s loaded: %s", _RUNTIME_CONF, values or "(none, startup values)")
    return values


_LOGGER = None


def _rt_logger():
    # freetoken's own logger (stdout, like every other engine line); a plain stdlib logger
    # was silent in the live server (11-19: neither INFO nor ERROR reached the log).
    global _LOGGER
    if _LOGGER is None:
        from freetoken.utils import init_logger
        _LOGGER = init_logger("freetoken.runtime_conf")
    return _LOGGER
