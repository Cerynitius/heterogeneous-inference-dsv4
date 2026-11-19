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
_RUNTIME_CONF = _os.getenv("FREETOKEN_RUNTIME_CONF", "").strip()
_RT_TYPES: dict[str, tuple[type, ...]] = {
    "default_max_output_tokens": (int,),
    "temperature": (int, float),
    "top_p": (int, float),
    "top_k": (int,),
    "force_temperature": (int, float, type(None)),
    "default_reasoning_effort": (str, type(None)),
    "default_thinking": (bool,),
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
