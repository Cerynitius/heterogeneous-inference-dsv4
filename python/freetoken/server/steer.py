"""Steer: a user message sent while the previous answer is still being generated.

Clients steer by dropping the running stream and sending the history + the partial answer they
saw + the new user message. Rendered by the template as usual, that partial answer loses its
reasoning (a user message follows it), so the model forgets its own thinking and the prefix cache
stops matching at that turn. Instead:

1. Cut: the new request extends a generation that did not end on its own (still running, or ended
   by a client disconnect) with that partial answer -> the old generation is aborted at once.
2. EOS + continue: the partial answer is rendered exactly as the model produced it --
   ``<think>`` + its reasoning + ``</think>`` + the content the client saw + EOS -- followed by
   the new messages. The old prompt is then a strict prefix of the new one (cache hit) and the
   model keeps its own thinking for the steered turn.

Fallback: anything that does not match exactly (history differs, tools or thinking switch differ,
the partial content is not what the model wrote, tool calls in the partial answer, images,
non-thinking mode) leaves the request untouched -- the normal template rendering. Hot switch:
runtime.json ``steer_continue`` (default on).
"""

from __future__ import annotations

import json
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from freetoken.runtime_conf import runtime_overrides
from freetoken.utils import init_logger

logger = init_logger(__name__)

STEER_KEY = "_steer_reasoning"   # message key read by the DSV4 renderer (tokenizer/tokenize.py)
THINK_END = "</think>"
_KEEP = 64                       # generations remembered
_TTL_S = 1800.0                  # ... for at most this long


@dataclass
class _Gen:
    uid: int
    messages: list
    fp: str
    t0: float
    text: list = field(default_factory=list)   # raw output pieces (reasoning </think> content)
    finished: bool = False                     # the scheduler finished it (any reason)
    cut: bool = False                          # aborted (client disconnect, or cut by a steer)


_SPECIAL = re.compile(r"<｜[^｜<>]{1,64}｜>")   # special tokens: the API strips them from what clients see


def _cut(raw: str, client: str) -> str | None:
    """The prefix of the model's raw text whose visible form (special tokens removed, leading
    whitespace ignored) is exactly the client's text; None if the client's text is not a prefix."""
    vis, pos, last = [], [], 0
    for m in list(_SPECIAL.finditer(raw)) + [None]:
        stop = m.start() if m is not None else len(raw)
        for k in range(last, stop):
            vis.append(raw[k])
            pos.append(k)
        if m is not None:
            last = m.end()
    v = "".join(vis)
    lead = len(v) - len(v.lstrip())
    c = _SPECIAL.sub("", client).strip()   # some special tokens reach the client, some do not
    if not v[lead:].startswith(c):
        return None
    end = lead + len(c)
    return raw[: pos[end - 1] + 1] if end > 0 else ""


def _diff(model: str, client: str) -> str:
    model, client = _SPECIAL.sub("", model), _SPECIAL.sub("", client)
    i = next((k for k, (x, y) in enumerate(zip(model, client)) if x != y), min(len(model), len(client)))
    return f" (first difference at char {i}: model {model[i:i + 24]!r}, client {client[i:i + 24]!r})"


def enabled() -> bool:
    v = runtime_overrides().get("steer_continue")
    return True if v is None else bool(v)


def fingerprint(spec: Any) -> str:
    """Everything besides the messages that shapes the rendered prompt."""
    return json.dumps([spec.template_tools, spec.chat_template_kwargs], sort_keys=True, default=str,
                      ensure_ascii=False)


class SteerRegistry:
    def __init__(self) -> None:
        self._gens: "OrderedDict[int, _Gen]" = OrderedDict()

    # ---- bookkeeping (FrontendManager) ----
    def record(self, uid: int, messages: list, fp: str) -> None:
        self._gens[uid] = _Gen(uid, messages, fp, time.monotonic())
        now = time.monotonic()
        while self._gens and (len(self._gens) > _KEEP or now - next(iter(self._gens.values())).t0 > _TTL_S):
            self._gens.popitem(last=False)

    def observe(self, msg: Any) -> None:
        g = self._gens.get(getattr(msg, "uid", None))
        if g is None:
            return
        piece = getattr(msg, "incremental_output", None)
        if piece:
            g.text.append(piece)
        if getattr(msg, "finished", False):
            g.finished = True

    def mark_cut(self, uid: int) -> None:
        g = self._gens.get(uid)
        if g is not None:
            g.cut = True

    # ---- the steer decision (submit_generation) ----
    def prepare(self, spec: Any, thinking: bool) -> tuple[list | None, int | None]:
        """(messages to render, uid to abort) for a steer, else (None, None)."""
        if not thinking or spec.images or not enabled():
            return None, None
        msgs = spec.messages
        fp = fingerprint(spec)
        g = None
        for cand in reversed(self._gens.values()):
            n = len(cand.messages)
            if cand.fp != fp or n + 2 > len(msgs):
                continue
            if cand.finished and not cand.cut:
                continue          # ended on its own: an ordinary next turn, template rules apply
            if msgs[:n] == cand.messages:
                g = cand
                break
        if g is None:
            return None, None
        n = len(g.messages)
        part, rest = msgs[n], msgs[n + 1:]

        def no(why: str):
            logger.info("steer fallback (extends uid %d): %s", g.uid, why)
            return None, None

        if part.get("role") != "assistant":
            return no("no partial answer after the cut turn")
        if not any(m.get("role") == "user" for m in rest):
            return no("no new user message")
        if part.get("tool_calls"):
            return no("partial answer carries tool calls")
        content = part.get("content") or ""
        if not isinstance(content, str):
            return no("partial answer content is not plain text")
        raw = "".join(g.text)
        if THINK_END in raw:
            reasoning, body = raw.split(THINK_END, 1)
        else:
            reasoning, body = raw, ""
        c_cli = content.strip()
        if c_cli:
            body_keep = _cut(body, c_cli) if THINK_END in raw else None
            if body_keep is None:
                return no("partial content is not what the model wrote" + _diff(body.lstrip(), c_cli))
            reasoning_keep = reasoning                 # reasoning was complete
        else:
            body_keep = ""
            r_cli = (part.get("reasoning_content") or "").strip()
            if r_cli:
                reasoning_keep = _cut(reasoning, r_cli)
                if reasoning_keep is None:
                    return no("partial reasoning is not what the model wrote" + _diff(reasoning.lstrip(), r_cli))
            else:
                reasoning_keep = reasoning             # everything generated before the cut
        if not reasoning_keep.strip() and not body_keep:
            return no("nothing was generated before the cut")
        newpart = dict(part)
        newpart["content"] = body_keep
        newpart[STEER_KEY] = reasoning_keep
        cut_uid = None if g.finished else g.uid
        g.cut = True
        logger.info(
            "steer: continues uid %d (%s), keeps %d reasoning + %d content chars, then %d new message(s)",
            g.uid, "cut now" if cut_uid is not None else "already ended", len(reasoning_keep),
            len(body_keep), len(rest),
        )
        return msgs[:n] + [newpart] + list(rest), cut_uid
