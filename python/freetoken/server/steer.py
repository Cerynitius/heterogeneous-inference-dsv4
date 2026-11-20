"""Steer: a user message sent while the previous answer is still being generated.

Clients steer by dropping the running stream and sending the history + the partial answer they
saw + the new user message. Rendered by the template as usual, that partial answer loses its
reasoning (a user message follows it), so the model forgets its own thinking and the prefix cache
stops matching at that turn. Instead:

1. Cut: the new request extends a generation that did not end on its own (still running, or ended
   by a client disconnect) with that partial answer -> the old generation is aborted at once.
   Identical histories from different clients: a running generation is cut only when it is the
   one generation the partial answer matches; one that already ended is preferred.
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
_KEEP = 64                       # ended generations remembered (running ones are never dropped)
_TTL_S = 1800.0                  # ... for at most this long after they ended
_RUN_TTL_S = 4 * 3600.0          # a "running" entry this old lost its finish message: drop it


@dataclass
class _Gen:
    uid: int
    messages: list
    fp: str
    t0: float
    text: list = field(default_factory=list)   # raw output pieces (reasoning </think> content)
    finished: bool = False                     # the scheduler finished it (any reason)
    cut: bool = False                          # aborted (client disconnect, or cut by a steer)
    t_end: float = 0.0                         # when it finished or was cut


_SPECIAL = re.compile(r"<｜[^｜<>]{1,64}｜>|</?think>")   # special tokens + think tags: the API strips them from what clients see


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
        now = time.monotonic()
        self._gens[uid] = _Gen(uid, messages, fp, now)
        # Drop only generations that ended: a long running one must still be found when its
        # client steers, however many short requests passed meanwhile (11-21: a FIFO cap of 64
        # evicted streams that ran a few seconds under 4-way load).
        ended = [g for g in self._gens.values() if g.finished or g.cut]
        drop = {g.uid for g in ended if now - g.t_end > _TTL_S}
        over = len(self._gens) - len(drop) - _KEEP
        if over > 0:
            drop.update(g.uid for g in sorted((g for g in ended if g.uid not in drop),
                                              key=lambda g: g.t_end)[:over])
        drop.update(g.uid for g in self._gens.values()
                    if not (g.finished or g.cut) and now - g.t0 > _RUN_TTL_S)
        for k in drop:
            del self._gens[k]

    def observe(self, msg: Any) -> None:
        g = self._gens.get(getattr(msg, "uid", None))
        if g is None:
            return
        piece = getattr(msg, "incremental_output", None)
        if piece:
            g.text.append(piece)
        if getattr(msg, "finished", False):
            g.finished = True
            g.t_end = g.t_end or time.monotonic()

    def mark_cut(self, uid: int) -> None:
        g = self._gens.get(uid)
        if g is not None:
            g.cut = True
            g.t_end = g.t_end or time.monotonic()

    # ---- the steer decision (submit_generation) ----
    def prepare(self, spec: Any, thinking: bool) -> tuple[list | None, int | None]:
        """(messages to render, uid to abort) for a steer, else (None, None)."""
        if not thinking or spec.images or not enabled():
            return None, None
        msgs = spec.messages
        fp = fingerprint(spec)
        # Every remembered generation this request extends. Several can match when different
        # clients sent the same history (same system prompt + "hi"): the partial answer picks the
        # ones whose text it is, and a still-running generation is cut only when it is the single
        # such match -- never another client's stream on a guess.
        ok, why, ordinary = [], None, False
        for cand in reversed(self._gens.values()):
            n = len(cand.messages)
            if cand.fp != fp or n + 2 > len(msgs):
                continue
            if msgs[:n] != cand.messages:
                continue
            res = self._continue(cand, msgs)
            if cand.finished and not cand.cut:
                # Ended on its own. The client saw all of it -> an ordinary next turn (template
                # rules apply). Only part of it -> the client cut it on its side and the engine
                # finished before it noticed the disconnect (polled every 0.5 s): a steer.
                if isinstance(res, str) or res[1]:
                    ordinary = True
                    continue
            if isinstance(res, str):
                why = why or (cand.uid, res)
            else:
                ok.append((cand, res[0]))
        if not ok:
            if ordinary:
                return None, None
            if why is not None:
                logger.info("steer fallback (extends uid %d): %s", *why)
            return None, None
        ended = [x for x in ok if x[0].finished or x[0].cut]
        running = [x for x in ok if not (x[0].finished or x[0].cut)]
        if ended:
            g, out = ended[0]
            cut_uid, how = None, "already ended"
        elif len(running) == 1:
            g, out = running[0]
            cut_uid, how = g.uid, "cut now"
        else:
            g, out = running[0]
            cut_uid, how = None, f"{len(running)} identical running generations, none cut"
        if cut_uid is not None:
            g.cut = True
            g.t_end = g.t_end or time.monotonic()
        newpart = out[len(g.messages)]
        logger.info(
            "steer: continues uid %d (%s), keeps %d reasoning + %d content chars, then %d new message(s)",
            g.uid, how, len(newpart[STEER_KEY]), len(newpart["content"]), len(out) - len(g.messages) - 1,
        )
        return out, cut_uid

    @staticmethod
    def _continue(g: _Gen, msgs: list) -> "tuple[list, bool] | str":
        """(the steer rendering of ``msgs`` as a continuation of ``g``, whether the client's partial
        answer is ALL of g's visible output), or why it is not a continuation."""
        n = len(g.messages)
        part, rest = msgs[n], msgs[n + 1:]
        if part.get("role") != "assistant":
            return "no partial answer after the cut turn"
        if not any(m.get("role") == "user" for m in rest):
            return "no new user message"
        if part.get("tool_calls"):
            return "partial answer carries tool calls"
        content = part.get("content") or ""
        if not isinstance(content, str):
            return "partial answer content is not plain text"
        raw = "".join(g.text)
        if THINK_END in raw:
            reasoning, body = raw.split(THINK_END, 1)
        else:
            reasoning, body = raw, ""
        c_cli = content.strip()
        if c_cli:
            body_keep = _cut(body, c_cli) if THINK_END in raw else None
            if body_keep is None:
                return "partial content is not what the model wrote" + _diff(body.lstrip(), c_cli)
            reasoning_keep = reasoning                 # reasoning was complete
        else:
            body_keep = ""
            r_cli = (part.get("reasoning_content") or "").strip()
            if r_cli:
                reasoning_keep = _cut(reasoning, r_cli)
                if reasoning_keep is None:
                    return "partial reasoning is not what the model wrote" + _diff(reasoning.lstrip(), r_cli)
            else:
                reasoning_keep = reasoning             # everything generated before the cut
        if not reasoning_keep.strip() and not body_keep:
            return "nothing was generated before the cut"
        newpart = dict(part)
        newpart["content"] = body_keep
        newpart[STEER_KEY] = reasoning_keep
        vis = lambda t: _SPECIAL.sub("", t).strip()  # noqa: E731
        complete = vis(reasoning_keep) == vis(reasoning) and vis(body_keep) == vis(body)
        return msgs[:n] + [newpart] + list(rest), complete
