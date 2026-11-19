"""Server-side web search for the request filter, the way hosted APIs run their built-in tools.

A client declares the hosted tool; the model calls it; this module runs the search and continues the
generation itself. The engine only ever sees an ordinary function tool `web_search(query)`.

  Anthropic Messages   tools: [{"type": "web_search_20250305", "name": "web_search", "max_uses": 5,
                                "allowed_domains": [...], "blocked_domains": [...]}]
                       -> server_tool_use + web_search_tool_result blocks, usage.server_tool_use
  OpenAI Responses     tools: [{"type": "web_search"}]   (also web_search_preview*)
                       -> web_search_call output items (action.type = "search")
  Chat Completions     "web_search_options": {...}  or  tools: [{"type": "web_search"}]
                       -> searches stay internal; the client receives the final answer

The search backend (WEBSEARCH_URL, loopback only) is not reachable from outside: only the model, through
this loop, can query it. Each upstream round is a normal streaming request to the engine; the prompt of
round n+1 is round n's prompt + its output + the tool results, so the prefix cache serves it.
"""
from __future__ import annotations

import asyncio, base64, copy, json, logging, os, time, uuid, zlib
from collections import OrderedDict
from typing import Any, AsyncIterator, Awaitable, Callable

import httpx

log = logging.getLogger("reqfilter.websearch")

WEBSEARCH_URL = os.environ.get("WEBSEARCH_URL", "http://127.0.0.1:1930")
TOOL = "web_search"
DESC = ("Search the web. Returns the top results with title, URL, publication date and the page text. "
        "Use it for recent events, facts that may have changed, or anything you are not sure about. "
        "Write the query in the language most likely to find good sources; search again with a "
        "different query if the results do not answer the question.")
SCHEMA = {"type": "object", "properties": {"query": {"type": "string", "description": "the search query"}},
          "required": ["query"]}
MAX_QUERY_CHARS = 400
KEEPALIVE_S = 5.0

ws_client = httpx.AsyncClient(base_url=WEBSEARCH_URL, timeout=httpx.Timeout(40.0, connect=5.0), trust_env=False)

Post = Callable[[str, bytes], Awaitable[httpx.Response]]


# ----------------------------------------------------------------------------- search + result text

class SearchError(Exception):
    def __init__(self, code: str, msg: str = ""):
        super().__init__(msg or code)
        self.code = code


async def run_search(query: str, site: list[str] | None = None, exclude: list[str] | None = None) -> list[dict]:
    """-> [{"title", "url", "published", "body"}]; raises SearchError(code)."""
    if not query.strip():
        raise SearchError("invalid_tool_input", "empty query")
    if len(query) > MAX_QUERY_CHARS:
        raise SearchError("query_too_long")
    params = {"q": query, "n": 8, "fetch": 5}
    if site:
        params["site"] = ",".join(site)
    if exclude:
        params["exclude"] = ",".join(exclude)
    t0 = time.monotonic()
    try:
        r = await ws_client.get("/search", params=params)
    except Exception as e:  # noqa: BLE001
        log.error("web_search %r: backend unreachable: %s: %s", query, type(e).__name__, e)
        raise SearchError("unavailable", f"{type(e).__name__}: {e}")
    if r.status_code != 200:
        log.error("web_search %r: backend HTTP %d: %s", query, r.status_code, r.text[:300])
        raise SearchError("unavailable", f"HTTP {r.status_code}")
    out = []
    for x in r.json().get("results", []):
        out.append({"title": x.get("title") or x["url"], "url": x["url"], "published": x.get("published") or None,
                    "body": (x.get("text") or x.get("snippet") or "").strip()})
    log.info("web_search %r -> %d results %.1fs", query, len(out), time.monotonic() - t0)
    return out


def result_text(query: str, results: list[dict]) -> str:
    """What the model reads. Deterministic in (query, results): a replayed history renders the same bytes."""
    if not results:
        return f'Web search results for "{query}": no results.'
    parts = [f'Web search results for "{query}":']
    for i, r in enumerate(results, 1):
        head = f"[{i}] {r['title']}\nURL: {r['url']}"
        if r.get("published"):
            head += f"\nPublished: {r['published']}"
        parts.append(head + ("\n" + r["body"] if r.get("body") else ""))
    return "\n\n".join(parts)


def error_text(query: str, code: str) -> str:
    return f'Web search for "{query}" failed: {code}.'


def _pack(s: str) -> str:
    return base64.b64encode(zlib.compress(s.encode())).decode()


def _unpack(s: str | None) -> str:
    try:
        return zlib.decompress(base64.b64decode(s or "")).decode()
    except Exception:  # noqa: BLE001  (not ours, e.g. a hosted API's encrypted payload)
        return ""


class _LRU(OrderedDict):
    def __init__(self, cap: int):
        super().__init__()
        self.cap = cap

    def put(self, k, v):
        self[k] = v
        self.move_to_end(k)
        while len(self) > self.cap:
            self.popitem(last=False)


# Responses web_search_call items carry no result payload: the text the model read is kept here, by item id,
# so a client that replays the item gets the same tool result back into the prompt.
_responses_results = _LRU(4096)


def _sum_usage(a: dict | None, b: dict | None) -> dict:
    a = copy.deepcopy(a or {})
    for k, v in (b or {}).items():
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            a[k] = (a.get(k) or 0) + v if isinstance(a.get(k) or 0, (int, float)) else v
        elif isinstance(v, dict):
            a[k] = _sum_usage(a.get(k) if isinstance(a.get(k), dict) else {}, v)
        elif k not in a:
            a[k] = v
    return a


async def _sse(resp: httpx.Response) -> AsyncIterator[tuple[str | None, str]]:
    """(event, data) per SSE message; comment lines are dropped."""
    ev, data = None, []
    async for line in resp.aiter_lines():
        if line == "":
            if data:
                yield ev, "\n".join(data)
            ev, data = None, []
        elif line.startswith(":"):
            continue
        elif line.startswith("event:"):
            ev = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
    if data:
        yield ev, "\n".join(data)


async def _with_keepalive(coro, ping: Callable[[], bytes]):
    """Run `coro`, yielding ping frames every KEEPALIVE_S while it runs; the last item is ("done", result)."""
    task = asyncio.ensure_future(coro)
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=KEEPALIVE_S)
            if done:
                break
            yield ping()
        yield ("done", task.result())
    finally:
        if not task.done():
            task.cancel()


async def _search_or_error(query: str, site, exclude) -> tuple[list[dict] | None, str | None]:
    try:
        return await run_search(query, site, exclude), None
    except SearchError as e:
        return None, e.code


def _args(raw: str) -> dict:
    try:
        v = json.loads(raw or "{}")
        return v if isinstance(v, dict) else {}
    except ValueError:
        return {}


# ----------------------------------------------------------------------------- base

class Loop:
    """One client request: prepares the engine request, then runs search rounds."""

    path = ""

    def __init__(self, doc: dict, max_uses: int):
        self.doc = doc
        self.stream = doc.get("stream") is True
        self.max_uses = max_uses
        self.uses = 0
        self.usage: dict | None = None
        self.site: list[str] = []
        self.exclude: list[str] = []
        self.error: str | None = None

    async def _drain(self, first, post) -> None:
        async for _ in self.events(first, post):
            pass
        if self.error:
            raise RuntimeError(self.error)

    def first_body(self) -> bytes:
        return json.dumps(self.body, ensure_ascii=False).encode()

    async def events(self, first: httpx.Response, post: Post) -> AsyncIterator[bytes]:
        raise NotImplementedError

    async def collect(self, first: httpx.Response, post: Post) -> dict:
        raise NotImplementedError

    async def _search(self, query: str):
        """-> (results | None, error_code | None), counting uses."""
        if self.uses >= self.max_uses:
            return None, "max_uses_exceeded"
        self.uses += 1
        return await _search_or_error(query, self.site, self.exclude)


async def _next_round(post: Post, path: str, body: dict) -> httpx.Response:
    resp = await post(path, json.dumps(body, ensure_ascii=False).encode())
    if resp.status_code != 200:
        text = (await resp.aread()).decode(errors="replace")[:500]
        await resp.aclose()
        raise RuntimeError(f"engine HTTP {resp.status_code} on search round: {text}")
    return resp


# ----------------------------------------------------------------------------- Anthropic Messages

def _is_anthropic_ws(t) -> bool:
    return isinstance(t, dict) and str(t.get("type", "")).startswith("web_search_")


def anthropic_history(messages: list) -> list:
    """server_tool_use / web_search_tool_result blocks (ours, replayed by the client) -> the tool_use /
    tool_result turns the engine rendered them as. Other messages are returned unchanged."""
    if not any(isinstance(m, dict) and isinstance(m.get("content"), list) and
               any(isinstance(b, dict) and b.get("type") in ("server_tool_use", "web_search_tool_result")
                   for b in m["content"]) for m in messages):
        return messages
    out = []
    for m in messages:
        if not (isinstance(m, dict) and m.get("role") == "assistant" and isinstance(m.get("content"), list)):
            out.append(m)
            continue
        queries: dict[str, str] = {}
        cur: list = []
        for b in m["content"]:
            t = b.get("type") if isinstance(b, dict) else None
            if t == "server_tool_use":
                if b.get("name") != TOOL:
                    continue
                queries[b.get("id")] = str((b.get("input") or {}).get("query", ""))
                cur.append({"type": "tool_use", "id": b.get("id"), "name": TOOL, "input": b.get("input") or {}})
            elif t == "web_search_tool_result":
                tid = b.get("tool_use_id")
                if tid not in queries:
                    continue
                q, c = queries[tid], b.get("content")
                if isinstance(c, list):
                    text = result_text(q, [{"title": r.get("title") or r.get("url"), "url": r.get("url"),
                                            "published": r.get("page_age"), "body": _unpack(r.get("encrypted_content"))}
                                           for r in c if isinstance(r, dict)])
                else:
                    text = error_text(q, (c or {}).get("error_code", "unavailable"))
                if cur:
                    out.append({"role": "assistant", "content": cur})
                out.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": tid, "content": text}]})
                cur = []
            else:
                cur.append(b)
        if cur:
            out.append({"role": "assistant", "content": cur})
    return out


def anthropic_rewrite_tools(doc: dict) -> dict | None:
    """Replace the hosted web_search tool by the function tool; returns the hosted tool's config or None."""
    tools = doc.get("tools")
    if not isinstance(tools, list):
        return None
    hosted = [t for t in tools if _is_anthropic_ws(t)]
    if not hosted:
        return None
    doc["tools"] = [t for t in tools if not _is_anthropic_ws(t) and not (isinstance(t, dict) and t.get("name") == TOOL)]
    doc["tools"].append({"name": TOOL, "description": DESC, "input_schema": SCHEMA})
    return hosted[0]


class AnthropicLoop(Loop):
    path = "/v1/messages"

    def __init__(self, doc: dict, hosted: dict, default_max_uses: int):
        super().__init__(doc, int(hosted.get("max_uses") or default_max_uses))
        self.site = [str(d).lower() for d in hosted.get("allowed_domains") or []]
        self.exclude = [str(d).lower() for d in hosted.get("blocked_domains") or []]
        self.body = dict(doc, stream=True)
        self.blocks: list[dict] = []          # client-visible content, in client index order
        self.stop_reason = None
        self.stop_sequence = None
        self.msg_start: dict | None = None

    def _frame(self, obj: dict) -> bytes:
        return f"event: {obj['type']}\ndata: {json.dumps(obj, ensure_ascii=False)}\n\n".encode()

    def _ping(self) -> bytes:
        return self._frame({"type": "ping"})

    def _flush(self, s: dict):
        """Give a held block its client index and send what was held. Whitespace-only text blocks (the
        model's blank line before a tool call) are never flushed: kept in the round's history only."""
        s["out"] = len(self.blocks)
        self.blocks.append(s["block"])
        for ev in s["held"]:
            if ev["type"] == "content_block_start":
                ev = dict(ev, content_block=dict(ev["content_block"], text="")) if ev["content_block"].get("type") == "text" else ev
            yield self._frame(dict(ev, index=s["out"]))
        s["held"] = []

    async def events(self, first, post):
        resp = first
        try:
            while True:
                turn: list[dict] = []          # this round's assistant blocks, engine form
                calls: list[dict] = []         # web_search calls of this round
                client_tools = False
                local: dict[int, dict] = {}    # upstream index -> {"kind", "out", "block"}
                stop = None
                async for _, data in _sse(resp):
                    ev = json.loads(data)
                    t = ev.get("type")
                    if t == "message_start":
                        u = (ev.get("message") or {}).get("usage") or {}
                        if self.msg_start is None:
                            self.msg_start = ev["message"]
                            yield self._frame(ev)
                        self.usage = _sum_usage(self.usage, {k: v for k, v in u.items() if k != "output_tokens"})
                    elif t == "content_block_start":
                        cb = ev["content_block"]
                        if cb.get("type") == "tool_use" and cb.get("name") == TOOL:
                            c = {"kind": "call", "id": "srvtoolu_" + uuid.uuid4().hex[:24], "json": ""}
                            local[ev["index"]] = c
                            calls.append(c)
                        else:
                            if cb.get("type") == "tool_use":
                                client_tools = True
                            blk = copy.deepcopy(cb)
                            turn.append(blk)
                            s = {"kind": "fwd", "out": None, "block": blk, "json": "", "held": [ev]}
                            local[ev["index"]] = s
                            if cb.get("type") != "text" or (cb.get("text") or "").strip():
                                for f in self._flush(s):
                                    yield f
                    elif t == "content_block_delta":
                        s = local.get(ev["index"])
                        if s is None:
                            continue
                        d = ev["delta"]
                        if s["kind"] == "call":
                            s["json"] += d.get("partial_json", "")
                            continue
                        b = s["block"]
                        if d.get("type") == "text_delta":
                            b["text"] = (b.get("text") or "") + d.get("text", "")
                        elif d.get("type") == "thinking_delta":
                            b["thinking"] = (b.get("thinking") or "") + d.get("thinking", "")
                        elif d.get("type") == "signature_delta":
                            b["signature"] = (b.get("signature") or "") + d.get("signature", "")
                        elif d.get("type") == "input_json_delta":
                            s["json"] += d.get("partial_json", "")
                        if s["out"] is None:           # whitespace-only text so far: held back
                            s["held"].append(ev)
                            if (b.get("text") or "").strip():
                                for f in self._flush(s):
                                    yield f
                            continue
                        yield self._frame(dict(ev, index=s["out"]))
                    elif t == "content_block_stop":
                        s = local.get(ev["index"])
                        if s is None or s["kind"] == "call" or s["out"] is None:
                            continue
                        if s["block"].get("type") == "tool_use" and s["json"]:
                            s["block"]["input"] = _args(s["json"])
                        yield self._frame(dict(ev, index=s["out"]))
                    elif t == "message_delta":
                        stop = (ev.get("delta") or {}).get("stop_reason")
                        self.stop_sequence = (ev.get("delta") or {}).get("stop_sequence")
                        u = ev.get("usage") or {}
                        self.usage = _sum_usage(self.usage, {"output_tokens": u.get("output_tokens", 0)})
                    elif t == "ping":
                        yield self._frame(ev)
                    elif t == "error":
                        yield self._frame(ev)
                        return
                await resp.aclose()
                if not calls or stop != "tool_use":
                    self.stop_reason = stop
                    break
                # -- run this round's searches, show them to the client, feed the results back
                results_msg = []
                for c in calls:
                    a = _args(c["json"])
                    q = str(a.get("query", ""))
                    out = len(self.blocks)
                    stu = {"type": "server_tool_use", "id": c["id"], "name": TOOL, "input": a}
                    self.blocks.append(stu)
                    yield self._frame({"type": "content_block_start", "index": out,
                                       "content_block": {"type": "server_tool_use", "id": c["id"], "name": TOOL, "input": {}}})
                    yield self._frame({"type": "content_block_delta", "index": out,
                                       "delta": {"type": "input_json_delta", "partial_json": json.dumps(a, ensure_ascii=False)}})
                    yield self._frame({"type": "content_block_stop", "index": out})
                    res = err = None
                    async for item in _with_keepalive(self._search(q), self._ping):
                        if isinstance(item, tuple):
                            res, err = item[1]
                        else:
                            yield item
                    if err:
                        content: Any = {"type": "web_search_tool_result_error", "error_code": err}
                        text = error_text(q, err)
                    else:
                        content = [{"type": "web_search_result", "url": r["url"], "title": r["title"],
                                    "page_age": r["published"], "encrypted_content": _pack(r["body"])} for r in res]
                        text = result_text(q, res)
                    out = len(self.blocks)
                    wsr = {"type": "web_search_tool_result", "tool_use_id": c["id"], "content": content}
                    self.blocks.append(wsr)
                    yield self._frame({"type": "content_block_start", "index": out, "content_block": wsr})
                    yield self._frame({"type": "content_block_stop", "index": out})
                    turn.append({"type": "tool_use", "id": c["id"], "name": TOOL, "input": a})
                    results_msg.append({"type": "tool_result", "tool_use_id": c["id"], "content": text})
                if client_tools:               # the client must answer its own calls first
                    self.stop_reason = "tool_use"
                    break
                self.body["messages"] = list(self.body["messages"]) + [
                    {"role": "assistant", "content": turn}, {"role": "user", "content": results_msg}]
                tc = self.body.get("tool_choice")
                if isinstance(tc, dict) and tc.get("type") in ("any", "tool"):
                    self.body["tool_choice"] = {"type": "auto"}
                resp = await _next_round(post, self.path, self.body)
            self.usage = _sum_usage(self.usage, {"server_tool_use": {"web_search_requests": self.uses}})
            yield self._frame({"type": "message_delta",
                               "delta": {"stop_reason": self.stop_reason, "stop_sequence": self.stop_sequence},
                               "usage": self.usage})
            yield self._frame({"type": "message_stop"})
        except Exception as e:  # noqa: BLE001
            log.error("anthropic web_search loop: %s: %s", type(e).__name__, e)
            yield self._frame({"type": "error", "error": {"type": "api_error", "message": f"{type(e).__name__}: {e}"}})
            self.error = f"{type(e).__name__}: {e}"
        finally:
            await resp.aclose()

    async def collect(self, first, post):
        await self._drain(first, post)
        m = dict(self.msg_start or {"id": "msg_" + uuid.uuid4().hex, "type": "message", "role": "assistant",
                                    "model": self.doc.get("model")})
        m.update(content=self.blocks, stop_reason=self.stop_reason, stop_sequence=self.stop_sequence, usage=self.usage)
        return m


# ----------------------------------------------------------------------------- OpenAI Responses

RESPONSES_HOSTED = ("web_search", "web_search_preview", "web_search_2025_08_26", "web_search_preview_2025_03_11")


def responses_history(inp):
    """web_search_call items (ours) -> function_call + function_call_output."""
    if not isinstance(inp, list) or not any(isinstance(i, dict) and i.get("type") == "web_search_call" for i in inp):
        return inp
    out = []
    for i in inp:
        if isinstance(i, dict) and i.get("type") == "web_search_call":
            q = str(((i.get("action") or {}).get("query")) or "")
            text = _responses_results.get(i.get("id")) or f'Web search results for "{q}": no longer available.'
            out.append({"type": "function_call", "call_id": i.get("id"), "name": TOOL,
                        "arguments": json.dumps({"query": q}, ensure_ascii=False)})
            out.append({"type": "function_call_output", "call_id": i.get("id"), "output": text})
        else:
            out.append(i)
    return out


def responses_rewrite_tools(doc: dict) -> bool:
    tools = doc.get("tools")
    if not isinstance(tools, list) or not any(isinstance(t, dict) and t.get("type") in RESPONSES_HOSTED for t in tools):
        return False
    doc["tools"] = [t for t in tools if not (isinstance(t, dict) and (t.get("type") in RESPONSES_HOSTED or t.get("name") == TOOL))]
    doc["tools"].append({"type": "function", "name": TOOL, "description": DESC, "parameters": SCHEMA})
    tc = doc.get("tool_choice")
    if isinstance(tc, dict) and tc.get("type") in RESPONSES_HOSTED:
        doc["tool_choice"] = {"type": "function", "name": TOOL}
    return True


class ResponsesLoop(Loop):
    path = "/v1/responses"

    def __init__(self, doc: dict, default_max_uses: int):
        super().__init__(doc, default_max_uses)
        self.body = dict(doc, stream=True)
        if isinstance(self.body.get("input"), str):
            self.body["input"] = [{"type": "message", "role": "user", "content": self.body["input"]}]
        self.items: list[dict] = []            # client-visible output items
        self.seq = 0
        self.snapshot: dict | None = None
        self.final: dict | None = None
        self.final_type = "response.completed"
        self.rounds = 0

    def _frame(self, obj: dict) -> bytes:
        obj = dict(obj, sequence_number=self.seq)
        self.seq += 1
        return f"event: {obj['type']}\ndata: {json.dumps(obj, ensure_ascii=False)}\n\n".encode()

    def _ping(self) -> bytes:
        return b": keepalive\n\n"

    def _flush(self, s: dict):
        s["out"] = len(self.items)
        self.items.append(s["item"])
        for ev in s["held"]:
            yield self._frame(dict(ev, output_index=s["out"]))
        s["held"] = []

    async def events(self, first, post):
        resp = first
        try:
            while True:
                turn: list[dict] = []
                calls: list[dict] = []
                client_tools = False
                local: dict[int, dict] = {}
                status = None
                async for _, data in _sse(resp):
                    ev = json.loads(data)
                    t = ev.get("type", "")
                    if t in ("response.created", "response.in_progress"):
                        if self.rounds == 0:
                            if self.snapshot is None:
                                self.snapshot = ev["response"]
                            yield self._frame(dict(ev, response=dict(ev["response"], id=self.snapshot["id"])))
                        continue
                    if t in ("response.completed", "response.incomplete", "response.failed"):
                        self.final = ev["response"]
                        self.final_type = t
                        status = self.final.get("status")
                        self.usage = _sum_usage(self.usage, self.final.get("usage"))
                        continue
                    if t == "error":
                        yield self._frame(ev)
                        return
                    oi = ev.get("output_index")
                    if t == "response.output_item.added":
                        it = ev["item"]
                        if it.get("type") == "function_call" and it.get("name") == TOOL:
                            c = {"kind": "call", "args": it.get("arguments") or ""}
                            local[oi] = c
                            calls.append(c)
                            continue
                        if it.get("type") == "function_call":
                            client_tools = True
                        s = {"kind": "fwd", "out": None, "item": it, "held": [ev], "text": ""}
                        local[oi] = s
                        if it.get("type") != "message":
                            for f in self._flush(s):
                                yield f
                        continue
                    s = local.get(oi)
                    if s is None:
                        yield self._frame(ev)
                        continue
                    if s["kind"] == "fwd" and s["out"] is None:
                        if t == "response.output_item.done":
                            turn.append(ev["item"])     # whitespace-only message: history only
                            continue
                        s["held"].append(ev)
                        s["text"] += ev.get("delta", "") if t == "response.output_text.delta" else ""
                        if s["text"].strip():
                            for f in self._flush(s):
                                yield f
                        continue
                    if s["kind"] == "call":
                        if t == "response.function_call_arguments.delta":
                            s["args"] += ev.get("delta", "")
                        elif t == "response.function_call_arguments.done":
                            s["args"] = ev.get("arguments", s["args"])
                        elif t == "response.output_item.done":
                            s["args"] = ev["item"].get("arguments") or s["args"]
                        continue
                    if t == "response.output_item.done":
                        self.items[s["out"]] = ev["item"]
                        turn.append(ev["item"])
                    yield self._frame(dict(ev, output_index=s["out"]))
                await resp.aclose()
                if not calls or status != "completed":
                    break
                feed = []
                for c in calls:
                    a = _args(c["args"])
                    q = str(a.get("query", ""))
                    wid = "ws_" + uuid.uuid4().hex[:24]
                    out = len(self.items)
                    item = {"type": "web_search_call", "id": wid, "status": "in_progress", "action": {"type": "search", "query": q}}
                    self.items.append(item)
                    yield self._frame({"type": "response.output_item.added", "output_index": out, "item": item})
                    yield self._frame({"type": "response.web_search_call.in_progress", "output_index": out, "item_id": wid})
                    yield self._frame({"type": "response.web_search_call.searching", "output_index": out, "item_id": wid})
                    res = err = None
                    async for x in _with_keepalive(self._search(q), self._ping):
                        if isinstance(x, tuple):
                            res, err = x[1]
                        else:
                            yield x
                    text = error_text(q, err) if err else result_text(q, res)
                    _responses_results.put(wid, text)
                    done = {"type": "web_search_call", "id": wid, "status": "failed" if err else "completed",
                            "action": {"type": "search", "query": q,
                                       "sources": [{"type": "url", "url": r["url"]} for r in res or []]}}
                    self.items[out] = done
                    yield self._frame({"type": "response.web_search_call.completed", "output_index": out, "item_id": wid})
                    yield self._frame({"type": "response.output_item.done", "output_index": out, "item": done})
                    feed += [{"type": "function_call", "call_id": wid, "name": TOOL,
                              "arguments": json.dumps(a, ensure_ascii=False)},
                             {"type": "function_call_output", "call_id": wid, "output": text}]
                if client_tools:
                    break
                self.rounds += 1
                self.body["input"] = list(self.body["input"]) + turn + feed
                tc = self.body.get("tool_choice")
                if tc == "required" or isinstance(tc, dict):
                    self.body["tool_choice"] = "auto"
                resp = await _next_round(post, self.path, self.body)
            yield self._frame({"type": self.final_type, "response": self._response()})
        except Exception as e:  # noqa: BLE001
            log.error("responses web_search loop: %s: %s", type(e).__name__, e)
            yield self._frame({"type": "error", "code": "server_error", "message": f"{type(e).__name__}: {e}", "param": None})
            self.error = f"{type(e).__name__}: {e}"
        finally:
            await resp.aclose()

    def _response(self) -> dict:
        r = dict(self.final or self.snapshot or {})
        if self.snapshot:
            r["id"] = self.snapshot["id"]
            r["created_at"] = self.snapshot.get("created_at", r.get("created_at"))
        r["output"] = self.items
        r["usage"] = self.usage
        r["tools"] = self.doc.get("tools", r.get("tools"))
        return r

    async def collect(self, first, post):
        await self._drain(first, post)
        return self._response()


# ----------------------------------------------------------------------------- Chat Completions

def chat_rewrite_tools(doc: dict) -> bool:
    tools = doc.get("tools") if isinstance(doc.get("tools"), list) else []
    hosted = "web_search_options" in doc or any(isinstance(t, dict) and t.get("type") in RESPONSES_HOSTED for t in tools)
    if not hosted:
        return False
    doc.pop("web_search_options", None)
    doc["tools"] = [t for t in tools if not (isinstance(t, dict) and (t.get("type") in RESPONSES_HOSTED or
                                                                     (t.get("function") or {}).get("name") == TOOL))]
    doc["tools"].append({"type": "function", "function": {"name": TOOL, "description": DESC, "parameters": SCHEMA}})
    return True


class ChatLoop(Loop):
    path = "/v1/chat/completions"

    def __init__(self, doc: dict, default_max_uses: int):
        super().__init__(doc, default_max_uses)
        self.want_usage = bool((doc.get("stream_options") or {}).get("include_usage"))
        self.body = dict(doc, stream=True, stream_options={"include_usage": True})
        self.content = ""
        self.reasoning = ""
        self.client_calls: list[dict] = []
        self.finish = None
        self.meta: dict | None = None          # id / created / model of the first chunk
        self.sent_role = False

    def _frame(self, obj: dict) -> bytes:
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode()

    def _ping(self) -> bytes:
        return b": keepalive\n\n"

    def _chunk(self, delta: dict, finish=None) -> dict:
        m = self.meta or {}
        return {"id": m.get("id"), "object": "chat.completion.chunk", "created": m.get("created"), "model": m.get("model"),
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

    async def events(self, first, post):
        resp = first
        try:
            while True:
                calls: dict[int, dict] = {}    # upstream tool index -> web_search call
                fwd: dict[int, int] = {}       # upstream tool index -> client index
                turn_calls: list[dict] = []
                content = reasoning = pending = ""
                finish = None
                async for _, data in _sse(resp):
                    if data.strip() == "[DONE]":
                        break
                    ch = json.loads(data)
                    if "error" in ch and not ch.get("choices"):
                        yield self._frame(ch)
                        return
                    if self.meta is None:
                        self.meta = {k: ch.get(k) for k in ("id", "created", "model")}
                    if ch.get("usage"):
                        self.usage = _sum_usage(self.usage, ch["usage"])
                    for c in ch.get("choices") or []:
                        d = dict(c.get("delta") or {})
                        if c.get("finish_reason"):
                            finish = c["finish_reason"]
                        if d.get("content"):
                            content += d["content"]
                            if not self.content:       # nothing shown yet: hold leading whitespace
                                pending += d["content"]
                                if pending.strip():
                                    d["content"] = pending.lstrip()
                                    self.content = d["content"]
                                else:
                                    d.pop("content")
                            else:
                                self.content += d["content"]
                        reasoning += d.get("reasoning_content") or ""
                        tcs = []
                        for tc in d.get("tool_calls") or []:
                            i = tc.get("index", 0)
                            fn = tc.get("function") or {}
                            if i not in calls and i not in fwd:
                                if fn.get("name") == TOOL:
                                    calls[i] = {"id": tc.get("id") or "call_" + uuid.uuid4().hex[:24], "args": ""}
                                else:
                                    fwd[i] = len(self.client_calls)
                                    self.client_calls.append({"id": tc.get("id"), "type": "function",
                                                              "function": {"name": fn.get("name"), "arguments": ""}})
                            if i in calls:
                                calls[i]["args"] += fn.get("arguments") or ""
                                continue
                            cc = self.client_calls[fwd[i]]
                            cc["function"]["arguments"] += fn.get("arguments") or ""
                            tcs.append(dict(tc, index=fwd[i]))
                        if "tool_calls" in d:
                            if tcs:
                                d["tool_calls"] = tcs
                            else:
                                d.pop("tool_calls")
                        if d.get("role"):
                            if self.sent_role:
                                d.pop("role")
                            self.sent_role = True
                        if d:
                            yield self._frame(self._chunk(d))
                self.reasoning += reasoning
                await resp.aclose()
                if not calls or finish != "tool_calls":
                    self.finish = finish
                    break
                feed = []
                for c in calls.values():
                    a = _args(c["args"])
                    q = str(a.get("query", ""))
                    res = err = None
                    async for x in _with_keepalive(self._search(q), self._ping):
                        if isinstance(x, tuple):
                            res, err = x[1]
                        else:
                            yield x
                    turn_calls.append({"id": c["id"], "type": "function",
                                       "function": {"name": TOOL, "arguments": json.dumps(a, ensure_ascii=False)}})
                    feed.append({"role": "tool", "tool_call_id": c["id"], "content": error_text(q, err) if err else result_text(q, res)})
                if fwd:
                    self.finish = "tool_calls"
                    break
                asst = {"role": "assistant", "content": content or None, "tool_calls": turn_calls}
                if reasoning:
                    asst["reasoning_content"] = reasoning
                self.body["messages"] = list(self.body["messages"]) + [asst] + feed
                if self.body.get("tool_choice") not in (None, "auto", "none"):
                    self.body["tool_choice"] = "auto"
                resp = await _next_round(post, self.path, self.body)
            yield self._frame(self._chunk({}, self.finish))
            if self.want_usage:
                yield self._frame(dict(self._chunk({}), choices=[], usage=self.usage))
            yield b"data: [DONE]\n\n"
        except Exception as e:  # noqa: BLE001
            log.error("chat web_search loop: %s: %s", type(e).__name__, e)
            yield self._frame({"error": {"type": "server_error", "message": f"{type(e).__name__}: {e}"}})
            self.error = f"{type(e).__name__}: {e}"
        finally:
            await resp.aclose()

    async def collect(self, first, post):
        await self._drain(first, post)
        m = self.meta or {}
        msg: dict = {"role": "assistant", "content": self.content or None}
        if self.reasoning:
            msg["reasoning_content"] = self.reasoning
        if self.client_calls:
            msg["tool_calls"] = self.client_calls
        return {"id": m.get("id"), "object": "chat.completion", "created": m.get("created"), "model": m.get("model"),
                "choices": [{"index": 0, "message": msg, "finish_reason": self.finish}], "usage": self.usage}


# ----------------------------------------------------------------------------- entry

def prepare(path: str, doc: dict, enabled: bool, default_max_uses: int) -> Loop | None:
    """Rewrites `doc` in place (hosted tool -> function tool, replayed search items -> tool turns).
    Returns a Loop when this request needs server-side searching, else None (plain proxying)."""
    if path in ("/v1/messages", "/v1/messages/count_tokens"):
        if isinstance(doc.get("messages"), list):
            doc["messages"] = anthropic_history(doc["messages"])
        hosted = anthropic_rewrite_tools(doc)
        if hosted is None or path != "/v1/messages":
            return None
        if not enabled:
            doc["tools"] = [t for t in doc["tools"] if t.get("name") != TOOL]
            return None
        return AnthropicLoop(doc, hosted, default_max_uses)
    if path == "/v1/responses":
        doc["input"] = responses_history(doc.get("input"))
        if not responses_rewrite_tools(doc):
            return None
        if not enabled:
            doc["tools"] = [t for t in doc["tools"] if t.get("name") != TOOL]
            return None
        return ResponsesLoop(doc, default_max_uses)
    if path == "/v1/chat/completions":
        if not chat_rewrite_tools(doc):
            return None
        if not enabled:
            doc["tools"] = [t for t in doc["tools"] if (t.get("function") or {}).get("name") != TOOL]
            return None
        return ChatLoop(doc, default_max_uses)
    return None
