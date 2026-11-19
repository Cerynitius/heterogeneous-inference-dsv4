"""Request filter in front of `ft serve` (deploy/serve.sh).

External traffic reaches this proxy first and is forwarded to the engine on 127.0.0.1.
- Only inference endpoints pass (allow-list); cache rebuild, /generate, admin and stats routes get 403.
- Request bodies must be JSON objects; debug fields (ignore_eos) are removed.
- Generation requests with empty input get 400 "input tokens must be > 0".
- Non-streaming generation requests whose prompt is shorter than nonstream_min_chars characters get
  400 "backend doesn't support non-stream".
- Requests that declare the hosted web_search tool are served by servertools.py (server-side search).
- Streaming responses are passed through. When the client disconnects, the upstream connection is
  kept for disconnect_grace_s seconds, then closed, and the engine aborts the generation (a
  disconnect is also noticed while a non-streaming request is still waiting for its response).
Settings live in rules.json (REQFILTER_RULES); edits apply to the next request, no restart.
"""
import asyncio, json, logging, os, re
import httpx, uvicorn
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # run with python -I
import servertools  # noqa: E402
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

UPSTREAM = os.environ.get("REQFILTER_UPSTREAM", "http://127.0.0.1:1920")
FALLBACK = os.environ.get("REQFILTER_FALLBACK", "")   # second upstream when the first refuses; "" = none
LISTEN_PORT = int(os.environ.get("REQFILTER_PORT", "1922"))
ALLOW = [
    ("GET", re.compile(r"/health")),
    ("GET", re.compile(r"/v1/models")),
    ("POST", re.compile(r"/v1/chat/completions")),
    ("POST", re.compile(r"/v1/completions")),
    ("POST", re.compile(r"/v1/messages")),
    ("POST", re.compile(r"/v1/messages/count_tokens")),
    ("POST", re.compile(r"/v1/responses")),
    ("GET", re.compile(r"/v1/responses/[A-Za-z0-9_\-]{1,128}")),
    ("POST", re.compile(r"/v1/responses/[A-Za-z0-9_\-]{1,128}/cancel")),
]
STRIP_FIELDS = ("ignore_eos",)
HOP = {"host", "content-length", "connection", "keep-alive", "transfer-encoding", "te", "upgrade",
       "proxy-connection", "proxy-authorization"}
log = logging.getLogger("reqfilter")
RULES_FILE = os.environ.get("REQFILTER_RULES", os.path.join(os.path.dirname(os.path.abspath(__file__)), "rules.json"))
RULES_DEFAULT = {"nonstream_min_chars": 1000, "disconnect_grace_s": 15, "web_search": 1, "web_search_max_uses": 8,
                 "web_search_results": 8, "web_search_fetch": 5,
                 "upstream": UPSTREAM, "fallback": FALLBACK, "websearch_url": servertools.WEBSEARCH_URL}
RULES_URLS = ("upstream", "fallback", "websearch_url")   # strings; "fallback" may be "" (none)
_rules = {"key": None, "val": dict(RULES_DEFAULT)}
GEN_PATHS = ("/v1/chat/completions", "/v1/completions", "/v1/messages", "/v1/responses")
PROMPT_FIELDS = ("messages", "prompt", "input", "system", "instructions", "tools")
NONSTREAM_MSG = "backend doesn't support non-stream"


def rules() -> dict:
    """rules.json, re-read when its mtime/size changes; a bad file keeps the last good values."""
    try:
        st = os.stat(RULES_FILE)
        key = (st.st_mtime_ns, st.st_size)
    except FileNotFoundError:
        key = None
    if key != _rules["key"]:
        _rules["key"] = key
        val = dict(RULES_DEFAULT)
        if key is not None:
            try:
                doc = json.load(open(RULES_FILE))
                bad = sorted(set(doc) - set(RULES_DEFAULT))
                if bad:
                    raise ValueError(f"unknown keys {bad}; allowed {sorted(RULES_DEFAULT)}")
                for k in RULES_DEFAULT:
                    n = doc.get(k, val[k])
                    if k in RULES_URLS:
                        if not isinstance(n, str) or not (n.startswith("http") or (k == "fallback" and n == "")):
                            raise ValueError(f"{k} must be an http(s) URL{' or empty' if k == 'fallback' else ''}, got {n!r}")
                    elif not isinstance(n, (int, float)) or isinstance(n, bool) or n < 0:
                        raise ValueError(f"{k} must be a number >= 0, got {n!r}")
                    val[k] = n
            except Exception as e:  # noqa: BLE001
                log.error("rules.json rejected, keeping last values: %s: %s", type(e).__name__, e)
                return _rules["val"]
        _rules["val"] = val
        servertools.settings.update(websearch_url=val["websearch_url"], n=int(val["web_search_results"]),
                                    fetch=int(val["web_search_fetch"]))
        log.info("rules: %s", val)
    return _rules["val"]


def _chars(x) -> int:
    if isinstance(x, str):
        return len(x)
    if isinstance(x, dict):
        return sum(_chars(v) for k, v in x.items() if k not in ("role", "type"))
    if isinstance(x, list):
        return sum(_chars(v) for v in x)
    return 0


def prompt_chars(doc: dict, fields=PROMPT_FIELDS) -> int:
    return sum(_chars(doc.get(f)) for f in fields)


INPUT_FIELDS = ("messages", "prompt", "input", "system", "instructions")   # tool schemas are not input
_upstreams: dict[str, httpx.AsyncClient] = {}


def upstream_client(url: str) -> httpx.AsyncClient:
    """One pooled client per upstream URL (rules.json `upstream` / `fallback`, hot)."""
    c = _upstreams.get(url)
    if c is None:
        c = _upstreams[url] = httpx.AsyncClient(base_url=url, timeout=httpx.Timeout(None, connect=10.0),
                                                trust_env=False,
                                                limits=httpx.Limits(max_connections=64, max_keepalive_connections=16))
    return c


def deny(request: Request, status: int, why: str):
    log.warning("BLOCK %s %s %s from %s (xff=%s)", status, request.method, request.url.path,
                request.client.host if request.client else "?", request.headers.get("x-forwarded-for", "-"))
    return JSONResponse({"error": {"type": "forbidden", "message": why}}, status_code=status)


async def handle(request: Request):
    path, method = request.url.path, request.method
    if not any(m == method and rx.fullmatch(path) for m, rx in ALLOW):
        return deny(request, 403, f"{method} {path} is not allowed on this endpoint")
    body = b""
    loop = None
    if method == "POST":
        raw = await request.body()
        if path.endswith("/cancel") and not raw:
            body = b""
        else:
            try:
                doc = json.loads(raw or b"null")
            except ValueError:
                return deny(request, 400, "request body must be JSON")
            if not isinstance(doc, dict):
                return deny(request, 400, "request body must be a JSON object")
            for f in STRIP_FIELDS:
                doc.pop(f, None)
            if path in GEN_PATHS and prompt_chars(doc, INPUT_FIELDS) == 0:
                log.warning("BLOCK 400 empty input %s from %s (xff=%s)", path,
                            request.client.host if request.client else "?",
                            request.headers.get("x-forwarded-for", "-"))
                msg = "input tokens must be > 0"
                if path == "/v1/messages":
                    return JSONResponse({"type": "error", "error": {"type": "invalid_request_error", "message": msg}},
                                        status_code=400)
                return JSONResponse({"error": {"type": "invalid_request_error", "code": "empty_input", "message": msg}},
                                    status_code=400)
            if path in GEN_PATHS and doc.get("stream") is not True:
                n = prompt_chars(doc)
                if n < rules()["nonstream_min_chars"]:
                    log.warning("BLOCK 400 non-stream %s (%d prompt chars) from %s (xff=%s)", path, n,
                                request.client.host if request.client else "?",
                                request.headers.get("x-forwarded-for", "-"))
                    if path == "/v1/messages":
                        err = {"type": "error", "error": {"type": "invalid_request_error", "message": NONSTREAM_MSG}}
                    else:
                        err = {"error": {"type": "invalid_request_error", "code": "stream_required",
                                         "message": NONSTREAM_MSG}}
                    return JSONResponse(err, status_code=400)
            if path in ("/v1/messages", "/v1/messages/count_tokens", "/v1/responses", "/v1/chat/completions"):
                r = rules()
                loop = servertools.prepare(path, doc, bool(r["web_search"]), int(r["web_search_max_uses"]))
            body = json.dumps(doc, ensure_ascii=False).encode()
    headers = [(k, v) for k, v in request.headers.items() if k.lower() not in HOP]
    if body:
        headers = [(k, v) for k, v in headers if k.lower() != "content-type"] + [("content-type", "application/json")]
    if loop is not None:
        return await run_loop(request, loop, headers, path)
    disc = asyncio.create_task(_wait_disconnect(request))
    send = asyncio.create_task(_send(method, path, request.query_params, headers, body))
    await asyncio.wait({send, disc}, return_when=asyncio.FIRST_COMPLETED)
    if not send.done():
        # The client left before the response head (e.g. a non-stream request still generating).
        grace = rules()["disconnect_grace_s"]
        log.info("client gone before response on %s, keeping upstream %ss", path, grace)
        await asyncio.wait({send}, timeout=grace)
        if not send.done():
            send.cancel()
            await asyncio.gather(send, return_exceptions=True)
            log.info("upstream request on %s cancelled (client gone > %ss)", path, grace)
        elif send.exception() is None:
            await send.result().aclose()
        return Response(status_code=499)
    try:
        up = send.result()
    except httpx.HTTPError as e:
        disc.cancel()
        log.error("upstream error %s %s: %s: %s", method, path, type(e).__name__, e)
        return JSONResponse({"error": {"type": "upstream_unavailable", "message": f"{type(e).__name__}: {e}"}},
                            status_code=502)
    out_headers = {k: v for k, v in up.headers.items() if k.lower() not in HOP and k.lower() != "content-encoding"}
    return StreamingResponse(relay(up, disc, path), status_code=up.status_code, headers=out_headers)


async def run_loop(request: Request, loop, headers, path: str):
    """A request that declared the hosted web_search tool: servertools runs the search rounds."""
    async def post(p: str, b: bytes) -> httpx.Response:
        return await _send("POST", p, {}, headers, b)

    log.info("web_search loop on %s (stream=%s, max_uses=%d)", path, loop.stream, loop.max_uses)
    disc = asyncio.create_task(_wait_disconnect(request))
    send = asyncio.create_task(post(path, loop.first_body()))
    await asyncio.wait({send, disc}, return_when=asyncio.FIRST_COMPLETED)
    if not send.done():
        grace = rules()["disconnect_grace_s"]
        await asyncio.wait({send}, timeout=grace)
        if not send.done():
            send.cancel()
            await asyncio.gather(send, return_exceptions=True)
        elif send.exception() is None:
            await send.result().aclose()
        return Response(status_code=499)
    try:
        up = send.result()
    except httpx.HTTPError as e:
        disc.cancel()
        log.error("upstream error POST %s: %s: %s", path, type(e).__name__, e)
        return JSONResponse({"error": {"type": "upstream_unavailable", "message": f"{type(e).__name__}: {e}"}},
                            status_code=502)
    if up.status_code != 200:                  # engine rejected the request: pass its answer through
        out_headers = {k: v for k, v in up.headers.items() if k.lower() not in HOP and k.lower() != "content-encoding"}
        return StreamingResponse(relay(up, disc, path), status_code=up.status_code, headers=out_headers)
    if loop.stream:
        return StreamingResponse(relay_gen(loop.events(up, post), disc, path), status_code=200,
                                 headers={"content-type": "text/event-stream; charset=utf-8", "cache-control": "no-cache"})
    coll = asyncio.create_task(loop.collect(up, post))
    await asyncio.wait({coll, disc}, return_when=asyncio.FIRST_COMPLETED)
    if not coll.done():
        grace = rules()["disconnect_grace_s"]
        await asyncio.wait({coll}, timeout=grace)
        if not coll.done():
            coll.cancel()
            await asyncio.gather(coll, return_exceptions=True)
            return Response(status_code=499)
    disc.cancel()
    try:
        return JSONResponse(coll.result())
    except Exception as e:  # noqa: BLE001
        log.error("web_search loop on %s failed: %s: %s", path, type(e).__name__, e)
        return JSONResponse({"error": {"type": "server_error", "message": f"{type(e).__name__}: {e}"}}, status_code=502)


async def relay_gen(gen, disc: asyncio.Task, path: str):
    """relay() for a servertools event generator: same disconnect race and grace, then the generator is
    cancelled, which closes its upstream response and aborts the generation."""
    nxt = None
    finished = False
    try:
        while True:
            nxt = asyncio.ensure_future(gen.__anext__())
            await asyncio.wait({nxt, disc}, return_when=asyncio.FIRST_COMPLETED)
            if not nxt.done():
                break
            try:
                chunk = nxt.result()
            except StopAsyncIteration:
                finished = True
                break
            nxt = None
            yield chunk
    finally:
        disc.cancel()
        if not finished:
            t = asyncio.create_task(_close_gen_later(gen, nxt, path))
            _closers.add(t)
            t.add_done_callback(_closers.discard)


async def _close_gen_later(gen, nxt, path: str) -> None:
    grace = rules()["disconnect_grace_s"]
    log.info("client gone on %s (web_search loop), keeping it %ss", path, grace)
    await asyncio.sleep(grace)
    if nxt is not None and not nxt.done():
        nxt.cancel()
        await asyncio.gather(nxt, return_exceptions=True)
    else:
        await gen.aclose()
    log.info("web_search loop on %s closed", path)


async def _wait_disconnect(request: Request) -> None:
    """Resolves when the client goes away: the body is already read, so the next ASGI message is
    http.disconnect. Starlette (ASGI spec >= 2.4) only notices a disconnect when a write fails and
    then skips the response's background task, so before this a non-stream request (nothing is
    written until the engine is done) or a dropped stream left the upstream open and the engine
    generating for nobody."""
    while (await request.receive())["type"] != "http.disconnect":
        pass


async def _send(method, path, params, headers, body):
    r = rules()
    up = upstream_client(r["upstream"])
    req = up.build_request(method, path, params=params, headers=headers, content=body or None)
    try:
        return await up.send(req, stream=True)
    except httpx.ConnectError as e:
        if not r["fallback"]:
            raise
        log.error("upstream %s unreachable (%s: %s), using %s", r["upstream"], type(e).__name__, e, r["fallback"])
        fb = upstream_client(r["fallback"])
        req = fb.build_request(method, path, params=params, headers=headers, content=body or None)
        return await fb.send(req, stream=True)


async def relay(up: httpx.Response, disc: asyncio.Task, path: str):
    """Relay the upstream body in this one task: every read races the disconnect watcher, so a
    dropped client is seen even while the upstream is silent. After a drop the upstream is kept
    for disconnect_grace_s, then closed, which aborts the generation."""
    it = up.aiter_raw()
    nxt = None
    finished = False
    try:
        while True:
            nxt = asyncio.ensure_future(it.__anext__())
            await asyncio.wait({nxt, disc}, return_when=asyncio.FIRST_COMPLETED)
            if not nxt.done():
                break                                  # client gone, upstream still producing
            try:
                chunk = nxt.result()
            except StopAsyncIteration:
                finished = True
                break
            except (httpx.StreamClosed, httpx.ReadError, httpx.RemoteProtocolError):
                break
            nxt = None
            yield chunk
    finally:
        disc.cancel()
        if finished:
            await up.aclose()
        else:
            # Starlette may be cancelling this task (client gone), so the hold + close runs detached.
            t = asyncio.create_task(_close_later(up, nxt, path))
            _closers.add(t)
            t.add_done_callback(_closers.discard)


_closers: set = set()


async def _close_later(up: httpx.Response, nxt, path: str) -> None:
    grace = rules()["disconnect_grace_s"]
    log.info("client gone on %s, keeping upstream %ss", path, grace)
    await asyncio.sleep(grace)
    if nxt is not None and not nxt.done():
        nxt.cancel()
        await asyncio.gather(nxt, return_exceptions=True)
    await up.aclose()
    log.info("upstream on %s closed", path)

app = Starlette(routes=[Route("/{p:path}", handle, methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])])

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    import socket
    sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)   # one socket for IPv4 + IPv6
    sock.bind(("::", LISTEN_PORT))
    cfg = uvicorn.Config(app, log_level="warning", access_log=False, proxy_headers=False)
    uvicorn.Server(cfg).run(sockets=[sock])
