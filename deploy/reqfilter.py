"""Request filter in front of `ft serve` (deploy/serve.sh).

External traffic reaches this proxy first and is forwarded to the engine on 127.0.0.1.
- Only inference endpoints pass (allow-list); cache rebuild, /generate, admin and stats routes get 403.
- Request bodies must be JSON objects; debug fields (ignore_eos) are removed.
- Streaming responses are passed through; when the client disconnects the upstream connection is
  closed and the engine aborts the generation.
"""
import json, logging, os, re
import httpx, uvicorn
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

UPSTREAM = os.environ.get("REQFILTER_UPSTREAM", "http://127.0.0.1:1920")
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
client = httpx.AsyncClient(base_url=UPSTREAM, timeout=httpx.Timeout(None, connect=10.0), trust_env=False,
                           limits=httpx.Limits(max_connections=64, max_keepalive_connections=16))


def deny(request: Request, status: int, why: str):
    log.warning("BLOCK %s %s %s from %s (xff=%s)", status, request.method, request.url.path,
                request.client.host if request.client else "?", request.headers.get("x-forwarded-for", "-"))
    return JSONResponse({"error": {"type": "forbidden", "message": why}}, status_code=status)


async def handle(request: Request):
    path, method = request.url.path, request.method
    if not any(m == method and rx.fullmatch(path) for m, rx in ALLOW):
        return deny(request, 403, f"{method} {path} is not allowed on this endpoint")
    body = b""
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
            body = json.dumps(doc, ensure_ascii=False).encode()
    headers = [(k, v) for k, v in request.headers.items() if k.lower() not in HOP]
    if body:
        headers = [(k, v) for k, v in headers if k.lower() != "content-type"] + [("content-type", "application/json")]
    req = client.build_request(method, path, params=request.query_params, headers=headers, content=body or None)
    try:
        up = await client.send(req, stream=True)
    except httpx.HTTPError as e:
        log.error("upstream error %s %s: %s: %s", method, path, type(e).__name__, e)
        return JSONResponse({"error": {"type": "upstream_unavailable", "message": f"{type(e).__name__}: {e}"}},
                            status_code=502)
    out_headers = {k: v for k, v in up.headers.items() if k.lower() not in HOP and k.lower() != "content-encoding"}
    return StreamingResponse(up.aiter_raw(), status_code=up.status_code, headers=out_headers,
                             background=BackgroundTask(up.aclose))


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
