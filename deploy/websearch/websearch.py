"""Web search backend for server-side model tools (loopback only).

GET /search?q=...&n=8&fetch=5[&lang=..][&site=a.com,b.org][&exclude=c.com]
    SearXNG results; the first `fetch` pages are downloaded in parallel and their main text extracted.
    -> {"query", "results": [{"title", "url", "published", "snippet", "text"}], "elapsed_s"}
GET /fetch?url=...  -> {"url", "title", "text"}
GET /health

Pages are fetched through the outbound proxy (WEBSEARCH_PROXY, optional).
Hosts that resolve to loopback / private / link-local addresses are refused.
"""
import asyncio, ipaddress, logging, os, socket, time
from urllib.parse import urlsplit

import httpx, trafilatura, uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

SEARXNG = os.environ.get("WEBSEARCH_SEARXNG", "http://127.0.0.1:8888")
PROXY = os.environ.get("WEBSEARCH_PROXY", "") or None
PORT = int(os.environ.get("WEBSEARCH_PORT", "1930"))
FETCH_TIMEOUT = float(os.environ.get("WEBSEARCH_FETCH_TIMEOUT", "6"))
MAX_PAGE_BYTES = 3 * 1024 * 1024
TEXT_CHARS = int(os.environ.get("WEBSEARCH_TEXT_CHARS", "3000"))
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/131.0.0.0 Safari/537.36")

log = logging.getLogger("websearch")
sx = httpx.AsyncClient(base_url=SEARXNG, timeout=httpx.Timeout(15.0), trust_env=False)
web = httpx.AsyncClient(proxy=PROXY, timeout=httpx.Timeout(FETCH_TIMEOUT), trust_env=False,
                        follow_redirects=False, headers={"User-Agent": UA, "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"},
                        limits=httpx.Limits(max_connections=64, max_keepalive_connections=16))


def _public_host(host: str) -> bool:
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return True            # unresolved locally (e.g. only the proxy resolves it): not a local address
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
            return False
    return True


async def _check_url(url: str) -> str | None:
    p = urlsplit(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        return f"unsupported url {url!r}"
    if not await asyncio.to_thread(_public_host, p.hostname):
        return f"refused non-public host {p.hostname}"
    return None


async def fetch_page(url: str) -> dict:
    """Download (following up to 5 checked redirects) and extract the main text."""
    t0 = time.monotonic()
    for _ in range(6):
        err = await _check_url(url)
        if err:
            return {"url": url, "error": err}
        try:
            async with web.stream("GET", url) as r:
                if r.is_redirect:
                    url = str(r.url.join(r.headers.get("location", "")))
                    continue
                if r.status_code >= 400:
                    return {"url": url, "error": f"HTTP {r.status_code}"}
                ctype = r.headers.get("content-type", "")
                if "html" not in ctype and "text" not in ctype and "xml" not in ctype:
                    return {"url": url, "error": f"unsupported content-type {ctype!r}"}
                buf = bytearray()
                async for chunk in r.aiter_bytes():
                    buf += chunk
                    if len(buf) > MAX_PAGE_BYTES:
                        break
                html = bytes(buf).decode(r.encoding or "utf-8", errors="replace")
        except Exception as e:  # noqa: BLE001
            return {"url": url, "error": f"{type(e).__name__}: {e}"}
        break
    else:
        return {"url": url, "error": "too many redirects"}
    def _extract():
        text = trafilatura.extract(html, include_comments=False, include_tables=True, favor_recall=True) or ""
        meta = trafilatura.extract_metadata(html)
        return text, (meta.title if meta else None)
    text, title = await asyncio.to_thread(_extract)
    log.info("fetch %s %d chars %.1fs", url, len(text), time.monotonic() - t0)
    return {"url": url, "title": title, "text": text}


def _domain_ok(url: str, site: list[str], exclude: list[str]) -> bool:
    host = (urlsplit(url).hostname or "").lower()
    match = lambda d: host == d or host.endswith("." + d)  # noqa: E731
    if site and not any(match(d) for d in site):
        return False
    return not any(match(d) for d in exclude)


async def search(request: Request):
    q = (request.query_params.get("q") or "").strip()
    if not q:
        return JSONResponse({"error": "missing q"}, status_code=400)
    n = min(int(request.query_params.get("n", "8")), 20)
    nfetch = min(int(request.query_params.get("fetch", "5")), n)
    lang = request.query_params.get("lang", "auto")
    site = [d.strip().lower() for d in request.query_params.get("site", "").split(",") if d.strip()]
    exclude = [d.strip().lower() for d in request.query_params.get("exclude", "").split(",") if d.strip()]
    t0 = time.monotonic()
    sq = q + (" " + " OR ".join(f"site:{d}" for d in site) if site else "")
    try:
        r = await sx.get("/search", params={"q": sq, "format": "json", "language": lang, "safesearch": 0})
        r.raise_for_status()
        data = r.json()
    except Exception as e:  # noqa: BLE001
        log.error("searxng %r: %s: %s", q, type(e).__name__, e)
        return JSONResponse({"error": f"search backend: {type(e).__name__}: {e}"}, status_code=502)
    seen, results = set(), []
    for it in data.get("results", []):
        url = it.get("url")
        if not url or url in seen or not _domain_ok(url, site, exclude):
            continue
        seen.add(url)
        results.append({"title": (it.get("title") or "").strip(), "url": url,
                        "published": it.get("publishedDate"), "snippet": (it.get("content") or "").strip(), "text": ""})
        if len(results) >= n:
            break
    pages = await asyncio.gather(*(fetch_page(x["url"]) for x in results[:nfetch]))
    for x, p in zip(results, pages):
        x["text"] = (p.get("text") or "")[:TEXT_CHARS]
    log.info("search %r -> %d results, %d fetched, %.1fs (unresponsive %s)", q, len(results),
             sum(1 for p in pages if p.get("text")), time.monotonic() - t0, data.get("unresponsive_engines"))
    return JSONResponse({"query": q, "results": results, "elapsed_s": round(time.monotonic() - t0, 2)})


async def fetch(request: Request):
    url = request.query_params.get("url") or ""
    p = await fetch_page(url)
    if p.get("error"):
        return JSONResponse(p, status_code=502)
    return JSONResponse(p)


async def health(_):
    return JSONResponse({"ok": True})


app = Starlette(routes=[Route("/search", search), Route("/fetch", fetch), Route("/health", health)])

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("trafilatura").setLevel(logging.ERROR)
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning", access_log=False)
