"""Web search backend for server-side model tools (loopback only).

GET /search?q=...&n=8&fetch=5[&lang=..][&site=a.com,b.org][&exclude=c.com]
    SearXNG results; the first `fetch` pages are downloaded in parallel and their main text extracted.
    -> {"query", "results": [{"title", "url", "published", "snippet", "text"}], "elapsed_s"}
GET /fetch?url=...  -> {"url", "title", "text"}
GET /health

Primary backend: Tavily (key in WEBSEARCH_TAVILY_KEY_FILE; basic search + raw page text, 1 credit/search).
Fallback when Tavily fails or its quota is used up: SearXNG + own page fetch.
Each result's text = the paragraphs of the page that best match the query (up to WEBSEARCH_TEXT_CHARS).
Results are cached per query (WEBSEARCH_CACHE_TTL seconds, sqlite) so repeated searches cost no credits.

Pages are fetched through the outbound proxy (WEBSEARCH_PROXY, optional).
Hosts that resolve to loopback / private / link-local addresses are refused.
"""
import asyncio, ipaddress, json, logging, math, os, re, socket, sqlite3, time
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
TAVILY_KEY_FILE = os.environ.get("WEBSEARCH_TAVILY_KEY_FILE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "tavily.key"))
TAVILY_DEPTH = os.environ.get("WEBSEARCH_TAVILY_DEPTH", "basic")
CACHE_TTL = float(os.environ.get("WEBSEARCH_CACHE_TTL", str(6 * 3600)))
CACHE_DB = os.environ.get("WEBSEARCH_CACHE_DB", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache.db"))
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/131.0.0.0 Safari/537.36")

log = logging.getLogger("websearch")
sx = httpx.AsyncClient(base_url=SEARXNG, timeout=httpx.Timeout(15.0), trust_env=False)
web = httpx.AsyncClient(proxy=PROXY, timeout=httpx.Timeout(FETCH_TIMEOUT), trust_env=False,
                        follow_redirects=False, headers={"User-Agent": UA, "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"},
                        limits=httpx.Limits(max_connections=64, max_keepalive_connections=16))


tavily = httpx.AsyncClient(base_url="https://api.tavily.com", proxy=PROXY, timeout=httpx.Timeout(30.0, connect=10.0),
                           trust_env=False)


def _tavily_key() -> str | None:
    try:
        return open(TAVILY_KEY_FILE).read().strip() or None
    except OSError:
        return None


# ---------------------------------------------------------------- query-relevant passages

_CJK = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")


def _terms(text: str) -> list[str]:
    """Latin words + CJK character bigrams, lower-cased."""
    text = text.lower()
    out = re.findall(r"[a-z0-9][a-z0-9.+_-]*", text)
    for run in re.findall(r"[\u3400-\u9fff\uf900-\ufaff]+", text):
        out += [run[i:i + 2] for i in range(len(run) - 1)] or [run]
    return out


def best_passages(query: str, text: str, limit: int) -> str:
    """Paragraphs of `text` ranked by BM25 against `query`, the best ones kept (in page order) up to `limit` chars."""
    if len(text) <= limit:
        return text
    paras = [p.strip() for p in re.split(r"\n\s*\n|\n(?=[#*\-\d])", text) if len(p.strip()) >= 20]
    chunks = []
    for p in paras:                               # split very long paragraphs
        while len(p) > 1200:
            cut = max(p.rfind("。", 0, 1000), p.rfind(". ", 0, 1000))
            cut = cut + 1 if cut > 200 else 1000
            chunks.append(p[:cut]); p = p[cut:].strip()
        if p:
            chunks.append(p)
    if not chunks:
        return text[:limit]
    q = set(_terms(query))
    docs = [_terms(c) for c in chunks]
    N, avg = len(docs), sum(len(d) for d in docs) / len(docs) or 1
    df = {t: sum(1 for d in docs if t in d) for t in q}
    scores = []
    for i, d in enumerate(docs):
        tf = {}
        for t in d:
            if t in q:
                tf[t] = tf.get(t, 0) + 1
        sc = sum(math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5)) * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * len(d) / avg))
                 for t, f in tf.items())
        scores.append((sc - 0.01 * i, i))         # ties -> earlier paragraph
    keep, used = set(), 0
    for sc, i in sorted(scores, reverse=True):
        if used + len(chunks[i]) > limit and keep:
            continue
        keep.add(i); used += len(chunks[i]) + 2
        if used >= limit:
            break
    return "\n\n".join(chunks[i] for i in sorted(keep))[:limit]


_URL = r"\((?:[^()\s]|\([^()]*\))*(?:\s+\"[^\"]*\")?\)"      # (url) with one level of parens, optional title
_IMG = re.compile(r"!\[[^\]]*\]" + _URL)
_LINK = re.compile(r"\[([^\]]*)\]" + _URL)


def clean_markdown(md: str) -> str:
    """Tavily raw markdown -> text: images dropped, links -> their text, navigation lines (mostly links) dropped."""
    out = []
    for line in md.splitlines():
        line = _IMG.sub("", line)
        link_chars = sum(len(m.group(1).strip()) for m in _LINK.finditer(line))
        text = _LINK.sub(lambda m: m.group(1), line).strip()
        body = re.sub(r"[\s#*>|`\-\d.]", "", text)
        if not body:
            out.append("") if out and out[-1] else None
            continue
        if link_chars >= 0.6 * len(body) or text.startswith("[Skip"):
            continue
        out.append(text)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


# ---------------------------------------------------------------- cache

_db = sqlite3.connect(CACHE_DB, check_same_thread=False)
_db.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, t REAL, v TEXT)")


def cache_get(k: str):
    row = _db.execute("SELECT t, v FROM cache WHERE k = ?", (k,)).fetchone()
    return json.loads(row[1]) if row and time.time() - row[0] < CACHE_TTL else None


def cache_put(k: str, v) -> None:
    _db.execute("INSERT OR REPLACE INTO cache VALUES (?, ?, ?)", (k, time.time(), json.dumps(v, ensure_ascii=False)))
    _db.execute("DELETE FROM cache WHERE t < ?", (time.time() - CACHE_TTL,))
    _db.commit()


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


async def tavily_search(q: str, n: int, site: list[str], exclude: list[str]) -> list[dict]:
    key = _tavily_key()
    if not key:
        raise RuntimeError("no Tavily key")
    body = {"query": q, "max_results": n, "search_depth": TAVILY_DEPTH, "include_raw_content": "markdown",
            "include_answer": False}
    if site:
        body["include_domains"] = site
    if exclude:
        body["exclude_domains"] = exclude
    r = await tavily.post("/search", json=body, headers={"Authorization": f"Bearer {key}"})
    if r.status_code != 200:
        raise RuntimeError(f"Tavily HTTP {r.status_code}: {r.text[:200]}")
    return [{"title": (x.get("title") or "").strip(), "url": x["url"], "published": x.get("published_date"),
             "snippet": (x.get("content") or "").strip(), "raw": clean_markdown(x.get("raw_content") or ""),
             "src": "tavily"} for x in r.json().get("results", []) if x.get("url")]


HARD_HOST_TTL = 24 * 3600
_hard_hosts: dict[str, float] = {}     # host -> time our own fetch last failed there


def _host(url: str) -> str:
    return (urlsplit(url).hostname or "").lower()


async def tavily_extract(urls: list[str]) -> dict[str, str]:
    """url -> cleaned page text, for pages our own fetch could not read (1 credit per 5 extracted pages)."""
    key = _tavily_key()
    if not key or not urls:
        return {}
    try:
        r = await tavily.post("/extract", json={"urls": urls, "format": "markdown"},
                              headers={"Authorization": f"Bearer {key}"})
        if r.status_code != 200:
            log.warning("tavily extract HTTP %d: %s", r.status_code, r.text[:200])
            return {}
        d = r.json()
    except Exception as e:  # noqa: BLE001
        log.warning("tavily extract: %s: %s", type(e).__name__, e)
        return {}
    for f in d.get("failed_results", []):
        log.info("tavily extract failed %s: %s", f.get("url"), f.get("error"))
    return {x["url"]: clean_markdown(x.get("raw_content") or "") for x in d.get("results", []) if x.get("url")}


async def searxng_search(q: str, n: int, lang: str, site: list[str], exclude: list[str]) -> list[dict]:
    sq = q + (" " + " OR ".join(f"site:{d}" for d in site) if site else "")
    r = await sx.get("/search", params={"q": sq, "format": "json", "language": lang, "safesearch": 0})
    r.raise_for_status()
    data = r.json()
    seen, out = set(), []
    for it in data.get("results", []):
        url = it.get("url")
        if not url or url in seen or not _domain_ok(url, site, exclude):
            continue
        seen.add(url)
        out.append({"title": (it.get("title") or "").strip(), "url": url, "published": it.get("publishedDate"),
                    "snippet": (it.get("content") or "").strip(), "raw": "", "src": "searxng"})
        if len(out) >= n:
            break
    return out


async def search(request: Request):
    q = (request.query_params.get("q") or "").strip()
    if not q:
        return JSONResponse({"error": "missing q"}, status_code=400)
    n = min(int(request.query_params.get("n", "8")), 20)
    nfetch = min(int(request.query_params.get("fetch", "5")), n)
    lang = request.query_params.get("lang", "auto")
    site = [d.strip().lower() for d in request.query_params.get("site", "").split(",") if d.strip()]
    exclude = [d.strip().lower() for d in request.query_params.get("exclude", "").split(",") if d.strip()]
    ck = json.dumps([q, n, nfetch, lang, site, exclude], ensure_ascii=False)
    hit = cache_get(ck)
    if hit is not None:
        log.info("search %r -> cache hit (%s)", q, hit.get("backend"))
        return JSONResponse(dict(hit, cached=True))
    t0 = time.monotonic()
    tv, sx_ = await asyncio.gather(tavily_search(q, n, site, exclude), searxng_search(q, n, lang, site, exclude),
                                   return_exceptions=True)
    for name, r in (("tavily", tv), ("searxng", sx_)):
        if isinstance(r, BaseException):
            log.warning("%s %r failed: %s: %s", name, q, type(r).__name__, r)
    tv = [] if isinstance(tv, BaseException) else tv
    sx_ = [] if isinstance(sx_, BaseException) else sx_
    if not tv and not sx_:
        return JSONResponse({"error": "search backends failed (see log)"}, status_code=502)
    # merge: interleave, Google (SearXNG) first for CJK queries (Tavily ranks Chinese pages poorly), dedupe by URL
    a, b = (sx_, tv) if _CJK.search(q) else (tv, sx_)
    results, seen = [], set()
    for i in range(max(len(a), len(b))):
        for lst in (a, b):
            if i < len(lst):
                u = lst[i]["url"].rstrip("/")
                if u not in seen:
                    seen.add(u)
                    results.append(lst[i])
    results = results[:n]
    backend = "+".join(sorted({x["src"] for x in results}))
    # page text: Tavily's raw content, else our own fetch, else Tavily extract (first `nfetch` results)
    need = [x for x in results[:nfetch] if len(x["raw"]) < 200]
    now = time.time()
    failed = [x for x in need if now - _hard_hosts.get(_host(x["url"]), 0) < HARD_HOST_TTL]   # known to block us
    own = [x for x in need if x not in failed]
    pages = await asyncio.gather(*(fetch_page(x["url"]) for x in own))
    for x, p in zip(own, pages):
        if len(p.get("text") or "") >= 300:
            x["raw"] = p["text"]
        else:
            log.info("fetch failed %s: %s", x["url"], p.get("error") or f"only {len(p.get('text') or '')} chars")
            _hard_hosts[_host(x["url"])] = now
            failed.append(x)
    if failed:
        ext = await tavily_extract([x["url"] for x in failed])
        for x in failed:
            if len(ext.get(x["url"], "")) > len(x["raw"]):
                x["raw"] = ext[x["url"]]
        log.info("tavily extract %d/%d recovered", sum(1 for x in failed if len(x["raw"]) >= 300), len(failed))
    for x in results:
        raw = x.pop("raw")
        lead = x["snippet"] if x["src"] == "tavily" else ""     # Tavily's snippet = its query-relevant chunks
        rest = best_passages(q, raw, max(0, TEXT_CHARS - len(lead) - 2)) if raw else ""
        x["text"] = (lead + "\n\n" + rest).strip() if rest and rest[:80] not in lead else (lead or rest or x["snippet"])
        x.pop("src")
    got = sum(1 for x in results if len(x["text"]) >= 500)
    log.info("search %r via %s -> %d results, %d with text >=500 chars (%d own fetches, %d extract), %.1fs",
             q, backend, len(results), got, len(own), len(failed), time.monotonic() - t0)
    out = {"query": q, "backend": backend, "results": results, "elapsed_s": round(time.monotonic() - t0, 2)}
    cache_put(ck, out)
    return JSONResponse(out)


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
