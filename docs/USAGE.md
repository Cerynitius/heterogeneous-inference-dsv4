# Usage

## Requirements

- Linux, two NVIDIA GPUs: a 96 GB card as GPU 0 and a 72 GB card as GPU 1 (RTX PRO 6000 + RTX PRO 5000 Blackwell)
- CUDA toolkit (`CUDA_HOME`) for the C++ extensions and the JIT-compiled CUDA kernels
- Python 3.12, about 90 GiB of free host RAM (64 GiB of it for the L2 prefix cache)
- Checkpoint: `deepseek-ai/DeepSeek-V4-Flash-0731`, downloaded to a local directory

## Install

```bash
git clone https://github.com/Cerynitius/heterogeneous-inference-dsv4
cd heterogeneous-inference-dsv4
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e .
```

`pip install -e .` installs the dependencies from `pyproject.toml` and builds `freetoken.kernel._pinned_tensor` and `freetoken.kernel._cpu_moe`.

## Run

```bash
DSV4_MODEL=/path/to/DeepSeek-V4-Flash-0731 deploy/serve.sh
```

The server is ready when `POST /v1/chat/completions` answers.

| Variable | Default | Meaning |
|---|---|---|
| `DSV4_MODEL` | (required) | checkpoint directory |
| `DSV4_PORT` | `1920` | engine port |
| `DSV4_HOST` | `127.0.0.1` | engine bind address |
| `DSV4_NAME` | `dsv4-flash` | served model name |
| `CUDA_VISIBLE_DEVICES` | `0,1` | large card first |
| `DSV4_MAX_RUNNING` | `4` | request slots allocated at startup; upper bound of `max_running_requests` |
| `FREETOKEN_DSV4_EP_N1` | `124` | routed experts per layer on GPU 1 |
| `FREETOKEN_DSV4_IDX_SPLIT` | `0.4` | share of deep prefill indexer queries scored on GPU 1, `0` = off |
| `FREETOKEN_DSV4_IDX_SPLIT_MIN_WORK` | `1.1e8` | split only when queries × key blocks of a call reach this |
| `FREETOKEN_DSV4_IDX_SPLIT_MARGIN_MB` | `768` | free MiB GPU 1 must keep after the split workspace, else the split stays off |
| `FREETOKEN_DSV4_L2_GB` | `64` | host L2 prefix cache size, GiB |
| `FREETOKEN_RUNTIME_CONF` | `deploy/runtime.json` | hot-reloadable defaults file |
| `FT` | `ft` | engine CLI |

Limits: 1,048,576 tokens per request (input + output); concurrent requests = `max_running_requests` in `deploy/runtime.json` (2), at most `DSV4_MAX_RUNNING`.

## Request filter

```bash
pip install httpx starlette uvicorn
REQFILTER_UPSTREAM=http://127.0.0.1:1920 REQFILTER_PORT=1922 python deploy/reqfilter.py
```

Listens on `REQFILTER_PORT` (IPv4 and IPv6) and forwards only:
`GET /health`, `GET /v1/models`, `POST /v1/chat/completions`, `POST /v1/completions`,
`POST /v1/messages`, `POST /v1/messages/count_tokens`, `POST /v1/responses`,
`GET /v1/responses/{id}`, `POST /v1/responses/{id}/cancel`.

Other routes get 403. A generation request with empty input gets 400 `input tokens must be > 0`.

Rules in `deploy/rules.json` (`REQFILTER_RULES`), re-read on change, no restart:

| Key | Default | Meaning |
|---|---|---|
| `nonstream_min_chars` | `1000` | non-streaming generation requests with a shorter prompt get 400 `backend doesn't support non-stream` |
| `disconnect_grace_s` | `15` | seconds the upstream is kept after the client disconnects; then the generation is aborted |
| `web_search` | `1` | `0` = hosted web_search tools are removed from requests |
| `web_search_max_uses` | `8` | searches per request when the tool sets no `max_uses` |
| `web_search_results` | `8` | results per search |
| `web_search_fetch` | `5` | results per search whose page text is fetched |
| `upstream` | `REQFILTER_UPSTREAM` | engine address |
| `fallback` | `REQFILTER_FALLBACK` | second upstream used when the first refuses connections, `""` = none |
| `websearch_url` | `WEBSEARCH_URL` | search backend |

A file with an unknown key or a bad value is rejected; the last valid values stay.

| Variable | Default | Meaning |
|---|---|---|
| `REQFILTER_PORT` | `1922` | listen port |
| `REQFILTER_RULES` | `deploy/rules.json` | rules file |
| `REQFILTER_UPSTREAM` | `http://127.0.0.1:1920` | `upstream` when rules.json omits it |
| `REQFILTER_FALLBACK` | empty | `fallback` when rules.json omits it |
| `WEBSEARCH_URL` | `http://127.0.0.1:1930` | `websearch_url` when rules.json omits it |

## Web search (server-side tool)

Start the search backend (SearXNG in Docker on 127.0.0.1:8888, backend on 127.0.0.1:1930, both loopback only):

```bash
pip install httpx starlette uvicorn trafilatura
deploy/websearch/run.sh
```

Settings in `deploy/websearch/conf.json` (`WEBSEARCH_CONF`), re-read on change, no restart. A missing key uses the variable; a bad file is rejected and the last valid values stay.

| Key | Variable | Default | Meaning |
|---|---|---|---|
| `searxng` | `WEBSEARCH_SEARXNG` | `http://127.0.0.1:8888` | SearXNG address |
| `proxy` | `WEBSEARCH_PROXY` | empty | outbound proxy for page fetches and Tavily |
| `fetch_timeout` | `WEBSEARCH_FETCH_TIMEOUT` | `6` | seconds per page fetch |
| `text_chars` | `WEBSEARCH_TEXT_CHARS` | `3000` | page text per result |
| `tavily_key_file` | `WEBSEARCH_TAVILY_KEY_FILE` | `deploy/websearch/tavily.key` | Tavily API key file; missing = SearXNG only |
| `tavily_depth` | `WEBSEARCH_TAVILY_DEPTH` | `basic` | Tavily `search_depth` (`basic` 1 credit, `advanced` 2) |
| `cache_ttl` | `WEBSEARCH_CACHE_TTL` | `21600` | seconds a query's results are reused |
| `hard_host_ttl` | | `86400` | seconds a site that blocked a direct fetch goes straight to Tavily extract |

Startup only:

| Variable | Default | Meaning |
|---|---|---|
| `WEBSEARCH_PORT` | `1930` | listen port (127.0.0.1) |
| `WEBSEARCH_CONF` | `deploy/websearch/conf.json` | settings file |
| `WEBSEARCH_CACHE_DB` | `deploy/websearch/cache.db` | cache file |

Each search queries Tavily and SearXNG in parallel and merges the results (SearXNG first for CJK queries). Pages without text are fetched directly; pages that fail are read with Tavily extract.

Clients declare the hosted tool; the request filter runs the searches and continues the generation.

Anthropic Messages:

```json
"tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 5,
           "allowed_domains": ["example.com"], "blocked_domains": []}]
```

Returns `server_tool_use` and `web_search_tool_result` blocks; `usage.server_tool_use.web_search_requests` counts searches. Send the blocks back unchanged in later turns.

OpenAI Responses:

```json
"tools": [{"type": "web_search"}]
```

Returns `web_search_call` output items (`action.type` = `search`).

Chat Completions:

```json
"web_search_options": {}
```

Searches stay internal; the reply contains the final answer.

## Runtime defaults (no restart)

Edit `deploy/runtime.json`; the next request uses the new values. Sampling and thinking keys apply only when the request does not set the field itself; a missing key uses the startup value.

| Key | Type | Meaning |
|---|---|---|
| `default_max_output_tokens` | int ≥ 1 | output budget when `max_tokens` is omitted |
| `default_thinking` | bool | thinking mode when the request sends no thinking switch |
| `default_reasoning_effort` | `"low"` / `"high"` / `"max"` / null | effort in thinking mode |
| `temperature` | number | default temperature |
| `top_p` | number | default top_p |
| `top_k` | int | default top_k |
| `force_temperature` | number / null | every request samples at this temperature |
| `max_running_requests` | int ≥ 1 | requests run at once, at most `DSV4_MAX_RUNNING`; lowering it does not stop running requests |
| `prefill_chunk_tokens` | int ≥ 1 | prefill chunk size, at most the startup chunk (8192) |
| `idx_split` | 0–1 | share of deep prefill indexer queries scored on GPU 1, `0` = off |
| `idx_split_min_work` | number ≥ 0 | split only when queries × key blocks of a call reach this |
| `sched_decode_share` | 0–0.95 | GPU time running decodes get while prompts prefill |
| `sched_reserve_tokens` | int ≥ 0 | future tokens reserved per running request, `0` = all of `max_tokens` |
| `sched_short_first` | bool | short prompts overtake a long chunked prefill |
| `sched_aligned_chunks` | bool | a new prompt waits for a whole chunk instead of a leftover slice |
| `swa_eviction_interval` | int ≥ 1 | decode steps between frees of out-of-window sliding pages |
| `ep_prefill_split` | int ≥ 1 | MoE sub-chunks per prefill chunk across the two cards |
| `ep_prefill_split_ratio` | 0–1 | size ratio of consecutive sub-chunks |
| `forward_unknown_tools` | bool | pass through tool calls whose name the request did not declare |

An invalid file is rejected; the engine logs the error and keeps the last valid values. Deleting the file returns to the startup defaults.

## API

OpenAI Chat Completions, OpenAI Responses and Anthropic Messages:

```bash
curl http://127.0.0.1:1920/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "dsv4-flash",
  "messages": [{"role": "user", "content": "hello"}],
  "chat_template_kwargs": {"enable_thinking": false}
}'
```

| Field | Effect |
|---|---|
| `chat_template_kwargs.enable_thinking` | `true` / `false`: thinking on or off |
| `reasoning_effort` | `low`, `high`, `max` |
| `thinking: {"type": "enabled" \| "disabled"}` | Anthropic / DeepSeek thinking switch |
| `tools` | tool calling (always runs in thinking mode) |
