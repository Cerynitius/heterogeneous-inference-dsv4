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
| `FREETOKEN_DSV4_EP_N1` | `124` | routed experts per layer on GPU 1 |
| `FREETOKEN_DSV4_IDX_SPLIT` | `0.4` | share of deep prefill indexer queries scored on GPU 1, `0` = off |
| `FREETOKEN_DSV4_IDX_SPLIT_MIN_WORK` | `1.1e8` | split only when queries × key blocks of a call reach this |
| `FREETOKEN_DSV4_IDX_SPLIT_MARGIN_MB` | `768` | free MiB GPU 1 must keep after the split workspace, else the split stays off |
| `FREETOKEN_DSV4_L2_GB` | `64` | host L2 prefix cache size, GiB |
| `FREETOKEN_RUNTIME_CONF` | `deploy/runtime.json` | hot-reloadable defaults file |
| `FT` | `ft` | engine CLI |

Limits: 1,048,576 tokens per request (input + output), 2 concurrent requests.

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

| Variable | Default | Meaning |
|---|---|---|
| `REQFILTER_UPSTREAM` | `http://127.0.0.1:1920` | engine address |
| `REQFILTER_PORT` | `1922` | listen port |
| `REQFILTER_RULES` | `deploy/rules.json` | rules file |
| `REQFILTER_FALLBACK` | empty | second upstream used when the first refuses connections |
| `WEBSEARCH_URL` | `http://127.0.0.1:1930` | search backend |

## Web search (server-side tool)

Start the search backend (SearXNG in Docker on 127.0.0.1:8888, backend on 127.0.0.1:1930, both loopback only):

```bash
pip install httpx starlette uvicorn trafilatura
deploy/websearch/run.sh
```

| Variable | Default | Meaning |
|---|---|---|
| `WEBSEARCH_PORT` | `1930` | listen port (127.0.0.1) |
| `WEBSEARCH_SEARXNG` | `http://127.0.0.1:8888` | SearXNG address |
| `WEBSEARCH_PROXY` | empty | outbound proxy for page fetches |
| `WEBSEARCH_TEXT_CHARS` | `3000` | page text per result |

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

Edit `deploy/runtime.json`; the next request uses the new values. Keys apply only when the request does not set the field itself.

| Key | Type | Meaning |
|---|---|---|
| `default_max_output_tokens` | int ≥ 1 | output budget when `max_tokens` is omitted |
| `default_thinking` | bool | thinking mode when the request sends no thinking switch |
| `default_reasoning_effort` | `"low"` / `"high"` / `"max"` / null | effort in thinking mode |
| `temperature` | number | default temperature |
| `top_p` | number | default top_p |
| `top_k` | int | default top_k |
| `force_temperature` | number / null | every request samples at this temperature |

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
