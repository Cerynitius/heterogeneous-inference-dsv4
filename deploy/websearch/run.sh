#!/usr/bin/env bash
# SearXNG on 127.0.0.1:8888 + search backend on 127.0.0.1:1930 (both loopback only).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
CONF=${SEARXNG_CONF:-$HERE/searxng}
mkdir -p "$CONF"
[ -f "$CONF/settings.yml" ] || sed "s/CHANGE_ME/$(openssl rand -hex 32)/" "$HERE/settings.yml" > "$CONF/settings.yml"
docker run -d --name websearch-searxng --restart unless-stopped --network host \
  -e GRANIAN_HOST=127.0.0.1 -e GRANIAN_PORT=8888 -v "$CONF:/etc/searxng" searxng/searxng:latest
exec python "$HERE/websearch.py"
