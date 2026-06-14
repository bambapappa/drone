#!/usr/bin/env bash
# Robust local server (re)start. Kills any stale instance, starts uvicorn
# unbuffered in the background, waits for /health, prints the URL.
# Usage: scripts/serve.sh [SOURCE_FILE]
#   SOURCE defaults to $SOURCE env or the first file in videos/.
# Env (optional): MODEL, IMGSZ, CONF, TILES, PORT.
set -u
cd "$(dirname "$0")/.."

PORT="${PORT:-8000}"
[ -n "${1:-}" ] && export SOURCE="$1"

# 1. Stop any previous instance on this port (does not touch files/context).
pkill -9 -f "uvicorn app.main" 2>/dev/null || true
for _ in 1 2 3 4 5; do ss -ltn 2>/dev/null | grep -q ":${PORT}\b" || break; sleep 1; done

# 2. Start fresh, unbuffered so logs are truthful.
export PYTHONUNBUFFERED=1
nohup python3 -m uvicorn app.main:app --host 0.0.0.0 --port "${PORT}" --no-access-log \
  > /tmp/drone-serve.log 2>&1 &
echo "startar (pid $!), logg: /tmp/drone-serve.log"

# 3. Wait for health (first run may fetch the model).
for i in $(seq 1 60); do
  if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
    echo "✅ uppe på http://localhost:${PORT}  (öppna /)"
    exit 0
  fi
  sleep 2
done
echo "❌ kom inte upp inom 120 s — se /tmp/drone-serve.log:"; tail -15 /tmp/drone-serve.log
exit 1
