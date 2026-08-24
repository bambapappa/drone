#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"

echo "Stoppar Drone-servrar på port 8001 och 8000 ..."

# Native review/realtime processes started by this project.
for pattern in \
  'uvicorn review.main:app' \
  'uvicorn app.main:app' \
  'python.*-m uvicorn review.main' \
  'python.*-m uvicorn app.main'; do
  pkill -TERM -f "$pattern" 2>/dev/null || true
done

# A Docker review container may own the host port instead of uvicorn directly.
if command -v docker >/dev/null 2>&1; then
  ids="$(docker ps --filter publish=8001 --format '{{.ID}}' 2>/dev/null || true)"
  if [[ -n "$ids" ]]; then
    docker stop $ids >/dev/null || true
  fi
fi

echo "Klart. Du kan nu dubbelklicka på START_DRONE.command."
read -r -p "Tryck Enter för att stänga detta fönster ... " _
