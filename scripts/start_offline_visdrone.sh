#!/usr/bin/env bash
# Starta den fristående analysen med VisDrone på en Mac med Colima/Docker.
# Kör från valfri katalog: bash scripts/start_offline_visdrone.sh videos/film.mp4
set -euo pipefail

usage() {
  cat <<'EOF'
Användning: bash scripts/start_offline_visdrone.sh <film-i-videos/>

Exempel:
  bash scripts/start_offline_visdrone.sh videos/test.mp4
  bash scripts/start_offline_visdrone.sh test.mp4

Filmen måste ligga i projektets videos/-mapp. Skriptet startar Colima vid
behov, hämtar VisDrone-s-vikter om de saknas, bygger offline-tjänsterna,
kör analysen och startar sedan granskningsvyn på http://localhost:8001.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if [[ $# -ne 1 ]]; then
  usage >&2
  exit 2
fi

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$root_dir"

input="$1"
if [[ "$input" == videos/* ]]; then
  video_rel="$input"
elif [[ "$input" == */* ]]; then
  echo "Fel: filmen måste ligga under $root_dir/videos/." >&2
  echo "Flytta eller kopiera den dit och kör exempelvis: $0 videos/$(basename "$input")" >&2
  exit 2
else
  video_rel="videos/$input"
fi

if [[ ! -f "$video_rel" ]]; then
  echo "Fel: hittar inte $root_dir/$video_rel" >&2
  exit 2
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Fel: Docker-klienten saknas. Installera Docker CLI och Colima först." >&2
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  if ! command -v colima >/dev/null 2>&1; then
    echo "Fel: Docker-daemonen kör inte och Colima är inte installerat." >&2
    exit 1
  fi
  echo "Docker kör inte; startar Colima ..."
  if ! colima start; then
    cat >&2 <<'EOF'

Colima kunde inte starta. Om feltexten nämner "host agent is not", starta om
Macen och kör samma kommando igen. Radera inte Colima-profilen utan att först
spara eventuella Docker-volymer som du behöver.
EOF
    exit 1
  fi
  docker context use colima >/dev/null
fi

if ! docker info >/dev/null 2>&1; then
  echo "Fel: Docker-daemonen svarar fortfarande inte efter Colima-start." >&2
  exit 1
fi

if ! docker buildx version >/dev/null 2>&1; then
  echo "Fel: Docker buildx-plugin saknas. Installera den med: brew install docker-buildx" >&2
  exit 1
fi

model_host="models/visdrone-yolov8s.pt"
if [[ ! -s "$model_host" ]]; then
  echo "Hämtar VisDrone-s-vikter ..."
  python3 scripts/fetch_visdrone.py --size s
fi

if [[ ! -s "$model_host" ]]; then
  echo "Fel: VisDrone-vikterna saknas efter nedladdning: $root_dir/$model_host" >&2
  exit 1
fi

compose=(docker compose -f docker-compose.yml -f docker-compose.offline.yml)
export MODEL="/models/visdrone-yolov8s.pt"

echo "Bygger offline-tjänster ..."
"${compose[@]}" build analyze review

echo "Analyserar $video_rel med VisDrone-s ..."
"${compose[@]}" run --rm analyze "/videos/${video_rel#videos/}"

echo "Startar granskningsvyn ..."
"${compose[@]}" up -d review

for _ in {1..20}; do
  if curl -fsS http://localhost:8001/health >/dev/null 2>&1; then
    echo "Klart. Öppna http://localhost:8001 och välj den nya körningen."
    exit 0
  fi
  sleep 1
done

echo "Analysen är klar men granskningsvyn svarade inte inom 20 sekunder." >&2
echo "Kontrollera den med: ${compose[*]} logs review" >&2
exit 1
