#!/usr/bin/env bash
# Starta den fristående analysen med VisDrone på en Mac med Colima/Docker.
# Kör från valfri katalog: bash scripts/start_offline_visdrone.sh videos/film.mp4
set -euo pipefail

usage() {
  cat <<'EOF'
Användning: bash scripts/start_offline_visdrone.sh [--fresh] <film-i-videos/>

Exempel:
  bash scripts/start_offline_visdrone.sh videos/test.mp4
  bash scripts/start_offline_visdrone.sh --fresh videos/test.mp4

Filmen måste ligga i projektets videos/-mapp. Skriptet startar Colima vid
behov, hämtar VisDrone-s-vikter om de saknas, bygger offline-tjänsterna,
kör analysen och startar sedan granskningsvyn på http://localhost:8001. En
tidigare komplett körning med samma film och konfiguration återanvänds. Använd
--fresh för att medvetet skapa en ny körning.
EOF
}

fresh=false
input=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)
      usage
      exit 0
      ;;
    --fresh)
      fresh=true
      ;;
    -*)
      echo "Fel: okänd flagga: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      if [[ -n "$input" ]]; then
        usage >&2
        exit 2
      fi
      input="$1"
      ;;
  esac
  shift
done

if [[ -z "$input" ]]; then
  usage >&2
  exit 2
fi

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$root_dir"

if [[ "$input" == videos/* ]]; then
  video_rel="$input"
elif [[ "$input" == */* ]]; then
  echo "Fel: filmen måste ligga under $root_dir/videos/." >&2
  echo "Flytta eller kopiera den dit och kör exempelvis: $0 videos/$(basename "$input")" >&2
  exit 2
else
  video_rel="videos/$input"
fi

video_abs="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$video_rel")"
videos_abs="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' videos)"
if [[ "$video_abs" != "$videos_abs/"* || ! -f "$video_abs" ]]; then
  echo "Fel: hittar inte $root_dir/$video_rel" >&2
  exit 2
fi
video_rel="videos/${video_abs#"$videos_abs/"}"

if ! command -v docker >/dev/null 2>&1; then
  echo "Fel: Docker-klienten saknas. Installera Docker CLI och Colima först." >&2
  exit 1
fi

docker_cmd=(docker --context colima)

if ! "${docker_cmd[@]}" info >/dev/null 2>&1; then
  if ! command -v colima >/dev/null 2>&1; then
    echo "Fel: Colima/Docker-kontexten är inte tillgänglig och Colima är inte installerat." >&2
    exit 1
  fi
  echo "Docker kör inte; startar Colima ..."
  if ! colima_output="$(colima start 2>&1)"; then
    printf '%s\n' "$colima_output" >&2
    if grep -qi 'host agent is not' <<<"$colima_output"; then
      cat >&2 <<'EOF'

Colima kunde inte starta på grund av VZ-felet "host agent is not". Starta om
Macen och kör samma kommando igen. Radera eller återskapa inte Colima-profilen
automatiskt; det kan förstöra lokala Docker-volymer.
EOF
    fi
    exit 1
  fi
  printf '%s\n' "$colima_output"
fi

if ! "${docker_cmd[@]}" info >/dev/null 2>&1; then
  echo "Fel: Docker-daemonen svarar fortfarande inte efter Colima-start." >&2
  exit 1
fi

if ! "${docker_cmd[@]}" buildx version >/dev/null 2>&1; then
  echo "Fel: Docker buildx-plugin saknas. Installera den med: brew install docker-buildx" >&2
  exit 1
fi

model_host="models/visdrone-yolov8s.pt"
echo "Kontrollerar VisDrone-s-vikter ..."
python3 scripts/fetch_visdrone.py --size s

if [[ ! -s "$model_host" ]]; then
  echo "Fel: VisDrone-vikterna saknas efter nedladdning: $root_dir/$model_host" >&2
  exit 1
fi

compose=("${docker_cmd[@]}" compose -f docker-compose.yml -f docker-compose.offline.yml)
export MODEL="/models/visdrone-yolov8s.pt"

echo "Bygger offline-tjänster ..."
"${compose[@]}" build analyze review

echo "Analyserar $video_rel med VisDrone-s ..."
analysis_args=(run --rm analyze "/videos/${video_rel#videos/}")
if [[ "$fresh" == false ]]; then
  analysis_args+=(--reuse-latest)
fi
"${compose[@]}" "${analysis_args[@]}"

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
