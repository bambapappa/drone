#!/usr/bin/env bash
# Native Mac/Linux launcher for the complete offline workflow:
# prerequisites -> model -> analysis -> review server.
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$root_dir"

fresh=false
video=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --fresh) fresh=true ;;
    --help|-h)
      cat <<'EOF'
Användning: bash scripts/start_workstation.sh [--fresh] [film]

Filmen ska ligga i videos/. Utan filmargument används videos/drone-halva2-brand.mp4
om den finns, annars den första .mp4-filen i videos/. --fresh tvingar en ny analys.
EOF
      exit 0
      ;;
    -*) echo "Fel: okänd flagga: $1" >&2; exit 2 ;;
    *)
      [[ -z "$video" ]] || { echo "Fel: ange bara en film." >&2; exit 2; }
      video="$1"
      ;;
  esac
  shift
done

mkdir -p videos analysis-output models
if [[ -z "$video" && -f videos/drone-halva2-brand.mp4 ]]; then
  video="videos/drone-halva2-brand.mp4"
elif [[ -z "$video" ]]; then
  video="$(find videos -maxdepth 1 -type f \( -iname '*.mp4' -o -iname '*.mov' \) -print -quit)"
fi
if [[ -z "$video" || ! -f "$video" ]]; then
  echo "Hittar ingen film i $root_dir/videos/. Lägg filmen där och kör skriptet igen." >&2
  exit 2
fi

if [[ ! -x .venv/bin/python ]]; then
  command -v python3 >/dev/null 2>&1 || { echo "Python 3 krävs." >&2; exit 1; }
  echo "Skapar projektets Python-miljö ..."
  python3 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -e ".[dev]"
fi

model="models/visdrone-yolov8m.pt"
if [[ ! -s "$model" ]]; then
  echo "Hämtar VisDrone-m-modellen ..."
  .venv/bin/python scripts/fetch_visdrone.py --size m --out "$model" --verify
fi

if [[ "$(uname -s)" == "Darwin" ]]; then
  device="mps"
else
  device="cpu"
fi

echo "Analyserar: $video"
analysis_args=("$video" --output analysis-output --model "$model" --device "$device"
  --imgsz 1536 --tiles 2 --detect-conf 0.05 --display-conf 0.30)
if [[ "$fresh" == false ]]; then
  analysis_args+=(--reuse-latest)
fi
.venv/bin/python analysis/cli.py "${analysis_args[@]}"

echo "Startar granskningsservern på http://localhost:8001 ..."
export ANALYSIS_OUTPUT_DIR="$root_dir/analysis-output"
export VIDEO_DIR="$root_dir/videos"
if command -v open >/dev/null 2>&1; then
  open http://localhost:8001 >/dev/null 2>&1 || true
fi
exec .venv/bin/python -m uvicorn review.main:app --host 0.0.0.0 --port 8001
