# Efteranalys (icke-realtid)

Live-läget är byggt för låg fördröjning: det detekterar på en bakgrundstråd
några gånger i sekunden och bär boxarna mellan detektionerna med optiskt
flöde. Rätt avvägning för en *direktström* — men för **efteranalys** bryr vi
oss inte om fördröjning, vi vill ha bästa möjliga markering.

Offline-läget kör därför detektionen på **varje ruta** (ingen flödesgissning,
ingen realtidspacing) och skriver en självständig *bunt* som en spelare i
webbläsaren spolar fram och tillbaka i — exakt det du behöver för att stega
igenom och själv bedöma *vad modellen såg och vad den missade*, och för att
i efterhand fråga: *missade vi någon? var beslutet rätt?*

## Snabbstart

```bash
# 1. Lägg filmen i videos/ (eller ladda upp via panelen).
# 2. Kör analysen — modell/trösklar tas från .env, precis som servern:
python scripts/analyze_offline.py videos/film.mp4

# Skarp insatskonfig (se docs/DEMO.md):
MODEL=models/visdrone-yolov8s.pt IMGSZ=1280 CONF=0.2 TILES=2 \
    python scripts/analyze_offline.py videos/film.mp4

# 3. Öppna spelaren:
make serve            # eller: redan körande server
# http://localhost:8000/player
```

Du kan också starta analysen **från spelaren**: knappen **＋ Ny analys** →
välj film → *Kör analys* (kör som bakgrundsprocess, framsteg visas live).

## Spelaren

- **Spola fram/tillbaka:** dra i tidslinjen, eller ⏪ för bakåtuppspelning.
- **Stega ruta för ruta:** ⟨ / ⟩ (eller piltangenter), `mellanslag` = spela/pausa.
- **Hastighet:** 0.25×–4×.
- **Lager:** Boxar · ID · Spår · Beteende · Miljö · Bas (samma togglar som live,
  samma renderare — `static/overlay.js`).
- **Händelser:** klickbar tidslinje (person upptäckt, STILLA/LIGGER, mot fara,
  rök/brand indikerad). Klicka en händelse → hoppar dit i filmen.
- **Lägesbild:** rutans personer/lägesinfo vid aktuell tidpunkt.

## Bunten

`analyses/<namn>/` (gitignorad):

| Fil | Innehåll |
|---|---|
| `meta.json` | källa, fps, modell/imgsz/conf/tiles, sammanfattning |
| `frames.jsonl` | en metadatapost per analyserad ruta (samma schema som live-strömmen, utan JPEG) |
| `events.json` | tidslinje av noterbara övergångar |
| `state.json` | framsteg, skrivs live under körning |

Bilderna dupliceras **inte** in i bunten — spelaren strömmar originalfilmen
och lägger annoteringarna ovanpå, synkat på tid.

## Diagnostisera missade detektioner

Det här är rätt yta för att jaga undercount. Kör samma film med olika modeller
och upplösningar och jämför ruta för ruta:

```bash
python scripts/analyze_offline.py videos/film.mp4 --out analyses/film-nano
MODEL=models/visdrone-yolov8s.pt IMGSZ=1280 CONF=0.2 TILES=2 \
    python scripts/analyze_offline.py videos/film.mp4 --out analyses/film-visdrone
```

Växla mellan analyserna i spelarens rullgardin. `scripts/eval_detection.py`
ger samma jämförelse i siffror (antal/konfidens/hastighet) på provrutor.

> Markering av facitrutor (ground truth) för mätning och eventuell träning är
> nästa steg — bunten är redan ett bra utgångsläge att rätta mot.

## Tips

- `--stride N` analyserar var N:te ruta (snabbare, glesare tidslinje).
- Ingen GPU? Det går, bara långsammare — offline har ingen realtidsbudget.
- Split-screen IR: sätt `ANALYSIS_ROI` som vanligt; analysen beskär likadant.
  (Spelarens overlay antar att filmen visas obeskuren — utan ROI ligger
  markeringarna exakt; med ROI är de relativa till den beskurna vyn.)
