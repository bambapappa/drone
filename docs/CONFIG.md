# Konfiguration

Denna sida gäller **realtidspipelinen** (`app/`). Alla inställningar sätts
som miljövariabler (versalt fältnamn) eller i en `.env`-fil i repo-roten.
Defaultvärden lever i `app/core/config.py` och är **identiska för allt
material** — samma konfiguration ger förutsägbart utfall på osedd film.
Inga per-video-trösklar.

Det fristående offline-batchverktyget (`analysis/`, se
[ARCHITECTURE.md](ARCHITECTURE.md)) konfigureras i stället via CLI-flaggor
(`analyze --help`), inte dessa miljövariabler.

Granskningsvyn (`review/`) tar två egna miljövariabler:

| Variabel | Default | Beskrivning |
|---|---|---|
| `ANALYSIS_OUTPUT_DIR` | `analysis-output` | Var sidecar-arkiven som `analyze` skrev ligger (läses och annoteras). |
| `VIDEO_DIR` | `videos` | Var originalfilmerna ligger, för uppspelning i `<video>`-elementet. |

Defaultvärdena matchar `docker-compose.offline.yml`s volymmonteringar, så
`uvicorn review.main:app` (eller `make review`) fungerar utan konfiguration
från repo-roten.

Fas 3:s jämförelse-/debriefingslut (`/api/runs/{id}/comparison` och
`.../debrief`) tar en valfri `tolerance_s`-frågeparameter (default `60.0`,
se `review/comparison.py` för motiveringen) i stället för en miljövariabel
— den är per-körning/exportval, inte en tjänstenivå-inställning.

Fas 4:s IRRATIONELLT-trösklar (`OfflineConfig`s `irr_*`-fält,
`analysis/irrational.py`) och faromarkören (`review/hazard.py`,
`POST /api/runs/{id}/hazard-marker`) följer samma mönster som `beh_*`:
inga CLI-flaggor eller miljövariabler ännu — trösklarna är konstruktörs-
defaultvärden, motiverade i modulens docstring, tills ett behov av att
justera dem per körning uppstår.

## Videokälla

| Variabel | Default | Beskrivning |
|---|---|---|
| `SOURCE` | _(tom)_ | Filväg, RTSP/HTTP-URL eller kameraindex (`0`). Tom = första filen i `VIDEO_DIR`. |
| `VIDEO_DIR` | `videos` | Mapp som skannas efter filmer och dit uppladdningar sparas. |
| `LOOP` | `true` | Spela om filer när de tar slut (ignoreras för live-strömmar). |

Filer spelas i sin egen bildtakt (realtid) så att systemet beter sig som mot
en live-feed. Scenklipp vid omspelning hanteras (spår nollställs, ID:n
återidentifieras via utseende).

## Modell & detektering

| Variabel | Default | Beskrivning |
|---|---|---|
| `MODEL` | `yolo11n.pt` | Valfri Ultralytics `.pt`. Klassnamn introspekteras. För drönarfilm: `models/visdrone-yolov8s.pt`. |
| `DEVICE` | `cpu` | `cpu`, `mps` (Apple-GPU/M-serie) eller `cuda`. |
| `IMGSZ` | `640` | Inferensupplösning. Högre = hittar mindre människor, långsammare. 1280 rekommenderas på GPU. |
| `CONF` | `0.30` | Detektionströskel. 0.20 för eftersök (recall före precision). |
| `IOU` | `0.50` | NMS-överlapp. |
| `TILES` | `1` | `N×N` rutad inferens (1–3). 2 fördubblar effektiv upplösning för små mål; tyngre. Kräver GPU/stark CPU. |
| `HUMAN_CLASSES` | `person,pedestrian,people` | Klassnamn som räknas som människa (täcker COCO + VisDrone). |
| `THREAT_CLASSES` | _(tom)_ | Hotklasser att flagga (utlyft ur PoC 1). Sätt t.ex. `knife` med COCO-modell för att återaktivera rörledningen. |

## Värmekamera / bild-i-bild / beskärning

| Variabel | Default | Beskrivning |
|---|---|---|
| `PIP_AUTODETECT` | `true` | Autodetektera IR-ruta (valfritt hörn) eller 50% split och exkludera den. Manuella inställningar nedan vinner. |
| `IGNORE_REGIONS` | _(tom)_ | `;`-separerade `x,y,w,h` (0..1) att exkludera från analys. För hörn-PiP. |
| `ANALYSIS_ROI` | _(tom)_ | `x,y,w,h` (0..1) att beskära **före** all analys. För split: t.ex. `0,0,0.5,1` (vänster halva). |

## Utström (till webbläsaren)

| Variabel | Default | Beskrivning |
|---|---|---|
| `MAX_FPS` | `24` | Bildtakt ut (boxar uppdateras alltid i denna takt, oberoende av detektionstakt). |
| `OUT_WIDTH` | `960` | Bredd på utströmmen (bandbredd; ~8 Mbit/s vid 960/q70). |
| `JPEG_QUALITY` | `70` | JPEG-kvalitet 1–100. |

## Boxutjämning (följsamhet)

| Variabel | Default | Beskrivning |
|---|---|---|
| `SMOOTH_TAU_POS` | `0.12` | Positions-tidskonstant (s). Lägre = snappare, högre = mjukare. |
| `SMOOTH_TAU_SIZE` | `0.18` | Storleks-tidskonstant (s). |
| `SMOOTH_SLEW` | `3.0` | Max glidhastighet (boxstorlekar/s) för detektionskorrigeringar → ingen teleportering. |

## Återigenkänning / unik-räkning

| Variabel | Default | Beskrivning |
|---|---|---|
| `REID_SIM_THRESH` | `0.86` | Min. utseendelikhet (HSV-histogram) för att återanvända person-ID. |
| `REID_MAX_GAP_S` | `60` | Hur länge en tappad person kan återidentifieras. |
| `REID_MAX_DIST_FRAC` | `0.45` | Max rimligt förflyttningsavstånd vid återinträde (andel av bilddiagonal/s). |

En person räknas som **unik** först efter ≥ 2 s existens (motverkar
ID-fladder på små lågkonfidenta detektioner).

## Beteende (irrationellt)

| Variabel | Default | Beskrivning |
|---|---|---|
| `BEH_STILL_SPEED` | `0.10` | Hastighet under denna (kroppslängder/s) räknas som stilla. |
| `BEH_STILL_TIME_S` | `4.0` | Sekunder stilla innan **STILLA** flaggas. |
| `BEH_PRONE_ASPECT` | `1.4` | Bredd/höjd över detta = liggande pose ("LIGGER"). |
| `BEH_TOWARD_SPEED` | `0.25` | Min. fart mot faran för **MOT FARA**. |
| `BEH_TOWARD_ANGLE_DEG` | `40` | Riktningstolerans mot faropunkten. |
| `BEH_TOWARD_TIME_S` | `1.5` | Ihållande tid innan MOT FARA flaggas. |
| `BEH_WINDOW_S` / `BEH_MIN_HISTORY_S` | `6.0` / `3.0` | Analysfönster / krävd historik. |

Mätt i **kamerastabiliserade** koordinater (drönarrörelse kompenseras) och
normaliserat mot personens boxhöjd → skalinvariant.

## Lägesbild (rök/eld/bas)

| Variabel | Default | Beskrivning |
|---|---|---|
| `FIRE_REQUIRE_SMOKE` | `true` | Flagga eld bara om rörlig rök finns intill (stoppar falsklarm på röda tegeltak). |
| `HAZARD_MIN_AREA` | `0.004` | Min. andel av bilden för rök/eld-blob. |
| `HAZARD_HOLD_S` | `2.0` | Sekunder ihållande innan rök/eld rapporteras. |
| `SMOKE_FLOW_EMA` | `0.15` | Utjämning av rökens driftriktning. |
| `BASE_MARGIN` | `0.08` | Håll basförslaget så långt från bildkanten. |
| `BASE_HYSTERESIS` | `0.15` | Flytta basmarkören bara vid större förändring (ingen vandring). |

## Övrigt

| Variabel | Default | Beskrivning |
|---|---|---|
| `APP_NAME` / `VERSION` / `DEBUG` | — | Standard FastAPI-metadata. |

Se [DECISIONS.md](../DECISIONS.md) för *varför* varje default är satt som det är
(B1–B21), och [DEMO.md](DEMO.md) för rekommenderad demokonfiguration.
