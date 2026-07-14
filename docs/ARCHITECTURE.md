# Arkitektur

Översikt av hur systemet hänger ihop. *Varför* bakom varje val: [DECISIONS.md](../DECISIONS.md).

## Dataflöde

```
videofil / RTSP / kamera
        │  (VideoSource: pacar filer till realtid, loopar, hanterar scenklipp)
        ▼
┌─ Pipeline (en per källa, två trådar) ───────────────────────────────────┐
│                                                                          │
│  RENDER-tråd  (~MAX_FPS, t.ex. 24 fps)                                   │
│    läs bild → ev. ROI-beskärning → IR-PiP-autodetekt                     │
│    → global kamerarörelse (gles LK-flöde)                                │
│    → flytta varje box med lokalt optiskt flöde + EMA/slew-utjämning      │
│    → slå ihop senaste detektionsresultat (rörelsekompenserat)            │
│    → JPEG-koda + bygg metadata → broadcasta                              │
│                                                                          │
│  DETECT-tråd  (så fort hårdvaran hinner, alltid på senaste bilden)       │
│    YOLO (+ ev. tiling) → BoT-SORT-spårning                               │
│    → personregister (utseende-re-ID, unik-räkning)                       │
│    → beteendeanalys (STILLA / MOT FARA i stabiliserade koord.)           │
│    → lägesbild (rök/eld-heuristik, rökdrift, basförslag)                 │
└──────────────────────────────────────────────────────────────────────────┘
        │  binärt WS-paket per bild:  [u32 metalängd][meta-JSON][JPEG]
        ▼
FastAPI ── WebSocket /ws/stream ──► webbklient (canvas ritar bild + lager)
        └─ REST /api/* (källval, faromarkering, status, uppladdning)
```

**Nyckelidé (B3):** detektion är långsam (~2–15 Hz), men boxarna ritas i full
bildtakt genom att render-tråden flyttar dem med optiskt flöde varje bild och
bara *korrigerar* mot detektionen när den blir klar — mjukt, aldrig hack/hopp.

## Trådar & synk

- **Render** äger visningen och spårens visningsläge. Den lämnar senaste bilden
  som ett "jobb" till detect-tråden (nyaste vinner — gammalt jobb kastas).
- **Detect** tar alltid senaste jobbet, kör tung analys, lämnar tillbaka ett
  resultat som render slår ihop nästa varv. Rörelsen som hann ske under
  analysen kompenseras (`global_offset` + per-spår flödesackumulering).
- **Broadcaster** fan-outar paket till alla WS-klienter med kö-längd 1 per
  klient (en långsam mobil tappar bilder men får aldrig växande fördröjning).

## Moduler (`app/vision/`)

| Fil | Ansvar |
|---|---|
| `pipeline.py` | Orkestrering: de två trådarna, spårens livscykel, paketbygge, källbyte (`PipelineManager`). |
| `sources.py` | `VideoSource`: fil/RTSP/kamera, realtidspacing, loop, scenklippsflagga. |
| `detector.py` | `Detector`: YOLO + BoT-SORT. Tiling-läget driver BoT-SORT manuellt. Klassnamn → människa/hot. |
| `tiling.py` | `N×N` överlappande rutor + global NMS (små mål på hög höjd). |
| `flow.py` | `GlobalMotion` (kamerarörelse), `local_box_flow` (boxflöde), `BoxFilter` (EMA+slew-utjämning). |
| `registry.py` | `PersonRegistry`: tracker-ID → stabilt `P1,P2…`, utseende-re-ID, unik-räkning (≥2 s bekräftelse). |
| `behavior.py` | `BehaviorAnalyzer`: STILLA / MOT FARA i kamerastabiliserade, höjd­normaliserade koordinater. |
| `situation.py` | Rök/eld-heuristik, rökdrift, basförslag (utväg/vändyta, undvik medvind). |
| `pip.py` | `PipAutoDetector`: hittar IR-bild-i-bild (hörn) eller 50% split via gråskale-signal. |
| `broadcast.py` | `Broadcaster`: trådsäker fan-out till WS-klienter. |

## App & API (`app/`)

| Fil | Ansvar |
|---|---|
| `main.py` | FastAPI-app, lifespan (startar pipeline mot defaultkälla), serverar statiskt GUI. |
| `routers/stream.py` | `/ws/stream` (binär ström) + `/api/*` (källa, fara, status, uppladdning). |
| `routers/health.py` | `/health`. |
| `core/config.py` | Alla inställningar (env/`.env`). Se [CONFIG.md](CONFIG.md). |
| `static/` | Webbklient: `index.html`, `app.js` (canvas + lager + togglar), `style.css` (mobile-first). |

## Metadata-paketet (server → klient)

Per bild skickas JSON + JPEG atomiskt. JSON innehåller bl.a.: `persons`
(box, `pid`, status, spår), `stats` (synliga, unika, irrationella),
`hazards` (rök/eld + drift), `base` (position + motivering), `danger`,
`wh`, `fps`. Klienten ritar — lager-togglar är därför rena klientval och
kräver ingen serveromrendering.

## Verktyg (`scripts/`)

| Skript | Syfte |
|---|---|
| `serve.sh` | Robust server-(om)start (`make serve`). |
| `make_demo_video.py` | Syntetiskt testklipp (panorering + eld/rök). |
| `eval_detection.py` | Jämför modeller/upplösningar/tiling på film (recall-mätning). |
| `fetch_visdrone.py` | Hämtar VisDrone-vikter till `models/`. |
| `snapshot.py` | Annoterad stillbild via WS utan webbläsare (felsökning/SSH). |
| `integration_check.py` | End-to-end-kontroll mot körande server (fps, boxhopp, räkning, lägesbild). |

## Test

`tests/` kör utan ML-beroende (snabbt, deterministiskt): beteende, register,
filter/rörelse, lägesbild, tiling, IR-PiP, config-validering, API.
`integration_check.py` täcker hela den körande pipelinen.

## Offline-analysverktyg (`analysis/`)

Fristående batchverktyg för sekventiell, deterministisk analys av en
inspelad film i efterhand (träning/utvärdering) — ett tillägg vid sidan av
ovanstående realtidspipeline, inte en ersättning. Delar inga körande
komponenter med `app/vision/`; de rena analysmodulerna (detektor/tiling,
register, beteende, lägesbild, flöde, PiP) är utklippta till egna,
oförändrade kopior i `analysis/`.

Skillnad mot realtidspipelinen: en process, sekventiella pass i stället för
render-/detect-trådar, och **videotid** (`t = bildruta/fps`, PTS-korrigerad)
i stället för `time.monotonic()`. Resultatet skrivs till ett versionerat
sidecar-arkiv (`manifest.json` + JSONL i `frames/`, `detections/`,
`tracklets/`, `persons/`) i stället för att strömmas till en webbklient. Körs
via `analyze <film>` (CLI) eller `docker compose -f docker-compose.yml -f
docker-compose.offline.yml run --rm analyze <film>`.

Modulkarta, artefaktschema och passordning (P1 detektion → P2 spårning →
P3 identitet) finns i [AGENTS.md](../AGENTS.md).
