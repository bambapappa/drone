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
`tracklets/`, `persons/`, `events/` samt en separat tilläggsbar
`annotations/`-logg) i stället för att strömmas till en webbklient. Körs
via `analyze <film>` (CLI) eller `docker compose -f docker-compose.yml -f
docker-compose.offline.yml run --rm analyze <film>`.

Modulkarta, artefaktschema och passordning (P1 detektion → P2 spårning →
P3 identitet → P5 händelser) finns i [AGENTS.md](../AGENTS.md).

### Granskningsvy (`review/`)

Tunn klient + REST-API ovanpå sidecar-arkivet — importerar aldrig
analysmotorn, läser bara det `analyze` skrev och skriver enbart till den
separata `annotations/`-loggen (bokmärken, skärmdumpar). Uppspelning är
native HTML5 `<video>`; overlay-canvasen synkas mot videons `currentTime`
via `requestVideoFrameCallback` och `frames/meta`-PTS-indexet i stället för
att strömmas från servern. Skärmdumpar komponeras klientsidan (video +
canvas → PNG) — det finns bara en annoterad-bild-renderare, `snapshot.py`
(se verktygstabellen ovan) förblir kvar enbart som fristående felsöknings-
verktyg för realtidspipelinen. Körs via `uvicorn review.main:app` eller
`docker compose -f docker-compose.yml -f docker-compose.offline.yml up
review` (port 8001). Detaljer i [AGENTS.md](../AGENTS.md).

### Utvärderingslager (fas 3)

Tre nya rena moduler ovanpå samma artefakt + `annotations/`-logg, ingen av
dem rör analysmotorn:

| Fil | Ansvar |
|---|---|
| `review/operator_notes.py` | Tolkar en inklistrad/uppladdad textklump av operatörens fältanteckningar till `(t, text)`-rader. Överseende med format (se modulens docstring): tid och text i valfri ordning, kommatecken *eller* semikolon som fältavskiljare, decimalkomma i sekunder, valfri CSV-rubrikrad, kommentarer/tomrader hoppas över. En rad som inte går att tolka blir en varning, inte ett avbrutet import. |
| `review/comparison.py` | Parar ihop AI-händelser och importerade anteckningar på tidsnärhet (deterministisk giriga-närmast-par-algoritm, ingen textmatchning) till tre hinkar: hittad av båda / endast AI / endast operatör, med tidsskillnad för matchade par. Ren funktion — inget sparas, alltid samma resultat för samma indata. |
| `review/debrief.py` | Renderar en fristående HTML-debriefingsrapport (inline CSS, ingen server behövs) från ett jämförelseresultat — svensk text rakt igenom, bildreferenser (`ruta N–M`) i stället för inbäddade miniatyrbilder (se modulens docstring för varför). |

Granskningsbeslut (bekräfta/avvisa/kommentera) skrivs till
`annotations/verdicts.jsonl` — **inte** till `events/<pass>.jsonl`, vars
`review`-fält motorn skriver en gång och sedan aldrig rör (se
`analysis/events.py`). API:t slår ihop det senaste verdiktet ovanpå
händelsen vid läsning (`review/routes.py:_merge_verdict`); en omkörning av
analysen skriver om `events/` men rör aldrig `annotations/`.

### IRRATIONELLT, faromarkör och tidslinje (fas 4)

**`analysis/irrational.py`** härleder en tredje personbunden kategori,
IRRATIONELLT, ur samma tracklet-trajektorier STILLA/MOT FARA redan läser
(foot-center-position, kroppslängds-normaliserad hastighet — ingen ny
koordinatkonvention). Fem rena delsignalfunktioner (`_eval_erratic`,
`_eval_sprint`, `_eval_counterflow`, `_eval_oscillation`,
`_eval_freeze_bolt`), vägda samman till en poäng med ett
uthållighetskrav (mirrorar `BehaviorAnalyzer`s `still_since`/
`toward_since`-hysteres). Varje händelses `evidence.sub_signals` namnger
exakt vilka delsignaler som slog till och deras uppmätta värden — aldrig
en bar etikett, samma disciplin som P3:s `assoc_audit`. Precedensregel:
STILLA vinner över IRRATIONELLT på samma bildruta (implementerad genom att
`derive_events` samlar STILLA-händelsernas täckta bildrutor per tracklet
och tvingar dem till "ej utlöst" i ensemblen); MOT FARA har ingen
precedensrelation till någotdera.

**Faromarkör (`review/hazard.py`):** granskaren kan retroaktivt placera
eller flytta en faropunkt i granskningsvyn; `recompute_mot_fara` läser
P2:s redan lagrade tracklets och kör om `analysis.events.derive_behavior_events`
med den nya faropunkten — ingen ny P1–P3-körning. Detta är medvetet en
**delad ren funktion**, inte en dubblering av beteendematematiken (som
uppdraget explicit varnar för), och är förenligt med gränssnittsregeln
"granskningsvyn importerar aldrig motorn": den regeln skyddar mot att
anropa P1/P2/P3:s tunga, tillståndsbärande, modelldrivna pass, inte mot
att återanvända en redan ren, redan artefakt-baserad härledningsfunktion
— exakt den distinktion rapportens §6 drar ("allt som kan uttryckas som en
fråga mot artefakten är billigt"). Den omräknade MOT FARA-mängden rör
aldrig `events/<pass>.jsonl`; den slås ihop vid läsning precis som
verdikter (`review/routes.py:_apply_hazard_override`). Att flytta markören
carry:ar best-effort fram en tidigare avgiven Fas 3-verdikt från den
ursprungliga MOT FARA-händelsen till dess omräknade motsvarighet
(`_carry_forward_mot_fara_reviews`, nyckelad på tracklet_id + tidsnärhet —
inte person_id, som är null när P3 inte kördes). Faromarkörens
position lagras i `annotations/hazard_marker.jsonl` (senaste-rad-vinner,
som verdikter). Se DECISIONS.md B26 för fullständigt resonemang.

**Tidslinje:** en ny flik i granskningsvyn (`review/static/app.js:renderTimeline`)
ritar en SVG-remsa med en rad per person (STILLA/MOT FARA/IRRATIONELLT-
spann), en rad för farhändelser, en för bokmärken och en för importerade
operatörsanteckningar — klick söker videon dit. Ren klientlogik över
samma data som redan hämtas för händelselistan; inget nytt REST-anrop
utöver det befintliga `/events`.
