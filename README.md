# Insatsdrönare — PoC för räddningstjänst

Realtidsanalys av drönarvideo i webbläsaren. **Kärnan: hitta och följa
människor** — rita boxar som följer mjukt, räkna unika personer, och flagga
den som ligger/står stilla (t.ex. skadad) eller rör sig mot faran. Ovanpå det
en enkel lägesbild (rök/eld, rökdrift, basförslag med utväg).

Byggt för att bete sig likadant på inspelad övningsfilm som på en riktig
drönarström — samma kodväg, inga per-video-inställningar, förutsägbart på
osedd film.

📖 **[Demo-körschema](docs/DEMO.md)** · **[Konfiguration](docs/CONFIG.md)** ·
**[Arkitektur](docs/ARCHITECTURE.md)** · **[Designbeslut](DECISIONS.md)**

---

## Snabbstart

```bash
make serve                       # startar servern, väntar på health, skriver ut URL
# öppna http://localhost:8000  (laptop, platta eller mobil)
```

Eller via Docker (CPU): `docker compose up --build`.

Saknas film? `python scripts/make_demo_video.py` skapar `videos/demo.mp4`.
För skarp körning, se **[docs/DEMO.md](docs/DEMO.md)** — i korthet: VisDrone-modell,
`DEVICE=mps` på M-serie-Mac, `IMGSZ=1280`, `CONF=0.20`.

## Vad systemet gör

- **Persondetektering & spårning** — YOLO + BoT-SORT med kamerarörelse­kompensation.
  Boxarna uppdateras i full bildtakt via optiskt flöde, så de följer mjukt även
  när detektionen är långsam (inget hack/hopp).
- **Unik-räkning** — stabila ID:n `P1, P2…`; personer som lämnar och kommer
  tillbaka återidentifieras på utseende (apparens, per session — ingen biometri).
- **Beteende** — **STILLA** (röd, ligger/står stilla länge) och **MOT FARA**
  (orange, rör sig mot markerad fara), mätt i kamerastabiliserade koordinater.
- **Lägesbild** — rök/eld-heuristik (eld kräver samtidig rök → inga falsklarm på
  tegeltak), rökens driftriktning, och basförslag som väger in **utväg** (öppen
  korridor till bildkant) och **vändyta**, bort från rökens medvind.
- **Värmekamera** — IR-bild-i-bild (valfritt hörn) eller 50%-split känns igen
  **automatiskt** och exkluderas/beskärs så att folk inte dubbelräknas.

*(Hotdetektion — vapen/farligt gods — är utlyft ur PoC 1; rörledningen finns kvar
bakom `THREAT_CLASSES`.)*

## GUI

- **Statusrad:** synliga nu · unika totalt · irrationella · fps (video·analys).
- **Lager (togglas per klient):** Boxar · ID · Spår · Beteende · Miljö (rök/eld) · Bas.
- **🎯 Markera fara:** tryck knappen och sedan i bilden — punkten följer
  kamerarörelsen; personer mot den flaggas orange.
- **Panel:** lägesbild med basmotivering, källbyte och filuppladdning.
- Färger: grön = OK · **röd = STILLA** (LIGGER om posen indikerar) · **orange = MOT FARA**.

Mobile-first för fältpersonal, fungerar lika bra på storbild i ledningscentral.
Flera klienter samtidigt med olika lagerval; en långsam klient får lägre
bildfrekvens men aldrig växande fördröjning.

## Modell

Standard `yolo11n.pt` (COCO). **För riktig drönarfilm rekommenderas VisDrone**
(tränad på små människor från höjd — COCO-modeller är nära blinda där):

```bash
python scripts/fetch_visdrone.py     # -> models/visdrone-yolov8s.pt
# .env:  MODEL=models/visdrone-yolov8s.pt
```

Vilken Ultralytics-`.pt` som helst fungerar — klassnamnen introspekteras.
`models/` monteras in i Docker automatiskt. Detaljer: [docs/CONFIG.md](docs/CONFIG.md).

## Offline-analys (batchläge)

Utöver realtids-PoC:n ovan finns ett fristående batchverktyg i `analysis/` för
att köra sekventiell, deterministisk analys mot en inspelad film i efterhand
(träning/utvärdering) — samma analysmoduler, men egen kodväg och körtid; ett
tillägg, inte en ersättning för realtidspipelinen.

```bash
docker compose -f docker-compose.yml -f docker-compose.offline.yml run --rm analyze /videos/film.mp4
# eller nativt (efter make venv):
analyze /videos/film.mp4
```

Resultatet skrivs till ett versionerat sidecar-arkiv (`manifest.json` + JSONL)
under `analysis-output/`. Nuvarande fas kör ingest (PTS-index, videohash,
IR-PiP-lås) samt P1 (detektion) och P2 (spårning); modulkarta och detaljer i
[AGENTS.md](AGENTS.md).

## Utveckling

```bash
make venv           # skapar .venv och installerar allt (Mac/Linux)
source .venv/bin/activate
make dev            # uvicorn med reload på :8000
make test           # enhetstester (ML-fritt, snabbt, deterministiskt)
make lint           # ruff check + format
make demo-video     # videos/demo.mp4
make serve          # robust (om)start av servern
make check          # integrationskontroll mot körande server
```

Källor: `SOURCE` kan vara filväg, `rtsp://…`-URL eller kameraindex (`0`).
Filer spelas i realtid och loopas (`LOOP=true`).

## Begränsningar (PoC 1)

Medvetna avgränsningar: hotdetektion utlyft, georeferens kräver drönartelemetri,
rök/eld + öppen-mark är bildheuristik (märkt som förslag — räddningsledaren
beslutar), re-ID är apparensbaserad per session. Detaljer och PoC 2/MCP-tankar
i **[DECISIONS.md](DECISIONS.md)**.
