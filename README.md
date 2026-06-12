# Insatsdrönare — PoC för räddningstjänst

Realtidsanalys av drönarvideo i webbläsaren: detekterar och följer människor,
räknar unika personer, flaggar irrationellt beteende (stilla / rör sig mot
fara), gör en enkel lägesbild (rök/eld, rökdrift, förslag på basplats) och
larmar vid hot. Byggt för att fungera likadant på inspelad övningsfilm som på
en riktig drönarström — samma kodväg, inga per-video-inställningar.

Arkitektur och alla designbeslut: se **[DECISIONS.md](DECISIONS.md)**.

## Snabbstart (Docker)

```bash
# 1. Lägg en eller flera filmer i videos/
cp /sökväg/till/övningsfilm.mp4 videos/

# 2. Bygg och starta (modellen bakas in i imagen vid bygget)
docker compose up --build

# 3. Öppna http://localhost:8000 i webbläsaren (mobil eller dator)
```

Utan egen film? Generera ett testklipp (panorerande kamera över en stillbild
med människor + syntetisk eld/rök):

```bash
python scripts/make_demo_video.py        # skriver videos/demo.mp4
```

## GUI

- **Statusrad:** synliga nu · unika totalt · irrationella · hotlarm · fps (video·analys).
- **Lager (togglas per klient):** Boxar, ID, Spår, Beteende, Hot, Miljö (rök/eld), Bas.
- **🎯 Markera fara:** tryck på knappen och sedan i bilden — personer som rör
  sig mot punkten flaggas orange (MOT FARA). Punkten följer kamerarörelsen.
- **Panel:** lägesbild med motivering av basförslaget, hotlista, källval och
  uppladdning av film.
- Färger: grön = OK, **röd = stilla/livlös** (LIGGER om posen indikerar det),
  **orange = rör sig mot faran**, mörkröd = hotobjekt.

Anpassat för liten skärm (mobil för fältpersonal) och storbild (ledningscentral).
Flera klienter kan titta samtidigt med olika lagerval; en långsam klient får
lägre bildfrekvens men aldrig växande fördröjning.

## Källor

`SOURCE` i `.env` (eller källväljaren i GUI:t):

| Typ | Exempel |
|---|---|
| Fil i `videos/` | `SOURCE=` (tom = första filen), eller välj i GUI |
| RTSP/HTTP-ström | `SOURCE=rtsp://drönare:8554/stream` |
| Kamera | `SOURCE=0` |

Filer spelas i realtid (egen fps, loopas med `LOOP=true`) så att systemet
beter sig exakt som mot en live-ström.

## Modell

Standard är `yolo11n.pt` (COCO). Vilken Ultralytics-modell som helst kan
användas — klassnamnen introspekteras, så en VisDrone-tränad modell
(`pedestrian`/`people`) fungerar direkt.

**VisDrone** (rekommenderas för riktig drönarfilm — tränad på små människor
från hög höjd):

```bash
python scripts/fetch_visdrone.py          # hämtar vikter till models/ (kräver internet)
# sätt sedan i .env:
MODEL=models/visdrone-yolov8s.pt          # samma sökväg funkar nativt och i Docker
```

`models/` monteras in i containern automatiskt. Obs: VisDrone saknar
vapenklasser, så hotlagret blir tyst med den modellen (`THREAT_CLASSES`
kräver en modell som har klasserna — rörledningen är klar för en
specialmodell i PoC 2). Annan officiell modell bakas in med
`docker compose build --build-arg MODEL=yolo11s.pt`.

## Värmekamera / split-screen

För film där IR- och vanlig bild visas sida vid sida (eller bild-i-bild):
beskär analysen till den ena vyn så personer inte dubbelräknas:

```bash
ANALYSIS_ROI=0,0,0.5,1      # vänstra halvan ("x,y,w,h", normaliserat 0..1)
```

Ren IR-film fungerar ofta acceptabelt med RGB-tränade modeller (white-hot),
men med sänkt träffsäkerhet — testa mot ditt material.

## Lokal utveckling

```bash
pip install uv
uv pip install --system -r pyproject.toml && uv pip install --system ruff pytest pytest-asyncio
make dev            # uvicorn med reload på :8000

make test           # enhetstester (ML-fritt: beteende, re-ID, filter, lägesbild, API)
make lint
make demo-video     # generera videos/demo.mp4
make check          # integrationskontroll mot körande server
python scripts/snapshot.py --out snap.jpg   # annoterad stillbild utan webbläsare
```

## Viktigaste inställningarna (`.env`)

| Variabel | Default | Beskrivning |
|---|---|---|
| `MODEL` | `yolo11n.pt` | Ultralytics-vikter (COCO/VisDrone/egna) |
| `IMGSZ` | `640` | Inferensupplösning (högre = bättre små mål, långsammare) |
| `CONF` | `0.30` | Detektionströskel |
| `MAX_FPS` | `24` | Utströmmens bildfrekvens |
| `OUT_WIDTH` | `960` | Utströmmens bredd (bandbredd) |
| `BEH_STILL_TIME_S` | `4.0` | Sekunder utan rörelse innan STILLA |
| `BEH_TOWARD_ANGLE_DEG` | `40` | Riktningstolerans för MOT FARA |

Alla trösklar ligger i `app/core/config.py`, är env-styrbara och identiska för
allt material — utfallet på osedd film är förutsägbart.

## Begränsningar (PoC 1) och vägen framåt

Medvetna avgränsningar — georeferens kräver drönartelemetri, hotmodellen är
COCO-begränsad, rök/eld är heuristik, re-ID är apparensbaserad per session.
Detaljer och PoC 2/MCP-tankar i [DECISIONS.md](DECISIONS.md).
