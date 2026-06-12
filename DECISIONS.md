# Designlogg — Drone Insats-PoC

Löpande logg över arkitektur- och designbeslut. Nyaste beslut läggs till längst ned i respektive sektion. Språk: svenska för dokumentation, engelska i kod.

## Mål (PoC 1)

Ett körbart system (Docker) som tar emot en videoström — i PoC:n en inspelad övningsfilm som strömmas internt, senare en riktig drönarfeed — och i nära realtid:

1. Detekterar människor och ritar boxar som **följer mjukt utan att hoppa eller lagga**.
2. Återigenkänner personer som setts tidigare (per session) och håller **totalräkning av unika personer**.
3. Klassar **irrationellt beteende**: ligger/står stilla länge, eller rör sig mot markerad fara. Markeras i avvikande färg och räknas separat.
4. Gör en enkel **lägesbedömning**: rök-/branddetektion (heuristik), rökens driftriktning, och **förslag på basplats** för räddningsledning (uppvind, bort från faran).
5. Flaggar **hot** (t.ex. kniv via objektdetektion) med larmbanner.
6. Visar allt i ett **webb-GUI anpassat för liten skärm** (mobil/platta för fältpersonal, större skärm för ledningscentral) med **togglebara lager**.

Uttryckligt krav: inget fejkat — byts filmen mot en annan eller mot en riktig feed ska utfallet vara förutsägbart även på osedd film.

## Arkitekturöversikt

```
videofil / RTSP-URL
        │
        ▼
┌─ Pipeline (bakgrundstrådar) ─────────────────────────────┐
│  Tråd A "render" (~24 fps):                              │
│    läs bild → global kameraflödesskattning (LK)          │
│    → propagera boxar med lokalt optiskt flöde            │
│    → One Euro-filter per box → JPEG + metadata-paket     │
│  Tråd B "detect" (så fort CPU:n hinner, ~5–10 Hz):       │
│    YOLO + BoT-SORT-tracking → person-register (re-ID)    │
│    → beteendeanalys → lägesbild (rök/eld/bas) → hot      │
└──────────────────────────────────────────────────────────┘
        │  binärt WS-paket: [meta-JSON][JPEG]
        ▼
FastAPI ── WebSocket /ws/stream ──► webbklient (canvas)
        └─ REST /api/* (källval, faromarkering, status)
```

## Beslut

### B1. Detektion: Ultralytics YOLO, modell utbytbar via env
- **Val:** `yolo11n.pt` (COCO) som standard, konfigurerbar via `MODEL` (env). Modellens klassnamn introspekteras: klasser med namn `person`, `pedestrian`, `people` behandlas som människa — därmed fungerar både COCO-modeller och VisDrone-tränade modeller (t.ex. yolov8/yolo11 finetunad på VisDrone) utan kodändring.
- **Varför:** COCO-vikter är officiella, reproducerbara och nedladdningsbara vid Docker-build (förutsägbarhet). VisDrone-vikter ger bättre träff på små människor från hög höjd men är tredjeparts — de stöds genom att montera in en .pt-fil och sätta `MODEL=/models/visdrone.pt`.
- **Avvägning:** nano-modell på CPU för att hålla realtid. Större modell (`yolo11s/m`) kan väljas via env om GPU finns.

### B2. Tracking: BoT-SORT (inbyggd i Ultralytics) med kamerarörelsekompensation
- **Val:** `model.track(persist=True)` med egen tracker-konfig (`botsort_drone.yaml`): hög `track_buffer` (≈8 s) så att korta ocklusioner inte byter ID, `gmc_method: sparseOptFlow` för drönarrörelse.
- **Varför:** BoT-SORT + GMC är beprövat för rörlig kamera; inget egenbygge att underhålla.

### B3. Mjuka boxar: render-tråd med optiskt flöde + flödesframmatat EMA/slew-filter
- **Problem:** På CPU hinner YOLO bara ~5–10 Hz. Att rita boxar enbart vid detektion ger hack/lagg — uttryckligen oacceptabelt.
- **Val:** Två trådar. Render-tråden går i visningstakt (~24 fps) och flyttar varje aktiv box med **lokalt optiskt flöde** (Lucas–Kanade på punkter inom boxen) varje bildruta; detektionskorrigeringar kompenseras för den rörelse som hunnit ske medan YOLO räknade. Visningsboxen filtreras varje bildruta: flödesrörelsen matas fram **1:1** (kamerarörelse släpar aldrig), och kvarvarande korrektionsresidual jagas ikapp via EMA med **slew-begränsning** (max 3 boxstorlekar/s) — stora korrektioner blir korta glid, aldrig teleportering.
- **Förkastat under utveckling (mätt, inte gissat):** One Euro-filter — dess hastighetsterm tolkar en stor detektionskorrigering som snabb rörelse och släpper igenom ~90 % av hoppet i en enda bildruta. Integrationsmätningen (max boxhopp per paket) gick från 0,37 → 0,07 (normerade enheter) med flödesframmatning + slew.
- **Dessutom:** spår som inte återdetekterats inom en kort frist (≈3,5 detektionsperioder, max 1 s) tas bort i stället för att spöklika boxar driver vid bildkanten; vid återinträde återfår personen sitt ID via registret och boxen glider in från senast kända läge.
- **Varför:** Boxarna uppdateras varje visad bildruta i stället för i detektionstakt → följsamt utan hopp, ärligt beräknat (flödet mäts i den faktiska bilden, inget gissas fram ur tomma intet).

### B4. Återigenkänning (re-ID): personregister ovanpå tracker-ID
- **Val:** Eget register som mappar tracker-ID → stabilt person-ID (`P1, P2, …`). När ett nytt tracker-ID dyker upp jämförs HSV-färghistogram (utseende) mot galleri av tidigare sedda personer vars spår tappats; matchning (cosinuslikhet + rimlighetskontroll i stabiliserade koordinater) återanvänder person-ID:t, annars skapas nytt. Unika-räknaren = antal person-ID.
- **Varför:** Tracker-ID dör vid längre ocklusion/utträde ur bild. Histogram-galleri är billigt, körbart på CPU och ger rimlig återigenkänning inom en insats. **Begränsning (medveten):** detta är apparens-baserat per session — ingen biometrisk identifiering, och två personer i likadana kläder kan förväxlas. PoC 2: riktig re-ID-embedding (t.ex. OSNet).

### B5. Beteendeanalys i kamerastabiliserade koordinater
- **Problem:** Drönaren rör sig — "ligger stilla" går inte att mäta i råa pixelkoordinater.
- **Val:** Global kamerarörelse skattas per bildruta (median av glesa LK-flödesvektorer); personpositioner ackumuleras i ett stabiliserat koordinatsystem (translation kompenseras; rotation/zoom ignoreras i PoC). Hastighet normaliseras med personens boxhöjd → ungefär "kroppslängder/sekund", skalinvariant.
- **Regler (fasta trösklar, samma för all film = förutsägbart):**
  - **STILLA** (röd): normerad fart < 0,10 kroppslängder/s i ≥ 4 s (kräver ≥ 3 s historik). Liggande pose (bredd/höjd > 1,4) skärper bedömningen.
  - **MOT FARA** (orange): faropunkt satt av operatör (tap i bilden) eller autodetekterad eld; rörelseriktning inom 40° mot faran, fart > 0,25 kl/s, avstånd minskar, ihållande ≥ 1,5 s.
  - Övriga: OK (grön).
- **Varför trösklar och inte ML:** transparent, justerbart, deterministiskt; ingen träningsdata för "irrationellt beteende" finns. PoC 2 kan lära trösklar ur data.

### B6. Lägesbild: heuristisk rök/eld + basplatsförslag
- **Val:** Eld = färgheuristik (mättade röd/orange-regioner), rök = lågmättade gråa regioner med rörelse; rökens driftriktning = medianflöde (Farnebäck, nedskalat) i rökmasken, EMA-utjämnad. Basplatsförslag = riktning **motvind** (motsatt rökdrift) om rök finns, annars bort från faropunkten; placeras mot bildkant med hysteres så markören inte vandrar. Motivering visas i klartext i GUI:t.
- **Varför:** Ärlig, billig beräkning ur själva bilden. Tydligt skyltad som *heuristik/förslag* — beslutet är alltid räddningsledarens. **Begränsning:** allt är i bildkoordinater (ingen georeferens utan drönartelemetri — PoC 2).

### B7. Hotdetektion: klassbaserad via samma YOLO-pass
- **Val:** Detekterade objekt vars klassnamn finns i `THREAT_CLASSES` (default: `knife`) flaggas med röd box + larmbanner (debounce 2 s). Miljöhot (eld/rök) listas separat.
- **Varför:** COCO saknar skjutvapen/farligt gods-skyltar; en specialmodell (eller öppen-vokabulär som YOLO-World) är PoC 2. Rörledningen (detektion → larm → GUI) är dock på plats och en hotmodell kan bytas in via `MODEL`/`THREAT_CLASSES` utan kodändring.

### B8. Leverans till klient: WebSocket med JPEG + metadata per bildruta
- **Val:** Binärt WS-meddelande per bildruta: `[längd][meta-JSON][JPEG]`. Klienten ritar bilden på canvas och lagren (boxar, ID, spår, status, bas, rök, hot) ovanpå — **togglar är därmed rena klientval** och kräver ingen serveromrendering. Per-klient-kö med längd 1 (äldsta kastas) → en långsam mobil ger sig själv lägre fps men aldrig växande fördröjning.
- **Förkastat:** (a) Server ritar in grafiken i bilden — omöjliggör per-klient-togglar. (b) HLS/WebRTC + separat metadatakanal — bättre bandbredd men synkdrift mellan video och boxar är exakt det "hopp" som inte får finnas; WebRTC-stacken är också tung för en PoC. Paketering bild+meta atomiskt garanterar perfekt synk.
- **Bandbredd:** ~24 fps × ~40 kB ≈ 8 Mbit/s vid 960 px bredd, JPEG q70 — OK på WiFi/5G; `OUT_WIDTH`/`JPEG_QUALITY`/`MAX_FPS` är env-justerbara.

### B9. Källabstraktion: fil och ström är samma kodväg
- **Val:** `SOURCE` kan vara en filväg, RTSP/HTTP-URL eller kameraindex; allt läses via OpenCV/FFmpeg. Filer pacas till sin egen fps (uppspelning i realtid, loopbar med `LOOP=true`) så att systemet beter sig som mot en live-ström. Byte av källa i GUI:t startar om pipelinen.
- **Varför:** uppfyller kravet att PoC:n kör inspelade filmer idag och drönarfeed imorgon utan kodändring och med förutsägbart utfall.

### B10. GUI: ett statiskt SPA, mobile-first, svenska
- **Val:** Ren HTML/JS/canvas utan byggsteg, mörkt tema, stora tryckytor. Statuschips överst (synliga, unika totalt, irrationella, hot, fps), togglechips nederst, hopfällbar lägespanel. Tap på bilden i "markera fara"-läge sätter faropunkt (delas globalt — en lägesbild för alla klienter).
- **Varför:** Inget ramverk = trivialt att köra i Docker och att granska; fungerar på mobil, platta och storbild.

### B11. Drift: Docker, CPU-default, modell bakas in i imagen
- **Val:** `python:3.12-slim` + ffmpeg-bibliotek; PyTorch CPU-wheels (mindre image, ingen CUDA); YOLO-vikter laddas ned vid build så att containern är körbar offline. `./videos` monteras som volym; uppladdning även möjlig via GUI.
- **Varför:** "körbart från typ docker" utan nätberoende vid demo.

### B12. Förutsägbarhet på osedd film
- Inga per-video-inställningar, inga inlärda trösklar, ingen cache mellan körningar. Samma konfiguration ⇒ samma beteende på nytt material. Alla trösklar samlade i `app/core/config.py` och dokumenterade i README.

### B13. Python-version
- Lokal utvecklings-/CI-miljö kan vara 3.11; `requires-python` sänkt till `>=3.11` (Docker kör 3.12). Ultralytics stödjer båda.

## Medvetna begränsningar i PoC 1 (kandidater för PoC 2)

- Ingen georeferens (allt i bildkoordinater) — kräver drönartelemetri (GPS/gimbal) → riktiga positioner, vindriktning från väderdata, basförslag på karta.
- Apparens-re-ID per session, ej över sessioner; ingen biometri (avsiktligt, även integritetsskäl).
- Hotmodell begränsad till COCO-klasser; PoC 2: specialtränad vapen-/farligt gods-modell eller öppen-vokabulärmodell.
- Rök/eld är färg-/rörelseheuristik, ej tränad segmentering.
- En källa åt gången, ingen inspelning/replay, ingen autentisering (demo på betrott nät).
- PoC 2+/MCP-tankar: flera samtidiga drönare, händelselogg med tidsstämplar, MCP-server som exponerar lägesbilden (personantal, flaggade, hot, basförslag) som verktyg för LLM-agenter i ledningsstöd, larmintegration, kartvy.

### B14. Scenklipp (filloop, källglapp) hanteras explicit
- **Problem:** Demofilmer loopas; ompositioneringen är ett hårt scenklipp som teleporterar alla boxar och förgiftar kamerarörelseskattningen. Riktiga strömmar kan ha motsvarande glitchar (tappade bildrutor, I-frame-hopp).
- **Val:** Loopomstart signaleras från källan; pipelinen rensar då visningsspår, tracker-tillstånd och beteendehistorik. Person-ID:n överlever ändå — registret återidentifierar på utseende. Kamerarörelseskattningen ignorerar dessutom orimligt stora skift (> 25 % av bildbredden per bildruta) som säkerhetsnät.

## Verifiering (körs mot riktig pipeline, ej mockad)

- `tests/` — 35 enhetstester för logikmodulerna (beteende, register, filter, lägesbild, API) utan ML-beroende.
- `scripts/make_demo_video.py` — genererar testklipp ur en stillbild med riktiga människor (ultralytics-exempelbild): panorerande kamera + syntetisk eld/rök. Testtillgång, inte del av analysen.
- `scripts/integration_check.py` — kör mot live server via WebSocket och verifierar: ≥ 8 fps ut, personer i ≥ 50 % av paketen, begränsat antal person-ID (ingen ID-explosion), max boxhopp < 0,08 mellan konsekutiva paket, STILLA-flaggning under kamerapanorering (stabiliseringstest), eld/rök/basförslag, faropunkts-API.
- Resultat 2026-06-12: 24 fps ut / ~9 Hz analys på 4 CPU-kärnor, 5 unika personer stabila över filmloopar, max boxhopp 0,07.

## Logg

- 2026-06-12: Repo var en tom FastAPI-mall. Arkitektur enligt ovan vald; PoC 1 påbörjad. Beslut B1–B13 nedtecknade innan implementation.
- 2026-06-12: Boxutjämning omarbetad efter mätning (B3 uppdaterad): One Euro förkastad till förmån för flödesframmatning + EMA/slew. Scenklippshantering tillagd (B14). Integrationskontroll grön.
- 2026-06-12: Kodgranskningsrunda (7 vinklar). Åtgärdat: JPEG-kodning hoppas över när inga klienter tittar (analysen fortsätter ackumulera), dött One Euro-filter borttaget, `VideoSource.frame_no` publik. Noterat för PoC 2 (medvetet ej åtgärdat nu): hothållning är global (ej per hottyp/plats), diskontinuitetshantering bör generaliseras bortom filloop (RTSP-glapp), GUI-rendering finns i två varianter (webbklient + snapshot-debugverktyg) som måste hållas i synk.
