# Demo-körschema (MacBook Pro M5)

Mål för demon: **visa persondetektering på riktig drönarfilm i webbläsaren** —
boxar som följer människor mjukt, total räkning av unika personer, och (bonus)
liggande/stilla skadade markerade. Allt annat (lägesbild, basförslag,
IR-maskning) är grädde på moset.

> Snabbaste vägen: `make serve` och öppna `http://localhost:8000`.
> Allt nedan är detaljerna runt det.

---

## 1. Engångsuppsättning

```bash
git clone https://github.com/bambapappa/drone && cd drone

# Isolerad virtuell miljö (krävs på macOS — systemets Python är skyddad).
python3 -m venv .venv
source .venv/bin/activate                   # gör i varje nytt terminalfönster
pip install --upgrade pip
pip install -e ".[dev]"                      # app + beroenden (torch m. MPS) + testverktyg
#   OBS: citattecknen runt ".[dev]" krävs i zsh (Macs standardskal)
#   Genväg för raderna ovan:  make venv

# Hämta VisDrone-vikterna (tränade för små människor från drönarhöjd)
python scripts/fetch_visdrone.py            # -> models/visdrone-yolov8s.pt
```

> **Inget `--system`.** Den flaggan är bara till för Docker-imagen/CI. På din
> Mac installerar du i venv:en ovan. `make serve` hittar `.venv` automatiskt,
> så du behöver inte aktivera den för att starta servern.

Lägg dina filmer i `videos/`. Saknas film helt finns ett syntetiskt testklipp:
`python scripts/make_demo_video.py` skriver `videos/demo.mp4`.

## 2. Konfiguration för demon (`.env`)

Skapa `.env` i repo-roten:

```ini
MODEL=models/visdrone-yolov8s.pt   # VisDrone slår COCO stort på drönarfilm
DEVICE=mps                         # Apple-GPU på M5 (cpu funkar men långsammare)
IMGSZ=1280                         # hög upplösning -> hittar små människor
CONF=0.20                          # recall före precision (hellre en för mycket)
TILES=1                            # börja här; höj till 2 om du tappar små mål
MAX_FPS=24
```

Tumregel för **personer** (demons prio):
- Hittar den för få? Höj `IMGSZ` (1280→1536) eller sätt `TILES=2`.
- Hackar/laggar det? Sänk `IMGSZ` (1280→960) eller `TILES=1`. Boxföljsamheten
  påverkas inte — den går alltid i full bildtakt via optiskt flöde.

## 3. Starta

```bash
make serve            # startar robust, väntar på /health, skriver ut URL
# eller direkt:  bash scripts/serve.sh videos/din-film.mp4
```

Öppna **http://localhost:8000**. Funkar på laptop, platta och mobil samtidigt.

Docker-alternativ (CPU, ingen MPS): `docker compose up --build`.

## 4. Vad du visar i GUI:t

| Element | Vad det visar |
|---|---|
| **Statusrad överst** | `synliga` (nu), `unika` (totalt under passet), `irrationella`, fps |
| **Gröna boxar** | detekterade människor, följer mjukt med ID `P1, P2…` |
| **Röda boxar (STILLA)** | någon ligger/står stilla länge — t.ex. skadad |
| **Orange (MOT FARA)** | rör sig mot markerad fara |
| **Lager-chips nederst** | toggla Boxar / ID / Spår / Beteende / Miljö / Bas |
| **🎯 Markera fara** | tryck knappen, tryck i bilden → sätter faropunkt |
| **Panel-knapp** | lägesbild + basmotivering + källbyte + filuppladdning |

Demoflöde-förslag:
1. Starta med ambulans-/masskadefilmen → peka på att räddare räknas (gröna)
   och **liggande skadade flaggas röda (STILLA)**.
2. Visa **unika-räknaren** stiga medan folk rör sig in/ut — ID:n återanvänds.
3. Toggla **Spår** för att visa rörelsemönster, **Bas** för basförslaget.
4. Byt film i panelen → poängen att utfallet är förutsägbart på osedd film.

## 5. Per scenario

| Film/typ | Tips |
|---|---|
| Masskada / ambulans (Full HD) | Bäst för personer; `IMGSZ=1280`, ev. `TILES=2` |
| Lägenhetsbrand | Tät folksamling; IR-rutan maskas automatiskt |
| Trafik m. IR-bild-i-bild | IR-rutan i hörnet detekteras & exkluderas automatiskt |
| Skog/nadir (svår) | Höj `IMGSZ`/`TILES`; glesa, små mål — räkna med missar |

IR-bild-i-bild (hörn eller 50% split) känns igen **automatiskt** och
exkluderas så att folk inte dubbelräknas — du behöver inte ställa något.
Vill du tvinga: `IGNORE_REGIONS=0.66,0,0.34,0.44` (hörn) eller
`ANALYSIS_ROI=0,0,0.5,1` (vänster halva).

## 6. Felsökning

| Symptom | Åtgärd |
|---|---|
| Servern svarar inte | `make serve` startar om rent (dödar gammal instans, väntar på health) |
| Sidan visar "Väntar på videoström" | kolla `videos/` har en film; se `/tmp/drone-serve.log` |
| För få personer hittas | höj `IMGSZ`, sätt `TILES=2`, sänk `CONF` till 0.15 |
| Lågt fps / hackigt | sänk `IMGSZ`/`TILES`; på Docker saknas MPS (kör native för fart) |
| Vill se vad analysen ser utan webbläsare | `python scripts/snapshot.py --out snap.jpg` |

Modellval, trösklar och alla env-variabler: se **[CONFIG.md](CONFIG.md)**.
Hur det fungerar inuti: **[ARCHITECTURE.md](ARCHITECTURE.md)**.
