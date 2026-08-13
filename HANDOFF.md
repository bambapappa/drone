# Session handoff — scenkoordinatlager

## Mål

Bygg ett gemensamt lokalt scenkoordinatlager så att faromarkör, `MOT FARA`,
värmekarta och rörelseanalys följer den fysiska scenen när drönaren rör sig.
Ingen GPS/georeferens fabriceras; visuellt tapp ger ett nytt segment och synlig
osäkerhet.

## Aktuell status

- Grenen `codex/scene-coordinate-layer` är mergad via PR #8 (CI grön).
- Arkitektur accepterad och dokumenterad som `DECISIONS.md` B29.
- Implementation, svensk UI/handledning och dokumentation är klara.
- Full lokal verifiering: 450 tester, ruff, formatkontroll och JS-syntax gröna.
- Återstår: verkligt acceptanstest på filmen som utlöste arbetet efter
  distribution.

## Beslut och kontrakt

- P2:s GMC är en gemensam källa, inte en separat review-renderarberäkning.
- Per bildruta: `scene_to_frame`, `frame_to_scene`, `scene_segment`, kvalitet.
- Per tracklet-rad: scenfotpunkt och scenkroppshöjd, rå `xyxy` bevaras.
- Segment korsas aldrig genom gissning; UI visar utanför bild respektive okänd.
- Gamla sidecars utan scenfält fortsätter fungera i råpixel-läge.

## Nästa steg

1. Commit och kör no-mistakes till CI-grön PR.
2. Kör om arbetsfilmen; kontrollera markör/värmekarta under cirkelflygning,
   utanför bild och efter visuellt tapp.
3. Mät ackumulerad drift/parallax. Om 2D-affin GMC inte räcker är nästa steg
   loop-closure/SLAM eller fusion med GPU-endpoint + drönartelemetri, inte
   hårdare gissning i detta lager.
4. På Mac med Colima: kör `bash scripts/start_offline_visdrone.sh
   videos/film.mp4` för den reproducerbara offline-starten med VisDrone-s;
   `--fresh` är det uttryckliga valet för en ny analys.

## Bevara

- Append-only-regeln för mänskliga annoteringar.
- Svenska-only GUI.
- `.serena/` eller andra orelaterade lokala filer ska aldrig stageas.
