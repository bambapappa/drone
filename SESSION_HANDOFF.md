# Session handoff

## 2026-08-15 — Superpowers Task 3

- **Status och SHA-roller:** `DONE_WITH_CONCERNS` på branch `fix/smoke-scene-compensated-motion`. Kod under test är exakt `c82d07fba818e2e44e883540efcb9516361120b0`. Initial Task 3-resultatdokumentation är exakt `7c18c5a01124808877de1331b82edcecc94bfcb1`. Första auditability-uppföljningen är `ae745e206850922b408c1887756c4f0b79c52ec6`. Kvalitetsuppföljningen är exakt `09ea6bbb652bf1c1da1358635ae6887e2bd3f1c9`; den ersätter eventspannssampling med full 3 706/3 705-bilders replay, P5:s egen eventdiff och kryptografiskt låsta sidecars. Samtliga uppföljningar är endast dokumentation; ingen kod eller tröskel ändras. Denna handoff-only-commit anger föregångarens SHA i stället för att låtsas kunna självreferera sin egen.
- **Brandhalva, använd körning:** `drone-halva2-brand_analysis/eb365b175b07`. Manifest: P1–P5 `complete`, 3 706 bildrutor; P5 23 events = HAZARD 8 (3 smoke, 5 fire), MOT_FARA 11, IRRATIONELL 4. Rökspann: 13,20–19,36, 24,80–30,84 och 137,40–143,40 s. Alla fem fire är öppna falsklarm eftersom filmen har rök men inga lågor. Den ofullständiga brandkörningen `4cee3cfea615` ska fortsatt ignoreras.
- **Olyckshalva, ny körning:** `drone-halva1-olycka_analysis/fd10f349f3d7`, skapad utan `--resume` med VisDrone-s/MPS, 1280 och display-conf 0,20. Kommando exit 0; manifest P1–P5 `complete`, 3 705 bildrutor; P5 20 events = IRRATIONELL 12, STILLA 8, HAZARD 0, MOT_FARA 0. Negativkontrollen har alltså 0 smoke och 0 fire.
- **Slutsats:** A′+textur detekterar verklig rök i separat brandfilm och förblir tyst för rök i separat olycksfilm; smoke-position och drift är mätt icke-triviala. MOT_FARA-kedjan ger 11/0 som bonus. Precisionen är inte grön eftersom brandhalvan samtidigt ger fem fire-falsklarm.
- **Öppna concerns:** filmglobal PiP-mask övermaskerar sammanklippt film, därför är split-halvor det ärliga testet för en-film-per-körning. `OfflineConfig.to_dict()` saknar `ignore_regions` och övriga hazard-fält; manifestet kan inte bära full körproveniens. `smoke_drift` använder fortfarande owarpad föregående bildruta. Fire-falsklarmen är kvar.
- **Återstår — Task 4:** stryk den nu inaktuella anspråksraden i `HANDOFF.md` och gör Task 4:s slutverifiering. `HANDOFF.md` är avsiktligt orörd i Task 3.

## 2026-08-10

- Review phase fixed the person statistic so the Swedish header keeps `unique_count` as its headline while showing the API's projected `count` with P3's immutable uncertainty band.
- Focused API regression was attempted with `pytest tests/test_review_api_phase5.py -k persons -v`, but collection is blocked because the available Python environment lacks `cv2`.
- Documentation pass aligned the Swedish guide's person-list wording with the corrected projection and found no other stale owner documentation.
