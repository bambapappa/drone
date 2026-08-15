# Session handoff

## 2026-08-15 — Superpowers Task 4, verifiering före PR

- **Status:** Implementationsdelen är klar på `fix/smoke-scene-compensated-motion`; PR är medvetet inte öppnad ännu, eftersom separat spec- och kvalitetsgranskning återstår. Anspråksraden för arbetet är borttagen ur `HANDOFF.md` enligt planen.
- **Grindar:** `source .venv/bin/activate && make test` gav **483 passed**. `make lint` gav `ruff check` grön och `ruff format --check` grön efter att formatteraren mekaniskt rättade två testfiler från Task 1/2 (`tests/test_analysis_situation.py`, `tests/test_events.py`). Den första testkörningen utan aktiverad venv stoppade under collection eftersom system-Python saknade `cv2`; omkörningen i planens projekt-venv är den giltiga grinden.
- **Scene-motion-kontroll:** produktionssökning efter `scene_motion=True` ger exakt en träff, i `analysis/events.py` inne i `derive_events`. Ytterligare träffar finns bara i de avsiktliga enhetstesterna.
- **Replay efter Task 4-formattering:** whole-branch-granskningen fångade att replaykvittots gamla `git diff` även låste `tests/`, vilket gjorde det okörbart efter den mekaniska formatteringen trots oförändrad produktionsmotor. Kvittot låser nu endast `analysis/` mot `c82d07f`, kontrollerar ocommittade produktionsfiler och använder alltid aktiva `RuntimeError`-kontroller i stället för optimeringskänsliga `assert`. Det exakta Markdown-kommandot kördes från Task 4-HEAD mot båda splitfilmerna: 3 706 + 3 705 bildrutor, exakt P5-hazardmatch och dokumenterad stdout byte-lika (`replay_stdout_exact=true`, 11 rader). Artefakt- och resultathashar är oförändrade.
- **Publiceringsgräns:** grenen är fortsatt stackad på PR #10 och måste ange att #10 mergas först. Ingen merge får ske automatiskt. Task 3:s `DONE_WITH_CONCERNS` står kvar: rök 3/0, men fem öppna fire-falsklarm i brandhalvan samt kända PiP-/proveniensluckor.

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
