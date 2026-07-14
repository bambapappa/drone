# Project agent memory

## Architecture

Two independent code paths in this repo:

- **`app/`** — the realtime PoC (FastAPI + WS + live web GUI). Runs via `docker compose up`.
- **`analysis/`** — the new offline training/evaluation tool (batch, sequential,
  multi-pass). Carved from `app/vision/` analyzers; shares no runtime with the
  realtime path. Runs via `docker compose -f docker-compose.yml -f docker-compose.offline.yml run --rm analyze /videos/film.mp4`.

Key architecture doc: the report at the path in this task's brief (design of record
for the offline tool). Code-level citations are in the companion scout report.

### Offline tool: module map

| Module | Role |
|--------|------|
| `analysis/ingest.py` | PTS index, video hash, PiP detection, `FrameStore` |
| `analysis/store.py` | Versioned JSONL sidecar + manifest.json |
| `analysis/orchestrator.py` | Sequential multi-pass driver (P1→P5) |
| `analysis/cli.py` | `analyze <video>` entry point |
| `analysis/tiling.py` | N×N tile grid + NMS (carried forward, unchanged) |
| `analysis/registry.py` | `PersonRegistry` (carried forward, unchanged) — live re-ID reference; P3 reimplements its gate math offline |
| `analysis/behavior.py` | `BehaviorAnalyzer` (carried forward, unchanged) |
| `analysis/situation.py` | `SituationAnalyzer` (carried forward, unchanged) |
| `analysis/flow.py` | `GlobalMotion`, `BoxFilter`, `local_box_flow` |
| `analysis/pip.py` | `PipAutoDetector` (carried forward) |
| `analysis/detector.py` | P1: stateless YOLO detection only, no tracker |
| `analysis/tracker.py` | P2: BoT-SORT + GMC, driven from P1's persisted detections |
| `analysis/embedding.py` | P1 appearance embedder: OSNet primary + HSV fallback below the ReID floor |
| `analysis/identity.py` | P3: global tracklet association into persons (constrained agglomerative clustering + `assoc_audit`) |

The carved-out analyzer modules in `analysis/` are independent copies of their
`app/vision/` originals. The realtime `app/vision/` modules are left untouched.
Both code paths evolve independently — this is an addition, not a replacement.

### Timebase

The offline tool uses **video time**: `t = frame_no / fps` (PTS-corrected).
The realtime PoC uses `time.monotonic()`. The analyzers themselves accept `t`
as a float parameter and are unchanged — only the caller swaps the timebase.
BoT-SORT's `track_buffer` is re-expressed as `fps × seconds` (not a fixed 120).

### Artifact schema

Sidecar store at `<output>/<run_id>/`:
- `manifest.json` — video hash, config hash, model, seed, pass log
- `frames/<pass>.jsonl` — per-frame metadata (P1)
- `detections/<pass>.jsonl` — P1's raw per-detection output (never tracker-adjusted), with raw appearance embedding + `embedding_method` ("osnet"|"hsv")
- `tracklets/<pass>.jsonl` — P2's per-(track_id, frame) tracker/Kalman-adjusted boxes, referencing back to `det_id`
- `persons/<pass>.jsonl` — P3's per-identity records: `person_id, tracklet_ids, embedding_centroids, first/last_seen, confirmation_state, assoc_audit`
- `checkpoints/<pass>/` — P1 resumable state only; P2/P3 always re-run in full (cheap, deterministic given P1's output)

### Identity design (Phase 1, P3)

`person_id` is the only stable public identity. `tracklet_id` (and `det_id`)
are P1/P2-internal lineage — a track split at a resume boundary or an
occlusion is indistinguishable, and P3 re-associates across both.

P3 = global tracklet association (not the live registry's online-greedy
match). Three gates, strongest first, in `analysis/identity.py`:

1. **Hard temporal-overlap exclusion** — two tracklets sharing even one frame
   are never the same person. Impossible live (you never have the full frame
   set), trivial offline. The biggest correctness lever the offline tool has
   over the live one for identity.
2. **Spatio-temporal plausibility** — generalizes `registry.py:_match_lost`'s
   `max_dist_frac × diag × (1+gap_s)` gate. Positions are raw frame pixels
   (no GMC stabilization persisted in Phase 0), so this is a plausibility
   filter, not a motion model; appearance + temporal exclusion do the real
   identity work.
3. **Appearance similarity** — best same-method cosine of embedding centroids
   (osnet and hsv vectors are different-dimensional and never compared
   across methods).

**Unique count is honest, never one precise number**: confirmed persons + an
uncertainty band from near-merges ("N unika, varav M osäkra sammanslagningar").
Same-clothing confusion is a fundamental appearance-method limit (DECISIONS
B4); the tool surfaces it via `assoc_audit`, never hides it.

**Determinism**: P3 is pure-numpy constrained agglomerative clustering (no
scipy/sklearn — tighter determinism), no RNG, fixed evaluation order →
byte-identical persons/assoc_audit across re-runs. Mirrors P1/P2's guarantee.

### ReID embeddings

P1 computes one appearance vector per detection crop. OSNet (TorchScript,
`--reid-weights PATH`) when weights are present; HSV-only (the carried-forward
`appearance_hist`) otherwise. The HSV fallback is **deliberate** below a crop
floor (`--reid-floor`, default 32 px): 10 px people at altitude are below any
ReID model's input. `embedding_method` tags each detection so a consumer
knows which space its vector is in. Weights follow the YOLO/VisDrone
provenance pattern (path on disk, recorded in the manifest, never committed
or auto-fetched with a guessed URL).

## Captain's product decisions (durable)

- The offline tool's review UI (Phase 2+) must be **Swedish-only** — no English
  strings in any user-facing GUI text. The existing realtime GUI is already Swedish.
- The offline tool must be runnable via `docker compose` end to end, like the
  realtime PoC is today (`docker compose -f docker-compose.yml -f docker-compose.offline.yml run analyze <video>`).
- This is an addition alongside the realtime PoC, not a replacement or a
  dual-mode rewrite of it.

## Build / test / lint

```bash
make venv && source .venv/bin/activate    # set up dev environment
make test                                  # pytest tests/ -v
make lint                                  # ruff check + format check
```

CI runs `ruff check` + `ruff format --check` on `app/`, `tests/`, `scripts/`, `analysis/`,
then `pytest tests/ -v`.

## Docker

```bash
# Realtime PoC
docker compose up --build

# Offline analysis
docker compose -f docker-compose.yml -f docker-compose.offline.yml run --rm analyze /videos/film.mp4
docker compose -f docker-compose.yml -f docker-compose.offline.yml run --rm analyze /videos/film.mp4 --tiles 2 --imgsz 1280
```

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
