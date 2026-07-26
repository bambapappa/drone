"""Multi-config comparison: diff the event logs of two runs on the same video.

Phase 5 (report §5.7): run the engine twice with different configs (heavier
model, more tiles, different thresholds) on the same film and diff the
results — this directly answers "does the heavier config find more?" and
doubles as the project's own model-evaluation harness (superseding
`scripts/eval_detection.py`'s sample-frame methodology with full-film event
logs).

**This is a batch capability, not a live UI action.** Producing the second
run means re-running P1's inference — hours on CPU — so the tool never
launches an analysis from the review UI (that would put the heavy engine
behind a button that looks instant; the exact scope-discipline smell report
§6 warns about). The workflow is: run `analyze` twice yourself (CLI/compose,
different flags), then diff the two finished sidecars — here, via the REST
endpoint the UI's "Körningar" tab calls, or headless via
`python -m review.run_compare <run_dir_a> <run_dir_b>`.

**The diff reads frozen engine output only** (`events/<pass>.jsonl` as
written, no verdict overlay, no hazard-marker recompute, no identity
corrections): the question is what the *configs* found, so per-run human
review layers must not leak into the comparison. Same-category events are
matched by time proximity (the shared deterministic greedy matcher);
cross-category matches are meaningless ("config A's STILLA is config B's
HAZARD" is not a correspondence). Default tolerance is 10 s — much tighter
than the operator comparison's 60 s, because both sides are machine-derived
from the same footage and a real correspondence should align closely; it's
a parameter, not a protocol constant.

**Same video is enforced by hash**, not filename: diffing runs of different
footage produces a number-shaped answer with no meaning, so mismatched
`video_hash` is a refusal, not a warning.

Deterministic: pure function of the two sidecars + tolerance. No RNG, no
wall-clock reads.
"""

from __future__ import annotations

import sys
from typing import Any

from analysis.orchestrator import OfflineOrchestrator
from analysis.store import ArtifactStore
from review.comparison import greedy_time_match

DEFAULT_RUN_TOLERANCE_S = 10.0


class RunCompareError(ValueError):
    """Raised when two runs cannot be meaningfully compared. The message is
    Swedish — it is surfaced verbatim in the UI and CLI."""


def _recorded_config(store: ArtifactStore) -> dict[str, Any]:
    """The run's recorded OfflineConfig dict, from the earliest pass that
    recorded one (P1 normally; fall back through later passes for partial
    sidecars). Every pass records the same config.to_dict(), so which pass
    supplies it doesn't matter."""
    passes = store.manifest.get("passes", {})
    for name in (
        OfflineOrchestrator.P1_PASS_NAME,
        OfflineOrchestrator.P2_PASS_NAME,
        OfflineOrchestrator.P3_PASS_NAME,
        OfflineOrchestrator.P5_PASS_NAME,
    ):
        cfg = passes.get(name, {}).get("meta", {}).get("config")
        if cfg:
            return cfg
    return {}


def _config_diff(cfg_a: dict[str, Any], cfg_b: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Fields whose recorded values differ between the two runs — the 'what
    was actually changed' half of the answer, straight from the manifests'
    provenance record rather than from what the user remembers passing."""
    diff: dict[str, dict[str, Any]] = {}
    for key in sorted(set(cfg_a) | set(cfg_b)):
        va, vb = cfg_a.get(key), cfg_b.get(key)
        if va != vb:
            diff[key] = {"a": va, "b": vb}
    return diff


def _pass_stats(store: ArtifactStore) -> dict[str, Any]:
    passes = store.manifest.get("passes", {})

    def stats(name: str) -> dict[str, Any]:
        return passes.get(name, {}).get("stats", {}) or {}

    return {
        "detections": stats(OfflineOrchestrator.P1_PASS_NAME).get("total_detections"),
        "tracklet_rows": stats(OfflineOrchestrator.P2_PASS_NAME).get("total_tracklet_rows"),
        "persons": stats(OfflineOrchestrator.P3_PASS_NAME).get("persons_out"),
        "events": stats(OfflineOrchestrator.P5_PASS_NAME).get("events_out"),
        "events_by_category": stats(OfflineOrchestrator.P5_PASS_NAME).get("by_category", {}),
    }


def compare_runs(
    store_a: ArtifactStore,
    store_b: ArtifactStore,
    tolerance_s: float = DEFAULT_RUN_TOLERANCE_S,
) -> dict[str, Any]:
    """Diff two completed runs of the same video. Returns a JSON-able dict:
    config diff, per-run pass stats, and per-category event buckets
    (both / only_a / only_b, with delta_s = B's onset - A's onset on
    matched pairs). Raises RunCompareError when the runs aren't comparable
    (different video, or P5 missing on either side)."""
    if store_a.manifest.get("video_hash") != store_b.manifest.get("video_hash"):
        raise RunCompareError(
            "körningarna avser olika videofiler (video_hash skiljer) — jämförelsen kräver samma film"
        )
    p5 = OfflineOrchestrator.P5_PASS_NAME
    for label, store in (("a", store_a), ("b", store_b)):
        if store.manifest.get("passes", {}).get(p5, {}).get("status") != "complete":
            raise RunCompareError(f"körning {store.run_id} ({label}) saknar färdig händelsehärledning (P5)")

    events_a = list(store_a.iter_events(p5))
    events_b = list(store_b.iter_events(p5))
    categories = sorted({e["category"] for e in events_a} | {e["category"] for e in events_b})

    both: list[dict[str, Any]] = []
    only_a: list[dict[str, Any]] = []
    only_b: list[dict[str, Any]] = []
    for cat in categories:
        cat_a = [e for e in events_a if e["category"] == cat]
        cat_b = [e for e in events_b if e["category"] == cat]
        triples = greedy_time_match(
            cat_a,
            cat_b,
            t_a="t_start",
            t_b="t_start",
            id_a="event_id",
            id_b="event_id",
            tolerance_s=tolerance_s,
        )
        matched_a = {a["event_id"] for a, _, _ in triples}
        matched_b = {b["event_id"] for _, b, _ in triples}
        both.extend({"a": a, "b": b, "delta_s": round(d, 3)} for a, b, d in triples)
        only_a.extend(e for e in cat_a if e["event_id"] not in matched_a)
        only_b.extend(e for e in cat_b if e["event_id"] not in matched_b)

    both.sort(key=lambda m: (m["a"]["t_start"], m["a"]["event_id"]))
    only_a.sort(key=lambda e: (e["t_start"], e["event_id"]))
    only_b.sort(key=lambda e: (e["t_start"], e["event_id"]))

    cfg_a, cfg_b = _recorded_config(store_a), _recorded_config(store_b)
    return {
        "run_a": {"run_id": store_a.run_id, "created_at": store_a.manifest.get("created_at")},
        "run_b": {"run_id": store_b.run_id, "created_at": store_b.manifest.get("created_at")},
        "video_hash": store_a.manifest.get("video_hash"),
        "tolerance_s": tolerance_s,
        "config_diff": _config_diff(cfg_a, cfg_b),
        "stats": {"a": _pass_stats(store_a), "b": _pass_stats(store_b)},
        "counts": {"both": len(both), "only_a": len(only_a), "only_b": len(only_b)},
        "both": both,
        "only_a": only_a,
        "only_b": only_b,
    }


def main(argv: list[str] | None = None) -> int:
    """Headless CLI: `python -m review.run_compare <run_dir_a> <run_dir_b>`.

    Respects the same FEATURE_RUN_COMPARE toggle as the REST surface so the
    feature has exactly one on/off switch across every entry point."""
    import argparse
    import json

    from review.config import ReviewSettings

    ap = argparse.ArgumentParser(description="Jämför två analyskörningar av samma film.")
    ap.add_argument("run_dir_a", help="Sökväg till första körningens sidecar-katalog")
    ap.add_argument("run_dir_b", help="Sökväg till andra körningens sidecar-katalog")
    ap.add_argument("--tolerance-s", type=float, default=DEFAULT_RUN_TOLERANCE_S)
    ap.add_argument(
        "--json", action="store_true", help="Skriv hela diffen som JSON i stället för sammanfattning"
    )
    args = ap.parse_args(argv)

    if not ReviewSettings.from_env().feature_run_compare:
        print("Funktionen är avstängd (FEATURE_RUN_COMPARE=0).", file=sys.stderr)
        return 2

    try:
        store_a = ArtifactStore.open_readonly(args.run_dir_a)
        store_b = ArtifactStore.open_readonly(args.run_dir_b)
        result = compare_runs(store_a, store_b, tolerance_s=args.tolerance_s)
    except (FileNotFoundError, RunCompareError) as exc:
        print(f"Fel: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    print(f"Jämförelse: {result['run_a']['run_id']} (A) mot {result['run_b']['run_id']} (B)")
    if result["config_diff"]:
        print("Konfigurationsskillnader:")
        for key, v in result["config_diff"].items():
            print(f"  {key}: A={v['a']!r}  B={v['b']!r}")
    else:
        print("Konfigurationsskillnader: inga (identisk konfiguration)")
    sa, sb = result["stats"]["a"], result["stats"]["b"]
    print(f"Detektioner:    A={sa['detections']}  B={sb['detections']}")
    print(f"Tracklet-rader: A={sa['tracklet_rows']}  B={sb['tracklet_rows']}")
    print(f"Personer:       A={sa['persons']}  B={sb['persons']}")
    print(f"Händelser:      A={sa['events']}  B={sb['events']}")
    c = result["counts"]
    print(
        f"Händelsediff (±{result['tolerance_s']:g}s): {c['both']} gemensamma, "
        f"{c['only_a']} endast A, {c['only_b']} endast B"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
