# B32 split-acceptans — reproducerbart mätkvitto

Datum: 2026-08-15

P5 persisterar tid och area men inte hazardposition eller `smoke_drift`. Detta kvitto återspelar därför varje bildruta genom samma `SituationAnalyzer`, bygger fullständiga fire/smoke-tidslinjer och anropar P5:s egen rena `_diff_and_number_hazard_events`. De nyhärledda eventen måste vara exakt lika den hash-låsta P5-loggens HAZARD-event innan position eller drift redovisas.

Inga filmer eller sidecars ingår i Git. Sätt `B32_VIDEO_DIR` till katalogen som innehåller de två splitfilmerna.

## Låst proveniens

- Kod under test: `c82d07fba818e2e44e883540efcb9516361120b0`.
- Båda manifest: `code_version=source-sha256:12df3be4d7dcbf91bc38376fb0c5f1a4001d74ee44cca7bc72ef2719d266715f`, `config_hash=5a5a9c8cde6a14e7`.
- Brandfilm `drone-halva2-brand.mp4`: full SHA-256 `e107bf6338f06346c1154d833fc3439ba609941308e2585ae5e7712cea825212`, run `eb365b175b07`.
- Olycksfilm `drone-halva1-olycka.mp4`: full SHA-256 `e421020d46d014de1ba8814e5cc9630e4de2e02be7ff9998c0bf49f7a6bd9ebe`, run `fd10f349f3d7`.
- Varje använd `manifest.json`, `frames/p2_track.jsonl` och `events/p5_events.jsonl` verifieras mot SHA-256 i kommandot nedan.
- Den ofullständiga brandkörningen `4cee3cfea615` används inte.

## Exakt replaykommando

Kör från reporoten efter att `B32_VIDEO_DIR` har satts. `PYTHONDONTWRITEBYTECODE=1` gör replayen läsande. Kommandot stoppar om produktionsträdet `analysis/` avviker från kod-SHA, om en lokal produktionsfil är ocommittad, om video/sidecar/hash/proveniens/passstatus avviker, eller om nyhärledda HAZARD-event inte exakt matchar den persisterade P5-loggen. Tester ingår inte i replaymotorn och får därför formatteras utan att ogiltigförklara den låsta produktionskoden.

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - <<'PY'
import hashlib
import json
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np

from analysis.events import _diff_and_number_hazard_events
from analysis.orchestrator import OfflineConfig
from analysis.situation import SituationAnalyzer, WORK_W

CODE_SHA = "c82d07fba818e2e44e883540efcb9516361120b0"
CONFIG_HASH = "5a5a9c8cde6a14e7"
CODE_VERSION = (
    "source-sha256:"
    "12df3be4d7dcbf91bc38376fb0c5f1a4001d74ee44cca7bc72ef2719d266715f"
)
VIDEO_DIR = Path(os.environ["B32_VIDEO_DIR"])
CASES = [
    {
        "name": "brand",
        "video_name": "drone-halva2-brand.mp4",
        "video_sha256": (
            "e107bf6338f06346c1154d833fc3439ba609941308e2585ae5e7712cea825212"
        ),
        "manifest_video_hash": (
            "a04a5305f00c7e6ad364ebfec8691e8dffb209a6efe15d354b9b8d9f95de3e20"
        ),
        "run_id": "eb365b175b07",
        "run": Path("drone-halva2-brand_analysis/eb365b175b07"),
        "ignore_regions": [],
        "artifact_sha256": {
            "manifest.json": (
                "26efe26832256f015cad31dc30a0e753e67291442aa40ca54b023235b435952e"
            ),
            "frames/p2_track.jsonl": (
                "94317a17954be19786d2f02a814b533817e5c18dc0f0ea8f64b1a1f1739161a9"
            ),
            "events/p5_events.jsonl": (
                "6dfcbf14088a93c57a7b3e039cc915b5c51280c3a860c78ebe96ea3a190a9e50"
            ),
        },
    },
    {
        "name": "olycka",
        "video_name": "drone-halva1-olycka.mp4",
        "video_sha256": (
            "e421020d46d014de1ba8814e5cc9630e4de2e02be7ff9998c0bf49f7a6bd9ebe"
        ),
        "manifest_video_hash": (
            "7d7ff50516ad9d45cf142db7fb00845eabbc9de26374dae7bc52550a7484f8d9"
        ),
        "run_id": "fd10f349f3d7",
        "run": Path("drone-halva1-olycka_analysis/fd10f349f3d7"),
        "ignore_regions": [(0.64, 0.0, 0.36, 0.46)],
        "artifact_sha256": {
            "manifest.json": (
                "e56ae39c14cab11d7d6509f7a58b8fdb8c8f347187ca7a9dfe907c563413a69d"
            ),
            "frames/p2_track.jsonl": (
                "d234314b4ce8d8ea5560f42755059a5fd8485e4d0539face0f081e9b37a08d39"
            ),
            "events/p5_events.jsonl": (
                "284270666e3fe1395bfacdb070bddf65df6735121ee81641e4b83dbd1299c32d"
            ),
        },
    },
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


subprocess.run(
    ["git", "diff", "--quiet", CODE_SHA, "--", "analysis"],
    check=True,
)
status = subprocess.run(
    ["git", "status", "--porcelain", "--", "analysis"],
    check=True,
    capture_output=True,
    text=True,
).stdout
require(status == "", f"analysis/ has uncommitted changes: {status}")
print(json.dumps({"analysis_tree_matches": True, "code_sha": CODE_SHA}, sort_keys=True))

for case in CASES:
    video = VIDEO_DIR / case["video_name"]
    run = case["run"]
    require(sha256(video) == case["video_sha256"], f"video hash mismatch: {video.name}")
    for relative, expected in case["artifact_sha256"].items():
        require(sha256(run / relative) == expected, f"artifact hash mismatch: {relative}")

    manifest = json.loads((run / "manifest.json").read_text())
    require(manifest["run_id"] == case["run_id"], "run_id mismatch")
    require(manifest["config_hash"] == CONFIG_HASH, "config_hash mismatch")
    require(manifest["code_version"] == CODE_VERSION, "code_version mismatch")
    require(manifest["video_hash"] == case["manifest_video_hash"], "manifest video hash mismatch")
    require(
        all(item["status"] == "complete" for item in manifest["passes"].values()),
        "incomplete pass",
    )

    meta = manifest["passes"]["p1_detect"]["meta"]
    fps = float(meta["fps"])
    width = int(meta["width"])
    expected_frames = int(meta["total_frames"])
    scene_frames = {}
    for line in (run / "frames/p2_track.jsonl").read_text().splitlines():
        row = json.loads(line)
        scene_frames[int(row["frame_no"])] = row
    persisted = [
        json.loads(line)
        for line in (run / "events/p5_events.jsonl").read_text().splitlines()
    ]
    persisted_hazards = [
        event for event in persisted if event["category"] == "HAZARD"
    ]

    config = OfflineConfig()
    analyzer = SituationAnalyzer(
        min_area=config.hazard_min_area,
        hold_s=config.hazard_hold_s,
        flow_ema=config.smoke_flow_ema,
        base_margin=config.base_margin,
        base_hysteresis=config.base_hysteresis,
        fire_require_smoke=config.fire_require_smoke,
        smoke_window_s=config.hazard_smoke_window_s,
        smoke_texture_min=config.hazard_texture_min,
    )
    lag = max(2, round(config.hazard_smoke_window_s * fps))
    warp_scale = WORK_W / float(width)
    scene_history = {}
    timelines = {"fire": [], "smoke": []}
    active = {"fire": {}, "smoke": {}}
    capture = cv2.VideoCapture(str(video))
    frame_no = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        scene = scene_frames.get(frame_no)
        reference = scene_history.get(frame_no - lag)
        previous_to_current = None
        if (
            scene is not None
            and scene.get("frame_to_scene") is not None
            and scene.get("scene_to_frame") is not None
            and reference is not None
            and reference.get("frame_to_scene") is not None
            and int(scene.get("scene_segment", -1))
            == int(reference.get("scene_segment", -2))
        ):
            relative = np.asarray(
                scene["scene_to_frame"], dtype=np.float32
            ) @ np.asarray(reference["frame_to_scene"], dtype=np.float32)
            previous_to_current = np.asarray(relative, dtype=np.float32)[:2, :3]
            previous_to_current[:, 2] *= warp_scale
        state = analyzer.update(
            frame,
            frame_no / fps,
            danger_norm=None,
            ignore=case["ignore_regions"],
            prev_to_cur=previous_to_current,
            scene_motion=True,
            ref_lag=lag,
        )
        if scene is not None:
            scene_history[frame_no] = scene
            scene_history.pop(frame_no - lag, None)
        for kind in ("fire", "smoke"):
            hazard = getattr(state, kind)
            timelines[kind].append(
                (frame_no, hazard is not None, hazard.area if hazard else 0.0)
            )
            if hazard is not None:
                active[kind][frame_no] = (hazard.pos, state.smoke_drift)
        frame_no += 1
    capture.release()
    require(frame_no == expected_frames, "decoded frame count mismatch")

    derived = [
        event.to_dict()
        for event in _diff_and_number_hazard_events(
            timelines["fire"], timelines["smoke"], fps
        )
    ]
    require(derived == persisted_hazards, "derived hazards differ from persisted P5 hazards")
    if case["name"] == "olycka":
        require(frame_no == 3705, "accident control did not replay all 3705 frames")
        require(not any(row[1] for row in timelines["fire"]), "accident control has active fire")
        require(not any(row[1] for row in timelines["smoke"]), "accident control has active smoke")
        require(derived == [], "accident control derived hazards")

    print(
        json.dumps(
            {
                "case": case["name"],
                "video": case["video_name"],
                "video_sha256": case["video_sha256"],
                "run_id": manifest["run_id"],
                "manifest_video_hash": manifest["video_hash"],
                "config_hash": manifest["config_hash"],
                "code_version": manifest["code_version"],
                "artifact_sha256": case["artifact_sha256"],
                "pass_statuses": {
                    key: value["status"]
                    for key, value in manifest["passes"].items()
                },
                "frames_replayed": frame_no,
                "persisted_events": len(persisted),
                "derived_hazards": len(derived),
                "active_fire_frames": len(active["fire"]),
                "active_smoke_frames": len(active["smoke"]),
                "hazard_match_exact": True,
                "replay": {
                    "lag": lag,
                    "ignore_regions": case["ignore_regions"],
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    for event in derived:
        kind = event["evidence"]["kind"]
        first = event["evidence"]["frame_start"]
        last = event["evidence"]["frame_end"]
        rows = [active[kind][number] for number in range(first, last + 1)]
        positions = np.array([row[0] for row in rows])
        drifts = np.array([row[1] for row in rows])
        magnitudes = np.linalg.norm(drifts, axis=1)
        print(
            json.dumps(
                {
                    "case": case["name"],
                    "event_id": event["event_id"],
                    "kind": kind,
                    "frames": [first, last],
                    "t": [event["t_start"], event["t_end"]],
                    "area_mean": event["evidence"]["area_mean"],
                    "area_peak": event["evidence"]["area_peak"],
                    "pos_first": [round(value, 6) for value in positions[0]],
                    "pos_last": [round(value, 6) for value in positions[-1]],
                    "pos_min": [
                        round(value, 6) for value in positions.min(axis=0)
                    ],
                    "pos_max": [
                        round(value, 6) for value in positions.max(axis=0)
                    ],
                    "drift_first": [round(value, 8) for value in drifts[0]],
                    "drift_last": [round(value, 8) for value in drifts[-1]],
                    "drift_mean": [
                        round(value, 8) for value in drifts.mean(axis=0)
                    ],
                    "drift_mag_max": round(float(magnitudes.max()), 8),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
PY
```

## Maskinläsbart utfall

Detta är stdout från exakt kommandot ovan, omkört 2026-08-15 med `B32_VIDEO_DIR` satt till den lokala splitfilmskatalogen. Positioner är normaliserade bildkoordinater. `smoke_drift` följer motorns befintliga normaliserade Farneback-värde; den kända begränsningen att driftvägen använder owarpad föregående bildruta kvarstår.

```jsonl
{"analysis_tree_matches": true, "code_sha": "c82d07fba818e2e44e883540efcb9516361120b0"}
{"active_fire_frames":1723,"active_smoke_frames":455,"artifact_sha256":{"events/p5_events.jsonl":"6dfcbf14088a93c57a7b3e039cc915b5c51280c3a860c78ebe96ea3a190a9e50","frames/p2_track.jsonl":"94317a17954be19786d2f02a814b533817e5c18dc0f0ea8f64b1a1f1739161a9","manifest.json":"26efe26832256f015cad31dc30a0e753e67291442aa40ca54b023235b435952e"},"case":"brand","code_version":"source-sha256:12df3be4d7dcbf91bc38376fb0c5f1a4001d74ee44cca7bc72ef2719d266715f","config_hash":"5a5a9c8cde6a14e7","derived_hazards":8,"frames_replayed":3706,"hazard_match_exact":true,"manifest_video_hash":"a04a5305f00c7e6ad364ebfec8691e8dffb209a6efe15d354b9b8d9f95de3e20","pass_statuses":{"p1_detect":"complete","p2_track":"complete","p3_identity":"complete","p5_events":"complete"},"persisted_events":23,"replay":{"ignore_regions":[],"lag":8},"run_id":"eb365b175b07","video":"drone-halva2-brand.mp4","video_sha256":"e107bf6338f06346c1154d833fc3439ba609941308e2585ae5e7712cea825212"}
{"area_mean":0.01476,"area_peak":0.03062,"case":"brand","drift_first":[0.00038052,6.828e-05],"drift_last":[0.00047396,4.146e-05],"drift_mag_max":0.00115685,"drift_mean":[0.00042791,3.62e-06],"event_id":"hazard-000000","frames":[59,720],"kind":"fire","pos_first":[0.670069,0.74893],"pos_last":[0.693502,0.521869],"pos_max":[0.900273,0.925604],"pos_min":[0.180386,0.226426],"t":[2.36,28.84]}
{"area_mean":0.00748,"area_peak":0.01014,"case":"brand","drift_first":[0.00024865,8.971e-05],"drift_last":[0.00026959,-8.618e-05],"drift_mag_max":0.00057362,"drift_mean":[0.00020727,-8.295e-05],"event_id":"hazard-000001","frames":[330,483],"kind":"smoke","pos_first":[0.776746,0.120261],"pos_last":[0.779398,0.074897],"pos_max":[0.799726,0.925879],"pos_min":[0.14523,0.028603],"t":[13.2,19.36]}
{"area_mean":0.00812,"area_peak":0.01111,"case":"brand","drift_first":[0.00061125,0.00011384],"drift_last":[0.00016006,-5.011e-05],"drift_mag_max":0.00115685,"drift_mean":[0.00057785,1.307e-05],"event_id":"hazard-000002","frames":[620,770],"kind":"smoke","pos_first":[0.90056,0.117247],"pos_last":[0.94947,0.05],"pos_max":[0.96855,0.125898],"pos_min":[0.768125,0.043981],"t":[24.8,30.84]}
{"area_mean":0.00684,"area_peak":0.01347,"case":"brand","drift_first":[0.0001745,-9.544e-05],"drift_last":[0.00015944,0.00010926],"drift_mag_max":0.0005484,"drift_mean":[0.00027156,6.332e-05],"event_id":"hazard-000003","frames":[779,944],"kind":"fire","pos_first":[0.647356,0.532585],"pos_last":[0.601396,0.548227],"pos_max":[0.648454,0.549889],"pos_min":[0.601277,0.463001],"t":[31.16,37.8]}
{"area_mean":0.00622,"area_peak":0.01333,"case":"brand","drift_first":[-0.00063211,-0.00012207],"drift_last":[-0.00035727,-0.00018134],"drift_mag_max":0.00064379,"drift_mean":[-0.00021124,-0.0001374],"event_id":"hazard-000004","frames":[1729,2106],"kind":"fire","pos_first":[0.460606,0.144613],"pos_last":[0.86088,0.556516],"pos_max":[0.943437,0.6598],"pos_min":[0.368097,0.056187],"t":[69.16,84.28]}
{"area_mean":0.0071,"area_peak":0.00972,"case":"brand","drift_first":[0.00010973,-6.136e-05],"drift_last":[2.429e-05,-2.405e-05],"drift_mag_max":0.0008353,"drift_mean":[0.00019164,-5.061e-05],"event_id":"hazard-000005","frames":[2158,2341],"kind":"fire","pos_first":[0.718239,0.080808],"pos_last":[0.685644,0.621782],"pos_max":[0.794342,0.659454],"pos_min":[0.521968,0.080808],"t":[86.32,93.68]}
{"area_mean":0.00777,"area_peak":0.0166,"case":"brand","drift_first":[0.00033524,0.00010871],"drift_last":[-0.00025628,-0.00025311],"drift_mag_max":0.00068397,"drift_mean":[0.00010751,1.772e-05],"event_id":"hazard-000006","frames":[2786,3118],"kind":"fire","pos_first":[0.773235,0.377099],"pos_last":[0.514089,0.477778],"pos_max":[0.784542,0.487125],"pos_min":[0.199432,0.377099],"t":[111.44,124.76]}
{"area_mean":0.00714,"area_peak":0.01278,"case":"brand","drift_first":[0.00046789,-0.00029143],"drift_last":[0.00035352,-0.0002226],"drift_mag_max":0.00072742,"drift_mean":[0.0003157,-0.0001801],"event_id":"hazard-000007","frames":[3435,3584],"kind":"smoke","pos_first":[0.411899,0.099644],"pos_last":[0.488982,0.52646],"pos_max":[0.966887,0.89976],"pos_min":[0.083048,0.016517],"t":[137.4,143.4]}
{"active_fire_frames":0,"active_smoke_frames":0,"artifact_sha256":{"events/p5_events.jsonl":"284270666e3fe1395bfacdb070bddf65df6735121ee81641e4b83dbd1299c32d","frames/p2_track.jsonl":"d234314b4ce8d8ea5560f42755059a5fd8485e4d0539face0f081e9b37a08d39","manifest.json":"e56ae39c14cab11d7d6509f7a58b8fdb8c8f347187ca7a9dfe907c563413a69d"},"case":"olycka","code_version":"source-sha256:12df3be4d7dcbf91bc38376fb0c5f1a4001d74ee44cca7bc72ef2719d266715f","config_hash":"5a5a9c8cde6a14e7","derived_hazards":0,"frames_replayed":3705,"hazard_match_exact":true,"manifest_video_hash":"7d7ff50516ad9d45cf142db7fb00845eabbc9de26374dae7bc52550a7484f8d9","pass_statuses":{"p1_detect":"complete","p2_track":"complete","p3_identity":"complete","p5_events":"complete"},"persisted_events":20,"replay":{"ignore_regions":[[0.64,0.0,0.36,0.46]],"lag":8},"run_id":"fd10f349f3d7","video":"drone-halva1-olycka.mp4","video_sha256":"e421020d46d014de1ba8814e5cc9630e4de2e02be7ff9998c0bf49f7a6bd9ebe"}
```

Fullt utfall: 3 706 brandbilder gav exakt samma åtta HAZARD som P5 (1 723 aktiva fire-bilder, 455 aktiva smoke-bilder). Alla 3 705 olycksbilder kördes; både fire- och smoke-tidslinjen var falsk på varje bildruta, den rena P5-diffhjälparen härledde noll HAZARD och detta matchade eventloggen exakt. Resultatet ändrar inte `DONE_WITH_CONCERNS`: fem `kind=fire` i brandhalvan är fortsatt falsklarm, filmglobal PiP-övermaskning och `OfflineConfig.to_dict()`-luckan kvarstår.
