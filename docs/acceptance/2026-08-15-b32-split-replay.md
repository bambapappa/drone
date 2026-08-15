# B32 split-acceptans — reproducerbart mätkvitto

Datum: 2026-08-15

Detta kvitto gör positions- och driftvärdena i `DECISIONS.md` reproducerbara utan att committa källfilmer eller sidecars. P5:s eventformat persisterar tid och area men inte hazardposition eller `smoke_drift`; därför återspelas exakt samma `SituationAnalyzer` mot källfilmen och de persisterade P2-scentransformerna.

## Låst proveniens

- Kod under test: `c82d07fba818e2e44e883540efcb9516361120b0`.
- `git diff --quiet c82d07f HEAD -- analysis tests` gav exit 0 före replay: dokumentationscommitarna efter `c82d07f` ändrade inte analyskod eller tester.
- Båda manifest: `code_version=source-sha256:12df3be4d7dcbf91bc38376fb0c5f1a4001d74ee44cca7bc72ef2719d266715f`, `config_hash=5a5a9c8cde6a14e7`.
- Brand: `/Users/bambapappa/.zclaude/jobs/110f4c45/tmp/drone-halva2-brand.mp4`, full SHA-256 `e107bf6338f06346c1154d833fc3439ba609941308e2585ae5e7712cea825212`, run `eb365b175b07`.
- Olycka: `/Users/bambapappa/.zclaude/jobs/110f4c45/tmp/drone-halva1-olycka.mp4`, full SHA-256 `e421020d46d014de1ba8814e5cc9630e4de2e02be7ff9998c0bf49f7a6bd9ebe`, run `fd10f349f3d7`.
- Den ofullständiga brandkörningen `4cee3cfea615` används inte.

Fulla filmhashar verifierades separat med:

```sh
shasum -a 256 \
  /Users/bambapappa/.zclaude/jobs/110f4c45/tmp/drone-halva1-olycka.mp4 \
  /Users/bambapappa/.zclaude/jobs/110f4c45/tmp/drone-halva2-brand.mp4
```

## Exakt replaykommando

Kör från reporoten med sidecars kvar på sina dokumenterade relativa sökvägar. Skriptet stoppar på hashavvikelse, icke-komplett manifest eller om en persisterad HAZARD saknar motsvarande analyzer-state på någon bildruta.

```sh
.venv/bin/python - <<'PY'
import hashlib, json
from pathlib import Path
import cv2
import numpy as np
from analysis.orchestrator import OfflineConfig
from analysis.situation import SituationAnalyzer, WORK_W

CASES = [
    {"name":"brand", "video":Path("/Users/bambapappa/.zclaude/jobs/110f4c45/tmp/drone-halva2-brand.mp4"), "run":Path("drone-halva2-brand_analysis/eb365b175b07"), "sha256":"e107bf6338f06346c1154d833fc3439ba609941308e2585ae5e7712cea825212", "ignore":[]},
    {"name":"olycka", "video":Path("/Users/bambapappa/.zclaude/jobs/110f4c45/tmp/drone-halva1-olycka.mp4"), "run":Path("drone-halva1-olycka_analysis/fd10f349f3d7"), "sha256":"e421020d46d014de1ba8814e5cc9630e4de2e02be7ff9998c0bf49f7a6bd9ebe", "ignore":[(0.64,0.0,0.36,0.46)]},
]
def file_sha256(path):
    digest=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()
for case in CASES:
    video,run=case["video"],case["run"]
    actual_sha=file_sha256(video); assert actual_sha==case["sha256"]
    manifest=json.loads((run/"manifest.json").read_text())
    assert all(p["status"]=="complete" for p in manifest["passes"].values())
    meta=manifest["passes"]["p1_detect"]["meta"]
    fps,w,h=float(meta["fps"]),int(meta["width"]),int(meta["height"])
    scene={}
    for line in (run/"frames/p2_track.jsonl").read_text().splitlines():
        row=json.loads(line); scene[int(row["frame_no"])]=row
    events=[json.loads(line) for line in (run/"events/p5_events.jsonl").read_text().splitlines()]
    hazards=[event for event in events if event["category"]=="HAZARD"]
    by_frame={}
    for event in hazards:
        for frame_no in range(event["evidence"]["frame_start"],event["evidence"]["frame_end"]+1):
            by_frame.setdefault(frame_no,[]).append(event)
    cfg=OfflineConfig()
    sit=SituationAnalyzer(min_area=cfg.hazard_min_area,hold_s=cfg.hazard_hold_s,flow_ema=cfg.smoke_flow_ema,base_margin=cfg.base_margin,base_hysteresis=cfg.base_hysteresis,fire_require_smoke=cfg.fire_require_smoke,smoke_window_s=cfg.hazard_smoke_window_s,smoke_texture_min=cfg.hazard_texture_min)
    k=max(2,round(cfg.hazard_smoke_window_s*fps)); scale=WORK_W/float(w); hist={}
    samples={event["event_id"]:[] for event in hazards}
    cap=cv2.VideoCapture(str(video)); frame_no=0
    while True:
        ok,frame=cap.read()
        if not ok: break
        rec,ref=scene.get(frame_no),hist.get(frame_no-k); prev_to_cur=None
        if rec is not None and rec.get("frame_to_scene") is not None and rec.get("scene_to_frame") is not None and ref is not None and ref.get("frame_to_scene") is not None and int(rec.get("scene_segment",-1))==int(ref.get("scene_segment",-2)):
            rel=np.asarray(rec["scene_to_frame"],dtype=np.float32) @ np.asarray(ref["frame_to_scene"],dtype=np.float32)
            prev_to_cur=np.asarray(rel,dtype=np.float32)[:2,:3]; prev_to_cur[:,2]*=scale
        state=sit.update(frame,frame_no/fps,danger_norm=None,ignore=case["ignore"],prev_to_cur=prev_to_cur,scene_motion=True,ref_lag=k)
        if rec is not None:
            hist[frame_no]=rec; hist.pop(frame_no-k,None)
        for event in by_frame.get(frame_no,[]):
            hazard=state.fire if event["evidence"]["kind"]=="fire" else state.smoke
            assert hazard is not None,(case["name"],event["event_id"],frame_no)
            samples[event["event_id"]].append((frame_no,hazard.pos,state.smoke_drift))
        frame_no+=1
    cap.release()
    print(json.dumps({"case":case["name"],"video":str(video),"video_sha256":actual_sha,"run_id":manifest["run_id"],"manifest_video_hash":manifest["video_hash"],"config_hash":manifest["config_hash"],"code_version":manifest["code_version"],"pass_statuses":{key:value["status"] for key,value in manifest["passes"].items()},"frames":frame_no,"events":len(events),"hazards":len(hazards),"replay":{"k":k,"ignore_regions":case["ignore"]}},sort_keys=True,separators=(",",":")))
    for event in hazards:
        rows=samples[event["event_id"]]; pos=np.array([row[1] for row in rows]); drift=np.array([row[2] for row in rows]); mag=np.linalg.norm(drift,axis=1)
        print(json.dumps({"case":case["name"],"event_id":event["event_id"],"kind":event["evidence"]["kind"],"frames":[rows[0][0],rows[-1][0]],"t":[event["t_start"],event["t_end"]],"area_mean":event["evidence"]["area_mean"],"area_peak":event["evidence"]["area_peak"],"pos_first":[round(x,6) for x in pos[0]],"pos_last":[round(x,6) for x in pos[-1]],"pos_min":[round(x,6) for x in pos.min(axis=0)],"pos_max":[round(x,6) for x in pos.max(axis=0)],"drift_first":[round(x,8) for x in drift[0]],"drift_last":[round(x,8) for x in drift[-1]],"drift_mean":[round(x,8) for x in drift.mean(axis=0)],"drift_mag_max":round(float(mag.max()),8)},sort_keys=True,separators=(",",":")))
PY
```

## Maskinläsbart utfall

Detta är stdout från kommandot ovan, omkört 2026-08-15. Positioner är normaliserade bildkoordinater. `smoke_drift` följer motorns befintliga normaliserade Farneback-värde; den kända begränsningen att driftvägen använder owarpad föregående bildruta kvarstår.

```jsonl
{"case":"brand","code_version":"source-sha256:12df3be4d7dcbf91bc38376fb0c5f1a4001d74ee44cca7bc72ef2719d266715f","config_hash":"5a5a9c8cde6a14e7","events":23,"frames":3706,"hazards":8,"manifest_video_hash":"a04a5305f00c7e6ad364ebfec8691e8dffb209a6efe15d354b9b8d9f95de3e20","pass_statuses":{"p1_detect":"complete","p2_track":"complete","p3_identity":"complete","p5_events":"complete"},"replay":{"ignore_regions":[],"k":8},"run_id":"eb365b175b07","video":"/Users/bambapappa/.zclaude/jobs/110f4c45/tmp/drone-halva2-brand.mp4","video_sha256":"e107bf6338f06346c1154d833fc3439ba609941308e2585ae5e7712cea825212"}
{"area_mean":0.01476,"area_peak":0.03062,"case":"brand","drift_first":[0.00038052,6.828e-05],"drift_last":[0.00047396,4.146e-05],"drift_mag_max":0.00115685,"drift_mean":[0.00042791,3.62e-06],"event_id":"hazard-000000","frames":[59,720],"kind":"fire","pos_first":[0.670069,0.74893],"pos_last":[0.693502,0.521869],"pos_max":[0.900273,0.925604],"pos_min":[0.180386,0.226426],"t":[2.36,28.84]}
{"area_mean":0.00748,"area_peak":0.01014,"case":"brand","drift_first":[0.00024865,8.971e-05],"drift_last":[0.00026959,-8.618e-05],"drift_mag_max":0.00057362,"drift_mean":[0.00020727,-8.295e-05],"event_id":"hazard-000001","frames":[330,483],"kind":"smoke","pos_first":[0.776746,0.120261],"pos_last":[0.779398,0.074897],"pos_max":[0.799726,0.925879],"pos_min":[0.14523,0.028603],"t":[13.2,19.36]}
{"area_mean":0.00812,"area_peak":0.01111,"case":"brand","drift_first":[0.00061125,0.00011384],"drift_last":[0.00016006,-5.011e-05],"drift_mag_max":0.00115685,"drift_mean":[0.00057785,1.307e-05],"event_id":"hazard-000002","frames":[620,770],"kind":"smoke","pos_first":[0.90056,0.117247],"pos_last":[0.94947,0.05],"pos_max":[0.96855,0.125898],"pos_min":[0.768125,0.043981],"t":[24.8,30.84]}
{"area_mean":0.00684,"area_peak":0.01347,"case":"brand","drift_first":[0.0001745,-9.544e-05],"drift_last":[0.00015944,0.00010926],"drift_mag_max":0.0005484,"drift_mean":[0.00027156,6.332e-05],"event_id":"hazard-000003","frames":[779,944],"kind":"fire","pos_first":[0.647356,0.532585],"pos_last":[0.601396,0.548227],"pos_max":[0.648454,0.549889],"pos_min":[0.601277,0.463001],"t":[31.16,37.8]}
{"area_mean":0.00622,"area_peak":0.01333,"case":"brand","drift_first":[-0.00063211,-0.00012207],"drift_last":[-0.00035727,-0.00018134],"drift_mag_max":0.00064379,"drift_mean":[-0.00021124,-0.0001374],"event_id":"hazard-000004","frames":[1729,2106],"kind":"fire","pos_first":[0.460606,0.144613],"pos_last":[0.86088,0.556516],"pos_max":[0.943437,0.6598],"pos_min":[0.368097,0.056187],"t":[69.16,84.28]}
{"area_mean":0.0071,"area_peak":0.00972,"case":"brand","drift_first":[0.00010973,-6.136e-05],"drift_last":[2.429e-05,-2.405e-05],"drift_mag_max":0.0008353,"drift_mean":[0.00019164,-5.061e-05],"event_id":"hazard-000005","frames":[2158,2341],"kind":"fire","pos_first":[0.718239,0.080808],"pos_last":[0.685644,0.621782],"pos_max":[0.794342,0.659454],"pos_min":[0.521968,0.080808],"t":[86.32,93.68]}
{"area_mean":0.00777,"area_peak":0.0166,"case":"brand","drift_first":[0.00033524,0.00010871],"drift_last":[-0.00025628,-0.00025311],"drift_mag_max":0.00068397,"drift_mean":[0.00010751,1.772e-05],"event_id":"hazard-000006","frames":[2786,3118],"kind":"fire","pos_first":[0.773235,0.377099],"pos_last":[0.514089,0.477778],"pos_max":[0.784542,0.487125],"pos_min":[0.199432,0.377099],"t":[111.44,124.76]}
{"area_mean":0.00714,"area_peak":0.01278,"case":"brand","drift_first":[0.00046789,-0.00029143],"drift_last":[0.00035352,-0.0002226],"drift_mag_max":0.00072742,"drift_mean":[0.0003157,-0.0001801],"event_id":"hazard-000007","frames":[3435,3584],"kind":"smoke","pos_first":[0.411899,0.099644],"pos_last":[0.488982,0.52646],"pos_max":[0.966887,0.89976],"pos_min":[0.083048,0.016517],"t":[137.4,143.4]}
{"case":"olycka","code_version":"source-sha256:12df3be4d7dcbf91bc38376fb0c5f1a4001d74ee44cca7bc72ef2719d266715f","config_hash":"5a5a9c8cde6a14e7","events":20,"frames":3705,"hazards":0,"manifest_video_hash":"7d7ff50516ad9d45cf142db7fb00845eabbc9de26374dae7bc52550a7484f8d9","pass_statuses":{"p1_detect":"complete","p2_track":"complete","p3_identity":"complete","p5_events":"complete"},"replay":{"ignore_regions":[[0.64,0.0,0.36,0.46]],"k":8},"run_id":"fd10f349f3d7","video":"/Users/bambapappa/.zclaude/jobs/110f4c45/tmp/drone-halva1-olycka.mp4","video_sha256":"e421020d46d014de1ba8814e5cc9630e4de2e02be7ff9998c0bf49f7a6bd9ebe"}
```

Replayresultatet styrker de exakta positions-/driftvärdena i B32 och negativa kontrollen `hazards=0`. Det ändrar inte bedömningen `DONE_WITH_CONCERNS`: fem `kind=fire` i brandhalvan är fortsatt falsklarm, och replayens explicita `ignore_regions` visar samtidigt varför `OfflineConfig.to_dict()`-luckan är viktig.
