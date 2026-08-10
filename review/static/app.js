/* Review UI — thin client over the analysis artifact.
 *
 * Architecture (report §2.5): the native HTML5 <video> element plays the
 * original file — play/pause/scrub/frame-step come free from the browser,
 * no WS streaming, no server-side frame pushing (the realtime PoC's ~210
 * lines of WS plumbing do NOT carry over). An overlay canvas draws boxes /
 * flags / trails synced to the video via requestVideoFrameCallback +
 * the ingest PTS index (frame_no ↔ pts_ms).
 *
 * The draw functions (drawPerson/drawTrail/drawHazards + the COLORS and
 * STATUS_TEXT tables) are ported from the realtime PoC's app.js and adjusted
 * to read artifact rows instead of WS `meta` packets; the rest is new
 * (event list, bookmarks, screenshots, export).
 *
 * Screenshots composite the video frame + overlay canvas to PNG
 * client-side (report §2.5 dual-renderer fix: there is no second
 * server-side screenshot path — snapshot.py is retired by this canvas).
 *
 * All user-facing strings are Swedish (AGENTS.md product rule). Internal
 * identifiers and category enum values stay English (codebase convention).
 */

"use strict";

// ---------- i18n: category enum → Swedish display label ----------
const CATEGORY_LABEL = {
  STILLA: "STILLA",
  MOT_FARA: "MOT FARA",
  IRRATIONELL: "IRRATIONELLT",
  HAZARD: "FARA",
};

// ---------- i18n: review-verdict state → Swedish display label ----------
const REVIEW_STATE_LABEL = {
  unreviewed: "ogranskad",
  confirmed: "bekräftad",
  rejected: "avvisad",
};

// ---------- color tokens shared with the realtime PoC ----------
const COLORS = {
  ok: "#2ecc71",
  still: "#ff4757",
  toward_danger: "#ffa502",
  irrationell: "#a55eea", // new in Phase 4 — distinct from the still/toward hues
  base: "#34c3ff",
  danger: "#ff4757",
  smoke: "#aab4be",
  fire: "#ff6b35",
};

// category -> color, for the timeline strip and active-event badges.
const CATEGORY_COLOR = {
  STILLA: COLORS.still,
  MOT_FARA: COLORS.toward_danger,
  IRRATIONELL: COLORS.irrationell,
  HAZARD: COLORS.fire,
};

// ---------- app state ----------
const state = {
  runId: null,
  runSummary: null,
  events: [],
  bookmarks: [],
  screenshots: [],
  operatorNotes: [],
  frames: [],          // [{frame_no, pts_ms}] sorted window around the playhead
  frameStep: null,     // ms/frame, learned from the first loaded window
  boxesByFrame: null,  // lazy: Map<frame_no, box[]>
  trailCache: { from: null, to: null, frames: {} }, // sliding tracklet window
  activeEventId: null,
  layers: { boxes: true, ids: true, status: true, trails: false, heatmap: false },
  _reviewPauseHandler: null, // current jumpToEvent auto-pause listener, if any
  hazardMarker: { active: false, x: null, y: null }, // Phase 4 retroactive marker
  hazardPlacementArmed: false, // true while waiting for the next canvas click
  // Phase 5. `features` mirrors GET /api/features; until the fetch lands we
  // assume everything on (matching the server defaults) so a failed fetch
  // degrades to showing the controls — the server still 404s disabled ones.
  features: { dossier: true, ground_truth: true, run_compare: true, clip_export: true, heatmap: true },
  persons: [],             // corrected projection from GET /persons
  personsCount: null,
  personsUniqueCount: 0,   // served unique_count (confirmed + manual, no transients)
  personsEngineUncertainty: null, // immutable P3 band + run/pass provenance
  selectedPersonId: null,
  corrections: [],         // live identity-correction ops
  correctionsSkipped: [],  // ops the projection couldn't apply
  gtEntries: [],           // ground-truth ("facit") rows
  heatmap: { canvas: null, loading: false, personId: null }, // offscreen render cache
  clipRecording: false,    // one clip export at a time
};

// ---------- Phase 5: feature gating ----------
// Hide every [data-feature] element whose toggle is off. Server-side the
// corresponding endpoints 404 too — this is cosmetics over that, not the
// enforcement.
// Gating is ADD-ONLY: it never removes `hidden`. Tab panes carry both
// `hidden` (the tab machinery owns it) and data-feature, so unhiding an
// enabled pane here would stack every Phase 5 pane in the sidebar until the
// first tab click.
function applyFeatureGates() {
  document.querySelectorAll("[data-feature]").forEach((el) => {
    const enabled = !!state.features[el.dataset.feature];
    if (!enabled) el.classList.add("hidden");
    // If a hidden tab was active, fall back to the events tab.
    if (!enabled && el.classList.contains("tab") && el.classList.contains("on")) {
      const eventsTab = document.querySelector('#sidebar-tabs .tab[data-tab="events"]');
      if (eventsTab) eventsTab.click();
    }
  });
}

async function refreshFeatures() {
  try {
    const r = await fetch("/api/features");
    if (r.ok) state.features = await r.json();
  } catch (_) { /* keep optimistic defaults; server enforces regardless */ }
  applyFeatureGates();
}

// ---------- DOM ----------
const $ = (sel) => document.querySelector(sel);
const video = $("#video");
const canvas = $("#overlay");
const ctx = canvas.getContext("2d");
const overlayMsg = $("#overlay-msg");
const frameInfo = $("#frame-info");

// ---------- small utilities ----------
function toast(msg, kind = "info") {
  const el = $("#toast");
  el.textContent = msg;
  el.className = kind;
  setTimeout(() => el.classList.add("hidden"), 2400);
}
function fmtT(t) {
  // seconds → m:ss.s
  const m = Math.floor(t / 60);
  const s = (t - m * 60).toFixed(1);
  return `${m}:${s.padStart(4, "0")}`;
}
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

// =====================================================================
// Run picker
// =====================================================================

async function refreshRuns() {
  const r = await fetch("/api/runs");
  const j = await r.json();
  const sel = $("#run-select");
  if (!j.runs.length) {
    sel.innerHTML = '<option value="">— inga körningar hittades —</option>';
    return;
  }
  sel.innerHTML = j.runs
    .map((run) => {
      const p5 = run.passes["p5_events"] || {};
      const evCount = p5.stats?.events_out ?? "–";
      const label = `${run.run_id} · ${run.video_filename || "?"} · ${evCount} händelser`;
      return `<option value="${esc(run.run_id)}">${esc(label)}</option>`;
    })
    .join("");
}

$("#btn-load-run").onclick = async () => {
  const rid = $("#run-select").value;
  if (!rid) return;
  await loadRun(rid);
};

async function loadRun(rid) {
  state.runId = rid;
  state.activeEventId = null;
  state.trailCache = { from: null, to: null, frames: {} };

  // 1. Run summary (drives UI visibility + video URL).
  const [sumRes, annRes, opRes] = await Promise.all([
    fetch(`/api/runs/${rid}`).then((r) => r.json()),
    fetch(`/api/runs/${rid}/annotations`).then((r) => r.json()),
    fetch(`/api/runs/${rid}/operator-notes`).then((r) => r.json()),
  ]);
  state.runSummary = sumRes;
  state.bookmarks = annRes.bookmarks || [];
  state.screenshots = annRes.screenshots || [];
  state.operatorNotes = opRes.notes || [];

  $("#main-empty").hidden = true;
  $("#main-review").hidden = false;

  // 2. Video element: native HTML5 playback. Range requests for seek are
  // handled by the server's FileResponse; nothing else to wire.
  if (sumRes.video_available) {
    video.src = `/api/runs/${rid}/video`;
    // Second hidden element for dossier crops — seeks independently of the
    // main player (same file, browser Range requests handle both).
    $("#crop-video").src = `/api/runs/${rid}/video`;
    overlayMsg.textContent = "Laddar video …";
    overlayMsg.classList.remove("hidden");
    video.addEventListener("loadeddata", () => {
      overlayMsg.classList.add("hidden");
      syncCanvasSize();
    }, { once: true });
    video.addEventListener("loadedmetadata", () => renderTimeline(), { once: true });
    video.addEventListener("error", () => {
      overlayMsg.textContent = "Kunde inte läsa in videon.";
      overlayMsg.classList.remove("hidden");
    });
  } else {
    video.removeAttribute("src");
    $("#crop-video").removeAttribute("src");
    overlayMsg.textContent = "Videofilen saknas i VIDEO_DIR — endast arkivet kan granskas.";
    overlayMsg.classList.remove("hidden");
  }

  // 3. Hazard marker state (Phase 4) before events — get_events already
  // serves MOT_FARA recomputed against it when active, so this just
  // syncs the button UI/legend to match what the event fetch below returns.
  state.hazardPlacementArmed = false;
  await refreshHazardMarker();

  // 4. Events (optional — a run may have skipped P5).
  try {
    const evRes = await fetch(`/api/runs/${rid}/events`).then((r) => r.json());
    state.events = evRes.events || [];
  } catch (e) {
    state.events = [];
  }
  renderEvents();
  renderBookmarks();
  renderScreenshots();
  renderOperatorNotes();
  updateStats();
  refreshComparison();

  // Phase 5 data. Each fetch is independent and failure-tolerant: a
  // disabled feature's endpoint answers 404 and the pane simply stays in
  // its empty state (the tab itself is hidden by applyFeatureGates).
  state.selectedPersonId = null;
  state.heatmap = { canvas: null, loading: false, personId: null };
  $("#person-detail").classList.add("hidden");
  await Promise.all([refreshPersons(), refreshGroundTruth()]);
  renderRunCompareOptions();
  $("#rc-result").classList.add("hidden");

  // 5. PTS index — bridges media-time (seconds) ↔ frame_no. We pull a
  // window around the current playhead on demand rather than the whole
  // file: long films have tens of thousands of frames and the index is
  // only consulted for sync, not for rendering.
  state.frames = [];
  state.frameStep = null;
  state.boxesByFrame = null;
  await ensureFramesWindow(0);
}

// Frames of margin loaded either side of the estimated playhead position —
// bounds the fetch/scan cost regardless of the film's total frame count.
const FRAMES_WINDOW_HALF = 500;
const FRAMES_WINDOW_MARGIN_MS = 1000;

async function ensureFramesWindow(tMs) {
  const frames = state.frames;
  const covered =
    frames.length &&
    tMs >= frames[0].pts_ms - FRAMES_WINDOW_MARGIN_MS &&
    tMs <= frames[frames.length - 1].pts_ms + FRAMES_WINDOW_MARGIN_MS;
  if (covered) return;

  const estimate = state.frameStep ? Math.round(tMs / state.frameStep) : 0;
  const from = Math.max(0, estimate - FRAMES_WINDOW_HALF);
  const to = estimate + FRAMES_WINDOW_HALF;
  try {
    const r = await fetch(
      `/api/runs/${state.runId}/frames/meta?from=${from}&to=${to}`
    ).then((r) => r.json());
    const loaded = r.frames || [];
    if (loaded.length > 1 && !state.frameStep) {
      const first = loaded[0];
      const last = loaded[loaded.length - 1];
      if (last.frame_no > first.frame_no) {
        state.frameStep = (last.pts_ms - first.pts_ms) / (last.frame_no - first.frame_no);
      }
    }
    state.frames = loaded;
  } catch (_) { /* keep the stale window rather than clearing it */ }
}

// =====================================================================
// Overlay canvas — ported draw layer (report §2.5: ~140 lines carry over)
// =====================================================================

function syncCanvasSize() {
  // Size the canvas to match the video's intrinsic pixels; CSS scales it
  // to fit the stage. Drawing happens in video-pixel space, which is what
  // the artifact's normalized boxes need to scale against.
  if (!video.videoWidth || !video.videoHeight) return;
  if (canvas.width !== video.videoWidth) canvas.width = video.videoWidth;
  if (canvas.height !== video.videoHeight) canvas.height = video.videoHeight;
}

function currentFrameNo() {
  // Map video.currentTime (media seconds) → nearest frame_no using the PTS
  // index. pts_ms values come from the ingest decode pass and are the
  // ground-truth sync; we do a linear scan (the index is sorted and the
  // client only needs one lookup per rAF tick).
  if (!state.frames.length) return null;
  const tMs = video.currentTime * 1000;
  let best = state.frames[0];
  let bestDelta = Math.abs(best.pts_ms - tMs);
  for (let i = 1; i < state.frames.length; i++) {
    const d = Math.abs(state.frames[i].pts_ms - tMs);
    if (d < bestDelta) { bestDelta = d; best = state.frames[i]; }
    else if (state.frames[i].pts_ms > tMs) break; // sorted, past the target
  }
  return best.frame_no;
}

async function fetchBoxesForFrame(frameNo) {
  if (!state.runId || frameNo == null) return [];
  if (state.boxesByFrame && state.boxesByFrame.has(frameNo)) {
    return state.boxesByFrame.get(frameNo);
  }
  try {
    const r = await fetch(`/api/runs/${state.runId}/tracklets?frame=${frameNo}`).then((r) => r.json());
    if (!state.boxesByFrame) state.boxesByFrame = new Map();
    // Tiny LRU: keep the last 64 frames' boxes to bound memory on long films.
    if (state.boxesByFrame.size > 64) {
      const firstKey = state.boxesByFrame.keys().next().value;
      state.boxesByFrame.delete(firstKey);
    }
    state.boxesByFrame.set(frameNo, r.boxes || []);
    return r.boxes || [];
  } catch (_) { return []; }
}

// Frames of lookahead buffered past the current playhead per fetch, so as
// playback advances tick by tick the cached window keeps covering the
// requested [frameNo-span, frameNo] range instead of refetching every tick.
const TRAIL_WINDOW_LOOKAHEAD = 150;

async function fetchTrailWindow(frameNo, span = 30) {
  const needFrom = Math.max(0, frameNo - span);
  const cache = state.trailCache;
  if (cache.to != null && frameNo <= cache.to && needFrom >= cache.from) {
    return cache.frames;
  }
  const from = needFrom;
  const to = frameNo + TRAIL_WINDOW_LOOKAHEAD;
  try {
    const r = await fetch(
      `/api/runs/${state.runId}/tracklets/range?from=${from}&to=${to}`
    ).then((r) => r.json());
    state.trailCache = { from, to, frames: r.frames || {} };
    return state.trailCache.frames;
  } catch (_) { return {}; }
}

let lastDrawnFrame = null;
async function drawOverlay() {
  // requestVideoFrameCallback gives us a frame-accurate tick when the
  // browser composites a new video frame; we fall back to rAF where
  // unsupported (Safari < 14, etc.) — the slight latency is acceptable
  // for review overlay, unlike the realtime PoC where it would matter.
  if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) {
    video.requestVideoFrameCallback(drawOverlay);
  } else {
    requestAnimationFrame(drawOverlay);
  }
  if (!video.readyState || video.readyState < 2) return;
  syncCanvasSize();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (state.layers.heatmap) drawHeatmapLayer(canvas.width, canvas.height);
  drawHazardMarker(canvas.width, canvas.height);

  await ensureFramesWindow(video.currentTime * 1000);
  const frameNo = currentFrameNo();
  if (frameNo == null) { frameInfo.classList.add("hidden"); return; }
  if (frameNo !== lastDrawnFrame) {
    frameInfo.textContent = `ruta ${frameNo}`;
    frameInfo.classList.remove("hidden");
    lastDrawnFrame = frameNo;
  }

  const W = canvas.width, H = canvas.height;
  // Artifact boxes are in pixel space already (P2's xyxy is tracker/Kalman-
  // adjusted frame pixels). The overlay canvas is sized to video intrinsic
  // pixels, so we draw 1:1 — no normalization math here, unlike the
  // realtime PoC where the WS meta contract carried normalized [0..1] boxes.
  const boxes = await fetchBoxesForFrame(frameNo);

  // Active event highlight: if the current time is inside an event's
  // [t_start, t_end], underline the relevant person box (if any) in red.
  const activeEvents = state.events.filter(
    (e) => video.currentTime >= e.t_start && video.currentTime <= e.t_end
  );

  if (state.layers.trails) {
    const trailFrames = await fetchTrailWindow(frameNo);
    drawTrails(trailFrames, frameNo, W, H);
  }
  if (state.layers.boxes) {
    for (const b of boxes) drawPerson(b, W, H, activeEvents);
  } else if (state.layers.status) {
    // Even with boxes off, still flag the still/toward persons (minimal
    // prominence — a thin colored bar above the box).
    for (const b of boxes) {
      const st = statusFor(b, activeEvents);
      if (st && st !== "ok") drawStatusFlag(b, st, W, H);
    }
  }

  // Active event badge top-right.
  if (activeEvents.length) {
    drawActiveBadges(activeEvents, W, H);
  }
}

function statusFor(box, activeEvents) {
  // Derive a behavior status for this box from active events at the current
  // time. The artifact doesn't store per-frame status post-P5 (we diffed
  // it into events) — so we reverse-derive from the active-event set: if
  // an event covers this person/tracklet at currentTime, that's the status.
  if (!activeEvents || !activeEvents.length) return "ok";
  for (const ev of activeEvents) {
    if (ev.category === "STILLA" && ev.person_id != null && ev.person_id === box.person_id) {
      return "still";
    }
    if (ev.category === "MOT_FARA" && ev.person_id != null && ev.person_id === box.person_id) {
      return "toward_danger";
    }
    if (ev.category === "IRRATIONELL" && ev.person_id != null && ev.person_id === box.person_id) {
      return "irrationell";
    }
  }
  return "ok";
}

// ---- draw primitives (ported from realtime PoC app.js, ~140 lines) ----

function drawPerson(b, W, H, activeEvents) {
  const [x0, y0, x1, y1] = b.xyxy;
  const status = state.layers.status ? statusFor(b, activeEvents) : "ok";
  const color = COLORS[status] || COLORS.ok;
  const lw = Math.max(2, W / 480);
  ctx.lineWidth = status === "ok" ? lw : lw * 1.6;
  ctx.strokeStyle = color;
  ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);

  if (!state.layers.ids && status === "ok") return;
  let label = "";
  if (state.layers.ids && b.person_id != null) label = `P${b.person_id}`;
  else if (state.layers.ids) label = `T${b.tracklet_id}`;
  if (state.layers.status && status === "still") label += `${label ? " · " : ""}STILLA`;
  if (state.layers.status && status === "toward_danger") label += `${label ? " · " : ""}MOT FARA`;
  if (state.layers.status && status === "irrationell") label += `${label ? " · " : ""}IRRATIONELLT`;
  if (!label) return;

  ctx.font = `bold ${Math.max(11, W / 60)}px system-ui, sans-serif`;
  ctx.textBaseline = "bottom";
  const tw = ctx.measureText(label).width;
  const th = parseInt(ctx.font, 10) + 6;
  const ly = y0 > th ? y0 : y0 + (y1 - y0) + th;
  ctx.fillStyle = status === "ok" ? "rgba(10,14,18,.75)" : color;
  ctx.fillRect(x0 - lw / 2, ly - th, tw + 10, th);
  ctx.fillStyle = status === "ok" ? color : "#0c1014";
  ctx.fillText(label, x0 + 4, ly - 3);
}

function drawStatusFlag(b, st, W, H) {
  const [x0, y0, x1, y1] = b.xyxy;
  const lw = Math.max(2, W / 480);
  ctx.lineWidth = lw;
  ctx.strokeStyle = COLORS[st] || COLORS.ok;
  ctx.beginPath();
  ctx.moveTo(x0, y0 - 6); ctx.lineTo(x0 + 16, y0 - 6);
  ctx.stroke();
}

function drawTrails(frames, lastFrameNo, W, H) {
  // Group trail points by tracklet_id across the window.
  const byTrack = new Map();
  for (let f = lastFrameNo; f >= lastFrameNo - 30 && f >= 0; f--) {
    const boxes = frames[f] || frames[String(f)] || [];
    for (const b of boxes) {
      if (!byTrack.has(b.tracklet_id)) byTrack.set(b.tracklet_id, []);
      const [x0, y0, x1, y1] = b.xyxy;
      byTrack.get(b.tracklet_id).push([(x0 + x1) / 2, y1]);
    }
  }
  const lw = Math.max(2, W / 480);
  ctx.lineWidth = lw;
  for (const [tid, pts] of byTrack) {
    if (pts.length < 2) continue;
    ctx.strokeStyle = COLORS.ok + "66";
    ctx.beginPath();
    pts.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)));
    ctx.stroke();
  }
}

function drawActiveBadges(events, W, H) {
  ctx.font = `bold ${Math.max(12, W / 70)}px system-ui, sans-serif`;
  ctx.textBaseline = "top";
  ctx.textAlign = "right";
  let y = 12;
  for (const ev of events) {
    const label = CATEGORY_LABEL[ev.category] || ev.category;
    const w = ctx.measureText(label).width + 16;
    ctx.fillStyle = CATEGORY_COLOR[ev.category] || COLORS.fire;
    ctx.fillRect(W - w - 8, y, w, 24);
    ctx.fillStyle = "#0c1014";
    ctx.fillText(label, W - 16, y + 5);
    y += 28;
  }
  ctx.textAlign = "left"; // reset
}

function drawHazardMarker(W, H) {
  // The manually placed hazard marker (Phase 4, report §5.1) — same pixel
  // space as tracklet boxes, drawn as a small pin so the reviewer always
  // sees where MOT_FARA is currently being computed against.
  if (!state.hazardMarker.active) return;
  const { x, y } = state.hazardMarker;
  const r = Math.max(6, W / 90);
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fillStyle = COLORS.toward_danger;
  ctx.fill();
  ctx.lineWidth = 2;
  ctx.strokeStyle = "#0c1014";
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x, y - r);
  ctx.lineTo(x, y - r - 14);
  ctx.stroke();
  ctx.font = `bold ${Math.max(11, W / 70)}px system-ui, sans-serif`;
  ctx.textBaseline = "bottom";
  ctx.fillStyle = COLORS.toward_danger;
  ctx.fillText("FAROMARKÖR", x + r + 4, y + r);
}

// =====================================================================
// Layer toggles
// =====================================================================

document.querySelectorAll("#toggles .chip[data-layer]").forEach((btn) => {
  const key = btn.dataset.layer;
  btn.classList.toggle("on", !!state.layers[key]);
  btn.onclick = () => {
    setLayer(key, !state.layers[key]);
    if (key === "heatmap" && state.layers.heatmap) ensureHeatmap(state.heatmap.personId);
  };
});

function setLayer(key, on) {
  state.layers[key] = on;
  const chip = document.querySelector(`#toggles .chip[data-layer="${key}"]`);
  if (chip) chip.classList.toggle("on", on);
  lastDrawnFrame = null; // force redraw
}

// =====================================================================
// Phase 5: dwell/coverage heatmap layer (report §5.9)
// =====================================================================

// Fetch the dwell grid once per (run, person filter) and pre-render it to a
// tiny offscreen canvas at grid resolution; drawOverlay just stretches that
// image every tick, so the per-frame cost is one drawImage.
async function ensureHeatmap(personId) {
  if (!state.runId || state.heatmap.loading) return;
  if (state.heatmap.canvas && state.heatmap.personId === personId) return;
  state.heatmap.loading = true;
  try {
    const q = personId != null ? `?person_id=${personId}` : "";
    const r = await fetch(`/api/runs/${state.runId}/heatmap${q}`);
    if (!r.ok) { toast("Kunde inte hämta värmekartan", "error"); return; }
    const hm = await r.json();
    state.heatmap.canvas = renderHeatmapCanvas(hm);
    state.heatmap.personId = personId;
    lastDrawnFrame = null;
  } catch (_) {
    toast("Kunde inte hämta värmekartan", "error");
  } finally {
    state.heatmap.loading = false;
  }
}

function renderHeatmapCanvas(hm) {
  if (!hm.grid_w || !hm.grid_h || !hm.max_s) return null;
  const off = document.createElement("canvas");
  off.width = hm.grid_w;
  off.height = hm.grid_h;
  const octx = off.getContext("2d");
  const img = octx.createImageData(hm.grid_w, hm.grid_h);
  for (let gy = 0; gy < hm.grid_h; gy++) {
    for (let gx = 0; gx < hm.grid_w; gx++) {
      const v = hm.cell_s[gy][gx] / hm.max_s; // 0..1
      if (v <= 0) continue;
      const i = (gy * hm.grid_w + gx) * 4;
      // Cold→hot ramp: blue → yellow → red, alpha grows with dwell.
      img.data[i] = Math.round(255 * Math.min(1, v * 2));
      img.data[i + 1] = Math.round(v < 0.5 ? 255 * v * 2 * 0.8 : 255 * (1 - v) * 1.6);
      img.data[i + 2] = Math.round(255 * Math.max(0, 1 - v * 2));
      img.data[i + 3] = Math.round(60 + 150 * v);
    }
  }
  octx.putImageData(img, 0, 0);
  return off;
}

function drawHeatmapLayer(W, H) {
  const off = state.heatmap.canvas;
  if (!off) return;
  ctx.save();
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(off, 0, 0, W, H);
  // Always say WHICH map is on screen — a per-person map looks like a
  // sparse whole-run map otherwise.
  const pid = state.heatmap.personId;
  const caption = pid != null ? `Värmekarta: person P${pid}` : "Värmekarta: hela körningen";
  ctx.font = `bold ${Math.max(11, W / 80)}px system-ui, sans-serif`;
  ctx.textBaseline = "bottom";
  const pad = Math.max(6, W / 160);
  const tw = ctx.measureText(caption).width;
  const th = Math.max(11, W / 80) + pad;
  ctx.fillStyle = "rgba(12,16,20,0.72)";
  ctx.fillRect(pad, H - th - pad, tw + pad * 2, th);
  ctx.fillStyle = "#e8edf2";
  ctx.fillText(caption, pad * 2, H - pad * 1.6);
  ctx.restore();
}

// The dossier's per-person filter: the same /heatmap endpoint with
// ?person_id=, resolved through the CORRECTED tracklet→person map server
// side. Clicking again returns to the whole-run map.
function renderPersonHeatmapControl(pid) {
  const btn = $("#pd-heatmap-btn");
  const label = $("#pd-heatmap-state");
  if (!btn) return;
  const showing = state.layers.heatmap && state.heatmap.personId === pid;
  btn.textContent = showing ? "Visa hela körningen" : "Värmekarta för personen";
  label.textContent = state.layers.heatmap
    ? (state.heatmap.personId != null ? `visar P${state.heatmap.personId}` : "visar hela körningen")
    : "";
  btn.onclick = () => {
    const toPerson = !(state.layers.heatmap && state.heatmap.personId === pid);
    setLayer("heatmap", true);
    ensureHeatmap(toPerson ? pid : null).then(() => renderPersonHeatmapControl(pid));
  };
}

// =====================================================================
// Sidebar tabs
// =====================================================================

document.querySelectorAll("#sidebar-tabs .tab").forEach((btn) => {
  btn.onclick = () => {
    document.querySelectorAll("#sidebar-tabs .tab").forEach((b) => b.classList.toggle("on", b === btn));
    document.querySelectorAll(".tab-pane").forEach((p) => p.classList.toggle("hidden", p.dataset.pane !== btn.dataset.tab));
    if (btn.dataset.tab === "timeline") renderTimeline();
  };
});

// =====================================================================
// Phase 4: timeline strip — per-person flag lanes, hazard/bookmark/operator
// markers, click-to-seek. Turns a long film into the "5-second visual scan"
// the report frames this feature around.
// =====================================================================

const TIMELINE_PX_PER_S = 8;
const TIMELINE_LANE_H = 22;
const TIMELINE_LANE_GAP = 6;
const TIMELINE_LABEL_W = 96;

function timelineDurationS() {
  if (video.duration && isFinite(video.duration) && video.duration > 0) return video.duration;
  let maxT = 0;
  for (const e of state.events) maxT = Math.max(maxT, e.t_end || 0);
  for (const b of state.bookmarks) maxT = Math.max(maxT, b.t || 0);
  for (const n of state.operatorNotes) maxT = Math.max(maxT, n.t || 0);
  return maxT + 5;
}

function renderTimeline() {
  const svg = $("#timeline-svg");
  const empty = $("#timeline-empty");
  const hasContent = state.events.length || state.bookmarks.length || state.operatorNotes.length;
  if (!state.runId || !hasContent) {
    svg.classList.add("hidden");
    empty.classList.remove("hidden");
    empty.textContent = state.runId ? "Inget att visa på tidslinjen ännu." : "Ingen körning inläst.";
    $("#timeline-legend").textContent = "";
    return;
  }
  empty.classList.add("hidden");
  svg.classList.remove("hidden");

  // Group person-keyed flags by person_id; HAZARD (person_id=null) gets its
  // own "Fara" lane instead of being lumped with "okänd person".
  const byPerson = new Map();
  const hazardRow = [];
  for (const e of state.events) {
    if (e.category === "HAZARD") { hazardRow.push(e); continue; }
    const pid = e.person_id != null ? e.person_id : "okänd";
    if (!byPerson.has(pid)) byPerson.set(pid, []);
    byPerson.get(pid).push(e);
  }
  const personIds = [...byPerson.keys()].sort((a, b) => {
    if (a === "okänd") return 1;
    if (b === "okänd") return -1;
    return a - b;
  });

  const rows = personIds.map((pid) => ({
    label: pid === "okänd" ? "Okänd person" : `Person P${pid}`,
    kind: "spans",
    items: byPerson.get(pid),
  }));
  if (hazardRow.length) rows.push({ label: "Fara", kind: "spans", items: hazardRow });
  if (state.bookmarks.length) rows.push({ label: "Bokmärken", kind: "points", items: state.bookmarks });
  if (state.operatorNotes.length) {
    rows.push({
      label: "Operatör",
      kind: "points",
      items: state.operatorNotes.map((n) => ({ t: n.t, label: n.text })),
    });
  }

  const duration = timelineDurationS();
  const width = TIMELINE_LABEL_W + Math.max(200, duration * TIMELINE_PX_PER_S);
  const height = rows.length * (TIMELINE_LANE_H + TIMELINE_LANE_GAP) + TIMELINE_LANE_GAP + 20;
  svg.setAttribute("width", width);
  svg.setAttribute("height", height);
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);

  const parts = [];
  for (let t = 0; t <= duration; t += 30) {
    const x = TIMELINE_LABEL_W + t * TIMELINE_PX_PER_S;
    parts.push(`<line x1="${x}" y1="0" x2="${x}" y2="${height}" class="tl-grid"></line>`);
    parts.push(`<text x="${x + 3}" y="12" class="tl-axis">${fmtT(t)}</text>`);
  }

  rows.forEach((row, i) => {
    const y = 20 + i * (TIMELINE_LANE_H + TIMELINE_LANE_GAP);
    parts.push(`<text x="4" y="${y + TIMELINE_LANE_H / 2 + 4}" class="tl-label">${esc(row.label)}</text>`);
    if (row.kind === "spans") {
      for (const e of row.items) {
        const x0 = TIMELINE_LABEL_W + e.t_start * TIMELINE_PX_PER_S;
        const w = Math.max(2, (e.t_end - e.t_start) * TIMELINE_PX_PER_S);
        const color = CATEGORY_COLOR[e.category] || COLORS.ok;
        const label = CATEGORY_LABEL[e.category] || e.category;
        const tip = `${label} · ${fmtT(e.t_start)}–${fmtT(e.t_end)} · v ${e.confidence.toFixed(2)}`;
        parts.push(
          `<rect x="${x0}" y="${y}" width="${w}" height="${TIMELINE_LANE_H}" rx="3" fill="${color}" ` +
          `class="tl-span" data-seek="${e.t_start}"><title>${esc(tip)}</title></rect>`
        );
      }
    } else {
      for (const item of row.items) {
        const x = TIMELINE_LABEL_W + item.t * TIMELINE_PX_PER_S;
        const cy = y + TIMELINE_LANE_H / 2;
        const tip = `${esc(item.label || "")} · ${fmtT(item.t)}`;
        parts.push(
          `<circle cx="${x}" cy="${cy}" r="5" class="tl-marker" data-seek="${item.t}"><title>${tip}</title></circle>`
        );
      }
    }
  });

  svg.innerHTML = parts.join("");
  svg.querySelectorAll("[data-seek]").forEach((el) => {
    el.addEventListener("click", () => {
      video.currentTime = Math.max(0, parseFloat(el.dataset.seek));
    });
  });

  $("#timeline-legend").textContent = state.hazardMarker.active
    ? `faromarkör placerad (${state.hazardMarker.x.toFixed(0)}, ${state.hazardMarker.y.toFixed(0)})`
    : "";
}

// =====================================================================
// Phase 4: retroactive hazard marker — place/move by clicking the video,
// instant recompute of MOT_FARA (report §5.1)
// =====================================================================

async function refreshHazardMarker() {
  if (!state.runId) return;
  try {
    const r = await fetch(`/api/runs/${state.runId}/hazard-marker`).then((res) => res.json());
    state.hazardMarker = r.active ? { active: true, x: r.x, y: r.y } : { active: false, x: null, y: null };
  } catch (_) {
    state.hazardMarker = { active: false, x: null, y: null };
  }
  updateHazardMarkerButtons();
}

function updateHazardMarkerButtons() {
  const btn = $("#btn-hazard-marker");
  const clearBtn = $("#btn-hazard-clear");
  btn.textContent = state.hazardPlacementArmed
    ? "📍 Klicka i bilden …"
    : state.hazardMarker.active
      ? "📍 Flytta faromarkör"
      : "📍 Faromarkör";
  btn.classList.toggle("on", state.hazardPlacementArmed);
  clearBtn.classList.toggle("hidden", !state.hazardMarker.active);
}

async function setHazardMarker(x, y) {
  const r = await fetch(`/api/runs/${state.runId}/hazard-marker`, {
    method: "POST",
    body: new URLSearchParams({ x: String(x), y: String(y) }),
  });
  if (!r.ok) { toast("Kunde inte flytta faromarkören", "error"); return; }
  const row = await r.json();
  state.hazardMarker = { active: true, x: row.x, y: row.y };
  updateHazardMarkerButtons();
  toast("Faromarkör flyttad — MOT FARA omberäknad", "success");
  await reloadEventsAfterHazardChange();
}

async function reloadEventsAfterHazardChange() {
  // MOT_FARA is recomputed server-side at read time (review/hazard.py) — a
  // plain re-fetch of the event log is all "instant recompute" requires,
  // no batch job, no polling.
  const evRes = await fetch(`/api/runs/${state.runId}/events`).then((r) => r.json());
  state.events = evRes.events || [];
  lastDrawnFrame = null; // force overlay redraw so MOT_FARA badges update now
  renderEvents();
  renderTimeline();
}

$("#btn-hazard-marker").onclick = () => {
  if (!state.runId) return;
  state.hazardPlacementArmed = !state.hazardPlacementArmed;
  updateHazardMarkerButtons();
  if (state.hazardPlacementArmed) toast("Klicka i bilden för att placera faromarkören", "info");
};

$("#btn-hazard-clear").onclick = async () => {
  if (!state.runId) return;
  const r = await fetch(`/api/runs/${state.runId}/hazard-marker`, { method: "DELETE" });
  if (r.ok) {
    state.hazardMarker = { active: false, x: null, y: null };
    updateHazardMarkerButtons();
    toast("Faromarkör borttagen — återgår till AI-detekterad fara", "success");
    await reloadEventsAfterHazardChange();
  } else {
    toast("Kunde inte ta bort faromarkören", "error");
  }
};

$("#stage").addEventListener("click", (e) => {
  if (!state.hazardPlacementArmed || !state.runId) return;
  if (video.readyState < 1 || !canvas.width || !canvas.height) return;
  const rect = canvas.getBoundingClientRect();
  const dispX = e.clientX - rect.left;
  const dispY = e.clientY - rect.top;
  if (dispX < 0 || dispY < 0 || dispX > rect.width || dispY > rect.height) return;
  const x = (dispX / rect.width) * canvas.width;
  const y = (dispY / rect.height) * canvas.height;
  state.hazardPlacementArmed = false;
  updateHazardMarkerButtons();
  setHazardMarker(x, y);
});

// =====================================================================
// Event list / review queue — jump-to-timestamp + confirm/reject/note
// =====================================================================

function sortedEvents() {
  const mode = $("#event-sort") ? $("#event-sort").value : "time";
  const list = [...state.events];
  if (mode === "confidence") list.sort((a, b) => b.confidence - a.confidence);
  else list.sort((a, b) => a.t_start - b.t_start);
  return list;
}

function renderEvents() {
  const ul = $("#event-list");
  const header = ul.closest(".card").querySelector("h3");
  if (header) {
    const c = header.querySelector(".count");
    if (c) c.textContent = state.events.length;
    else header.insertAdjacentHTML("beforeend", `<span class="count">${state.events.length}</span>`);
  }
  if (!state.events.length) {
    ul.innerHTML = '<li class="dim">Inga händelser.</li>';
    renderTimeline();
    return;
  }
  ul.innerHTML = sortedEvents().map((ev) => {
    const cls = state.activeEventId === ev.event_id ? "active" : "";
    const review = ev.review || { state: "unreviewed", note: null };
    const cat = `<span class="cat-tag cat-${ev.category}">${CATEGORY_LABEL[ev.category] || ev.category}</span>`;
    const pid = ev.person_id != null ? `P${ev.person_id}` : "—";
    const dur = (ev.t_end - ev.t_start).toFixed(1);
    const meta = `<span class="meta">${fmtT(ev.t_start)} · ${dur}s · ${pid} · v ${ev.confidence.toFixed(2)}</span>`;
    // IRRATIONELL's evidence has no bare "kind" — it names which sub-signals
    // fired (report §4: never a bare label). HAZARD keeps its fire/smoke tag.
    const note = ev.evidence && ev.evidence.kind
      ? `<span class="note">typ: ${ev.evidence.kind}</span>`
      : ev.evidence && ev.evidence.summary
        ? `<span class="note">${esc(ev.evidence.summary)}</span>`
        : "";
    const badge = `<span class="review-badge">${REVIEW_STATE_LABEL[review.state] || "ogranskad"}</span>`;
    return `<li class="${cls} review-${esc(review.state)}" data-event-id="${esc(ev.event_id)}">
      <span class="label">${cat}</span>
      ${meta}
      ${note}
      ${badge}
      <div class="review-actions">
        <button type="button" class="btn-confirm" data-id="${esc(ev.event_id)}">Bekräfta</button>
        <button type="button" class="btn-reject" data-id="${esc(ev.event_id)}">Avvisa</button>
        <button type="button" class="btn-note" data-id="${esc(ev.event_id)}">Anteckning</button>
        ${state.features.clip_export ? `<button type="button" class="btn-clip" data-id="${esc(ev.event_id)}" title="Exportera annoterat videoklipp ±5 s runt händelsen">🎬 Klipp</button>` : ""}
      </div>
      <form class="note-form hidden" data-id="${esc(ev.event_id)}">
        <input type="text" class="note-input" placeholder="Anteckning" maxlength="4000" value="${esc(review.note || "")}">
        <button type="submit" class="primary">Spara</button>
      </form>
    </li>`;
  }).join("");
  ul.querySelectorAll("li[data-event-id]").forEach((li) => {
    li.onclick = (e) => {
      if (e.target.closest("button, input, form")) return;
      jumpToEvent(li.dataset.eventId);
    };
  });
  ul.querySelectorAll(".btn-confirm").forEach((btn) => {
    btn.onclick = (e) => { e.stopPropagation(); setEventReview(btn.dataset.id, { state: "confirmed" }); };
  });
  ul.querySelectorAll(".btn-reject").forEach((btn) => {
    btn.onclick = (e) => { e.stopPropagation(); setEventReview(btn.dataset.id, { state: "rejected" }); };
  });
  ul.querySelectorAll(".btn-note").forEach((btn) => {
    btn.onclick = (e) => {
      e.stopPropagation();
      btn.closest("li").querySelector(".note-form").classList.toggle("hidden");
    };
  });
  ul.querySelectorAll(".btn-clip").forEach((btn) => {
    btn.onclick = (e) => { e.stopPropagation(); exportClip(btn.dataset.id); };
  });
  ul.querySelectorAll(".note-form").forEach((form) => {
    form.onclick = (e) => e.stopPropagation();
    form.onsubmit = async (e) => {
      e.preventDefault();
      const note = form.querySelector(".note-input").value.trim();
      await setEventReview(form.dataset.id, { note });
      form.classList.add("hidden");
    };
  });
  renderTimeline();
}

async function setEventReview(eventId, fields) {
  // `fields` may include state and/or note — omitted fields carry forward
  // their previous value server-side (see review/annotations.py's
  // set_verdict). Re-fetches the event list afterward so the merged verdict
  // (annotations layer overlaid on the frozen engine table) is authoritative
  // rather than guessed at client-side.
  const body = new URLSearchParams();
  if (fields.state !== undefined) body.set("state", fields.state);
  if (fields.note !== undefined) body.set("note", fields.note);
  const r = await fetch(`/api/runs/${state.runId}/events/${eventId}/review`, { method: "POST", body });
  if (r.ok) {
    const evRes = await fetch(`/api/runs/${state.runId}/events`).then((res) => res.json());
    state.events = evRes.events || [];
    renderEvents();
    toast("Granskning sparad", "success");
  } else {
    toast("Kunde inte spara granskning", "error");
  }
}

$("#event-sort").onchange = renderEvents;

$("#btn-next-unreviewed").onclick = () => {
  const list = sortedEvents();
  if (!list.length) { toast("Inga händelser att granska", "info"); return; }
  const currentIdx = list.findIndex((e) => e.event_id === state.activeEventId);
  const isUnreviewed = (e) => (e.review ? e.review.state : "unreviewed") === "unreviewed";
  const next = list.slice(currentIdx + 1).find(isUnreviewed) || list.find(isUnreviewed);
  if (!next) { toast("Inga fler ogranskade händelser", "success"); return; }
  jumpToEvent(next.event_id);
};

// ~5s lead-in/lead-out context window around an event, per the review-queue
// spec (report §5.2: "auto-seek ... with a small (~5s) context window").
const REVIEW_CONTEXT_S = 5.0;

function jumpToEvent(eid) {
  const ev = state.events.find((e) => e.event_id === eid);
  if (!ev) return;
  state.activeEventId = eid;
  try { video.currentTime = Math.max(0, ev.t_start - REVIEW_CONTEXT_S); } catch (_) {}
  video.play().catch(() => {}); // ignore autoplay rejection — user gesture already happened

  // Auto-pause once playback runs REVIEW_CONTEXT_S past the event's offset,
  // bounding the context clip instead of letting playback continue
  // indefinitely (the reviewer can always resume manually).
  if (state._reviewPauseHandler) {
    video.removeEventListener("timeupdate", state._reviewPauseHandler);
  }
  const pauseAt = ev.t_end + REVIEW_CONTEXT_S;
  const onTime = () => {
    if (video.currentTime >= pauseAt) {
      video.pause();
      video.removeEventListener("timeupdate", onTime);
      state._reviewPauseHandler = null;
    }
  };
  state._reviewPauseHandler = onTime;
  video.addEventListener("timeupdate", onTime);

  renderEvents();
}

// =====================================================================
// Phase 5: annotated clip export (report §5.8)
// =====================================================================

// Records the SAME composite the screenshot button captures (video frame +
// overlay canvas), continuously, via canvas.captureStream + MediaRecorder —
// the browser stays the single annotated-frame renderer (report §2.5's
// dual-renderer rule; no server-side video encoder is ever added for this).
// The recording runs in real time: a 20-second clip takes 20 seconds, since
// the overlay must actually be drawn against the playing video.
async function exportClip(eventId) {
  if (!state.features.clip_export) return;
  if (state.clipRecording) { toast("En klippinspelning pågår redan", "error"); return; }
  const ev = state.events.find((e) => e.event_id === eventId);
  if (!ev) return;
  if (!video.src || video.readyState < 2) { toast("Videon är inte redo", "error"); return; }
  if (!window.MediaRecorder || !HTMLCanvasElement.prototype.captureStream) {
    toast("Webbläsaren saknar stöd för klippinspelning (MediaRecorder)", "error");
    return;
  }

  const t0 = Math.max(0, ev.t_start - REVIEW_CONTEXT_S);
  const t1 = ev.t_end + REVIEW_CONTEXT_S;
  state.clipRecording = true;
  toast(`Spelar in klipp (${(t1 - t0).toFixed(0)} s i realtid) …`, "info");

  // A pending jumpToEvent auto-pause would cut the recording short.
  if (state._reviewPauseHandler) {
    video.removeEventListener("timeupdate", state._reviewPauseHandler);
    state._reviewPauseHandler = null;
  }

  const out = document.createElement("canvas");
  out.width = video.videoWidth;
  out.height = video.videoHeight;
  const outCtx = out.getContext("2d");
  const mime = MediaRecorder.isTypeSupported("video/webm;codecs=vp9") ? "video/webm;codecs=vp9" : "video/webm";
  const rec = new MediaRecorder(out.captureStream(30), { mimeType: mime });
  const chunks = [];
  rec.ondataavailable = (e) => { if (e.data && e.data.size) chunks.push(e.data); };
  const recStopped = new Promise((resolve) => { rec.onstop = resolve; });

  let rafId = null;
  const compose = () => {
    outCtx.drawImage(video, 0, 0, out.width, out.height);
    outCtx.drawImage(canvas, 0, 0, out.width, out.height);
    rafId = requestAnimationFrame(compose);
  };

  let finished = false;
  const finish = async () => {
    if (finished) return;
    finished = true;
    cancelAnimationFrame(rafId);
    video.removeEventListener("timeupdate", onTime);
    video.removeEventListener("ended", finish);
    video.pause();
    rec.stop();
    await recStopped;
    state.clipRecording = false;
    if (!chunks.length) { toast("Klippet blev tomt", "error"); return; }
    const blob = new Blob(chunks, { type: "video/webm" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${state.runId}_${eventId}_klipp.webm`;
    a.click();
    URL.revokeObjectURL(url);
    toast("Klipp exporterat (WebM)", "success");
  };
  const onTime = () => { if (video.currentTime >= t1) finish(); };

  try {
    await seekOnce(video, t0);
    compose();
    rec.start(250);
    video.addEventListener("timeupdate", onTime);
    video.addEventListener("ended", finish);
    await video.play();
  } catch (_) {
    state.clipRecording = false;
    cancelAnimationFrame(rafId);
    video.removeEventListener("timeupdate", onTime);
    video.removeEventListener("ended", finish);
    if (rec.state !== "inactive") rec.stop();
    toast("Kunde inte starta klippinspelningen", "error");
  }
}

// =====================================================================
// Bookmarks
// =====================================================================

function renderBookmarks() {
  const ul = $("#bookmark-list");
  const header = ul.closest(".card").querySelector("h3");
  if (header) {
    let c = header.querySelector(".count");
    if (!c) { header.insertAdjacentHTML("beforeend", '<span class="count">0</span>'); c = header.querySelector(".count"); }
    c.textContent = state.bookmarks.length;
  }
  if (!state.bookmarks.length) {
    ul.innerHTML = '<li class="dim">Inga bokmärken.</li>';
    renderTimeline();
    return;
  }
  const sorted = [...state.bookmarks].sort((a, b) => a.t - b.t);
  ul.innerHTML = sorted.map((b) => `
    <li data-id="${esc(b.annotation_id)}">
      <span class="label">${esc(b.label)}</span>
      <span class="meta">${fmtT(b.t)}</span>
      ${b.note ? `<span class="note">${esc(b.note)}</span>` : ""}
      <button class="del" data-id="${esc(b.annotation_id)}" title="Ta bort">Ta bort</button>
    </li>
  `).join("");
  ul.querySelectorAll("li[data-id]").forEach((li) => {
    li.onclick = (e) => {
      if (e.target.classList.contains("del")) return;
      const id = li.dataset.id;
      const b = state.bookmarks.find((x) => x.annotation_id === id);
      if (b) { video.currentTime = b.t; }
    };
  });
  ul.querySelectorAll("button.del").forEach((btn) => {
    btn.onclick = async (e) => {
      e.stopPropagation();
      const id = btn.dataset.id;
      const r = await fetch(`/api/runs/${state.runId}/bookmarks/${id}`, { method: "DELETE" });
      if (r.ok) {
        state.bookmarks = state.bookmarks.filter((b) => b.annotation_id !== id);
        renderBookmarks();
        updateStats();
        toast("Bokmärke borttaget", "success");
      } else { toast("Kunde inte ta bort bokmärke", "error"); }
    };
  });
  renderTimeline();
}

$("#btn-bookmark").onclick = () => {
  if (!state.runId) return;
  const form = $("#bookmark-form");
  form.classList.toggle("hidden");
  if (!form.classList.contains("hidden")) {
    $("#bm-label").focus();
    $("#bm-label").value = "";
    $("#bm-note").value = "";
  }
};

$("#bm-cancel").onclick = () => { $("#bookmark-form").classList.add("hidden"); };

$("#bookmark-form").onsubmit = async (e) => {
  e.preventDefault();
  const t = video.currentTime;
  const label = $("#bm-label").value.trim();
  const note = $("#bm-note").value.trim() || null;
  if (!label) return;
  const r = await fetch(`/api/runs/${state.runId}/bookmarks`, {
    method: "POST",
    body: new URLSearchParams({ t: String(t), label, note: note || "" }),
  });
  if (r.ok) {
    const row = await r.json();
    state.bookmarks.push(row);
    renderBookmarks();
    updateStats();
    $("#bookmark-form").classList.add("hidden");
    toast("Bokmärke sparat", "success");
  } else { toast("Kunde inte spara bokmärke", "error"); }
};

// =====================================================================
// Screenshots — client-side composite (video + overlay canvas → PNG)
// =====================================================================

function renderScreenshots() {
  const ul = $("#screenshot-list");
  const header = ul.closest(".card").querySelector("h3");
  if (header) {
    let c = header.querySelector(".count");
    if (!c) { header.insertAdjacentHTML("beforeend", '<span class="count">0</span>'); c = header.querySelector(".count"); }
    c.textContent = state.screenshots.length;
  }
  if (!state.screenshots.length) {
    ul.innerHTML = '<li class="dim">Inga skärmdumpar.</li>';
    return;
  }
  const sorted = [...state.screenshots].sort((a, b) => a.t - b.t);
  ul.innerHTML = sorted.map((s) => `
    <li data-id="${esc(s.annotation_id)}">
      <span class="label">${esc(s.label)}</span>
      <span class="meta">${fmtT(s.t)}${s.png_filename ? " · png" : ""}</span>
      ${s.note ? `<span class="note">${esc(s.note)}</span>` : ""}
      <button class="del" data-id="${esc(s.annotation_id)}">Ta bort</button>
    </li>
  `).join("");
  ul.querySelectorAll("li[data-id]").forEach((li) => {
    li.onclick = (e) => {
      if (e.target.classList.contains("del")) return;
      const id = li.dataset.id;
      const s = state.screenshots.find((x) => x.annotation_id === id);
      if (s) { video.currentTime = s.t; }
    };
  });
  ul.querySelectorAll("button.del").forEach((btn) => {
    btn.onclick = async (e) => {
      e.stopPropagation();
      const id = btn.dataset.id;
      const r = await fetch(`/api/runs/${state.runId}/screenshots/${id}`, { method: "DELETE" });
      if (r.ok) {
        state.screenshots = state.screenshots.filter((s) => s.annotation_id !== id);
        renderScreenshots();
        toast("Skärmdump borttagen", "success");
      } else { toast("Kunde inte ta bort skärmdump", "error"); }
    };
  });
}

$("#btn-screenshot").onclick = async () => {
  if (!state.runId) return;
  if (video.readyState < 2) { toast("Videon är inte redo", "error"); return; }
  syncCanvasSize();
  // Composite the current video frame + overlay canvas into an offscreen
  // canvas, export to PNG. This is the single annotated-frame renderer
  // (report §2.5) — there is no second server-side path.
  const off = document.createElement("canvas");
  off.width = video.videoWidth;
  off.height = video.videoHeight;
  const offCtx = off.getContext("2d");
  offCtx.drawImage(video, 0, 0, off.width, off.height);
  offCtx.drawImage(canvas, 0, 0, off.width, off.height);
  const blob = await new Promise((resolve) => off.toBlob(resolve, "image/png"));
  if (!blob) { toast("Kunde inte skapa PNG", "error"); return; }

  const label = `Skärmdump ${fmtT(video.currentTime)}`;
  const fd = new FormData();
  fd.append("t", String(video.currentTime));
  fd.append("label", label);
  fd.append("png", blob, "frame.png");
  const r = await fetch(`/api/runs/${state.runId}/screenshots`, { method: "POST", body: fd });
  if (r.ok) {
    const row = await r.json();
    state.screenshots.push(row);
    renderScreenshots();
    toast("Skärmdump sparad", "success");
    // Also trigger a client-side download so the reviewer keeps a local copy.
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = `${state.runId}_${row.annotation_id}.png`;
    a.click();
    URL.revokeObjectURL(url);
  } else { toast("Kunde inte spara skärmdump", "error"); }
};

// =====================================================================
// Export — CSV / JSON of the AI event log
// =====================================================================

$("#btn-export-csv").onclick = () => downloadExport("csv");
$("#btn-export-json").onclick = () => downloadExport("json");

async function downloadExport(fmt) {
  if (!state.runId) return;
  const r = await fetch(`/api/runs/${state.runId}/export?format=${fmt}`);
  if (!r.ok) { toast("Inga händelser att exportera", "error"); return; }
  const blob = await r.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${state.runId}_events.${fmt}`;
  a.click();
  URL.revokeObjectURL(url);
  toast(`Exporterade ${state.events.length} händelser (${fmt.toUpperCase()})`, "success");
}

// =====================================================================
// Phase 3: operator field-notes import
// =====================================================================

function renderOperatorWarnings(warnings) {
  const ul = $("#operator-warnings");
  if (!warnings || !warnings.length) {
    ul.classList.add("hidden");
    ul.innerHTML = "";
    return;
  }
  ul.classList.remove("hidden");
  ul.innerHTML = warnings
    .map((w) => `<li>Rad ${w.line}: ${esc(w.reason)} — "${esc(w.raw_line)}"</li>`)
    .join("");
}

function renderOperatorNotes() {
  const ul = $("#operator-note-list");
  if (!state.operatorNotes.length) {
    ul.innerHTML = '<li class="dim">Inga anteckningar importerade.</li>';
    renderTimeline();
    return;
  }
  const sorted = [...state.operatorNotes].sort((a, b) => a.t - b.t);
  ul.innerHTML = sorted.map((n) => `
    <li data-id="${esc(n.annotation_id)}">
      <span class="label">${esc(n.text)}</span>
      <span class="meta">${fmtT(n.t)}</span>
      <button class="del" data-id="${esc(n.annotation_id)}" title="Ta bort">Ta bort</button>
    </li>
  `).join("");
  ul.querySelectorAll("li[data-id]").forEach((li) => {
    li.onclick = (e) => {
      if (e.target.classList.contains("del")) return;
      const n = state.operatorNotes.find((x) => x.annotation_id === li.dataset.id);
      if (n) video.currentTime = n.t;
    };
  });
  ul.querySelectorAll("button.del").forEach((btn) => {
    btn.onclick = async (e) => {
      e.stopPropagation();
      const id = btn.dataset.id;
      const r = await fetch(`/api/runs/${state.runId}/operator-notes/${id}`, { method: "DELETE" });
      if (r.ok) {
        state.operatorNotes = state.operatorNotes.filter((n) => n.annotation_id !== id);
        renderOperatorNotes();
        toast("Anteckning borttagen", "success");
        refreshComparison();
      } else {
        toast("Kunde inte ta bort anteckning", "error");
      }
    };
  });
  renderTimeline();
}

$("#operator-import-form").onsubmit = async (e) => {
  e.preventDefault();
  if (!state.runId) return;
  const textarea = $("#operator-import-text");
  const text = textarea.value;
  if (!text.trim()) return;
  const r = await fetch(`/api/runs/${state.runId}/operator-notes/import`, {
    method: "POST",
    body: new URLSearchParams({ text }),
  });
  if (r.ok) {
    const body = await r.json();
    state.operatorNotes.push(...body.imported);
    renderOperatorNotes();
    renderOperatorWarnings(body.warnings);
    textarea.value = "";
    toast(`${body.imported.length} anteckningar importerade`, "success");
    refreshComparison();
  } else {
    toast("Kunde inte importera anteckningar", "error");
  }
};

// =====================================================================
// Phase 3: AI-vs-operator comparison + HTML debrief export
// =====================================================================

async function refreshComparison() {
  if (!state.runId) return;
  const tol = parseFloat($("#tolerance-input").value) || 60;
  try {
    const r = await fetch(`/api/runs/${state.runId}/comparison?tolerance_s=${tol}`);
    if (!r.ok) { renderComparison(null); return; }
    renderComparison(await r.json());
  } catch (_) {
    renderComparison(null);
  }
}

function renderComparison(cmp) {
  const ul = $("#compare-list");
  if (!cmp) {
    $("#cmp-both").textContent = "–";
    $("#cmp-ai-only").textContent = "–";
    $("#cmp-op-only").textContent = "–";
    ul.innerHTML = '<li class="dim">Ingen jämförelse ännu.</li>';
    return;
  }
  $("#cmp-both").textContent = cmp.counts.both;
  $("#cmp-ai-only").textContent = cmp.counts.ai_only;
  $("#cmp-op-only").textContent = cmp.counts.operator_only;

  const rows = [];
  cmp.both.forEach((m) => rows.push({
    t: m.event.t_start,
    html: `<li>
      <span class="label">hittad av båda · ${esc(CATEGORY_LABEL[m.event.category] || m.event.category)}</span>
      <span class="meta">AI ${fmtT(m.event.t_start)} · operatör ${fmtT(m.note.t)} · Δ ${m.delta_s.toFixed(1)}s</span>
      <span class="note">${esc(m.note.text)}</span>
    </li>`,
  }));
  cmp.ai_only.forEach((ev) => rows.push({
    t: ev.t_start,
    html: `<li>
      <span class="label">endast AI · ${esc(CATEGORY_LABEL[ev.category] || ev.category)}</span>
      <span class="meta">${fmtT(ev.t_start)} · v ${ev.confidence.toFixed(2)}</span>
    </li>`,
  }));
  cmp.operator_only.forEach((n) => rows.push({
    t: n.t,
    html: `<li>
      <span class="label">endast operatör</span>
      <span class="meta">${fmtT(n.t)}</span>
      <span class="note">${esc(n.text)}</span>
    </li>`,
  }));
  rows.sort((a, b) => a.t - b.t);
  ul.innerHTML = rows.length ? rows.map((r) => r.html).join("") : '<li class="dim">Ingen jämförelse ännu.</li>';
}

$("#btn-refresh-comparison").onclick = refreshComparison;

$("#btn-export-debrief").onclick = () => {
  if (!state.runId) return;
  const tol = parseFloat($("#tolerance-input").value) || 60;
  window.location.href = `/api/runs/${state.runId}/debrief?tolerance_s=${tol}`;
};

// =====================================================================
// Phase 5: person dossier + manual split/merge (report §5.5, §3.3-4)
// =====================================================================

const PERSON_STATE_LABEL = {
  confirmed: "bekräftad",
  transient: "transient",
  manual: "manuellt korrigerad",
};

async function refreshPersons() {
  state.persons = [];
  state.personsCount = null;
  state.personsUniqueCount = 0;
  state.personsEngineUncertainty = null;
  state.corrections = [];
  state.correctionsSkipped = [];
  if (state.runId) {
    try {
      // Both endpoints are ungated reads: /persons predates Phase 5 and the
      // corrections list stays visible even with the dossier toggle off
      // (recorded human work is never hidden by a config flag).
      const [pr, cr] = await Promise.all([
        fetch(`/api/runs/${state.runId}/persons`),
        fetch(`/api/runs/${state.runId}/identity-corrections`),
      ]);
      if (pr.ok) {
        const pj = await pr.json();
        state.persons = pj.persons || [];
        state.personsCount = pj.count ?? state.persons.length;
        state.personsUniqueCount = pj.unique_count ?? 0;
        state.personsEngineUncertainty = pj.engine_uncertainty || null;
      }
      if (cr.ok) {
        const j = await cr.json();
        state.corrections = j.corrections || [];
        state.correctionsSkipped = j.skipped || [];
      }
    } catch (_) { /* run without P3 → 409; pane shows its empty state */ }
  }
  renderPersons();
  renderCorrections();
  updateStats();
}

function renderPersons() {
  const ul = $("#person-list");
  $("#persons-legend").textContent = state.persons.some((p) => p.corrected)
    ? "korrigeringar tillämpade"
    : "";
  if (!state.persons.length) {
    ul.innerHTML = '<li class="dim">Inga personer (identitetspasset har inte körts).</li>';
    return;
  }
  ul.innerHTML = state.persons.map((p) => {
    const cls = state.selectedPersonId === p.person_id ? "active" : "";
    const stateLabel = PERSON_STATE_LABEL[p.confirmation_state] || p.confirmation_state;
    return `<li class="${cls}" data-pid="${p.person_id}">
      <span class="label">P${p.person_id}${p.corrected ? " ✎" : ""}</span>
      <span class="meta">${fmtT(p.first_seen)}–${fmtT(p.last_seen)} · ${p.tracklet_ids.length} spår · ${esc(stateLabel)}</span>
    </li>`;
  }).join("");
  ul.querySelectorAll("li[data-pid]").forEach((li) => {
    li.onclick = () => selectPerson(parseInt(li.dataset.pid, 10));
  });
}

async function selectPerson(pid) {
  state.selectedPersonId = pid;
  renderPersons();
  const p = state.persons.find((x) => x.person_id === pid);
  const detail = $("#person-detail");
  if (!p) { detail.classList.add("hidden"); return; }
  detail.classList.remove("hidden");
  $("#pd-title").textContent = `Person P${pid}`;
  const stateLabel = PERSON_STATE_LABEL[p.confirmation_state] || p.confirmation_state;
  const methods = Object.keys(p.embedding_centroids || {}).join(", ") || "–";
  $("#pd-meta").innerHTML = `
    <span>Synlig: ${esc(fmtT(p.first_seen))}–${esc(fmtT(p.last_seen))}</span>
    <span>Status: ${esc(stateLabel)}</span>
    <span>Spår: ${p.tracklet_ids.map((t) => `T${t}`).join(", ")}</span>
    <span>Utseendemetod: ${esc(methods)}</span>`;

  renderPersonHeatmapControl(pid);
  renderPersonEvents(pid);
  renderPersonAudit(p);
  renderMergeForm(pid);
  renderSplitForm(p);
  // Trajectory + crops fetch/seek in the background; the pane is usable
  // meanwhile.
  renderPersonTrajectory(pid);
  renderPersonCrops(pid);
}

function renderPersonEvents(pid) {
  const ul = $("#pd-events");
  const evs = state.events.filter((e) => e.person_id === pid);
  if (!evs.length) { ul.innerHTML = '<li class="dim">Inga händelser för personen.</li>'; return; }
  ul.innerHTML = evs.map((ev) => `
    <li data-event-id="${esc(ev.event_id)}">
      <span class="label"><span class="cat-tag cat-${ev.category}">${CATEGORY_LABEL[ev.category] || ev.category}</span></span>
      <span class="meta">${fmtT(ev.t_start)}–${fmtT(ev.t_end)} · v ${ev.confidence.toFixed(2)}</span>
    </li>`).join("");
  ul.querySelectorAll("li[data-event-id]").forEach((li) => {
    li.onclick = () => jumpToEvent(li.dataset.eventId);
  });
}

function renderPersonAudit(p) {
  const ul = $("#pd-audit");
  const audit = p.assoc_audit || [];
  if (!audit.length) {
    ul.innerHTML = '<li class="dim">Inget associationsspår (personen är ett enda spår).</li>';
    return;
  }
  ul.innerHTML = audit.map((a) => {
    const merged = a.rule === "merged";
    const what = merged ? "sammanslagen" : `blockerad (${esc(a.rule ? a.rule.replace("blocked:", "") : "?")})`;
    const sim = a.appearance_sim != null ? ` · likhet ${a.appearance_sim.toFixed(3)} (${esc(a.method || "?")})` : "";
    const gap = a.gap_s != null ? ` · lucka ${a.gap_s.toFixed(1)}s` : "";
    return `<li>
      <span class="label">T${a.tracklet_a} + T${a.tracklet_b}: ${what}</span>
      <span class="meta">${sim}${gap}</span>
    </li>`;
  }).join("");
}

async function renderPersonTrajectory(pid) {
  const holder = $("#pd-trajectory");
  holder.innerHTML = '<span class="dim">Laddar rörelsebana …</span>';
  try {
    const r = await fetch(`/api/runs/${state.runId}/persons/${pid}/trajectory`);
    if (!r.ok) { holder.innerHTML = '<span class="dim">Rörelsebanan kunde inte hämtas.</span>'; return; }
    const j = await r.json();
    if (state.selectedPersonId !== pid) return; // selection changed mid-fetch
    const pts = j.points || [];
    if (pts.length < 2) { holder.innerHTML = '<span class="dim">För få punkter för en bana.</span>'; return; }
    const W = video.videoWidth || Math.max(...pts.map((p) => p.x)) + 20;
    const H = video.videoHeight || Math.max(...pts.map((p) => p.y)) + 20;
    const poly = pts.map((p) => `${p.x.toFixed(0)},${p.y.toFixed(0)}`).join(" ");
    const a = pts[0], b = pts[pts.length - 1];
    holder.innerHTML = `
      <svg viewBox="0 0 ${W} ${H}" class="pd-traj-svg" preserveAspectRatio="xMidYMid meet">
        <rect x="0" y="0" width="${W}" height="${H}" class="pd-traj-frame"></rect>
        <polyline points="${poly}" class="pd-traj-line"></polyline>
        <circle cx="${a.x}" cy="${a.y}" r="${Math.max(4, W / 120)}" class="pd-traj-start"><title>Start ${fmtT(a.t)}</title></circle>
        <circle cx="${b.x}" cy="${b.y}" r="${Math.max(4, W / 120)}" class="pd-traj-end"><title>Slut ${fmtT(b.t)}</title></circle>
      </svg>`;
  } catch (_) {
    holder.innerHTML = '<span class="dim">Rörelsebanan kunde inte hämtas.</span>';
  }
}

// ---- appearance crops (client-rendered, report §2.5 single-renderer rule) ----

function seekOnce(v, t) {
  return new Promise((resolve) => {
    const onSeek = () => { v.removeEventListener("seeked", onSeek); resolve(true); };
    v.addEventListener("seeked", onSeek);
    v.currentTime = t;
    setTimeout(() => { v.removeEventListener("seeked", onSeek); resolve(false); }, 3000);
  });
}

async function renderPersonCrops(pid) {
  const holder = $("#pd-crops");
  const cropVideo = $("#crop-video");
  if (!cropVideo.src || cropVideo.readyState === 0) {
    holder.innerHTML = '<span class="dim">Videofilen saknas — inga bildutsnitt.</span>';
    return;
  }
  holder.innerHTML = '<span class="dim">Laddar bildutsnitt …</span>';
  try {
    // Sample up to 5 moments spread over the person's visible span (via the
    // trajectory endpoint), fetch that frame's boxes, and crop the matching
    // box out of a hidden second <video> element — the browser is still the
    // only renderer of annotated pixels.
    const r = await fetch(`/api/runs/${state.runId}/persons/${pid}/trajectory?max_points=200`);
    if (!r.ok) { holder.innerHTML = '<span class="dim">Kunde inte hämta utsnittspunkter.</span>'; return; }
    const pts = (await r.json()).points || [];
    if (!pts.length) { holder.innerHTML = '<span class="dim">Inga utsnittspunkter.</span>'; return; }
    const n = Math.min(5, pts.length);
    const samples = [];
    for (let i = 0; i < n; i++) samples.push(pts[Math.floor((i * (pts.length - 1)) / Math.max(1, n - 1))]);

    holder.innerHTML = "";
    for (const s of samples) {
      if (state.selectedPersonId !== pid) return;
      const boxesRes = await fetch(`/api/runs/${state.runId}/tracklets?frame=${s.frame_no}`).then((x) => x.json()).catch(() => null);
      const box = boxesRes && (boxesRes.boxes || []).find((b) => b.tracklet_id === s.tracklet_id);
      if (!box) continue;
      const ok = await seekOnce(cropVideo, s.t);
      if (!ok || state.selectedPersonId !== pid) continue;
      const [x0, y0, x1, y1] = box.xyxy;
      const padX = (x1 - x0) * 0.25, padY = (y1 - y0) * 0.15;
      const sx = Math.max(0, x0 - padX), sy = Math.max(0, y0 - padY);
      const sw = Math.min(cropVideo.videoWidth - sx, (x1 - x0) + padX * 2);
      const sh = Math.min(cropVideo.videoHeight - sy, (y1 - y0) + padY * 2);
      if (sw <= 0 || sh <= 0) continue;
      const c = document.createElement("canvas");
      const dispH = 110;
      c.height = dispH;
      c.width = Math.max(24, Math.round((sw / sh) * dispH));
      c.getContext("2d").drawImage(cropVideo, sx, sy, sw, sh, 0, 0, c.width, c.height);
      const wrap = document.createElement("div");
      wrap.className = "pd-crop";
      wrap.appendChild(c);
      const cap = document.createElement("small");
      cap.textContent = fmtT(s.t);
      wrap.appendChild(cap);
      wrap.onclick = () => { video.currentTime = s.t; };
      holder.appendChild(wrap);
    }
    if (!holder.children.length) holder.innerHTML = '<span class="dim">Inga bildutsnitt kunde skapas.</span>';
  } catch (_) {
    holder.innerHTML = '<span class="dim">Bildutsnitten kunde inte skapas.</span>';
  }
}

// ---- split/merge forms ----

function renderMergeForm(pid) {
  const sel = $("#pd-merge-target");
  const others = state.persons.filter((p) => p.person_id !== pid);
  sel.innerHTML = others.length
    ? others.map((p) => `<option value="${p.person_id}">P${p.person_id} (${fmtT(p.first_seen)}–${fmtT(p.last_seen)})</option>`).join("")
    : '<option value="">— inga andra personer —</option>';
  $("#pd-merge-form").onsubmit = async (e) => {
    e.preventDefault();
    const other = sel.value;
    if (!other) return;
    await postCorrection({
      op: "merge",
      person_ids: `${pid},${other}`,
      reason: $("#pd-merge-reason").value.trim() || "",
    });
  };
}

function renderSplitForm(p) {
  const holder = $("#pd-split-tracklets");
  if ((p.tracklet_ids || []).length < 2) {
    holder.innerHTML = '<span class="dim">Personen består av ett enda spår — inget att dela.</span>';
    $("#pd-split-form").onsubmit = (e) => e.preventDefault();
    return;
  }
  holder.innerHTML = p.tracklet_ids.map((t) =>
    `<label class="pd-split-choice"><input type="checkbox" value="${t}"> T${t}</label>`
  ).join("");
  $("#pd-split-form").onsubmit = async (e) => {
    e.preventDefault();
    const chosen = [...holder.querySelectorAll("input:checked")].map((i) => i.value);
    if (!chosen.length) { toast("Markera minst ett spår att dela ut", "error"); return; }
    if (chosen.length === p.tracklet_ids.length) {
      toast("Kan inte dela ut samtliga spår", "error");
      return;
    }
    await postCorrection({
      op: "split",
      person_id: String(p.person_id),
      tracklet_ids: chosen.join(","),
      reason: $("#pd-split-reason").value.trim() || "",
    });
  };
}

async function postCorrection(fields) {
  const r = await fetch(`/api/runs/${state.runId}/identity-corrections`, {
    method: "POST",
    body: new URLSearchParams(fields),
  });
  if (!r.ok) {
    let detail = "Korrigeringen avvisades";
    try { detail = (await r.json()).detail || detail; } catch (_) {}
    toast(detail, "error");
    return;
  }
  const body = await r.json();
  toast(body.warning || "Korrigering sparad", body.warning ? "info" : "success");
  await reloadAfterCorrectionChange();
}

async function reloadAfterCorrectionChange() {
  // A correction changes the tracklet→person join everywhere: person list,
  // overlay labels (cached per frame), served events' person_id, per-person
  // heatmap. Drop the caches and re-fetch, keeping the heatmap filter the
  // reviewer had chosen when that person still exists.
  const heatPerson = state.heatmap.personId;
  state.boxesByFrame = null;
  state.heatmap = { canvas: null, loading: false, personId: null };
  const evRes = await fetch(`/api/runs/${state.runId}/events`).then((r) => r.json()).catch(() => null);
  if (evRes) state.events = evRes.events || [];
  await refreshPersons();
  const still = state.persons.find((p) => p.person_id === state.selectedPersonId);
  if (still) selectPerson(still.person_id);
  else { state.selectedPersonId = null; $("#person-detail").classList.add("hidden"); }
  renderEvents();
  lastDrawnFrame = null;
  if (state.layers.heatmap) {
    const stillThere = heatPerson != null && state.persons.some((p) => p.person_id === heatPerson);
    ensureHeatmap(stillThere ? heatPerson : null);
  }
}

function renderCorrections() {
  const ul = $("#correction-list");
  const skippedById = {};
  state.correctionsSkipped.forEach((s) => { skippedById[s.annotation_id] = s.reason; });
  if (!state.corrections.length) {
    ul.innerHTML = '<li class="dim">Inga korrigeringar.</li>';
    return;
  }
  ul.innerHTML = state.corrections.map((c) => {
    const what = c.op === "merge"
      ? `Slå ihop ${(c.person_ids || []).map((p) => `P${p}`).join(" + ")}`
      : `Dela P${c.person_id}: ut ${(c.tracklet_ids || []).map((t) => `T${t}`).join(", ")}`;
    const skipped = skippedById[c.annotation_id];
    return `<li>
      <span class="label">${esc(what)}</span>
      ${c.reason ? `<span class="note">${esc(c.reason)}</span>` : ""}
      ${skipped ? `<span class="note warn">Tillämpas inte: ${esc(skipped)}</span>` : ""}
      <button class="del" data-id="${esc(c.annotation_id)}">Ångra</button>
    </li>`;
  }).join("");
  ul.querySelectorAll("button.del").forEach((btn) => {
    btn.onclick = async () => {
      const r = await fetch(`/api/runs/${state.runId}/identity-corrections/${btn.dataset.id}`, { method: "DELETE" });
      if (r.ok) { toast("Korrigering ångrad", "success"); await reloadAfterCorrectionChange(); }
      else toast("Kunde inte ångra korrigeringen", "error");
    };
  });
}

// =====================================================================
// Phase 5: ground truth "facit" (report §5.6)
// =====================================================================

async function refreshGroundTruth() {
  state.gtEntries = [];
  if (state.runId && state.features.ground_truth) {
    try {
      const r = await fetch(`/api/runs/${state.runId}/ground-truth`);
      if (r.ok) state.gtEntries = (await r.json()).entries || [];
    } catch (_) {}
  }
  renderGroundTruth();
  refreshGtScore();
}

function renderGroundTruth() {
  const ul = $("#gt-list");
  if (!state.gtEntries.length) {
    ul.innerHTML = '<li class="dim">Inget facit inlagt.</li>';
    return;
  }
  const sorted = [...state.gtEntries].sort((a, b) => a.t - b.t);
  ul.innerHTML = sorted.map((g) => `
    <li data-id="${esc(g.annotation_id)}">
      <span class="label">${esc(g.text)}</span>
      <span class="meta">${fmtT(g.t)}</span>
      <button class="del" data-id="${esc(g.annotation_id)}">Ta bort</button>
    </li>`).join("");
  ul.querySelectorAll("li[data-id]").forEach((li) => {
    li.onclick = (e) => {
      if (e.target.classList.contains("del")) return;
      const g = state.gtEntries.find((x) => x.annotation_id === li.dataset.id);
      if (g) video.currentTime = g.t;
    };
  });
  ul.querySelectorAll("button.del").forEach((btn) => {
    btn.onclick = async (e) => {
      e.stopPropagation();
      const r = await fetch(`/api/runs/${state.runId}/ground-truth/${btn.dataset.id}`, { method: "DELETE" });
      if (r.ok) {
        state.gtEntries = state.gtEntries.filter((g) => g.annotation_id !== btn.dataset.id);
        renderGroundTruth();
        refreshGtScore();
        toast("Facitpost borttagen", "success");
      } else toast("Kunde inte ta bort facitposten", "error");
    };
  });
}

$("#gt-import-form").onsubmit = async (e) => {
  e.preventDefault();
  if (!state.runId) return;
  const textarea = $("#gt-import-text");
  if (!textarea.value.trim()) return;
  const r = await fetch(`/api/runs/${state.runId}/ground-truth/import`, {
    method: "POST",
    body: new URLSearchParams({ text: textarea.value }),
  });
  if (!r.ok) { toast("Kunde inte importera facit", "error"); return; }
  const body = await r.json();
  state.gtEntries.push(...body.imported);
  const wUl = $("#gt-warnings");
  if (body.warnings.length) {
    wUl.classList.remove("hidden");
    wUl.innerHTML = body.warnings.map((w) => `<li>Rad ${w.line}: ${esc(w.reason)} — "${esc(w.raw_line)}"</li>`).join("");
  } else { wUl.classList.add("hidden"); wUl.innerHTML = ""; }
  textarea.value = "";
  toast(`${body.imported.length} facitposter importerade`, "success");
  renderGroundTruth();
  refreshGtScore();
};

async function refreshGtScore() {
  if (!state.runId || !state.features.ground_truth) return;
  const tol = parseFloat($("#gt-tolerance-input").value) || 60;
  try {
    const r = await fetch(`/api/runs/${state.runId}/ground-truth/score?tolerance_s=${tol}`);
    renderGtScore(r.ok ? await r.json() : null);
  } catch (_) { renderGtScore(null); }
}

function renderGtScore(score) {
  const ul = $("#gt-score-list");
  if (!score || !score.counts.gt_total) {
    $("#gt-ai-found").textContent = "–";
    $("#gt-op-found").textContent = "–";
    $("#gt-total").textContent = score ? score.counts.gt_total : "–";
    ul.innerHTML = '<li class="dim">Ingen poängsättning ännu — importera facit först.</li>';
    return;
  }
  const c = score.counts;
  $("#gt-ai-found").textContent = `${c.ai_found}/${c.gt_total}`;
  $("#gt-op-found").textContent = `${c.operator_found}/${c.gt_total}`;
  $("#gt-total").textContent = c.gt_total;
  const rows = score.entries.map((e) => {
    const ai = e.ai
      ? `AI ✓ (Δ ${e.ai.delta_s > 0 ? "+" : ""}${e.ai.delta_s.toFixed(0)}s)`
      : "AI –";
    const op = e.operator
      ? `Operatör ✓ (Δ ${e.operator.delta_s > 0 ? "+" : ""}${e.operator.delta_s.toFixed(0)}s)`
      : "Operatör –";
    return `<li data-t="${e.gt.t}">
      <span class="label">${esc(e.gt.text)}</span>
      <span class="meta">${fmtT(e.gt.t)} · ${ai} · ${op}</span>
    </li>`;
  });
  if (c.ai_unmatched) {
    rows.push(`<li class="dim">${c.ai_unmatched} AI-händelse(r) utan motsvarande facitpost.</li>`);
  }
  if (c.operator_unmatched) {
    rows.push(`<li class="dim">${c.operator_unmatched} operatörsanteckning(ar) utan motsvarande facitpost.</li>`);
  }
  ul.innerHTML = rows.join("");
  ul.querySelectorAll("li[data-t]").forEach((li) => {
    li.onclick = () => { video.currentTime = parseFloat(li.dataset.t); };
  });
}

$("#btn-refresh-gt").onclick = refreshGtScore;

// =====================================================================
// Phase 5: multi-config run comparison (report §5.7)
// =====================================================================

async function renderRunCompareOptions() {
  const sel = $("#rc-other-run");
  if (!state.runId || !state.features.run_compare) return;
  try {
    const j = await fetch("/api/runs").then((r) => r.json());
    // Only runs of the SAME video are comparable (the endpoint enforces it
    // by hash; the picker simply doesn't offer meaningless pairs).
    const others = (j.runs || []).filter(
      (r) => r.run_id !== state.runId && r.video_hash === state.runSummary?.video_hash
    );
    sel.innerHTML = others.length
      ? others.map((r) => `<option value="${esc(r.run_id)}">${esc(r.run_id)} · ${esc(r.created_at || "")}</option>`).join("")
      : '<option value="">— inga andra körningar av samma film —</option>';
  } catch (_) {
    sel.innerHTML = '<option value="">— kunde inte hämta körningar —</option>';
  }
}

$("#btn-run-compare").onclick = async () => {
  const other = $("#rc-other-run").value;
  if (!state.runId || !other) { toast("Ingen annan körning att jämföra med", "info"); return; }
  const tol = parseFloat($("#rc-tolerance-input").value) || 10;
  const r = await fetch(`/api/runs/${state.runId}/compare/${other}?tolerance_s=${tol}`);
  if (!r.ok) {
    let detail = "Jämförelsen misslyckades";
    try { detail = (await r.json()).detail || detail; } catch (_) {}
    toast(detail, "error");
    return;
  }
  renderRunCompare(await r.json());
};

function renderRunCompare(result) {
  $("#rc-result").classList.remove("hidden");
  $("#rc-both").textContent = result.counts.both;
  $("#rc-only-a").textContent = result.counts.only_a;
  $("#rc-only-b").textContent = result.counts.only_b;

  const cfgUl = $("#rc-config-diff");
  const keys = Object.keys(result.config_diff);
  cfgUl.innerHTML = keys.length
    ? keys.map((k) => {
        const v = result.config_diff[k];
        return `<li><span class="label">${esc(k)}</span><span class="meta">denna: ${esc(JSON.stringify(v.a))} · andra: ${esc(JSON.stringify(v.b))}</span></li>`;
      }).join("")
    : '<li class="dim">Identisk konfiguration (samma inställningar).</li>';

  const sa = result.stats.a, sb = result.stats.b;
  $("#rc-pass-stats").innerHTML = [
    ["Detektioner", sa.detections, sb.detections],
    ["Tracklet-rader", sa.tracklet_rows, sb.tracklet_rows],
    ["Personer", sa.persons, sb.persons],
    ["Händelser", sa.events, sb.events],
  ].map(([label, a, b]) =>
    `<li><span class="label">${label}</span><span class="meta">denna: ${a ?? "–"} · andra: ${b ?? "–"}</span></li>`
  ).join("");

  const evUl = $("#rc-events");
  const rows = [];
  result.both.forEach((m) => rows.push({
    t: m.a.t_start,
    html: `<li data-t="${m.a.t_start}">
      <span class="label">gemensam · <span class="cat-tag cat-${m.a.category}">${CATEGORY_LABEL[m.a.category] || m.a.category}</span></span>
      <span class="meta">${fmtT(m.a.t_start)} · Δ ${m.delta_s > 0 ? "+" : ""}${m.delta_s.toFixed(1)}s</span>
    </li>`,
  }));
  result.only_a.forEach((e) => rows.push({
    t: e.t_start,
    html: `<li data-t="${e.t_start}">
      <span class="label">endast denna · <span class="cat-tag cat-${e.category}">${CATEGORY_LABEL[e.category] || e.category}</span></span>
      <span class="meta">${fmtT(e.t_start)} · v ${e.confidence.toFixed(2)}</span>
    </li>`,
  }));
  result.only_b.forEach((e) => rows.push({
    t: e.t_start,
    html: `<li data-t="${e.t_start}">
      <span class="label">endast andra · <span class="cat-tag cat-${e.category}">${CATEGORY_LABEL[e.category] || e.category}</span></span>
      <span class="meta">${fmtT(e.t_start)} · v ${e.confidence.toFixed(2)}</span>
    </li>`,
  }));
  rows.sort((a, b) => a.t - b.t);
  evUl.innerHTML = rows.length ? rows.map((r) => r.html).join("") : '<li class="dim">Inga händelser i någon av körningarna.</li>';
  evUl.querySelectorAll("li[data-t]").forEach((li) => {
    li.onclick = () => { video.currentTime = parseFloat(li.dataset.t); };
  });
}

// =====================================================================
// Header stats
// =====================================================================

function updateStats() {
  $("#st-events").querySelector("b").textContent = state.events.length;
  // The headline is "unika personer", so it uses the served unique_count
  // (confirmed + manually corrected, transients excluded) rather than the
  // full projected list — a transient track is not an established person.
  // Falls back to the engine stat only before /persons has loaded.
  const p3 = state.runSummary?.passes?.["p3_identity"];
  const personStat = $("#st-persons");
  personStat.querySelector("b").textContent = state.personsCount !== null
    ? state.personsUniqueCount
    : (p3?.stats?.confirmed_persons ?? p3?.stats?.persons_out ?? "–");
  const uncertainty = state.personsEngineUncertainty;
  const totalCountEl = personStat.querySelector(".person-count");
  const uncertaintyEl = personStat.querySelector(".engine-uncertainty");
  const uncertainMerges = uncertainty?.uncertain_merges ?? 0;
  totalCountEl.classList.toggle("hidden", state.personsCount === null);
  totalCountEl.textContent = state.personsCount === null ? "" : `totalt: ${state.personsCount}`;
  uncertaintyEl.classList.toggle("hidden", !uncertainty);
  uncertaintyEl.textContent = uncertainty
    ? `motorn: ${uncertainMerges} osäkra sammanslagningar`
    : "";
  personStat.title = uncertainty
    ? `Unika personer: ${state.personsUniqueCount}. Totalt i den korrigerade projektionen: ${state.personsCount}. Motorns osäkerhetsband hör till körning ${uncertainty.run_id} och räknas inte om av manuella korrigeringar.`
    : "Unika personer";
  $("#st-bookmarks").querySelector("b").textContent = state.bookmarks.length;
}

// =====================================================================
// Boot
// =====================================================================

refreshRuns().catch(() => toast("Kunde inte hämta körningar", "error"));
refreshFeatures(); // Phase 5 toggle state → hide disabled tabs/chips/buttons
// Start the overlay loop — it self-schedules via requestVideoFrameCallback
// (or rAF fallback) and no-ops until a run is loaded.
drawOverlay();
