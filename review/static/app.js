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
  layers: { boxes: true, ids: true, status: true, trails: false },
  _reviewPauseHandler: null, // current jumpToEvent auto-pause listener, if any
  hazardMarker: { active: false, x: null, y: null }, // Phase 4 retroactive marker
  hazardPlacementArmed: false, // true while waiting for the next canvas click
};

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
    state.layers[key] = !state.layers[key];
    btn.classList.toggle("on", state.layers[key]);
    lastDrawnFrame = null; // force redraw
  };
});

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
// Header stats
// =====================================================================

function updateStats() {
  $("#st-events").querySelector("b").textContent = state.events.length;
  const p3 = state.runSummary?.passes?.["p3_identity"];
  $("#st-persons").querySelector("b").textContent = p3?.stats?.confirmed_persons ?? p3?.stats?.persons_out ?? "–";
  $("#st-bookmarks").querySelector("b").textContent = state.bookmarks.length;
}

// =====================================================================
// Boot
// =====================================================================

refreshRuns().catch(() => toast("Kunde inte hämta körningar", "error"));
// Start the overlay loop — it self-schedules via requestVideoFrameCallback
// (or rAF fallback) and no-ops until a run is loaded.
drawOverlay();
