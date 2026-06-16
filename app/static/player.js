/* Offline analysis player. Streams the original video and overlays the
 * pre-computed annotations (shared Overlay renderer) synced by time, with
 * scrubbing, frame-stepping, reverse play and a clickable event timeline —
 * the after-action surface: step through and judge what the model saw. */

"use strict";

const video = document.getElementById("video");
const canvas = document.getElementById("view");
const ctx = canvas.getContext("2d");
const overlayMsg = document.getElementById("overlay-msg");

const layers = { boxes: true, ids: true, trails: false, status: true, hazards: true, base: true };
try { Object.assign(layers, JSON.parse(localStorage.getItem("layers") || "{}")); } catch (_) {}

let bundle = null; // { name, meta, frames, fvals, events }

/* ---------- loading ---------- */

const qsName = new URLSearchParams(location.search).get("a");

async function refreshAnalyses() {
  const sel = document.getElementById("analysis-select");
  const list = await fetch("/api/analyses").then(r => r.json()).catch(() => ({ analyses: [] }));
  const cur = bundle ? bundle.name : qsName;
  sel.innerHTML = list.analyses.length
    ? list.analyses.map(a => {
        const tag = a.status === "running" ? ` (kör ${a.pct ?? "?"}%)` : "";
        return `<option value="${a.name}" ${a.name === cur ? "selected" : ""}>${a.name}${tag}</option>`;
      }).join("")
    : '<option value="">— inga analyser —</option>';
  sel.onchange = () => loadBundle(sel.value);
  return list.analyses;
}

async function loadBundle(name) {
  if (!name) return;
  overlayMsg.textContent = "Laddar analys …";
  overlayMsg.classList.remove("hidden");
  try {
    const [meta, framesTxt, events] = await Promise.all([
      fetch(`/api/analysis/${name}`).then(r => r.json()),
      fetch(`/api/analysis/${name}/frames`).then(r => r.text()),
      fetch(`/api/analysis/${name}/events`).then(r => r.json()).catch(() => ({ events: [] })),
    ]);
    const frames = framesTxt.trim().split("\n").filter(Boolean).map(JSON.parse);
    frames.sort((a, b) => a.f - b.f);
    bundle = { name, meta, frames, fvals: frames.map(f => f.f), events: events.events || [] };
    video.src = `/api/analysis/${name}/video`;
    video.load();
    canvas.width = meta.wh[0]; canvas.height = meta.wh[1];
    document.getElementById("vwrap").style.aspectRatio = `${meta.wh[0]} / ${meta.wh[1]}`;
    renderEvents();
    overlayMsg.classList.add("hidden");
    history.replaceState(null, "", `?a=${encodeURIComponent(name)}`);
  } catch (e) {
    overlayMsg.textContent = "Kunde inte ladda analysen.";
    overlayMsg.classList.remove("hidden");
  }
}

/* ---------- frame lookup ---------- */

function metaAtTime(tsec) {
  if (!bundle || !bundle.frames.length) return null;
  const fno = Math.round(tsec * bundle.meta.fps);
  const fv = bundle.fvals;
  let lo = 0, hi = fv.length - 1, ans = 0;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (fv[mid] <= fno) { ans = mid; lo = mid + 1; } else { hi = mid - 1; }
  }
  return bundle.frames[ans];
}

/* ---------- render loop ---------- */

function draw() {
  requestAnimationFrame(draw);
  if (!bundle) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const meta = metaAtTime(video.currentTime);
  if (meta) { Overlay.draw(ctx, meta, layers); updateReadout(meta); }
  updateTransport();
}
requestAnimationFrame(draw);

/* ---------- transport ---------- */

const btnPlay = document.getElementById("btn-play");
const btnRev = document.getElementById("btn-rev");
const scrub = document.getElementById("scrub");
const timecode = document.getElementById("timecode");
const speedSel = document.getElementById("speed");

let reverse = false, revLast = 0;

function fmt(s) {
  s = Math.max(0, s | 0);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

function updateTransport() {
  if (!video.duration) return;
  if (document.activeElement !== scrub) scrub.value = Math.round((video.currentTime / video.duration) * 1000);
  timecode.textContent = `${fmt(video.currentTime)} / ${fmt(video.duration)}`;
  const playing = !video.paused && !reverse;
  btnPlay.textContent = playing ? "⏸" : "▶";
  btnPlay.classList.toggle("on", playing);
  btnRev.classList.toggle("on", reverse);
}

function stopReverse() { reverse = false; btnRev.classList.remove("on"); }

function startReverse() {
  video.pause();
  reverse = true; revLast = performance.now();
  btnRev.classList.add("on");
  requestAnimationFrame(reverseTick);
}

function reverseTick(ts) {
  if (!reverse) return;
  const dt = (ts - revLast) / 1000; revLast = ts;
  video.currentTime = Math.max(0, video.currentTime - parseFloat(speedSel.value) * dt);
  if (video.currentTime <= 0) { stopReverse(); }
  requestAnimationFrame(reverseTick);
}

btnPlay.onclick = () => {
  stopReverse();
  if (video.paused) video.play(); else video.pause();
};
btnRev.onclick = () => { if (reverse) stopReverse(); else startReverse(); };

function step(dir) {
  stopReverse(); video.pause();
  const dt = 1 / (bundle ? bundle.meta.fps : 25);
  video.currentTime = Math.min(video.duration || 0, Math.max(0, video.currentTime + dir * dt));
}
document.getElementById("btn-step-back").onclick = () => step(-1);
document.getElementById("btn-step-fwd").onclick = () => step(1);

scrub.addEventListener("input", () => {
  stopReverse();
  if (video.duration) video.currentTime = (scrub.value / 1000) * video.duration;
});
speedSel.onchange = () => { video.playbackRate = parseFloat(speedSel.value); };

document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (e.code === "Space") { e.preventDefault(); btnPlay.onclick(); }
  else if (e.code === "ArrowLeft") { e.preventDefault(); step(-1); }
  else if (e.code === "ArrowRight") { e.preventDefault(); step(1); }
});

/* ---------- readout + events ---------- */

const situationList = document.getElementById("situation-list");
const curStats = document.getElementById("cur-stats");
const eventList = document.getElementById("event-list");

function esc(s) {
  return String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function updateReadout(meta) {
  const s = meta.stats;
  curStats.textContent = `· synliga ${s.visible} · unika ${s.unique} · irrat. ${s.irr_now}`;
  const sit = [];
  if (meta.base) for (const r of meta.base.reasons) sit.push(`<li>${esc(r)}</li>`);
  if (meta.hazards.fire) sit.push(`<li class="alert">Brand indikerad (${(meta.hazards.fire.area * 100).toFixed(1)} %)</li>`);
  if (meta.hazards.smoke) sit.push(`<li class="note">Rök indikerad</li>`);
  for (const p of meta.persons) {
    if (p.st === "still") sit.push(`<li class="alert">P${p.pid} STILLA${p.prone ? " – LIGGER" : ""}</li>`);
    else if (p.st === "toward_danger") sit.push(`<li class="note">P${p.pid} mot fara</li>`);
  }
  situationList.innerHTML = sit.length ? sit.join("") : '<li class="dim">Inget att rapportera.</li>';
}

const SEV_CLASS = { alert: "alert", warn: "note", info: "dim" };

function renderEvents() {
  document.getElementById("ev-count").textContent = `(${bundle.events.length})`;
  if (!bundle.events.length) { eventList.innerHTML = '<li class="dim">Inga händelser.</li>'; return; }
  eventList.innerHTML = bundle.events.map((e, i) =>
    `<li class="${SEV_CLASS[e.sev] || ""}" data-i="${i}"><b>${fmt(e.t)}</b> ${esc(e.text)}</li>`
  ).join("");
  eventList.querySelectorAll("li[data-i]").forEach(li => {
    li.onclick = () => {
      stopReverse(); video.pause();
      video.currentTime = bundle.events[+li.dataset.i].t;
    };
  });
}

/* ---------- layer toggles ---------- */

document.querySelectorAll("#toggles .chip[data-layer]").forEach(btn => {
  const key = btn.dataset.layer;
  btn.classList.toggle("on", !!layers[key]);
  btn.onclick = () => {
    layers[key] = !layers[key];
    btn.classList.toggle("on", layers[key]);
    localStorage.setItem("layers", JSON.stringify(layers));
  };
});

const panel = document.getElementById("panel");
document.getElementById("btn-newpanel").onclick = function () {
  document.getElementById("new-card").scrollIntoView({ behavior: "smooth" });
};

/* ---------- create analysis ---------- */

async function refreshVideos() {
  const v = await fetch("/api/videos").then(r => r.json()).catch(() => ({ videos: [] }));
  document.getElementById("new-video").innerHTML = v.videos.length
    ? v.videos.map(x => `<option value="${esc(x.name)}">${esc(x.name)} (${x.size_mb} MB)</option>`).join("")
    : '<option value="">— inga filmer i videos/ —</option>';
}

document.getElementById("btn-run").onclick = async () => {
  const name = document.getElementById("new-video").value;
  const stride = parseInt(document.getElementById("new-stride").value) || 1;
  const status = document.getElementById("run-status");
  if (!name) { status.textContent = "Ingen film vald."; return; }
  status.textContent = "Startar analys …";
  const res = await fetch("/api/analyze", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, stride }),
  });
  const j = await res.json().catch(() => ({}));
  if (!res.ok) { status.textContent = j.detail || "Kunde inte starta."; return; }
  pollRun(j.name, status);
};

async function pollRun(name, status) {
  const st = await fetch(`/api/analysis/${name}/state`).then(r => r.json()).catch(() => null);
  if (!st) { status.textContent = "Väntar …"; setTimeout(() => pollRun(name, status), 1500); return; }
  if (st.status === "done") {
    status.textContent = "Klart.";
    await refreshAnalyses();
    loadBundle(name);
    return;
  }
  status.textContent = `Analyserar … ${st.pct ?? 0}% (${st.done}/${st.total || "?"})`;
  setTimeout(() => pollRun(name, status), 1500);
}

/* ---------- boot ---------- */

(async function () {
  await Promise.all([refreshAnalyses(), refreshVideos()]);
  const analyses = await refreshAnalyses();
  const pick = qsName || (analyses[0] && analyses[0].name);
  if (pick) loadBundle(pick);
})();
