/* Insatsdrönare PoC — klient.
 * Tar emot binära WS-paket [u32 metalängd][meta-JSON][JPEG], ritar bilden på
 * canvas och lagren ovanpå. Togglar är rena klientval. */

"use strict";

const canvas = document.getElementById("view");
const ctx = canvas.getContext("2d");
const overlayMsg = document.getElementById("overlay-msg");
const threatBanner = document.getElementById("threat-banner");
const dangerHint = document.getElementById("danger-hint");

const COLORS = {
  ok: "#2ecc71",
  still: "#ff4757",
  toward_danger: "#ffa502",
  threat: "#ff3838",
  base: "#34c3ff",
  danger: "#ff4757",
  smoke: "#aab4be",
  fire: "#ff6b35",
};
const STATUS_TEXT = { still: "STILLA", toward_danger: "MOT FARA" };

const layers = { boxes: true, ids: true, trails: false, status: true, threats: true, hazards: true, base: true };
try {
  Object.assign(layers, JSON.parse(localStorage.getItem("layers") || "{}"));
} catch (_) {}

let latest = null;      // { meta, bitmap }
let dangerArmed = false;
let lastPacketAt = 0;

/* ---------- WebSocket ---------- */

let ws = null;
let retryMs = 500;

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws/stream`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => { retryMs = 500; };
  ws.onmessage = async (ev) => {
    const buf = ev.data;
    const view = new DataView(buf);
    const metaLen = view.getUint32(0);
    const meta = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, metaLen)));
    const jpeg = new Blob([new Uint8Array(buf, 4 + metaLen)], { type: "image/jpeg" });
    try {
      const bitmap = await createImageBitmap(jpeg);
      if (latest && latest.bitmap) latest.bitmap.close();
      latest = { meta, bitmap };
      lastPacketAt = performance.now();
    } catch (_) { /* skip broken frame */ }
  };
  ws.onclose = () => {
    setTimeout(connect, retryMs);
    retryMs = Math.min(retryMs * 2, 8000);
  };
  ws.onerror = () => ws.close();
}
connect();

/* ---------- Render loop ---------- */

let drawnFrame = null;

function draw() {
  requestAnimationFrame(draw);
  if (!latest) return;
  const { meta, bitmap } = latest;
  if (drawnFrame === latest) return;
  drawnFrame = latest;

  if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
  }
  ctx.drawImage(bitmap, 0, 0);

  const W = canvas.width, H = canvas.height;
  const lw = Math.max(2, W / 480);
  ctx.font = `bold ${Math.max(11, W / 60)}px system-ui, sans-serif`;
  ctx.textBaseline = "bottom";

  if (layers.hazards) drawHazards(meta, W, H, lw);
  if (layers.trails) for (const p of meta.persons) drawTrail(p, W, H, lw);
  if (layers.boxes) for (const p of meta.persons) drawPerson(p, W, H, lw);
  if (layers.threats) for (const t of meta.threats) drawThreat(t, W, H, lw);
  if (meta.danger) drawDanger(meta.danger, W, H, lw);
  if (layers.base && meta.base) drawBase(meta.base, W, H, lw);

  updateHud(meta);
}
requestAnimationFrame(draw);

setInterval(() => {
  if (!latest) return;
  if (performance.now() - lastPacketAt > 2500) {
    overlayMsg.textContent = "Väntar på videoström …";
    overlayMsg.classList.remove("hidden");
  }
}, 1000);

function drawPerson(p, W, H, lw) {
  const [x, y, w, h] = p.box;
  const status = layers.status ? p.st : "ok";
  const color = COLORS[status] || COLORS.ok;
  const X = x * W, Y = y * H, BW = w * W, BH = h * H;

  ctx.lineWidth = status === "ok" ? lw : lw * 1.6;
  ctx.strokeStyle = color;
  ctx.strokeRect(X, Y, BW, BH);

  if (!layers.ids && status === "ok") return;
  let label = layers.ids ? `P${p.pid}` : "";
  if (layers.status && STATUS_TEXT[p.st]) label += `${label ? " · " : ""}${STATUS_TEXT[p.st]}`;
  if (layers.status && p.prone && p.st === "still") label += " · LIGGER";
  if (!label) return;

  const tw = ctx.measureText(label).width;
  const th = parseInt(ctx.font, 10) + 6;
  const ly = Y > th ? Y : Y + BH + th;
  ctx.fillStyle = status === "ok" ? "rgba(10,14,18,.75)" : color;
  ctx.fillRect(X - lw / 2, ly - th, tw + 10, th);
  ctx.fillStyle = status === "ok" ? color : "#0c1014";
  ctx.fillText(label, X + 4, ly - 3);
}

function drawTrail(p, W, H, lw) {
  if (!p.trail || p.trail.length < 2) return;
  ctx.lineWidth = lw;
  ctx.strokeStyle = (COLORS[layers.status ? p.st : "ok"] || COLORS.ok) + "88";
  ctx.beginPath();
  p.trail.forEach(([tx, ty], i) => {
    const X = tx * W, Y = ty * H;
    i === 0 ? ctx.moveTo(X, Y) : ctx.lineTo(X, Y);
  });
  ctx.stroke();
}

function drawThreat(t, W, H, lw) {
  const [x, y, w, h] = t.box;
  const X = x * W, Y = y * H;
  ctx.lineWidth = lw * 2;
  ctx.strokeStyle = COLORS.threat;
  ctx.strokeRect(X, Y, w * W, h * H);
  const label = `⚠ ${t.cls.toUpperCase()}`;
  const th = parseInt(ctx.font, 10) + 6;
  ctx.fillStyle = COLORS.threat;
  ctx.fillRect(X - lw, Y - th, ctx.measureText(label).width + 10, th);
  ctx.fillStyle = "#fff";
  ctx.fillText(label, X + 4, Y - 3);
}

function drawDanger(d, W, H, lw) {
  const X = d.pos[0] * W, Y = d.pos[1] * H;
  ctx.lineWidth = lw * 1.5;
  ctx.strokeStyle = COLORS.danger;
  const r = Math.max(14, W / 50);
  ctx.beginPath();
  ctx.arc(X, Y, r, 0, Math.PI * 2);
  ctx.moveTo(X - r * 0.6, Y - r * 0.6); ctx.lineTo(X + r * 0.6, Y + r * 0.6);
  ctx.moveTo(X + r * 0.6, Y - r * 0.6); ctx.lineTo(X - r * 0.6, Y + r * 0.6);
  ctx.stroke();
  ctx.fillStyle = COLORS.danger;
  ctx.fillText(d.off ? "FARA (utanför bild)" : "FARA", X + r + 4, Y + 4);
}

function drawBase(b, W, H, lw) {
  const X = b.pos[0] * W, Y = b.pos[1] * H;
  ctx.lineWidth = lw * 1.5;
  ctx.strokeStyle = COLORS.base;
  ctx.fillStyle = COLORS.base;
  const s = Math.max(16, W / 45);
  ctx.beginPath(); // flag
  ctx.moveTo(X, Y); ctx.lineTo(X, Y - s * 1.6);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(X, Y - s * 1.6); ctx.lineTo(X + s, Y - s * 1.25); ctx.lineTo(X, Y - s * 0.9);
  ctx.closePath(); ctx.fill();
  ctx.fillText("BAS (förslag)", X + 6, Y + parseInt(ctx.font, 10));
}

function drawHazards(meta, W, H, lw) {
  const hz = meta.hazards || {};
  if (hz.fire) {
    const X = hz.fire.pos[0] * W, Y = hz.fire.pos[1] * H;
    ctx.fillStyle = COLORS.fire;
    ctx.font = `bold ${Math.max(16, W / 40)}px system-ui`;
    ctx.fillText("🔥", X - 10, Y + 10);
    ctx.font = `bold ${Math.max(11, W / 60)}px system-ui, sans-serif`;
    ctx.fillText("BRAND (heuristik)", X + 14, Y + 4);
  }
  if (hz.smoke) {
    const X = hz.smoke.pos[0] * W, Y = hz.smoke.pos[1] * H;
    const [dx, dy] = hz.smoke.drift;
    const mag = Math.hypot(dx, dy);
    ctx.strokeStyle = COLORS.smoke;
    ctx.fillStyle = COLORS.smoke;
    ctx.lineWidth = lw * 1.4;
    if (mag > 1e-4) {
      const k = Math.min(0.25, mag * 40) * W;
      const ux = dx / mag, uy = dy / mag;
      const X2 = X + ux * k, Y2 = Y + uy * k;
      ctx.beginPath(); ctx.moveTo(X, Y); ctx.lineTo(X2, Y2); ctx.stroke();
      const a = Math.atan2(uy, ux), ah = Math.max(8, W / 90);
      ctx.beginPath();
      ctx.moveTo(X2, Y2);
      ctx.lineTo(X2 - ah * Math.cos(a - 0.5), Y2 - ah * Math.sin(a - 0.5));
      ctx.lineTo(X2 - ah * Math.cos(a + 0.5), Y2 - ah * Math.sin(a + 0.5));
      ctx.closePath(); ctx.fill();
    }
    ctx.fillText("RÖK", X + 6, Y - 6);
  }
}

/* ---------- HUD / panel ---------- */

const elVisible = document.querySelector("#st-visible b");
const elUnique = document.querySelector("#st-unique b");
const elIrr = document.querySelector("#st-irr b");
const elIrrBox = document.getElementById("st-irr");
const elThreat = document.getElementById("st-threat");
const elFps = document.querySelector("#st-fps b");
const situationList = document.getElementById("situation-list");
const threatList = document.getElementById("threat-list");
const srcInfo = document.getElementById("src-info");

let lastHud = 0;

function updateHud(meta) {
  overlayMsg.classList.add("hidden");
  const now = performance.now();
  if (now - lastHud < 250) return;
  lastHud = now;

  const s = meta.stats;
  elVisible.textContent = s.visible;
  elUnique.textContent = s.unique;
  elIrr.textContent = s.irr_now;
  elIrrBox.classList.toggle("active", s.irr_now > 0);
  elThreat.classList.toggle("hidden", !s.threat);
  elFps.textContent = `${meta.fps}·${meta.det_fps}`;

  threatBanner.classList.toggle("hidden", !s.threat);
  if (s.threat && meta.threats.length) {
    threatBanner.textContent = `⚠ HOT: ${[...new Set(meta.threats.map(t => t.cls.toUpperCase()))].join(", ")}`;
  }

  const sit = [];
  if (meta.base) for (const r of meta.base.reasons) sit.push(`<li>${esc(r)}</li>`);
  if (meta.hazards.fire) sit.push(`<li class="alert">Brand indikerad (${(meta.hazards.fire.area * 100).toFixed(1)} % av bilden)</li>`);
  if (meta.hazards.smoke) sit.push(`<li class="note">Rök indikerad — drift utritad i bild</li>`);
  if (meta.danger) sit.push(`<li class="note">Faropunkt markerad${meta.danger.off ? " (utanför bild)" : ""}</li>`);
  if (s.irr_total > 0) sit.push(`<li>Irrationellt beteende hos ${s.irr_total} person(er) under passet</li>`);
  situationList.innerHTML = sit.length ? sit.join("") : '<li class="dim">Inget att rapportera.</li>';

  threatList.innerHTML = meta.threats.length
    ? meta.threats.map(t => `<li class="alert">${esc(t.cls)} (${Math.round(t.conf * 100)} %)</li>`).join("")
    : '<li class="dim">Inga hot upptäckta.</li>';
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/* ---------- Toggles ---------- */

document.querySelectorAll("#toggles .chip[data-layer]").forEach(btn => {
  const key = btn.dataset.layer;
  btn.classList.toggle("on", !!layers[key]);
  btn.onclick = () => {
    layers[key] = !layers[key];
    btn.classList.toggle("on", layers[key]);
    localStorage.setItem("layers", JSON.stringify(layers));
    drawnFrame = null; // redraw with new layers
  };
});

const panel = document.getElementById("panel");
document.getElementById("btn-panel").onclick = function () {
  panel.classList.toggle("hidden");
  this.classList.toggle("on");
};

/* ---------- Danger marking ---------- */

const btnDanger = document.getElementById("btn-danger");
btnDanger.onclick = () => {
  dangerArmed = !dangerArmed;
  btnDanger.classList.toggle("armed", dangerArmed);
  dangerHint.classList.toggle("hidden", !dangerArmed);
};

document.getElementById("btn-clear-danger").onclick = () =>
  fetch("/api/danger", { method: "DELETE" });

canvas.addEventListener("click", (ev) => {
  if (!dangerArmed) return;
  const r = canvas.getBoundingClientRect();
  const x = (ev.clientX - r.left) / r.width;
  const y = (ev.clientY - r.top) / r.height;
  fetch("/api/danger", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ x, y }),
  });
  dangerArmed = false;
  btnDanger.classList.remove("armed");
  dangerHint.classList.add("hidden");
});

/* ---------- Source control ---------- */

const videoSelect = document.getElementById("video-select");

async function refreshSources() {
  try {
    const [vids, state] = await Promise.all([
      fetch("/api/videos").then(r => r.json()),
      fetch("/api/state").then(r => r.json()),
    ]);
    const current = state.pipeline.source || "";
    videoSelect.innerHTML = vids.videos
      .map(v => `<option value="${esc(v.name)}" ${current.endsWith(v.name) ? "selected" : ""}>${esc(v.name)} (${v.size_mb} MB)</option>`)
      .join("") || '<option value="">— inga filmer i videos/ —</option>';
    const p = state.pipeline;
    srcInfo.textContent =
      `Status: ${p.status}${p.error ? " — " + p.error : ""} · Modell: ${state.config.model} · ` +
      `${p.render_fps ?? "–"} fps video, ${p.detect_fps ?? "–"} Hz analys`;
    if (p.status === "error") {
      overlayMsg.textContent = p.error || "Fel i pipeline";
      overlayMsg.classList.remove("hidden");
    } else if (p.status === "idle" || !p.source) {
      overlayMsg.innerHTML = "Ingen videokälla.<br>Lägg filmer i <code>videos/</code> eller ladda upp via panelen.";
      overlayMsg.classList.remove("hidden");
      panel.classList.remove("hidden");
    }
  } catch (_) { /* server gone; ws reconnect handles it */ }
}
refreshSources();
setInterval(refreshSources, 4000);

document.getElementById("btn-switch").onclick = async () => {
  if (!videoSelect.value) return;
  overlayMsg.textContent = "Byter källa …";
  overlayMsg.classList.remove("hidden");
  await fetch("/api/source", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name: videoSelect.value }),
  });
};

document.getElementById("upload").addEventListener("change", async function () {
  if (!this.files.length) return;
  const status = document.getElementById("upload-status");
  status.textContent = "Laddar upp …";
  const fd = new FormData();
  fd.append("file", this.files[0]);
  try {
    const res = await fetch("/api/upload", { method: "POST", body: fd });
    const j = await res.json();
    status.textContent = res.ok ? `Klart: ${j.name}` : (j.detail || "Fel vid uppladdning");
    refreshSources();
  } catch (e) {
    status.textContent = "Fel vid uppladdning";
  }
  this.value = "";
});
