/* Insatsdrönare PoC — klient.
 * Tar emot binära WS-paket [u32 metalängd][meta-JSON][JPEG], ritar bilden på
 * canvas och lagren ovanpå. Togglar är rena klientval. */

"use strict";

const canvas = document.getElementById("view");
const ctx = canvas.getContext("2d");
const overlayMsg = document.getElementById("overlay-msg");
const dangerHint = document.getElementById("danger-hint");

const layers = { boxes: true, ids: true, trails: false, status: true, hazards: true, base: true };
try {
  Object.assign(layers, JSON.parse(localStorage.getItem("layers") || "{}"));
} catch (_) {}

let latest = null;      // { meta, bitmap }
let dangerArmed = false;
let lastPacketAt = 0;

/* ---------- WebSocket ---------- */

let ws = null;
let retryMs = 500;
let seq = 0; // decode-order guard: an older frame must never replace a newer

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws/stream`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => { retryMs = 500; };
  ws.onmessage = async (ev) => {
    const mySeq = ++seq;
    const buf = ev.data;
    const view = new DataView(buf);
    const metaLen = view.getUint32(0);
    const meta = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, metaLen)));
    const jpeg = new Blob([new Uint8Array(buf, 4 + metaLen)], { type: "image/jpeg" });
    try {
      const bitmap = await createImageBitmap(jpeg);
      if (latest && latest.seq > mySeq) { bitmap.close(); return; }
      if (latest && latest.bitmap) latest.bitmap.close();
      latest = { meta, bitmap, seq: mySeq };
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
  Overlay.draw(ctx, meta, layers);
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

/* ---------- HUD / panel ---------- */

const elVisible = document.querySelector("#st-visible b");
const elUnique = document.querySelector("#st-unique b");
const elIrr = document.querySelector("#st-irr b");
const elIrrBox = document.getElementById("st-irr");
const elFps = document.querySelector("#st-fps b");
const situationList = document.getElementById("situation-list");
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
  elFps.textContent = `${meta.fps}·${meta.det_fps}`;

  const sit = [];
  if (meta.base) for (const r of meta.base.reasons) sit.push(`<li>${esc(r)}</li>`);
  if (meta.hazards.fire) sit.push(`<li class="alert">Brand indikerad (${(meta.hazards.fire.area * 100).toFixed(1)} % av bilden)</li>`);
  if (meta.hazards.smoke) sit.push(`<li class="note">Rök indikerad — drift utritad i bild</li>`);
  if (meta.danger) sit.push(`<li class="note">Faropunkt markerad${meta.danger.off ? " (utanför bild)" : ""}</li>`);
  if (s.irr_total > 0) sit.push(`<li>Irrationellt beteende hos ${s.irr_total} person(er) under passet</li>`);
  situationList.innerHTML = sit.length ? sit.join("") : '<li class="dim">Inget att rapportera.</li>';
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
  // object-fit: contain letterboxes the content when max-height clips the
  // element — map the click into the actual video rectangle.
  const r = canvas.getBoundingClientRect();
  const scale = Math.min(r.width / canvas.width, r.height / canvas.height);
  const cw = canvas.width * scale, ch = canvas.height * scale;
  const ox = r.left + (r.width - cw) / 2, oy = r.top + (r.height - ch) / 2;
  const x = (ev.clientX - ox) / cw;
  const y = (ev.clientY - oy) / ch;
  if (x < 0 || x > 1 || y < 0 || y > 1) return; // tapped the letterbox bars
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
