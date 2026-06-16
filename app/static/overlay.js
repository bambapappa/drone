/* Shared overlay renderer — used by both the live client (app.js) and the
 * offline player (player.js) so they draw pixel-identically and stay in sync.
 * Pure drawing: given a 2D context, a metadata record and the active layer
 * toggles, it paints boxes/ids/trails/behavior/hazards/base/danger. It never
 * touches the base image (live draws a bitmap, the player shows a <video>). */

"use strict";

window.Overlay = (function () {
  const COLORS = {
    ok: "#2ecc71",
    still: "#ff4757",
    toward_danger: "#ffa502",
    base: "#34c3ff",
    danger: "#ff4757",
    smoke: "#aab4be",
    fire: "#ff6b35",
  };
  const STATUS_TEXT = { still: "STILLA", toward_danger: "MOT FARA" };

  function draw(ctx, meta, layers) {
    const W = ctx.canvas.width, H = ctx.canvas.height;
    const lw = Math.max(2, W / 480);
    ctx.font = `bold ${Math.max(11, W / 60)}px system-ui, sans-serif`;
    ctx.textBaseline = "bottom";

    if (layers.hazards) drawHazards(ctx, meta, W, H, lw);
    if (layers.trails) for (const p of meta.persons) drawTrail(ctx, p, W, H, lw, layers);
    if (layers.boxes) for (const p of meta.persons) drawPerson(ctx, p, W, H, lw, layers);
    if (meta.danger) drawDanger(ctx, meta.danger, W, H, lw);
    if (layers.base && meta.base) drawBase(ctx, meta.base, W, H, lw);
  }

  function drawPerson(ctx, p, W, H, lw, layers) {
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

  function drawTrail(ctx, p, W, H, lw, layers) {
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

  function drawDanger(ctx, d, W, H, lw) {
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

  function drawBase(ctx, b, W, H, lw) {
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

  function drawHazards(ctx, meta, W, H, lw) {
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

  return { COLORS, STATUS_TEXT, draw };
})();
