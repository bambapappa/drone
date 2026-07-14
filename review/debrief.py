"""Render the self-contained HTML training-debrief report.

Report §2.4/§5.3: the operator-comparison feature's actual deliverable is a
document a training lead can hand to people who weren't at the exercise —
"what did the AI find, what did the operator find, where did they diverge".
This module renders that as one standalone HTML string: inline CSS, no
external requests, no server needed to view it (open the file directly).

**Frame refs, not thumbnails.** The brief allows either. Thumbnails would
require decoding the source video at export time (VIDEO_DIR may not even be
mounted where the debrief is generated or later opened) and would make
debrief generation non-deterministic in wall-clock cost and dependent on a
live decoder session. A frame reference (frame_no + timestamp) is exact,
free, and always available since every event already carries `frame_start`/
`frame_end` in its evidence — sufficient for someone to locate the moment in
the review UI or the original footage without embedding image bytes.

**Deterministic given the same inputs.** No randomness, no wall-clock reads
inside this module — `generated_at` is passed in by the caller so the
rendered HTML is a pure function of (run metadata, comparison result,
generated_at), matching the "reproducible given the same events + same
imported notes" requirement from the brief.
"""

from __future__ import annotations

from html import escape
from typing import Any

from review.comparison import ComparisonResult, Match

# Mirrors review/static/app.js's CATEGORY_LABEL — kept in sync manually since
# this is a small, stable, rarely-changing table (see AGENTS.md's category
# registry note); category enum values themselves stay English everywhere.
CATEGORY_LABEL = {
    "STILLA": "STILLA",
    "MOT_FARA": "MOT FARA",
    "HAZARD": "FARA",
}

REVIEW_STATE_LABEL = {
    "unreviewed": "ogranskad",
    "confirmed": "bekräftad",
    "rejected": "avvisad",
}


def _fmt_t(seconds: float) -> str:
    """Seconds -> `[H:]MM:SS` — the same clock format the operator-notes
    parser accepts, so a reader can cross-reference the two directly."""
    seconds = max(0.0, seconds)
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _fmt_delta(delta_s: float) -> str:
    if abs(delta_s) < 0.5:
        return "samtidigt (± under en sekund)"
    direction = "AI upptäckte tidigare" if delta_s > 0 else "operatören noterade tidigare"
    return f"{direction}, {_fmt_t(abs(delta_s))} (mm:ss)"


def _frame_ref(evidence: dict[str, Any]) -> str:
    fs, fe = evidence.get("frame_start"), evidence.get("frame_end")
    if fs is None:
        return "–"
    return f"ruta {fs}" if fe is None or fe == fs else f"ruta {fs}–{fe}"


def _state_cell(ev: dict[str, Any]) -> str:
    review = ev.get("review") or {}
    raw_state = review.get("state", "unreviewed")
    label = REVIEW_STATE_LABEL.get(raw_state, "ogranskad")
    return f'<td><span class="state state-{escape(raw_state)}">{escape(label)}</span></td>'


def _event_row(ev: dict[str, Any]) -> str:
    cat = CATEGORY_LABEL.get(ev["category"], ev["category"])
    person = f"P{ev['person_id']}" if ev.get("person_id") is not None else "–"
    return (
        "<tr>"
        f"<td>{escape(_fmt_t(ev['t_start']))}–{escape(_fmt_t(ev['t_end']))}</td>"
        f'<td><span class="cat-tag cat-{escape(ev["category"])}">{escape(cat)}</span></td>'
        f"<td>{escape(person)}</td>"
        f"<td>{ev['confidence']:.2f}</td>"
        f"<td>{escape(_frame_ref(ev.get('evidence') or {}))}</td>"
        f"{_state_cell(ev)}"
        "</tr>"
    )


def _note_row(note: dict[str, Any]) -> str:
    return f"<tr><td>{escape(_fmt_t(note['t']))}</td><td>{escape(note.get('text', ''))}</td></tr>"


def _match_row(m: Match) -> str:
    ev, note = m.event, m.note
    cat = CATEGORY_LABEL.get(ev["category"], ev["category"])
    person = f"P{ev['person_id']}" if ev.get("person_id") is not None else "–"
    return (
        "<tr>"
        f"<td>{escape(_fmt_t(ev['t_start']))}</td>"
        f"<td>{escape(_fmt_t(note['t']))}</td>"
        f'<td><span class="cat-tag cat-{escape(ev["category"])}">{escape(cat)}</span></td>'
        f"<td>{escape(person)}</td>"
        f"<td>{escape(note.get('text', ''))}</td>"
        f"<td>{escape(_fmt_delta(m.delta_s))}</td>"
        f"<td>{escape(_frame_ref(ev.get('evidence') or {}))}</td>"
        f"{_state_cell(ev)}"
        "</tr>"
    )


def render_debrief_html(
    run_id: str,
    comparison: ComparisonResult,
    generated_at: str,
    video_filename: str | None = None,
) -> str:
    """Render the full standalone HTML debrief document as a string."""
    counts = comparison.counts
    both_rows = "".join(_match_row(m) for m in comparison.both) or (
        '<tr><td colspan="8" class="empty">Inga händelser hittade av båda.</td></tr>'
    )
    ai_only_rows = "".join(_event_row(e) for e in comparison.ai_only) or (
        '<tr><td colspan="6" class="empty">Inga AI-unika händelser.</td></tr>'
    )
    operator_only_rows = "".join(_note_row(n) for n in comparison.operator_only) or (
        '<tr><td colspan="2" class="empty">Inga operatörsunika observationer.</td></tr>'
    )

    video_line = f"Video: {escape(video_filename)}<br>" if video_filename else ""

    return f"""<!doctype html>
<html lang="sv">
<head>
<meta charset="utf-8">
<title>Insatsdebriefing – {escape(run_id)}</title>
<style>
{_CSS}
</style>
</head>
<body>
<header>
  <h1>Insatsdebriefing</h1>
  <p class="meta">
    Körning: <code>{escape(run_id)}</code><br>
    {video_line}
    Genererad: {escape(generated_at)}<br>
    Tidstolerans för matchning: ±{comparison.tolerance_s:.0f} s
  </p>
  <div class="summary">
    <div class="stat"><b>{counts["both"]}</b><small>hittad av båda</small></div>
    <div class="stat"><b>{counts["ai_only"]}</b><small>endast AI</small></div>
    <div class="stat"><b>{counts["operator_only"]}</b><small>endast operatör</small></div>
  </div>
</header>

<main>
  <section>
    <h2>Hittad av båda</h2>
    <table>
      <thead><tr>
        <th>AI-tid</th><th>Operatörstid</th><th>Kategori</th><th>Person</th>
        <th>Operatörens anteckning</th><th>Tidsskillnad</th><th>Bildreferens</th><th>Granskningsstatus</th>
      </tr></thead>
      <tbody>{both_rows}</tbody>
    </table>
  </section>

  <section>
    <h2>Endast AI</h2>
    <table>
      <thead><tr>
        <th>Tid</th><th>Kategori</th><th>Person</th><th>Konfidens</th><th>Bildreferens</th><th>Granskningsstatus</th>
      </tr></thead>
      <tbody>{ai_only_rows}</tbody>
    </table>
  </section>

  <section>
    <h2>Endast operatör</h2>
    <table>
      <thead><tr><th>Tid</th><th>Anteckning</th></tr></thead>
      <tbody>{operator_only_rows}</tbody>
    </table>
  </section>
</main>

<footer>
  <p>Genererad av drönare-verktygets granskningsvy (fas 3, utvärderingslager).
  Denna rapport är fristående — den behöver ingen server för att visas.</p>
</footer>
</body>
</html>
"""


_CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body {
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  max-width: 1100px; margin: 0 auto; padding: 24px;
  background: #fff; color: #1a1f24;
}
@media (prefers-color-scheme: dark) {
  body { background: #14181d; color: #e8edf2; }
  table { background: #1d232b; }
  th { background: #262e38 !important; }
  tr:nth-child(even) td { background: #1a2027; }
  code { background: #262e38; }
}
header { border-bottom: 2px solid #34c3ff; padding-bottom: 16px; margin-bottom: 24px; }
h1 { margin: 0 0 8px; font-size: 26px; }
h2 { font-size: 18px; margin: 28px 0 8px; }
.meta { color: #667; font-size: 13px; line-height: 1.7; }
code { background: #eef2f6; padding: 1px 6px; border-radius: 4px; }
.summary { display: flex; gap: 12px; margin-top: 14px; }
.stat { background: #eef2f6; border-radius: 8px; padding: 8px 16px; text-align: center; }
.stat b { display: block; font-size: 22px; }
.stat small { text-transform: uppercase; letter-spacing: .5px; font-size: 10px; color: #667; }
table { border-collapse: collapse; width: 100%; margin-bottom: 8px; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid #dde3e9; font-size: 13px; }
th { background: #f4f7fa; font-size: 11px; text-transform: uppercase; letter-spacing: .5px; color: #667; }
td.empty { color: #889; font-style: italic; text-align: center; padding: 14px; }
.cat-tag { display: inline-block; padding: 1px 7px; border-radius: 4px; font-size: 11px; font-weight: 700; }
.cat-STILLA { background: #5c1119; color: #ff8a95; }
.cat-MOT_FARA { background: #5a3a00; color: #ffcb7a; }
.cat-HAZARD { background: #3a1d10; color: #ff9d75; }
.state { font-size: 12px; }
.state-confirmed { color: #1e8e4a; font-weight: 600; }
.state-rejected { color: #c0392b; font-weight: 600; }
.state-unreviewed { color: #889; }
footer { margin-top: 32px; color: #889; font-size: 12px; border-top: 1px solid #dde3e9; padding-top: 12px; }
"""
