"""Tests for review.debrief — the standalone HTML training-debrief report.

Covers: presence of the three Swedish bucket sections, correct rendering of
matched/AI-only/operator-only rows, empty-state text when a bucket has
nothing, HTML-escaping of operator-authored free text (XSS safety — this is
the one place untrusted human input flows into an HTML document), and that
the output is a single self-contained document (no external resource refs)."""

from __future__ import annotations

from review.comparison import ComparisonResult, Match
from review.debrief import render_debrief_html


def _event(
    event_id="ev-1",
    category="STILLA",
    person_id=1,
    t_start=10.0,
    t_end=15.0,
    confidence=0.75,
    evidence=None,
    state="unreviewed",
):
    return {
        "event_id": event_id,
        "category": category,
        "person_id": person_id,
        "t_start": t_start,
        "t_end": t_end,
        "confidence": confidence,
        "evidence": evidence or {"frame_start": 250, "frame_end": 375},
        "review": {"state": state, "note": None, "reviewer": None, "reviewed_at": None},
    }


def _note(annotation_id="n-1", t=12.0, text="observation"):
    return {"annotation_id": annotation_id, "t": t, "text": text, "raw_line": None}


class TestSections:
    def test_all_three_swedish_bucket_headings_present(self):
        comparison = ComparisonResult(tolerance_s=60.0, both=[], ai_only=[], operator_only=[])
        html = render_debrief_html("run-1", comparison, generated_at="2026-07-15T00:00:00Z")
        assert "Hittad av båda" in html
        assert "Endast AI" in html
        assert "Endast operatör" in html

    def test_empty_buckets_show_swedish_empty_state(self):
        comparison = ComparisonResult(tolerance_s=60.0, both=[], ai_only=[], operator_only=[])
        html = render_debrief_html("run-1", comparison, generated_at="2026-07-15T00:00:00Z")
        assert "Inga händelser hittade av båda" in html
        assert "Inga AI-unika händelser" in html
        assert "Inga operatörsunika observationer" in html

    def test_counts_summary_reflects_bucket_sizes(self):
        ev = _event()
        note = _note()
        comparison = ComparisonResult(
            tolerance_s=60.0,
            both=[Match(event=ev, note=note, delta_s=2.0)],
            ai_only=[_event(event_id="ev-2")],
            operator_only=[_note(annotation_id="n-2")],
        )
        html = render_debrief_html("run-1", comparison, generated_at="2026-07-15T00:00:00Z")
        # One stat block each for both/ai_only/operator_only counts.
        assert ">1<" in html


class TestMatchedRows:
    def test_matched_row_includes_category_person_delta(self):
        ev = _event(category="MOT_FARA", person_id=3, t_start=100.0)
        note = _note(t=90.0, text="figurant sprang mot faran")
        comparison = ComparisonResult(
            tolerance_s=60.0, both=[Match(event=ev, note=note, delta_s=10.0)], ai_only=[], operator_only=[]
        )
        html = render_debrief_html("run-1", comparison, generated_at="now")
        assert "MOT FARA" in html
        assert "P3" in html
        assert "figurant sprang mot faran" in html
        assert "AI upptäckte tidigare" in html

    def test_operator_first_delta_phrasing(self):
        ev = _event(t_start=100.0)
        note = _note(t=110.0)
        comparison = ComparisonResult(
            tolerance_s=60.0, both=[Match(event=ev, note=note, delta_s=10.0)], ai_only=[], operator_only=[]
        )
        html = render_debrief_html("run-1", comparison, generated_at="now")
        assert "AI upptäckte tidigare" in html

    def test_ai_first_delta_phrasing(self):
        ev = _event(t_start=100.0)
        note = _note(t=90.0)
        comparison = ComparisonResult(
            tolerance_s=60.0, both=[Match(event=ev, note=note, delta_s=-10.0)], ai_only=[], operator_only=[]
        )
        html = render_debrief_html("run-1", comparison, generated_at="now")
        assert "operatören noterade tidigare" in html

    def test_near_simultaneous_delta_phrasing(self):
        ev = _event(t_start=100.0)
        note = _note(t=100.2)
        comparison = ComparisonResult(
            tolerance_s=60.0, both=[Match(event=ev, note=note, delta_s=0.2)], ai_only=[], operator_only=[]
        )
        html = render_debrief_html("run-1", comparison, generated_at="now")
        assert "samtidigt" in html


class TestReviewState:
    def test_confirmed_state_shown_in_swedish(self):
        ev = _event(state="confirmed")
        comparison = ComparisonResult(tolerance_s=60.0, both=[], ai_only=[ev], operator_only=[])
        html = render_debrief_html("run-1", comparison, generated_at="now")
        assert "bekräftad" in html

    def test_rejected_state_shown_in_swedish(self):
        ev = _event(state="rejected")
        comparison = ComparisonResult(tolerance_s=60.0, both=[], ai_only=[ev], operator_only=[])
        html = render_debrief_html("run-1", comparison, generated_at="now")
        assert "avvisad" in html


class TestFrameRefs:
    def test_frame_ref_range_shown(self):
        ev = _event(evidence={"frame_start": 100, "frame_end": 200})
        comparison = ComparisonResult(tolerance_s=60.0, both=[], ai_only=[ev], operator_only=[])
        html = render_debrief_html("run-1", comparison, generated_at="now")
        assert "ruta 100" in html and "200" in html

    def test_missing_frame_ref_degrades_gracefully(self):
        ev = _event(evidence={})
        comparison = ComparisonResult(tolerance_s=60.0, both=[], ai_only=[ev], operator_only=[])
        html = render_debrief_html("run-1", comparison, generated_at="now")
        assert html  # doesn't raise


class TestEscaping:
    def test_operator_note_text_is_escaped(self):
        note = _note(text="<script>alert(1)</script>")
        comparison = ComparisonResult(tolerance_s=60.0, both=[], ai_only=[], operator_only=[note])
        html = render_debrief_html("run-1", comparison, generated_at="now")
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;" in html

    def test_matched_note_text_is_escaped(self):
        ev = _event()
        note = _note(text="<img src=x onerror=alert(1)>")
        comparison = ComparisonResult(
            tolerance_s=60.0, both=[Match(event=ev, note=note, delta_s=0.0)], ai_only=[], operator_only=[]
        )
        html = render_debrief_html("run-1", comparison, generated_at="now")
        assert "<img src=x" not in html

    def test_run_id_is_escaped(self):
        comparison = ComparisonResult(tolerance_s=60.0, both=[], ai_only=[], operator_only=[])
        html = render_debrief_html('"><script>x</script>', comparison, generated_at="now")
        assert "<script>x</script>" not in html


class TestSelfContained:
    def test_no_external_resource_references(self):
        comparison = ComparisonResult(tolerance_s=60.0, both=[], ai_only=[], operator_only=[])
        html = render_debrief_html("run-1", comparison, generated_at="now", video_filename="film.mp4")
        assert "http://" not in html
        assert "https://" not in html
        assert "<link" not in html
        assert "<script" not in html  # static document — no JS at all

    def test_video_filename_included_when_present(self):
        comparison = ComparisonResult(tolerance_s=60.0, both=[], ai_only=[], operator_only=[])
        html = render_debrief_html("run-1", comparison, generated_at="now", video_filename="brandfilm.mp4")
        assert "brandfilm.mp4" in html

    def test_is_valid_looking_html_document(self):
        comparison = ComparisonResult(tolerance_s=60.0, both=[], ai_only=[], operator_only=[])
        html = render_debrief_html("run-1", comparison, generated_at="now")
        assert html.strip().startswith("<!doctype html>")
        assert "<html" in html and "</html>" in html
