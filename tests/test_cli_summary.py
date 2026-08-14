"""Tests for the CLI unique-person summary helper (analysis.cli.format_unique_summary).

The #2 fix: a rescue context must never let a bare unique count read as
"nobody seen" when short sightings (transients) exist. The 2s confirm
threshold (DECISIONS B16) is NOT changed — this is presentation only.
"""

from __future__ import annotations

from analysis.cli import format_unique_summary


def test_no_transient_no_uncertain_is_bare_count():
    assert (
        format_unique_summary(confirmed=3, uncertain=0, transient=0, confirm_s=2.0)
        == "Unika personer:   3 unika"
    )


def test_transient_surfaces_on_same_line_when_present():
    # The dangerous case: 0 confirmed, 2 short sightings. Must NOT read as a
    # bare "0 unika" — transients appear on the same line.
    s = format_unique_summary(confirmed=0, uncertain=0, transient=2, confirm_s=2.0)
    assert "0 unika" in s
    assert "2 korta observationer" in s
    assert "2s" in s


def test_uncertain_and_transient_both_shown():
    s = format_unique_summary(confirmed=5, uncertain=1, transient=3, confirm_s=2.0)
    assert "5 unika" in s
    assert "1 osäkra sammanslagningar" in s  # plural per established CLI convention
    assert "3 korta observationer" in s
