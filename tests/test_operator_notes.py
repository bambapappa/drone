"""Tests for the operator field-notes import parser (review.operator_notes).

Covers the forgiving-parsing surface documented in the module: time-then-text
and text-then-time ordering, comma vs semicolon delimiter (incl. the
decimal-comma-vs-field-delimiter collision), H:MM:SS / M:SS / plain-seconds
formats, header-based CSV, and graceful degradation (a bad line becomes a
warning, not a crash, and never drops the rest of the import)."""

from __future__ import annotations

from review.operator_notes import parse_operator_notes


class TestFreeformLines:
    def test_text_then_time_mmss(self):
        result = parse_operator_notes("2 personer vid fordonet, 14:32")
        assert len(result.notes) == 1
        note = result.notes[0]
        assert note.t == 14 * 60 + 32
        assert note.text == "2 personer vid fordonet"
        assert not result.warnings

    def test_time_then_text(self):
        result = parse_operator_notes("14:32, 2 personer vid fordonet")
        assert len(result.notes) == 1
        assert result.notes[0].t == 872.0
        assert result.notes[0].text == "2 personer vid fordonet"

    def test_hmmss_format(self):
        result = parse_operator_notes("1:04:32, brand vid ladan")
        assert result.notes[0].t == 3600 + 4 * 60 + 32

    def test_plain_seconds_dot_decimal(self):
        result = parse_operator_notes("kort observation, 872.5")
        assert result.notes[0].t == 872.5
        assert result.notes[0].text == "kort observation"

    def test_plain_seconds_comma_decimal_with_comma_delimiter(self):
        # The decimal comma is not the LAST comma on the line, so rpartition
        # on ',' still isolates the time field correctly.
        result = parse_operator_notes("872,5, kort observation")
        assert result.notes[0].t == 872.5
        assert result.notes[0].text == "kort observation"

    def test_plain_seconds_comma_decimal_with_semicolon_delimiter(self):
        result = parse_operator_notes("872,5; kort observation")
        assert result.notes[0].t == 872.5
        assert result.notes[0].text == "kort observation"

    def test_text_then_seconds_comma_decimal_semicolon_delim(self):
        result = parse_operator_notes("kort observation; 872,5")
        assert result.notes[0].t == 872.5
        assert result.notes[0].text == "kort observation"

    def test_extra_whitespace_tolerated(self):
        result = parse_operator_notes("  2 personer vid fordonet  ,   14:32  ")
        assert result.notes[0].t == 872.0
        assert result.notes[0].text == "2 personer vid fordonet"

    def test_blank_lines_and_comments_ignored(self):
        blob = "\n".join(
            [
                "# fältanteckningar, insats 2026-07-14",
                "",
                "2 personer vid fordonet, 14:32",
                "   ",
                "# nästa observation",
                "rök vid ladan, 15:10",
            ]
        )
        result = parse_operator_notes(blob)
        assert len(result.notes) == 2
        assert not result.warnings

    def test_bom_stripped(self):
        result = parse_operator_notes("﻿2 personer vid fordonet, 14:32")
        assert len(result.notes) == 1
        assert result.notes[0].t == 872.0


class TestMultipleLines:
    def test_multiple_notes_preserve_order(self):
        blob = "2 personer vid fordonet, 14:32\nrök vid ladan, 15:10\nfigurant ligger still, 2:00"
        result = parse_operator_notes(blob)
        assert [n.t for n in result.notes] == [872.0, 910.0, 120.0]

    def test_one_bad_line_does_not_drop_others(self):
        blob = "\n".join(
            [
                "2 personer vid fordonet, 14:32",
                "helt oläslig rad utan tid",
                "rök vid ladan, 15:10",
            ]
        )
        result = parse_operator_notes(blob)
        assert len(result.notes) == 2
        assert len(result.warnings) == 1
        assert result.warnings[0].line_no == 2
        assert "oläslig" in result.warnings[0].raw_line


class TestInvalidTimes:
    def test_minutes_60_or_more_rejected(self):
        result = parse_operator_notes("text, 1:65")
        assert not result.notes
        assert len(result.warnings) == 1

    def test_seconds_60_or_more_rejected(self):
        result = parse_operator_notes("text, 1:60")
        assert not result.notes
        assert len(result.warnings) == 1

    def test_time_only_no_text_is_a_warning(self):
        result = parse_operator_notes("14:32")
        assert not result.notes
        assert len(result.warnings) == 1


class TestHeaderCsv:
    def test_swedish_header_comma(self):
        blob = "tid,anteckning\n14:32,2 personer vid fordonet\n15:10,rök vid ladan"
        result = parse_operator_notes(blob)
        assert len(result.notes) == 2
        assert result.notes[0].t == 872.0
        assert result.notes[0].text == "2 personer vid fordonet"

    def test_english_header_semicolon(self):
        blob = "time;text\n14:32;2 people at the vehicle\n15:10;smoke near the barn"
        result = parse_operator_notes(blob)
        assert len(result.notes) == 2
        assert result.notes[1].text == "smoke near the barn"

    def test_header_columns_out_of_canonical_order(self):
        blob = "anteckning,tid\n2 personer vid fordonet,14:32"
        result = parse_operator_notes(blob)
        assert len(result.notes) == 1
        assert result.notes[0].t == 872.0
        assert result.notes[0].text == "2 personer vid fordonet"

    def test_header_row_with_embedded_delimiter_in_quoted_field(self):
        blob = 'tid,anteckning\n14:32,"2 personer, en vid bilen"'
        result = parse_operator_notes(blob)
        assert len(result.notes) == 1
        assert result.notes[0].text == "2 personer, en vid bilen"

    def test_header_row_bad_time_warns(self):
        blob = "tid,anteckning\nokänd,2 personer vid fordonet"
        result = parse_operator_notes(blob)
        assert not result.notes
        assert len(result.warnings) == 1


class TestEmptyInput:
    def test_empty_string(self):
        result = parse_operator_notes("")
        assert result.notes == []
        assert result.warnings == []

    def test_only_comments_and_blanks(self):
        result = parse_operator_notes("# bara kommentarer\n\n   \n")
        assert result.notes == []
        assert result.warnings == []
