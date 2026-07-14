"""Parse imported operator field notes into timestamped rows.

Report §2.4/§5.3: the operator's live field notes are short, timestamped
observations jotted down during the exercise — e.g. "2 personer vid
fordonet, 14:32". This module turns a pasted/uploaded blob of such lines
into `(t_seconds, text)` rows the comparison engine (`review/comparison.py`)
can align against AI events.

**Timebase: video-relative elapsed time, not wall-clock.** The offline tool
uses video time everywhere else (`t = frame_no / fps`, per AGENTS.md's
"Timebase" section) and the manifest/ingest schema has no concept of the
video's real-world recording start time to anchor a wall-clock "14:32"
against. Rather than inventing a new wall-clock-alignment concept solely for
this feature, operator note timestamps are interpreted the same way as
every other timestamp in this tool: elapsed time since the exercise/video
started, as `[H:]MM:SS` or plain seconds. A "14:32" note in a ~16-minute
exercise film reads naturally as 14 minutes 32 seconds in — this is also
almost certainly what an operator scribbling notes during a live exercise
means (they are timing against the exercise clock, not reciting a wall-clock
time for someone to reverse-engineer later).

**Forgiving by design.** This is field data, transcribed under time
pressure, so the parser tolerates the variance that actually shows up in
handwritten-then-typed notes rather than demanding one exact shape:
  - time and text may appear in either order on a line ("14:32, text" or
    "text, 14:32") — whichever token parses as a time wins;
  - the field delimiter may be a comma or a semicolon. Semicolon is tried
    FIRST when present: Swedish-locale spreadsheet exports use semicolon as
    the field separator specifically because comma is the decimal mark, so
    a line like "872,5; kort observation" must split on ';', not on the
    comma inside the decimal seconds value. When no semicolon is present, a
    plain-seconds value's decimal comma is naturally not the last comma in
    a well-formed two-field line, so splitting on the *last* comma still
    resolves correctly for the common case;
  - a leading header row is recognized by column name (Swedish or English:
    t/tid/time/tidpunkt/klockslag for the timestamp, text/anteckning/note/
    notering/kommentar/beskrivning for the text) and parsed as real CSV
    (quoted fields, embedded delimiters) instead of the line-splitting
    heuristic;
  - blank lines and lines starting with '#' are skipped;
  - a line that can't be parsed is reported as a warning (with its line
    number and the raw text) rather than aborting the whole import — one
    garbled line should never lose the rest of a field report.

**Known ambiguity, accepted:** a comma-delimited (no semicolon) line whose
free text ends in a bare number can collide with a decimal-comma seconds
value (e.g. "kort observation, 872,5" — the text itself contains a comma
before a trailing number). This is an inherent ambiguity in mixing
comma-as-delimiter with comma-as-decimal-mark; the fix is to use a
semicolon delimiter or `H:MM:SS` time format, which is why this module
documents that recommendation rather than trying to disambiguate free text.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass

_HEADER_TIME_NAMES = {"t", "tid", "time", "tidpunkt", "klockslag"}
_HEADER_TEXT_NAMES = {"text", "anteckning", "note", "notering", "kommentar", "beskrivning"}

# H:MM:SS(.f) or M(M):SS(.f) — seconds may use '.' or ',' as the decimal mark.
_CLOCK_RE = re.compile(r"^(?:(\d+):)?(\d{1,2}):(\d{2}(?:[.,]\d+)?)$")
_SECONDS_RE = re.compile(r"^\d+(?:[.,]\d+)?$")

# Semicolon first: see module docstring on why comma-decimal + comma-delimiter
# is ambiguous and semicolon is the escape hatch.
_LINE_DELIMS = (";", ",")


@dataclass(frozen=True)
class ParsedNote:
    t: float
    text: str
    raw_line: str


@dataclass(frozen=True)
class ParseWarning:
    line_no: int
    raw_line: str
    reason: str


@dataclass(frozen=True)
class ParseResult:
    notes: list[ParsedNote]
    warnings: list[ParseWarning]


def _parse_time_token(token: str) -> float | None:
    token = token.strip()
    if not token:
        return None
    m = _CLOCK_RE.match(token)
    if m:
        hours = int(m.group(1)) if m.group(1) else 0
        minutes = int(m.group(2))
        seconds = float(m.group(3).replace(",", "."))
        if minutes >= 60 or seconds >= 60:
            return None
        return hours * 3600 + minutes * 60 + seconds
    if _SECONDS_RE.match(token):
        return float(token.replace(",", "."))
    return None


def _split_line(line: str) -> tuple[float, str] | None:
    """Split one freeform line into (t, text), trying each delimiter and
    both field orders. Returns None if neither delimiter yields a line with
    one time-like field and one non-empty text field."""
    for delim in _LINE_DELIMS:
        if delim not in line:
            continue
        left, _, right = line.rpartition(delim)
        left, right = left.strip(), right.strip()
        t = _parse_time_token(right)
        if t is not None and left:
            return t, left
        t = _parse_time_token(left)
        if t is not None and right:
            return t, right
    return None


def _detect_header(first_line: str) -> tuple[str, int, int] | None:
    """If `first_line` looks like a CSV header (has both a recognized time
    column and text column name), return (delimiter, time_col_idx,
    text_col_idx). Otherwise None."""
    for delim in _LINE_DELIMS:
        if delim not in first_line:
            continue
        tokens = [t.strip().lower() for t in first_line.split(delim)]
        time_col = next((i for i, t in enumerate(tokens) if t in _HEADER_TIME_NAMES), None)
        text_col = next((i for i, t in enumerate(tokens) if t in _HEADER_TEXT_NAMES), None)
        if time_col is not None and text_col is not None:
            return delim, time_col, text_col
    return None


def parse_operator_notes(text: str) -> ParseResult:
    """Parse a pasted/uploaded blob of operator field notes.

    Returns every line that could be parsed as a `(t, text)` note, plus a
    warning per line that couldn't. Never raises on malformed input — an
    import is best-effort by design (see module docstring)."""
    text = text.lstrip("﻿")  # tolerate a Windows-exported BOM
    lines = text.splitlines()

    header: tuple[str, int, int] | None = None
    header_line_no: int | None = None
    for i, raw in enumerate(lines):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        header = _detect_header(stripped)
        header_line_no = i
        break

    notes: list[ParsedNote] = []
    warnings: list[ParseWarning] = []

    for i, raw in enumerate(lines):
        if i == header_line_no and header is not None:
            continue
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if header is not None:
            delim, time_col, text_col = header
            fields = next(csv.reader([stripped], delimiter=delim))
            if len(fields) <= max(time_col, text_col):
                warnings.append(ParseWarning(i + 1, raw, "för få fält på raden för rubriken"))
                continue
            t = _parse_time_token(fields[time_col])
            note_text = fields[text_col].strip()
            if t is None:
                warnings.append(ParseWarning(i + 1, raw, f"kunde inte tolka tid: {fields[time_col]!r}"))
                continue
            if not note_text:
                warnings.append(ParseWarning(i + 1, raw, "tom anteckningstext"))
                continue
            notes.append(ParsedNote(t=t, text=note_text, raw_line=raw))
            continue

        parsed = _split_line(stripped)
        if parsed is None:
            warnings.append(
                ParseWarning(i + 1, raw, "kunde inte tolka raden (varken tid eller text hittades)")
            )
            continue
        t, note_text = parsed
        notes.append(ParsedNote(t=t, text=note_text, raw_line=raw))

    return ParseResult(notes=notes, warnings=warnings)
