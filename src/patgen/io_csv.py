"""CSV loading helpers tolerant of unknown encodings and unknown schemas."""

from __future__ import annotations

import csv
import glob
import os
import sys
from collections.abc import Iterator, Sequence
from typing import Union

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

ENCODINGS = ("utf-8-sig", "utf-8", "cp1256", "cp1252", "latin-1")
TEXT_COLUMN_HINTS = (
    "text", "message", "msg", "body", "sms", "content", "description", "raw",
    "نص", "الرساله", "الرسالة", "الوصف",
)


PathLike = Union[str, "os.PathLike[str]"]


def expand_inputs(patterns: Sequence[PathLike]) -> list[str]:
    paths: list[str] = []
    for pattern in patterns:
        text = os.fspath(pattern)
        matched = sorted(glob.glob(text))
        paths.extend(matched or [text])
    return paths


def _open(path: str):
    """Open a CSV with the first encoding that decodes the whole header row."""
    last_error: Exception | None = None
    for encoding in ENCODINGS:
        try:
            fh = open(path, newline="", encoding=encoding, errors="strict")
            fh.readline()
            fh.seek(0)
            return fh
        except (UnicodeDecodeError, LookupError) as exc:
            last_error = exc
            continue
    if last_error is not None:
        # Last resort: never fail on a single bad byte, the normalizer will
        # drop the replacement characters.
        return open(path, newline="", encoding="utf-8", errors="replace")
    raise RuntimeError(f"cannot open {path}")


def detect_text_column(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lowered = [h.strip().lower() for h in header]
    for hint in TEXT_COLUMN_HINTS:
        for i, name in enumerate(lowered):
            if name == hint:
                return header[i]
    for hint in TEXT_COLUMN_HINTS:
        for i, name in enumerate(lowered):
            if hint in name:
                return header[i]
    # Fall back to the column with the longest average content.
    best_index, best_score = 0, -1.0
    for i in range(len(header)):
        values = [row[i] for row in rows if i < len(row)]
        if not values:
            continue
        score = sum(len(v) for v in values) / len(values)
        if score > best_score:
            best_index, best_score = i, score
    return header[best_index]


def read_texts(
    paths: Sequence[PathLike], column: str | None = None, limit: int | None = None
) -> Iterator[str]:
    count = 0
    for path in expand_inputs(paths):
        with _open(path) as fh:
            reader = csv.reader(fh)
            try:
                header = next(reader)
            except StopIteration:
                continue
            sample = []
            for _ in range(50):
                try:
                    sample.append(next(reader))
                except StopIteration:
                    break
            name = column or detect_text_column(header, sample)
            if name not in header:
                raise KeyError(f"column {name!r} not in {path} (has {header})")
            index = header.index(name)
            for row in [*sample, *reader]:
                if index >= len(row):
                    continue
                value = row[index]
                if not value:
                    continue
                yield value
                count += 1
                if limit is not None and count >= limit:
                    return
