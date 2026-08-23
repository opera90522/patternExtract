"""Optional, user supplied slot vocabularies.

The learner names slots from the corpus alone. A lexicon is only a rename
layer on top: it maps words that appear next to a slot to the name your
downstream systems expect ("bal" and "رصيد" -> ``balance``). Shipped lexicons
live in :mod:`patgen.lexicons` and are examples, not defaults — nothing in the
pipeline assumes a domain unless one is passed.

File format (either direction works)::

    {"balance": ["balance", "bal", "رصيد"]}      # name -> words
    {"balance": "balance", "bal": "balance"}      # word -> name
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from importlib import resources
from pathlib import Path

BUNDLED = "patgen.lexicons"


def available() -> list[str]:
    """Names of the lexicons shipped with the package."""
    return sorted(
        path.name.removesuffix(".json")
        for path in resources.files(BUNDLED).iterdir()
        if path.name.endswith(".json")
    )


def load(source: str | Path) -> dict[str, str]:
    """Load a bundled lexicon by name, or a JSON file by path."""
    text = str(source)
    if text in available():
        raw = json.loads(resources.files(BUNDLED).joinpath(f"{text}.json").read_text("utf-8"))
    else:
        raw = json.loads(Path(text).read_text("utf-8"))
    return normalize_lexicon(raw)


def normalize_lexicon(raw: Mapping[str, object]) -> dict[str, str]:
    """Accept both directions of the mapping and return word -> slot name."""
    out: dict[str, str] = {}
    for key, value in raw.items():
        if isinstance(value, str):
            out[key.lower()] = value
            continue
        if isinstance(value, (list, tuple)):
            for word in value:
                out[str(word).lower()] = key
            continue
        raise ValueError(f"lexicon entry {key!r} must be a string or a list of strings")
    return out
