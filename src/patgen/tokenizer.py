"""Tokenizer shared by the learning and production paths."""

from __future__ import annotations

import re
from dataclasses import dataclass

ARABIC_RANGE = "\u0600-\u06ff\u0750-\u077f\ufb50-\ufdff\ufe70-\ufeff"

_TOKEN_RE = re.compile(
    r"""
    (?P<url>(?:https?://|www\.)[^\s]+)
  | (?P<email>[\w.+-]+@[\w-]+\.[\w.-]+)
  | (?P<num>\d[\d,._:/\\-]*\d|\d)
  | (?P<word>[a-zA-Z]+(?:['\u2019][a-zA-Z]+)*|[""" + ARABIC_RANGE + r"""]+)
  | (?P<mixed>[^\s\w]+|\w+)
    """,
    re.VERBOSE,
)


@dataclass(frozen=True, slots=True)
class Token:
    text: str
    kind: str  # url | email | num | word | mixed
    start: int
    end: int

    @property
    def is_arabic(self) -> bool:
        return bool(_ARABIC_CHAR.search(self.text))


_ARABIC_CHAR = re.compile("[" + ARABIC_RANGE + "]")
_PUNCT_ONLY = re.compile(r"^[^\w\s]+$")


def tokenize(text: str) -> list[Token]:
    """Split normalized text into typed tokens.

    Punctuation runs are kept as their own tokens: they are highly stable in
    SMS templates and therefore useful anchors, but they must not glue onto
    neighbouring values (``sar.100`` and ``sar 100`` should generalize alike).
    """
    tokens: list[Token] = []
    for m in _TOKEN_RE.finditer(text):
        kind = m.lastgroup or "mixed"
        raw = m.group()
        if kind == "mixed" and _PUNCT_ONLY.match(raw):
            for i, ch in enumerate(raw):
                tokens.append(Token(ch, "punct", m.start() + i, m.start() + i + 1))
            continue
        tokens.append(Token(raw, kind, m.start(), m.end()))
    return tokens
