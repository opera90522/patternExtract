"""Entity layer: turn tokens into placeholders before clustering.

Masking before clustering is what keeps the template count bounded: two SMS
that differ only in an amount or a card tail collapse into one cluster on the
very first pass instead of relying on the similarity threshold.

Every entity type has two faces:

* a token-level detector used by the learning path (:func:`mask_tokens`);
* a regex over *canonical text* (tokens joined by single spaces) used by the
  production matcher, so both paths agree on what a value looks like.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from .tokenizer import Token

# ---------------------------------------------------------------------------
# slot naming was here; it is now corpus-driven in .naming and optional
# via .lexicon. This module only does token-level entity masking.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# entity vocabulary
# ---------------------------------------------------------------------------

URL = "URL"
EMAIL = "EMAIL"
IBAN = "IBAN"
CARD = "CARD"
ACCOUNT = "ACCOUNT"
PHONE = "PHONE"
DATE = "DATE"
TIME = "TIME"
DATETIME = "DATETIME"
AMOUNT = "AMOUNT"
PERCENT = "PERCENT"
CURRENCY = "CURRENCY"
CODE = "CODE"
REF = "REF"
NUM = "NUM"
TEXT = "TEXT"  # free wildcard discovered by clustering (merchant names, ...)

#: Regex used by the matcher, expressed over canonical (space joined) text.
ENTITY_REGEX = {
    URL: r"(?:https?://|www\.)\S+",
    EMAIL: r"[^\s@]+@[^\s@]+\.[^\s@]+",
    IBAN: r"[a-z]{2}\d{2}[a-z0-9]{10,30}",
    CARD: r"(?:\*|x)(?: ?(?:\*|x))* ?\d{2,6}|\d{12,19}",
    ACCOUNT: r"(?:\*|x)(?: ?(?:\*|x))* ?\d{2,10}",
    PHONE: r"\+ ?\d{8,15}|00?\d{8,13}|9\d{10,14}",
    DATE: r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}",
    TIME: r"\d{1,2}:\d{2}(?::\d{2})?(?: ?(?:am|pm))?",
    DATETIME: r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}[ t]\d{1,2}:\d{2}(?::\d{2})?",
    AMOUNT: r"\d[\d,]*\.\d{1,3}|\d{1,3}(?:,\d{3})+",
    PERCENT: r"\d+(?:\.\d+)? ?%",
    CURRENCY: "",  # filled below from the currency lexicon
    CODE: r"[a-z0-9]{4,10}",
    REF: r"[a-z0-9][a-z0-9.-]{3,}",
    NUM: r"\d+",
    TEXT: r"\S+",
}

CURRENCY_WORDS = frozenset(
    """sar aed usd eur egp kwd qar bhd omd jod gbp
    ريال درهم جنيه دينار دولار يورو""".split()
)

#: Currency symbols that the tokenizer necessarily splits ("ر.س" -> ر . س).
#: Matched as a run of up to four tokens whose concatenation is a known symbol.
CURRENCY_SYMBOLS = frozenset(["ر.س", "رس", "د.إ", "د.ا", "ج.م", "د.ك", "ل.س", "$", "€", "£"])
MAX_CURRENCY_TOKENS = 4

#: Digit runs next to these words are references/codes, never phone numbers.
_REF_CONTEXT = frozenset(
    "ref reference trx txn id no مرجع رقم العمليه العملية".split()
)
_PHONE_CONTEXT = frozenset("call mobile phone tel اتصل جوال هاتف".split())


def _currency_regex() -> str:
    """Match currencies in canonical text, where "ر.س" appears as "ر . س"."""
    alternatives = []
    for symbol in sorted(CURRENCY_SYMBOLS | CURRENCY_WORDS, key=len, reverse=True):
        alternatives.append(" ?".join(re.escape(ch) for ch in symbol))
    return "|".join(alternatives)


ENTITY_REGEX[CURRENCY] = _currency_regex()

# ---------------------------------------------------------------------------
# token level detectors
# ---------------------------------------------------------------------------

_RE_IBAN = re.compile(r"^[a-z]{2}\d{2}[a-z0-9]{10,30}$")
_RE_LONG_DIGITS = re.compile(r"^\d{12,19}$")
# A bare 9 digit run is far more often a reference than a phone number, so a
# phone needs either an explicit "+" or a plausible national length.
_RE_PHONE = re.compile(r"^(?:\+\d{8,15}|0\d{8,13}|00\d{8,13}|9\d{10,14})$")
_RE_DATE = re.compile(r"^\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}$")
_RE_TIME = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?$")
_RE_AMOUNT = re.compile(r"^\d[\d,]*\.\d{1,3}$|^\d{1,3}(,\d{3})+$")
_RE_INT = re.compile(r"^\d+$")
_RE_REF = re.compile(r"^(?=[a-z0-9.-]*\d)(?=[a-z0-9.-]*[a-z])[a-z0-9][a-z0-9.-]{3,}$")
_RE_MASKCHARS = re.compile(r"^[*x#]+$")


@dataclass(frozen=True, slots=True)
class MaskedToken:
    """A token, or a run of tokens, after entity substitution."""

    key: str  # what the clusterer sees: literal text or "<TYPE>"
    entity: str | None  # entity type when masked, else None
    value: str  # original (canonical) text of the span
    start_index: int  # index of first source token
    end_index: int  # index after the last source token

    @property
    def is_entity(self) -> bool:
        return self.entity is not None


def _classify_number(tok: Token, prev: Token | None, nxt: Token | None) -> str:
    text = tok.text
    if _RE_DATE.match(text):
        return DATE
    if _RE_TIME.match(text):
        return TIME
    if _RE_AMOUNT.match(text):
        return AMOUNT
    prev_text = prev.text if prev is not None else ""
    if prev_text in _REF_CONTEXT:
        return NUM
    if prev_text in _PHONE_CONTEXT:
        return PHONE
    if _RE_LONG_DIGITS.match(text) and not _RE_PHONE.match(text):
        return CARD
    if _RE_PHONE.match(text) or prev_text == "+":
        return PHONE
    if nxt is not None and nxt.text == "%":
        return PERCENT
    if _RE_INT.match(text):
        # A plain integer next to a currency word is money, not a counter.
        if (prev is not None and prev.text in CURRENCY_WORDS) or (
            nxt is not None and nxt.text in CURRENCY_WORDS
        ):
            return AMOUNT
        return NUM
    return REF


def _currency_run(tokens: Sequence[Token], start: int) -> int:
    """Length of a currency symbol spelled over several tokens, else 0."""
    limit = min(MAX_CURRENCY_TOKENS, len(tokens) - start)
    for width in range(limit, 0, -1):
        joined = "".join(t.text for t in tokens[start : start + width])
        if joined in CURRENCY_SYMBOLS:
            return width
    return 0


def mask_tokens(tokens: Sequence[Token]) -> list[MaskedToken]:
    """Replace value-bearing tokens (and short runs) with typed placeholders."""
    out: list[MaskedToken] = []
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        prev = tokens[i - 1] if i else None
        nxt = tokens[i + 1] if i + 1 < n else None

        # multi token: split currency symbols -> "ر . س"
        width = _currency_run(tokens, i)
        if width:
            span = tokens[i : i + width]
            out.append(
                MaskedToken(
                    f"<{CURRENCY}>",
                    CURRENCY,
                    " ".join(t.text for t in span),
                    i,
                    i + width,
                )
            )
            i += width
            continue

        # multi token: masked card / account tails -> "**** 1234", "xxxx1234"
        j = i
        while j < n and (_RE_MASKCHARS.match(tokens[j].text) or tokens[j].text == "*"):
            j += 1
        if j > i and j < n and tokens[j].kind == "num" and len(tokens[j].text) <= 6:
            span = tokens[i : j + 1]
            value = " ".join(t.text for t in span)
            out.append(MaskedToken(f"<{CARD}>", CARD, value, i, j + 1))
            i = j + 1
            continue

        # multi token: date followed by time -> single DATETIME slot
        if (
            tok.kind == "num"
            and _RE_DATE.match(tok.text)
            and nxt is not None
            and _RE_TIME.match(nxt.text)
        ):
            out.append(
                MaskedToken(
                    f"<{DATETIME}>", DATETIME, f"{tok.text} {nxt.text}", i, i + 2
                )
            )
            i += 2
            continue

        if tok.kind == "url":
            entity: str | None = URL
        elif tok.kind == "email":
            entity = EMAIL
        elif tok.kind == "num":
            entity = _classify_number(tok, prev, nxt)
        elif tok.kind == "word":
            if _RE_IBAN.match(tok.text):
                entity = IBAN
            elif tok.text in CURRENCY_WORDS or tok.text in CURRENCY_SYMBOLS:
                entity = CURRENCY
            elif _RE_REF.match(tok.text):
                entity = REF
            else:
                entity = None
        else:
            entity = None

        if entity is None:
            out.append(MaskedToken(tok.text, None, tok.text, i, i + 1))
        else:
            out.append(MaskedToken(f"<{entity}>", entity, tok.text, i, i + 1))
        i += 1
    return out


# Slot naming is now handled by patgen.naming, which derives names from the
# corpus and optional user lexicon rather than a fixed domain vocabulary.
