"""Text normalization for messy bilingual (Arabic/English) SMS text.

The learning and production paths must see byte-identical input, so every
transformation lives here and is applied through :func:`normalize`.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Characters that carry no information for template mining but do break
# tokenization: bidi marks, zero width joiners, soft hyphen, BOM.
_INVISIBLE = re.compile(
    "[\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff\u00ad\u180e]"
)
_CONTROL = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WS = re.compile(r"\s+")

# Arabic diacritics (tashkeel) and tatweel.
_TASHKEEL = re.compile("[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed\u0640]")

_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

_ARABIC_LETTER_FOLD = str.maketrans(
    {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ة": "ه",
        "ى": "ي",
        "ؤ": "و",
        "ئ": "ي",
        "ک": "ك",
        "ی": "ي",
    }
)

# Punctuation that appears in both scripts, folded to the ASCII form.
_PUNCT_FOLD = str.maketrans(
    {
        "،": ",",
        "؛": ";",
        "؟": "?",
        "٪": "%",
        "۔": ".",
        "٫": ".",
        "٬": ",",
        "“": '"',
        "”": '"',
        "„": '"',
        "‘": "'",
        "’": "'",
        "–": "-",
        "—": "-",
        "‐": "-",
        "−": "-",
        "…": "...",
    }
)

# Replacement char and lone surrogates left over from a broken decode.
_UNDECODABLE = re.compile("[\ufffd\ud800-\udfff]+")

_MOJIBAKE_HINT = re.compile("[ÂÃÐØÙÚÛâãðøùÅå][\u0080-\u00ff\u0152-\u0178]")
# A run of characters that could plausibly be UTF-8 bytes shown as cp1252.
_MOJIBAKE_RUN = re.compile("[\u0080-\u024f\u0192\u02c6\u2013-\u201e\u2020-\u2122]+")


@dataclass(frozen=True)
class NormalizeConfig:
    fix_mojibake: bool = True
    fold_arabic: bool = True
    strip_tashkeel: bool = True
    arabic_digits_to_ascii: bool = True
    lowercase: bool = True
    drop_undecodable: bool = True


DEFAULT_CONFIG = NormalizeConfig()


def _repair_run(run: str, max_rounds: int = 3) -> str:
    for _ in range(max_rounds):
        if not _MOJIBAKE_HINT.search(run):
            return run
        for codec in ("cp1252", "latin-1"):
            try:
                candidate = run.encode(codec, errors="strict").decode(
                    "utf-8", errors="strict"
                )
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
            if candidate == run:
                return run
            run = candidate
            break
        else:
            return run
    return run


_RESIDUE_CLASS = "[\u00a1-\u00ff\u0152-\u0192\u02c6\u2013-\u2122]"
# Latin-1/cp1252 leftovers glued to Arabic: the other half of the UTF-8 pair
# was dropped by the broken pipeline, so nothing can decode them back.
_RESIDUE = re.compile(
    f"(?:{_RESIDUE_CLASS}{{1,3}}\\s*(?=[\u0600-\u06ff])"
    f"|(?<=[\u0600-\u06ff])\\s*{_RESIDUE_CLASS}{{1,3}})"
)


def drop_mojibake_residue(text: str) -> str:
    """Delete undecodable mojibake fragments sitting next to Arabic text."""
    return _RESIDUE.sub(" ", text)


def repair_mojibake(text: str) -> str:
    """Undo UTF-8 text that was decoded as cp1252/latin-1 ("Ø£Ù..." garbage).

    Repair is applied per suspicious *run* rather than to the whole string:
    real exports are often only partially corrupted (one field passed through
    a broken pipeline), and a whole-string round trip fails as soon as any
    correctly decoded Arabic is present. Runs are retried while they keep
    decoding, since double and triple encoding are both common.
    """
    if not _MOJIBAKE_HINT.search(text):
        return text
    return drop_mojibake_residue(_MOJIBAKE_RUN.sub(lambda m: _repair_run(m.group()), text))


def normalize(text: str, config: NormalizeConfig = DEFAULT_CONFIG) -> str:
    if not text:
        return ""
    if config.fix_mojibake:
        text = repair_mojibake(text)
    text = unicodedata.normalize("NFKC", text)
    if config.drop_undecodable:
        text = _UNDECODABLE.sub(" ", text)
    text = _INVISIBLE.sub("", text)
    text = _CONTROL.sub(" ", text)
    if config.strip_tashkeel:
        text = _TASHKEEL.sub("", text)
    if config.arabic_digits_to_ascii:
        text = text.translate(_ARABIC_DIGITS)
    text = text.translate(_PUNCT_FOLD)
    if config.fold_arabic:
        text = text.translate(_ARABIC_LETTER_FOLD)
    if config.lowercase:
        text = text.lower()
    return _WS.sub(" ", text).strip()
