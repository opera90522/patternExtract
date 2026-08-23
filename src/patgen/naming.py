"""Slot naming learned from the corpus, with no domain vocabulary built in.

A slot's name should be the word the writer of the message already used for
it ("balance <AMOUNT>", "order <REF>", "cpu <PERCENT>"). That word is found by
looking at the nearest informative literal on the left of the value and
scoring each candidate by how strongly it predicts the slot's entity type.

* rare words across templates get a higher weight (idf);
* words that consistently sit next to the same entity type get a higher weight
  (purity);
* common glue words (prepositions, pronouns, copulas) are ignored.

The result is scaled by the best score in the corpus, so the same threshold
works for a 20-template corpus and a 2000-template one.

An optional lexicon (:mod:`patgen.lexicon`) can still be supplied to rename
slots into a house vocabulary. It maps a literal word to a semantic name, and
``_COMPATIBILITY`` makes sure the semantic name only applies to entity types
that can reasonably hold it (e.g. a word mapped to ``amount`` is not used to
name a free-text merchant slot).
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from .entities import (
    ACCOUNT,
    AMOUNT,
    CARD,
    CODE,
    CURRENCY,
    DATE,
    DATETIME,
    EMAIL,
    IBAN,
    NUM,
    PERCENT,
    PHONE,
    REF,
    TEXT,
    TIME,
    URL,
)
from .template import Literal, Part, Slot, Template

#: How far to the left a qualifier may sit ("available balance is <AMOUNT>").
LEFT_WINDOW = 3
#: Proximity discount per extra word, so the nearest strong candidate wins.
DECAY = 0.5
#: Scaled score under which we treat a word as glue and fall back to the entity.
MIN_SCORE = 0.2

#: Generic glue words across the corpora we expect (English + Arabic starters).
#: These are not domain rules; they are language-level function words that
#: almost never name a slot.
DEFAULT_STOPLIST = frozenset(
    """
    a an the is are was were be been being have has had do does did will would
    shall should can could may might must of in on at to for with from by about
    into through during before after above below between under and or but yet so
    if than then as this that these those my your his her its our their me you
    him her it us them what which who whose where when why how all any both each
    few more most other some such no nor not only own same so than too very just
    now
    declined confirmed updated shipped delivered activated succeeded completed
    received rejected approved sent arrived looking resolved replied started
    failed passed done
    في من إلى على عن مع ب ل ك و أو لكن ثم هناك هذا هذه ذو ذلك التي الذي هم هي
    نحن أنا أنت هو هي إنه إنها أن قد كان لم لا ليس ما كم هل لقد كانت يكون تم
    قام يجب عليه عليها لك له لنا علي عنده لديه لديها به عند كان قد سوف لا
    بك بكم لك لكم لدي لدى لديك لديكم
    تم تأكيد تمت رد شكرا نجح فشل اكتمل وصل بدأ حل
    """.split()
)

#: Entity types whose meaning is obvious from their syntax; context naming is
#: skipped for these unless an explicit lexicon maps a nearby word to a key.
_NO_CONTEXT: frozenset[str] = frozenset({
    CURRENCY, DATE, TIME, DATETIME, PHONE, EMAIL, URL, PERCENT
})

#: Default slot key for each entity type. These are used when the corpus does
#: not provide a clear label, so the extracted JSON is still human-readable.
_ENTITY_DEFAULTS = {
    AMOUNT: "amount",
    CURRENCY: "currency",
    DATE: "date",
    TIME: "time",
    DATETIME: "datetime",
    PHONE: "phone",
    CARD: "card",
    ACCOUNT: "account",
    IBAN: "iban",
    EMAIL: "email",
    URL: "url",
    PERCENT: "percent",
    NUM: "num",
    CODE: "code",
    REF: "ref",
    TEXT: "text",
}

#: Semantic slot name -> entity types that are allowed to carry it.
#: This is generic type safety, not domain knowledge.
_COMPATIBILITY: dict[str, frozenset[str]] = {
    "amount": frozenset({AMOUNT, NUM, PERCENT}),
    "balance": frozenset({AMOUNT, NUM, PERCENT}),
    "fee": frozenset({AMOUNT, NUM, PERCENT}),
    "limit": frozenset({AMOUNT, NUM, PERCENT}),
    "purchase": frozenset({AMOUNT, NUM, PERCENT}),
    "payment": frozenset({AMOUNT, NUM, PERCENT}),
    "transfer": frozenset({AMOUNT, NUM, PERCENT}),
    "withdrawal": frozenset({AMOUNT, NUM, PERCENT}),
    "deposit": frozenset({AMOUNT, NUM, PERCENT}),
    "card": frozenset({CARD, NUM, TEXT, ACCOUNT}),
    "account": frozenset({ACCOUNT, NUM, TEXT, CARD, IBAN}),
    "iban": frozenset({IBAN, ACCOUNT, TEXT}),
    "merchant": frozenset({TEXT, NUM, REF, ACCOUNT}),
    "beneficiary": frozenset({TEXT, NUM, REF, ACCOUNT}),
    "service": frozenset({TEXT, NUM, REF}),
    "ref": frozenset({REF, NUM, CODE, TEXT}),
    "reference": frozenset({REF, NUM, CODE, TEXT}),
    "otp": frozenset({CODE, NUM, REF}),
    "code": frozenset({CODE, NUM, REF}),
    "pin": frozenset({CODE, NUM, REF}),
    "password": frozenset({CODE, NUM, REF}),
    "date": frozenset({DATE, DATETIME, TIME}),
    "time": frozenset({TIME, DATETIME, DATE}),
    "phone": frozenset({PHONE, NUM, TEXT}),
    "mobile": frozenset({PHONE, NUM, TEXT}),
    "currency": frozenset({CURRENCY, AMOUNT, NUM}),
    "percent": frozenset({PERCENT, AMOUNT, NUM}),
}

_IDENT = re.compile(r"[^\w]+", re.UNICODE)


def _is_word(text: str) -> bool:
    return len(text) > 1 and any(ch.isalnum() for ch in text)


def _safe_key(name: str) -> str:
    name = _IDENT.sub("_", name.lower()) or "slot"
    if name[0].isdigit():
        name = f"s_{name}"
    return name


def _left_candidates(
    parts: Sequence[Part], index: int, stoplist: frozenset[str]
) -> list[tuple[str, float]]:
    """Words left of the slot that could name it, with proximity weight.

    Other slots are skipped so that a unit word like a currency does not block
    a label further back ("available balance <CURRENCY> <AMOUNT>").
    """
    out: list[tuple[str, float]] = []
    distance = 0
    seen = 0
    for j in range(index - 1, -1, -1):
        if seen >= LEFT_WINDOW:
            break
        part = parts[j]
        if isinstance(part, Slot):
            continue
        text = part.text
        if not _is_word(text):
            continue
        if text.lower() in stoplist:
            continue
        out.append((text, DECAY**distance))
        distance += 1
        seen += 1
    return out


@dataclass
class SlotNamer:
    """Names slots from corpus statistics, optionally through a lexicon."""

    scores: Mapping[str, float]
    lexicon: Mapping[str, str]
    stoplist: frozenset[str] = field(default=DEFAULT_STOPLIST)
    min_score: float = MIN_SCORE

    @classmethod
    def fit(
        cls,
        templates: Sequence[Template],
        lexicon: Mapping[str, str] | None = None,
        stoplist: frozenset[str] | None = None,
        min_score: float = MIN_SCORE,
    ) -> SlotNamer:
        document_frequency: Counter = Counter()
        next_to: dict[str, Counter] = defaultdict(Counter)
        for template in templates:
            document_frequency.update(
                {p.text for p in template.parts if isinstance(p, Literal) and _is_word(p.text)}
            )
            for index, part in enumerate(template.parts):
                if not isinstance(part, Slot):
                    continue
                for word, weight in _left_candidates(
                    template.parts, index, stoplist or DEFAULT_STOPLIST
                ):
                    next_to[word][part.entity] += weight
        total = max(len(templates), 1)
        raw: dict[str, float] = {}
        for word, entities in next_to.items():
            idf = math.log(1 + total / document_frequency.get(word, 1))
            purity = max(entities.values()) / sum(entities.values())
            raw[word] = idf * purity
        ceiling = max(raw.values(), default=1.0) or 1.0
        return cls(
            scores={word: value / ceiling for word, value in raw.items()},
            lexicon=lexicon or {},
            stoplist=stoplist or DEFAULT_STOPLIST,
            min_score=min_score,
        )

    def _is_compatible(self, name: str, entity: str) -> bool:
        allowed = _COMPATIBILITY.get(name)
        return allowed is None or entity in allowed

    def _lexicon_name(self, word: str, entity: str, taken: set[str]) -> str | None:
        mapped = self.lexicon.get(word.lower())
        if not mapped:
            return None
        if mapped not in taken and self._is_compatible(mapped, entity):
            return mapped
        return None

    def _fallback(self, entity: str) -> str:
        return _ENTITY_DEFAULTS.get(entity, entity.lower())

    def name(self, parts: Sequence[Part], index: int, taken: set[str]) -> str:
        """Best available name for the slot at ``index``."""
        slot = parts[index]
        assert isinstance(slot, Slot)

        # Lexicon takes precedence over corpus scores.
        for word, _ in _left_candidates(parts, index, self.stoplist):
            mapped = self._lexicon_name(word, slot.entity, taken)
            if mapped:
                return mapped

        ranked = sorted(
            (
                (self.scores.get(word, 0.0) * weight, word)
                for word, weight in _left_candidates(parts, index, self.stoplist)
            ),
            reverse=True,
        )
        if slot.entity in _NO_CONTEXT:
            return self._fallback(slot.entity)

        for score, word in ranked:
            if score < self.min_score:
                break
            if word in taken:
                continue
            mapped = self.lexicon.get(word.lower(), word)
            if mapped != word:
                if mapped not in taken and self._is_compatible(mapped, slot.entity):
                    return mapped
                continue
            return word

        return self._fallback(slot.entity)

    def assign(self, template: Template) -> None:
        """Give every slot of ``template`` a unique, readable key."""
        used: Counter = Counter()
        taken: set[str] = set()
        for index, part in enumerate(template.parts):
            if not isinstance(part, Slot):
                continue
            raw = self.name(template.parts, index, taken)
            name = _safe_key(raw)
            taken.add(name)
            used[name] += 1
            part.key = name if used[name] == 1 else f"{name}_{used[name]}"

    def assign_all(self, templates: Sequence[Template]) -> None:
        """Name slots for every template."""
        for template in templates:
            self.assign(template)
