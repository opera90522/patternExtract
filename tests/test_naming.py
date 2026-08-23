"""Slot naming is corpus-driven and optional lexicon-aware."""

from patgen.entities import AMOUNT, NUM, TEXT
from patgen.naming import SlotNamer
from patgen.template import Literal, Slot, Template


def _make(*parts: str | Slot, count: int = 1) -> Template:
    return Template(
        "tx",
        [(Literal(p) if isinstance(p, str) else p) for p in parts],
        count=count,
    )


def test_generic_name_uses_most_predictive_word():
    # "order" is rare and always before <NUM>; "your" is common and not.
    templates = [
        _make("your", "order", Slot("", NUM, 1, 1), count=5),
        _make("your", "balance", Slot("", AMOUNT, 1, 1), count=5),
    ]
    SlotNamer.fit(templates).assign_all(templates)
    assert templates[0].slots[0].key == "order"
    assert templates[1].slots[0].key == "balance"


def test_lexicon_overrides_corpus_scores():
    templates = [
        _make("payment", Slot("", AMOUNT, 1, 1), count=5),
    ]
    SlotNamer.fit(templates, lexicon={"payment": "amount"}).assign_all(templates)
    assert templates[0].slots[0].key == "amount"


def test_glue_words_do_not_become_names():
    # Many templates contain "is" before a slot, so "is" has low purity and score.
    templates = [
        _make("is", Slot("", NUM, 1, 1), "otp", count=3),
        _make("is", Slot("", NUM, 1, 1), "ref", count=3),
    ]
    SlotNamer.fit(templates).assign_all(templates)
    # "is" is low purity, so we fall back to the entity name.
    assert templates[0].slots[0].key == "num"


def test_unique_keys_for_multiple_same_entity_slots():
    templates = [
        _make("amount", Slot("", AMOUNT, 1, 1), "fee", Slot("", AMOUNT, 1, 1), count=3),
    ]
    SlotNamer.fit(templates).assign_all(templates)
    keys = [s.key for s in templates[0].slots]
    assert keys == ["amount", "fee"]


def test_text_slots_named_by_rare_predictive_word():
    templates = [
        _make("merchant", Slot("", TEXT, 1, 2), count=3),
    ]
    SlotNamer.fit(templates).assign_all(templates)
    assert templates[0].slots[0].key == "merchant"
