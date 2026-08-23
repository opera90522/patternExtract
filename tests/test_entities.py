from patgen.entities import AMOUNT, CARD, CURRENCY, DATE, NUM, mask_tokens
from patgen.learn import prepare
from patgen.naming import SlotNamer
from patgen.template import Literal, Slot, Template
from patgen.tokenizer import tokenize


def keys(text: str) -> list[str]:
    return [m.key for m in mask_tokens(tokenize(text))]


def test_masks_amount_currency_and_card():
    assert keys("purchase of sar 1,250.00 on card ****1234") == [
        "purchase",
        "of",
        f"<{CURRENCY}>",
        f"<{AMOUNT}>",
        "on",
        "card",
        f"<{CARD}>",
    ]


def test_arabic_currency_symbol_is_one_token():
    assert keys("مبلغ 50.00 ر.س") == ["مبلغ", f"<{AMOUNT}>", f"<{CURRENCY}>"]


def test_dates_are_typed():
    assert f"<{DATE}>" in keys("on 29/07/2024")
    assert f"<{DATE}>" in keys("بتاريخ 2024-11-16")


def test_otp_digits_stay_numeric():
    assert keys("your otp is 483920") == ["your", "otp", "is", f"<{NUM}>"]


def test_slot_naming_uses_context():
    templates = [
        Template(
            "t1",
            [
                Literal("available"),
                Literal("balance"),
                Slot("", AMOUNT, 1, 1),
            ],
            count=5,
        ),
        Template(
            "t2",
            [
                Literal("your"),
                Literal("otp"),
                Literal("is"),
                Slot("", NUM, 1, 1),
            ],
            count=5,
        ),
        Template(
            "t3",
            [
                Literal("fee"),
                Slot("", AMOUNT, 1, 1),
            ],
            count=5,
        ),
    ]
    SlotNamer.fit(templates).assign_all(templates)
    assert templates[0].slots[0].key == "balance"
    assert templates[1].slots[0].key == "otp"
    assert templates[2].slots[0].key == "fee"


def test_prepare_masks_after_normalization():
    prepared = prepare("Ø±.Ø³ 1,000.00 خصم")
    assert prepared.keys[:2] == [f"<{CURRENCY}>", f"<{AMOUNT}>"]
    assert prepared.canonical.startswith("ر . س")
