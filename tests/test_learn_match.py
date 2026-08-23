import json

from patgen import LearnConfig, TemplateLibrary, TemplateMatcher, learn_templates

ENGLISH = [
    f"Purchase of SAR {i}0.50 at store{i} on card ****{1000 + i}. Available balance SAR {i}00.00"
    for i in range(1, 40)
]
ARABIC = [
    f"سحب نقدي {i}00.00 ر.س من الحساب ***{100 + i} بتاريخ 0{i % 9 + 1}/07/2024"
    for i in range(1, 40)
]
OTP = [f"Your OTP is {100000 + i}. Do not share it with anyone" for i in range(1, 40)]


def build():
    return learn_templates(ENGLISH + ARABIC + OTP, LearnConfig(min_support=2))


def test_learns_one_template_per_family():
    library = build()
    assert 3 <= len(library.templates) <= 6


def test_templates_expose_finance_slots():
    library = build()
    slots = {slot.key for template in library.templates for slot in template.slots}
    assert {"amount", "balance", "card", "account", "date", "otp"} <= slots


def test_matcher_extracts_entities():
    matcher = TemplateMatcher(build())
    result = matcher.match(
        "Purchase of SAR 90.50 at store9 on card ****1009. Available balance SAR 900.00"
    )
    assert result is not None
    assert result.entities["amount"] == "90.50"
    assert result.entities["balance"] == "900.00"
    assert result.entities["card"] == "****1009"


def test_matcher_handles_mojibake_and_arabic_digits():
    matcher = TemplateMatcher(build())
    result = matcher.match("Ø³Ø­Ø¨ Ù†Ù‚Ø¯ÙŠ ٥٠٠.٠٠ ر.س من الحساب ***105 بتاريخ 05/07/2024")
    assert result is not None
    assert result.entities["amount"] == "500.00"
    assert result.entities["account"] == "***105"


def test_matcher_returns_none_for_unknown_traffic():
    matcher = TemplateMatcher(build())
    assert matcher.match("hello there, how are you doing today ?") is None


def test_library_round_trips_through_json(tmp_path):
    library = build()
    path = tmp_path / "model.json"
    library.save(path)
    json.loads(path.read_text(encoding="utf-8"))
    reloaded = TemplateLibrary.load(path)
    assert [t.text for t in reloaded.templates] == [t.text for t in library.templates]
    assert TemplateMatcher(reloaded).match(OTP[0]) is not None


def test_coverage_on_training_traffic():
    messages = ENGLISH + ARABIC + OTP
    matcher = TemplateMatcher(build())
    matched = sum(1 for result in matcher.match_many(messages) if result is not None)
    assert matched == len(messages)
