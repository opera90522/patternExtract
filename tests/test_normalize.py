from patgen.normalize import normalize, repair_mojibake


def test_repairs_fully_corrupted_arabic():
    assert repair_mojibake("Ø±ØµÙŠØ¯Ùƒ") == "رصيدك"


def test_repairs_corruption_in_the_middle_only():
    text = "Bill payment Ø±.Ø³ 37,955.74 to ACME"
    assert repair_mojibake(text) == "Bill payment ر.س 37,955.74 to ACME"


def test_leaves_clean_text_alone():
    text = "Your OTP is 1234"
    assert repair_mojibake(text) == text


def test_normalize_folds_tashkeel_and_arabic_digits():
    assert normalize("رَصيدُك ٢٥٠٫٥٠ ريال") == "رصيدك 250.50 ريال"


def test_normalize_folds_hamza_and_taa_marbuta():
    assert normalize("أحمد") == "احمد"
    assert normalize("خدمة") == "خدمه"
    assert normalize("علىٰ") == "علي"


def test_normalize_strips_bidi_and_collapses_space():
    assert normalize("\u200fسحب  نقدي\u200e") == "سحب نقدي"


def test_normalize_lowercases_latin():
    assert normalize("Your OTP") == "your otp"
