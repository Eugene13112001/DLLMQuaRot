import pytest

from blockkv import gsm8k


@pytest.mark.parametrize("text,want", [
    ("so the total is \\boxed{1,234}.", "1234"),
    ("first \\boxed{3} then \\boxed{\\frac{7}{1}} wait no \\boxed{18}", "18"),
    ("\\boxed{\\$42.00}", "42"),
    ("blah\n#### 72", "72"),
    ("The answer is $18.", "18"),
    ("He has 3 apples and 4 pears, 7 fruits", "7"),
    ("-5 degrees", "-5"),
    ("no number", ""),
])
def test_extract(text, want):
    assert gsm8k.extract(text) == want


def test_gold_and_compare():
    assert gsm8k.gold("Natalia sold 48+24 = <<48+24=72>>72 clips.\n#### 72") == "72"
    assert gsm8k.gold("...\n#### 1,000") == "1000"
    assert gsm8k.is_correct("72.0", "72") and not gsm8k.is_correct("71", "72")
