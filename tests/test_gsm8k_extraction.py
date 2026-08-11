"""Pulling the final answer out of a chain-of-thought completion.

Every string here is a real completion from LLaDA-1.5 quantized to W4A4. A
scoring bug here is invisible in the aggregate -- the run finishes, the number
looks plausible, and the method gets blamed for the parser's mistakes.
"""

from __future__ import annotations

from dllmquant.eval import extract_answer, gold_answer


def test_unmarked_answer_in_prose_is_a_known_limitation():
    """The real answer is 7, but the sentence ends '...in 4 weeks'.

    With no marker there is no syntactic way to tell which number is the
    answer, so extraction still returns 4 here. This is documented rather than
    papered over with unit blacklists: the fix belongs in the prompt, which now
    asks for an explicit "The answer is N" line.
    """
    text = "Claire will eat 7 dozens of eggs in 4 weeks."
    assert extract_answer(text) == 4.0  # wrong, and knowably so

    # The same completion with the marker the prompt now requires:
    assert extract_answer(text + "\nThe answer is 7") == 7.0


def test_boxed_wins_over_trailing_text():
    text = (r"So, Skylar needs to pay \(\boxed{34}\) for 16 glasses.")
    assert extract_answer(text) == 34.0


def test_boxed_alone():
    text = r"the total number of bolts of fiber required is \boxed{3}."
    assert extract_answer(text) == 3.0


def test_last_boxed_wins_when_several():
    text = r"first \boxed{5}, corrected to \boxed{12}."
    assert extract_answer(text) == 12.0


def test_answer_is_phrasing():
    assert extract_answer("Therefore, the answer is 260 sheep.") == 260.0
    assert extract_answer("Final Answer: $57500") == 57500.0


def test_bold_marker():
    text = "So, the total cost is **$694** for everything."
    assert extract_answer(text) == 694.0


def test_falls_back_to_the_last_number():
    text = "Adding them up gives 180 + 10 + 80 = 270"
    assert extract_answer(text) == 270.0


def test_commas_and_currency():
    assert extract_answer(r"profit is \boxed{70,000}") == 70000.0
    assert extract_answer("the answer is $1,150.") == 1150.0


def test_no_number_at_all():
    assert extract_answer("I cannot determine the answer.") is None


def test_gold_answer_reads_the_dataset_format():
    assert gold_answer("Janet sells 16 - 3 - 4 = 9 eggs.\n#### 18") == 18.0


def test_real_completions_from_the_w4a4_run():
    """Regression on the exact strings that were mis-scored."""
    cases = [
        ("Therefore, Janet makes $24 every day at the farmers' market.", 24.0),
        ("So, there are **20 cups** of feed** left for the final meal.", 20.0),
        ("the total distance covered by each train in the two days is "
         "**230 miles**.", 230.0),
        ("So, the total number of vacuum cleaners she started with is "
         r"\( \boxed{15} \).", 15.0),
        ("Marissa needs to walk the remaining distance at a speed of "
         "**3 miles per hour**.", 3.0),
    ]
    for text, want in cases:
        assert extract_answer(text) == want, text


# ------------------------------------------------------- truncation counting


def _result(completions):
    from dllmquant.eval.gsm8k import EvalResult

    samples = [
        {"question": "q", "completion": c, "pred": 1.0, "gold": 1.0, "correct": ok}
        for c, ok in completions
    ]
    return EvalResult(
        correct=sum(ok for _, ok in completions),
        total=len(completions),
        samples=samples,
    )


def test_a_reply_that_stops_on_an_operator_is_counted_as_cut_off():
    """A diffusion LM fills a fixed canvas: needing one more line does not get
    you one, it gets you a wrong answer. Reporting accuracy without this
    invites reading a shortfall in gen_length as a shortfall in reasoning."""
    res = _result([
        ("Total = 40 + 20 +", False),
        ("So the total is 60.\n\nThe answer is 60", True),
    ])
    assert res.cut_off == 1
    assert res.cut_off_wrong == 1
    assert "cut off mid-answer" in res.summary()


def test_a_finished_reply_is_not_counted():
    res = _result([("The answer is 18", True), ("The answer is 26.", True)])
    assert res.cut_off == 0
    assert "cut off" not in res.summary()


def test_the_count_is_a_floor_not_a_guess():
    """'The answer is 8' for a model that meant 800 looks complete, and is
    deliberately not counted -- overstating truncation would be worse than
    understating it."""
    res = _result([("Total = $800\n\nThe answer is 8", False)])
    assert res.cut_off == 0
