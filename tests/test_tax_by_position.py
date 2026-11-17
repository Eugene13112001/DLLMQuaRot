from blockkv.tax_by_position import by_position, segments


def test_segments_split_steps_by_answer_position():
    assert segments([4, 4, 4, 4], [0, 8, 16]) == [[8, 2], [8, 2]]


def test_tax_that_grows_along_the_answer():
    # ref: 4 tokens/step everywhere; run: 4 in the first half, 2 in the second
    ra = {i: dict(gen_tokens=64, steps=16, per_step=[4] * 16, text="x") for i in range(5)}
    rb = {i: dict(gen_tokens=64, steps=24, per_step=[4] * 8 + [2] * 16, text="x") for i in range(5)}
    rows = by_position(ra, rb, cap=3072, edges=[0, 32, 64])
    assert abs(rows[0]["ratio"] - 1) < 1e-9 and abs(rows[1]["ratio"] - 0.5) < 1e-9
