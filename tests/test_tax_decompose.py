import math

from blockkv.tax_decompose import decompose


def rows(spec):
    """{idx: (gen_tokens, steps)} -> rows."""
    return {i: dict(gen_tokens=t, steps=s) for i, (t, s) in spec.items()}


def test_pure_tax_has_no_length_share():
    ra = rows({0: (100, 40), 1: (200, 80)})
    rb = rows({0: (100, 50), 1: (200, 100)})          # same answers, 25% more steps
    d = decompose(ra, rb, cap=512)
    assert abs(d["steps"] - 1.25) < 1e-9 and abs(d["length"] - 1) < 1e-9
    assert abs(d["from_length"]) < 1e-9 and abs(d["tps"] - 0.8) < 1e-9


def test_runaways_are_split_off():
    ra = rows({0: (100, 40), 1: (200, 80), 2: (150, 60)})
    rb = rows({0: (100, 40), 1: (200, 80), 2: (500, 200)})   # one answer runs away, same tok/step
    d = decompose(ra, rb, cap=500)
    assert d["loops_run"] == 1 and d["n_clean"] == 2
    assert abs(d["from_length"] - 1) < 1e-6                     # all the extra steps come from length
    assert abs(d["clean"]["steps"] - 1) < 1e-9


def test_identity_holds():
    ra = rows({0: (120, 50), 1: (300, 90)})
    rb = rows({0: (160, 80), 1: (310, 120)})
    d = decompose(ra, rb, cap=512)
    assert math.isclose(d["steps"], d["length"] / d["tps"])
