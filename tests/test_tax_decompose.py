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


def test_repetition_counts_as_a_loop_below_the_cap():
    from blockkv.tax_decompose import repetitive
    varied = " ".join(f"row {i} col {i * 7} value {i * 13}" for i in range(200))
    looped = varied[:500] + " the answer is 42 and so" * 200
    assert not repetitive(varied) and repetitive(looped)
    ra = {0: dict(gen_tokens=100, steps=40, text=varied), 1: dict(gen_tokens=300, steps=100, text=varied)}
    rb = {0: dict(gen_tokens=100, steps=50, text=varied), 1: dict(gen_tokens=400, steps=200, text=looped)}
    rep = dict(n=10, tail=600, min_distinct=0.3)
    d = decompose(ra, rb, cap=9216, repeat=rep)
    assert d["loops_run"] == 1 and d["cap_run"] == 0 and d["n_clean"] == 1
    assert abs(d["in_loops"] - 100 / 110) < 1e-9
    assert decompose(ra, rb, cap=9216)["loops_run"] == 0          # cap only: not caught


def test_clean_tok_per_step_interval_and_task_filter():
    from blockkv.tax_decompose import main
    import json, tempfile, os
    ra = {f"tom:{i}": dict(idx=f"tom:{i}", gen_tokens=100 + i, steps=40 + i % 3, text="a b") for i in range(10)}
    rb = {f"tom:{i}": dict(idx=f"tom:{i}", gen_tokens=100 + i, steps=50 + i % 3, text="a b") for i in range(10)}
    ra["path:0"] = dict(idx="path:0", gen_tokens=100, steps=40, text="a b")
    rb["path:0"] = dict(idx="path:0", gen_tokens=100, steps=400, text="a b")
    d = decompose(ra, rb, cap=3072)
    lo, hi = d["clean_tps_ci"]
    assert lo <= d["clean"]["tps"] - 1 <= hi
    with tempfile.TemporaryDirectory() as t:
        for name, r in (("a", ra), ("b", rb)):
            with open(os.path.join(t, name + ".jsonl"), "w") as f:
                f.write("\n".join(json.dumps(x) for x in r.values()))
        main(["--cap", "3072", "--tasks", "tom", f"{t}/a.jsonl::{t}/b.jsonl"])
