from blockkv.step_time import cost_model, fit_line

GB = 2 ** 30


def test_fit_line_exact():
    a, b = fit_line([0, 1, 2], [5, 7, 9])
    assert abs(a - 5) < 1e-12 and abs(b - 2) < 1e-12


def test_cost_model_saving_and_break_even():
    # 40 ms fixed + 10 ms per GiB of cache
    rows = [dict(block=32, batch=1, cache_bytes=g * GB, ms_step=40 + 10 * g) for g in (1, 2, 4)]
    out = {r["cache_bytes"] // GB: r for r in cost_model(rows, bits=4.0)}
    r = out[4]
    assert abs(r["ms_per_gb"] - 10) < 1e-6 and abs(r["ms_fixed"] - 40) < 1e-6
    # 4 GiB read at 4/16 of the bytes: saves 10 * 4 * 0.75 = 30 ms of 80
    assert abs(r["est_ms_2bit"] - 50) < 1e-6
    assert abs(r["break_even"] - (80 / 50 - 1)) < 1e-4


def test_memory_bound_ideal():
    # 16 GB of weights, 16 GB of cache, 4 TB/s: 8 ms at 16 bit; cache at 4/16: 5 ms
    rows = [dict(block=32, batch=1, cache_bytes=16e9, ms_step=50.0), dict(block=32, batch=1, cache_bytes=8e9, ms_step=45.0)]
    r = [x for x in cost_model(rows, bits=4.0, weight_bytes=16e9, bandwidth_tbs=4.0) if x["cache_bytes"] == 16e9][0]
    assert abs(r["ideal_ms_16"] - 8.0) < 1e-9 and abs(r["ideal_ms_2bit"] - 5.0) < 1e-9
    assert abs(r["ideal_break_even"] - 0.6) < 1e-9


def test_cost_model_groups_by_batch():
    rows = [dict(block=32, batch=b, cache_bytes=g * GB, ms_step=b * (40 + 10 * g)) for b in (1, 4) for g in (1, 2)]
    by = {(r["batch"], r["cache_bytes"] // GB): r for r in cost_model(rows)}
    assert abs(by[(4, 2)]["ms_per_gb"] - 40) < 1e-6 and abs(by[(1, 2)]["ms_per_gb"] - 10) < 1e-6
