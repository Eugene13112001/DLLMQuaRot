from blockkv.bitsieve_timing import cache_tokens, median_tpob, summarize


def run(cfg, batch, ctx, tpobs, clean=True):
    return dict(config=cfg, batch=batch, context=ctx, clean=clean,
                rows=[dict(mean_tpob_ms=t) for t in tpobs])


def test_cache_tokens_rounds_up_to_blocks():
    assert cache_tokens(1024, 128) == 1152
    assert cache_tokens(1000, 128) == 1152
    assert cache_tokens(16384, 128, block=32) == 16512


def test_median_tpob():
    assert median_tpob([dict(mean_tpob_ms=3), dict(mean_tpob_ms=1), dict(mean_tpob_ms=2)]) == 2
    assert median_tpob([]) is None


def test_summary_prefers_clean_runs_and_computes_speedup():
    runs = [run("official_dense_bf16", 1, 1024, [100, 102, 98]),
            run("dense_kivi2_k2v2_r32", 1, 1024, [300], clean=False),     # dirty first attempt
            run("dense_kivi2_k2v2_r32", 1, 1024, [120, 125, 130]),        # clean retry
            run("official_dense_bf16", 16, 16384, [200]),
            run("dense_kivi2_k2v2_r32", 16, 16384, [100])]
    s = {(r["batch"], r["context"]): r for r in summarize(runs, stat="median")}
    a = s[(1, 1024)]
    assert a["official_dense_bf16_tpob_ms"] == 100 and a["dense_kivi2_k2v2_r32_tpob_ms"] == 125
    assert a["dense_kivi2_k2v2_r32_clean"] and abs(a["speedup"] - 0.8) < 1e-9
    assert s[(16, 16384)]["speedup"] == 2.0
    m = {(r["batch"], r["context"]): r for r in summarize(runs)}[(1, 1024)]     # default: minimum
    assert m["official_dense_bf16_tpob_ms"] == 98 and m["dense_kivi2_k2v2_r32_tpob_ms"] == 120
    assert abs(m["speedup"] - 98 / 120) < 1e-4 and abs(m["speedup_median"] - 0.8) < 1e-9
