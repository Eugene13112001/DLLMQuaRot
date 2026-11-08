import torch

from blockkv.audit_cache import Snapshot, Totals, audit_layer, levels
from blockkv.kvquant import KVQuant, cache_layers
from tests.test_kvquant import H, D, LAYERS, kv_at, stored


def audited(block, total, quant):
    snap = Snapshot(quant)
    stored(block, total, snap)
    return snap, [audit_layer(kq, vq, kc, vc, 2, 32)
                  for (kq, vq), (kc, vc) in zip(cache_layers(snap.cache), snap.clean)]


def test_levels_counts_distinct_values():
    assert levels(torch.tensor([[1., 1., 2., 5.], [3., 3., 3., 3.]]), 1).tolist() == [3, 1]


def test_snapshot_keeps_the_exact_cache():
    snap, _ = audited(16, 128, KVQuant(2, 32))
    for layer, (k, v) in enumerate(snap.clean):
        ref_k, ref_v = kv_at(0, 128, layer)
        assert torch.equal(k, ref_k) and torch.equal(v, ref_v)


def test_a_quantized_cache_passes():
    _, results = audited(8, 104, KVQuant(2, 32))           # 96 quantized, 8 left in the tail
    for r in results:
        assert r["tail_exact"]
        for side in ("k", "v"):
            assert r[side]["over"] == 0 and r[side]["max_levels"] <= 4 and r[side]["changed"] > 0
            assert r[side]["groups"] > 0


def test_an_uncompressed_side_is_caught():
    _, results = audited(16, 96, KVQuant(2, 32, only="v"))
    assert all(r["k"]["over"] > 0 and r["k"]["changed"] == 0 and r["v"]["over"] == 0 for r in results)


def test_a_touched_tail_is_caught():
    _, results = audited(32, 96, KVQuant(2, 32))
    assert all(r["tail_exact"] for r in results)
    kq, vq = kv_at(0, 40, 0)
    damaged = kq.clone()
    damaged[:, :, 35] += 1
    assert not audit_layer(damaged, vq, kq, vq, 16, 32)["tail_exact"]


def test_centered_error_ignores_the_constant_offset():
    k = torch.randn(1, H, 32, D)
    v = torch.randn(1, H, 32, D)
    shifted = k + 100
    plain = audit_layer(k.half() + 0.25, v.half(), k.half(), v.half(), 16, 32)["k"]
    offset = audit_layer(shifted.half() + 0.25, v.half(), shifted.half(), v.half(), 16, 32)["k"]
    assert offset["err"] / offset["energy"] < plain["err"] / plain["energy"] / 100    # relative error collapses
    ratio = (offset["err"] / offset["spread"]) / (plain["err"] / plain["spread"])
    assert 0.1 < ratio < 10                                                           # centered error does not


def test_totals_add_up_over_layers():
    _, results = audited(32, 64, KVQuant(2, 32))
    totals = Totals()
    for layer, r in enumerate(results):
        totals.add(layer, r)
    assert sorted(totals.sums) == list(range(LAYERS)) and totals.tail_broken == 0
    row = totals.row(0, "k")
    assert row["max_levels"] <= 4 and row["over"] == 0 and 0 < row["centered"] < 1
