import torch

from blockkv.kv_probe import SCHEMES, cache_errors, make_quant, step_metrics


class Layer:
    def __init__(self, keys, values):
        self.keys, self.values = keys, values


class Cache:
    def __init__(self, keys, values):
        self.layers = [Layer(keys, values)]


def test_every_default_scheme_builds():
    for s in SCHEMES:
        assert make_quant(s).honest_bits > 0
    assert make_quant("k4v2").k_bits == 4 and make_quant("2v").only == "v" and make_quant("3rot").rotate


def test_centered_error_exposes_a_constant_key_offset():
    """A per-channel constant offset (k_proj bias) inflates the energy, not the token-to-token spread:
    rotation + per-token rounding looks fine by energy but not by the centered measure."""
    g = torch.Generator().manual_seed(0)
    signal = torch.randn(1, 2, 64, 128, generator=g)
    bias = torch.zeros(128)
    bias[:16] = 40.0                                      # large constant offsets in a few channels
    k = signal + bias
    v = torch.randn(1, 2, 64, 128, generator=g)
    res = {}
    for s in ("4", "4rot"):
        c = Cache(k.clone(), v.clone())
        q = Cache(k.clone(), v.clone())
        make_quant(s)(q, 0, 64)
        e = cache_errors(c, q)["k"]
        res[s] = (e[0] / e[1]).item(), (e[0] / e[2]).item()
    rel_rot, crel_rot = res["4rot"]
    rel_kivi, crel_kivi = res["4"]
    assert crel_rot > 10 * rel_rot                         # the offset hides the damage by energy
    assert crel_rot > 5 * crel_kivi                        # per-channel keys remove the offset first


def test_step_metrics_counts_flips_and_threshold_falls():
    lc = torch.tensor([[10.0, 0.0, 0.0], [5.0, 4.0, 0.0]])
    lq = torch.tensor([[3.0, 2.9, 0.0], [4.0, 5.0, 0.0]])
    m = step_metrics(lc, lq, 0.9)
    assert m["n"] == 2 and m["flip"] == 1 and m["above"] == 1 and m["fall_below"] == 1
    assert m["conf_drop"] > 0 and m["kl"] > 0


def test_block_mass_counts_masked_rows_on_the_last_q_keys():
    from blockkv.kv_probe import block_mass
    # 1 layer, 1 head, q_len 2, k_len 5 (3 cached + 2 block); row 0 masked, row 1 not
    a = torch.tensor([[[[0.1, 0.1, 0.2, 0.3, 0.3],
                        [0.5, 0.5, 0.0, 0.0, 0.0]]]])
    masked = torch.tensor([True, False])
    assert abs(block_mass((a,), masked) - 0.6) < 1e-6
    assert block_mass(None, masked) is None and block_mass((None,), masked) is None


def test_attention_recorder_uniform_share_and_passthrough():
    from blockkv.kv_probe import AttnRecorder
    rec = AttnRecorder()
    calls = []
    fn = rec.wrap(lambda q, k, v, softmax_scale=None, causal=False: calls.append(1) or "out")
    q = torch.zeros(1, 4, 8, 16)                  # q_len 4, 8 query heads -> uniform attention
    k = torch.randn(1, 20, 2, 16)                 # k_len 20 = 16 cached + 4 block, 2 kv heads (GQA)
    rec.active, rec.masked = True, torch.tensor([True, True, False, True])
    assert fn(q, k, k, softmax_scale=0.25) == "out" and calls == [1]
    assert abs(rec.vals[0] - 4 / 20) < 1e-6
    rec.active = False
    fn(q, k, k)
    assert len(rec.vals) == 1                     # inactive: nothing recorded
