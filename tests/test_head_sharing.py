"""The probe is only worth running if its two rows actually differ."""

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from dllmquant.head_sharing import (  # noqa: E402
    make_head_sharing_attention,
    repeat_heads,
    stochastic_round,
)


def test_stochastic_round_is_unbiased_and_bounded():
    """Error stays inside one step and averages to nothing over draws."""
    torch.manual_seed(0)
    x = torch.randn(1, 2, 64, 32)
    step = (x.amax(-1, keepdim=True) - x.amin(-1, keepdim=True)) / 15

    draws = torch.stack([stochastic_round(x, 4) for _ in range(200)])
    assert ((draws - x).abs() <= step + 1e-5).all()
    # Unbiased: the mean over draws returns to the original, not to a grid.
    # Read as a mean, not a max -- over four thousand elements the largest
    # residual is a few standard errors of the mean by construction, so a
    # bound on the max would be testing the tail of the draw count.
    assert (draws.mean(0) - x).abs().mean() < 0.03 * float(step.mean())


def test_replicating_before_rounding_decorrelates_the_error():
    """The whole point: deterministic rounding would give identical copies."""
    torch.manual_seed(0)
    k = torch.randn(1, 4, 32, 16)
    rep = 4

    shared = repeat_heads(stochastic_round(k, 3), rep)
    independent = stochastic_round(repeat_heads(k, rep), 3)
    base = repeat_heads(k, rep)

    # Shared: the four readers of one stored head got the same value.
    for h in range(4):
        block = shared[:, h * rep:(h + 1) * rep]
        assert torch.equal(block[:, 0], block[:, 1])

    # Independent: they did not.
    assert not torch.equal(independent[:, 0], independent[:, 1])

    # And the magnitude is the same, which is what makes the rows comparable.
    a = (shared - base).abs().mean()
    b = (independent - base).abs().mean()
    assert abs(float(a) - float(b)) < 0.15 * float(a)


def test_wrapper_is_a_no_op_without_fan_out():
    """MHA has nothing to decorrelate, and the probe must not pretend."""
    seen = {}

    def inner(module, q, k, v, mask, **kw):
        seen["heads"] = k.shape[1]
        return k, None

    class M:
        num_key_value_groups = 1

    fn = make_head_sharing_attention(inner, "independent", 3)
    q = torch.randn(1, 4, 8, 16)
    fn(M(), q, q.clone(), q.clone(), None)
    assert seen["heads"] == 4, "an MHA module must reach attention untouched"


def test_wrapper_restores_the_group_count():
    """A raising kernel must not leave the module rewired."""
    class M:
        num_key_value_groups = 4

    def boom(*a, **k):
        raise RuntimeError("kernel failed")

    m = M()
    fn = make_head_sharing_attention(boom, "shared", 4)
    k = torch.randn(1, 2, 8, 16)
    try:
        fn(m, torch.randn(1, 8, 8, 16), k, k.clone(), None)
    except RuntimeError:
        pass
    assert m.num_key_value_groups == 4
