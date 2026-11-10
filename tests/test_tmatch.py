import torch

from blockkv import decode
from blockkv.tmatch import t_match
from tests.test_decode import FakeSDAR, MASK, STOP, target


def test_temperature_rescales_confidence_not_the_token():
    logits = torch.tensor([[1.0, 0.0, 0.0], [5.0, 4.0, 0.0]])
    tok1, conf1 = decode._greedy(logits)
    tok2, conf2 = decode._greedy(logits, 0.5)
    assert torch.equal(tok1, tok2) and (conf2 > conf1).all()


def run(temp):
    prompt = torch.tensor([[target(p) for p in range(8)]])
    return decode.sdar_generate(FakeSDAR(8 + 60), prompt, MASK, 8, 8, 0.9, 128, [STOP], commit_temp=temp)


def test_lower_temperature_commits_more_per_step_and_writes_the_same():
    (gen1, st1), (gen2, st2) = run(1.0), run(0.1)
    assert torch.equal(gen1, gen2)
    assert st2.steps < st1.steps and st2.forced < st1.forced


def test_t_match_interpolates_down_to_the_reference():
    pts = [(1.0, 120.0), (0.9, 105.0), (0.8, 95.0)]
    assert t_match(pts, 130.0) == 1.0                      # no tax: T stays 1
    assert abs(t_match(pts, 100.0) - 0.85) < 1e-9          # halfway between 0.9 and 0.8
    assert t_match(pts, 90.0) is None                      # not reached in the grid
