"""The teacher-forced accounting behind the trajectory sweep.

The table it feeds separates two things that look alike: a cached run that
committed a different token, and a run that merely forked earlier and is now
decoding different text. Holding the canvas to a reference is what separates
them, and the accounting below is the part that can silently be wrong -- it
was, and the giveaway was a control column printing 1/32 where it had to
print 100%.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_spec = importlib.util.spec_from_file_location(
    "check_block_reuse",
    Path(__file__).resolve().parents[1] / "scripts" / "check_block_reuse.py",
)
cbr = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = cbr
_spec.loader.exec_module(cbr)

MASK = 99


def test_a_masked_position_is_not_a_commit():
    prev = torch.full((1, 8), MASK)
    x = prev.clone()
    x[0, 2] = 5

    assert cbr.newly_committed(x, prev, 0, 8, MASK) == [2]
    # And nothing outside the block counts, however it changed.
    assert cbr.newly_committed(x, prev, 4, 8, MASK) == []


def test_a_position_commits_exactly_once():
    """Over a whole block every generated position is counted once, which is
    the invariant the first column violated."""
    block = 8
    prev = torch.full((1, block), MASK)
    counted = []
    for step in range(block):
        x = prev.clone()
        x[0, step] = step + 1                       # one commit per step
        counted += cbr.newly_committed(x, prev, 0, block, MASK)
        prev = x.clone()

    assert sorted(counted) == list(range(block))
    assert len(counted) == block


def test_an_all_mask_start_does_not_count_the_whole_block():
    """The bug, pinned: against a canvas that is not the mask id, every
    position of a fresh block reads as newly committed."""
    block = torch.full((1, 6), MASK)
    wrong_prev = torch.full((1, 6), -1)

    assert cbr.newly_committed(block, wrong_prev, 0, 6, MASK) == []
    # Nothing has committed yet, so nothing may be counted -- with either
    # starting canvas, because the test is on x, not on prev.
    assert cbr.newly_committed(block, block, 0, 6, MASK) == []


def test_a_committed_position_is_not_counted_again():
    prev = torch.full((1, 4), MASK)
    prev[0, 1] = 7
    x = prev.clone()

    assert cbr.newly_committed(x, prev, 0, 4, MASK) == []
    x[0, 1] = 8                                     # overwritten, not committed
    assert cbr.newly_committed(x, prev, 0, 4, MASK) == []
