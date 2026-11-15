import json

import torch

from blockkv.lmdeploy_gsm8k import StepCounter, read_done

M, U, C = 0, 1, 2


def test_counter_counts_only_calls_that_commit():
    c = StepCounter()
    c.record(torch.tensor([M, M, M, M]), torch.tensor([U, M, U, M]), M, U)    # 2 committed
    c.record(torch.tensor([U, U, U, U]), torch.tensor([C, C, C, C]), M, U)    # cache pass, no commit
    c.record(torch.tensor([U, M, U, M]), torch.tensor([U, U, U, U]), M, U)    # 2 committed
    assert c.snapshot() == (3, 2, 4)


def test_read_done_drops_a_cut_last_line(tmp_path):
    p = tmp_path / "o.jsonl"
    p.write_text(json.dumps(dict(idx=0)) + "\n" + json.dumps(dict(idx=1)) + "\n" + '{"idx": 2, "te')
    assert read_done(p) == {0, 1}
    assert len(p.read_text().splitlines()) == 2
