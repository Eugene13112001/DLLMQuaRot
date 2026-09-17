from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from logit_compare import logit_errors, verdicts  # noqa: E402


def cells(pc, pt, qr):
    return {"per-channel": pc, "per-token": pt, "QuaRot": qr}


def test_h1_when_quarot_does_not_improve():
    v = verdicts(cells(0.1, 0.9, 0.8), cells(0.1, 0.3, 0.85), None)
    assert v["proxy orders the known answers"] is True
    assert v["H1"] is True and v["H2"] is False and v["H3"] is None


def test_h2_when_quarot_improves_but_stays_above_the_rescued_cell():
    v = verdicts(cells(0.1, 0.9, 0.8), cells(0.1, 0.3, 0.5), None)
    assert v["H1"] is False and v["H2"] is True


def test_h3_needs_the_other_model():
    v = verdicts(cells(0.1, 0.9, 0.8), cells(0.1, 0.3, 0.5), cells(0.1, 0.4, 0.6))
    assert v["H3"] is True


def test_proxy_that_misorders_known_answers_is_flagged():
    v = verdicts(cells(0.1, 0.2, 0.8), cells(0.1, 0.3, 0.5), None)
    assert v["proxy orders the known answers"] is False


def test_reads_the_logit_cells_from_a_dump():
    d = {"config": {"logit_error": True}, "errors": {
        "L/2/token/128": [[0.1, 0.3]], "L/2/channel/128": [[0.5, 0.7]],
        "L/2/channel+rot/128": [[0.2, 0.2]]}}
    e = logit_errors(d, 2, 128)
    assert abs(e["per-channel"] - 0.2) < 1e-12
    assert abs(e["per-token"] - 0.6) < 1e-12
    assert abs(e["QuaRot"] - 0.2) < 1e-12
