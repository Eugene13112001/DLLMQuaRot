import json

from blockkv import summarize


def write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def row(idx, correct, tokens, steps):
    return dict(idx=idx, correct=correct, gen_tokens=tokens, steps=steps, forced=0, per_step=[1] * steps,
                text=str(tokens))


def test_a_runaway_answer_does_not_move_the_robust_numbers(tmp_path, capsys):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write(a, [row(0, True, 100, 50), row(1, True, 200, 100), row(2, True, 300, 150)])
    write(b, [row(0, True, 100, 50), row(1, True, 200, 100), row(2, False, 4096, 4000)])
    summarize.compare(str(a), str(b))
    out = capsys.readouterr().out
    assert "runaway answers a=0 b=1" in out
    assert "steps: median a=100 b=100 (+0.0%); mean without runaways a=75.0 b=75.0 (+0.0%, n=2)" in out
    assert "mean steps a=100.0" in out


def test_compare_without_step_counts(tmp_path, capsys):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write(a, [row(0, True, 100, 0)])
    write(b, [row(0, True, 110, 0)])
    summarize.compare(str(a), str(b))
    out = capsys.readouterr().out
    assert "gen_tokens: median a=100 b=110" in out and "mean steps" not in out


def test_table_has_the_robust_columns(tmp_path, capsys):
    a = tmp_path / "a.jsonl"
    write(a, [row(0, True, 100, 50), row(1, False, 4096, 4000), row(2, True, 300, 150)])
    summarize.table([str(a)])
    head, line = capsys.readouterr().out.splitlines()
    assert "med.st" in head and "runaw" in head
    assert line.split()[6:8] == ["150", "1"]
