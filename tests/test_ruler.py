import json

import pytest

from blockkv import run_ruler, summarize_ruler


def test_score_follows_the_kvpress_rules():
    # qa tasks: any reference is enough
    assert run_ruler.score("qa_1", " The answer is Paris.\n", ["london", "PARIS"]) == 1.0
    assert run_ruler.score("qa_2", "no idea", ["paris"]) == 0.0
    # the others: the share of the references found
    assert run_ruler.score("vt", "VAR A, VAR B", ["var a", "var b", "var c", "var d"]) == 0.5
    assert run_ruler.score("niah_single_1", "the number is 123\x01", ["123"]) == 1.0


def test_pick_is_fixed_and_balanced():
    rows = [{"task": "a"}] * 10 + [{"task": "b"}] * 4
    first = run_ruler.pick(rows, 3, 0)
    assert first == run_ruler.pick(rows, 3, 0)
    assert sum(i < 10 for i in first) == 3 and sum(i >= 10 for i in first) == 3
    assert len(run_ruler.pick(rows, 8, 0)) == 8 + 4     # a task smaller than per_task gives all it has
    assert run_ruler.pick(rows, 0, 0) == list(range(14))


class Tok:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert not tokenize and add_generation_prompt
        return f"<u>{messages[0]['content']}</u><a>"


def test_prompt_keeps_the_order():
    row = dict(context="CTX ", question="Q?", answer_prefix="The answer:")
    assert run_ruler.prompt_text(Tok(), row) == "<u>CTX Q?</u><a>The answer:"


def test_load_reads_a_length_folder(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    (tmp_path / "4096").mkdir()
    pd.DataFrame(dict(task=["vt"], answer=[["x", "y"]], context=["c"])).to_parquet(tmp_path / "4096" / "t.parquet")
    assert run_ruler.load(tmp_path, 4096)[0]["answer"] == ["x", "y"]
    with pytest.raises(SystemExit):
        run_ruler.load(tmp_path, 8192)


def r(task, score, steps=10):
    return dict(task=task, score=score, steps=steps, prompt_tokens=100)


def test_every_task_counts_the_same():
    rows = [r("a", 1.0)] * 9 + [r("b", 0.0)]
    assert summarize_ruler.task_mean(rows) == 0.5


def test_paired_difference_is_task_balanced():
    ref = {0: r("a", 1.0), 1: r("a", 1.0), 2: r("b", 1.0)}
    run = {0: r("a", 0.0), 1: r("a", 0.0), 2: r("b", 1.0), 7: r("b", 0.0)}   # idx 7 has no pair
    diff, ci = summarize_ruler.paired(ref, run)
    assert diff == -0.5 and ci == 0


def test_table_pairs_a_run_with_its_block(tmp_path, capsys):
    def write(name, rows):
        (tmp_path / name).write_text("".join(json.dumps(dict(x, idx=i)) + "\n" for i, x in enumerate(rows)), encoding="utf-8")
    write("sdar8b_b4.jsonl", [r("a", 1.0, 10), r("a", 1.0, 20)])
    write("sdar8b_b4_kv2.jsonl", [r("a", 0.5, 12), r("a", 0.5, 24)])
    write("sdar8b_b16.jsonl", [r("a", 1.0)])
    summarize_ruler.table(tmp_path)
    out = capsys.readouterr().out.splitlines()
    line = next(x for x in out if "b4_kv2" in x)
    assert line.split()[3:5] == ["50.0", "-50.0"]
    assert line.split()[-1] == "+20.0"
