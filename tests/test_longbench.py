import json
import zipfile

from blockkv import longbench


class Tok:
    """Whitespace 'tokenizer' for the cutting rule."""

    def __call__(self, text, add_special_tokens=False):
        class R:
            pass
        r = R()
        r.input_ids = text.split(" ")
        return r

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(ids)


def test_load_from_zip_and_items_ids(tmp_path):
    z = tmp_path / "data.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("data/qmsum.jsonl", json.dumps(dict(context="a b", input="q", answers=["x"])) + "\n")
        zf.writestr("data/gov_report.jsonl", "".join(json.dumps(dict(context=str(i), answers=["y"])) + "\n" for i in range(3)))
    it = longbench.items(["qmsum", "gov_report"], z, None, 0, per_task=2)
    assert [i for i, _, _ in it] == ["qmsum:0", "gov_report:0", "gov_report:1"]


def test_middle_cut_keeps_head_and_tail():
    row = dict(context=" ".join(f"w{i}" for i in range(1000)))
    text = longbench.raw_prompt("gov_report", row, Tok(), 100)
    assert len(text.split(" ")) <= 101
    assert text.startswith("You are given a report") and text.rstrip().endswith("Summary:")


def test_score_is_best_reference():
    assert longbench.score("the cat sat", ["a dog", "the cat sat"]) == 1.0
    assert 0 < longbench.score("the cat", ["the cat sat on"]) < 1
