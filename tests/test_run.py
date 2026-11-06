from types import SimpleNamespace

from blockkv import run_gsm8k


class Tok:
    eos_token_id = 151643
    unk_token_id = None

    def convert_tokens_to_ids(self, t):
        return {"<|im_end|>": 151645, "<|endoftext|>": 151643}[t]


def test_stop_ids_follow_each_evaluation():
    model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[151645, 151643]))
    assert run_gsm8k.stop_ids("sdar", Tok(), model) == [151643, 151645]
    # Fast-dLLM v2 generate() stops on <|im_end|> only, whatever the generation config says.
    assert run_gsm8k.stop_ids("fastdllm", Tok(), model) == [151645]


def test_repo_names():
    assert run_gsm8k.repo("sdar8b", 4) == ("sdar", "JetLM/SDAR-8B-Chat")
    assert run_gsm8k.repo("sdar8b", 16) == ("sdar", "JetLM/SDAR-8B-Chat-b16")
    assert run_gsm8k.repo("fastdllm", 32) == ("fastdllm", "Efficient-Large-Model/Fast_dLLM_v2_7B")


def test_resume_drops_a_cut_last_line(tmp_path):
    out = tmp_path / "r.jsonl"
    out.write_text('{"idx": 0, "correct": true}\n{"idx": 1, "correct": false}\n{"idx": 2, "corr',
                   encoding="utf-8")
    assert run_gsm8k.read_done(out) == {0, 1}
    assert out.read_text(encoding="utf-8").count("\n") == 2
    assert run_gsm8k.read_done(tmp_path / "missing.jsonl") == set()
