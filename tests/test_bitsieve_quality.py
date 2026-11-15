import json

from blockkv.bitsieve_quality import main, read


def row(i, score, steps, stop=True, text="x"):
    return dict(id=i, score=score, prediction=text,
                runtime=dict(generated_tokens_per_request=32 * len(steps), denoising_steps_per_block=steps, stop_found=stop))


def test_paired_numbers(tmp_path, capsys):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text("".join(json.dumps(r) + "\n" for r in [row(0, 1.0, [10, 10]), row(1, 0.0, [8]), row(2, 1.0, [5])]))
    b.write_text("".join(json.dumps(r) + "\n" for r in [row(0, 1.0, [12, 12]), row(1, 1.0, [8]), row(2, 0.0, [9], stop=False)]))
    assert read(a)["0"]["steps"] == 20 and read(b)["2"]["gen_tokens"] >= 2000
    main([str(a), str(b)])
    out = json.loads(capsys.readouterr().out)
    assert out["n"] == 3 and out["only_ref"] == 1 and out["only_run"] == 1 and out["diff"] == 0
    assert out["steps_ref"] == round(33 / 3, 2) and out["steps_run"] == round(41 / 3, 2)
    assert out["runaway_run"] == 1 and out["runaway_ref"] == 0
