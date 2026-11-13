import json

from blockkv import summarize, tax_table


def write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def rows(steps, correct, tokens=100, extra=None):
    out = []
    for i, (s, c) in enumerate(zip(steps, correct)):
        r = dict(idx=i, correct=c, gen_tokens=tokens, steps=s, per_step=[2] * s, text=f"t{i}{s}{c}")
        if extra:
            r.update(extra)
        out.append(r)
    return out


def test_paired_tokens_per_step_and_runaway_share(tmp_path):
    a = rows([10, 10, 10, 10], [1, 1, 1, 1])
    b = rows([12, 12, 12, 30], [1, 1, 0, 1])
    b[3]["gen_tokens"] = summarize.RUNAWAY          # the last answer ran away
    p = summarize.paired({r["idx"]: r for r in a}, {r["idx"]: r for r in b})
    assert p["calm"] == 3
    assert p["tps_a"] == 2 and p["tps_b"] == 2      # per_step = 2 every step
    assert abs(p["extra"] - (2 + 2 + 2 + 20) / 4) < 1e-9
    assert abs(p["extra_runaway"] - 20 / 4) < 1e-9
    assert abs(p["calm_steps_b"] / p["calm_steps_a"] - 1.2) < 1e-9


def test_tax_table_row_and_markdown(tmp_path, capsys):
    ref, run = tmp_path / "ref.jsonl", tmp_path / "run_kv2.jsonl"
    write(ref, rows([10] * 4, [1, 1, 1, 1]))
    write(run, rows([12] * 4, [1, 1, 1, 0], extra=dict(kv_bits_avg=3.0, kv_relerr_k=0.1, kv_relerr_v=0.2,
                                                          kv_quantized=50, kv_tokens=100)))
    r = tax_table.row(str(ref), str(run))
    assert r["bits"] == "3.00" and r["err_k"] == "0.1000" and r["steps"].startswith("+20.0%")
    assert r["eff_k"] == "0.0500" and r["eff_v"] == "0.1000"     # half of the cache is quantized
    assert r["acc"].startswith("-25.0") and r["runaway"] == "0→0"
    import sys
    sys.argv = ["tax_table", "--markdown", f"{ref}::{run}"]
    tax_table.main()
    assert "`run_kv2`" in capsys.readouterr().out


def test_compare_prints_the_new_lines(tmp_path, capsys):
    ref, run = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write(ref, rows([10] * 3, [1, 1, 1]))
    write(run, rows([11] * 3, [1, 1, 1]))
    summarize.compare(str(ref), str(run))
    out = capsys.readouterr().out
    assert "tokens per step without runaways" in out and "extra steps per problem +1.00" in out
