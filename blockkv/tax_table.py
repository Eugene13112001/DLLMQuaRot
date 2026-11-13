"""One row per compressed run against its uncompressed reference: honest bits, cache
error, accuracy change, step tax, tokens per step, extra steps from runaway answers.

    python -m blockkv.tax_table --ref out/stage1/full/sdar8b_b32_t0.9.jsonl \
        out/stage1_kv/full/sdar8b_b32_t0.9_kv{2,3,4}.jsonl \
        out/stage1/full/sdar8b_b4_t0.9.jsonl::out/stage1_kv/full/sdar8b_b4_t0.9_kv2_w128.jsonl

An item `REF::RUN` uses its own reference; a plain RUN uses --ref. Steps are means
without runaway answers (medians in brackets); `err` is the relative squared error of
the cached keys / values, as the quantizer measured it (noise runs measure the noise).
"""
import argparse
from pathlib import Path

from blockkv.summarize import kv, paired, read


def row(ref, run):
    ra, rb = read(ref), read(run)
    p = paired(ra, rb)
    rows = list(rb.values())
    d = 100 * (p["acc_b"] - p["acc_a"])
    out = dict(run=Path(run).stem, ref=Path(ref).stem, n=p["n"],
               bits=kv(rows, "kv_bits_avg", "5.2f").strip(),
               err_k=kv(rows, "kv_relerr_k", "7.4f").strip(), err_v=kv(rows, "kv_relerr_v", "7.4f").strip(),
               acc=f"{d:+.1f} ±{196 * p['se']:.1f}")
    if "steps_a" in p:
        out.update(steps=f"{100 * (p['calm_steps_b'] / p['calm_steps_a'] - 1):+.1f}% "
                         f"({p['med_steps_a']:.0f}→{p['med_steps_b']:.0f})",
                   tps=f"{p['tps_a']:.2f}→{p['tps_b']:.2f}",
                   runaway=f"{p['runaway_a']}→{p['runaway_b']}",
                   runaway_steps=(f"{100 * p['extra_runaway'] / p['extra']:.0f}%" if p["extra"] else "-"))
    return out


COLS = [("run", 34), ("n", 5), ("bits", 5), ("err_k", 7), ("err_v", 7), ("acc", 11), ("steps", 18),
        ("tps", 11), ("runaway", 7), ("runaway_steps", 13)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", help="uncompressed reference for plain RUN items")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--markdown", action="store_true", help="print a Markdown table with the file paths")
    args = ap.parse_args()
    rows = []
    for item in args.runs:
        ref, run = item.split("::") if "::" in item else (args.ref, item)
        if ref is None:
            raise SystemExit(f"{item}: no reference (use --ref or REF::RUN)")
        rows.append(row(ref, run))
    if args.markdown:
        print("| Прогон | Опора | Задач | Честные биты | Ошибка K / V | Точность | Шаги (медиана) | Токенов за шаг | Зацикл. | Доля лишних шагов в зацикл. |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            print(f"| `{r['run']}` | `{r['ref']}` | {r['n']} | {r['bits']} | {r['err_k']} / {r['err_v']} | {r['acc']} | "
                  f"{r.get('steps', '-')} | {r.get('tps', '-')} | {r.get('runaway', '-')} | {r.get('runaway_steps', '-')} |")
        return
    print(" ".join(f"{c:>{w}s}" if c != "run" else f"{c:{w}s}" for c, w in COLS))
    for r in rows:
        print(" ".join(f"{str(r.get(c, '-'))[-w:]:{w}s}" if c == "run" else f"{str(r.get(c, '-')):>{w}s}" for c, w in COLS))


if __name__ == "__main__":
    main()
