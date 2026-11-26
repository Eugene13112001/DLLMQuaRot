"""CPU-only sensitivity analysis and stratified blind review of saved generations."""
import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import fmean

from blockkv.tax_decompose import repetitive

THRESHOLDS = (0.2, 0.3, 0.4)
QA_CAP = {"qmsum": 512, "gov_report": 512, "2wikimqa": 32, "musique": 32, "narrativeqa": 128}


def csvfile(path, fields, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields.split())
        writer.writerows(rows)


def baseline(path, root):
    stem = path.stem
    for suffix in ("_kivi_local2", "_qmxmatchkivi2ts", "_kivis2r128"):
        if stem.endswith(suffix):
            return path.with_stem(stem.removesuffix(suffix) + "_base")
    if re.search(r"_q[1-9]\d*$", stem):
        return path.with_stem(re.sub(r"_q\d+$", "_q0", stem))
    if "_kv" in stem:
        ref = path.with_stem(stem.split("_kv")[0])
        if path.relative_to(root).parts[0] == "stage1_kv":
            return root / "stage1/full" / ref.name
        return ref
    return None


def classify(path, meta, root):
    group = path.relative_to(root).parts[0]
    for name in ("longbench", "longproc", "ruler", "gsm8k"):
        if name in str(path.relative_to(root)).lower() or meta.get("bench") == name:
            return name
    if group in ("stage1", "stage1_kv", "lmdeploy"):
        return "gsm8k"
    return None


def cap_flag(row, meta, bench, task):
    if row.get("finish") in ("length", "stop"):
        return row["finish"] == "length"
    cap = row.get("max_new_tokens")
    if not cap and bench == "gsm8k":
        cap = meta.get("max_new_tokens") or meta.get("gen_length")
    if not cap and bench == "longbench":
        cap = meta.get("max_new_tokens") or QA_CAP.get(task)
    if not cap and bench == "longproc":
        cap = {"0.5k": 1024, "2k": 3072, "8k": 9216}.get(task.split("_")[-1])
    return row["gen_tokens"] >= cap if cap else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("out"))
    ap.add_argument("--per-stratum", type=int, default=12)
    args = ap.parse_args()
    if args.per_stratum < 1:
        ap.error("--per-stratum must be positive")
    root = args.root
    out = root / "loop_audit" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True)
    rng = random.Random(20261126)
    runs, manifest = {}, []
    counts, samples = Counter(), defaultdict(list)
    for path in sorted(root.rglob("*.jsonl")):
        group = path.relative_to(root).parts[0]
        if group not in ("stage1", "stage1_kv", "longbench", "longproc", "ruler", "lmdeploy", "qumix"):
            manifest.append((path, "outside benchmark output directories", 0, 0))
            continue
        mp = path.with_suffix(".meta.json")
        meta = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
        meta = meta.get("run", meta)
        bench = classify(path, meta, root)
        rows, skipped, duplicates = {}, 0, 0
        with path.open(encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except ValueError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
                if not bench or not isinstance(r.get("text"), str) or not all(k in r for k in ("idx", "steps", "gen_tokens")):
                    skipped += 1
                    continue
                task = r.get("task") or bench
                key = task, str(r["idx"])
                if key in rows:
                    duplicates += 1
                    continue
                cap = cap_flag(r, meta, bench, task)
                flags = tuple(repetitive(r["text"], min_distinct=t) for t in THRESHOLDS)
                rows[key] = (r["steps"], r["gen_tokens"], cap, flags)
                stratum = bench, task, str(cap), flags[1]
                counts[stratum] += 1
                sample = samples[stratum]
                j = len(sample) if len(sample) < args.per_stratum else rng.randrange(counts[stratum])
                item = (path, key[1], r["text"])
                if j < len(sample):
                    sample[j] = item
                elif len(sample) < args.per_stratum:
                    sample.append(item)
        if rows:
            runs[path] = rows
        manifest.append((path, f"kept first occurrence; duplicates={duplicates}", len(rows), skipped))
    csvfile(out/"manifest.csv", "path status used skipped", manifest)
    results = []
    for path, rows in runs.items():
        ref = baseline(path, root)
        if ref not in runs:
            continue
        common = runs[ref].keys() & rows.keys()
        for task in sorted({k[0] for k in common}):
            pairs = [(runs[ref][k], rows[k]) for k in sorted(common) if k[0] == task]
            n_common = len(pairs)
            pairs = [(a,b) for a,b in pairs if a[0] > 0 and b[0] > 0]
            for i, threshold in enumerate(THRESHOLDS):
                for mode in ("all", "exclude_repeat", "exclude_cap", "exclude_both"):
                    keep = [(a,b) for a,b in pairs
                            if (mode not in ("exclude_cap","exclude_both") or a[2] is False and b[2] is False)
                            and (mode not in ("exclude_repeat","exclude_both") or not (a[3][i] or b[3][i]))]
                    delta = [b[0]-a[0] for a,b in keep]
                    results.append((ref,path,task,threshold,mode,n_common,len(pairs),len(keep),
                                    fmean(delta) if delta else "", 100*sum(delta)/sum(a[0] for a,b in keep) if delta else ""))
    csvfile(out/"tax.csv","reference run task min_distinct filter common_ids measured_pairs retained_pairs mean_extra_steps extra_steps_pct",results)
    selected = [(s,p,i,t) for s,items in samples.items() for p,i,t in items]
    rng.shuffle(selected)
    csvfile(out/"blind.csv","sample_id text label notes",((n,t,"","") for n,(_,_,_,t) in enumerate(selected,1)))
    csvfile(out/"key.csv","sample_id benchmark task cap_hit repeat_flag path idx population sampled",
            ((n,*s,p,i,counts[s],len(samples[s])) for n,(s,p,i,_) in enumerate(selected,1)))
    print(f"{len(runs)} generation files, {sum(map(len,runs.values()))} answers, {len(selected)} blinded examples. Output: {out}")


if __name__ == "__main__":
    main()

