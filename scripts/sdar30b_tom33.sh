#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
out=out/longproc/tom_tracking_8k/sdar30b_b32_bf16_first33
mkdir -p "$out"
exec 9>>"$out/run.lock"
flock -n 9 || { echo "This run is already active"; exit 1; }
py="$HOME/venvs/blockkv/bin/python"
lp="$HOME/third_party/LongProc"
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0

"$py" -u - "$lp" "$out" <<'PY'
import sys
import shutil
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE

lp, out = map(Path, sys.argv[1:])
cache = Path(HF_HUB_CACHE)
cache.mkdir(parents=True, exist_ok=True)
repo = "JetLM/SDAR-30B-A3B-Chat-b32"
rev = "c351bbc37d240aa6871f167e8f92d694281b0c22"
snap = cache / ("models--" + repo.replace("/", "--")) / "snapshots" / rev
gib = 2**30
print(f"Cache: {cache}; free: {shutil.disk_usage(cache).free/gib:.2f} GiB")
files = HfApi().model_info(repo, revision=rev, files_metadata=True).siblings
missing = 0
for f in files:
    if f.size is None:
        raise RuntimeError(f"Unknown size: {f.rfilename}")
    path = snap / f.rfilename
    if not path.is_file() or path.stat().st_size != f.size:
        missing += f.size
print(f"Missing files: {missing/gib:.2f} GiB; disk margin: 5 GiB")
if shutil.disk_usage(cache).free < missing + 5*gib:
    sys.exit("Insufficient disk space; stopping before download")
from flash_attn.ops.triton.layer_norm import rms_norm_fn
sys.path.insert(0, str(lp))
from longproc.longproc_data import load_longproc_data
data, _ = load_longproc_data("tom_tracking_8k", str(lp / "data"))
if len(data) != 200:
    sys.exit(f"Unexpected split size: {len(data)}")
path = snapshot_download(repo, revision=rev, cache_dir=str(cache), max_workers=2)
(out / "snapshot.txt").write_text(path + "\n")
PY

while true; do
    inventory=$(nvidia-smi --query-gpu=uuid,memory.free --format=csv,noheader,nounits)
    while IFS=, read -r uuid free; do
        if (( free >= 81920 )) && CUDA_VISIBLE_DEVICES="$uuid" "$py" -c \
            'import torch; torch.empty(1, device="cuda"); assert torch.cuda.is_bf16_supported() and torch.cuda.mem_get_info()[0] >= 80*2**30'; then
            export CUDA_VISIBLE_DEVICES="$uuid"
            echo "$(date -Is) GPU $uuid | BF16 | IDs 0..32 | cap 9216"
            "$py" -u -m blockkv.run_longproc --model sdar30b --block 32 --dtype bfloat16 \
                --model-path "$(cat "$out/snapshot.txt")" --longproc "$lp" \
                --datasets tom_tracking_8k --per-task 0 --limit 33 --threshold 0.9 \
                --prefill-chunk 512 --out "$out/results.jsonl"
            "$py" - "$out/results.jsonl" <<'PY'
import json, sys
from pathlib import Path
from statistics import mean
rows = [json.loads(s) for s in Path(sys.argv[1]).read_text().splitlines()]
expected = {f"tom_tracking_8k:{i}" for i in range(33)}
if len(rows) != 33 or {r["idx"] for r in rows} != expected:
    sys.exit("Incomplete or duplicate results; inspect results.meta.json")
if any("scorer_error" in r["metrics"] for r in rows):
    sys.exit("Scorer errors found; inspect results.jsonl")
for k in ("accuracy", "partial_accuracy"):
    print(f"{k} (33 items): {mean(r['metrics'][k] for r in rows):.4%}")
PY
            exit 0
        fi
    done < <(sort -t, -k2,2nr <<< "$inventory")
    echo "$(date -Is) Waiting for a CUDA GPU with 80 GiB free"
    sleep 60
done
