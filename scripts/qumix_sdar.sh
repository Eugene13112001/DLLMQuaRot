#!/usr/bin/env bash
# SDAR with the author QuMix code and a fresh checkpoint-specific calibration.
set -euo pipefail
cd "$(dirname "$0")/.."

author=third_party/qumix
block=${BLOCK:-32}
case "$block" in
    4|8|16|32) ;;
    *) echo "Unsupported SDAR block: $block" >&2; exit 2 ;;
esac
output=${OUT:-out/qumix/sdar8b_b${block}_t0.9}
model=${SDAR_DIR:-$output/model}
bench_data=${LONGBENCH_DATA:-$HOME/data/longbench/data.zip}
proc_data=${LONGPROC:-$HOME/third_party/LongProc}
read -r -a benches <<<"${BENCHES:-longbench longproc}"
mkdir -p "$output/calib"
exec 9>>"$output/run.lock"
flock -n 9 || { echo "QuMix is already running in $output" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES=${GPU:-0}

for bench in "${benches[@]}"; do
    case "$bench" in
        longbench)
            [[ -r "$bench_data" ]] || { echo "LongBench data unavailable: $bench_data" >&2; exit 2; }
            if [[ -z ${TASKS:-} ]]; then
                python -c 'from blockkv.longbench import require_metrics; require_metrics()'
            fi
            ;;
        longproc)
            [[ -d "$proc_data/longproc" && -d "$proc_data/data" ]] || {
                echo "LongProc checkout incomplete: $proc_data" >&2; exit 2;
            }
            ;;
        *) echo "Unknown benchmark: $bench" >&2; exit 2 ;;
    esac
done
python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable on the selected GPU.")
torch.empty(1, device="cuda")
PY

if [[ -z ${SDAR_DIR:-} ]]; then
    if [[ ! -f "$model/.qumix_prepared" ]]; then
        python - "$model" "$block" <<'PY'
import json
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

from blockkv.run_gsm8k import repo as model_repo

block = int(sys.argv[2])
_, repo = model_repo("sdar8b", block)
revision = "76fd4333a0c7e14a92e9873f05c295fd6954991a" if block == 32 else "main"


def complete(path):
    if path is None or not (path / "config.json").is_file():
        return False
    for name in ("model.safetensors", "pytorch_model.bin"):
        if (path / name).is_file():
            return True
        index = path / (name + ".index.json")
        if index.is_file():
            shards = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
            if shards and all((path / shard).is_file() for shard in shards):
                return True
    return False


root = os.environ.get("MODELS_DIR")
source = Path(root).expanduser() / repo.split("/")[-1] if root else None
if not complete(source):
    try:
        source = Path(snapshot_download(repo, revision=revision, local_files_only=True))
    except LocalEntryNotFoundError:
        source = None
if not complete(source):
    try:
        source = Path(snapshot_download(repo, revision=revision))
    except LocalEntryNotFoundError as error:
        raise SystemExit(
            f"{repo} weights unavailable. Point MODELS_DIR to a directory containing "
            f"{repo.split('/')[-1]} or populate the Hugging Face cache before retrying."
        ) from error
if not complete(source):
    raise SystemExit(f"SDAR checkpoint is incomplete: {source}")

target = Path(sys.argv[1]).resolve()
target.mkdir(parents=True, exist_ok=True)
if target == source.resolve():
    raise SystemExit("Use a separate output directory for the patched QuMix model.")
for entry in source.iterdir():
    if entry.name.startswith(".") or entry.name == "modeling_sdar.py":
        continue
    link = target / entry.name
    original = entry.resolve(strict=True)
    if link.is_symlink() and link.resolve() == original:
        continue
    link.symlink_to(original, target_is_directory=original.is_dir())
print(f"QuMix model uses existing checkpoint files: {source}", flush=True)
PY
        cp --remove-destination "$author/accuracy/sdar/sdar_model/modeling_sdar.py" "$model/modeling_sdar.py"
        touch "$model/.qumix_prepared"
    fi
fi

calib="$output/calib"
if [[ ! -s "$calib/same_kv.pt" ]]; then
    python -u -m blockkv.run_qumix --author-calibration --model "$model" \
        --calibset wikitext --ncalib 64 --calibgen 128 --prerope \
        --vbits 2 --block "$block" --steps "$block" --ladder --arms qmx2t --savecalib "$calib/raw.pt"
    python "$author/calibration/build_po2q.py" "$calib/raw.pt" "$calib/po2q.pt"
    python "$author/calibration/add_scale_tags.py" "$calib/po2q.pt" "$calib/scales.pt"
    python "$author/calibration/build_same_kv.py" --model sdar --src "$calib/scales.pt" \
        --dst "$calib/same_kv.tmp.pt"
    mv "$calib/same_kv.tmp.pt" "$calib/same_kv.pt"
fi

extra=()
[[ -z ${LIMIT:-} ]] || extra+=(--limit "$LIMIT")
for bench in "${benches[@]}"; do
    bench_out="$output/$bench"
    bench_extra=("${extra[@]}")
    case "$bench" in
        longbench)
            data=$bench_data
            if [[ -n ${TASKS:-} ]]; then
                read -r -a tasks <<<"$TASKS"
                bench_extra+=(--tasks "${tasks[@]}")
                task_dir=$(IFS=_; echo "${tasks[*]}")
                [[ "$task_dir" =~ ^[a-z0-9_]+$ ]] || { echo "Invalid TASKS" >&2; exit 2; }
                bench_out="${bench_out}_${task_dir}"
            fi
            ;;
        longproc) data=$proc_data ;;
        *) echo "Unknown benchmark: $bench" >&2; exit 2 ;;
    esac
    python -u -m blockkv.run_qumix --model "$model" --block "$block" --calibration "$calib/same_kv.pt" \
        --bench "$bench" --data "$data" --out "$bench_out" "${bench_extra[@]}"
done
