#!/usr/bin/env bash
# SDAR b32 with the author QuMix code and a fresh checkpoint-specific calibration.
set -euo pipefail
cd "$(dirname "$0")/.."

author=third_party/qumix
output=${OUT:-out/qumix/sdar8b_b32_t0.9}
model=${SDAR_DIR:-$output/model}
bench_data=${LONGBENCH_DATA:-$HOME/data/longbench/data.zip}
proc_data=${LONGPROC:-$HOME/third_party/LongProc}
mkdir -p "$output/calib"
exec 9>>"$output/run.lock"
flock -n 9 || { echo "QuMix is already running in $output" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES=${GPU:-0}

if [[ -z ${SDAR_DIR:-} ]]; then
    if [[ ! -f "$model/.qumix_prepared" ]]; then
        hf download JetLM/SDAR-8B-Chat-b32 --local-dir "$model"
        cp --remove-destination "$author/accuracy/sdar/sdar_model/modeling_sdar.py" "$model/modeling_sdar.py"
        touch "$model/.qumix_prepared"
    fi
fi

calib="$output/calib"
if [[ ! -s "$calib/same_kv.pt" ]]; then
    python -u "$author/accuracy/sdar/qumix_eval.py" --model "$model" \
        --calibonly --calibset wikitext --ncalib 64 --calibgen 128 --prerope \
        --vbits 2 --block 32 --steps 32 --ladder --arms qmx2t --savecalib "$calib/raw.pt"
    python "$author/calibration/build_po2q.py" "$calib/raw.pt" "$calib/po2q.pt"
    python "$author/calibration/add_scale_tags.py" "$calib/po2q.pt" "$calib/scales.pt"
    python "$author/calibration/build_same_kv.py" --model sdar --src "$calib/scales.pt" \
        --dst "$calib/same_kv.tmp.pt"
    mv "$calib/same_kv.tmp.pt" "$calib/same_kv.pt"
fi

extra=()
[[ -z ${LIMIT:-} ]] || extra+=(--limit "$LIMIT")
for bench in ${BENCHES:-longbench longproc}; do
    case "$bench" in
        longbench) data=$bench_data ;;
        longproc) data=$proc_data ;;
        *) echo "Unknown benchmark: $bench" >&2; exit 2 ;;
    esac
    python -u -m blockkv.run_qumix --model "$model" --calibration "$calib/same_kv.pt" \
        --bench "$bench" --data "$data" --out "$output/$bench" "${extra[@]}"
done
