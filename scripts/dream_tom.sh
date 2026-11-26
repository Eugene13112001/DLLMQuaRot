#!/usr/bin/env bash
# Run from the repository root on ext1. Optional argument selects a GPU index.
set -euo pipefail
out=out/longproc/tom_tracking_8k/dreamreasoner_b32_bf16_direct
mkdir -p "$out"
exec 9>>"$out/run.lock"
flock -n 9 || { echo 'DreamReasoner ToM is already running'; exit 1; }

rev=673ec4c230876e941d674116d66d28783c186b15
lp="$HOME/third_party/LongProc"
if [[ ! -d "$lp/.git" ]]; then
    mkdir -p "$HOME/third_party"
    git clone https://github.com/princeton-pli/LongProc.git "$lp"
fi
[[ "$(git -C "$lp" rev-parse HEAD)" == "$rev" ]] || {
    echo "Expected LongProc $rev at $lp"; exit 1;
}

base="$HOME/venvs/blockkv/bin/python"
envdir="$HOME/venvs/dreamreasoner-tom"
if [[ ! -x "$envdir/bin/python" ]]; then
    "$base" -m venv "$envdir"
fi
py="$envdir/bin/python"
# Reuse the existing CUDA torch build while keeping Dream's Transformers separate.
existing=$("$base" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
site=$("$py" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
echo "$existing" > "$site/blockkv.pth"
"$py" -m pip install --disable-pip-version-check transformers==5.7.0 pyyaml
if [[ -n "${1:-}" ]]; then cards=(-i "$1"); else cards=(); fi
while true; do
    gpu=""
    while IFS=, read -r idx uuid free; do
        uuid=${uuid//[[:space:]]/}
        if (( free >= 40960 )) && CUDA_VISIBLE_DEVICES="$uuid" "$py" -c \
            'import torch; torch.empty(1,device="cuda"); assert torch.cuda.is_bf16_supported()'; then
            gpu=$uuid; break
        fi
    done < <(nvidia-smi "${cards[@]}" --query-gpu=index,uuid,memory.free \
        --format=csv,noheader,nounits | sort -t, -k3,3nr)
    [[ -n "$gpu" ]] && break
    echo "$(date -Is) Waiting for 40 GiB free on a CUDA GPU"
    sleep 60
done
export CUDA_VISIBLE_DEVICES="$gpu"
echo "$(date -Is) GPU $gpu | $out"
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 "$py" -u -m blockkv.dream_tom --longproc "$lp" --out "$out"
