#!/usr/bin/env bash
# Run a script against the shadow install of transformers that LLaDA2.0 needs.
#
# The two LLaDA families need transformers versions whose ranges do not
# overlap, and the newer one lives in a directory on PYTHONPATH rather than in
# the venv, so LLaDA-1.5 keeps working untouched.  That path is easy to lose:
# it does not survive a new shell, and when it is missing the run does not
# fail at the point of the mistake -- it loads the venv's transformers and
# stops several seconds later complaining about a version, which reads like a
# problem with the install rather than with the command.
#
#   bash scripts/llada2.sh scripts/selfcheck.py --model inclusionAI/LLaDA2.0-mini \
#       --model-type llada2_moe --rotate
#
# Override the location with TF457=/somewhere/else.
set -euo pipefail

TF457="${TF457:-$HOME/tf457}"

if [ ! -d "$TF457/transformers" ]; then
    echo "no transformers under $TF457 -- install the shadow copy first:" >&2
    echo "  pip install --no-index --no-deps --target=$TF457 <wheels>/core/*.whl" >&2
    echo "or point TF457 at where it actually lives." >&2
    exit 1
fi

export PYTHONPATH="$TF457${PYTHONPATH:+:$PYTHONPATH}"

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    # A shared node: the emptiest card now is not the emptiest card in ten
    # minutes, but starting on the fullest one is a guaranteed waste.
    CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=index,memory.free \
        --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1)"
    export CUDA_VISIBLE_DEVICES
    echo "picked GPU $CUDA_VISIBLE_DEVICES (most free memory)" >&2
fi

exec python "$@"
