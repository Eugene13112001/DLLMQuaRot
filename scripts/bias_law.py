"""Does the k_proj bias predict, layer by layer, how much the wrong key axis costs?

The counterpart of gamma_law.py for a model with no QK-Norm. A bias adds the same vector to
every key before RoPE, so a layer whose bias is concentrated in a few loud rotary pairs should
hold a few loud fixed channels, and there a scale per token should cost much more than a
scale per channel. The features are read from the bias alone -- no activations -- and
correlated across layers with the measured axis cost of a check_key_error dump.

The bias enters the key additively, not multiplicatively like a norm gain, so its size only
matters relative to the rest of the key. Two kinds of feature are reported: the shape of the
bias (how concentrated it is in a few pairs) and its size relative to the key's typical
magnitude, read from the dump's crest-free statistics where available.

    python scripts/bias_law.py out/ke_cfd.json --bits 4
    python scripts/bias_law.py out/ke_dfd.json --bits 4 \\
        --checkpoint Efficient-Large-Model/Fast_dLLM_v2_7B     # dump taken before k_bias was recorded

RoPE pairs channel i with i + d/2 across the whole head here, so the per-pair magnitude
sqrt(b_i^2 + b_{i+d/2}^2) is what survives the rotation and is what is measured.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gamma_law import boot_ci, cell, layer_ratio, pair_gains, perm_p, spearman  # noqa: E402


def bias_features(bias: List[float], kv_heads: int) -> Dict[str, float]:
    """Shape of a layer's bias, per KV head then averaged: concentration in a few pairs."""
    d = len(bias) // kv_heads
    feats: Dict[str, List[float]] = {"pair max/median": [], "pair top-share": [], "max |b|": []}
    for h in range(kv_heads):
        b = bias[h * d:(h + 1) * d]
        pairs = sorted(pair_gains(b, d))
        n = len(pairs)
        med = (pairs[n // 2] if n % 2 else (pairs[n // 2 - 1] + pairs[n // 2]) / 2) or 1e-12
        energy = sum(x * x for x in pairs) or 1e-12
        top = sum(x * x for x in pairs[-max(1, n // 16):])
        feats["pair max/median"].append(pairs[-1] / med)
        feats["pair top-share"].append(top / energy)
        feats["max |b|"].append(max(abs(x) for x in b))
    return {k: sum(v) / len(v) for k, v in feats.items()}


def biases_from_checkpoint(model: str, layers: List[int]) -> List[List[float]]:
    """Read the k_proj biases straight from the safetensors shards, no model load."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    root = Path(snapshot_download(model, allow_patterns=["*.safetensors", "*.json"]))
    wanted = {f"model.layers.{i}.self_attn.k_proj.bias": i for i in layers}
    found: Dict[int, List[float]] = {}
    for shard in sorted(root.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as fh:
            for key in fh.keys():
                if key in wanted:
                    found[wanted[key]] = [float(x) for x in fh.get_tensor(key).float()]
    missing = [i for i in layers if i not in found]
    if missing:
        raise SystemExit(f"no k_proj bias found for layers {missing} in {model}")
    return [found[i] for i in layers]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--kv-heads", type=int, default=4)
    ap.add_argument("--checkpoint", default=None,
                    help="read the biases from this checkpoint when the dump predates k_bias")
    args = ap.parse_args()

    with open(args.dump, encoding="utf-8") as fh:
        d = json.load(fh)
    layers = d["config"]["layers"]
    biases = d.get("k_bias")
    if not biases or any(b is None for b in biases):
        if not args.checkpoint:
            raise SystemExit("the dump carries no k_bias; pass --checkpoint to read it from the "
                             "weights")
        biases = biases_from_checkpoint(args.checkpoint, layers)

    feats = [bias_features(b, args.kv_heads) for b in biases]
    # The bias relative to the key it is added to, when the dump was taken with --pre-bias:
    # an additive term's effect on a per-token scale depends on that ratio, not on the
    # bias alone, and the shape features above cannot see it.
    share = d.get("bias_share")
    if share:
        for i in range(len(layers)):
            feats[i]["||b|| / ||k||"] = sum(row[i] for row in share) / len(share)
    meas = {"axis (K space)": layer_ratio(d, cell("K", args.bits, "channel", args.group),
                                          cell("K", args.bits, "token", args.group))}
    lc_tok, lc_ch = (cell("Lc", args.bits, a, args.group) for a in ("channel", "token"))
    if lc_tok in d["errors"] and lc_ch in d["errors"]:
        meas["axis (centered logits)"] = layer_ratio(d, lc_tok, lc_ch)

    names = list(feats[0])
    print(f"{d['config']['model']}: {len(layers)} layers, {args.bits} bits, group {args.group}")
    print(f"{'layer':>6} " + " ".join(f"{m:>22}" for m in meas) + " "
          + " ".join(f"{n:>16}" for n in names))
    for i, li in enumerate(layers):
        print(f"{li:>6} " + " ".join(f"{meas[m][i]:>21.2f}x" for m in meas) + " "
              + " ".join(f"{feats[i][n]:>16.3f}" for n in names))
    print()
    depth = [float(li) for li in layers]
    for m, ys in meas.items():
        print(f"  rho(depth, {m}) = {spearman(depth, ys):+.2f}")
        for n in names:
            xs = [f[n] for f in feats]
            rho = spearman(xs, ys)
            lo, hi = boot_ci(xs, ys)
            print(f"  rho({n:>16}, {m:<22}) = {rho:+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]"
                  f"  permutation p = {perm_p(xs, ys):.4g}")
    print("\nRead: a law needs a clear positive rho that is not just the depth trend; compare "
          "rho(depth, ·) above. With 28 layers the smallest permutation p is about 1/10000.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
