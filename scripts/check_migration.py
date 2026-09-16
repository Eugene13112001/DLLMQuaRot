"""Does moving the K-norm gain into the Q-norm really leave the model alone? Ask the checkpoint.

The whole point of the migration is that it is an identity on attention, so the two-bit collapse
cells can be re-run with the fixed-channel peaks gone and nothing else changed. That claim is
checked on a stand-in in the tests; here it is checked on the real weights, in one forward pass,
before anything spends hours of GPU on the cells themselves:

* the attention probabilities of a block, before and after -- the quantity the migration promises
  not to touch;
* the largest key value and the crest factor of the stored tensor, before and after -- the
  quantity it promises to change;
* the spread of the gains, so the print says how coarse the power-of-two flattening turned out.

    bash scripts/llada2.sh scripts/check_migration.py --model inclusionAI/LLaDA2.0-mini \\
        --model-type llada2_moe --alpha 1.0
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from dllmquant.algos.smooth_qk import _norms, gain_scales, migrate_qk_gains  # noqa: E402
from dllmquant.config import DLLMQuantConfig, QuantConfig  # noqa: E402
from dllmquant.models import build_adapter  # noqa: E402

from check_block_cache import text_ids  # noqa: E402


def key_stats(adapter, block, x):
    probe = adapter.make_probe(block)
    with probe, torch.no_grad():
        adapter.model(x, **adapter.forward_kwargs(x))
        k = probe.parts.key_states.detach().float()
        p = probe.parts.attn_probs.detach().float().clone()
    flat = k.reshape(k.shape[1], -1)
    crest = float((flat.abs().amax(-1) / flat.pow(2).mean(-1).sqrt()).mean())
    return p, float(k.abs().max()), crest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-type", default="llada2_moe", choices=["llada2_moe", "llada"])
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--mode", default="shrink", choices=["shrink", "geomean"])
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--done", default=None,
                    help="write this marker file when the check passes, so a queue can gate on it")
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="blocks to inspect; default is every block")
    args = ap.parse_args()

    cfg = DLLMQuantConfig(model_path=args.model, model_type=args.model_type,
                          dtype=args.dtype, device=args.device,
                          weight=QuantConfig(n_bits=16), activation=QuantConfig(n_bits=16))
    adapter = build_adapter(cfg)
    adapter.load()
    print(adapter.describe())

    depth = len(adapter.blocks)
    idx = args.layers or list(range(depth))
    x = text_ids(adapter, args.seq_len, seed=0).unsqueeze(0).to(adapter.model.device)

    before = {li: key_stats(adapter, adapter.blocks[li], x) for li in idx}
    rd = adapter._probe_rotary_dim() or adapter.head_dim
    gains = {li: _norms(adapter.blocks[li])[1].weight.detach().float().cpu() for li in idx}

    undo = migrate_qk_gains(adapter, args.alpha, mode=args.mode)
    try:
        after = {li: key_stats(adapter, adapter.blocks[li], x) for li in idx}
    finally:
        undo()

    spread = lambda t: float(t.max() / t.median().clamp_min(1e-9))
    print()
    print(f"{'layer':>6} {'attn drift':>12} {'|K| max':>20} {'crest K':>16} {'gain max/median':>22}")
    print("-" * 80)
    worst, louder, c0s, c1s = 0.0, [], [], []
    for li in idx:
        p0, m0, c0 = before[li]
        p1, m1, c1 = after[li]
        drift = float((p1 - p0).abs().max())
        worst = max(worst, drift)
        c0s.append(c0)
        c1s.append(c1)
        if c1 > c0 * 1.01 or m1 > m0 * 1.01:
            louder.append(li)
        g = gains[li].abs()
        flat = g / gain_scales(gains[li], rd, args.alpha, mode=args.mode).float()
        print(f"{li:>6} {drift:>12.2e} {m0:>9.2f} -> {m1:<8.2f} {c0:>7.2f} -> {c1:<7.2f} "
              f"{spread(g):>10.1f} -> {spread(flat):<10.1f}")

    mean0, mean1 = sum(c0s) / len(c0s), sum(c1s) / len(c1s)
    print()
    print("attention probabilities moved by at most {:.2e}".format(worst))
    print("crest of the stored keys, mean over layers: {:.2f} -> {:.2f}".format(mean0, mean1))
    if worst > 1e-3:
        print("STOP: that is not rounding. The factor is not commuting with RoPE -- check the "
              "rotary width and the pairing before running anything on this.")
        return 1
    if louder:
        print("WARNING: keys got louder on layers {}".format(louder))
    if mean1 >= mean0 * 0.95:
        print("STOP: the model is unchanged, but the keys did not lose their peaks either, so the "
              "two-bit cells would test nothing. Do not spend cards on them.")
        return 1
    print("The model is unchanged and the stored keys lost their peaks. The two-bit cells can be "
          "re-run: whatever changes there is the gain.")
    if args.done:
        with open(args.done, "w", encoding="utf-8") as fh:
            fh.write("attention drift {:.3e}, crest {:.2f} -> {:.2f}, alpha {}, {}".format(
                worst, mean0, mean1, args.alpha, args.mode))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
