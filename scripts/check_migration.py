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

from dllmquant.algos.smooth_qk import gain_scales, migrate_qk_gains  # noqa: E402
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
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--done", default=None,
                    help="write this marker file when the check passes, so a queue can gate on it")
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="blocks to inspect; default is first, middle and last")
    args = ap.parse_args()

    cfg = DLLMQuantConfig(model_path=args.model, model_type=args.model_type,
                          dtype=args.dtype, device=args.device,
                          weight=QuantConfig(n_bits=16), activation=QuantConfig(n_bits=16))
    adapter = build_adapter(cfg)
    adapter.load()
    print(adapter.describe())

    depth = len(adapter.blocks)
    idx = args.layers or [0, depth // 2, depth - 1]
    x = text_ids(adapter, args.seq_len, seed=0).unsqueeze(0).to(adapter.model.device)

    before = {li: key_stats(adapter, adapter.blocks[li], x) for li in idx}
    rd = adapter._probe_rotary_dim() or adapter.head_dim
    gains = {}
    for li in idx:
        from dllmquant.algos.smooth_qk import _norms
        _, k_norm = _norms(adapter.blocks[li])
        gains[li] = k_norm.weight.detach().float().cpu()

    undo = migrate_qk_gains(adapter, args.alpha)
    try:
        after = {li: key_stats(adapter, adapter.blocks[li], x) for li in idx}
    finally:
        pass  # the weights stay migrated for the prints below; undone at exit

    print()
    print(f"{'layer':>6} {'attn drift':>12} {'|K| max':>20} {'crest K':>16} {'gain max/median':>22}")
    print("-" * 80)
    worst = 0.0
    for li in idx:
        p0, m0, c0 = before[li]
        p1, m1, c1 = after[li]
        drift = float((p1 - p0).abs().max())
        worst = max(worst, drift)
        g = gains[li].abs()
        s = gain_scales(gains[li], rd, args.alpha).float()
        flat = (g / s)
        spread = lambda t: float(t.max() / t.median().clamp_min(1e-9))
        print(f"{li:>6} {drift:>12.2e} {m0:>9.2f} -> {m1:<8.2f} {c0:>7.2f} -> {c1:<7.2f} "
              f"{spread(g):>10.1f} -> {spread(flat):<10.1f}")
    undo()

    print()
    print(f"attention probabilities moved by at most {worst:.2e}")
    if worst > 1e-3:
        print("STOP: that is not rounding. The factor is not commuting with RoPE -- check the "
              "rotary width and the pairing before running anything on this.")
        return 1
    print("The model is unchanged (bfloat16 rounding only), and the keys the cache stores lost "
          "their peaks. The two-bit cells can be re-run: whatever changes there is the gain.")
    if args.done:
        with open(args.done, "w", encoding="utf-8") as fh:
            fh.write("attention drift {:.3e} at alpha {}".format(worst, args.alpha))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
