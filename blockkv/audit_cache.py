"""Audit the KV cache after real Fast-dLLM v2 generations.

Does not trust KVQuant's own counters: every committed block is copied before it
is quantized, and the finished cache is compared with those copies.

    python -m blockkv.audit_cache --data ~/data/gsm8k_test.jsonl --limit 20

Per layer: how many distinct levels each group really holds (at most 2^bits), the
share of numbers that changed, whether the tail is untouched, and the error
relative to the signal energy and to its spread around the group mean (the first
is small whenever a channel carries a large constant offset). Per problem: how
much the next block's prediction moves when the cache is swapped for the exact one.
"""
import argparse
import copy
from collections import defaultdict

import torch

from blockkv import decode, gsm8k
from blockkv.kvquant import KVQuant, cache_layers

REPO, MASK, STOP = "Efficient-Large-Model/Fast_dLLM_v2_7B", 151665, 151645


class Snapshot:
    """`on_store` hook: copies each new block, then lets `quant` compress it."""

    def __init__(self, quant):
        self.quant, self.clean, self.cache = quant, None, None

    def __call__(self, cache, start, end):
        new = [(k[:, :, start:end].clone(), v[:, :, start:end].clone()) for k, v in cache_layers(cache)]
        self.clean = new if self.clean is None else [
            (torch.cat([k0, k1], dim=2), torch.cat([v0, v1], dim=2))
            for (k0, v0), (k1, v1) in zip(self.clean, new)]
        self.cache = cache
        self.quant(cache, start, end)


def levels(x, dim):
    """Number of distinct values along `dim`."""
    return (x.sort(dim).values.diff(dim=dim) != 0).sum(dim) + 1


def _side(q, clean, dim, limit):
    err = (q - clean).pow(2).sum()
    spread = (clean - clean.mean(dim, keepdim=True)).pow(2).sum()
    n = levels(q, dim)
    return dict(over=(n > limit).sum(), groups=torch.tensor(n.numel()), max_levels=n.max(),
                changed=(q != clean).sum(), numbers=torch.tensor(q.numel()),
                err=err, energy=clean.pow(2).sum(), spread=spread)


def audit_layer(kq, vq, kc, vc, bits, group):
    """Raw sums for one layer: {"k": ..., "v": ..., "tail_exact": bool}."""
    assert kq.shape == kc.shape and vq.shape == vc.shape
    b, h, t, d = kq.shape
    n = t // group * group
    tail_exact = torch.equal(kq[:, :, n:], kc[:, :, n:]) and torch.equal(vq[:, :, n:], vc[:, :, n:])
    if n == 0:
        return dict(k=None, v=None, tail_exact=tail_exact)
    shape_k = (b, h, n // group, group, d)
    shape_v = (b, h, n, d // group, group)
    kq, kc = (x[:, :, :n].float().reshape(shape_k) for x in (kq, kc))
    vq, vc = (x[:, :, :n].float().reshape(shape_v) for x in (vq, vc))
    limit = 2 ** bits
    return dict(k=_side(kq, kc, 3, limit), v=_side(vq, vc, 4, limit), tail_exact=tail_exact)


class Totals:
    """Adds `audit_layer` results over problems, per layer."""

    def __init__(self):
        self.sums = defaultdict(lambda: {"k": defaultdict(float), "v": defaultdict(float)})
        self.tail_broken = 0

    def add(self, layer, res):
        self.tail_broken += not res["tail_exact"]
        for side in ("k", "v"):
            if res[side] is None:
                continue
            s = self.sums[layer][side]
            s["max_levels"] = max(s["max_levels"], int(res[side]["max_levels"]))
            for name, x in res[side].items():
                if name != "max_levels":
                    s[name] += float(x)

    def row(self, layer, side):
        s = self.sums[layer][side]
        return dict(max_levels=int(s["max_levels"]), over=s["over"] / max(s["groups"], 1),
                    changed=s["changed"] / max(s["numbers"], 1), rel=s["err"] / max(s["energy"], 1e-30),
                    centered=s["err"] / max(s["spread"], 1e-30))

    def report(self):
        print(f"{'layer':>5} | {'K lev':>5} {'K>lim':>6} {'K chg':>6} {'K rel':>8} {'K cent':>8}"
              f" | {'V lev':>5} {'V>lim':>6} {'V chg':>6} {'V rel':>8} {'V cent':>8}")
        for layer in sorted(self.sums):
            k, v = self.row(layer, "k"), self.row(layer, "v")
            print(f"{layer:5d} | {k['max_levels']:5d} {k['over']:6.3f} {k['changed']:6.3f} {k['rel']:8.4f} {k['centered']:8.4f}"
                  f" | {v['max_levels']:5d} {v['over']:6.3f} {v['changed']:6.3f} {v['rel']:8.4f} {v['centered']:8.4f}")
        print(f"layers with an altered tail: {self.tail_broken}")


def swap_clean(cache, clean):
    """A copy of `cache` holding the exact tensors."""
    other = copy.deepcopy(cache)
    other.key_cache = [k for k, _ in clean]
    other.value_cache = [v for _, v in clean]
    return other


def next_block_shift(model, cache, clean, block=32):
    """(share of positions whose top token changes, mean KL) for an all-mask next block."""
    window = torch.full((1, block), MASK, dtype=torch.long, device=model.device)
    run = lambda c: model(input_ids=window, use_cache=True, past_key_values=c,
                          update_past_key_values=False).logits[0].float()
    q, c = run(cache), run(swap_clean(cache, clean))
    kl = (c.log_softmax(-1).exp() * (c.log_softmax(-1) - q.log_softmax(-1))).sum(-1).mean()
    return float((q.argmax(-1) != c.argmax(-1)).float().mean()), float(kl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--bits", type=int, default=2)
    ap.add_argument("--threshold", type=float, default=1.0)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        REPO, torch_dtype=torch.bfloat16, trust_remote_code=True,
        device_map="cuda", low_cpu_mem_usage=True).eval()

    totals, shifts = Totals(), []
    with torch.no_grad():
        for i, p in enumerate(gsm8k.load(args.data)[:args.limit]):
            text = tok.apply_chat_template(
                [{"role": "user", "content": gsm8k.PROMPT_STYLES["fastdllm"].format(question=p["question"])}],
                tokenize=False, add_generation_prompt=True)
            ids = tok(text, return_tensors="pt").input_ids.cuda()
            snap = Snapshot(KVQuant(args.bits, 32))
            decode.fastdllm_generate(model, ids, MASK, 32, 8, args.threshold, 2048, [STOP], on_store=snap)
            for layer, ((kq, vq), (kc, vc)) in enumerate(zip(cache_layers(snap.cache), snap.clean)):
                totals.add(layer, audit_layer(kq, vq, kc, vc, args.bits, 32))
            shifts.append(next_block_shift(model, snap.cache, snap.clean))
            print(f"[{i + 1}/{args.limit}] cache {snap.cache.get_seq_length()} tokens, "
                  f"top-1 changed {shifts[-1][0]:.3f}, KL {shifts[-1][1]:.4f}", flush=True)

    totals.report()
    print(f"next block, quantized vs exact cache: top-1 changed {sum(s[0] for s in shifts) / len(shifts):.3f}, "
          f"KL {sum(s[1] for s in shifts) / len(shifts):.4f}")


if __name__ == "__main__":
    main()
