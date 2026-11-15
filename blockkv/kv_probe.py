"""One-step effect of KV compression schemes, measured on the exact decoding states.

    python -m blockkv.kv_probe --model sdar8b --block 32 --limit 30 --out out/probe/sdar8b_b32.json
    python -m blockkv.kv_probe --model fastdllm --block 32 --limit 30 --out out/probe/fastdllm_b32.json

The model decodes GSM8K without compression (threshold 0.9, our decoders). On every `--every`-th
denoising forward (at most `--per-problem` per problem) the same forward is repeated with a copy of
the cache compressed by each scheme, as DECAL compares a quantized model with a full-precision shadow
on identical states. Per scheme, averaged over the probed masked positions:

  relerr_k / relerr_v      ||dK||^2 / ||K||^2 over the cache (what KVQuant reports)
  crelerr_k / crelerr_v    ||dK||^2 / ||K - mean_tokens(K)||^2: relative to the part that varies from
                           token to token. A constant offset in a channel (e.g. a k_proj bias) adds
                           the same q.b to every logit of a query, so softmax ignores it; the first
                           measure counts it in the denominator, this one does not.
  conf_drop                mean drop of the top-1 probability
  gap_shrink               mean relative change of the top-1 minus top-2 logit gap (DECAL's measure)
  flip                     share of positions whose argmax token changes
  fall_below               share of positions above the threshold exactly that drop to or below it
  rise_above               the reverse
  kl                       KL(exact || compressed) of the next-token distribution

Schemes are written like the stage1_kv.sh specs: "2", "3", "4", "2k", "2v", "k4v2", "k2v4",
"2rot" (Hadamard, keys per token), "2kb0.0625" (Kitty-like boost of key channels to 8 bits).
"""
import argparse
import copy
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import torch

from blockkv import decode, gsm8k
from blockkv.kvquant import KVQuant, cache_layers
from blockkv.run_gsm8k import DEFAULTS, mask_id, repo, resolve_path, stop_ids

SCHEMES = ["2", "3", "4", "2k", "2v", "k4v2", "k2v4", "2rot", "3rot", "4rot",
           "2kb0.0625", "2kb0.125", "2kb0.25"]


def make_quant(spec, group=32, head_dim=128):
    """KVQuant from a scheme spec."""
    m = re.fullmatch(r"k(\d+)v(\d+)", spec)
    if m:
        return KVQuant(int(m[1]), group, k_bits=int(m[1]), v_bits=int(m[2]), head_dim=head_dim)
    m = re.fullmatch(r"(\d+)rot", spec)
    if m:
        return KVQuant(int(m[1]), group, key_mode="token", rotate=True, head_dim=head_dim)
    m = re.fullmatch(r"(\d+)kb([\d.]+)", spec)
    if m:
        return KVQuant(int(m[1]), group, key_boost=float(m[2]), key_boost_bits=8, head_dim=head_dim)
    m = re.fullmatch(r"(\d+)([kv]?)", spec)
    if m:
        return KVQuant(int(m[1]), group, only=m[2] or None, head_dim=head_dim)
    raise ValueError(spec)


def cache_errors(clean, quant):
    """Sums for relerr and crelerr over all layers: {k: [err, energy, spread], v: [...]}."""
    out = {"k": torch.zeros(3, dtype=torch.float64), "v": torch.zeros(3, dtype=torch.float64)}
    for (kc, vc), (kq, vq) in zip(cache_layers(clean), cache_layers(quant)):
        for name, c, q in (("k", kc, kq), ("v", vc, vq)):
            c, q = c.float(), q.float()
            out[name] += torch.tensor([(q - c).pow(2).sum().item(), c.pow(2).sum().item(),
                                       (c - c.mean(2, keepdim=True)).pow(2).sum().item()], dtype=torch.float64)
    return out


def step_metrics(lc, lq, threshold):
    """lc, lq: (n, vocab) logits of the probed masked positions, exact and compressed."""
    pc, pq = lc.float().softmax(-1), lq.float().softmax(-1)
    c1, ic = pc.max(-1)
    q_at = pq.gather(-1, ic[:, None])[:, 0]
    q1, iq = pq.max(-1)
    top_c, top_q = lc.float().topk(2, -1).values, lq.float().topk(2, -1).values
    gap_c, gap_q = top_c[:, 0] - top_c[:, 1], top_q[:, 0] - top_q[:, 1]
    kl = (pc * (pc.clamp_min(1e-30).log() - pq.clamp_min(1e-30).log())).sum(-1)
    return dict(n=len(c1), conf_drop=(c1 - q1).sum().item(), conf_at_exact=(c1 - q_at).sum().item(),
                gap_shrink=((gap_c - gap_q) / gap_c.clamp_min(1e-6)).sum().item(),
                flip=(ic != iq).sum().item(),
                above=(c1 > threshold).sum().item(),
                fall_below=((c1 > threshold) & (q1 <= threshold)).sum().item(),
                rise_above=((c1 <= threshold) & (q1 > threshold)).sum().item(),
                kl=kl.sum().item())


class Prober:
    """Wraps model.forward: on sampled non-storing forwards, reruns them on compressed cache copies."""

    def __init__(self, model, family, mid, schemes, threshold, every, per_problem, head_dim):
        self.model, self.family, self.mid = model, family, mid
        self.schemes, self.threshold, self.every, self.per_problem = schemes, threshold, every, per_problem
        self.head_dim = head_dim
        self.orig = model.forward
        self.calls = 0
        self.probed = 0
        self.stats = {s: defaultdict(float) for s in schemes}
        model.forward = self.forward

    def new_problem(self):
        self.calls, self.probed = 0, 0

    def forward(self, *args, **kw):
        out = self.orig(*args, **kw)
        store = kw.get("store_kv", kw.get("update_past_key_values", True))
        cache = kw.get("past_key_values")
        if store or cache is None:
            return out
        self.calls += 1
        if self.calls % self.every or self.probed >= self.per_problem:
            return out
        ids = args[0] if args else kw["input_ids"]
        masked = ids[0] == self.mid
        if not masked.any():
            return out
        self.probed += 1
        lc = self._masked_logits(out.logits, masked)
        length = cache_layers(cache)[0][0].shape[2]
        with torch.no_grad():
            for s in self.schemes:
                cq = copy.deepcopy(cache)
                make_quant(s, head_dim=self.head_dim)(cq, 0, length)
                kq = dict(kw, past_key_values=cq)
                lq = self._masked_logits(self.orig(*args, **kq).logits, masked)
                st = self.stats[s]
                for k, v in step_metrics(lc, lq, self.threshold).items():
                    st[k] += v
                e = cache_errors(cache, cq)
                for name in "kv":
                    st[f"err_{name}"] += e[name][0].item()
                    st[f"energy_{name}"] += e[name][1].item()
                    st[f"spread_{name}"] += e[name][2].item()
                st["probes"] += 1
                del cq
        return out

    def _masked_logits(self, logits, masked):
        lg = logits[0]
        if self.family == "fastdllm":               # Fast-dLLM v2 predicts position i from i - 1
            lg = torch.cat([lg[:1], lg[:-1]], 0)
        return lg[masked]

    def summary(self):
        res = {}
        for s, st in self.stats.items():
            n = max(st["n"], 1)
            res[s] = dict(probes=int(st["probes"]), positions=int(st["n"]),
                          honest_bits=round(make_quant(s, head_dim=self.head_dim).honest_bits, 3),
                          relerr_k=st["err_k"] / max(st["energy_k"], 1e-30),
                          relerr_v=st["err_v"] / max(st["energy_v"], 1e-30),
                          crelerr_k=st["err_k"] / max(st["spread_k"], 1e-30),
                          crelerr_v=st["err_v"] / max(st["spread_v"], 1e-30),
                          conf_drop=st["conf_drop"] / n, conf_at_exact=st["conf_at_exact"] / n,
                          gap_shrink=st["gap_shrink"] / n, flip=st["flip"] / n,
                          fall_below=st["fall_below"] / max(st["above"], 1),
                          rise_above=st["rise_above"] / max(n - st["above"], 1), kl=st["kl"] / n)
        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["sdar8b", "fastdllm"])
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0, help="the problems are a fixed random subset")
    ap.add_argument("--every", type=int, default=4, help="probe every N-th denoising forward")
    ap.add_argument("--per-problem", type=int, default=12)
    ap.add_argument("--schemes", nargs="+", default=SCHEMES)
    ap.add_argument("--small-block", type=int, default=8)
    ap.add_argument("--gen-length", type=int, default=512, help="answer limit (GSM8K answers are ~160-320 tokens)")
    ap.add_argument("--data")
    ap.add_argument("--model-path")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    family, repo_id = repo(args.model, args.block)
    path = resolve_path(repo_id, args.model_path)
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=getattr(torch, DEFAULTS[family]["dtype"]),
                                                 trust_remote_code=True, device_map="cuda",
                                                 low_cpu_mem_usage=True).eval()
    mid, stops = mask_id(family, tok, None), stop_ids(family, tok, model)
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    head_dim = getattr(model.config, "head_dim", None) or head_dim
    prober = Prober(model, family, mid, args.schemes, args.threshold, args.every, args.per_problem, head_dim)

    problems = gsm8k.load(args.data)
    idxs = sorted(random.Random(args.seed).sample(range(len(problems)), args.limit))
    t0 = time.time()
    for n, i in enumerate(idxs):
        prober.new_problem()
        text = tok.apply_chat_template([{"role": "user", "content": gsm8k.PROMPT_STYLES[family].format(
            question=problems[i]["question"])}], tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt", add_special_tokens=family != "sdar").input_ids.cuda()
        with torch.no_grad():
            if family == "sdar":
                decode.sdar_generate(model, ids, mid, args.block, args.block, args.threshold, args.gen_length, stops)
            else:
                decode.fastdllm_generate(model, ids, mid, args.block, args.small_block, args.threshold,
                                         args.gen_length, stops)
        print(f"[{n + 1}/{len(idxs)}] idx={i} probes={prober.probed} {time.time() - t0:.0f}s", flush=True)

    res = dict(args=vars(args), head_dim=head_dim, schemes=prober.summary())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"{'scheme':12s} {'bits':>5s} {'relK':>7s} {'crelK':>7s} {'relV':>7s} {'crelV':>7s} {'conf':>7s} "
          f"{'gap':>6s} {'flip':>6s} {'below':>6s} {'kl':>7s}")
    for s, r in res["schemes"].items():
        print(f"{s:12s} {r['honest_bits']:5.2f} {r['relerr_k']:7.4f} {r['crelerr_k']:7.4f} {r['relerr_v']:7.4f} "
              f"{r['crelerr_v']:7.4f} {r['conf_drop']:7.4f} {r['gap_shrink']:6.3f} {r['flip']:6.3f} "
              f"{r['fall_below']:6.3f} {r['kl']:7.4f}")


if __name__ == "__main__":
    main()
