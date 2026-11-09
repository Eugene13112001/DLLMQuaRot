"""Block diffusion decoding loops for SDAR and Fast-dLLM v2, written out so that
the KV cache of every committed block passes through one place (`on_store`).

Both loops reproduce the models' own greedy decoding:
  * SDAR       — `block_diffusion_generate` from JetAstra/SDAR generate.py,
                 strategy `low_confidence_dynamic`, argmax instead of sampling.
  * Fast-dLLM v2 — `generate()` in the model's modeling.py with temperature 0,
                 no block cache (sub-blocks re-run the whole block window).

`on_store(cache, start, end)` is called right after positions [start, end) are
written into the cache: prompt prefill and each finished block. Stage 1.2 hooks
the quantizer here; the current block is never in the cache, so it stays exact.
"""
import math
from dataclasses import dataclass, field

import torch


@dataclass
class Stats:
    steps: int = 0          # denoising forwards (one per unmasking step)
    nfe: int = 0            # all forwards: prefill + steps + block commits
    forced: int = 0         # tokens committed with confidence <= threshold
    blocks: int = 0         # generated blocks
    cache_len: int = 0      # tokens in the KV cache at the end
    per_step: list = field(default_factory=list)  # tokens committed at each step

    def record(self, n_commit, n_forced):
        self.steps += 1
        self.nfe += 1
        self.forced += n_forced
        self.per_step.append(n_commit)


def first_stop(tokens, stop_ids):
    """Index of the first stop token in a 1-D tensor, or None."""
    hits = torch.isin(tokens, torch.tensor(list(stop_ids), device=tokens.device)).nonzero()
    return int(hits[0]) if len(hits) else None


def _greedy(logits):
    probs = torch.softmax(logits.float(), dim=-1)
    conf, tok = probs.max(dim=-1)
    return tok, conf


@torch.no_grad()
def sdar_generate(model, prompt_ids, mask_id, block_length, denoising_steps,
                  threshold, gen_length, stop_ids, on_store=None, prefill_chunk=None):
    """Returns (generated token ids up to and including the stop token, Stats).

    `prefill_chunk`: feed the prompt to the cache in pieces of about this many tokens
    (whole blocks), so that a very long prompt fits in memory; `on_store` still sees
    the whole prompt once, after the last piece, as with a single forward.
    """
    from transformers.cache_utils import DynamicCache

    dev = prompt_ids.device
    P, B = prompt_ids.shape[1], block_length
    num_blocks = math.ceil((P + gen_length) / B)
    total = num_blocks * B

    block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=dev))
    attn = block_mask.repeat_interleave(B, 0).repeat_interleave(B, 1).unsqueeze(0)
    pos = torch.arange(total, device=dev).unsqueeze(0)

    x = torch.full((1, total), mask_id, dtype=torch.long, device=dev)
    x[:, :P] = prompt_ids
    cache = DynamicCache()
    st = Stats()

    def fwd(lo, hi, cur, store, hook=True):
        st.nfe += store
        out = model(cur, attention_mask=attn[:, lo:hi, :hi], position_ids=pos[:, lo:hi],
                    past_key_values=cache, use_cache=True, store_kv=store)
        if store:
            st.cache_len = hi
            if on_store is not None and hook:
                on_store(cache, lo, hi)
        return out

    # Split block_length into denoising_steps counts, remainder to the earliest steps.
    base, rem = divmod(B, denoising_steps)
    n_transfer = [base + (i < rem) for i in range(denoising_steps)]

    prefill = (P // B) * B
    if prefill > 0:
        step = max(B, prefill_chunk // B * B) if prefill_chunk else prefill
        for lo in range(0, prefill, step):
            hi = min(lo + step, prefill)
            fwd(lo, hi, x[:, lo:hi], True, hook=False)
        if on_store is not None:
            on_store(cache, 0, prefill)

    for nb in range(P // B, num_blocks):
        lo, hi = nb * B, (nb + 1) * B
        cur = x[:, lo:hi].clone()
        step = 0
        while True:
            masked = cur[0] == mask_id
            if not masked.any():
                fwd(lo, hi, cur, True)
                break
            logits = fwd(lo, hi, cur, False).logits[0]
            tok, conf = _greedy(logits)
            conf = torch.where(masked, conf, -torch.inf)
            k = max(1, min(n_transfer[min(step, denoising_steps - 1)], int(masked.sum())))
            high = conf > threshold
            commit = high if int(high.sum()) >= k else torch.zeros_like(masked).scatter(0, conf.topk(k).indices, True)
            cur[0, commit] = tok[commit]
            st.record(int(commit.sum()), int((commit & ~high).sum()))
            step += 1
        x[:, lo:hi] = cur
        st.blocks += 1
        if first_stop(x[0, P:hi], stop_ids) is not None:
            break

    gen = x[0, P:hi]
    s = first_stop(gen, stop_ids)
    return (gen if s is None else gen[:s + 1]), st


@torch.no_grad()
def fastdllm_generate(model, prompt_ids, mask_id, block_size, small_block_size,
                      threshold, max_new_tokens, stop_ids, on_store=None):
    """Returns (generated token ids up to and including the stop token, Stats)."""
    P, B = prompt_ids.shape[1], block_size
    assert P > B, "prompt shorter than one block: the reference generate() is not mirrored there"
    st = Stats()

    def fwd(window, store, cache):
        st.nfe += store
        out = model(input_ids=window, use_cache=True, past_key_values=cache,
                    update_past_key_values=store)
        if store:
            n = st.cache_len = out.past_key_values.get_seq_length()
            if on_store is not None:
                on_store(out.past_key_values, n - window.shape[1], n)
        return out

    x = prompt_ids
    out = fwd(x[:, :(P // B) * B], True, None)
    cache = out.past_key_values
    if P % B == 0:
        x = torch.cat([x, out.logits[:, -1:].argmax(-1)], dim=1)

    def finished(seq):
        gen = seq[0, P:]
        s = first_stop(gen, stop_ids)
        return s is not None and not (gen[:s] == mask_id).any()

    done = False
    for _ in range(max_new_tokens // B):
        if first_stop(x[0, P:], stop_ids) is not None:
            break
        pad = B - x.shape[1] % B
        x = torch.cat([x, torch.full((1, pad), mask_id, dtype=torch.long, device=x.device)], dim=1)
        for sb in range(B // small_block_size):
            start = -B + sb * small_block_size
            end = None if sb * small_block_size + small_block_size == B else start + small_block_size
            while (x[0, start:end] == mask_id).any():
                logits = fwd(x[:, -B:], False, cache).logits
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)[0, start:end]
                tok, conf = _greedy(logits)
                masked = x[0, start:end] == mask_id
                conf = torch.where(masked, conf, -torch.inf)
                high = conf > threshold
                commit = high.clone()
                commit[conf.argmax()] = True
                commit &= masked
                x[0, start:end] = torch.where(commit, tok, x[0, start:end])
                st.record(int(commit.sum()), int((commit & ~high).sum()))
                if finished(x):
                    done = True
                    break
            if done:
                break
        if done:
            break
        out = fwd(x[:, -B:], True, cache)
        cache = out.past_key_values
        st.blocks += 1
        x = torch.cat([x, out.logits[:, -1:].argmax(-1)], dim=1)

    gen = x[0, P:]
    s = first_stop(gen, stop_ids)
    return (gen if s is None else gen[:s + 1]), st
