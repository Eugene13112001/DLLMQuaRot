"""SDAR's block-diffusion sampler, greedy, counting its own steps.

This is ``block_diffusion_generate`` from the authors' ``generate.py`` (JetAstra/SDAR) with
the ``low_confidence_dynamic`` strategy, changed in two ways and no others:

* **greedy**: the token is the argmax and its confidence the softmax probability of it. The
  script samples with ``multinomial`` even at temperature 1, so two runs differ; the model
  card reports greedy numbers, and the comparison across bit widths needs a run to repeat.
* **it counts**: denoising forwards, forced commits, tokens per step. A forced commit is the
  script's fallback branch -- fewer masked positions clear the threshold than the step's
  quota, so the quota is filled with the most confident ones anyway. That is where
  irreversible mistakes concentrate (DECAL), and a compressed cache that makes the model
  less sure shows up as more steps before it shows up as lower accuracy.

The cache is handed in, as the script does: ``DynamicCache()`` built by the caller and
written only on the ``store_kv=True`` passes -- the aligned part of the prompt once, then
each finished block once. The current block is recomputed at every step from the store and
never written, so a store that rounds on write (``dllmquant.committed_cache``) rounds
exactly the prompt and the finished blocks.

The model's own code is loaded with ``trust_remote_code`` at a pinned revision; the hash of
``modeling_sdar.py`` is the same for the b4 and b32 checkpoints (checked 2026-11-02).
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import torch

# block size -> checkpoint. The unsuffixed one is block 4: its card sets block_length=4.
SDAR_8B = {
    4: "JetLM/SDAR-8B-Chat",
    8: "JetLM/SDAR-8B-Chat-b8",
    16: "JetLM/SDAR-8B-Chat-b16",
    32: "JetLM/SDAR-8B-Chat-b32",
    64: "JetLM/SDAR-8B-Chat-b64",
}
SDAR_30B = {
    4: "JetLM/SDAR-30B-A3B-Chat",
    8: "JetLM/SDAR-30B-A3B-Chat-b8",
    16: "JetLM/SDAR-30B-A3B-Chat-b16",
    32: "JetLM/SDAR-30B-A3B-Chat-b32",
    64: "JetLM/SDAR-30B-A3B-Chat-b64",
}
MASK_ID = 151669
STOP_IDS = (151645, 151643)   # generation_config.eos_token_id


def transfer_quota(block_length: int, steps: int) -> torch.Tensor:
    """Tokens a step must commit at least: the script's ``get_num_transfer_tokens``."""
    base, rem = divmod(block_length, steps)
    q = torch.full((steps,), base, dtype=torch.int64)
    q[:rem] += 1
    return q


def choose(logits: torch.Tensor, masked: torch.Tensor, quota: int, threshold: float
           ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    """One step of ``low_confidence_dynamic``, greedy. Batch of one.

    Returns (tokens, positions to commit, forced). Forced is the fallback: fewer than
    ``quota`` masked positions are above the threshold, so the top ``quota`` are taken.
    """
    probs = torch.softmax(logits.float(), dim=-1)
    conf, x0 = probs.max(dim=-1)
    conf = torch.where(masked, conf, torch.full_like(conf, -float("inf")))
    sure = conf[0] > threshold
    take = torch.zeros_like(masked)
    if int(sure.sum()) >= quota:
        take[0] = sure
        return x0, take, False
    idx = torch.topk(conf[0], quota).indices
    take[0, idx] = True
    return x0, take, True


@torch.no_grad()
def sdar_generate(
    model,
    prompt_ids: torch.Tensor,
    *,
    cache,
    block_length: int,
    gen_length: int = 2048,
    denoising_steps: Optional[int] = None,
    threshold: float = 0.9,
    mask_id: int = MASK_ID,
    stop_ids: Iterable[int] = STOP_IDS,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Generate one answer. Returns (ids up to the last decoded block, counts).

    ``denoising_steps`` defaults to the block length, i.e. at least one token per step --
    the card's setting at block 4 (4 and 4), carried to every block size so that the
    threshold, not the quota, decides how many tokens a step commits.
    """
    steps_cap = denoising_steps or block_length
    ids = prompt_ids if prompt_ids.dim() > 1 else prompt_ids.unsqueeze(0)
    dev = model.device
    ids = ids.to(dev)
    prompt_length = ids.shape[1]
    stops = torch.tensor(list(stop_ids), device=dev)

    num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
    total = num_blocks * block_length
    block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=dev))
    attn = block_mask.repeat_interleave(block_length, 0) \
                     .repeat_interleave(block_length, 1).unsqueeze(0)
    pos = torch.arange(total, device=dev).unsqueeze(0)

    x = torch.full((1, total), mask_id, dtype=torch.long, device=dev)
    x[:, :prompt_length] = ids
    prefill_blocks = prompt_length // block_length
    prefill = prefill_blocks * block_length
    if prefill > 0:
        model(x[:, :prefill], attention_mask=attn[:, :prefill, :prefill],
              position_ids=pos[:, :prefill], past_key_values=cache,
              use_cache=True, store_kv=True)

    quota = transfer_quota(block_length, steps_cap)
    counts = {"steps": 0, "forced_steps": 0, "forced_tokens": 0, "committed": 0,
              "blocks": 0, "prefill_tokens": prefill}
    end = prefill
    for b in range(prefill_blocks, num_blocks):
        lo, hi = b * block_length, (b + 1) * block_length
        cur = x[:, lo:hi].clone()
        cur_attn = attn[:, lo:hi, :hi]
        cur_pos = pos[:, lo:hi]
        for step in range(steps_cap + 1):
            masked = cur == mask_id
            if not masked.any():
                model(cur, attention_mask=cur_attn, position_ids=cur_pos,
                      past_key_values=cache, use_cache=True, store_kv=True)
                break
            logits = model(cur, attention_mask=cur_attn, position_ids=cur_pos,
                           past_key_values=cache, use_cache=True, store_kv=False).logits
            q = int(quota[min(step, steps_cap - 1)])
            x0, take, forced = choose(logits, masked, q, threshold)
            cur[take] = x0[take]
            n = int(take.sum())
            counts["steps"] += 1
            counts["committed"] += n
            if forced:
                counts["forced_steps"] += 1
                counts["forced_tokens"] += n
        x[:, lo:hi] = cur
        counts["blocks"] += 1
        end = hi
        if torch.isin(x[:, prompt_length:end], stops).any():
            break
    counts["stored_tokens"] = end
    return x[:, :end], counts


def load_sdar(name: str, *, revision: Optional[str] = None, dtype: str = "bfloat16",
              device: str = "cuda"):
    """Model and tokenizer, remote code at ``revision`` (None: the hub's main)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        name, revision=revision, trust_remote_code=True,
        torch_dtype=getattr(torch, dtype), device_map=device)
    model.eval()
    tok = AutoTokenizer.from_pretrained(name, revision=revision, trust_remote_code=True)
    mask = tok.convert_tokens_to_ids(tok.mask_token) if tok.mask_token else MASK_ID
    if mask != MASK_ID:
        raise RuntimeError(f"{name}: mask token id {mask}, expected {MASK_ID}")
    return model, tok


__all__ = ["SDAR_8B", "SDAR_30B", "MASK_ID", "STOP_IDS", "sdar_generate", "choose",
           "transfer_quota", "load_sdar"]
