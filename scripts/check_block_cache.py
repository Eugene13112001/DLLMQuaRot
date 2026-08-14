"""What does storing the KV cache in 4 bits actually cost?

This is the project's subject, measured directly and with nothing else mixed
in. Block-causal attention makes the prefix exactly reusable -- a position in
an earlier block attends only to blocks that are already fixed -- so a cached
prefix and a recomputed one differ for exactly one reason: the bits it was
stored in. Staleness, the other half of a cache's error, is structurally zero
here and enters only when the *current* block is cached too.

    bash scripts/llada2.sh scripts/check_block_cache.py \
        --model inclusionAI/LLaDA2.0-mini --model-type llada2_moe

Two things are reported per configuration:

* the relative error in the logits, which is what quality follows from;
* the share of positions whose argmax survives -- in a diffusion LM the
  argmax is the token that gets committed, and a commitment is irreversible,
  so an ordering that flips matters more than a logit that moves.

Both are measured at several mask ratios, because the K/V of a masked
position and a decoded one are not alike: at a high mask ratio nearly every
position carries the same `[MASK]` embedding row, their K/V are near-identical,
and their rounding errors add coherently instead of cancelling.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dllmquant.cache import BlockKVCache, KVCacheConfig  # noqa: E402
from dllmquant.calib.prompts import load_prompts  # noqa: E402
from dllmquant.config import DLLMQuantConfig  # noqa: E402
from dllmquant.models import build_adapter  # noqa: E402
from dllmquant.rotate import dtype_invariance_tol  # noqa: E402
from dllmquant.models.llada2_local import (  # noqa: E402
    block_causal_mask,
    forward_window,
    install_block_cache,
    logits_for_window,
    refresh_prefix,
)


def masked_canvas(adapter, prompt: torch.Tensor, total: int, mask_ratio: float,
                  seed: int = 0) -> torch.Tensor:
    """Prompt, then a tail that is `mask_ratio` masked and otherwise decoded.

    Standing in for a real trajectory state: what matters for the cache is the
    proportion of positions still carrying the mask embedding, not how they
    got there.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.full((1, total), adapter.mask_id, dtype=torch.long)
    x[0, : prompt.shape[-1]] = prompt

    tail = torch.arange(prompt.shape[-1], total)
    n_decoded = int(round(len(tail) * (1.0 - mask_ratio)))
    if n_decoded:
        chosen = tail[torch.randperm(len(tail), generator=g)[:n_decoded]]
        vocab = int(adapter.model.config.vocab_size)
        x[0, chosen] = torch.randint(0, vocab - 1, (n_decoded,), generator=g)
    return x


@torch.no_grad()
def full_logits(model, x, block_length):
    dtype = next(model.parameters()).dtype
    mask = block_causal_mask(
        x.shape[-1], block_length, batch_size=x.shape[0],
        device=x.device, dtype=dtype,
    )
    positions = torch.arange(x.shape[-1], device=x.device).unsqueeze(0)
    return forward_window(model, x, positions, mask)


def compare(reference: torch.Tensor, actual: torch.Tensor) -> tuple:
    ref, act = reference.float(), actual.float()
    denom = ref.abs().mean().clamp(min=1e-8)
    rel = float((act - ref).abs().mean() / denom)
    agree = float((act.argmax(-1) == ref.argmax(-1)).float().mean())
    return rel, agree


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-type", default="llada2_moe", choices=["llada2_moe"])
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--blocks", type=int, default=8,
                    help="total sequence length, in blocks")
    ap.add_argument("--bits", type=int, nargs="+", default=[16, 8, 4, 3, 2])
    ap.add_argument("--group-sizes", type=int, nargs="+", default=[32, 64, 128],
                    help="channels sharing one scale, along head_dim. The "
                         "coarsest useful value is head_dim itself -- one "
                         "scale per head per token -- because quantize_kv "
                         "groups along that axis and cannot span heads. For "
                         "this checkpoint head_dim is 128, so 128 is already "
                         "the whole head and anything larger is the same run.")
    ap.add_argument("--mask-ratios", type=float, nargs="+",
                    default=[1.0, 0.5, 0.0])
    ap.add_argument("--kv-pairs", nargs="*",
                    default=["4,4", "8,4", "4,8", "3,4", "4,3", "2,4", "4,2"],
                    help="K,V bit pairs for the asymmetry sweep. Pairs are "
                         "listed so that mirror images sit next to each other: "
                         "8,4 and 4,8 cost the same memory, so whichever wins "
                         "says which side of attention is worth the bits. "
                         "Empty to skip.")
    ap.add_argument("--kv-pair-group", type=int, default=128,
                    help="group size for the asymmetry sweep, kept fixed so "
                         "the only thing varying is where the bits go")
    args = ap.parse_args()

    cfg = DLLMQuantConfig(
        model_path=args.model, model_type=args.model_type,
        dtype=args.dtype, device=args.device,
    )
    adapter = build_adapter(cfg)
    adapter.load()
    print(adapter.describe())

    model = adapter.model
    device = next(model.parameters()).device
    head_dim = adapter.head_dim
    n_layers = len(adapter.blocks)

    total = args.blocks * args.block_length
    prefix_len = total - args.block_length  # cache everything but the last block
    prompt = adapter.encode_prompts(load_prompts(1), max_len=args.block_length)[0]

    floor = dtype_invariance_tol(args.dtype)
    print(f"\nsequence {total} tokens = {args.blocks} blocks of "
          f"{args.block_length}; prefix cached = {prefix_len}, "
          f"window recomputed = {args.block_length}")
    print(f"{args.dtype}: the 16-bit control row is exact only in exact "
          f"arithmetic; noise floor here is {floor:.0e}, and argmax is what "
          f"must hold at 100%")

    states = install_block_cache(model, BlockKVCache(KVCacheConfig(), n_layers))

    for mask_ratio in args.mask_ratios:
        x = masked_canvas(adapter, prompt, total, mask_ratio).to(device)
        reference = full_logits(model, x, args.block_length)[:, prefix_len:]
        print(f"\n--- mask ratio {mask_ratio:.2f} " + "-" * 44)
        print(f"{'bits':>5} {'group':>7} {'rel. logit error':>18} {'argmax kept':>13}")

        groups = sorted({min(g, head_dim) for g in args.group_sizes})
        for group_size in groups:
            for bits in args.bits:
                cache = BlockKVCache(
                    KVCacheConfig(
                        enabled=True, decoded_bits=bits, masked_bits=bits,
                        group_size=group_size,
                    ),
                    n_layers,
                )
                for state in states:
                    state.cache = cache

                refresh_prefix(model, states, x, prefix_len, args.block_length)
                windowed = logits_for_window(
                    model, states, x, prefix_len, total, args.block_length
                )
                rel, agree = compare(reference, windowed)
                label = f"{group_size}=head" if group_size == head_dim else str(group_size)
                flag = ""
                if bits >= 16 and (rel > floor or agree < 1.0):
                    # 16 bits stores the tensor unchanged, so what is left is
                    # the plumbing -- but "unchanged" is only exact in exact
                    # arithmetic. The windowed forward multiplies a 32-row
                    # query against the keys where the full one multiplies
                    # 256, and in bfloat16 a different accumulation order is a
                    # different answer. The floor is the same one the rotation
                    # check uses, for the same reason; below it, argmax is the
                    # signal that matters, because a token that flips is a
                    # token committed differently and commitment is final.
                    flag = "   <-- NOT EXACT: the cache path itself is wrong"
                print(f"{bits:>5} {label:>7} {rel:>18.3e} {100*agree:>12.2f}%{flag}")

    print("\nThe 16-bit row is the control: quantize_kv returns the tensor "
          "untouched there, so it measures the windowed forward against the "
          "full one and nothing else -- but only in exact arithmetic. The "
          "windowed forward multiplies a 32-row query against the keys where "
          "the full one multiplies every row, and a different accumulation "
          f"order is a different answer in {args.dtype}. Below the floor of "
          f"{floor:.0e} the number to read is argmax: a logit that moves "
          "changes nothing, a logit ordering that flips changes which token "
          "gets committed, and commitment is irreversible.\n"
          "To see the control at 1e-04, rerun with --dtype float32.")

    if args.kv_pairs:
        run_kv_asymmetry(
            adapter, model, states, args, n_layers, head_dim, device, prefix_len, total
        )
    return 0


def run_kv_asymmetry(
    adapter, model, states, args, n_layers, head_dim, device, prefix_len, total
) -> None:
    """Does K deserve more bits than V, or the other way round?

    K and V are stored identically everywhere in this codebase, and there is no
    reason they should be. K is consumed by the softmax: an error there is
    exponentiated and redistributes attention mass across every position. V is
    summed against weights that are by then already decided, so its error
    passes through linearly. Which asymmetry that produces is an empirical
    question nobody has asked for a diffusion LM.

    Mirror pairs cost identical memory, so the comparison is free of any
    budget argument: if 8,4 beats 4,8, the bits belong on K.
    """
    pairs = []
    for spec in args.kv_pairs:
        k_bits, v_bits = (int(p) for p in spec.split(","))
        pairs.append((k_bits, v_bits))

    group = min(args.kv_pair_group, head_dim)
    prompt = adapter.encode_prompts(load_prompts(1), max_len=args.block_length)[0]

    print(f"\n=== K/V asymmetry, group {group} " + "=" * 34)
    print("mirror pairs cost the same memory; the winner says where bits belong")

    for mask_ratio in args.mask_ratios:
        x = masked_canvas(adapter, prompt, total, mask_ratio).to(device)
        reference = full_logits(model, x, args.block_length)[:, prefix_len:]
        print(f"\n--- mask ratio {mask_ratio:.2f} " + "-" * 44)
        print(f"{'K bits':>7} {'V bits':>7} {'rel. logit error':>18} {'argmax kept':>13}")

        for k_bits, v_bits in pairs:
            cache = BlockKVCache(
                KVCacheConfig(
                    enabled=True, group_size=group,
                    key_bits=k_bits, value_bits=v_bits,
                ),
                n_layers,
            )
            for state in states:
                state.cache = cache

            refresh_prefix(model, states, x, prefix_len, args.block_length)
            windowed = logits_for_window(
                model, states, x, prefix_len, total, args.block_length
            )
            rel, agree = compare(reference, windowed)
            print(f"{k_bits:>7} {v_bits:>7} {rel:>18.3e} {100*agree:>12.2f}%")


if __name__ == "__main__":
    raise SystemExit(main())
