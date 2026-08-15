"""What does storing the KV cache in 4 bits actually cost?

This is the project's subject, measured directly and with nothing else mixed
in. Block-causal attention makes the prefix exactly reusable -- a position in
an earlier block attends only to blocks that are already fixed -- so a cached
prefix and a recomputed one differ for exactly one reason: the bits it was
stored in. Staleness, the other half of a cache's error, is structurally zero
here and enters only when the *current* block is cached too.

    bash scripts/llada2.sh scripts/check_block_cache.py \
        --model inclusionAI/LLaDA2.0-mini --model-type llada2_moe

Four things are reported per configuration:

* the relative error in the logits, which is what quality follows from;
* the share of positions whose argmax survives -- in a diffusion LM the
  argmax is the token that gets committed, and a commitment is irreversible,
  so an ordering that flips matters more than a logit that moves;
* the same share restricted to the positions that would actually be committed
  this step, and
* whether those positions are even the same ones.

The last two exist because the plain argmax share counts every position in the
window equally, and the sampler does not. It commits the most confident
positions and leaves the rest masked for a later step, where they are decided
again from scratch. An argmax that flips on a position nobody was going to
commit costs nothing; the whole cost lives in the top of the confidence
ordering. Reporting only the flat share therefore states an upper bound on the
damage and invites reading it as the damage.

Two distinct failures hide in that top: the committed position can take a
different token, or a different position can become the most confident one and
get committed instead. `argmax@k` catches the first, `slots@k` the second.

Every table also opens with a **chance floor**: the same measurement taken with
a 16-bit cache whose entries have been shuffled along the token axis. Its
distribution is exactly that of a perfect cache and it carries none of the
information, so it is what these numbers look like when the cache has told the
model nothing. Neither the error nor the argmax share has a meaningful zero
without it -- two runs of the same model agree on a great deal by prior alone,
and a row sitting at the floor is dead however small its logit error looks.

The sweep is over the mask ratio of the **prefix**, which is the part that gets
cached, while the window stays masked because that is what a sampler step
reads. The two are separate for a reason: the K/V of a masked position and a
decoded one are not alike, so at a high prefix mask ratio nearly every cached
position carries the same `[MASK]` embedding row, their K/V are near-identical,
and their rounding errors add coherently instead of cancelling -- the usual
"quantization noise averages out" argument fails exactly there.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dllmquant.cache import BlockKVCache, KVCacheConfig  # noqa: E402
from dllmquant.calib.prompts import load_prompts  # noqa: E402
from dllmquant.config import DLLMQuantConfig  # noqa: E402
from dllmquant.models import build_adapter  # noqa: E402
from dllmquant.models.llada2_local import (  # noqa: E402
    block_causal_mask,
    forward_window,
    install_block_cache,
    logits_for_window,
    refresh_prefix,
)


def text_ids(adapter, total: int, seed: int = 0) -> torch.Tensor:
    """`total` tokens of actual text.

    What a decoded position holds is not a detail. An earlier version filled
    them with uniformly random ids, which is not a partly decoded trajectory --
    it is gibberish, and a model reading gibberish is uncertain about
    everything downstream of it. That flattens the very quantity the sweep is
    trying to resolve: with no coherent prefix there is little for the cache to
    carry, so destroying it costs little and every bit width scores alike.
    Real text puts the decision margins and the outlier structure where they
    are in deployment.
    """
    pieces, ids = load_prompts(64, seed=seed), []
    for i in range(len(pieces)):
        ids.extend(adapter.tokenizer(pieces[i % len(pieces)])["input_ids"])
        if len(ids) >= total:
            break
    while len(ids) < total:                      # short corpus, wrap around
        ids.extend(ids[: total - len(ids)])
    return torch.tensor(ids[:total], dtype=torch.long)


def masked_canvas(adapter, tokens: torch.Tensor, prompt_len: int, prefix_len: int,
                  prefix_ratio: float, window_ratio: float = 1.0,
                  seed: int = 0) -> torch.Tensor:
    """A canvas mid-trajectory: real text with holes punched in it.

    The prefix and the window are set separately, and an earlier version swept
    them together, which put every realistic configuration outside the table.
    The prefix is the closed blocks -- what gets cached -- and it empties of
    masks as the trajectory advances. The window is the block being decoded
    right now, and a sampler step looks at it while it is still mostly masked;
    a fully decoded window has nothing to commit and nothing to measure.

    So `prefix_ratio` is the variable of interest -- how far along the
    trajectory the cache was taken -- and `window_ratio` stays at 1.0, because
    that is what the step doing the reading looks like.
    """
    total = int(tokens.shape[-1])
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = tokens.reshape(1, total).clone()

    for lo, hi, ratio in (
        (prompt_len, prefix_len, prefix_ratio),
        (prefix_len, total, window_ratio),
    ):
        span = torch.arange(lo, hi)
        n_masked = int(round(len(span) * ratio))
        if n_masked:
            chosen = span[torch.randperm(len(span), generator=g)[:n_masked]]
            x[0, chosen] = adapter.mask_id
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


@dataclass
class Comparison:
    """How far a cached forward moved the decision, at three levels of strictness.

    Counts rather than rates, so measurements from several canvases pool by
    addition. One canvas is not enough to state any of these: a rate read off
    32 window positions has a standard error near nine points, and `argmax@k`
    off four has a *resolution* of twenty-five. Two bit widths whose true cost
    is identical will still print far apart, which is how the sweep first
    produced 8 bits scoring below a destroyed cache and 4 bits scoring far
    above it.
    """

    rel_sum: float = 0.0
    n_forward: int = 0
    kept: int = 0
    n_pos: int = 0
    kept_k: int = 0
    n_k: int = 0
    slots_hit: int = 0
    n_slots: int = 0

    def __add__(self, other: "Comparison") -> "Comparison":
        return Comparison(*(a + b for a, b in zip(
            (self.rel_sum, self.n_forward, self.kept, self.n_pos,
             self.kept_k, self.n_k, self.slots_hit, self.n_slots),
            (other.rel_sum, other.n_forward, other.kept, other.n_pos,
             other.kept_k, other.n_k, other.slots_hit, other.n_slots),
        )))

    @staticmethod
    def _rate(hit: int, n: int) -> float:
        return hit / n if n else float("nan")

    @property
    def rel(self) -> float:
        return self._rate(self.rel_sum, self.n_forward)

    @property
    def agree(self) -> float:
        return self._rate(self.kept, self.n_pos)

    @property
    def agree_k(self) -> float:
        return self._rate(self.kept_k, self.n_k)

    @property
    def slots_k(self) -> float:
        return self._rate(self.slots_hit, self.n_slots)

    @property
    def agree_k_se(self) -> float:
        """Binomial standard error on `argmax@k` -- the resolution of the row.

        Printed because a difference smaller than this is not a difference, and
        the whole sweep is a comparison between rows.
        """
        p, n = self.agree_k, self.n_k
        if not n or p != p:
            return float("nan")
        return (p * (1 - p) / n) ** 0.5


def compare(
    reference: torch.Tensor,
    actual: torch.Tensor,
    *,
    committable: torch.Tensor | None = None,
    top_k: int = 4,
) -> Comparison:
    """Compare a cached forward against the exact one.

    `committable` marks the window positions still carrying the mask -- the
    only ones a sampler step can commit. A decoded position is already fixed,
    so its logits are read by nothing and an argmax that flips there is not an
    error, it is noise being counted as one.

    Confidence is taken from the *reference*, because the question is whether
    the cache changes what the exact path would have committed. Ranking by the
    cached run's own confidence would let a quantizer that corrupts the
    ordering pick its own exam questions.
    """
    ref, act = reference.float(), actual.float()
    denom = ref.abs().mean().clamp(min=1e-8)
    rel = float((act - ref).abs().mean() / denom)

    kept = act.argmax(-1) == ref.argmax(-1)             # [B, W]
    out = Comparison(rel_sum=rel, n_forward=1,
                     kept=int(kept.sum()), n_pos=kept.numel())

    conf = ref.softmax(-1).max(-1).values                # [B, W]
    if committable is None:
        committable = torch.ones_like(conf, dtype=torch.bool)
    n_avail = int(committable.sum())
    if n_avail == 0:
        # Nothing in this window is masked, so nothing here would be committed
        # and the restricted numbers have no subject. Happens at mask ratio 0.
        return out

    k = min(max(top_k, 1), n_avail)
    ref_top = conf.masked_fill(~committable, -1.0).flatten().topk(k).indices
    out.kept_k = int(kept.flatten()[ref_top].sum())
    out.n_k = k

    act_conf = act.softmax(-1).max(-1).values
    act_top = act_conf.masked_fill(~committable, -1.0).flatten().topk(k).indices
    out.slots_hit = len(set(ref_top.tolist()) & set(act_top.tolist()))
    out.n_slots = k
    return out


def pct(value: float) -> str:
    return "     n/a" if value != value else f"{100 * value:7.2f}%"


def decision_margin(reference: torch.Tensor, committable: torch.Tensor,
                    top_k: int) -> tuple:
    """How close the about-to-be-committed decisions are to a tie.

    This is the quantity that decides how many bits the cache can afford, and
    it is not a constant of the model -- it collapses as the mask ratio rises.
    Every masked position carries the same embedding row, so their logits
    converge, and a decision taken between two candidates a hair apart flips
    under a perturbation that would be invisible anywhere else.

    Returned in absolute logit units next to the mean logit magnitude, because
    the sweep reports error relatively: a row at `rel` moves logits by roughly
    `rel * mean_abs`, and flips are expected once that approaches the margin.
    """
    ref = reference.float()
    top2 = ref.topk(2, dim=-1).values
    margins = top2[..., 0] - top2[..., 1]                # [B, W]
    mean_abs = float(ref.abs().mean())

    n_avail = int(committable.sum())
    if n_avail == 0:
        return float("nan"), mean_abs
    conf = ref.softmax(-1).max(-1).values
    k = min(max(top_k, 1), n_avail)
    top = conf.masked_fill(~committable, -1.0).flatten().topk(k).indices
    return float(margins.flatten()[top].mean()), mean_abs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-type", default="llada2_moe", choices=["llada2_moe"])
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--device-map", default=None,
                    help="spread the model over several visible cards, e.g. "
                         "'auto'. Use when no single card holds it: with "
                         "CUDA_VISIBLE_DEVICES=3,1 the layers are split and "
                         "the cost is one activation crossing per boundary, "
                         "about 1 MB -- far cheaper than sharing one busy "
                         "card, where neighbours cost a factor of thirty.")
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--blocks", type=int, default=8,
                    help="total sequence length, in blocks")
    ap.add_argument("--bits", type=int, nargs="+", default=[16, 8, 4, 3, 2])
    ap.add_argument("--group-sizes", type=int, nargs="+", default=[32, 128],
                    help="channels sharing one scale, along head_dim. The "
                         "coarsest useful value is head_dim itself -- one "
                         "scale per head per token -- because quantize_kv "
                         "groups along that axis and cannot span heads. For "
                         "this checkpoint head_dim is 128, so 128 is already "
                         "the whole head and anything larger is the same run.")
    ap.add_argument("--mask-ratios", type=float, nargs="+",
                    default=[0.0, 0.5, 1.0],
                    help="mask ratio of the cached *prefix*. 0.0 first because "
                         "it is the only one that occurs in the scheme we are "
                         "measuring: the prefix is the closed blocks, and a "
                         "closed block is fully decoded. The rest answer the "
                         "next question instead -- what it would cost to cache "
                         "the current, partly masked block as well, which is "
                         "where staleness stops being structurally zero. "
                         "Expect the high ratios to be degenerate: with every "
                         "cached position carrying the same embedding row "
                         "there is little information there to lose, and the "
                         "chance floor will say so.")
    ap.add_argument("--window-mask-ratio", type=float, default=1.0,
                    help="mask ratio of the block being decoded. 1.0 is what a "
                         "sampler step actually sees; lowering it leaves fewer "
                         "positions that could be committed, and at 0 there is "
                         "nothing to commit and the restricted metrics go n/a.")
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
    ap.add_argument("--commit-k", type=int, default=4,
                    help="how many positions the sampler commits before the "
                         "cache is rebuilt, i.e. how deep into the confidence "
                         "ordering an error can still reach. Under the "
                         "standard schedule (steps == gen_length) one token is "
                         "committed per step, so this is the first few commits "
                         "of the block. Raise it for a coarser schedule that "
                         "commits several at once.")
    ap.add_argument("--key-axes", nargs="+", default=["channel"],
                    choices=["channel", "token"],
                    help="grouping axis for K in the bits table. Pass both to "
                         "sweep them side by side, which is the only way to "
                         "tell the axis apart from the group size: a fine "
                         "group along the wrong axis partly rescues what the "
                         "right axis fixes outright, so the two leave the same "
                         "trace. Default is the axis every earlier number in "
                         "this project used.")
    ap.add_argument("--value-axis", default="channel",
                    choices=["channel", "token"],
                    help="grouping axis for V. Measured to make almost no "
                         "difference, unlike K's.")
    ap.add_argument("--axis-bits", type=int, default=3,
                    help="bit width for the grouping-axis sweep. Pick one where "
                         "the cache is damaged but not dead, or there is "
                         "nothing for the axis to rescue: at group 32 that is "
                         "3, since 4 costs nothing there.")
    ap.add_argument("--axis-group", type=int, default=32,
                    help="group size for the grouping-axis sweep")
    ap.add_argument("--skip-axis-sweep", action="store_true")
    ap.add_argument("--samples", type=int, default=4,
                    help="independent canvases (different prompt and different "
                         "decoded content) pooled into every row. One is not "
                         "enough: a rate off 32 window positions carries a "
                         "standard error near nine points and argmax@k off "
                         "commit-k positions has a resolution of 100/k, which "
                         "is more than the effects being compared. Cost is "
                         "linear in this.")
    args = ap.parse_args()

    cfg = DLLMQuantConfig(
        model_path=args.model, model_type=args.model_type,
        dtype=args.dtype, device=args.device, device_map=args.device_map,
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
    # One text per sample, each a different stretch of the corpus.
    texts = [text_ids(adapter, total, seed=i) for i in range(args.samples)]

    print(f"\nsequence {total} tokens = {args.blocks} blocks of "
          f"{args.block_length}; prefix cached = {prefix_len}, "
          f"window recomputed = {args.block_length}")
    print(f"{args.samples} canvases pooled per row: "
          f"{args.samples * args.block_length} positions behind `argmax`, "
          f"{args.samples * args.commit_k} behind `argmax@k`")

    states = install_block_cache(model, BlockKVCache(KVCacheConfig(), n_layers))

    canvas_cache: dict = {}

    def canvases(mask_ratio):
        """One canvas per sample: a different text, differently masked.

        Memoized because the K/V sweep asks for the same batch again, and the
        reference is a full forward over the whole sequence -- the most
        expensive thing here.
        """
        if mask_ratio not in canvas_cache:
            batch = []
            for i in range(args.samples):
                x = masked_canvas(adapter, texts[i], args.block_length, prefix_len,
                                  mask_ratio, args.window_mask_ratio,
                                  seed=i).to(device)
                reference = full_logits(model, x, args.block_length)[:, prefix_len:]
                batch.append((x, reference, x[:, prefix_len:total] == adapter.mask_id))
            canvas_cache[mask_ratio] = batch
        return canvas_cache[mask_ratio]

    def sweep(make_cache, batch, scramble: bool = False) -> Comparison:
        """One configuration, pooled over every canvas."""
        pooled = Comparison()
        for x, reference, committable in batch:
            cache = make_cache()
            for state in states:
                state.cache = cache
            refresh_prefix(model, states, x, prefix_len, args.block_length)
            if scramble:
                cache.scramble(torch.Generator().manual_seed(0))
            windowed = logits_for_window(
                model, states, x, prefix_len, total, args.block_length
            )
            pooled = pooled + compare(reference, windowed,
                                      committable=committable,
                                      top_k=args.commit_k)
        return pooled

    def flat_cache(bits, group, key_axis=None, value_axis=None):
        return lambda: BlockKVCache(
            KVCacheConfig(enabled=True, decoded_bits=bits, masked_bits=bits,
                          group_size=group,
                          key_axis=key_axis or args.key_axes[0],
                          value_axis=value_axis or args.value_axis),
            n_layers,
        )

    def row(bits_label, group_label, c, flag=""):
        se = c.agree_k_se
        se_txt = "     " if se != se else f"±{100 * se:3.0f}%"
        print(f"{bits_label:>5} {group_label:>10} {c.rel:>10.3e} "
              f"{pct(c.agree)} {pct(c.agree_k)} {se_txt} {pct(c.slots_k)}{flag}")

    for mask_ratio in args.mask_ratios:
        batch = canvases(mask_ratio)
        print(f"\n--- prefix mask ratio {mask_ratio:.2f} " + "-" * 37)

        margins = [decision_margin(ref, com, args.commit_k)
                   for _, ref, com in batch]
        margin = sum(m for m, _ in margins) / len(margins)
        mean_abs = sum(a for _, a in margins) / len(margins)
        committable = sum(int(c.sum()) for _, _, c in batch) // len(batch)
        print(f"window is {100 * args.window_mask_ratio:.0f}% masked: "
              f"{committable} of {args.block_length} positions could "
              f"be committed")
        print(f"decision margin at those positions: {margin:.3f} logits, mean "
              f"|logit| {mean_abs:.2f} -- a row at rel. err e moves logits by "
              f"about e*{mean_abs:.2f}, and flips once that reaches the margin")
        head = "group" if len(args.key_axes) == 1 else "group/Kaxis"
        print(f"{'bits':>5} {head:>10} {'rel. err':>10} "
              f"{'argmax':>8} {'argmax@k':>8} {'':>5} {'slots@k':>8}")

        # The chance floor first, so every row below is read against it. A
        # 16-bit cache whose entries have been shuffled has exactly the
        # distribution of a perfect one and none of its information, so this is
        # what "the cache told the model nothing" scores. Rows at this level
        # are dead no matter how small their logit error looks.
        chance = sweep(flat_cache(16, head_dim), batch, scramble=True)
        row("--", "scram", chance, "   <-- chance floor: no information")
        if chance.agree > 0.99:
            # The floor met the ceiling: destroying the cache outright changed
            # no decision, so nothing below can be attributed to the bits.
            #
            # Which of two things that means depends on the mask ratio, and the
            # difference matters. High: every cached position carries the same
            # embedding row, so there is no information in the prefix to lose
            # and the row is simply degenerate. Low: the prefix is full of real
            # decoded content, and a window that ignores it is not a fact about
            # diffusion, it is a defect -- and one the 16-bit control cannot
            # catch, because a window that never reads the prefix agrees with a
            # full forward perfectly. So the floor is a second plumbing check.
            if mask_ratio <= 0.5:
                print("      ^ FLOOR AT 100% WITH A DECODED PREFIX: the window "
                      "is not reading the cache at all. The 16-bit control "
                      "cannot see this -- it passes either way. Suspect the "
                      "mask or the prefix length before reading anything else.")
            else:
                print("      ^ the floor is at 100%: destroying the cache "
                      "changes nothing here, so no row below is interpretable. "
                      "Expected this high -- masked positions share an "
                      "embedding row, so there is little in the prefix to lose.")

        groups = sorted({min(g, head_dim) for g in args.group_sizes})
        for key_axis in args.key_axes:
            for group_size in groups:
                for bits in args.bits:
                    c = sweep(flat_cache(bits, group_size, key_axis), batch)

                    size = f"{group_size}=h" if group_size == head_dim else str(group_size)
                    label = size if len(args.key_axes) == 1 else f"{size}/{key_axis[:3]}"

                    flag = ""
                    if bits >= 16:
                        # At 16 bits quantize_kv hands the tensor back
                        # untouched, so this row is not a measurement of
                        # storage -- it *is* the arithmetic noise of the
                        # windowed path, and there is no independent standard
                        # to judge it against. An earlier version compared it
                        # to the rotation check's tolerance, which is a
                        # threshold for a different quantity in a different
                        # place; it fired at 6.6e-02 against a transplanted
                        # 6e-02 while every committed decision was intact.
                        # Raising that constant each time it fires makes the
                        # check unfalsifiable, so the criterion is now at the
                        # level that matters: a decision about to be committed
                        # must not move. Read the number as the table's noise
                        # scale. For a sharp check, --dtype float32.
                        if c.agree_k < 1.0:
                            flag = "   <-- a committed decision moved: the path is wrong"
                        else:
                            flag = "   <-- control: windowing noise, storage is a no-op"
                    elif c.rel >= chance.rel and c.agree <= chance.agree:
                        flag = "   <-- at the floor: carries nothing"
                    row(str(bits), label, c, flag)

    print("\nThe 16-bit row is the control: quantize_kv returns the tensor "
          "untouched there, so it is not a measurement of storage at all -- it "
          "is the arithmetic noise of the windowed path, and it sets the scale "
          "the rest of the table is read on. The windowed forward multiplies a "
          f"{args.block_length}-row query against the keys where the full one "
          "multiplies every row, and a different accumulation order is a "
          f"different answer in {args.dtype}. It is flagged only when a "
          "committed decision moves, because that is the failure that has "
          "consequences and any threshold on the error itself would be a "
          "number carried in from somewhere else. For a sharp control, rerun "
          "with --dtype float32.\n\n"
          f"argmax counts all {args.block_length} window positions alike and is "
          "therefore an upper bound on the damage: most of those positions stay "
          "masked this step and get decided again later, so a flip there costs "
          f"nothing. argmax@k restricts it to the {args.commit_k} positions the "
          "exact path was most confident about -- the ones about to be committed "
          "-- and is the number that predicts accuracy. slots@k asks the other "
          "half: whether quantization changed *which* positions those are. Both "
          "matter, and they fail differently -- argmax@k at 100% with slots@k "
          "below it means the tokens are right but the schedule moved.\n\n"
          "Read every row against the scrambled one at the top of its table. "
          "That row is a cache with a perfect distribution and no information, "
          "so it is where these numbers bottom out -- and they do not bottom "
          "out at zero, because two runs of one model agree by prior alone. A "
          "bit width at the floor is carrying nothing, and the ordering among "
          "rows at the floor is noise, not a ranking.")

    if not args.skip_axis_sweep:
        run_axis_sweep(args, n_layers, head_dim, canvases, sweep, flat_cache, row)

    if args.kv_pairs:
        run_kv_asymmetry(
            args, n_layers, head_dim, canvases, sweep, flat_cache, row
        )
    return 0


def run_axis_sweep(args, n_layers, head_dim, canvases, sweep, flat_cache, row) -> None:
    """Which direction should a group run in, for K and for V separately?

    A group must not straddle an outlier, because the outlier sets the scale
    and everyone sharing it pays. So the right axis is the one that runs
    *along* wherever the outliers live, and that is a property of the tensor,
    not a convention: KIVI found K's outliers sit in fixed channels and V's do
    not, and therefore quantized K along tokens and V along channels.

    Everything measured in this project used the channel axis for both, which
    is half wrong if KIVI's finding carries over from autoregressive models. It
    may not: a diffusion LM's attention sinks move during generation, so the
    channels K's outliers occupy need not stay put either.

    All four cells cost exactly the same memory -- the axis changes what a
    scale is shared across, not how many there are -- so this is a free choice
    and the only question is which way is better.
    """
    print(f"\n=== grouping axis, {args.axis_bits} bits, group {args.axis_group} "
          + "=" * 24)
    print("channel = a group runs along head_dim (KIVI calls this per-token)")
    print("token   = a group runs along tokens   (KIVI calls this per-channel)")
    print("KIVI's answer for autoregressive models: K token, V channel")
    print("all four cost identical memory")

    for mask_ratio in args.mask_ratios:
        batch = canvases(mask_ratio)
        print(f"\n--- prefix mask ratio {mask_ratio:.2f} " + "-" * 37)
        print(f"{'K axis':>5} {'V axis':>10} {'rel. err':>10} "
              f"{'argmax':>8} {'argmax@k':>8} {'':>5} {'slots@k':>8}")

        chance = sweep(flat_cache(16, head_dim), batch, scramble=True)
        row("--", "scram", chance, "   <-- chance floor: no information")

        best = None
        for k_axis in ("channel", "token"):
            for v_axis in ("channel", "token"):
                def make_cache(ka=k_axis, va=v_axis):
                    return BlockKVCache(
                        KVCacheConfig(
                            enabled=True, decoded_bits=args.axis_bits,
                            masked_bits=args.axis_bits,
                            group_size=args.axis_group,
                            key_axis=ka, value_axis=va,
                        ),
                        n_layers,
                    )
                c = sweep(make_cache, batch)
                note = ""
                if (k_axis, v_axis) == ("channel", "channel"):
                    note = "   <-- what every earlier number used"
                elif (k_axis, v_axis) == ("token", "channel"):
                    note = "   <-- KIVI's answer"
                row(k_axis, v_axis, c, note)
                if best is None or c.agree_k > best[0]:
                    best = (c.agree_k, k_axis, v_axis)

        if best:
            print(f"      best here: K {best[1]}, V {best[2]}")


def run_kv_asymmetry(
    args, n_layers, head_dim, canvases, sweep, flat_cache, row
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
    # The axis is not optional here. Grouping K along channels leaves its
    # outliers straddling everyone else's scale, which costs far more than any
    # bit reallocation -- so a K/V sweep on the wrong axis measures the axis
    # and calls it asymmetry. The first --key-axes value is used, and printed,
    # because an earlier version silently defaulted and produced a table
    # indistinguishable from the channel-axis one.
    key_axis, value_axis = args.key_axes[0], args.value_axis

    print(f"\n=== K/V asymmetry, group {group}, K along {key_axis}s, "
          f"V along {value_axis}s " + "=" * 8)
    print("mirror pairs cost the same memory; the winner says where bits belong")
    print("a gap smaller than the +- on argmax@k is not a result")

    for mask_ratio in args.mask_ratios:
        batch = canvases(mask_ratio)
        print(f"\n--- prefix mask ratio {mask_ratio:.2f} " + "-" * 37)
        print(f"{'K':>5} {'V':>10} {'rel. err':>10} "
              f"{'argmax':>8} {'argmax@k':>8} {'':>5} {'slots@k':>8}")

        chance = sweep(flat_cache(16, head_dim), batch, scramble=True)
        row("--", "scram", chance, "   <-- chance floor: no information")

        for k_bits, v_bits in pairs:
            def make_cache(k=k_bits, v=v_bits):
                return BlockKVCache(
                    KVCacheConfig(enabled=True, group_size=group,
                                  key_bits=k, value_bits=v,
                                  key_axis=key_axis, value_axis=value_axis),
                    n_layers,
                )
            c = sweep(make_cache, batch)
            flag = ("   <-- at the floor"
                    if c.rel >= chance.rel and c.agree <= chance.agree else "")
            row(str(k_bits), str(v_bits), c, flag)


if __name__ == "__main__":
    raise SystemExit(main())
