# Quantizing the cache of a diffusion LM

Working plan. Companion to HANDOFF.md, which records state; this records
intent and the reasoning behind the order of work.

---

## 1. The gap, as it actually stands

An earlier version of this section claimed no published diffusion-LM cache is
quantized. That was checked and it is false. Four framings that this project
leaned on are taken:

| framing | taken by |
|---|---|
| "nobody quantizes a dLLM cache" | **DART** (arXiv 2601.20706, Jan 2026) — W4A8KV4, MXINT4 cache, LLaDA-8B and LLaDA-MoE-7B-A1B |
| "K and V are not alike" | **KIVI** (2024) — K per-channel, V per-token, in autoregressive models |
| "a block-causal prefix is exactly reusable" | **LaCache** (arXiv 2607.16339) — lossless state memoization for dLLMs |
| "4-bit weights cost a dLLM almost nothing" | **Layer Collapse** (arXiv 2605.06366) — 3-bit GPTQ on LLaDA costs 1.8% GSM8K against 64.7% on Llama-3.1-8B |

So the territory is occupied and the claim has to move: not *first to compress
a dLLM cache*, but **what decides the tolerable precision of that cache, and
where the bits should go**. DART is a hardware paper — it showed a 4-bit cache
runs, because it needed one to build an NPU around. It did not ask why it runs,
where it stops running, or how to spend a bit budget. That question is open,
and three parts of it are unclaimed by anything found so far.

**The router.** In a MoE the expert choice is a top-k over 256 candidates —
discrete, with nothing to average. Measured here: the 19 routers are 0.061% of
the parameters, and at four bits they change 22% of expert selections, while
four-bit weights on the experts themselves cost nothing detectable (McNemar
p = 0.44). DART evaluates on a MoE and does not analyse the router at all.

**Attention sinks that move.** Sinks are the known obstacle to KV-cache
quantization: one position with an enormous key magnitude sets the scale for
its whole group, and KVQuant builds dedicated outlier machinery around it. That
machinery assumes the sink *stays put*. In a diffusion LM it does not — sinks
shift during generation (arXiv 2510.15731). If so, a statically calibrated
outlier set cannot transfer, and an entry written while its position was a sink
keeps that scale after the sink has left. Nobody has connected the two lines.

**Low-rank correction on the cache.** `K ≈ Q₃(K) + A·B`. Well worked out for
weights (LQER, CALDERA, ZeroQuant-V2); for a diffusion LM's cache, done by
nobody, DART included (checked: "low-rank decomposition, residual correction —
not discussed"). Evidence that there is structure to catch: refusal behaviour
in an MDLM lives in a roughly one-dimensional activation subspace (arXiv
2512.24143).

Underneath all three sits the two-error structure that made this project worth
starting, and it survives intact:

* **drift** — the entry was computed some steps ago and the state has moved;
* **rounding** — the entry is stored in four bits.

Refreshing more often cuts drift and costs compute; spending more bits cuts
rounding and costs memory. Whether one hides the other is still unstudied. The
autoregressive analogue does not answer it: there the cache is exact by
construction, so only rounding is at stake.

LLaDA2.0 makes the two separable, which is the reason to work on it rather than
LLaDA-8B. Its block-causal prefix is exact, so drift is structurally zero there
and a change in output is attributable to the bits alone. DART's models are
fully bidirectional, where the cache is approximate from the moment it is
written and the two errors cannot be told apart.

---

## 2. What is already measured

Numbers below are on real weights. Everything else in this document is intent.

### LLaDA-1.5 (dense, 8B) — the premise

| configuration | GSM8K |
|---|---|
| FP16 | 84.5 |
| W4A16 | 83.0 |
| W4A4, activations per-token | 26.0 |
| W4A4, activations group-128 | 78.0 |

Rotation lifts teacher-forced fidelity 42.5 → 88.9 and drops KL 2.64 → 0.132.
Activation crest factor is 39.1 at full mask against 6.9 at none.

Two methodological findings that constrain everything after them:

* **Teacher-forced fidelity misleads.** The 26.0 and 78.0 configurations —
  52 points apart on the task — differ by 2.3 points of fidelity, because the
  reference states come from the FP16 trajectory and the quantized model never
  walks its own path. `dllmquant/eval/trajectory.py` exists for this reason.
* **Activation granularity dominates everything else measured so far.**
  52 points from one knob.

### LLaDA2.0-mini (MoE, 16B-A1B) — this project's target

Adapter validated against the checkpoint (8 structural checks). Rotation
verified exact in float32: 2.17e-06 against a 1e-04 floor, routing unchanged.

| configuration | GSM8K, our protocol |
|---|---|
| FP16 | 77.0 |
| W4A16 | 74.5 |

Paired over the same 200 problems: 138 both right, 35 both wrong, 16 only
FP16, 11 only W4A16. Exact McNemar p = 0.44 — **4-bit weights cost nothing
measurable**, as on LLaDA-1.5.

Published GSM8K for this checkpoint is 94.24. Our 77.0 is not a failed
reproduction, it is a different protocol: zero-shot, a 256-token canvas, and
LLaDA-1.5's fixed-schedule sampler rather than the model's own
threshold-with-early-stop one. Comparisons *within* our protocol are valid;
the published number does not belong in the same table.

**Truncation confounds the base.** At a 256-token canvas, unambiguous
mid-expression cutoffs account for 20% of FP16 errors and 37% of W4A16
errors — a difference of 10 answers, larger than the entire 5-answer accuracy
gap. The measured "quantization damage" was mostly the generation budget.
`--gen-length` is now recorded in every result file and the cut-off share is
reported next to accuracy.

**Expert starvation is real and central.** 256 experts, top-8: from a single
prompt the median expert receives nothing and the busiest takes an eighth of
the layer's routes. Most expert projections therefore reach CGQ with no
calibration data at all and fall back to plain rounding. Solving that is a
prerequisite for any claim about *weight* quantization on this model.

It is not, however, orthogonal to the cache, as an earlier draft of this
document said. The deployment-regime cache measurements run on top of a
quantized rotated model, so how well that model was quantized sets the floor
under them -- and R3, which makes V quantizable in the first place, is part of
the same pipeline. Starvation therefore bounds the quality of the substrate
the cache study stands on, even though it has nothing to do with the cache's
own error.

### A hypothesis this project raised and then refuted

Under rotation in bf16, routing changes. Measured with a positional
comparison, the damage looked dramatic and strongly ordered by mask ratio
(51% of slots kept at full mask, 74–79% elsewhere), suggesting that a
degenerate all-`[MASK]` input makes expert selection maximally fragile.

That metric was wrong: replacing one expert shifts the others in the sorted
tuple, and a positional comparison counts one swap as several. With set
overlap the picture is flat — 87–91% kept, minimum at mask ratio 0.75, not
1.00.

**So: routing is perturbed by rotation, by roughly a tenth of slots, and the
mask ratio does not order it.** The degeneracy hypothesis is not supported by
this measurement and should not be carried forward as an assumption. It may
still hold for larger perturbations (weight quantization is one); that is a
question, not a finding.

---

## 3. Axes

Six, and the last one is transverse to the rest.

**What is cached.** K, V, or a block's output latents. K and V are currently
treated identically, which is unlikely to be right: K enters the softmax,
where error passes through an exponential and reshapes the attention
distribution; V is summed linearly against attention weights already fixed.
The autoregressive literature generally finds K needs more bits. Untested
here, and cheap to test.

**Which positions.** Block-causal attention splits the cache in two with
completely different error characters:

* the **prefix** (blocks already closed) is *exactly* reusable — a position in
  an earlier block attends only to blocks that are themselves closed, so its
  K/V are constant for the whole of the current block's decoding. Drift is
  identically zero and rounding is the only error;
* the **current block** is not — its tokens are revealed between steps, so
  reuse trades staleness for compute.

The prefix is therefore the clean instrument for measuring quantization cost
in isolation, and the current block is where the interesting trade lives.

A second split cuts across it: a masked position's K/V will be overwritten as
soon as its token is committed, while a decoded position's is final. Spending
equal precision on both spends it on something about to be discarded.
`KVCacheConfig` already exposes `decoded_bits` / `masked_bits`; never tested.

**Format.** Bits, group size, symmetric vs asymmetric, clipping. Group size
along `head_dim`, which is 128 on this model — so group 128 is one scale per
head per token, the coarsest meaningful setting, and the ladder is 128 / 64 /
32. This is the cache analogue of the knob that moved 52 points on
activations.

**Refresh policy.** `never` / `every_n` / `block` / `mask_ratio`. The adaptive
one uses a signal that is free at every step: everything moves early in the
trajectory and almost nothing moves late, so the interval can follow the mask
ratio. Written, never measured.

**Where the scales come from.** Currently nowhere: `quantize_kv` computes
amax/amin per group at write time, so the cache is dynamically quantized and
never calibrated. That is the right default -- the distribution moves along
the trajectory, and on LLaDA-1.5 the activation crest factor is 39.1 at full
mask against 6.9 at none, so one static scale fitted at one end is badly wrong
at the other.

But dynamic scales *are part of the cache*: one scale and one zero point per
group per token per head, which is the 6.25% overhead at group 128 and 25% at
group 32. For a structure whose entire purpose is to save memory, that is not
a rounding error -- it eats a sixth of what four-bit storage buys.

Static scales cost nothing per token but need calibration and break under
distribution shift. And here diffusion offers something an autoregressive
model cannot: **the mask ratio is known for free at every step**, so scales
can be calibrated per mask-ratio bucket -- the same four buckets TMAS already
sorts snapshots into -- and selected by the current state. Static storage
cost, distribution-following precision. Whether the shift within a bucket is
small enough for this to work is exactly the sort of question this project is
positioned to answer, and nothing in the autoregressive literature asks it
because there is no such signal there.

Three settings to compare: dynamic per group (what exists), one static set,
and static per mask-ratio bucket -- at equal *total* bits including the
scales, which is the only comparison that means anything.

**Correction.** Quantized matrix plus a low-rank approximation of the
residual, `W ≈ Q(W) + AB^T`. Not implemented.

**Mask ratio.** Transverse to all of the above. At a high mask ratio nearly
every position carries the same embedding row, so their K/V are near-identical
and round the same way — errors that would partly cancel if independent add
coherently instead. The usual "quantization noise averages over positions"
argument fails, and how badly is measurable.

---

## 4. Metrics, and why the obvious ones mislead

**Not relative logit error alone.** A logit that moves 3% changes nothing; a
logit ordering that flips changes the token that gets committed, and in a
diffusion LM commitment is irreversible. **Argmax agreement is the primary
number**, error is secondary.

**And not flat argmax agreement either.** Averaged over every position in the
window it weights them all alike, and the sampler does not: it commits the most
confident and leaves the rest masked to be decided again from scratch on a
later step. A flip on a position nobody was going to commit costs nothing, so
the flat share is an upper bound on the damage — and a loose one exactly where
the interesting regime is. At a high mask ratio the positions carry the same
`[MASK]` embedding row, their logits sit near a tie, and the flat share drops
on its own without any decision changing.

So `check_block_cache.py` reports two restricted numbers beside it, both taken
over the `--commit-k` positions the *reference* path was most confident about
(ranking by the quantized run's own confidence would let it choose its own exam
questions):

* `argmax@k` — do the positions about to be committed still take the same
  token. This is the number that predicts accuracy.
* `slots@k` — are they still the same positions. Quantization can leave every
  token right and still reorder which one is unmasked first, which changes the
  conditioning of everything after it.

The two fail independently, and a claim about a four-bit cache needs both.

**Not teacher-forced fidelity.** Measured to compress a 52-point task gap into
2.3 points, for a structural reason: forcing reference states prevents the
quantized model from walking its own trajectory, which is exactly where the
damage compounds. Use `trajectory.py`: `exact_match`, `final_agreement`,
`first_divergence_step`, `divergence_mask_ratio`.

**Drift decomposition.** `measure_drift` separates a cached tensor's total
error into staleness and rounding by quantizing a freshly computed tensor
separately. Without that separation there is nothing to tune the refresh
interval against — the whole point of the study.

**Routing agreement, for MoE.** Fraction of each token's expert set that
survives a perturbation, compared as a set. Already instrumented
(`routing_fingerprint`, `routing_overlap`) and already used to establish that
an exactly-invariant rotation still moves routing in bf16.

**End-task accuracy last**, on configurations that the cheaper metrics have
already selected — it costs hours and answers only one question.

---

## 5. Three tiers, by cost

**Tier 0 — one forward, no generation.** `scripts/check_block_cache.py`.
Bits × group × mask ratio on the prefix cache, with a 16-bit control row that
compares the windowed path against a full forward and must read ~0. Minutes on
a GPU, tens of minutes on CPU. Answers: what does four-bit storage cost, where
does argmax break, does K differ from V.

**Tier 1 — generation, no scoring.** Cache in the sampler. Trajectory
divergence, drift decomposition, routing stability across the trajectory, and
the refresh-policy sweep. Answers the actual research question: how drift and
rounding interact, and whether the adaptive policy beats a fixed interval.

**Tier 2 — end task.** GSM8K on the handful of configurations Tier 1 selects,
in **two regimes, both required**:

* *FP16 weights, quantized cache* — the isolating experiment. Everything that
  moved is the cache, so the cause is not in doubt.
* *quantized rotated weights, quantized cache* — the deployment experiment,
  and the one the claim is actually about. A four-bit cache beside sixteen-bit
  weights saves nothing worth reporting: the weights are the memory. And the
  two errors may add, or one may absorb the other, which is the same
  two-error question this plan opens with, asked one level up.

The second regime is not optional for a further reason that is easy to miss.
**R3 rotates the value subspace specifically to make V quantizable** -- V is
the tensor the cache stores. So how quantizable the cache is depends on
whether the model was rotated, and rotation arrives as part of the weight
pipeline. A cache measured on an unrotated FP16 model is a cache measured in a
configuration nobody would deploy.

Practically this means the long weight-quantization run is not a side quest:
`quantize.py --save` writes the rotated, quantized model once, and every
deployment-regime cache experiment then runs on top of it without paying for
the solve again.

---

## 6. Order of work

Four phases. The first is cheap and finishes what is half-measured; the second
and third are the two unclaimed results and need code that does not exist; the
fourth converts everything into task numbers. Weights stay FP16 throughout
except in phase D — that is the field's convention (KIVI, KVQuant, GEAR,
ZipCache, SKVQ all quantize the cache alone) and the only way to attribute a
shift to the cache.

### Phase A — close the open measurements (CPU, days)

Nothing here needs new ideas, and two items can change what phases B and C are
built on.

1. **K grouping axis.** `quantize_kv` groups K and V both along `head_dim`.
   KIVI established that K wants grouping along the token axis within a
   channel, because its outliers sit in channels. Untested here. The suspicion
   is that the measured "group 32 beats 128" is a weak shadow of it — a fine
   group along the wrong axis partly rescuing what the right axis would fix
   outright. **Before** low-rank, or low-rank is built on a base known to be
   wrong.
2. **`check_block_cache --samples 24`**, including the K/V pairs at group 32
   around three bits, which is the operating point there. Present bars are ±8
   against differences of 9–16 points.
3. **`trace_window_path`.** The 16-bit control sits at 90.6%, not 100%: window
   reuse alone moves three committed decisions in thirty-two, at a perturbation
   nine times below the decision margin. Smooth growth with depth is bfloat16;
   a jump at one layer is the router. If it is the router, this is a result in
   its own right — cache reuse in a MoE dLLM is not free at *any* width.
4. **`check_expert_weights`.** Why certainty weighting reached no expert layer.
5. **`measure_drift` on LLaDA-1.5** (needs `transformers==4.46.3` in a shadow
   directory) and **along a trajectory** rather than one snapshot, which is
   what puts numbers on the age and mask-ratio axes.

### Phase A' — the three axes block diffusion adds, and this plan missed

Every cache number so far is one operating point: the last block of eight, a
block length of 32, one sampler step, a prefix that is itself exact. The plan
was built around block-causal attention and then failed to sweep the things
block-causal attention introduces.

5a. **Block length.** Fixed at 32 throughout and never questioned. It pulls the
    cache in two directions at once: a longer block closes more positions per
    refresh, and lets staleness accumulate longer inside the block before that
    refresh comes. The bit optimum almost certainly depends on it. `--block-length`
    already exists; nothing has ever varied it.

5b. **Where the window sits.** Always the last block, always a 224-token
    prefix. In a real generation the prefix grows from 32 to 480, and the cost
    of quantizing it should grow with it -- more cached positions, more
    rounding accumulating inside one attention sum. One point was measured and
    silently treated as representative.

5c. **Compounding across blocks.** Attention across blocks is causal, so a
    quantized prefix reaches every later block, and each later block writes
    K/V that were computed from it. The table measures **one** window against a
    clean prefix. A real trajectory is sixteen blocks in sequence.

The third is the serious one: four bits can be free for one step and not free
for sixteen. Nothing in the analytic table can see that, and the only thing
that would is the end-task evaluation in phase D -- which leaves a gap between
the mechanism and the number, exactly where a reviewer will look.

It is also the same work as "current-block reuse" in phase C: both need the
cache wired into the sampler and measured along a trajectory rather than on a
snapshot. `cached_generate` exists and `measure_drift` already writes per
layer; what is missing is the loop that joins them.

Note which findings this does *not* touch. Prefix exactness is structural --
a closed block cannot change under block-causal attention, at any block length
or position -- so `stale(prefix)` = 0 holds everywhere by construction. The
grouping axis is a property of where K's outliers live, not of how long the
prefix is. What genuinely needs the sweep is "four bits are free", because
that one is about accumulated rounding and accumulation is exactly what
changes with length and with block count.

### Phase B — the router (new code, the strongest unclaimed result)

`check_router` perturbs *weights* and watches routing. The cache question is
the other direction: does a quantized **cache** change the expert choice, and
does a changed choice change the output?

6. Routing agreement as a function of **cache** bits, per layer, at several
   mask ratios. The instrument is `route_overlap` in `trace_window_path` —
   set comparison, not positional; the positional version inverted a result
   here once already.
7. **Is a flipped route absorbed?** Eight experts of 256 are summed with
   router weights; swapping the eighth-ranked one may cost nothing while
   swapping the first costs everything. Measure the output change against the
   rank of the flipped route. This decides whether "22% of routes change" is
   alarming or cosmetic, and nobody has asked.
8. **Per-layer refresh from router fragility.** If the routers of some layers
   are far more brittle, a uniform refresh interval is the wrong policy, and
   the mask ratio is known for free at every step.

### Phase C — sinks and low-rank (new code)

9. **Track sinks across the trajectory.** Which positions absorb
   disproportionate attention mass, per step and per layer; how large their
   keys are against the rest; whether they alone set their group's scale.
10. **Static versus dynamic outlier sets.** Calibrate a sink set once, KVQuant
    style, and measure how fast it decays as generation proceeds, against a set
    recomputed every step. A clean negative — the autoregressive machinery does
    not transfer — is as publishable as a positive, and follows from sinks
    moving.
11. **Low-rank correction**, `K ≈ Q₃(K) + A·B`. The hypothesis is specific, not
    exploratory: keep V at three bits flat and give the whole rank budget to K,
    because K is what the softmax exponentiates. Mandatory control — the same
    bits spent on a finer group, or simply spent flat.

    *The arithmetic this item was justified by was wrong.* A 224 × 128 prefix
    is 28672 entries and a rank-8 correction is 2816 numbers, so "10% overhead,
    3 bits + rank 8 ≈ 3.3 bits effective" — but that is 10% of the *count*, and
    the entries are three bits while the numbers are fp16. In bits the
    correction costs 1.57 per entry: 4.86 total against 4.29 for four bits
    flat. It reaches 3.3 only with four-bit factors, which have an error of
    their own, so factor precision is a swept axis in `check_lowrank.py` rather
    than a constant. On synthetic channels with this model's measured outlier
    gains the residual is genuinely low-rank — per-channel scales make the wide
    channels round coarsest — and the correction still loses to one flat bit.
    Which is what the survey in that script is for: it settles this on real
    tensors at the cost of one forward per canvas.
12. **Static scales bucketed by mask ratio.** Scales are dynamic now, as in
    KIVI. The diffusion move is to calibrate per mask-ratio bucket and select
    by the current one. External justification: attributes commit on distinct
    schedules — topic within the first 2% of denoising, sentiment over 20%
    (arXiv 2605.10971) — so an error early is unrecoverable and an error late
    is not. Compare at equal **total** bits, scales included.

    *Built, not yet run:* `check_static_scales.py`. It also carries the control
    that makes a bad result interpretable — one scale per channel taken from
    the canvas being stored, which is the same arithmetic as a perfectly
    calibrated static scale and therefore the ceiling. Without it, "granularity
    cost this much" and "staleness of the scales cost this much" arrive as one
    number.

### Phase D — task numbers (GPU, hours)

13. **`--kv-cache --kv-bits 16` against the 91.50% FP16 baseline.** Answers
    whether those three decisions in thirty-two matter downstream, and measures
    the speed-up, which sets the price of everything below.
14. **Four cells**: {FP16 weights, W4A4} × {FP16 cache, 4-bit cache}. The top
    row is the isolating experiment, the bottom row answers the reviewer who
    objects that nobody deploys a 4-bit cache beside FP16 weights.
15. **The same cache tables on the rotated model.** Every cache number so far
    was taken without rotation, and R3 exists precisely to make V fit into few
    bits.

### What each phase is worth

Phase A alone leaves a solid but narrow paper: bit and group sweeps with
decision-level metrics, on a model where drift and rounding separate. Phase B
adds the result that is hardest to dismiss, because DART works on a MoE and
never looks at the router. Phase C is the one that reads as a method rather
than a study. Phase D is what makes any of it citable as accuracy.

If time runs out, cut from the bottom: D can shrink to two cells, C to the
low-rank experiment alone, and B not at all.

---

## 7. What could invalidate this

**Our sampler is not the model's.** We run LLaDA-1.5's fixed schedule on both
models so the two are comparable; LLaDA2.0's own sampler commits by confidence
threshold and stops at EOS. That choice depresses our base (77.0 against a
published 94.24) and a depressed base masks damage. Before the final numbers,
either adopt the model's sampler for the baseline or state the protocol
explicitly and argue that relative comparisons survive it.

**A shared node.** Solve time per layer has varied 30× with neighbours'
load — 0.3 s against 9–17 s for the same work. Wall-clock estimates are
unreliable and checkpointing is not optional.

**MoE calibration starvation.** Weight-quantization claims on this model are
provisional until the median expert receives data. This does not block the
cache study — the cache is an activation, not a weight — but it does block
any claim of the form "CGQ helps here".

---

## 8. Inventory

**Built and tested:** `BlockKVCache` with the four refresh policies,
`quantize_kv`, `measure_drift`, the cache-aware attention path and windowed
forward (`llada2_local.py`), the block-causal mask builder,
`check_block_cache.py`, `routing_fingerprint` / `routing_overlap`,
`trajectory.py`, the vendored and hash-pinned model code.

**Tested only against stand-ins:** everything in `llada2_local.py` — the local
transformers is too old to import the vendored module. Tier 0 discharges this.

**Not built:** cache in the sampler; separate K/V formats; low-rank
correction; any evaluation path that turns the cache on.

**Measured:** nothing about the cache. That is the whole of the work ahead.
