# Quantizing the cache of a diffusion LM

Working plan. Companion to HANDOFF.md, which records state; this records
intent and the reasoning behind the order of work.

---

## 1. The gap

Block-wise diffusion decoders reuse computation across denoising steps and
accept that the reused values are stale: Fast-dLLM and its successors cache
K/V, later work caches intermediate latents, and all of it is governed by some
refresh policy that trades staleness for compute. The literature on that
trade is by now substantial.

**None of it is quantized.** Every published cache for a diffusion LM is kept
in the model's native precision.

That leaves a specific, unclaimed question. A quantized reused cache carries
*two* errors at once:

* **drift** — the entry was computed some steps ago and the state has moved;
* **rounding** — the entry is stored in four bits.

Both are controlled by overlapping knobs, and they are not obviously additive.
Refreshing more often cuts drift and costs compute; spending more bits cuts
rounding and costs memory. Whether one hides the other, whether the optimum is
"refresh rarely and store precisely" or "refresh often and store coarsely", is
unstudied for any diffusion LM.

The autoregressive analogue does not answer it. There the cache is *exact* by
construction — past tokens never change — so drift does not exist and only
rounding is at stake. The interaction is specific to diffusion.

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

1. **Tier 0 on the prefix.** Nothing to build; the exactness control also
   discharges the outstanding risk that our windowed path differs from the
   model's own (verified so far only against stand-ins).
2. **K and V separately.** A small change to `quantize_kv`'s call sites. Cheap,
   and an asymmetric answer would be worth reporting on its own.
3. **Cache into the sampler.** The one substantial piece of engineering left.
   Requires a branch in `_denoise` and flags in `evaluate.py`. Unlocks Tier 1
   and everything after it.
4. **Refresh policies and drift decomposition.** The core result.
5. **Masked vs decoded precision.** Diffusion-specific, cheap once (3) exists.
6. **Where the scales come from.** Dynamic (what exists) against one static
   set against static per mask-ratio bucket, compared at equal total bits with
   the scales counted -- they are a sixth of the budget at group 128. The
   bucketed variant is the one idea here that an autoregressive model cannot
   copy, because it needs a signal that says which regime the distribution is
   in, and the mask ratio is exactly that signal, available for free.
7. **Low-rank correction.** Last, and not as a headline. It is most defensible
   where data-driven compensation is *unavailable* — the starved experts,
   which have no calibration data and therefore get no CGQ help, while an SVD
   of the residual needs no data at all. It must be reported against the
   control that most papers in this area omit: **the same bits spent on a
   finer group instead**. Rank 8 on a 2048×512 expert projection costs about
   8% over 4-bit storage, i.e. roughly 4.3 bits; if group-64 at 4 bits does
   better for the same budget, the correction is not earning its complexity.

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
