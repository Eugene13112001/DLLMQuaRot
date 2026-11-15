# Step tax and rotation-based quantization in dLLMs (2024 – Nov 2026)

Verification legend: [TEXT] = read from the paper's full text (via WebFetch of arXiv HTML / alphaXiv); [ABS] = abstract page only; [SEC] = only from a search-engine summary / secondary source, not opened. arXiv HTML pages timed out repeatedly during this session, so several items are [ABS]/[SEC] only.

## Q1. Which papers report step-count / tokens-per-step changes under quantization in dLLMs?

### Takeaway
I could NOT find a paper named DECAL, nor any paper reporting "W4A4 raises steps by up to ~13% on LLaDA/Dream" or a "commit temperature". Five separate searches (DECAL + quantization, DECAL + step tax, "commit temperature" + dLLM, etc.) returned nothing. The DECAL claim is unverified and may be misremembered or mis-attributed. Treat it as unconfirmed. Every quantized-dLLM paper I opened runs with a FIXED step count (or a fixed commit policy) and does not measure how quantization changes steps or NFE. The "step tax" appears to be unmeasured in the literature I could reach. The one possible exception, the Oct 2026 KV-cache paper that uses threshold-0.9 decoding, could not be confirmed either way.

### Cited Findings
- DECAL: no search hit under that name. The closest name match is **DecoCal: "Decoding with Calibration in Diffusion Large Language Models"** (ACL 2026 long, 2026.acl-long.545). The search summary says it calibrates token-level confidence across diffusion steps and uses it to guide decoding. Nothing seen ties it to quantization, and the page timed out [SEC] — [ACL Anthology](https://aclanthology.org/2026.acl-long.545/). Possibly the source of a "DECAL" mix-up, but that is speculation.
- **FAIR-Calib** (Huang, Yang, Xu, Liu, Guo, Fu, Zhou, Zhang; arXiv 2606.06547, v2 19 Jul 2026, listed as ICML 2026):
  - Setup: W4A4 on LLaDA and Dream, with W3A4 and W8A8 in the appendix; no KV quantization.
  - Steps: the paper keeps decoding fixed, "same number of diffusion steps and commit policy", and reports NO change in steps or tokens/step. LLaDA uses a confidence-driven commit policy and Dream an entropy-driven one. No commit temperature [TEXT] — [alphaXiv](https://www.alphaxiv.org/abs/2606.06547.md).
- **T²Quant**, "Different Tasks, Different Timesteps" (Avivi, Rahimi, Mendelson, Baskin; arXiv 2610.31198, 22 Oct 2026):
  - Setup: fixed 128 denoising steps; the paper does not study the effect of quantization on steps or NFE.
  - Tokens-per-step ablation, LLaDA-8B-Instruct GSM8K, W4A4: the schedule keeps a ~5-point gain from 4 down to 2 tokens/step. At 8 tokens/step every row collapses (FP16 31.6) and the gain disappears.
  - Uniform W4A4 scores 44.1, T²Quant 49.4, FP16 52.9.
  - The reference list contains no paper on quantization increasing steps (no DECAL) [TEXT] — [arXiv](https://arxiv.org/html/2610.31198).
- **STaR-Quant** (arXiv 2606.04945): reports 1.65× / 1.64× / 1.69× speedup on LLaDA / LLaDA-1.5 / Dream, with averages 58.99→58.25 (LLaDA) and 66.94→65.35 (Dream). The search summary notes that fewer decoded tokens per step raise compute cost. Whether it measures quantization-induced step changes is unknown (page timed out) [SEC] — [arXiv](https://arxiv.org/html/2606.04945v1).
- **KV Cache Quantization for Diffusion Language Models** (Tai, Ding, Li, Luo, Zhang, Zhai, Guo, Liu; arXiv 2610.39853, 29 Oct 2026):
  - Setup [ABS]: adapts KIVI/KVQuant to Fast-dLLM prefix and dual cache, 4-bit and 2-bit, on LLaDA and Dream (GSM8K, HumanEval, MATH, MBPP).
  - The search snippet says it evaluates with confidence-threshold parallel decoding at threshold 0.9 [SEC].
  - Whether it reports steps or NFE is unknown; the full text timed out — [arXiv abs](https://www.arxiv.org/abs/2610.39853).
- **SoloQ** (Ghosh, Manjunath, Rhee, Kook, Xiao, Kim, Panda; arXiv 2610.07121, 7 Oct 2026):
  - Steps are fixed per benchmark, e.g. LLaDA-8B-Base gen/steps/block 1024/1024/1024; Dream 256 steps on GSM8K.
  - Nemotron-Labs-Diffusion runs at threshold 0.9 with block 32, but no step change under quantization was reported in the part I read (first 100k of 126k characters) [TEXT, partial] — [HF papers](https://huggingface.co/papers/2610.07121).
- DLLMQuant (2508.14090) and Quant-dLLM (2510.03274): the summaries report that quantization error accumulates or worsens over denoising steps. That is error per step, not step count [SEC] — [DLLMQuant](https://arxiv.org/pdf/2508.14090); [Quant-dLLM](https://www.arxiv.org/pdf/2510.03274).

### Inferences
- The standard protocol in dLLM PTQ papers fixes steps or the commit policy. A step-tax measurement under adaptive (threshold) decoding therefore appears to be an open gap. It is a plausible novelty point, provided DECAL truly does not exist.
- The T²Quant result that W4A4 collapses at 8 tokens/step suggests quantization interacts badly with high parallelism. That is indirect evidence of a tax.

### Gaps
- DECAL: title, arXiv id, ~13% figure and commit-temperature method could not be found or verified. The caller should supply its source (link or arXiv id).
- Full text of 2610.39853 (the closest match on topic: dLLM KV quantization with threshold decoding) was not read. Check its tables for steps or NFE.
- STaR-Quant and DecoCal full texts were not read.

## Q2. Mechanism: why would quantization change confidence or parallelism?

### Takeaway
The only explicit mechanism found is FAIR-Calib's. Quantization noise perturbs relative logits at borderline "write-frontier" positions and flips commit decisions, which are irreversible. FAIR-Calib measures this with logit margin and flip counts, not with step counts. No paper found links lower confidence or higher entropy to more steps under threshold decoding.

### Cited Findings
- FAIR-Calib: "PTQ error easily flips these borderline decisions at the write frontier". Diagnostics:
  - Mean frontier logit margin at commit steps: 6.56±0.62 for the baseline vs 7.23±0.64 for FAIR-Calib.
  - Teacher-forced commit flips per sequence: 2.9±1.4 vs 1.9±0.9 (N=32) [TEXT] — [alphaXiv](https://www.alphaxiv.org/abs/2606.06547.md).
- DLLMQuant's Certainty-Guided Quantization (CGQ) weights the Hessian toward tokens about to be unmasked or high-confidence tokens. The summary also reports a progressive decline in quality as iterations proceed [SEC] — [DLLMQuant](https://arxiv.org/html/2508.14090v1).
- T²Quant: sensitivity concentrates in early denoising windows. On GSM8K the first window carries 27% of the KL divergence, vs a uniform 6.25%. Divergence is measured at committed positions [TEXT] — [arXiv](https://arxiv.org/html/2610.31198).
- Background, not quantization: threshold decoding commits all tokens above τ (commonly 0.9) and falls back to the single top token. Lower confidence therefore means fewer tokens per step and more steps [SEC] — [ParallelBench](https://arxiv.org/pdf/2510.04767); [Fast-dLLM](https://arxiv.org/pdf/2505.22618).

### Inferences
- Combined mechanism (my inference, not a published result): quantization noise shrinks top-1 margins and confidence. Under a fixed τ, fewer positions clear the threshold, so the step count rises. FAIR-Calib's margin data supports the first link only.

### Gaps
- No published measurement of confidence or entropy distribution shift vs tokens/step under quantization was found.

## Q3. Rotation / outlier-smoothing methods applied to dLLMs, and whether the KV cache is included

### Takeaway
Rotation methods are applied to dLLMs widely, both as baselines and as foundations. "Quantization Meets dLLMs" (QuaRot, DuQuant, SmoothQuant), T²Quant (QuaRot-style rotations), FAIR-Calib (FlatQuant-style affine, with QuaRot as baseline) and SoloQ (its own K-RPBH Hadamard rotation) are the main cases. Only SoloQ includes the KV cache, and it rotates only the values; keys get KIVI-style per-channel quantization. None reports step changes.

### Cited Findings
- **Quantization Meets dLLMs** (arXiv 2508.14896, Aug 2025):
  - Methods: GPTQ and AWQ for weight-only; SmoothQuant, QuaRot and DuQuant for weight-activation. Rotation (QuaRot/DuQuant) clearly beats SmoothQuant on LLaDA-8B and LLaDA-8B-Instruct [SEC].
  - Recommended settings: 4-bit for weight-only, 8-bit as near-lossless for W-A [SEC].
  - QuaRot W4A4 drop reported as ~3.9% avg on Instruct and ~6.6% on Base [SEC]. The instruct model is more robust [SEC].
  - Activation outliers identified [SEC].
  - KV cache inclusion and steps: not verified, the full text timed out — [arXiv](https://arxiv.org/abs/2508.14896).
- **SoloQ** (2610.07121) [TEXT]:
  - Method: calibration-free; K-RPBH rotation (permutation → block randomized Hadamard → Haar mixer), with Lloyd–Max codebook or NVFP4.
  - KV cache: quantized at block commit only, with the active block kept in full precision. **Keys: KIVI-style per-channel asymmetric; values: SoloQ rotation path.**
  - Results: at 2-bit KV (~3 bits/entry) it averages 53.31 (codebook) / 53.53 (NVFP4) on Fast-dLLM v2, beating KIVI, OScar and TurboQuant baselines.
  - QuaRot is not applicable to Fast-dLLM v2 because of model dimensions [SEC snippet]. Folded QuaRot+GPTQ scores ~51 on LLaDA-Base vs FP16 ~59 [SEC].
  - Up to 2.61× lower peak memory and 2.24× faster with NVFP4. Rotation latency on Fast-dLLM v2 drops from ~98 ms to ~5.8 ms.
  - Steps fixed; no step tax reported — [HF papers](https://huggingface.co/papers/2610.07121).
- **T²Quant** (2610.31198): W4A4 (GPTQ weights, dynamic per-token activations) with QuaRot-style rotations; no KV cache; 128 steps fixed [TEXT] — [arXiv](https://arxiv.org/html/2610.31198).
- **FAIR-Calib** (2606.06547): FlatQuant-based learnable affine transforms, with QuaRot as a baseline; W4A4; no KV cache [TEXT] — [alphaXiv](https://www.alphaxiv.org/abs/2606.06547.md).
- **KV Cache Quantization for dLLMs** (2610.39853): KIVI/KVQuant adaptations, not rotation, per the abstract. Whether rotation baselines such as TurboQuant are included is unknown [ABS] — [arXiv abs](https://www.arxiv.org/abs/2610.39853).

### Inferences
- A QuaRot-like Hadamard rotation of KV in a dLLM, with steps measured under threshold decoding, does not appear in any paper found. SoloQ is the closest (rotation plus commit-time KV), but it uses fixed steps and does not rotate keys.

### Gaps
- Not checked: SpinQuant, DuQuant-on-KV, OstQuant or TurboQuant as the main method in a dLLM paper (TurboQuant appears only as a SoloQ KV baseline).
- KV handling in "Quantization Meets dLLMs" was not verified from the full text.

## Q4. Keys vs values sensitivity in dLLMs; systematic PTQ studies

### Takeaway
Two late-2026 papers treat keys as the more fragile part. dLLM-KVQ (2610.39853) keeps selected per-layer **key** entries in high precision and quantizes values with standard low bit. SoloQ uses per-channel (KIVI) quantization for keys and its rotation codebook for values. Neither, as far as I read, states a measured "keys more sensitive" ablation; the design choice implies it. Systematic PTQ studies (Quantization Meets dLLMs; the CoDA coding robustness study) focus on weights and activations.

### Cited Findings
- dLLM-KVQ (in 2610.39853):
  - Design: uses TRACE reveal-trajectory calibration to keep certain per-layer **key**-cache entries in high precision, chosen by output- and attention-level metrics, "while applying standard low-bit quantization to values".
  - Motivation: in dLLMs a reused cache already differs from full recomputation before any quantization, because bidirectional representations change as tokens are revealed [ABS] — [arXiv abs](https://www.arxiv.org/abs/2610.39853).
- SoloQ: asymmetric treatment of keys (per-channel KIVI) vs values (rotation) [TEXT] — [HF papers](https://huggingface.co/papers/2610.07121).
- "On the Quantization Robustness of Diffusion Language Models in Coding Benchmarks" (arXiv 2604.20079): finds dLLMs (CoDA) may be more quantization-resilient. Details were not read [SEC] — [arXiv](https://arxiv.org/pdf/2604.20079).
- DLLMQuant (2508.14090): timestep- and mask-aware calibration, CGQ, IA-AQ for attention modules. Whether it covers KV was not verified [SEC] — [arXiv](https://arxiv.org/pdf/2508.14090).

### Inferences
- The field is converging on "keys need more care" for dLLM KV, as in autoregressive LLMs (KIVI), but explicit K-vs-V sensitivity ablations for dLLMs were not confirmed. The user's own split-bit (k_bits / v_bits) measurements would be new evidence.

### Gaps
- Full texts of 2610.39853 (key/value ablations, steps under τ=0.9), 2508.14896 v3 and DLLMQuant were not read (arXiv timeouts). These should be read first.
