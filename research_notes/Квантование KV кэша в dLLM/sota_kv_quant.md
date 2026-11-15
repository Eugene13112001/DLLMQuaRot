# SOTA of KV-cache quantization for autoregressive LLMs (as of 2026-11-15), for transfer to block-diffusion LMs

Verification labels used below:
- **[paper]** — read in the paper's full text (arXiv HTML) in this session.
- **[abstract]** — from the arXiv abstract / listing page only.
- **[secondary]** — from a search snippet, aggregator, blog or press piece; not checked against the paper.
- **[background]** — from my prior knowledge of the literature (cutoff mid-2026), NOT re-verified in this session. Treat ids/details as needing a check before citing in the paper.

Process note: many arXiv fetches timed out in this session (OSCAR 2605.17757, the benchmark paper 2607.05399, OptR 2608.02691, Kitty full PDF). Their content below is snippet/abstract-level only.

## Q1. Which method is considered best at ~2–3 bits in 2025–2026 comparisons?

### Takeaway
There is no single accepted winner, and I found no 2025–2026 independent benchmark that puts all the strong methods side by side under matched memory. The two families that consistently beat KIVI at 2–3 bits are (a) **random/Hadamard rotation + good scalar quantizer with a few outlier channels at higher precision** (TurboQuant, ICLR 2026; RotateKV, IJCAI 2025; 2026 follow-ups OSCAR, OptR, FlagRot that tune the rotation), and (b) **KIVI-style per-channel keys with a small set of channels/tokens kept at higher precision** (Kitty 2025, KVQuant 2024, dLLM-KVQ 2026). Vector/codebook methods (CQ, CommVQ) reach 1–2 bits but need trained codebooks.

### Cited Findings

**Rotation + scalar quantization family**
- **TurboQuant** — "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate", Zandieh, Daliri, Hadian, Mirrokni (Google), arXiv 2504.19874, ICLR 2026 poster. [paper for method and tables; secondary for venue] — [arXiv HTML](https://arxiv.org/html/2504.19874); [venue via search summary](https://liner.com/review/turboquant-online-vector-quantization-with-nearoptimal-distortion-rate)
  - Core idea [paper]: multiply each vector by a random orthogonal matrix (QR of a Gaussian matrix); each coordinate then is ~N(0,1/d), so a fixed, precomputed Lloyd-Max scalar codebook per bit width is near-optimal. Inner-product variant: MSE quantizer at b−1 bits + 1-bit QJL (sign of a Gaussian projection) on the residual, with the residual norm stored, giving an unbiased inner-product estimate. MSE-only has a multiplicative bias (2/π at 1 bit) that shrinks with bits. — [arXiv HTML](https://arxiv.org/html/2504.19874)
  - Outliers / fractional bits [paper]: channels split into outlier and regular sets with separate TurboQuant instances; 2.5 bits = 32 channels at 3 bits + 96 channels at 2 bits (head_dim 128). The 3.5-bit split is not given. — [arXiv HTML](https://arxiv.org/html/2504.19874)
  - Calibration [paper]: none ("data-oblivious"); only codebooks precomputed once per bit width. — [arXiv HTML](https://arxiv.org/html/2504.19874)
  - Pre- vs post-RoPE: not stated in the extracted text. [paper, absence]
  - LongBench (LongBench-E, Llama-3.1-8B-Instruct, average) [paper]: Full 16-bit 50.06; KIVI 3-bit 48.50; KIVI 5-bit 50.16; PolarQuant 3.9-bit 49.78; TurboQuant 2.5-bit 49.44; TurboQuant 3.5-bit 50.06. Ministral-7B: Full 49.89, TurboQuant 2.5-bit 49.62. — [arXiv HTML](https://arxiv.org/html/2504.19874)
  - Needle-in-a-haystack (Llama-3.1-8B, 4k–104k, compression 0.25) [paper]: SnapKV 0.858, PyramidKV 0.895, KIVI 0.981, PolarQuant 0.995, Full 0.997, TurboQuant 0.997. — [arXiv HTML](https://arxiv.org/html/2504.19874)
  - Kernels [paper]: no fused attention kernel or end-to-end speed in the paper; only quantization time (e.g. 0.0013 s at d=1536 vs 239.75 s for PQ). Google's "up to 8x faster attention logits on H100" is a press claim [secondary]. — [arXiv HTML](https://arxiv.org/html/2504.19874); [techinformed](https://techinformed.com/google-publishes-turboquant-to-ease-ai-memory-strain/)
  - Code: no official Google code as of April 2026 per secondary sources; community implementations exist (e.g. github.com/0xSero/turboquant, "TurboQuant+") and one reports a speed regression vs q8_0 in llama.cpp. [secondary] — [search summary incl. 0xSero repo](https://github.com/0xSero/turboquant); [TurboQuant+ docs](https://docsearch.algolia.com/mcp/docs/repo/thetom/turboquant_plus)
  - Caveat [paper]: compression factor wording is inconsistent within the paper (">5x", "at least 4.5x", ">4x"). The "KV size (bits)" column does not obviously include the stored residual norm / per-vector norm overhead — verify before comparing at "equal memory". — [arXiv HTML](https://arxiv.org/html/2504.19874)
- **RotateKV** — "RotateKV: Accurate and Robust 2-Bit KV Cache Quantization for LLMs via Outlier-Aware Adaptive Rotations", Su et al., arXiv 2501.16383, IJCAI 2025 (pp. 6200–6208). [abstract/secondary] — [IJCAI](https://www.ijcai.org/proceedings/2025/690); [alphaxiv](https://alphaxiv.org/abs/2501.16383)
  - Core idea: channel reordering so the fast Walsh–Hadamard transform adapts to outlier channels; **pre-RoPE grouped-head rotation** (spreads outliers across heads); protection of attention-sink tokens (massive activations). [abstract]
  - Claims: <0.3 PPL degradation at 2 bits on WikiText-2 (LLaMA-2-13B); <1.7% drop on GSM8K; 3.97x lower peak memory, 5.75x larger batch, 2.32x decode speedup. [abstract] Baselines and exact bit accounting not verified.
- **FlagRot** — "FlagRot: Selective Rotation for Low-Bit KV Caches", Shao et al. (HKUST), arXiv 2611.03992, 2026-11-02. [abstract via papers.cool] — [papers.cool](https://papers.cool/arxiv/2611.03992)
  - Finding: rotation does not always help — it flattens spiky key segments but can create a peak in flat ones. Each 32-channel segment is stored rotated or not, chosen by calibrated attention error; query rotated at read time. For per-channel (KIVI) caches one decision per head; for per-token caches the flag is stored per key.
  - Claims: at 2.5 bits/value, KIVI's KL divergence over 12 models drops 18.7 → 12.4 (×1e-3); beats RotateKV on 9/12 models (KL); +2.9 RULER at 128k as a vLLM plugin; LongBench v1 +0.4 to +1.9 on 3 of 4 models where 2-bit KIVI loses accuracy. Code link not found.
- **OSCAR** — "Offline Spectral Covariance-Aware Rotation for 2-bit KV Cache Quantization", arXiv 2605.17757. [secondary, snippet only; fetch failed] Notes that methods using channel-wise precision, residual buffers, progressive bit allocation or adaptive layouts complicate paged-KV serving and fused decode kernels, i.e. it argues for a pure offline rotation. Numbers not obtained. — [arXiv PDF link](https://arxiv.org/pdf/2605.17757)
- **OptR** — "Output-Aware Rotation for INT2 KV-Cache Quantization", arXiv 2608.02691. [secondary, snippet only] Rotation chosen to minimise attention-output error instead of cache reconstruction error. Numbers not obtained. — [alphaxiv](https://www.alphaxiv.org/abs/2608.02691)
- QuaRot (2404.00456) and SpinQuant (2405.16406) quantize KV per head after a Hadamard (QuaRot) / learned (SpinQuant) rotation, mainly at 4 bits; they are weight+activation methods, not KV-specialised. [background]

**Per-channel keys + higher-precision exceptions**
- **Kitty** — "Kitty: Accurate and Efficient 2-bit KV Cache Quantization with Dynamic Channel-wise Precision Boost", Xia et al. (Together AI et al.), arXiv 2511.18643, 2025-11-23; no venue listed. [abstract] — [arXiv abs](https://arxiv.org/abs/2511.18643)
  - Core idea: rank key channels by sensitivity and keep a small fraction at higher precision ("near 2-bit" memory); mixed-precision key page split into two tensors both stored as 2-bit, enabling a paged layout and Triton dequant kernels.
  - Claims: tested on Qwen3 and LLaMA3, 7 tasks; ~8x KV memory reduction with "negligible" loss, up to 8x larger batch, 2.1–4.1x throughput. Code: https://github.com/Summer-Summer/Kitty (licence not checked).
  - Kitty's own re-evaluation says KIVI "fails to preserve model accuracy" at 2 bits, with average drops of roughly 2–16 points depending on model/task. [secondary snippet of the PDF] — [arXiv PDF](https://arxiv.org/pdf/2511.18643)
  - Relevance: Qwen3 is SDAR's base; this is the only strong 2-bit method I found evaluated on Qwen3. [abstract]
- **KVQuant** — Hooper et al., arXiv 2401.18079, NeurIPS 2024. Claims "2-bit KVQuant achieves similar accuracy to KIVI with 1.5x smaller bit-width", comparing with KIVI at group 32 + 128-token fp16 residual. [secondary snippet of the paper] — [arXiv HTML](https://arxiv.org/html/2401.18079)
  - Core idea [background]: per-channel **pre-RoPE** key quantization, non-uniform sensitivity-weighted datatype (needs calibration), dense-and-sparse outlier isolation (~1% in fp16), attention-sink kept in fp16; custom CUDA kernels; code github.com/SqueezeAILab/KVQuant.

**Vector / codebook / residual methods** [background unless noted]
- QJL (arXiv 2406.03482, Zandieh et al.): 1-bit JL sign sketch of keys with unbiased inner product; component of TurboQuant. PolarQuant (arXiv 2502.02617): polar-coordinate quantization after random preconditioning; 3.9 bits in TurboQuant's table [paper] — [TurboQuant HTML](https://arxiv.org/html/2504.19874).
- CQ coupled quantization (arXiv 2405.03917, NeurIPS 2024): jointly quantizes groups of channels with learned codebooks, down to 1 bit/channel; needs calibration.
- CommVQ (arXiv 2506.18879, ICML 2025): additive VQ with a RoPE-commutative codebook trained by EM; reports 2-bit and 1-bit KV; needs codebook training per model. (Search for its numbers failed in this session.)
- GEAR (arXiv 2403.05527): uniform quant + low-rank + sparse residual correction. MiKV (2402.18096) and ZipCache (2405.14256): importance-based mixed precision per token. SKVQ (2405.06219): channel reordering + clipped groups + full-precision sliding window. QServe (2405.04532): KV4 with SmoothAttention in a serving system. KVTuner (2502.04420, ICML 2025): layer-wise mixed precision for K vs V found offline. BitDecoding (2503.18773): tensor-core kernels for low-bit KV decode.
- KVTC (NVIDIA, ICLR 2026): PCA decorrelation + entropy coding, ~20x compression with a small penalty. [secondary] — [search summary](https://liner.com/review/turboquant-online-vector-quantization-with-nearoptimal-distortion-rate)

**Independent benchmark**
- "Benchmarking KV-Cache Optimizations across Task Quality and System Performance for Long-Context Serving", Agrawal & Mayer (Univ. Bayreuth), arXiv 2607.05399: evaluates KIVI and TurboQuant (quantization), SnapKV (eviction), CaM (merging) under matched models/data. [secondary; fetch timed out — results NOT obtained] — [arXiv](https://arxiv.org/abs/2607.05399)

### Inferences
- At ~2.5–3 bits, the rotation-based data-oblivious approach (TurboQuant) is the most cited new reference point in 2026, and 2026 papers (OSCAR, OptR, FlagRot) position themselves as improvements on rotation, i.e. rotation is the current "mainline".
- At strict 2 bits, the methods that reliably close the gap all keep *something* at higher precision (outlier channels in TurboQuant-2.5 / Kitty, sink tokens in RotateKV/KVQuant, salient tokens in dLLM-KVQ). Pure uniform 2-bit, rotated or not, is generally reported to lose accuracy.
- TurboQuant's "2.5 bits" is effectively "Hadamard-like rotation + per-channel mixed 2/3-bit + Lloyd-Max codebook" — structurally close to our existing Hadamard scheme plus outlier channels.

### Gaps
- No matched-memory head-to-head of TurboQuant vs Kitty vs RotateKV vs KVQuant vs CommVQ found; the one independent benchmark (2607.05399) could not be read.
- OSCAR/OptR numbers, code and licences not obtained.
- Licences of Kitty, KVQuant, CommVQ repos not checked.

## Q2. How large is the gap to KIVI-2bit at equal memory on long-context benchmarks?

### Takeaway
Reported gaps are small on LongBench averages (about 1 point) and larger on hard long-context retrieval/reasoning. Numbers are author-reported and settings differ.

### Cited Findings
- TurboQuant 2.5-bit 49.44 vs KIVI 3-bit 48.50 vs full 50.06 (LongBench-E avg, Llama-3.1-8B), i.e. TurboQuant at fewer bits is ~0.9 points above KIVI at more bits; NIAH 0.997 vs 0.981. [paper] — [arXiv HTML](https://arxiv.org/html/2504.19874)
- FlagRot: on LongBench v1, +0.4 to +1.9 points over 2-bit KIVI on 3 of 4 models; +2.9 on RULER-128k. [abstract] — [papers.cool](https://papers.cool/arxiv/2611.03992)
- Kitty: KIVI-2bit average drops of ~2–16 points vs full precision depending on model/task; Kitty claims near-zero loss. [secondary] — [arXiv PDF](https://arxiv.org/pdf/2511.18643)
- KVQuant: matches KIVI accuracy at 1.5x fewer bits. [secondary] — [arXiv HTML](https://arxiv.org/html/2401.18079)

### Inferences
- On a LongBench average the headroom over a well-configured KIVI (group 32, fp16 residual window) is ~1–2 points; differences show up more on RULER at long lengths and on reasoning (GSM8K/AIME with long CoT), which is where Kitty and FlagRot report gains.
- Our own observation that block-diffusion decoding re-reads the whole cache many times per block may amplify per-step errors, so the gap could be larger in dLLMs than in AR models — to be measured, not assumed.

### Gaps
- No RULER numbers for TurboQuant or Kitty read directly; bit accounting (scales/zero points/norms/residual windows) is not consistently reported across papers, so "equal memory" comparisons are not truly established.

## Q3. Which are simple to reimplement as fake quantization, and which need calibration or kernels?

### Takeaway
TurboQuant (MSE variant) and RotateKV-style pre-RoPE Hadamard are calibration-free and fake-quant friendly; Kitty-like channel boost needs only a cheap online/offline channel ranking; KVQuant, CQ, CommVQ, FlagRot, OSCAR and dLLM-KVQ need calibration data or trained codebooks.

### Cited Findings
- TurboQuant: no calibration; random rotation + fixed Lloyd-Max codebooks per bit width [paper] — [arXiv HTML](https://arxiv.org/html/2504.19874). Fake-quant = rotate, normalise per vector, nearest-centroid lookup, de-rotate; QJL residual stage only matters if estimating inner products rather than reconstructing.
- RotateKV: Hadamard (FWHT) with channel reordering, pre-RoPE grouped-head rotation, sink protection [abstract] — [IJCAI](https://www.ijcai.org/proceedings/2025/690). Channel reordering likely needs a small calibration pass (inference).
- Kitty: sensitivity ranking of key channels ("dynamic") + Triton kernels released [abstract] — [arXiv abs](https://arxiv.org/abs/2511.18643).
- FlagRot: per-segment rotate/no-rotate decision from "calibrated attention error" [abstract] — [papers.cool](https://papers.cool/arxiv/2611.03992).
- dLLM-KVQ: trajectory calibration (min-cost max-flow over reveal steps) + discrete policy search [paper] — [arXiv HTML](https://arxiv.org/html/2610.39853v1).
- KVQuant non-uniform datatype and CQ/CommVQ codebooks require calibration/training [background].

### Inferences (difficulty for our codebase, which already has KIVI and a Hadamard per-token scheme with k_bits/v_bits)
- TurboQuant-MSE fake quant: ~1 day. Reuse our Hadamard (or a random orthogonal matrix), store per-vector L2 norm (16 bits per 128 dims = +0.125 bit), Lloyd-Max centroids for N(0,1/d) (computable offline in a few lines), optional 32 outlier channels at b+1 bits for 2.5/3.5-bit points. QJL residual stage: +0.5 day, optional.
- Kitty-style channel boost on top of KIVI keys: ~1 day (pick top-k channels by per-channel range/sensitivity per head, quantize them at 4 bits, rest at 2 bits; account bits honestly).
- RotateKV pre-RoPE grouped-head rotation: ~1–2 days (requires hooking K before RoPE in the Qwen2/Qwen3 attention forward, and applying RoPE after dequant).
- KVQuant full / CommVQ / CQ: several days plus calibration; CommVQ requires codebook training.

### Gaps
- Whether TurboQuant quantizes pre- or post-RoPE is not stated in what I read.

## Q4. Any of them already applied to diffusion LLMs?

### Takeaway
Yes: one paper from 29 Oct 2026 (arXiv 2610.39853) applies KIVI and KVQuant to LLaDA and Dream under Fast-dLLM caching and proposes dLLM-KVQ. It does not cover SDAR, Fast-dLLM v2, rotation methods, TurboQuant or long context — so those remain open for us.

### Cited Findings
- "KV Cache Quantization for Diffusion Language Models", Tai, Ding, Li, Luo, Zhang, Zhai, Guo, Liu (SJTU / EIT Ningbo / PolyU), arXiv 2610.39853v1, 29 Oct 2026. [paper] — [arXiv HTML](https://arxiv.org/html/2610.39853v1)
  - Models: LLaDA-8B-Instruct, Dream-v0-Base-7B; Fast-dLLM prefix and dual cache; gen length 256, block 32, threshold 0.9.
  - Baselines: KIVI, KVQuant (adapted to Fast-dLLM). No QuaRot/rotation, no TurboQuant, no SDAR / Fast-dLLM v2.
  - Method: K per-channel over token groups, V per-token; per-layer budget of {0,1,2,4} K tokens per group kept in high precision, chosen by discrete search using trajectory calibration (TRACE). Group 32. Effective K bits 4.77 (prefix) / 4.60 (dual) at "4-bit" — overhead not fully accounted.
  - Average over GSM8K/HumanEval/MATH/MBPP (cache-only / KIVI / KVQuant / dLLM-KVQ): LLaDA prefix 2-bit 45.08 / 43.62 / 41.45 / 44.38; LLaDA dual 2-bit 43.09 / 42.84 / 40.60 / 43.25; Dream prefix 2-bit 51.82 / 48.98 / 48.75 / 49.86; Dream dual 2-bit 49.65 / 48.10 / 46.33 / 48.83. At 4 bits all methods within ~1.3 points of cache-only.
  - Notable: KVQuant is *worse* than KIVI at 2 bits on dLLMs in this paper; KIVI loses only 0.25–2.8 points at 2 bits on short-generation tasks.
  - Limitations stated: no LongBench, no speed/memory, no V-side policy, no code link.

### Inferences
- KIVI is a strong baseline on dLLMs for short tasks; differences between methods are ~1 point at 2 bits there. Long-context (LongBench/RULER) on block-diffusion models with rotation/codebook methods is not covered by anyone I found — a clear novelty slot for our paper.
- dLLM-KVQ must be cited and ideally compared against (or its "keep a few K tokens high-precision" idea reproduced cheaply), since reviewers will know it.

### Gaps
- No search hit for KV quantization on SDAR or Fast-dLLM v2 specifically (one search on dLLM KV quant 2026 failed; the other found only 2610.39853).

## Recommendation (part of the notes for the report writer)

Implement next, besides KIVI and our QuaRot-like Hadamard per-token scheme:

1. **TurboQuant (MSE variant, with outlier-channel 2.5/3.5-bit splits)** — the most cited 2026 reference point (ICLR 2026), calibration-free, matches our existing rotation code path, reported to beat KIVI-3bit at 2.5 bits on LongBench and to match full precision on NIAH [paper]. Fake-quant reimplementation is about 1 day: random orthogonal/Hadamard rotation, per-vector norm, precomputed Lloyd-Max centroids for N(0,1/d), 32 outlier channels at +1 bit; count the norm (and the outlier split) in bits. Optional QJL residual later. It also gives a clean "scalar vs codebook" axis versus our uniform Hadamard scheme.
2. **Kitty-style channel-wise precision boost on KIVI keys** (or equivalently, the outlier-channel split applied in the non-rotated basis) — the only strong 2-bit method evaluated on Qwen3 (SDAR's base), code public, ~1 day to fake-quant. It directly tests the "keep a few things at higher precision" principle that dLLM-KVQ (the only dLLM KV-quant paper) also relies on, but without trajectory calibration.

Lower priority: RotateKV's pre-RoPE grouped-head rotation + sink protection (1–2 days, needs hooking before RoPE); FlagRot (needs calibration, very recent, no code found); CommVQ/CQ/KVQuant (calibration or codebook training, multi-day). Must cite and discuss dLLM-KVQ (2610.39853) as the closest prior work.
