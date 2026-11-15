# KV-cache compression and quantization for diffusion language models (dLLMs), 2024 to Nov 2026

Verification labels used below:
- **[FULL]** means checked against the arXiv HTML full text in this session.
- **[ABS]** means only the arXiv abstract page or an aggregator abstract was read.
- **[SNIPPET]** means only a search-engine snippet or secondary summary was seen.
- **[LOCAL]** means the claim comes only from the user's own repository notes (C:\quantization\results\review_package.md). It was not re-checked online.
- **[UNVERIFIED]** means the source could not be reached (arXiv or OpenReview timed out, or a captcha blocked access).

arXiv and OpenReview often timed out during this session. Several items from the coordinator's list therefore stay unverified. They are flagged as such below.

## Q1. Which works quantize the dLLM KV cache, at what bits, on which models, and with what accuracy?

### Takeaway
Three works quantize the dLLM KV cache and give numbers checked against the full text:
- Tai et al. (2610.39853): KIVI, KVQuant and dLLM-KVQ at 4 and 2 bits on LLaDA and Dream. GSM8K loss is at most about 1 point on LLaDA and up to about 4.5 points on Dream at 2 bits.
- SoloQ (2610.07121): KV4 inside W4A4 on Fast-dLLM v2 and Nemotron-Labs-Diffusion. Its 2-bit KV-only table shows KIVI collapsing to 59.29 on Fast-dLLM v2 GSM8K, against 84.15 for FP16.
- MaskAhead (2610.06996): its Q-MaskAhead variant runs K4V4 combined with eviction.

The only other dLLM-specific KV quantizer found is in an NPU hardware paper (BAOS, 2601.20706). Most other "dLLM quantization" papers quantize weights and activations only.

### Cited Findings

**Tai et al., "KV Cache Quantization for Diffusion Language Models" (arXiv 2610.39853, submitted 29 Oct 2026, cs.CL)** [FULL]
- Authors: Yan Tai, Yunan Ding, Guangzhao Li, Siqi Luo, Chaoyang Zhang, Guangtao Zhai, Guodong Guo, Xiaohong Liu. [ABS] — [arXiv abs](https://arxiv.org/abs/2610.39853)
- Setting: KIVI and KVQuant are adapted to Fast-dLLM prefix cache and dual cache. The paper proposes two methods. TRACE builds calibration states from reconstructed reveal trajectories. dLLM-KVQ keeps selected per-layer key entries in high precision and applies standard low-bit quantization to values. — [arXiv abs](https://arxiv.org/abs/2610.39853)
- Models: GSAI-ML/LLaDA-8B-Instruct and Dream-org/Dream-v0-Base-7B. Bits: 4 and 2, with K and V both quantized. Group size 32. Generation length 256, block 32, Fast-dLLM confidence threshold 0.9. Benchmarks: GSM8K, HumanEval, MATH, MBPP. — [arXiv HTML](https://arxiv.org/html/2610.39853)
- GSM8K results. The reference is the unquantized cache-only run. No-cache full attention gives LLaDA 78.62 and Dream 74.15. — [arXiv HTML](https://arxiv.org/html/2610.39853)

  | Model | Setting | Reference | KIVI | KVQuant | dLLM-KVQ |
  |---|---|---|---|---|---|
  | LLaDA | 4b prefix | 78.47 | 77.33 | 77.48 | 77.94 |
  | LLaDA | 4b dual | 77.94 | 76.35 | 78.01 | 78.62 |
  | LLaDA | 2b prefix | 78.47 | 76.50 | 76.27 | 77.18 |
  | LLaDA | 2b dual | 77.94 | 77.56 | 77.48 | 77.79 |
  | Dream | 4b prefix | 74.60 | 73.69 | 74.83 | 74.53 |
  | Dream | 4b dual | 74.91 | 73.24 | 75.89 | 74.91 |
  | Dream | 2b prefix | 74.60 | 71.57 | 73.92 | 71.80 |
  | Dream | 2b dual | 74.91 | 70.43 | 72.71 | 71.72 |

- Decoding steps: none reported. The full text mentions no denoising-step count, NFE, forward-pass count or tokens per step, and does not say how quantization changes the step count under threshold-0.9 decoding. The threshold appears only as a setup parameter ("Fast-dLLM's confidence-threshold parallel decoding with threshold 0.9"). — [arXiv HTML](https://arxiv.org/html/2610.39853)
- Speed: no measurement. Appendix A.8 says runtime, throughput and peak memory are "outside the current experimental claims". The Limitations section says real speed depends on fused low-bit kernels. — [arXiv HTML](https://arxiv.org/html/2610.39853)
- Keys versus values: the paper argues that K matters more, because K error moves attention mass between positions while V error only perturbs content at positions already selected. It therefore spends its high-precision budget on K. There is no K-only or V-only ablation. The closest result is Table 3 (LLaDA, MBPP, 4-bit), which varies the number of protected K tokens per group (p):
  - Prefix cache: KIVI 26.60; fixed rule with p=4 25.60; dLLM-KVQ 26.60.
  - Dual cache: KIVI 25.60; dLLM-KVQ 26.80.
  - Appendix A.7 says V-side policy variants "showed weaker transfer in our diagnostics", with no numbers given. — [arXiv HTML](https://arxiv.org/html/2610.39853)
- Code: no URL is given. The paper only says released code or policy files are intended for research use. — [arXiv HTML](https://arxiv.org/html/2610.39853)

**SoloQ, "Calibration-Free Quantization for Diffusion Language Models" (arXiv 2610.07121, Oct 2026)** [FULL, first 100k of 133k characters]
- Main method: weight-activation quantization with a K-RPBH rotation and a rescaling correction. For block diffusion it adds commit-time KV quantization: a block's KV is quantized only when the block is committed, and the active block stays in full precision. Full-sequence dLLMs (LLaDA, Dream) have no persistent KV in this design, so KV quantization applies only to block models. — [arXiv HTML](https://arxiv.org/html/2610.07121)
- Quantizers:
  - Keys: KIVI-style per-channel asymmetric quantization.
  - Values: K-RPBH rotation, then Lloyd–Max (SoloQ-C) or NVFP4 (SoloQ-N). — [arXiv HTML](https://arxiv.org/html/2610.07121)
- W4A4 against W4A4KV4 (average / GSM8K). KV4 costs at most about 0.8 points. — [arXiv HTML](https://arxiv.org/html/2610.07121)
  - Fast-dLLM v2, SoloQ-C: 57.00/54.86 → 56.19/54.06.
  - Nemotron-Labs-Diffusion, SoloQ-N: 70.76/81.12 → 71.06/81.10.
- 2-bit KV-only test (Table 5): Fast-dLLM v2, FP16 weights and activations, about 3.00 bits per element. — [arXiv HTML](https://arxiv.org/html/2610.07121)

  | Method | GSM8K | IFEval |
  |---|---|---|
  | FP16 | 84.15 | 61.18 |
  | KIVI | 59.29 | 41.22 |
  | OScar | 52.39 | 40.85 |
  | TurboQuant | 0.30 | 7.76 |
  | SoloQ-C | 60.96 | 45.66 |
  | SoloQ-N | 60.65 | 46.40 |

- Decoding: Fast-dLLM v2 uses block 32, sub-block 8, threshold 1 (one token per step), greedy, up to 2048 tokens. Nemotron uses block 32, threshold 0.9, up to 8192 tokens, with EOS stopping disabled. The KV experiments report no step or NFE counts. The only forward-pass figure is for latency: 75 forward passes on a BF16-recorded Nemotron sequence, giving a 2.05× speedup. — [arXiv HTML](https://arxiv.org/html/2610.07121)
- Real speed: SoloQ-C uses 8-bit Tensor Core kernels on H200. SoloQ-N uses native NVFP4 kernels on RTX Pro 6000. Peak memory falls 2.31–2.61× and end-to-end speed rises 1.94–2.24× over BF16, but the KV contribution is not isolated. — [arXiv HTML](https://arxiv.org/html/2610.07121); [aiweekly summary](https://aiweekly.co/alerts/soloq-cuts-diffusion-llm-memory-261x-without-calibration-data)
- Keys versus values: no separate ablation in the portion read. Appendix D.4 covers attention sinks under KV quantization but was not read. — [arXiv HTML](https://arxiv.org/html/2610.07121)
- Code: the paper links https://github.com/Intelligent-Computing-Lab-Panda/SoloQ. [FULL] The user's notes say this link returns 404. [LOCAL] — [arXiv HTML](https://arxiv.org/html/2610.07121)

**MaskAhead / Q-MaskAhead, "Mask-Guided KV Cache Eviction in Block Diffusion Language Models" (arXiv 2610.06996, submitted 4 Oct 2026)** [FULL]
- Authors: Molodtsov, Alimaskina, Uskov, Zagitov, Beznosikov. — [arXiv abs](https://arxiv.org/abs/2610.06996)
- Models: Fast-dLLM-v2-7B, DreamReasoner-8B, LLaDA-2.0-mini. — [arXiv HTML](https://arxiv.org/html/2610.06996)
- Method: a single mask-query ranking serves two purposes. It selects a read subset for the current block and evicts persistent entries every 4 blocks. Each eviction keeps 4 sink tokens and a 64-token recent window. — [arXiv HTML](https://arxiv.org/html/2610.06996)
- Q-MaskAhead stores K4V4: keys per-channel asymmetric over 32-position groups, values per-token over 32-channel groups. It has no bf16 shadow copy. KIVI is cited only as related work and is not a baseline. — [arXiv HTML](https://arxiv.org/html/2610.06996)
- Budgets:
  - Reasoning: read min(N, max(128, 5%·N)); retain max(256, 10%·N).
  - Long-context QA: floors of 256 read and 512 retained. — [arXiv HTML](https://arxiv.org/html/2610.06996)
- Accuracy against dense (fraction of dense KV memory in parentheses): — [arXiv HTML](https://arxiv.org/html/2610.06996)
  - Fast-dLLM v2 GSM8K: dense 86.13; MaskAhead 84.91 (0.85); Q-MaskAhead 84.23 (0.32).
  - Fast-dLLM v2 MATH-500: dense 58.60; MaskAhead 54.60; Q-MaskAhead 54.00.
  - DreamReasoner MATH-500: dense 75.20; MaskAhead 68.00; Q-MaskAhead 67.40.
  - LongBench QA, average over 6 settings: MaskAhead cuts KV memory 9.5× for −1.2 F1; Q-MaskAhead 20.1× for −2.3 F1.
  - NIAH: dense 83.0, MAGE 83.5, MaskAhead 82.5, Q-MaskAhead 75.5.
  - No RULER results.
- Keys versus values (MATH-500, Fast-dLLM v2): keys are more sensitive. — [arXiv HTML](https://arxiv.org/html/2610.06996)
  - K2V4: 48.2. K4V2: 53.4. K2V2: 46.0. K4V4: 54.0.
  - Rotating and centering the keys raises HotpotQA K4V4 F1 from 34.75 to 37.33. The paper does not separate the two effects.
- Decoding: greedy. Threshold, steps per block, NFE and tokens per step are not reported. — [arXiv HTML](https://arxiv.org/html/2610.06996)
- Real speed: at batch 32, MaskAhead is 1.23× faster end to end and 1.68× faster in the decode stage. Q-MaskAhead is 1.20× and 1.64×. Throughput is 55.6, 68.2 and 66.5 tok/s for dense, MaskAhead and Q-MaskAhead, with fixed 256-token outputs and EOS disabled. Dense and plain K4V4 run out of memory at batch 64. Kernels are not described. — [arXiv HTML](https://arxiv.org/html/2610.06996)
- Baselines: HERALD, MAGE, H2O, H2O+CAOTE, Expected Attention, DapQ. — [arXiv HTML](https://arxiv.org/html/2610.06996)
- Code: https://github.com/brain-lab-research/MaskAhead (link given in the paper; whether the repository exists was not checked). — [arXiv HTML](https://arxiv.org/html/2610.06996)

**BitSieve (BitSieve-Fast-dLLMv2, anonymous.4open.science)** [LOCAL; the anonymous page returned only its title "Anonymous Github" to the fetcher]
- A fetch of the anonymous repository returned no content, so nothing could be verified online. — [anonymous.4open.science](https://anonymous.4open.science/r/BitSieve-Fast-dLLMv2)
- The user's notes describe:
  - Packed 2-bit KIVI kernels on Fast-dLLM v2. Configurations include `official_dense_bf16` and `dense_kivi2_k2v2_r32`, meaning 2-bit K and V with a residual window of 32.
  - Settings of threshold 0.95, up to 512 answer tokens and window 32.
  - GSM8K of 74.0 with KIVI 2-bit when reading only 64 cache entries per head, against 84.5 dense.
  - Speed measured with a fixed number of steps per block, so any step-count overhead cannot show up. [LOCAL]

**QuMix and its workshop version QWaF (Lee, Seo, Ko)** [LOCAL only; not found by any web search]
- Four web searches found no public page for "QuMix" or "QWaF". — [search 1](https://arxiv.org/html/2607.16339) (nearest hit LaCache, unrelated)
- The user's notes say QuMix evaluates KIVI on SDAR:
  - LongBench loss is about −0.5.
  - On RULER, the loss grows with input length (block 4: −1.8 → −8.2; 128 uncompressed tokens).
  - With Nemotron and block 32, RULER drops by −22 to −28. [LOCAL]
- Its venue, arXiv id and code could not be confirmed.

**BAOS inside "NPU Design for Diffusion Language Model Inference" (arXiv 2601.20706, 28 Jan 2026, revised 23 Apr 2026, cs.AR)** [ABS]
- Authors: Lou, Wu, Lau, … Mullins, Aaron Zhao. — [arXiv abs](https://arxiv.org/abs/2601.20706)
- The paper describes a dLLM NPU with its own ISA, compiler and RTL, synthesized in 7 nm. It quantizes the KV cache with Block-Adaptive Online Smoothing (BAOS), designed for the step-wise activation shifts of block-wise refinement. — [arXiv abs](https://arxiv.org/abs/2601.20706)
- The abstract gives no bit-width, model or numbers. The authors say they will open-source the code on acceptance. — [arXiv abs](https://arxiv.org/abs/2601.20706)
- The coordinator's label "DART" was not seen on the abstract page.

**Weight/activation-only dLLM quantization (no KV quantization; listed only to rule them out)**
- **"Quantization Meets dLLMs"** (Lin et al., arXiv 2508.14896, v3 Mar 2026) [FULL]: weight-only and weight-activation PTQ on LLaDA-8B and Dream-7B. It does not quantize the KV cache. It says only that the work "can be seamlessly integrated by quantizing dLLM caches". Code: https://github.com/FelixMessi/QDLM — [arXiv HTML](https://arxiv.org/html/2508.14896v3)
- **DLLMQuant** (Chen Xu, Dawei Yang; arXiv 2508.14090, Aug 2025) [SNIPPET]: W4A4 PTQ with three components (TMAS, IA-AQ, CGQ). Gains more than 10 points GSM8K on LLaDA at 4 bits. No KV-cache quantization appears in the abstract or snippets. Code is announced but not confirmed. — [arXiv](https://arxiv.org/abs/2508.14090)
- **STaR-Quant** (Yan, Wang, Wan, Yu, Tsang; arXiv 2606.04945, 3 Jun 2026) [ABS]: weight-activation PTQ with state-guided activation transforms and a temporal attention compensation step (a block-diagonal affine correction of quantized attention). Reports up to 1.69× speedup and 3.14× memory saving. The abstract does not say it quantizes the KV cache. No code link. — [arXiv abs](https://arxiv.org/abs/2606.04945)
- **LaCache** (arXiv 2607.16339, Jul 2026) [SNIPPET]: exact caching plus per-group FP8 for the FFN, not the KV cache. — [alphaXiv](https://www.alphaxiv.org/abs/2607.16339)
- **arXiv 2610.31198** (temporal task-aware quantization for diffusion LMs) [SNIPPET, title only]. — [opentrain](https://www.opentrain.ai/papers/different-tasks-different-timesteps-temporal-task-aware-quantization-for-diffusi--arxiv-2610.31198/)

**Related KV-quantization works on the coordinator's list that are not about dLLMs**
- **KVarN** (Muller et al., arXiv 2606.03458) [ABS]:
  - Calibration-free 2-bit KV quantizer: Hadamard rotation plus variance normalization along both axes of K and V.
  - Targets error accumulation over time in autoregressive reasoning (MATH500, AIME24, HumanEval).
  - Code with a vLLM implementation: https://github.com/huawei-csl/KVarN
  - Not dLLM. — [papers.cool](https://papers.cool/arxiv/2606.03458)
- **"Quantized Keys Steal Attention"** (Tuncer, Becker, Pfeil; arXiv 2605.26266, 25 May 2026) [ABS]:
  - Covers chunk-wise autoregressive *video* diffusion (MAGI-1, SkyReels-V2, HY-WorldPlay).
  - Finds a "Jensen bias": quantizing keys inflates the attention paid to cached keys and pulls attention away from the current, unquantized chunk.
  - A Taylor-based correction makes INT2 approach BF16 and beat INT4.
  - The mechanism of quantized cached keys "stealing" attention from the full-precision active block maps directly onto block dLLMs, but the paper does not test language models. — [papers.cool](https://papers.cool/arxiv/2605.26266)
- **Gu et al., "When Attention Sink Emerges in Language Models"** (arXiv 2410.10781, Sea AI Lab) [ABS]: attention-sink analysis of autoregressive LMs. The sink acts like a key bias. Code: https://github.com/sail-sg/Attention-Sink. Not about compression. — [HF papers](https://huggingface.co/papers/2410.10781)
- **arXiv 2605.06611** ("The Structural Origin of Attention Sink", Li, Jiang, Sun, Hu; 7 May 2026) [ABS]: attention-sink mechanism. No compression and no code. — [papers.cool](https://papers.cool/arxiv/2605.06611)
- **arXiv 2609.09085** ("It's Not RoPE that Creates Sinks", Kiya et al.; 8 Sep 2026; EMNLP 2026) [ABS]: sinks and massive activations explained by causal-mask self-concentration. Mentions relevance to low-bit quantization. No code. — [arXiv abs](https://arxiv.org/abs/2609.09085)

### Inferences
- The accuracy cost of 2-bit KV varies widely across models and decoders:
  - LLaDA with prefix cache (Tai): about −1 GSM8K.
  - Dream (Tai): −3 to −4.5.
  - Fast-dLLM v2 with threshold 1 and 2048 tokens (SoloQ): −25.
- The SoloQ number is an outlier against Tai and the user's own measurements (−0.2 to −3.4 [LOCAL]). It may depend on its long 2048-token generation or on implementation details. The user could not reproduce it [LOCAL].
- Two papers independently find keys more sensitive than values in dLLMs. MaskAhead shows it with numbers (K2V4 48.2 against K4V2 53.4). Tai et al. argue it and build their method on it. The video-diffusion "Jensen bias" paper gives a mechanism for key error that applies when quantized cache meets a full-precision active block. This exact situation exists in block dLLMs and in SoloQ's commit-time design.

### Gaps
- BRQ-KV (arXiv 2609.01084): arXiv and Hugging Face timed out or returned 404, and search found nothing. [UNVERIFIED]
- TraceKV (OpenReview qSAFcu…), iKVC, Step-Wise Endorsement (OpenReview IhSt4QyAGl), DECAL (OpenReview xQ57THZa4v) and "When Do Denoising Errors Propagate?" could not be checked. OpenReview showed a browser-verification wall or timed out, and search found no match for the exact titles. [UNVERIFIED]
  - DECAL facts in the user's notes (W4A4 overhead up to 13%, LLaDA and Dream, block 32) are [LOCAL].
- SoloQ Appendix D.4 (sinks under KV quantization) and the last 30% of the paper were not read.
- QuMix/QWaF: no public trace was found.

## Q2. Which works do token eviction, selection or sparse reading of the dLLM cache, and do they measure step counts under threshold decoding?

### Takeaway
There is an active line of selection and eviction work:
- Sparse-dLLM (2508.02558).
- SparseD (2509.24014, ICLR 2026).
- MaskKV (2510.09309, ACL Findings 2026).
- MAGE (2602.14209).
- HERALD (2606.21633).
- MaskAhead (2610.06996).

Most report throughput or latency, usually with fixed outputs and EOS disabled. None of the texts or abstracts read reports how eviction changes the number of steps under confidence-threshold decoding.

### Cited Findings
- **Sparse-dLLM** (arXiv 2508.02558, AAAI 2026) [SNIPPET]:
  - Training-free "delayed bidirectional sparse caching". Keeps pivotal tokens, evicts unimportant prefix and suffix entries based on attention, and delays cache updates by one step.
  - Up to 10× throughput with comparable accuracy on LLaDA and Dream.
  - Code: https://github.com/OpenMOSS/Sparse-dLLM (GitHub listing appeared in search). — [arXiv](https://arxiv.org/pdf/2508.02558); [GitHub](https://github.com/OpenMOSS/Sparse-dLLM)
- **MaskKV, "Mask Tokens as Prophet"** (arXiv 2510.09309, ACL Findings 2026) [SNIPPET]:
  - Eviction guided by mask-token attention, on LLaDA and Dream.
  - With only 256 KV pairs (under 5% of tokens), keeps 94% of full-cache LongBench on LLaDA.
  - Up to 31× faster at 32k prompt length. — [arXiv](https://arxiv.org/abs/2510.09309); [ACL Anthology](https://aclanthology.org/2026.findings-acl.170/)
- **SparseD** (arXiv 2509.24014, ICLR 2026) [SNIPPET]:
  - Full attention in early steps, then head-specific sparse patterns computed once and reused.
  - RULER-8k: 72.47 for SparseD against 55.08 for dKV-Cache.
  - Cache methods (dKV-Cache, Fast-dLLM) lose about 16% on RULER-8k on Dream-7B-Instruct.
  - Up to 1.50× speedup at 64k with 1024 steps, which is a fixed step count. — [arXiv](https://arxiv.org/pdf/2509.24014); [ICLR](https://iclr.cc/virtual/2026/poster/10008368)
- **MAGE** (Kwon et al., SNU; arXiv 2602.14209, 15 Feb 2026) [SNIPPET; arXiv and alphaXiv timed out]:
  - Uses the attention of the first all-[MASK] denoising step to choose KV entries for the whole block in block diffusion.
  - Near-lossless on LongBench and NIAH at a fraction of the KV budget.
  - Up to 6.82× end-to-end at 128K, and 3.35× faster than Quest and 2.28× faster than SparseD.
  - Notes that per-block constraints cut the recall of existing sparse estimators by up to 25%. — [arXiv](https://arxiv.org/abs/2602.14209); [HF](https://huggingface.co/papers/2602.14209)
- **HERALD** (Kwon, Kim, Park, Lee, Stoica, Jae W. Lee; arXiv 2606.21633, 23 Jun 2026) [SNIPPET]:
  - CPU holds the full KV cache and selects sparse entries per block; the GPU holds a small pool. Built on SGLang.
  - Tested on SDAR-8B-Chat and LLaDA 2.0-mini.
  - Near-lossless at a 5–10% KV budget.
  - Speed claims differ between listings: one gives up to 2.28× decode throughput, another 1.59× lower per-block latency and 2.47× throughput. — [arXiv](https://arxiv.org/pdf/2606.21633); [chatpaper](https://chatpaper.com/paper/301801)
- **MaskAhead**: see Q1. No threshold or step statistics. — [arXiv HTML](https://arxiv.org/html/2610.06996)
- Other dLLM cache or decoder works seen only as titles in search, not read: [SNIPPET]
  - FlashBlock.
  - Flash-dLLM (arXiv 2609.26796, "IO-Aware KV Caching and Parallel Decoding").
  - "Roll Out and Roll Back" (arXiv 2605.16941).
  - Fast-dLLM (2505.22618, code at NVlabs/Fast-dLLM).
  - Fast-dLLM v2 (2509.26328). It uses a block-level cache plus sub-block dual cache, and its adaptation needs about 1B fine-tuning tokens. — [Fast-dLLM v2](https://arxiv.org/pdf/2509.26328); [Flash-dLLM](https://arxiv.org/pdf/2609.26796); [GitHub Fast-dLLM](https://github.com/NVlabs/Fast-dLLM)

### Inferences
- The selection and eviction papers measure speed with fixed output lengths or fixed step counts (MaskAhead: 256 tokens with EOS disabled; SparseD: 1024 steps). That makes them structurally blind to any change in the number of steps that threshold decoding would produce.

### Gaps
- Full texts of MAGE, HERALD, SparseD, Sparse-dLLM and MaskKV were not read because arXiv timed out, so their decoding settings and step reporting are unconfirmed.
- dKV-Cache, dLLM-Cache and Elastic-Cache (the dLLM version) were not reached. Search returned only a vision-language "Elastic Cache".

## Q3. Does any work report that KV compression changes the number of decoding steps or tokens per step under confidence-threshold decoding?

### Takeaway
None found. Tai et al. use threshold 0.9 and explicitly report no steps, NFE or tokens per step. SoloQ reports no steps for its KV experiments. MaskAhead reports none. The eviction papers measure fixed-length or fixed-step throughput. The only source with this measurement is the user's own work: a 7–34% increase in steps at about 3 effective bits, and +10.7% under BitSieve settings. [LOCAL]

### Cited Findings
- Tai et al. use threshold 0.9. The full text contains no step, NFE or tokens-per-step statistic and no analysis of how quantization changes step counts. — [arXiv HTML](https://arxiv.org/html/2610.39853)
- SoloQ uses threshold 1 on Fast-dLLM v2 (one token per step, so the step count is fixed by construction) and threshold 0.9 on Nemotron. It reports no step counts for KV runs. — [arXiv HTML](https://arxiv.org/html/2610.07121)
- MaskAhead reports no threshold or step statistics, and its throughput is measured with fixed 256-token outputs. — [arXiv HTML](https://arxiv.org/html/2610.06996)
- SparseD measures speed at a fixed 1024 steps. — [arXiv](https://arxiv.org/pdf/2509.24014)

### Inferences
- The claim "KV compression changes tokens per step under threshold decoding" appears to be unreported in the literature found. That holds with the caveat that MAGE, HERALD, QuMix, TraceKV and the OpenReview items could not be read in full.

### Gaps
- Unread full texts (MAGE, HERALD, Sparse-dLLM, MaskKV, SparseD, BRQ-KV, TraceKV, iKVC, Step-Wise Endorsement, DECAL) could still contain step statistics.

## Q4. Which works have public code?

### Takeaway
Code is linked by:
- MaskAhead
- SoloQ (link reported dead by the user)
- Sparse-dLLM
- Fast-dLLM
- QDLM ("Quantization Meets dLLMs")
- KVarN (not dLLM)
- Attention-Sink (Gu et al.)

Tai et al., STaR-Quant and the 2601.20706 NPU paper link no code. BitSieve is on an anonymous mirror that could not be read.

### Cited Findings
- MaskAhead: https://github.com/brain-lab-research/MaskAhead (linked in the paper; existence not checked). — [arXiv HTML](https://arxiv.org/html/2610.06996)
- SoloQ: https://github.com/Intelligent-Computing-Lab-Panda/SoloQ (linked in the paper). — [arXiv HTML](https://arxiv.org/html/2610.07121). The user observed a 404. [LOCAL]
- Tai et al.: no code URL. — [arXiv HTML](https://arxiv.org/html/2610.39853)
- Sparse-dLLM: https://github.com/OpenMOSS/Sparse-dLLM — [GitHub](https://github.com/OpenMOSS/Sparse-dLLM)
- Fast-dLLM v1/v2: https://github.com/NVlabs/Fast-dLLM — [GitHub](https://github.com/NVlabs/Fast-dLLM)
- QDLM: https://github.com/FelixMessi/QDLM — [arXiv HTML](https://arxiv.org/html/2508.14896v3)
- KVarN: https://github.com/huawei-csl/KVarN — [papers.cool](https://papers.cool/arxiv/2606.03458)
- NPU/BAOS (2601.20706): code promised on acceptance. — [arXiv abs](https://arxiv.org/abs/2601.20706)
- STaR-Quant: none. — [arXiv abs](https://arxiv.org/abs/2606.04945)
- BitSieve: anonymous.4open.science page returned no content to the fetcher. — [link](https://anonymous.4open.science/r/BitSieve-Fast-dLLMv2)

### Inferences
- Among dLLM KV *quantization* papers, MaskAhead is the only one with a code link that appears live. Its existence was not checked here.

### Gaps
- The GitHub repositories were not opened to confirm they exist or contain the KV-quantization code. MAGE, HERALD, SparseD and MaskKV code status is unknown.
