# Real systems-level speed of quantized/compressed KV caches in dLLMs, and serving-engine support (2024 to Nov 2026)

Notes compiled 2026-11-15. Verification tags:
- **[V-text]**: read in the primary source's full text or official doc page (via fetch).
- **[V-abs]**: only from the abstract or the paper landing page.
- **[2nd]**: from a search-engine summary or a secondary aggregator, not opened directly.
- **[known]**: supplied by the requester as already known. Not re-verified here.

Caveat on method: many web searches in this session timed out ("search was cut off"). In particular, searches for vLLM dLLM support, dInfer, and a direct arXiv sweep for "dLLM + KV cache quantization" got no results back. So the absence findings below are weaker than they would normally be.

## Q1. Which dLLM papers measure real speed with a quantized KV cache, under which decoding, and what do they report?

### Takeaway
I found almost no dLLM paper that measures real end-to-end speed with a packed low-bit KV cache. BitSieve (known) is the only one identified, and it times a fixed 20 steps per block, which would hide any rise in step count. The other dLLM "quantization speed" numbers come from weight/activation quantization (DLLMQuant) or FFN-only FP8 (LaCache), not from the KV cache. The dLLM KV-cache speed papers (Fast-dLLM, dKV-Cache, FlashDLM, EntropyCache) are about caching or reuse, not quantization.

### Cited Findings
- **BitSieve** [known]: built on Fast-dLLM v2. Speed is measured with 20 fixed denoising steps per block and reported as TPOB (time per output block). Under fixed steps, any rise in the number of steps that quantization would cause under threshold decoding is invisible. I could not find a public page for "BitSieve" by web search, so it is not re-verified here.
- **DLLMQuant (arXiv 2508.14090v2, Aug 2025)**: Table 4 reports speed and memory of "our 4-bit implementation" vs FP16 on an RTX 4090 class setup. Average speedup is "over 1.6×" and memory saving "exceeding 3.2×". Table rows: LLaDA speedup 1.71, memory 15.89 → 4.91 GB (3.24×). LLaDA-1.5 speedup 1.70, memory 3.24×. Dream speedup 1.54, memory 15.89/13.95 → 4.44 GB (3.14×). The quantization is of weights and activations (the paper benchmarks W4A4, e.g. "AWQ suffers a 16% accuracy drop on LLADA under W4A4"). It is **not** KV-cache quantization. The text gives no details on the kernels, the decoding steps, or the generation length used for timing. [V-text: PDF text extracted locally from the arXiv PDF]. — [DLLMQuant](https://arxiv.org/abs/2508.14090)
  - Data-quality flag: in the extracted Table 4, the tokens/s columns read "34.59 50.14 1.71" for LLaDA. 50.14/34.59 = 1.45, not 1.71, so either the column parse is garbled or the table is internally inconsistent. Do not cite the tokens/s figures without checking the rendered PDF.
- **LaCache (arXiv 2607.16339, 2026)**: exact (lossless) caching (EmbedCache, RoPECache, FACache) plus "per-group FP8 quantization strategy for FFN layers". It does **not** quantize the KV cache. Reported results: about 1.3× end-to-end speedup over vanilla dLLM, and up to 40.2× when combined with existing accelerators. It targets semi-autoregressive (block) decoding. Hardware and step schedule are not on the abstract page. [V-abs] — [LaCache arXiv](https://arxiv.org/abs/2607.16339)
- **Fast-dLLM (arXiv 2505.22618)**: block-wise approximate KV cache plus confidence-threshold parallel decoding. Up to 27.6× throughput. The KV cache alone gives 2×–3.6×. No quantization. [2nd] — [Fast-dLLM](https://arxiv.org/abs/2505.22618)
- **dKV-Cache (NeurIPS 2025)**: 2–10× speedup on 7B LLaDA and Dream from delayed KV caching. No quantization. [2nd] — [dKV-Cache](https://arxiv.org/abs/2505.15781)
- **FlashDLM / FreeCache (ICLR 2026)**: up to 34× end-to-end with guided diffusion. Caching, not quantization. [2nd] — [FlashDLM](https://iclr.cc/virtual/2026/poster/10010125)
- **MaskKV (on LLaDA)**: KV *eviction*, not quantization. 256 KV pairs (<5% of tokens) keep 94% of full-cache LongBench performance. [2nd] — [arXiv 2510.09309](https://arxiv.org/abs/2510.09309v1)
- **dLLM-Serve (arXiv 2512.17077)**: a serving system that jointly budgets activation and KV memory and uses a head-centric *sparse* KV cache. No quantization found. [2nd] — [dLLM-Serve](https://arxiv.org/abs/2512.17077)
- **Systematic PTQ study of dLLMs (arXiv 2508.14896)**: covers weights and activations on LLaDA and Dream. No KV cache. [2nd] — [arXiv 2508.14896](https://arxiv.org/abs/2508.14896)
- Adjacent domain, video diffusion (not dLLM): "Quantized Keys Steal Attention" (May 2026) works on KV-cache quantization in chunk-wise autoregressive *video* diffusion and claims near-BF16 quality at INT2. vLLM-Omni has an experimental `diffusion_kv_cache_dtype` FP8 option, implemented only for the NPU Flash Attention backend, with NVIDIA GPUs marked unsupported. [2nd / V-text for vLLM-Omni docs via search snippet] — [arXiv 2605.26266](https://arxiv.org/abs/2605.26266); [vLLM-Omni quantized KV](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/quantization/quantized_kvcache/)

### Inferences
- The space "dLLM + packed low-bit KV + real timing under threshold (adaptive) decoding" appears empty as of Nov 2026, based on the searches that returned results. BitSieve times fixed steps. DLLMQuant times W4A4-type weight/activation quantization and does not state its step schedule. So no paper shows whether KV quantization raises the number of denoising steps under threshold decoding, or what that costs in wall time.
- DLLMQuant's 1.5–1.7× speedups come from weight/activation compression, so they are not evidence about KV-cache kernels.

### Gaps
- I could not confirm a public source for BitSieve. Direct arXiv searches for "dLLM KV cache quantization" timed out repeatedly, so a 2026 paper could have been missed.
- The kernel type and step schedule behind DLLMQuant's Table 4 are not stated in the text I extracted.

## Q2. Which serving engines support quantized KV for dLLMs, and has anyone reported correctness problems?

### Takeaway
LMDeploy's PyTorch engine is the only engine I found that officially marks a dLLM (SDAR, 1.7B–30B) as supporting KV INT8, with INT4 marked "No". Its default SDAR recipe uses threshold decoding. SGLang (LLaDA 2.x), JetEngine (SDAR/LLaDA) and the SDAR README document no KV quantization for dLLMs. I found no public bug report about quantized KV with block diffusion in any engine, and no benchmark of it either.

### Cited Findings
- **LMDeploy supported-models page (latest)**: PyTorchEngine on CUDA lists the row "SDAR | 1.7B-30B | LLM | FP16/BF16 Yes | KV INT8 Yes | KV INT4 No | W8A8 - | W4A16 -". The row has no footnote. No LLaDA row, and no other dLLM. [V-text] — [LMDeploy supported models](https://lmdeploy.readthedocs.io/en/latest/supported_models/supported_models.html)
- **LMDeploy KV quant doc (latest)**: classic `quant_policy` 4/8 (int4/int8, per-head per-token asymmetric) examples use TurboMind. TurboQuant `quant_policy=42` is "PytorchEngine only". It is unsupported with MLA and with speculative decoding, and needs a power-of-2 head_dim. SDAR and diffusion are not mentioned. [V-text] — [LMDeploy KV quant](https://lmdeploy.readthedocs.io/en/latest/quantization/kv_quant.html)
- **LMDeploy PR #4563 "FP8 kv cache quantization"**: merged 2026-05-29. Adds `--quant-policy fp8` (E4M3) and `fp8_e5m2` to the PyTorch backend's paged attention, with a scalar per-tensor K/V scale. On Qwen3.5-35B-A3B (TP2, ShareGPT) vs BF16 KV, E4M3 throughput ranged from flat to +15% and E5M2 from -3.7% to +14.4%. dLLM and SDAR are not mentioned. [V-text] — [lmdeploy PR 4563](https://github.com/InternLM/lmdeploy/pull/4563)
- **SDAR official README**: says SDAR has had LMDeploy deployment since 2025-10-10 ("industrial-grade inference solution"). The example config uses `dllm_unmasking_strategy="low_confidence_dynamic"`, `dllm_block_length=4`, `dllm_denoising_steps=4`, `dllm_confidence_threshold=0.9`, `dtype="float16"`, `cache_max_entry_count=0.8`. There is **no `quant_policy`** in the example. So LMDeploy's default dLLM path is threshold-based (adaptive), where any step increase from KV quantization would show up in wall time. JetEngine is reported at 1800+ tok/s on A800 and 3700+ tok/s on H200 (4B, block 4, batch 128), with no quantization. [V-text] — [JetAstra/SDAR](https://github.com/JetAstra/SDAR)
- **JetEngine (Labman42/JetEngine)**: a lightweight engine for block diffusion (SDAR, LLaDA, dLLM-Var). Supports dense and MoE, DP+TP, CUDA graphs, flash-attn. No KV quantization is mentioned in the aggregator summaries. The GitHub page fetch timed out. [2nd] — [JetEngine](https://github.com/Labman42/JetEngine)
- **SGLang dLLM support (LMSYS blog, 2025-12-19)**: built on the chunked-prefill pipeline. Supports LLaDA-2.0-mini/flash, `--dllm-algorithm LowConfidence` (threshold 0.95 in the CAP benchmark), "full KV cache support", TP, and CUDA graphs. Benchmarks on 8×H20 TP8: LLaDA2.0-flash-CAP (BF16) at 935 tok/s on a streaming code task. On HumanEval, MBPP, GSM8K and CRUXEval it reaches 500 TPS vs 383 TPS for LLaDA2.0-flash, up to 1.9× over AR baselines at small batch. No mention of KV dtype, FP8 or quantized KV for dLLMs. Roadmap: Fast-dLLM v2 decoding, non-block dLLMs (LLaDA, RND1). [V-text] — [LMSYS blog](https://lmsys.org/blog/2025-12-19-diffusion-llm)
- **SGLang** has a LLaDA-2.1 cookbook (H100/H200/B200, MI300X/MI325X/MI355X) and a DiffusionGemma cookbook (`--dllm-algorithm Gemma4Renoise`). PR #19020 (merged 2026-02-20) fixed SGLang **skipping KV-cache saving for dLLMs** ("DLLM needs to attend to kvcache"). That was a KV-cache correctness bug in the dLLM path, but not a quantization one. [2nd] — [SGLang LLaDA-2.1 cookbook](https://docs.sglang.io/cookbook/autoregressive/InclusionAI/LLaDA-2.1); [sglang PR 19020](https://github.com/sgl-project/sglang/pull/19020)
- **DLMRT**: no result for this name. I cannot confirm it exists or that it has FP8 KV. [no source]
- **vLLM**: no evidence found of native dLLM (text) serving with quantized KV. The searches timed out. vLLM-Omni's diffusion FP8 "KV" option is for image and video diffusion, and NPU only (see Q1). [2nd]

### Inferences
- LMDeploy SDAR + `quant_policy=8` is the one documented production path for a quantized-KV dLLM. It runs with threshold decoding by default (`low_confidence_dynamic`, threshold 0.9), so it is a natural setup in which a step increase would be visible. No public benchmark of it was found.
- The "KV INT8 Yes" flag probably means only that the PyTorch-engine paged-attention int8 path is not blocked for the SDAR model class. Neither the doc nor any issue shows an accuracy or speed validation for dLLMs. (This is an inference.)

### Gaps
- No LMDeploy, SGLang or JetEngine issue about quantized KV with SDAR or LLaDA was found. Targeted issue searches either returned nothing or timed out.
- I did not verify whether LMDeploy's FP8 KV (PR #4563) or TurboQuant (policy 42) works with the dLLM path. The supported-models table only lists INT8/INT4.
- dInfer (inclusionAI) and DLMRT could not be checked because the searches timed out.

## Q3. What does general-LLM literature say about when low-bit KV actually speeds up decoding?

### Takeaway
Low-bit KV speeds things up mainly when decoding is memory-bound on KV reads (long context, large batch) or when cache capacity limits batching. At short context or small batch, dequantization overhead often makes it break even or run slower. Examples: TurboQuant gives -7.3% (LMDeploy) and -20 to -34% throughput (vLLM). KIVI and QServe kernels fall below FP16 on A100 (BitDecoding's measurements). Plain FP8 KV is roughly free.

### Cited Findings
- **LMDeploy TurboQuant (`quant_policy=42`)**, H200, Qwen3-30B-A3B-Base, ShareGPT, concurrency 64, 5000 requests. Input/output throughput 2368.8/2186.7 → 2195.8/2027.0 tok/s (-7.3%). Request throughput 10.74 → 9.96 req/s. Mean E2E latency 5.888 → 6.348 s (+7.8%). TTFT +8.4%. TPOT 0.024 → 0.026 s (+8.3%). ITL unchanged. Summarized as about 5× KV memory reduction for about 7–8% overhead. [V-text] — [LMDeploy KV quant](https://lmdeploy.readthedocs.io/en/latest/quantization/kv_quant.html)
- **LMDeploy classic int8/int4 KV (TurboMind)**, throughput mode with batch 128/256 on 10k prompts. llama2-chat-7b RPS 14.98 → 19.01 (int8, 1.27×) → 20.81 (int4, 1.39×). llama2-13b 1.28× / 1.39×. internlm2-chat-7b only 1.05× / 1.07×. The gains come from about 2×/4× more KV blocks, so they are a capacity effect at high batch. GPU not stated. [V-text] — same page
- **vLLM TurboQuant study (2026-05-11)**, vLLM 0.20.2, Qwen3-30B-A3B on 2×H100 and Llama-3.3-70B on 4×H100.
  - Latency at batch 1–64 (in 1024 / out 256): FP8 KV adds negligible overhead. TQ is 10–60% slower (Qwen) and 10–68% slower (Llama-70B), and the overhead grows with batch.
  - Offline throughput: FP8 equals BF16. TQ reaches 73–80% (Qwen) and 66–75% (Llama) of BF16.
  - Online burst: TQ TPOT is 1.5–2.5× worse.
  - TQ helps only in the memory-saturated case: P99 TTFT below 3.5 s vs about 17 s for BF16. FP8 is best at about 1.3 s.
  [V-text] — [vLLM blog: TurboQuant](https://vllm.ai/blog/turboquant)
- **BitDecoding (arXiv 2503.18773; HPCA-track systems paper)**: reports that on A100, "KIVI and QServe perform worse than the FP16 baseline" (KIVI because of non-fused kernels, QServe because of Tensor Core underuse). On RTX 4090, KIVI shows "limited speedup in MHA and severe degradation in GQA". Atom/QServe dequantization takes "nearly half the kernel execution time", vs <15% (4-bit) and <35% (2-bit) for BitDecoding. BitDecoding's own results: up to 7.5× (4090), 4.8× (A100), 8.9× (H100) kernel speedup vs FlashDecoding-v2. On Llama-3.1-8B at 32K, INT4 gives 2.98× throughput. At 128K with batch 1, latency is about 3× lower. The half-precision residual cache adds an extra kernel launch, "negligible as sequence length grows". These are long-context results. The text gives no short-context numbers. [V-text for v2 HTML; v3 numbers from abstract = V-abs] — [BitDecoding](https://arxiv.org/abs/2503.18773)
- **KIVI (ICML 2024)**: Llama-2-7B on 1×A100-80GB. With batch pushed to OOM, KIVI allows up to 4× larger batch and 2.35–3.47× throughput (prompt 161, output 338). Peak memory is 2.6× lower. The speedup comes from capacity and batch size, and "can grow larger with longer context length and output length". [2nd] — [KIVI](https://arxiv.org/abs/2402.02750)
- **DiffKV (arXiv 2412.03131)**: K8V8 attention kernel reaches 1.7× vs a theoretical 2×. The gap "primarily due to the overhead of accessing quantization metadata and performing dequantization". [2nd] — [DiffKV](https://arxiv.org/abs/2412.03131)
- **InnerQ (arXiv 2602.23200)**: fused dequantize-GEMV kernels. Average 1.3× over KIVI and 1.2× over TurboQuant (Jetson Xavier NX). [2nd] — [InnerQ](https://arxiv.org/abs/2602.23200)
- **KVQuant (NeurIPS 2024)**: argues KV loading is memory-bandwidth-bound, so the cost of non-uniform dequantization is small with custom kernels. [2nd] — [KVQuant](https://www.stat.berkeley.edu/~mmahoney/pubs/neurips-2024-kvquant.pdf)
- **LMDeploy FP8 KV (PR #4563)**: from flat to +15% throughput vs BF16 on Qwen3.5-35B-A3B. [V-text] — [PR 4563](https://github.com/InternLM/lmdeploy/pull/4563)

### Inferences
- The general-LLM evidence follows one pattern. Low-bit KV gains come from (a) more batch capacity, or (b) fewer bytes per KV read at long context, with a well-fused Tensor Core kernel. At small batch and short context, dequantization overhead dominates, and generic int4/2-bit paths (KIVI, QServe, TurboQuant) run at or below FP16. FP8 is the near-free exception.
- dLLM block decoding behaves like small-batch, short-context work in this sense. Each denoising step attends a block of 4–32 query tokens over a prefix that is usually short (GSM8K-scale). Several query tokens per KV read raise arithmetic intensity relative to AR decoding, which further shrinks the bandwidth benefit of a smaller KV. So the AR evidence predicts little or negative wall-time gain from low-bit KV in typical dLLM benchmarks unless context is long or batch is large. Under threshold decoding, any rise in step count from quantization noise would add to that. This is an inference, not measured by any source found.

### Gaps
- No source measures low-bit KV with multi-token (block) queries per KV read, which is the dLLM case. Speculative-decoding and verification-kernel literature might, but it was not searched.
- The ROCm/AMD TurboQuant blog and short-context breakdowns timed out or were not reported.
