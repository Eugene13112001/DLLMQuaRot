# Third-party components

The code written for this paper is released under the Apache License 2.0 (`LICENSE`). The following files come
from other projects and keep their original licenses.

| Files | Origin | License |
|---|---|---|
| `accuracy/sdar/generate.py` | SDAR official block-diffusion generation code (JetAstra/SDAR), unmodified | MIT, Copyright (c) 2025 JetAstra (`accuracy/sdar/LICENSE.SDAR-generate`) |
| `accuracy/longbench/config/dataset2prompt.json`, `dataset2maxlen.json` | LongBench (THUDM/LongBench), unmodified | MIT, Copyright (c) 2023 THU-KEG & Zhipu AI (`accuracy/longbench/LICENSE`) |
| `accuracy/ruler/official/` (`synthetic.yaml`, `metrics.py`, `postprocess.py`) | NVIDIA RULER: task configuration and scoring, copied without changes (`postprocess.py` is an excerpt) | Apache-2.0 (`accuracy/ruler/official/LICENSE`) |
| `accuracy/sdar/sdar_model/modeling_sdar.py` | SDAR model code from JetLM/SDAR-8B-Chat (revision ac4528d2), modified: version-tolerant imports and SDPA for decoding attention (`modeling_sdar.patch` lists the changes) | Apache-2.0 |
| `serving/models/sdar-runtime/modeling_sdar.py`, `configuration_sdar.py` | SDAR model code distributed with JetLM/SDAR-8B-Chat (derived from Qwen3 / Hugging Face Transformers); `modeling_sdar.py` modified to import FlashAttention-4 and the RMSNorm below (see `serving/README.md`) | Apache-2.0 |
| `serving/models/sdar-runtime/sdar_norm.py` | FlashAttention Triton RMSNorm, unmodified | BSD-3-Clause, Copyright (c) 2024 Tri Dao (`serving/models/sdar-runtime/LICENSE.flash-attention`) |

LongBench metrics are re-implemented inside the harnesses and use the `rouge` and `fuzzywuzzy`/`python-Levenshtein`
packages, which are installed from PyPI and not redistributed here.

Models and datasets are not included. They are downloaded from their original sources: JetLM/SDAR-8B-Chat,
Gen-Verse/TraDo-8B-Instruct, nvidia/Nemotron-Labs-Diffusion-8B (Hugging Face), LongBench
(THUDM/LongBench `data.zip`), WikiText-2 (`wikitext`, `wikitext-2-raw-v1`), and the RULER task sources.
