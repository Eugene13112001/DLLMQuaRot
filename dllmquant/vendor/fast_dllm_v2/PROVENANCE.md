# Fast-dLLM-v2 modelling code, vendored

Copied byte-for-byte from the checkpoint's own repository. Nothing here is edited,
for the same reason as `llada2_moe`: the quantized cache has to sit inside attention,
and remote code fetched per run can change between results.

| | |
|---|---|
| Source | https://huggingface.co/Efficient-Large-Model/Fast_dLLM_v2_7B |
| Revision | `0661abf5f9f0ee338970d091052a26c8efa51974` |
| Fetched | 2026-09-20 |

    sha256  760e76af3a1fdf23e70868eea0d221aae186b41eafdfc35222e79e1d14b22f4c  configuration.py
    sha256  7eced912363bdfa22bb5618b68b680a9e8c6c89d775ad5debc116cc0e733d274  modeling.py

`tests/test_vendor.py` checks these hashes, so an accidental edit fails a test
instead of quietly becoming a fork nobody declared.

## What this model is here for

It is a block dLLM without QK-Norm and with a bias on `k_proj`, so it separates the
two candidate sources of fixed-channel key outliers that LLaDA2.0-mini confounds.
Its sampler is also the one the BitSieve tables use: blocks of 32, sub-blocks of 8,
commit by confidence.
