# LLaDA2.0 modelling code, vendored

Copied byte-for-byte from the checkpoint's own repository. Nothing in this
directory is edited — see "Why verbatim" below.

| | |
|---|---|
| Source | https://huggingface.co/inclusionAI/LLaDA2.0-mini |
| Revision | `dad945cac317da394b390f82c7b40691d8a881ed` |
| License | Apache 2.0 (Antgroup and the HuggingFace team; header retained) |
| Fetched | 2026-08-10 |

    sha256  375a2116aa72aae697fe240486abe1363927f68fa355c744ef4270fa1cec52c3  configuration_llada2_moe.py
    sha256  b9cac6e6f46473ed6aa5785ac57c1931ae8a68499b1149cf9e5c39b12015f47f  modeling_llada2_moe.py

`tests/test_vendor.py` checks these hashes, so an accidental edit to the copy
fails a test instead of quietly becoming a fork nobody declared.

## Why vendor at all

Two things this project has to do live *inside* attention, and remote code
cannot be reached from outside it:

* the quantized KV cache has to sit between the K/V projections and the
  attention call, which is the entire research question here;
* R4 rotates Q and K after RoPE, which by construction cannot be folded into
  the projection weights.

A checkpoint's remote code is also not a fixed target: it is downloaded at
load time and can change under a rerun, which would move measured numbers
with no commit to point at. Pinning the revision here makes the model's code
part of the experiment record.

## Why verbatim, with changes kept elsewhere

Edits go in `dllmquant/models/llada2_local.py`, as subclasses, not into these
files. That keeps two properties worth more than the convenience of editing
in place:

* `diff` against upstream stays meaningful — a new revision of the checkpoint
  can be compared against this one directly;
* what this project changed about the model is readable in one short file
  instead of being scattered through 1400 lines of someone else's code.

## Refreshing

    curl -fL -o modeling_llada2_moe.py \
      https://huggingface.co/inclusionAI/LLaDA2.0-mini/resolve/<rev>/modeling_llada2_moe.py

then update the revision and both hashes above, and rerun the selfcheck with
`--rotate --dtype float32`: the invariance number is the thing most likely to
notice a change in how attention is computed.
