from pathlib import Path

from blockkv.summarize_ruler import ref_key


def test_runs_pair_within_model_block_and_rms():
    assert ref_key(Path("fastdllm_b32_kv2.jsonl")) == ref_key(Path("fastdllm_b32.jsonl"))
    assert ref_key(Path("sdar8b_b64_kv2_w128.jsonl")) == ref_key(Path("sdar8b_b64.jsonl"))
    assert ref_key(Path("sdar8b_b32_kv2.jsonl")) != ref_key(Path("fastdllm_b32.jsonl"))
    assert ref_key(Path("sdar8b_b16_kv2_rms.jsonl")) == ref_key(Path("sdar8b_b16_rms.jsonl"))
