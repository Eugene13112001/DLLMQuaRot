"""Fix LMDeploy 0.18.0's quantized paged-attention kernel for block diffusion (several query
tokens per request), in the installed package.

    python -m blockkv.lmdeploy_patch check     # are the expected lines there? (no change)
    python -m blockkv.lmdeploy_patch apply     # backup to pagedattention.py.orig, then patch
    python -m blockkv.lmdeploy_patch revert    # restore the backup

The bug. In decoding, a dLLM sends the whole block (e.g. 16 tokens) per request. LMDeploy's
plain kernel `_fwd_grouped_split_kernel` handles that: its program tiles run over
kv_group_num * seq_len query rows and `cur_token = cur_batch * seq_len + offs_h // kv_group_num`.
The kernel used whenever the KV cache is quantized, `_fwd_grouped_split_quant_kernel`, has no
seq_len: it loads and stores only row `cur_batch`, so with a block of B tokens and batch 1 only the
first token of the block gets an attention output and the others read scratch memory. Every
quant_policy (int4, int8, fp8, TurboQuant) gives garbage on SDAR, while policy 0 is fine.

The fix gives the quant kernel the plain kernel's tiling, word for word: the seq_len argument,
the tile -> (kv head, query row) mapping, the row masks, and cur_token instead of cur_batch in the
query loads and in the scratch stores. The dequantization and the attention loop are untouched.
With seq_len = 1 (autoregressive decoding) the mapping reduces to the old one.
"""
import shutil
import sys
from pathlib import Path

OLD_HEADER = """    cur_batch = tl.program_id(2)
    cur_kv_head = tl.program_id(0)
    split_k_id = tl.program_id(1)

    if BLOCK_H < kv_group_num:
        HEAD_PER_CTA: tl.constexpr = BLOCK_H
    else:
        HEAD_PER_CTA: tl.constexpr = kv_group_num
    cur_head = cur_kv_head * HEAD_PER_CTA + tl.arange(0, BLOCK_H)
    mask_h = cur_head < cur_kv_head * HEAD_PER_CTA + HEAD_PER_CTA
    mask_h = mask_h & (cur_head < num_heads_q)
    if BLOCK_H < kv_group_num:
        cur_kv_head = (cur_kv_head * HEAD_PER_CTA) // kv_group_num
"""
NEW_HEADER = """    cur_batch = tl.program_id(2)
    tile_id = tl.program_id(0)
    split_k_id = tl.program_id(1)

    # blockkv patch: several query tokens per request, tiled as in _fwd_grouped_split_kernel
    HEADS_PER_REQ: tl.constexpr = kv_group_num * seq_len
    TILES_PER_GROUP: tl.constexpr = tl.cdiv(HEADS_PER_REQ, BLOCK_H)
    subtile_id = tile_id % TILES_PER_GROUP
    cur_kv_head = tile_id // TILES_PER_GROUP
    offs_h = subtile_id * BLOCK_H + tl.arange(0, BLOCK_H)
    cur_head = cur_kv_head * kv_group_num + offs_h % kv_group_num
    cur_token = cur_batch * seq_len + offs_h // kv_group_num

    mask_h = cur_head < cur_kv_head * kv_group_num + kv_group_num
    mask_h = mask_h & (cur_token < cur_batch * seq_len + seq_len)
    mask_h = mask_h & (cur_head < num_heads_q)
"""

# (old, new, expected count in the original file)
EDITS = [
    ("kv_group_num: tl.constexpr,\n    window_size: tl.constexpr,",
     "kv_group_num: tl.constexpr,\n    seq_len: tl.constexpr,\n    window_size: tl.constexpr,", 1),
    (OLD_HEADER, NEW_HEADER, 1),
    ("(cur_batch * stride_qbs + cur_head[:, None]", "(cur_token[:, None] * stride_qbs + cur_head[:, None]", 2),
    ("off_acc = (cur_batch * stride_obs", "off_acc = (cur_token[:, None] * stride_obs", 1),
    ("off_meta = (cur_batch * stride_obs", "off_meta = (cur_token * stride_obs", 3),
    ("kv_group_num=kv_group_num,\n                                              window_size=window_size,",
     "kv_group_num=kv_group_num,\n                                              seq_len=seq_len,\n"
     "                                              window_size=window_size,", 1),
]
MARK = "# blockkv patch:"


def target():
    import importlib.util
    spec = importlib.util.find_spec("lmdeploy")
    if spec is None:
        raise SystemExit("lmdeploy is not installed in this python")
    return Path(spec.origin).parent / "pytorch" / "kernels" / "cuda" / "pagedattention.py"


def patch_text(src):
    """Patched source, or raise if the file is not the one the patch was written for."""
    if MARK in src:
        raise ValueError("already patched")
    for old, new, n in EDITS:
        found = src.count(old)
        if found != n:
            raise ValueError(f"expected {n} x {old[:60]!r}, found {found}")
        src = src.replace(old, new)
    if "HEAD_PER_CTA" in src:
        raise ValueError("HEAD_PER_CTA left over")
    return src


def main(argv=None):
    cmd = (argv or sys.argv[1:] or ["check"])[0]
    path = target()
    backup = path.with_name(path.name + ".orig")
    src = path.read_text(encoding="utf-8")
    if cmd == "check":
        if MARK in src:
            print(f"patched: {path}")
        else:
            patch_text(src)
            print(f"not patched, patch applies cleanly: {path}")
    elif cmd == "apply":
        new = patch_text(src)
        shutil.copy2(path, backup)
        path.write_text(new, encoding="utf-8")
        print(f"patched {path} (original in {backup.name})")
    elif cmd == "revert":
        if not backup.exists():
            raise SystemExit(f"no backup {backup}")
        shutil.copy2(backup, path)
        print(f"restored {path}")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
