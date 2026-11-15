import pytest

from blockkv.lmdeploy_patch import EDITS, MARK, patch_text


def fake_source():
    return "\n#--\n".join(old for old, _, n in EDITS for _ in range(n))


def test_patch_replaces_every_site_and_marks_the_file():
    out = patch_text(fake_source())
    assert MARK in out and "HEAD_PER_CTA" not in out
    assert out.count("cur_token[:, None] * stride_qbs") == 2
    assert out.count("off_meta = (cur_token * stride_obs") == 3
    assert out.count("seq_len=seq_len,") == 1 and out.count("seq_len: tl.constexpr,") == 1


def test_patch_refuses_a_different_file_or_a_second_run():
    with pytest.raises(ValueError, match="expected 2"):
        patch_text(fake_source().replace("(cur_batch * stride_qbs + cur_head[:, None]", "x", 1))
    with pytest.raises(ValueError, match="already patched"):
        patch_text(patch_text(fake_source()))
