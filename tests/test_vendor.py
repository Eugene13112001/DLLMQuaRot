"""The vendored copy must stay a copy.

Editing it in place would be the easy way to add the KV cache hook, and it
would also make `diff` against a future checkpoint revision useless and leave
this project's changes to the model indistinguishable from Antgroup's code.
The hashes recorded in PROVENANCE.md are what keeps that decision honest.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

VENDOR = Path(__file__).resolve().parents[1] / "dllmquant" / "vendor" / "llada2_moe"
PROVENANCE = VENDOR / "PROVENANCE.md"


def _recorded_hashes() -> dict:
    text = PROVENANCE.read_text(encoding="utf-8")
    return {
        name: digest
        for digest, name in re.findall(r"sha256\s+([0-9a-f]{64})\s+(\S+)", text)
    }


def test_provenance_records_every_vendored_file():
    recorded = _recorded_hashes()
    on_disk = {p.name for p in VENDOR.glob("*.py") if p.name != "__init__.py"}
    assert on_disk == set(recorded), (
        "PROVENANCE.md and the directory disagree about what was vendored"
    )


@pytest.mark.parametrize("name", ["modeling_llada2_moe.py", "configuration_llada2_moe.py"])
def test_vendored_file_is_unmodified(name):
    recorded = _recorded_hashes()[name]
    actual = hashlib.sha256((VENDOR / name).read_bytes()).hexdigest()
    assert actual == recorded, (
        f"{name} no longer matches the hash in PROVENANCE.md. If the change is "
        "deliberate, it belongs in dllmquant/models/llada2_local.py instead; if "
        "the revision was refreshed, update PROVENANCE.md."
    )


def test_the_revision_is_pinned_in_one_place():
    from dllmquant.vendor.llada2_moe import REVISION

    assert re.fullmatch(r"[0-9a-f]{40}", REVISION)
    assert REVISION in PROVENANCE.read_text(encoding="utf-8")


def test_line_endings_are_lf():
    """The hashes are byte hashes, and this repository is edited on Windows."""
    for name in ("modeling_llada2_moe.py", "configuration_llada2_moe.py"):
        assert b"\r\n" not in (VENDOR / name).read_bytes()
