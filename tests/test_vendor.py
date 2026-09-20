"""The vendored copies must stay copies.

Editing one in place would be the easy way to add the KV cache hook, and it
would also make `diff` against a future checkpoint revision useless and leave
this project's changes to the model indistinguishable from the vendor's code.
The hashes recorded in PROVENANCE.md are what keeps that decision honest.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "dllmquant" / "vendor"
VENDORED = {
    "llada2_moe": "dllmquant.vendor.llada2_moe",
    "fast_dllm_v2": "dllmquant.vendor.fast_dllm_v2",
}


def _recorded_hashes(vendor: Path) -> dict:
    text = (vendor / "PROVENANCE.md").read_text(encoding="utf-8")
    return {
        name: digest
        for digest, name in re.findall(r"sha256\s+([0-9a-f]{64})\s+(\S+)", text)
    }


@pytest.mark.parametrize("name", sorted(VENDORED))
def test_provenance_records_every_vendored_file(name):
    vendor = ROOT / name
    recorded = _recorded_hashes(vendor)
    on_disk = {p.name for p in vendor.glob("*.py") if p.name != "__init__.py"}
    assert on_disk == set(recorded), (
        f"{name}: PROVENANCE.md and the directory disagree about what was vendored"
    )


@pytest.mark.parametrize("name", sorted(VENDORED))
def test_vendored_files_are_unmodified(name):
    vendor = ROOT / name
    for file, recorded in _recorded_hashes(vendor).items():
        actual = hashlib.sha256((vendor / file).read_bytes()).hexdigest()
        assert actual == recorded, (
            f"{file} no longer matches the hash in PROVENANCE.md. If the change is "
            "deliberate, it belongs in a local module (llada2_local.py is the "
            "precedent); if the revision was refreshed, update PROVENANCE.md."
        )


@pytest.mark.parametrize("name,module", sorted(VENDORED.items()))
def test_the_revision_is_pinned_in_one_place(name, module):
    import importlib

    revision = importlib.import_module(module).REVISION
    assert re.fullmatch(r"[0-9a-f]{40}", revision)
    assert revision in (ROOT / name / "PROVENANCE.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("name", sorted(VENDORED))
def test_line_endings_are_lf(name):
    """The hashes are byte hashes, and this repository is edited on Windows."""
    vendor = ROOT / name
    for file in _recorded_hashes(vendor):
        assert b"\r\n" not in (vendor / file).read_bytes()
