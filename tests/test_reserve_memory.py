"""reserve_device_memory: sizes the claim from the model and never touches CPU."""

import torch
import torch.nn as nn
import pytest

from dllmquant.models.base import reserve_device_memory


def test_cpu_is_a_no_op():
    assert reserve_device_memory(nn.Linear(4, 4), "cpu") == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_reservation_stays_in_the_cache():
    m = nn.Linear(1024, 1024)
    before = torch.cuda.memory_reserved()
    got = reserve_device_memory(m, "cuda", extra_gb=0.01)
    assert got == 1024 * 1024 * 4 + 1024 * 4 + int(0.01 * (1 << 30))
    assert torch.cuda.memory_reserved() - before >= got
