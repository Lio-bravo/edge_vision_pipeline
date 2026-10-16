#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 edge_vision_pipeline contributors
import pytest

from evp.errors import ConfigError
from evp.utils import resolve_device


@pytest.mark.parametrize("dev", ["auto", "AUTO", "", None, "  auto "])
def test_resolve_device_auto(dev):
    assert resolve_device(dev, cuda_available=True) == "0"
    assert resolve_device(dev, cuda_available=False) == "cpu"


@pytest.mark.parametrize("dev", ["cpu", "CPU", "mps"])
def test_resolve_device_non_cuda_passthrough(dev):
    assert resolve_device(dev, cuda_available=False) == dev.lower()


@pytest.mark.parametrize("dev", ["0", 0, "0,1", "cuda:0"])
def test_resolve_device_explicit_gpu(dev):
    assert resolve_device(dev, cuda_available=True) == str(dev)


@pytest.mark.parametrize("dev", ["0", 0, "cuda:1"])
def test_resolve_device_gpu_without_cuda_is_actionable(dev):
    with pytest.raises(ConfigError) as ei:
        resolve_device(dev, "train.device", cuda_available=False)
    assert "train.device" in str(ei.value) and "auto" in str(ei.value)
