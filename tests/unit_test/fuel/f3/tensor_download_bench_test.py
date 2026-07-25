# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

import pytest
import torch

from dev_tools.f3.tensor_download_bench import (
    MODE_DISK,
    MODE_MEMORY,
    build_transfer_payload,
    parse_modes,
    select_tensors,
    tensor_nbytes,
    unique_tensor_stats,
    unwrap_state_dict,
    validate_received_payload,
)


class FakeLazyTensorRef:
    def __init__(self, tensor, file_path=None, key=None):
        self.tensor = tensor
        self.file_path = file_path
        self.key = key

    def materialize(self):
        return self.tensor


def test_parse_modes():
    assert parse_modes("memory,disk") == (MODE_MEMORY, MODE_DISK)
    assert parse_modes(" disk, memory, disk ") == (MODE_DISK, MODE_MEMORY)
    with pytest.raises(argparse.ArgumentTypeError):
        parse_modes("gpu")


def test_unwrap_and_select_tensors_by_binary_size():
    state_dict = {
        "large": torch.zeros(32, dtype=torch.float32),
        "medium": torch.zeros(8, dtype=torch.float32),
        "small": torch.zeros(2, dtype=torch.float32),
        "metadata": "ignored",
    }
    assert unwrap_state_dict({"state_dict": state_dict}) is state_dict

    selected = select_tensors(state_dict, max_bytes=40)
    assert list(selected) == ["small", "medium"]
    assert sum(tensor_nbytes(tensor) for tensor in selected.values()) == 40


def test_aliases_only_count_once_toward_selection_limit():
    shared = torch.zeros(8, dtype=torch.float32)
    state_dict = {
        "a_shared": shared,
        "b_shared": shared,
        "z_other": torch.zeros(8, dtype=torch.float32),
    }

    selected = select_tensors(state_dict, max_bytes=32)
    assert list(selected) == ["a_shared", "b_shared"]
    assert unique_tensor_stats(selected) == (1, 32)


def test_memory_and_disk_payload_validation(tmp_path):
    tensors = {
        "weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "bias": torch.tensor([1.0, 2.0]),
    }
    payload = build_transfer_payload(tmp_path / "model.pt", tensors)

    memory_result = validate_received_payload(payload, MODE_MEMORY)
    assert memory_result["tensor_count"] == 2
    assert memory_result["tensor_bytes"] == 56
    assert memory_result["transfer_bytes"] == 56

    disk_payload = {**payload, "tensors": {key: FakeLazyTensorRef(value) for key, value in tensors.items()}}
    disk_result = validate_received_payload(disk_payload, MODE_DISK)
    assert disk_result["tensor_count"] == 2
    assert disk_result["sample_materialized_bytes"] == 8


def test_disk_validation_counts_aliases_by_file_and_key(tmp_path):
    shared = torch.arange(4, dtype=torch.float32)
    tensors = {"shared_a": shared, "shared_b": shared}
    payload = build_transfer_payload(tmp_path / "model.pt", tensors)
    payload["tensors"] = {
        key: FakeLazyTensorRef(shared, file_path="/tmp/chunk.safetensors", key="T0") for key in tensors
    }

    result = validate_received_payload(payload, MODE_DISK)
    assert result["tensor_count"] == 2
    assert result["unique_tensor_count"] == 1
