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

import os
import shutil
import tempfile

import pytest
import torch
from safetensors.torch import save as save_tensors

from nvflare.app_opt.pt.lazy_tensor_dict import LazyTensorDict
from nvflare.app_opt.pt.tensor_downloader import DiskTensorConsumer, _extract_safetensors_keys


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp(prefix="nvflare_test_disk_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


class TestExtractSafetensorsKeys:
    def test_single_key(self):
        data = save_tensors({"weight": torch.randn(3, 3)})
        keys = _extract_safetensors_keys(data)
        assert keys == ["weight"]

    def test_multiple_keys(self):
        data = save_tensors({"a": torch.randn(2), "b": torch.randn(2)})
        keys = _extract_safetensors_keys(data)
        assert set(keys) == {"a", "b"}

    def test_invalid_data(self):
        with pytest.raises(ValueError, match="too short"):
            _extract_safetensors_keys(b"short")

    def test_header_size_exceeds_payload(self):
        # Header says 100 bytes of JSON, but payload only has 2.
        data = (100).to_bytes(8, byteorder="little") + b"{}"
        with pytest.raises(ValueError, match="header size exceeds payload length"):
            _extract_safetensors_keys(data)

    def test_invalid_json_header(self):
        data = (4).to_bytes(8, byteorder="little") + b"nope"
        with pytest.raises(ValueError, match="invalid JSON header"):
            _extract_safetensors_keys(data)

    def test_non_object_json_header(self):
        data = (2).to_bytes(8, byteorder="little") + b"[]"
        with pytest.raises(ValueError, match="header must be JSON object"):
            _extract_safetensors_keys(data)


class TestDiskTensorConsumer:
    def test_writes_to_disk(self, temp_dir):
        consumer = DiskTensorConsumer(temp_dir)

        t1 = {"layer.weight": torch.randn(4, 3)}
        t2 = {"layer.bias": torch.randn(4)}
        items = [save_tensors(t1), save_tensors(t2)]

        result = consumer.consume_items(items, None)

        assert isinstance(result, dict)
        assert "layer.weight" in result
        assert "layer.bias" in result

        # Files exist on disk
        for file_path, key in result.values():
            assert os.path.exists(file_path)

    def test_key_to_file_mapping(self, temp_dir):
        consumer = DiskTensorConsumer(temp_dir)

        tensors = {"w1": torch.randn(2), "w2": torch.randn(3)}
        items = [save_tensors({"w1": tensors["w1"]}), save_tensors({"w2": tensors["w2"]})]

        result = consumer.consume_items(items, None)

        # Each key maps to a different file
        file1, _ = result["w1"]
        file2, _ = result["w2"]
        assert file1 != file2

    def test_no_tensor_deserialization(self, temp_dir):
        """Verify bytes written to disk match input (no deserialization)."""
        consumer = DiskTensorConsumer(temp_dir)

        original_bytes = save_tensors({"test": torch.randn(5)})
        result = consumer.consume_items([original_bytes], None)

        file_path, _ = result["test"]
        with open(file_path, "rb") as f:
            disk_bytes = f.read()
        assert disk_bytes == original_bytes

    def test_accumulates_across_calls(self, temp_dir):
        consumer = DiskTensorConsumer(temp_dir)

        batch1 = [save_tensors({"a": torch.randn(2)})]
        batch2 = [save_tensors({"b": torch.randn(3)})]

        result = consumer.consume_items(batch1, None)
        result = consumer.consume_items(batch2, result)

        assert set(result.keys()) == {"a", "b"}

    def test_download_failed_cleanup(self, temp_dir):
        consumer = DiskTensorConsumer(temp_dir)

        # Write some data first
        items = [save_tensors({"x": torch.randn(2)})]
        consumer.consume_items(items, None)
        assert os.path.exists(temp_dir)

        # Simulate download failure
        consumer.download_failed("ref123", "timeout")
        assert not os.path.exists(temp_dir)
        assert consumer.error == "timeout"

    def test_result_builds_lazy_tensor_dict(self, temp_dir):
        """Verify DiskTensorConsumer result can be used to create a LazyTensorDict."""
        consumer = DiskTensorConsumer(temp_dir)

        original = {"param": torch.randn(3, 2)}
        items = [save_tensors(original)]
        result = consumer.consume_items(items, None)

        ltd = LazyTensorDict(key_to_file=result, temp_dir=temp_dir)
        loaded = ltd["param"]
        assert torch.allclose(loaded, original["param"])
