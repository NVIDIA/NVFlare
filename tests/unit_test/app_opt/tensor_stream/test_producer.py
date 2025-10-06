# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import pytest
import torch
from safetensors.torch import load as load_tensors

from nvflare.apis.shareable import ReturnCode, Shareable
from nvflare.app_opt.tensor_stream.producer import TorchTensorsProducer
from nvflare.app_opt.tensor_stream.types import TensorBlobKeys


class TestTorchTensorsProducer:
    """Test cases for TorchTensorsProducer class."""

    @pytest.mark.parametrize(
        "tensors,expected_keys_len,expected_end",
        [
            ({"layer1.weight": torch.randn(2, 3), "layer1.bias": torch.randn(2)}, 2, 2),  # Valid tensors
            ({}, 0, 0),  # Empty tensors
            (None, 0, 0),  # None tensors
        ],
    )
    def test_init_with_various_tensors(self, tensors, expected_keys_len, expected_end):
        """Test initialization with various tensor configurations."""
        entry_timeout = 5.0
        root_key = "model"

        producer = TorchTensorsProducer(tensors=tensors, entry_timeout=entry_timeout, root_key=root_key)

        assert producer.entry_timeout == entry_timeout
        assert producer.root_key == root_key
        assert producer.last is False
        assert producer.tensors == tensors
        assert len(producer.tensors_keys) == expected_keys_len
        assert producer.start == 0
        assert producer.current == 0
        assert producer.end == expected_end
        assert producer.total_bytes == 0

        if tensors:
            assert producer.tensors_keys == list(tensors.keys())

    def test_produce_single_tensor(self, random_torch_tensors, mock_stream_context, mock_fl_context):
        """Test producing a single tensor."""
        # Use only one tensor for this test
        single_tensor = {"test_tensor": random_torch_tensors["layer1.weight"]}
        producer = TorchTensorsProducer(tensors=single_tensor.copy(), entry_timeout=5.0, root_key="model")

        shareable, timeout = producer.produce(mock_stream_context, mock_fl_context)

        assert timeout == 5.0
        assert isinstance(shareable, Shareable)
        assert TensorBlobKeys.SAFETENSORS_BLOB in shareable
        assert TensorBlobKeys.TENSOR_KEYS in shareable
        assert TensorBlobKeys.ROOT_KEY in shareable

        # Verify the tensor keys and root key
        assert shareable[TensorBlobKeys.TENSOR_KEYS] == ["test_tensor"]
        assert shareable[TensorBlobKeys.ROOT_KEY] == "model"

        # Verify the tensor can be loaded back
        blob = shareable[TensorBlobKeys.SAFETENSORS_BLOB]
        loaded_tensors = load_tensors(blob)
        assert "test_tensor" in loaded_tensors
        assert torch.allclose(loaded_tensors["test_tensor"], single_tensor["test_tensor"])

        # After producing, the tensor should **not** be removed from producer's tensors
        assert "test_tensor" in producer.tensors
        assert producer.last is True
        assert producer.current == 1
        assert producer.total_bytes > 0

    def test_produce_multiple_tensors(self, random_torch_tensors, mock_stream_context, mock_fl_context):
        """Test producing multiple tensors sequentially."""
        producer = TorchTensorsProducer(tensors=random_torch_tensors.copy(), entry_timeout=3.0, root_key="model")
        original_tensor_count = len(random_torch_tensors)

        produced_keys = []

        # Produce all tensors
        for i in range(original_tensor_count):
            shareable, timeout = producer.produce(mock_stream_context, mock_fl_context)

            assert timeout == 3.0
            assert isinstance(shareable, Shareable)

            # Verify structure
            assert TensorBlobKeys.SAFETENSORS_BLOB in shareable
            assert TensorBlobKeys.TENSOR_KEYS in shareable
            assert TensorBlobKeys.ROOT_KEY in shareable
            assert shareable[TensorBlobKeys.ROOT_KEY] == "model"

            # Each shareable should contain exactly one tensor
            tensor_keys = shareable[TensorBlobKeys.TENSOR_KEYS]
            assert len(tensor_keys) == 1
            produced_keys.extend(tensor_keys)

            # Verify the tensor data
            blob = shareable[TensorBlobKeys.SAFETENSORS_BLOB]
            loaded_tensors = load_tensors(blob)
            assert len(loaded_tensors) == 1

            # Check if it's the last tensor
            if i == original_tensor_count - 1:
                assert producer.last is True
            else:
                assert producer.last is False

        # Verify all tensors were produced
        assert len(produced_keys) == original_tensor_count
        assert set(produced_keys) == set(random_torch_tensors.keys())
        assert len(producer.tensors) == 10  # All tensors should still be present

    def test_produce_with_none_tensors(self, mock_stream_context, mock_fl_context):
        """Test producing when tensors is None."""
        producer = TorchTensorsProducer(tensors=None, entry_timeout=5.0, root_key="model")

        result, timeout = producer.produce(mock_stream_context, mock_fl_context)

        assert result is None
        assert timeout == 5.0

    def test_process_replies_success(self, random_torch_tensors, mock_stream_context, mock_fl_context):
        """Test processing successful replies."""
        producer = TorchTensorsProducer(tensors=random_torch_tensors, entry_timeout=5.0, root_key="model")

        # Mock successful replies
        replies = {"peer1": Shareable(), "peer2": Shareable()}

        # Test when not last (should return None to continue streaming)
        producer.last = False
        result = producer.process_replies(replies, mock_stream_context, mock_fl_context)
        assert result is None

        # Test when last (should return True for success)
        producer.last = True
        result = producer.process_replies(replies, mock_stream_context, mock_fl_context)
        assert result is True
        assert len(producer.tensors) == 0  # Tensors should be cleared

    def test_process_replies_with_errors(self, random_torch_tensors, mock_stream_context, mock_fl_context):
        """Test processing replies with errors."""
        producer = TorchTensorsProducer(tensors=random_torch_tensors, entry_timeout=5.0, root_key="model")

        # Mock replies with errors
        error_reply = Shareable()
        error_reply.set_return_code(ReturnCode.ERROR)

        success_reply = Shareable()
        success_reply.set_return_code(ReturnCode.OK)

        replies = {"peer1": success_reply, "peer2": error_reply}

        result = producer.process_replies(replies, mock_stream_context, mock_fl_context)
        assert result is False  # Should return False due to error
        assert len(producer.tensors) == 0  # Tensors should be cleared on error

    def test_tensor_sizes_and_bytes_tracking(self, random_torch_tensors, mock_stream_context, mock_fl_context):
        """Test that tensor sizes are properly tracked."""
        producer = TorchTensorsProducer(tensors=random_torch_tensors.copy(), entry_timeout=5.0, root_key="model")

        initial_total_bytes = producer.total_bytes
        assert initial_total_bytes == 0

        # Produce first tensor
        shareable, _ = producer.produce(mock_stream_context, mock_fl_context)

        # Check that bytes were tracked
        assert producer.total_bytes > initial_total_bytes
        blob_size = len(shareable[TensorBlobKeys.SAFETENSORS_BLOB])
        assert producer.total_bytes == blob_size

    def test_tensor_key_ordering(self, mock_stream_context, mock_fl_context):
        """Test that tensors are produced in the order of their keys."""
        # Create tensors with specific ordering
        ordered_tensors = {"a_first": torch.randn(2, 2), "b_second": torch.randn(3, 3), "c_third": torch.randn(4, 4)}

        producer = TorchTensorsProducer(tensors=ordered_tensors, entry_timeout=5.0, root_key="model")

        expected_order = ["a_first", "b_second", "c_third"]
        produced_keys = []

        for _ in range(len(ordered_tensors)):
            shareable, _ = producer.produce(mock_stream_context, mock_fl_context)
            tensor_keys = shareable[TensorBlobKeys.TENSOR_KEYS]
            produced_keys.extend(tensor_keys)

        assert produced_keys == expected_order

    def test_different_tensor_dtypes(self, mock_stream_context, mock_fl_context):
        """Test producing tensors with different data types."""
        mixed_dtype_tensors = {
            "float32_tensor": torch.randn(2, 2, dtype=torch.float32),
            "float64_tensor": torch.randn(2, 2, dtype=torch.float64),
            "int32_tensor": torch.randint(0, 100, (2, 2), dtype=torch.int32),
            "int64_tensor": torch.randint(0, 100, (2, 2), dtype=torch.int64),
            "bool_tensor": torch.randint(0, 2, (2, 2), dtype=torch.bool),
        }

        producer = TorchTensorsProducer(tensors=mixed_dtype_tensors.copy(), entry_timeout=5.0, root_key="model")

        for tensor_name in mixed_dtype_tensors.keys():
            original_tensor = mixed_dtype_tensors[tensor_name].clone()
            shareable, _ = producer.produce(mock_stream_context, mock_fl_context)

            # Load and verify the tensor
            blob = shareable[TensorBlobKeys.SAFETENSORS_BLOB]
            loaded_tensors = load_tensors(blob)

            assert tensor_name in loaded_tensors
            loaded_tensor = loaded_tensors[tensor_name]
            assert loaded_tensor.dtype == original_tensor.dtype
            assert torch.equal(loaded_tensor, original_tensor)
