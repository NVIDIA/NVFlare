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
from nvflare.app_opt.tensor_stream.producer import TensorProducer
from nvflare.app_opt.tensor_stream.types import TensorBlobKeys


class TestTorchTensorsProducer:
    """Test cases for TorchTensorsProducer class."""

    @pytest.mark.parametrize(
        "tensors,expected_keys_len",
        [
            ({"layer1.weight": torch.randn(2, 3), "layer1.bias": torch.randn(2)}, 2),  # Valid tensors
            ({}, 0),  # Empty tensors
        ],
    )
    def test_init_with_various_tensors(self, tensors, expected_keys_len):
        """Test initialization with various tensor configurations."""
        entry_timeout = 5.0
        root_key = "model"

        producer = TensorProducer(tensors=tensors, entry_timeout=entry_timeout, root_key=root_key)

        assert producer.entry_timeout == entry_timeout
        assert producer.root_key == root_key
        assert producer.last is False
        assert len(producer.tensors_keys) == expected_keys_len
        assert producer.total_bytes == 0

        if tensors:
            assert producer.tensors_keys == list(tensors.keys())

    def test_init_with_none_tensors(self):
        """Test initialization with None tensors raises ValueError."""
        with pytest.raises(ValueError, match="No tensors received"):
            TensorProducer(tensors=None, entry_timeout=5.0, root_key="model")

    def test_produce_single_tensor(self, random_torch_tensors, mock_stream_context, mock_fl_context):
        """Test producing a single tensor."""
        # Use only one tensor for this test
        single_tensor = {"test_tensor": random_torch_tensors["layer1.weight"]}
        producer = TensorProducer(tensors=single_tensor.copy(), entry_timeout=5.0, root_key="model")

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

        # After producing, check that last is not yet set (need to call produce again)
        assert producer.last is False
        assert producer.total_bytes > 0

        # Produce again to check for completion
        shareable2, _ = producer.produce(mock_stream_context, mock_fl_context)
        assert shareable2 is None  # No more data

    def test_produce_multiple_tensors(self, random_torch_tensors, mock_stream_context, mock_fl_context):
        """Test producing multiple tensors sequentially."""
        producer = TensorProducer(tensors=random_torch_tensors.copy(), entry_timeout=3.0, root_key="model")
        original_tensor_count = len(random_torch_tensors)

        produced_keys = []

        # Produce all tensors (may be in chunks)
        while True:
            shareable, timeout = producer.produce(mock_stream_context, mock_fl_context)

            if shareable is None:
                break

            assert timeout == 3.0
            assert isinstance(shareable, Shareable)

            # Verify structure
            assert TensorBlobKeys.SAFETENSORS_BLOB in shareable
            assert TensorBlobKeys.TENSOR_KEYS in shareable
            assert TensorBlobKeys.ROOT_KEY in shareable
            assert shareable[TensorBlobKeys.ROOT_KEY] == "model"

            # Collect the tensor keys from this shareable
            tensor_keys = shareable[TensorBlobKeys.TENSOR_KEYS]
            produced_keys.extend(tensor_keys)

            # Verify the tensor data
            blob = shareable[TensorBlobKeys.SAFETENSORS_BLOB]
            loaded_tensors = load_tensors(blob)
            assert len(loaded_tensors) == len(tensor_keys)

        # Verify all tensors were produced
        assert len(produced_keys) == original_tensor_count
        assert set(produced_keys) == set(random_torch_tensors.keys())

    def test_produce_with_none_tensors(self, mock_stream_context, mock_fl_context):
        """Test that initializing producer with None tensors raises ValueError."""
        with pytest.raises(ValueError, match="No tensors received"):
            TensorProducer(tensors=None, entry_timeout=5.0, root_key="model")

    def test_process_replies_success(self, random_torch_tensors, mock_stream_context, mock_fl_context):
        """Test processing successful replies."""
        producer = TensorProducer(tensors=random_torch_tensors, entry_timeout=5.0, root_key="model")

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

    def test_process_replies_with_errors(self, random_torch_tensors, mock_stream_context, mock_fl_context):
        """Test processing replies with errors."""
        producer = TensorProducer(tensors=random_torch_tensors, entry_timeout=5.0, root_key="model")

        # Mock replies with errors
        error_reply = Shareable()
        error_reply.set_return_code(ReturnCode.ERROR)

        success_reply = Shareable()
        success_reply.set_return_code(ReturnCode.OK)

        replies = {"peer1": success_reply, "peer2": error_reply}

        result = producer.process_replies(replies, mock_stream_context, mock_fl_context)
        assert result is False  # Should return False due to error

    def test_tensor_sizes_and_bytes_tracking(self, random_torch_tensors, mock_stream_context, mock_fl_context):
        """Test that tensor sizes are properly tracked."""
        producer = TensorProducer(tensors=random_torch_tensors.copy(), entry_timeout=5.0, root_key="model")

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

        producer = TensorProducer(tensors=ordered_tensors, entry_timeout=5.0, root_key="model")

        expected_order = ["a_first", "b_second", "c_third"]
        produced_keys = []

        while True:
            shareable, _ = producer.produce(mock_stream_context, mock_fl_context)
            if shareable is None:
                break
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

        producer = TensorProducer(tensors=mixed_dtype_tensors.copy(), entry_timeout=5.0, root_key="model")

        # Collect all produced tensors
        all_loaded_tensors = {}
        while True:
            shareable, _ = producer.produce(mock_stream_context, mock_fl_context)
            if shareable is None:
                break

            # Load and verify the tensors in this shareable
            blob = shareable[TensorBlobKeys.SAFETENSORS_BLOB]
            loaded_tensors = load_tensors(blob)
            all_loaded_tensors.update(loaded_tensors)

        # Verify all tensors and their dtypes
        for tensor_name, original_tensor in mixed_dtype_tensors.items():
            assert tensor_name in all_loaded_tensors
            loaded_tensor = all_loaded_tensors[tensor_name]
            assert loaded_tensor.dtype == original_tensor.dtype
            assert torch.equal(loaded_tensor, original_tensor)

    def test_chunking_behavior_with_default_chunk_size(self, mock_stream_context, mock_fl_context):
        """Test that producer chunks tensors correctly with default chunk size (25)."""
        # Create 30 tensors to test chunking (should be 2 chunks: 25 + 5)
        large_tensor_dict = {f"tensor_{i}": torch.randn(2, 2) for i in range(30)}

        producer = TensorProducer(tensors=large_tensor_dict, entry_timeout=5.0, root_key="model")

        chunks = []
        while True:
            shareable, _ = producer.produce(mock_stream_context, mock_fl_context)
            if shareable is None:
                break
            chunks.append(shareable[TensorBlobKeys.TENSOR_KEYS])

        # Should have 2 chunks: 25 and 5
        assert len(chunks) == 2
        assert len(chunks[0]) == 25
        assert len(chunks[1]) == 5

        # Verify all tensors are present
        all_keys = []
        for chunk in chunks:
            all_keys.extend(chunk)
        assert set(all_keys) == set(large_tensor_dict.keys())

    def test_empty_tensors_dict(self, mock_stream_context, mock_fl_context):
        """Test producing from an empty tensors dictionary."""
        producer = TensorProducer(tensors={}, entry_timeout=5.0, root_key="model")

        shareable, timeout = producer.produce(mock_stream_context, mock_fl_context)

        # Should return None immediately for empty dict
        assert shareable is None
        assert timeout == 5.0

    def test_log_completion_called(self, random_torch_tensors, mock_stream_context, mock_fl_context):
        """Test that log_completion is called when production is complete."""
        producer = TensorProducer(tensors=random_torch_tensors, entry_timeout=5.0, root_key="model")

        # Produce all tensors
        while True:
            shareable, _ = producer.produce(mock_stream_context, mock_fl_context)
            if shareable is None:
                break

        # Verify last flag is set
        assert producer.last is True

    def test_total_bytes_accumulation(self, mock_stream_context, mock_fl_context):
        """Test that total_bytes accumulates correctly across multiple produces."""
        tensors = {
            "tensor1": torch.randn(10, 10),
            "tensor2": torch.randn(20, 20),
            "tensor3": torch.randn(5, 5),
        }

        producer = TensorProducer(tensors=tensors, entry_timeout=5.0, root_key="model")

        total_bytes = 0
        while True:
            shareable, _ = producer.produce(mock_stream_context, mock_fl_context)
            if shareable is None:
                break
            blob_size = len(shareable[TensorBlobKeys.SAFETENSORS_BLOB])
            total_bytes += blob_size

        assert producer.total_bytes == total_bytes
        assert producer.total_bytes > 0
