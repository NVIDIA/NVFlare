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

from typing import Dict, List

import pytest
import torch
from safetensors.torch import save as save_tensors

from nvflare.apis.shareable import ReturnCode, Shareable
from nvflare.app_opt.tensor_stream.consumer import TensorConsumer, TensorConsumerFactory
from nvflare.app_opt.tensor_stream.producer import TensorProducer
from nvflare.app_opt.tensor_stream.types import TensorBlobKeys, TensorCustomKeys


def create_shareables_from_tensors(
    tensors: Dict[str, torch.Tensor], root_key: str = "", task_id: str = "test_task_123"
) -> List[Shareable]:
    """Helper function to create shareables from tensors using the producer logic.

    Args:
        tensors: Dictionary of tensors to convert to shareables
        root_key: Root key for the tensors
        task_id: Task ID for the tensors

    Returns:
        List of Shareable objects that would be produced by TorchTensorsProducer
    """
    shareables = []

    for tensor_name, tensor in tensors.items():
        shareable = Shareable()
        single_tensor = {tensor_name: tensor}

        shareable[TensorBlobKeys.SAFETENSORS_BLOB] = save_tensors(single_tensor)
        shareable[TensorBlobKeys.TENSOR_KEYS] = [tensor_name]
        shareable[TensorBlobKeys.ROOT_KEY] = root_key
        shareable[TensorBlobKeys.TASK_ID] = task_id

        shareables.append(shareable)

    return shareables


class TestTorchTensorsConsumerFactory:
    """Test cases for TorchTensorsConsumerFactory class."""

    def test_get_consumer(self, mock_stream_context, mock_fl_context):
        """Test that factory creates a consumer instance."""
        factory = TensorConsumerFactory()
        consumer = factory.get_consumer(mock_stream_context, mock_fl_context)

        assert isinstance(consumer, TensorConsumer)


class TestTorchTensorsConsumer:
    """Test cases for TorchTensorsConsumer class."""

    def test_init(self, mock_stream_context, mock_fl_context):
        """Test initialization of TorchTensorsConsumer."""
        consumer = TensorConsumer(mock_stream_context, mock_fl_context)

        assert consumer.tensors_map == {}
        assert consumer.total_bytes == {}
        assert consumer.logger is not None

    def test_consume_single_tensor(self, mock_stream_context, mock_fl_context):
        """Test consuming a single tensor shareable."""
        consumer = TensorConsumer(mock_stream_context, mock_fl_context)

        # Create a test tensor
        test_tensor = torch.randn(3, 4)
        tensor_name = "test_tensor"
        root_key = "model"
        task_id = "test_task_123"

        # Create shareable
        shareable = Shareable()
        shareable[TensorBlobKeys.SAFETENSORS_BLOB] = save_tensors({tensor_name: test_tensor})
        shareable[TensorBlobKeys.TENSOR_KEYS] = [tensor_name]
        shareable[TensorBlobKeys.ROOT_KEY] = root_key
        shareable[TensorBlobKeys.TASK_ID] = task_id

        # Consume the shareable
        success, reply = consumer.consume(shareable, mock_stream_context, mock_fl_context)

        assert success is True
        assert reply.get_return_code() == ReturnCode.OK

        # Verify tensor was stored correctly
        assert root_key in consumer.tensors_map
        assert tensor_name in consumer.tensors_map[root_key]
        assert torch.allclose(consumer.tensors_map[root_key][tensor_name], test_tensor)
        assert root_key in consumer.total_bytes
        assert consumer.total_bytes[root_key] > 0

    def test_consume_multiple_tensors_same_root_key(self, random_torch_tensors, mock_stream_context, mock_fl_context):
        """Test consuming multiple tensors with the same root key."""
        consumer = TensorConsumer(mock_stream_context, mock_fl_context)
        root_key = "state_dict"

        # Create shareables from tensors
        shareables = create_shareables_from_tensors(random_torch_tensors, root_key)

        # Consume all shareables
        for shareable in shareables:
            success, reply = consumer.consume(shareable, mock_stream_context, mock_fl_context)
            assert success is True
            assert reply.get_return_code() == ReturnCode.OK

        # Verify all tensors were stored correctly
        assert root_key in consumer.tensors_map
        assert len(consumer.tensors_map[root_key]) == len(random_torch_tensors)

        # Verify tensor contents
        for tensor_name, original_tensor in random_torch_tensors.items():
            assert tensor_name in consumer.tensors_map[root_key]
            assert torch.allclose(consumer.tensors_map[root_key][tensor_name], original_tensor)

        # Verify bytes tracking
        assert root_key in consumer.total_bytes
        assert consumer.total_bytes[root_key] > 0

    def test_consume_multiple_tensors_different_root_keys(self, mock_stream_context, mock_fl_context):
        """Test consuming tensors with different root keys."""
        consumer = TensorConsumer(mock_stream_context, mock_fl_context)

        # Create tensors for different root keys
        tensors_dict = {
            "encoder": {"encoder.layer1": torch.randn(5, 5), "encoder.layer2": torch.randn(3, 3)},
            "decoder": {"decoder.layer1": torch.randn(4, 4), "decoder.layer2": torch.randn(2, 2)},
        }

        # Create and consume shareables for each root key
        for root_key, tensors in tensors_dict.items():
            shareables = create_shareables_from_tensors(tensors, root_key)

            for shareable in shareables:
                success, reply = consumer.consume(shareable, mock_stream_context, mock_fl_context)
                assert success is True
                assert reply.get_return_code() == ReturnCode.OK

        # Verify all root keys and tensors are stored

        for root_key, expected_tensors in tensors_dict.items():
            assert root_key in consumer.tensors_map
            assert len(consumer.tensors_map[root_key]) == len(expected_tensors)

            for tensor_name, original_tensor in expected_tensors.items():
                assert tensor_name in consumer.tensors_map[root_key]
                assert torch.allclose(consumer.tensors_map[root_key][tensor_name], original_tensor)

    @pytest.mark.parametrize(
        "missing_field,setup_shareable",
        [
            ("SAFETENSORS_BLOB", lambda: _create_shareable_missing_blob()),
            ("TENSOR_KEYS", lambda: _create_shareable_missing_keys()),
            ("MISMATCHED_KEYS", lambda: _create_shareable_mismatched_keys()),
        ],
    )
    def test_consume_error_cases(self, mock_stream_context, mock_fl_context, missing_field, setup_shareable):
        """Test consuming shareable with various missing or invalid fields."""
        consumer = TensorConsumer(mock_stream_context, mock_fl_context)
        shareable = setup_shareable()

        success, reply = consumer.consume(shareable, mock_stream_context, mock_fl_context)

        assert success is False
        assert reply.get_return_code() == ReturnCode.ERROR

    def test_consume_missing_root_key_defaults_to_empty_string(self, mock_stream_context, mock_fl_context):
        """Test that missing ROOT_KEY defaults to empty string."""
        consumer = TensorConsumer(mock_stream_context, mock_fl_context)
        test_tensor = torch.randn(2, 2)
        task_id = "test_task_123"
        shareable = Shareable()
        shareable[TensorBlobKeys.SAFETENSORS_BLOB] = save_tensors({"test": test_tensor})
        shareable[TensorBlobKeys.TENSOR_KEYS] = ["test"]
        shareable[TensorBlobKeys.TASK_ID] = task_id
        # Note: ROOT_KEY is missing

        success, reply = consumer.consume(shareable, mock_stream_context, mock_fl_context)

        assert success is True
        assert reply.get_return_code() == ReturnCode.OK
        # Should be stored under empty string root key
        assert "" in consumer.tensors_map
        assert "test" in consumer.tensors_map[""]
        assert torch.allclose(consumer.tensors_map[""]["test"], test_tensor)

    def test_finalize_single_root_key_empty_string(self, random_torch_tensors, mock_stream_context, mock_fl_context):
        """Test finalize with single root key that is empty string."""
        consumer = TensorConsumer(mock_stream_context, mock_fl_context)
        root_key = ""  # Empty string for top-level tensors

        # Consume tensors with empty root key
        shareables = create_shareables_from_tensors(random_torch_tensors, root_key)
        for shareable in shareables:
            consumer.consume(shareable, mock_stream_context, mock_fl_context)

        # Finalize
        consumer.finalize(mock_stream_context, mock_fl_context)

        # Verify that tensors are stored directly in the context (not nested under root key)
        stored_tensors = mock_fl_context.get_custom_prop(TensorCustomKeys.SAFE_TENSORS_PROP_KEY)
        assert isinstance(stored_tensors, dict)
        assert len(stored_tensors) == len(random_torch_tensors)

        for tensor_name, original_tensor in random_torch_tensors.items():
            assert tensor_name in stored_tensors
            assert torch.allclose(stored_tensors[tensor_name], original_tensor)

        # Verify tensors are cleared after finalization
        assert consumer.tensors_map == {}

    def test_finalize_multiple_root_keys(self, mock_stream_context, mock_fl_context):
        """Test finalize with multiple root keys."""
        consumer = TensorConsumer(mock_stream_context, mock_fl_context)

        # Create tensors for different root keys
        tensors_dict = {
            "encoder": {"layer1": torch.randn(3, 3), "layer2": torch.randn(2, 2)},
            "decoder": {"layer1": torch.randn(4, 4), "layer2": torch.randn(5, 5)},
        }

        # Consume tensors
        for root_key, tensors in tensors_dict.items():
            shareables = create_shareables_from_tensors(tensors, root_key)
            for shareable in shareables:
                consumer.consume(shareable, mock_stream_context, mock_fl_context)

        # Finalize
        consumer.finalize(mock_stream_context, mock_fl_context)

        # Verify nested structure is preserved
        stored_tensors = mock_fl_context.get_custom_prop(TensorCustomKeys.SAFE_TENSORS_PROP_KEY)
        assert isinstance(stored_tensors, dict)
        assert set(stored_tensors.keys()) == {"encoder", "decoder"}

        for root_key, expected_tensors in tensors_dict.items():
            assert root_key in stored_tensors
            for tensor_name, original_tensor in expected_tensors.items():
                assert tensor_name in stored_tensors[root_key]
                assert torch.allclose(stored_tensors[root_key][tensor_name], original_tensor)

        # Verify tensors are cleared after finalization
        assert consumer.tensors_map == {}

    def test_producer_consumer_integration(self, random_torch_tensors, mock_stream_context, mock_fl_context):
        """Integration test: producer creates shareables, consumer reconstructs tensors."""
        # Create producer
        original_tensors = random_torch_tensors.copy()
        root_key = "model_weights"
        task_id = "test_task_123"
        producer = TensorProducer(tensors=original_tensors, task_id=task_id, entry_timeout=5.0, root_key=root_key)

        # Create consumer
        consumer = TensorConsumer(mock_stream_context, mock_fl_context)

        # Producer creates shareables, consumer consumes them
        shareables_created = []
        for _ in range(len(random_torch_tensors)):
            shareable, _ = producer.produce(mock_stream_context, mock_fl_context)
            if shareable is None:
                break
            shareables_created.append(shareable)

            success, reply = consumer.consume(shareable, mock_stream_context, mock_fl_context)
            assert success is True
            assert reply.get_return_code() == ReturnCode.OK

        # Finalize consumer
        consumer.finalize(mock_stream_context, mock_fl_context)

        # Verify that shareables were created (may be fewer than tensors due to chunking)
        assert len(shareables_created) > 0

        # Verify reconstructed tensors match original
        stored_tensors = mock_fl_context.get_custom_prop(TensorCustomKeys.SAFE_TENSORS_PROP_KEY)
        assert root_key in stored_tensors
        reconstructed_tensors = stored_tensors[root_key]

        assert len(reconstructed_tensors) == len(random_torch_tensors)
        for tensor_name, original_tensor in random_torch_tensors.items():
            assert tensor_name in reconstructed_tensors
            assert torch.allclose(reconstructed_tensors[tensor_name], original_tensor)

    def test_bytes_tracking_accuracy(self, mock_stream_context, mock_fl_context):
        """Test that byte tracking is accurate across multiple consumptions."""
        consumer = TensorConsumer(mock_stream_context, mock_fl_context)
        root_key = "model"

        # Create tensors of known sizes
        tensors = {
            "small": torch.randn(2, 2),  # Small tensor
            "medium": torch.randn(10, 10),  # Medium tensor
            "large": torch.randn(50, 50),  # Large tensor
        }

        total_expected_bytes = 0
        shareables = create_shareables_from_tensors(tensors, root_key)

        for shareable in shareables:
            blob_size = len(shareable[TensorBlobKeys.SAFETENSORS_BLOB])
            total_expected_bytes += blob_size

            consumer.consume(shareable, mock_stream_context, mock_fl_context)

        # Verify byte tracking
        assert root_key in consumer.total_bytes
        assert consumer.total_bytes[root_key] == total_expected_bytes

    def test_different_tensor_dtypes_reconstruction(self, mock_stream_context, mock_fl_context):
        """Test reconstruction of tensors with different data types."""
        consumer = TensorConsumer(mock_stream_context, mock_fl_context)
        root_key = "mixed_types"

        # Create tensors with different dtypes
        mixed_dtype_tensors = {
            "float32_tensor": torch.randn(2, 2, dtype=torch.float32),
            "float64_tensor": torch.randn(2, 2, dtype=torch.float64),
            "int32_tensor": torch.randint(0, 100, (2, 2), dtype=torch.int32),
            "int64_tensor": torch.randint(0, 100, (2, 2), dtype=torch.int64),
            "bool_tensor": torch.randint(0, 2, (2, 2), dtype=torch.bool),
        }

        # Consume all tensors
        shareables = create_shareables_from_tensors(mixed_dtype_tensors, root_key)
        for shareable in shareables:
            success, reply = consumer.consume(shareable, mock_stream_context, mock_fl_context)
            assert success is True

        # Finalize and verify dtypes are preserved
        consumer.finalize(mock_stream_context, mock_fl_context)
        stored_tensors = mock_fl_context.get_custom_prop(TensorCustomKeys.SAFE_TENSORS_PROP_KEY)

        for tensor_name, original_tensor in mixed_dtype_tensors.items():
            reconstructed_tensor = stored_tensors[root_key][tensor_name]
            assert reconstructed_tensor.dtype == original_tensor.dtype
            assert torch.equal(reconstructed_tensor, original_tensor)

    def test_consume_large_number_of_tensors(self, mock_stream_context, mock_fl_context):
        """Test consuming a large number of tensors (tests generator behavior)."""
        consumer = TensorConsumer(mock_stream_context, mock_fl_context)
        root_key = "large_model"

        # Create 50 tensors
        large_tensor_dict = {f"layer_{i}": torch.randn(10, 10) for i in range(50)}

        # Consume all tensors
        shareables = create_shareables_from_tensors(large_tensor_dict, root_key)
        for shareable in shareables:
            success, reply = consumer.consume(shareable, mock_stream_context, mock_fl_context)
            assert success is True
            assert reply.get_return_code() == ReturnCode.OK

        # Verify all tensors were stored
        assert root_key in consumer.tensors_map
        assert len(consumer.tensors_map[root_key]) == 50

        # Finalize and verify
        consumer.finalize(mock_stream_context, mock_fl_context)
        stored_tensors = mock_fl_context.get_custom_prop(TensorCustomKeys.SAFE_TENSORS_PROP_KEY)
        assert len(stored_tensors[root_key]) == 50

    def test_consume_empty_blob_error(self, mock_stream_context, mock_fl_context):
        """Test that consuming an empty blob raises an error."""
        consumer = TensorConsumer(mock_stream_context, mock_fl_context)
        task_id = "test_task_123"

        shareable = Shareable()
        shareable[TensorBlobKeys.SAFETENSORS_BLOB] = b""  # Empty blob
        shareable[TensorBlobKeys.TENSOR_KEYS] = ["test"]
        shareable[TensorBlobKeys.ROOT_KEY] = "model"
        shareable[TensorBlobKeys.TASK_ID] = task_id

        success, reply = consumer.consume(shareable, mock_stream_context, mock_fl_context)

        assert success is False
        assert reply.get_return_code() == ReturnCode.ERROR

    def test_consume_empty_tensor_keys_error(self, mock_stream_context, mock_fl_context):
        """Test that consuming with empty tensor keys raises an error."""
        consumer = TensorConsumer(mock_stream_context, mock_fl_context)
        task_id = "test_task_123"

        test_tensor = torch.randn(2, 2)
        shareable = Shareable()
        shareable[TensorBlobKeys.SAFETENSORS_BLOB] = save_tensors({"test": test_tensor})
        shareable[TensorBlobKeys.TENSOR_KEYS] = []  # Empty list
        shareable[TensorBlobKeys.ROOT_KEY] = "model"
        shareable[TensorBlobKeys.TASK_ID] = task_id

        success, reply = consumer.consume(shareable, mock_stream_context, mock_fl_context)

        assert success is False
        assert reply.get_return_code() == ReturnCode.ERROR

    def test_bytes_tracking_multiple_root_keys(self, mock_stream_context, mock_fl_context):
        """Test byte tracking with multiple root keys."""
        consumer = TensorConsumer(mock_stream_context, mock_fl_context)

        tensors_dict = {
            "encoder": {"layer1": torch.randn(10, 10)},
            "decoder": {"layer1": torch.randn(20, 20)},
        }

        for root_key, tensors in tensors_dict.items():
            shareables = create_shareables_from_tensors(tensors, root_key)
            for shareable in shareables:
                consumer.consume(shareable, mock_stream_context, mock_fl_context)

        # Verify separate byte tracking for each root key
        assert "encoder" in consumer.total_bytes
        assert "decoder" in consumer.total_bytes
        assert consumer.total_bytes["encoder"] > 0
        assert consumer.total_bytes["decoder"] > 0
        assert consumer.total_bytes["decoder"] > consumer.total_bytes["encoder"]  # decoder has bigger tensors

    def test_finalize_clears_tensors_map(self, random_torch_tensors, mock_stream_context, mock_fl_context):
        """Test that finalize clears the internal tensors_map."""
        consumer = TensorConsumer(mock_stream_context, mock_fl_context)

        shareables = create_shareables_from_tensors(random_torch_tensors, "model")
        for shareable in shareables:
            consumer.consume(shareable, mock_stream_context, mock_fl_context)

        # Before finalize, tensors_map should have data
        assert len(consumer.tensors_map) > 0

        # After finalize, tensors_map should be empty
        consumer.finalize(mock_stream_context, mock_fl_context)
        assert consumer.tensors_map == {}

    def test_consume_multiple_times_same_root_key_accumulates(self, mock_stream_context, mock_fl_context):
        """Test that consuming multiple times with the same root key accumulates tensors."""
        consumer = TensorConsumer(mock_stream_context, mock_fl_context)
        root_key = "model"

        # First batch
        batch1 = {"tensor1": torch.randn(2, 2), "tensor2": torch.randn(3, 3)}
        shareables1 = create_shareables_from_tensors(batch1, root_key)
        for shareable in shareables1:
            consumer.consume(shareable, mock_stream_context, mock_fl_context)

        # Second batch
        batch2 = {"tensor3": torch.randn(4, 4), "tensor4": torch.randn(5, 5)}
        shareables2 = create_shareables_from_tensors(batch2, root_key)
        for shareable in shareables2:
            consumer.consume(shareable, mock_stream_context, mock_fl_context)

        # Verify all tensors are accumulated under the same root key
        assert root_key in consumer.tensors_map
        assert len(consumer.tensors_map[root_key]) == 4
        assert set(consumer.tensors_map[root_key].keys()) == {"tensor1", "tensor2", "tensor3", "tensor4"}


# Helper functions for parameterized tests
def _create_shareable_missing_blob():
    """Create shareable missing SAFETENSORS_BLOB."""
    shareable = Shareable()
    shareable[TensorBlobKeys.TENSOR_KEYS] = ["test_tensor"]
    shareable[TensorBlobKeys.ROOT_KEY] = "model"
    shareable[TensorBlobKeys.TASK_ID] = "test_task_123"
    return shareable


def _create_shareable_missing_keys():
    """Create shareable missing TENSOR_KEYS."""
    test_tensor = torch.randn(2, 2)
    shareable = Shareable()
    shareable[TensorBlobKeys.SAFETENSORS_BLOB] = save_tensors({"test": test_tensor})
    shareable[TensorBlobKeys.ROOT_KEY] = "model"
    shareable[TensorBlobKeys.TASK_ID] = "test_task_123"
    return shareable


def _create_shareable_missing_root():
    """Create shareable missing ROOT_KEY."""
    test_tensor = torch.randn(2, 2)
    shareable = Shareable()
    shareable[TensorBlobKeys.SAFETENSORS_BLOB] = save_tensors({"test": test_tensor})
    shareable[TensorBlobKeys.TENSOR_KEYS] = ["test"]
    shareable[TensorBlobKeys.TASK_ID] = "test_task_123"
    return shareable


def _create_shareable_mismatched_keys():
    """Create shareable with mismatched tensor keys."""
    test_tensor = torch.randn(2, 2)
    shareable = Shareable()
    shareable[TensorBlobKeys.SAFETENSORS_BLOB] = save_tensors({"actual_tensor": test_tensor})
    shareable[TensorBlobKeys.TENSOR_KEYS] = ["expected_tensor"]  # Different key
    shareable[TensorBlobKeys.ROOT_KEY] = "model"
    shareable[TensorBlobKeys.TASK_ID] = "test_task_123"
    return shareable
