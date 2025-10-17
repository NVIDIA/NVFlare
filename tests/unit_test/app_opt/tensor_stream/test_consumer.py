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
    tensors: Dict[str, torch.Tensor], parent_keys: List[str] = None, task_id: str = "test_task_123"
) -> List[Shareable]:
    """Helper function to create shareables from tensors using the producer logic.

    Args:
        tensors: Dictionary of tensors to convert to shareables
        parent_keys: Parent keys for the tensors (replaces root_key)
        task_id: Task ID for the tensors

    Returns:
        List of Shareable objects that would be produced by TorchTensorsProducer
    """
    if parent_keys is None:
        parent_keys = []

    shareables = []

    for tensor_name, tensor in tensors.items():
        shareable = Shareable()
        single_tensor = {tensor_name: tensor}

        shareable[TensorBlobKeys.SAFETENSORS_BLOB] = save_tensors(single_tensor)
        shareable[TensorBlobKeys.TENSOR_KEYS] = [tensor_name]
        shareable[TensorBlobKeys.PARENT_KEYS] = parent_keys
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

        assert consumer.params == {}
        assert consumer.total_bytes == 0
        assert consumer.num_tensors == 0
        assert consumer.task_ids == set()
        assert consumer.logger is not None

    def test_consume_single_tensor(self, mock_stream_context, mock_fl_context):
        """Test consuming a single tensor shareable."""
        consumer = TensorConsumer(mock_stream_context, mock_fl_context)

        # Create a test tensor
        test_tensor = torch.randn(3, 4)
        tensor_name = "test_tensor"
        parent_keys = ["model"]
        task_id = "test_task_123"

        # Create shareable
        shareable = Shareable()
        shareable[TensorBlobKeys.SAFETENSORS_BLOB] = save_tensors({tensor_name: test_tensor})
        shareable[TensorBlobKeys.TENSOR_KEYS] = [tensor_name]
        shareable[TensorBlobKeys.PARENT_KEYS] = parent_keys
        shareable[TensorBlobKeys.TASK_ID] = task_id

        # Consume the shareable
        success, reply = consumer.consume(shareable, mock_stream_context, mock_fl_context)

        assert success is True
        assert reply.get_return_code() == ReturnCode.OK

        # Verify tensor was stored correctly
        assert "model" in consumer.params
        assert tensor_name in consumer.params["model"]
        assert torch.allclose(consumer.params["model"][tensor_name], test_tensor)
        assert consumer.total_bytes > 0
        assert consumer.num_tensors == 1
        assert task_id in consumer.task_ids

    def test_consume_multiple_tensors_same_parent_key(self, random_torch_tensors, mock_stream_context, mock_fl_context):
        """Test consuming multiple tensors with the same parent key."""
        consumer = TensorConsumer(mock_stream_context, mock_fl_context)
        parent_keys = ["state_dict"]

        # Create shareables from tensors
        shareables = create_shareables_from_tensors(random_torch_tensors, parent_keys)

        # Consume all shareables
        for shareable in shareables:
            success, reply = consumer.consume(shareable, mock_stream_context, mock_fl_context)
            assert success is True
            assert reply.get_return_code() == ReturnCode.OK

        # Verify all tensors were stored correctly
        assert "state_dict" in consumer.params
        assert len(consumer.params["state_dict"]) == len(random_torch_tensors)

        # Verify tensor contents
        for tensor_name, original_tensor in random_torch_tensors.items():
            assert tensor_name in consumer.params["state_dict"]
            assert torch.allclose(consumer.params["state_dict"][tensor_name], original_tensor)

        # Verify bytes tracking
        assert consumer.total_bytes > 0
        assert consumer.num_tensors == len(random_torch_tensors)

    def test_consume_multiple_tensors_different_parent_keys(self, mock_stream_context, mock_fl_context):
        """Test consuming tensors with different parent keys."""
        consumer = TensorConsumer(mock_stream_context, mock_fl_context)

        # Create tensors for different parent keys
        tensors_dict = {
            "encoder": {"encoder.layer1": torch.randn(5, 5), "encoder.layer2": torch.randn(3, 3)},
            "decoder": {"decoder.layer1": torch.randn(4, 4), "decoder.layer2": torch.randn(2, 2)},
        }

        # Create and consume shareables for each parent key
        for parent_key, tensors in tensors_dict.items():
            shareables = create_shareables_from_tensors(tensors, [parent_key])

            for shareable in shareables:
                success, reply = consumer.consume(shareable, mock_stream_context, mock_fl_context)
                assert success is True
                assert reply.get_return_code() == ReturnCode.OK

        # Verify all parent keys and tensors are stored
        for parent_key, expected_tensors in tensors_dict.items():
            assert parent_key in consumer.params
            assert len(consumer.params[parent_key]) == len(expected_tensors)

            for tensor_name, original_tensor in expected_tensors.items():
                assert tensor_name in consumer.params[parent_key]
                assert torch.allclose(consumer.params[parent_key][tensor_name], original_tensor)

    @pytest.mark.parametrize(
        "missing_field,setup_shareable",
        [
            ("SAFETENSORS_BLOB", lambda: _create_shareable_missing_blob()),
            ("TENSOR_KEYS", lambda: _create_shareable_missing_keys()),
            ("TASK_ID", lambda: _create_shareable_missing_task_id()),
            ("PARENT_KEYS", lambda: _create_shareable_missing_parent_keys()),
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

    def test_consume_empty_parent_keys_stores_at_root(self, mock_stream_context, mock_fl_context):
        """Test that empty PARENT_KEYS stores tensors at root level."""
        consumer = TensorConsumer(mock_stream_context, mock_fl_context)
        test_tensor = torch.randn(2, 2)
        task_id = "test_task_123"
        shareable = Shareable()
        shareable[TensorBlobKeys.SAFETENSORS_BLOB] = save_tensors({"test": test_tensor})
        shareable[TensorBlobKeys.TENSOR_KEYS] = ["test"]
        shareable[TensorBlobKeys.TASK_ID] = task_id
        shareable[TensorBlobKeys.PARENT_KEYS] = []  # Empty parent keys

        success, reply = consumer.consume(shareable, mock_stream_context, mock_fl_context)

        assert success is True
        assert reply.get_return_code() == ReturnCode.OK
        # Should be stored at root level
        assert "test" in consumer.params
        assert torch.allclose(consumer.params["test"], test_tensor)

    def test_finalize_empty_parent_keys(self, random_torch_tensors, mock_stream_context, mock_fl_context):
        """Test finalize with empty parent keys (top-level tensors)."""
        consumer = TensorConsumer(mock_stream_context, mock_fl_context)
        parent_keys = []  # Empty parent keys for top-level tensors

        # Consume tensors with empty parent keys
        shareables = create_shareables_from_tensors(random_torch_tensors, parent_keys)
        for shareable in shareables:
            consumer.consume(shareable, mock_stream_context, mock_fl_context)

        # Finalize
        consumer.finalize(mock_stream_context, mock_fl_context)

        # Verify that tensors are stored directly in the context (not nested)
        stored_tensors = mock_fl_context.get_custom_prop(TensorCustomKeys.SAFE_TENSORS_PROP_KEY)
        assert isinstance(stored_tensors, dict)
        assert len(stored_tensors) == len(random_torch_tensors)

        for tensor_name, original_tensor in random_torch_tensors.items():
            assert tensor_name in stored_tensors
            assert torch.allclose(stored_tensors[tensor_name], original_tensor)

        # Verify tensors are cleared after finalization
        assert consumer.params == {}

        # Verify task_id was stored
        task_id = mock_fl_context.get_custom_prop(TensorCustomKeys.TASK_ID)
        assert task_id == "test_task_123"

    def test_finalize_multiple_parent_keys(self, mock_stream_context, mock_fl_context):
        """Test finalize with multiple parent keys."""
        consumer = TensorConsumer(mock_stream_context, mock_fl_context)

        # Create tensors for different parent keys
        tensors_dict = {
            "encoder": {"layer1": torch.randn(3, 3), "layer2": torch.randn(2, 2)},
            "decoder": {"layer1": torch.randn(4, 4), "layer2": torch.randn(5, 5)},
        }

        # Consume tensors
        for parent_key, tensors in tensors_dict.items():
            shareables = create_shareables_from_tensors(tensors, [parent_key])
            for shareable in shareables:
                consumer.consume(shareable, mock_stream_context, mock_fl_context)

        # Finalize
        consumer.finalize(mock_stream_context, mock_fl_context)

        # Verify nested structure is preserved
        stored_tensors = mock_fl_context.get_custom_prop(TensorCustomKeys.SAFE_TENSORS_PROP_KEY)
        assert isinstance(stored_tensors, dict)
        assert set(stored_tensors.keys()) == {"encoder", "decoder"}

        for parent_key, expected_tensors in tensors_dict.items():
            assert parent_key in stored_tensors
            for tensor_name, original_tensor in expected_tensors.items():
                assert tensor_name in stored_tensors[parent_key]
                assert torch.allclose(stored_tensors[parent_key][tensor_name], original_tensor)

        # Verify tensors are cleared after finalization
        assert consumer.params == {}

        # Verify task_id was stored
        task_id = mock_fl_context.get_custom_prop(TensorCustomKeys.TASK_ID)
        assert task_id == "test_task_123"

    def test_producer_consumer_integration(self, random_torch_tensors, mock_stream_context, mock_fl_context):
        """Integration test: producer creates shareables, consumer reconstructs tensors."""
        # Create producer
        # Producer expects nested tensors with parent keys, so we nest them
        original_tensors = {"model_weights": random_torch_tensors.copy()}
        task_id = "test_task_123"
        producer = TensorProducer(tensors=original_tensors, task_id=task_id, entry_timeout=5.0)

        # Create consumer
        consumer = TensorConsumer(mock_stream_context, mock_fl_context)

        # Producer creates shareables, consumer consumes them
        shareables_created = []
        for _ in range(len(random_torch_tensors) + 5):  # Extra iterations to ensure we exhaust the producer
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
        assert "model_weights" in stored_tensors
        reconstructed_tensors = stored_tensors["model_weights"]

        assert len(reconstructed_tensors) == len(random_torch_tensors)
        for tensor_name, original_tensor in random_torch_tensors.items():
            assert tensor_name in reconstructed_tensors
            assert torch.allclose(reconstructed_tensors[tensor_name], original_tensor)

    def test_bytes_tracking_accuracy(self, mock_stream_context, mock_fl_context):
        """Test that byte tracking is accurate across multiple consumptions."""
        consumer = TensorConsumer(mock_stream_context, mock_fl_context)
        parent_keys = ["model"]

        # Create tensors of known sizes
        tensors = {
            "small": torch.randn(2, 2),  # Small tensor
            "medium": torch.randn(10, 10),  # Medium tensor
            "large": torch.randn(50, 50),  # Large tensor
        }

        total_expected_bytes = 0
        shareables = create_shareables_from_tensors(tensors, parent_keys)

        for shareable in shareables:
            blob_size = len(shareable[TensorBlobKeys.SAFETENSORS_BLOB])
            total_expected_bytes += blob_size

            consumer.consume(shareable, mock_stream_context, mock_fl_context)

        # Verify byte tracking
        assert consumer.total_bytes == total_expected_bytes
        assert consumer.num_tensors == len(tensors)

    def test_different_tensor_dtypes_reconstruction(self, mock_stream_context, mock_fl_context):
        """Test reconstruction of tensors with different data types."""
        consumer = TensorConsumer(mock_stream_context, mock_fl_context)
        parent_keys = ["mixed_types"]

        # Create tensors with different dtypes
        mixed_dtype_tensors = {
            "float32_tensor": torch.randn(2, 2, dtype=torch.float32),
            "float64_tensor": torch.randn(2, 2, dtype=torch.float64),
            "int32_tensor": torch.randint(0, 100, (2, 2), dtype=torch.int32),
            "int64_tensor": torch.randint(0, 100, (2, 2), dtype=torch.int64),
            "bool_tensor": torch.randint(0, 2, (2, 2), dtype=torch.bool),
        }

        # Consume all tensors
        shareables = create_shareables_from_tensors(mixed_dtype_tensors, parent_keys)
        for shareable in shareables:
            success, reply = consumer.consume(shareable, mock_stream_context, mock_fl_context)
            assert success is True

        # Finalize and verify dtypes are preserved
        consumer.finalize(mock_stream_context, mock_fl_context)
        stored_tensors = mock_fl_context.get_custom_prop(TensorCustomKeys.SAFE_TENSORS_PROP_KEY)

        for tensor_name, original_tensor in mixed_dtype_tensors.items():
            reconstructed_tensor = stored_tensors["mixed_types"][tensor_name]
            assert reconstructed_tensor.dtype == original_tensor.dtype
            assert torch.equal(reconstructed_tensor, original_tensor)

    def test_consume_large_number_of_tensors(self, mock_stream_context, mock_fl_context):
        """Test consuming a large number of tensors (tests generator behavior)."""
        consumer = TensorConsumer(mock_stream_context, mock_fl_context)
        parent_keys = ["large_model"]

        # Create 50 tensors
        large_tensor_dict = {f"layer_{i}": torch.randn(10, 10) for i in range(50)}

        # Consume all tensors
        shareables = create_shareables_from_tensors(large_tensor_dict, parent_keys)
        for shareable in shareables:
            success, reply = consumer.consume(shareable, mock_stream_context, mock_fl_context)
            assert success is True
            assert reply.get_return_code() == ReturnCode.OK

        # Verify all tensors were stored
        assert "large_model" in consumer.params
        assert len(consumer.params["large_model"]) == 50

        # Finalize and verify
        consumer.finalize(mock_stream_context, mock_fl_context)
        stored_tensors = mock_fl_context.get_custom_prop(TensorCustomKeys.SAFE_TENSORS_PROP_KEY)
        assert len(stored_tensors["large_model"]) == 50

    def test_consume_empty_blob_error(self, mock_stream_context, mock_fl_context):
        """Test that consuming an empty blob raises an error."""
        consumer = TensorConsumer(mock_stream_context, mock_fl_context)
        task_id = "test_task_123"

        shareable = Shareable()
        shareable[TensorBlobKeys.SAFETENSORS_BLOB] = b""  # Empty blob
        shareable[TensorBlobKeys.TENSOR_KEYS] = ["test"]
        shareable[TensorBlobKeys.PARENT_KEYS] = ["model"]
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
        shareable[TensorBlobKeys.PARENT_KEYS] = ["model"]
        shareable[TensorBlobKeys.TASK_ID] = task_id

        success, reply = consumer.consume(shareable, mock_stream_context, mock_fl_context)

        assert success is False
        assert reply.get_return_code() == ReturnCode.ERROR

    def test_bytes_tracking_multiple_parent_keys(self, mock_stream_context, mock_fl_context):
        """Test byte tracking with multiple parent keys accumulates total."""
        consumer = TensorConsumer(mock_stream_context, mock_fl_context)

        tensors_dict = {
            "encoder": {"layer1": torch.randn(10, 10)},
            "decoder": {"layer1": torch.randn(20, 20)},
        }

        total_expected_bytes = 0
        for parent_key, tensors in tensors_dict.items():
            shareables = create_shareables_from_tensors(tensors, [parent_key])
            for shareable in shareables:
                total_expected_bytes += len(shareable[TensorBlobKeys.SAFETENSORS_BLOB])
                consumer.consume(shareable, mock_stream_context, mock_fl_context)

        # Verify total byte tracking across all parent keys
        assert consumer.total_bytes == total_expected_bytes
        assert consumer.total_bytes > 0
        assert consumer.num_tensors == 2

    def test_finalize_clears_tensors_map(self, random_torch_tensors, mock_stream_context, mock_fl_context):
        """Test that finalize clears the internal tensors_map."""
        consumer = TensorConsumer(mock_stream_context, mock_fl_context)

        shareables = create_shareables_from_tensors(random_torch_tensors, ["model"])
        for shareable in shareables:
            consumer.consume(shareable, mock_stream_context, mock_fl_context)

        # Before finalize, tensors_map should have data
        assert len(consumer.params) > 0

        # After finalize, tensors_map should be empty
        consumer.finalize(mock_stream_context, mock_fl_context)
        assert consumer.params == {}

    def test_consume_multiple_times_same_parent_key_accumulates(self, mock_stream_context, mock_fl_context):
        """Test that consuming multiple times with the same parent key accumulates tensors."""
        consumer = TensorConsumer(mock_stream_context, mock_fl_context)
        parent_keys = ["model"]

        # First batch
        batch1 = {"tensor1": torch.randn(2, 2), "tensor2": torch.randn(3, 3)}
        shareables1 = create_shareables_from_tensors(batch1, parent_keys)
        for shareable in shareables1:
            consumer.consume(shareable, mock_stream_context, mock_fl_context)

        # Second batch
        batch2 = {"tensor3": torch.randn(4, 4), "tensor4": torch.randn(5, 5)}
        shareables2 = create_shareables_from_tensors(batch2, parent_keys)
        for shareable in shareables2:
            consumer.consume(shareable, mock_stream_context, mock_fl_context)

        # Verify all tensors are accumulated under the same parent key
        assert "model" in consumer.params
        assert len(consumer.params["model"]) == 4
        assert set(consumer.params["model"].keys()) == {"tensor1", "tensor2", "tensor3", "tensor4"}

    def test_finalize_multiple_task_ids_error(self, mock_stream_context, mock_fl_context):
        """Test that finalize raises error when multiple task_ids are present."""
        consumer = TensorConsumer(mock_stream_context, mock_fl_context)

        # Create tensors with different task_ids
        batch1 = {"tensor1": torch.randn(2, 2)}
        shareables1 = create_shareables_from_tensors(batch1, ["model"], task_id="task_1")
        for shareable in shareables1:
            consumer.consume(shareable, mock_stream_context, mock_fl_context)

        batch2 = {"tensor2": torch.randn(3, 3)}
        shareables2 = create_shareables_from_tensors(batch2, ["model"], task_id="task_2")
        for shareable in shareables2:
            consumer.consume(shareable, mock_stream_context, mock_fl_context)

        # Finalize should raise ValueError due to multiple task_ids
        with pytest.raises(ValueError, match="Expected one task_id, but found multiple"):
            consumer.finalize(mock_stream_context, mock_fl_context)

    def test_finalize_no_task_id_error(self, mock_stream_context, mock_fl_context):
        """Test that finalize raises error when no task_id is present."""
        consumer = TensorConsumer(mock_stream_context, mock_fl_context)

        # Finalize without consuming any shareables should raise ValueError
        with pytest.raises(ValueError, match="No valid task_id found in received shareables"):
            consumer.finalize(mock_stream_context, mock_fl_context)


# Helper functions for parameterized tests
def _create_shareable_missing_blob():
    """Create shareable missing SAFETENSORS_BLOB."""
    shareable = Shareable()
    shareable[TensorBlobKeys.TENSOR_KEYS] = ["test_tensor"]
    shareable[TensorBlobKeys.PARENT_KEYS] = ["model"]
    shareable[TensorBlobKeys.TASK_ID] = "test_task_123"
    return shareable


def _create_shareable_missing_keys():
    """Create shareable missing TENSOR_KEYS."""
    test_tensor = torch.randn(2, 2)
    shareable = Shareable()
    shareable[TensorBlobKeys.SAFETENSORS_BLOB] = save_tensors({"test": test_tensor})
    shareable[TensorBlobKeys.PARENT_KEYS] = ["model"]
    shareable[TensorBlobKeys.TASK_ID] = "test_task_123"
    return shareable


def _create_shareable_missing_task_id():
    """Create shareable missing TASK_ID."""
    test_tensor = torch.randn(2, 2)
    shareable = Shareable()
    shareable[TensorBlobKeys.SAFETENSORS_BLOB] = save_tensors({"test": test_tensor})
    shareable[TensorBlobKeys.TENSOR_KEYS] = ["test"]
    shareable[TensorBlobKeys.PARENT_KEYS] = ["model"]
    return shareable


def _create_shareable_missing_parent_keys():
    """Create shareable missing PARENT_KEYS."""
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
    shareable[TensorBlobKeys.PARENT_KEYS] = ["model"]
    shareable[TensorBlobKeys.TASK_ID] = "test_task_123"
    return shareable
