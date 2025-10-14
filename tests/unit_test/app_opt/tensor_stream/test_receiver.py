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

from unittest.mock import patch

import pytest
import torch

from nvflare.apis.dxo import DataKind
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.shareable import Shareable
from nvflare.app_opt.tensor_stream.consumer import TensorConsumerFactory
from nvflare.app_opt.tensor_stream.receiver import TensorReceiver
from nvflare.app_opt.tensor_stream.types import SAFE_TENSORS_PROP_KEY, TENSORS_CHANNEL, TensorTopics
from nvflare.client.config import ExchangeFormat

from .conftest import setup_mock_get_prop_with_task_id


class TestTensorReceiver:
    """Test cases for TensorReceiver class."""

    @pytest.mark.parametrize(
        "ctx_prop_key,format_type,channel,expected_topic,expected_channel",
        [
            (
                FLContextKey.TASK_DATA,
                ExchangeFormat.PYTORCH,
                "custom_channel",
                TensorTopics.TASK_DATA,
                "custom_channel",
            ),
            (
                FLContextKey.TASK_RESULT,
                ExchangeFormat.NUMPY,
                None,
                TensorTopics.TASK_RESULT,
                TENSORS_CHANNEL,
            ),  # Default channel
        ],
    )
    @patch("nvflare.app_opt.tensor_stream.receiver.get_topic_for_ctx_prop_key")
    def test_init_and_register(
        self,
        mock_get_topic,
        mock_streamable_engine,
        ctx_prop_key,
        format_type,
        channel,
        expected_topic,
        expected_channel,
    ):
        """Test TensorReceiver initialization and registration with various parameters."""
        mock_get_topic.return_value = expected_topic

        if channel is None:
            receiver = TensorReceiver(engine=mock_streamable_engine, ctx_prop_key=ctx_prop_key, format=format_type)
        else:
            receiver = TensorReceiver(
                engine=mock_streamable_engine, ctx_prop_key=ctx_prop_key, format=format_type, channel=channel
            )

        # Verify initialization
        assert receiver.engine == mock_streamable_engine
        assert receiver.ctx_prop_key == ctx_prop_key
        assert receiver.format == format_type
        assert receiver.channel == expected_channel
        assert receiver.tensors == {}
        assert receiver.logger is not None

        # Verify registration was called
        mock_streamable_engine.register_stream_processing.assert_called_once()
        call_args = mock_streamable_engine.register_stream_processing.call_args

        assert call_args.kwargs["channel"] == expected_channel
        assert call_args.kwargs["topic"] == expected_topic
        assert isinstance(call_args.kwargs["factory"], TensorConsumerFactory)
        assert call_args.kwargs["stream_done_cb"] == receiver._save_tensors_cb

        mock_get_topic.assert_called_once_with(ctx_prop_key)

    def test_save_tensors_cb_success(self, mock_streamable_engine, mock_fl_context, random_torch_tensors):
        """Test _save_tensors_cb with successful tensor reception."""
        receiver = TensorReceiver(
            engine=mock_streamable_engine, ctx_prop_key=FLContextKey.TASK_DATA, format=ExchangeFormat.PYTORCH
        )

        # Setup FL context with received tensors
        peer_name = "test_peer"
        task_id = "test_task_123"
        mock_fl_context.get_peer_context().get_identity_name.return_value = peer_name
        mock_fl_context.set_custom_prop("task_id", task_id)
        mock_fl_context.set_custom_prop(SAFE_TENSORS_PROP_KEY, random_torch_tensors)

        # Call the callback with success=True
        receiver._save_tensors_cb(True, mock_fl_context)

        # Verify tensors were stored using task_id as the key
        assert task_id in receiver.tensors
        assert receiver.tensors[task_id] == random_torch_tensors

    def test_save_tensors_cb_failure(self, mock_streamable_engine, mock_fl_context):
        """Test _save_tensors_cb with failed tensor reception."""
        receiver = TensorReceiver(
            engine=mock_streamable_engine, ctx_prop_key=FLContextKey.TASK_DATA, format=ExchangeFormat.PYTORCH
        )

        peer_name = "test_client"
        task_id = "test_task_456"
        mock_fl_context.get_peer_context().get_identity_name.return_value = peer_name
        mock_fl_context.set_custom_prop("task_id", task_id)

        # Call the callback with success=False - should raise ValueError
        with pytest.raises(ValueError, match=f"Failed to receive tensors from peer '{peer_name}' and task '{task_id}'"):
            receiver._save_tensors_cb(False, mock_fl_context)

        # Verify no tensors were stored
        assert task_id not in receiver.tensors
        assert len(receiver.tensors) == 0

    def test_save_tensors_cb_no_tensors(self, mock_streamable_engine, mock_fl_context):
        """Test _save_tensors_cb when no tensors are found in context."""
        receiver = TensorReceiver(
            engine=mock_streamable_engine, ctx_prop_key=FLContextKey.TASK_DATA, format=ExchangeFormat.PYTORCH
        )

        peer_name = "test_client"
        task_id = "test_task_789"
        mock_fl_context.get_peer_context().get_identity_name.return_value = peer_name
        mock_fl_context.set_custom_prop("task_id", task_id)
        mock_fl_context.set_custom_prop(SAFE_TENSORS_PROP_KEY, None)

        # Call the callback with success=True but no tensors - should raise ValueError
        with pytest.raises(ValueError, match=f"No tensors found from peer '{peer_name}' and task '{task_id}'"):
            receiver._save_tensors_cb(True, mock_fl_context)

        # Verify no tensors were stored
        assert task_id not in receiver.tensors
        assert len(receiver.tensors) == 0

    def test_save_tensors_cb_nested_tensors(self, mock_streamable_engine, mock_fl_context, sample_nested_tensors):
        """Test _save_tensors_cb with nested tensor structure."""
        receiver = TensorReceiver(
            engine=mock_streamable_engine, ctx_prop_key=FLContextKey.TASK_DATA, format=ExchangeFormat.PYTORCH
        )

        peer_name = "test_client"
        task_id = "test_task_nested"
        mock_fl_context.get_peer_context().get_identity_name.return_value = peer_name
        mock_fl_context.set_custom_prop("task_id", task_id)
        mock_fl_context.set_custom_prop(SAFE_TENSORS_PROP_KEY, sample_nested_tensors)

        # Call the callback
        receiver._save_tensors_cb(True, mock_fl_context)

        # Verify nested tensors were stored correctly using task_id as the key
        assert task_id in receiver.tensors
        assert receiver.tensors[task_id] == sample_nested_tensors
        assert "encoder" in receiver.tensors[task_id]
        assert "decoder" in receiver.tensors[task_id]

    def test_set_ctx_with_tensors_torch_format(
        self, mock_streamable_engine, mock_fl_context, sample_shareable_with_dxo, random_torch_tensors
    ):
        """Test set_ctx_with_tensors with torch format."""
        receiver = TensorReceiver(
            engine=mock_streamable_engine, ctx_prop_key=FLContextKey.TASK_DATA, format=ExchangeFormat.PYTORCH
        )

        # Setup: store tensors using task_id as the key
        peer_name = "test_client"
        task_id = "test_task_torch"
        receiver.tensors[task_id] = random_torch_tensors

        # Setup FL context
        mock_fl_context.get_peer_context().get_identity_name.return_value = peer_name
        mock_fl_context.get_prop.return_value = sample_shareable_with_dxo
        # Mock get_prop for FLContextKey.TASK_ID specifically
        setup_mock_get_prop_with_task_id(mock_fl_context, task_id, sample_shareable_with_dxo)

        # Call set_ctx_with_tensors
        receiver.set_ctx_with_tensors(mock_fl_context)

        # Verify the DXO was updated with tensors
        mock_fl_context.set_prop.assert_called_once()
        call_args = mock_fl_context.set_prop.call_args

        assert call_args.args[0] == FLContextKey.TASK_DATA  # Property key
        updated_shareable = call_args.args[1]  # Updated shareable
        assert call_args.kwargs["private"] is True
        assert call_args.kwargs["sticky"] is False

        # Verify DXO data was updated with torch tensors
        dxo_data = updated_shareable["DXO"]["data"]
        assert len(dxo_data) == len(random_torch_tensors)
        for name, tensor in random_torch_tensors.items():
            assert name in dxo_data
            assert torch.allclose(dxo_data[name], tensor)
            assert isinstance(dxo_data[name], torch.Tensor)

        # Verify tensors were removed from receiver after use
        assert task_id not in receiver.tensors

    def test_set_ctx_with_tensors_numpy_format(
        self, mock_streamable_engine, mock_fl_context, sample_shareable_with_dxo, random_torch_tensors
    ):
        """Test set_ctx_with_tensors with numpy format conversion."""
        receiver = TensorReceiver(
            engine=mock_streamable_engine,
            ctx_prop_key=FLContextKey.TASK_DATA,
            format=ExchangeFormat.NUMPY,  # Different format
        )

        # Setup: store tensors using task_id as the key
        peer_name = "test_client"
        task_id = "test_task_numpy"
        receiver.tensors[task_id] = random_torch_tensors

        # Setup FL context
        mock_fl_context.get_peer_context().get_identity_name.return_value = peer_name
        # Mock get_prop for FLContextKey.TASK_ID specifically
        setup_mock_get_prop_with_task_id(mock_fl_context, task_id, sample_shareable_with_dxo)

        # Call set_ctx_with_tensors
        receiver.set_ctx_with_tensors(mock_fl_context)

        # Verify the DXO was updated with numpy arrays
        call_args = mock_fl_context.set_prop.call_args
        updated_shareable = call_args.args[1]
        dxo_data = updated_shareable["DXO"]["data"]

        assert len(dxo_data) == len(random_torch_tensors)
        for name, original_tensor in random_torch_tensors.items():
            assert name in dxo_data
            # Verify it's a numpy array and values match
            numpy_array = dxo_data[name]
            # Check if it has numpy method (should be converted from torch tensor)
            assert hasattr(numpy_array, "numpy") or hasattr(numpy_array, "shape")
            if hasattr(numpy_array, "numpy"):
                numpy_array = numpy_array.numpy()
            assert torch.allclose(torch.from_numpy(numpy_array), original_tensor)

    def test_set_ctx_with_tensors_weight_diff(
        self, mock_streamable_engine, mock_fl_context, sample_shareable_with_weight_diff_dxo, random_torch_tensors
    ):
        """Test set_ctx_with_tensors with WEIGHT_DIFF data kind."""
        receiver = TensorReceiver(
            engine=mock_streamable_engine, ctx_prop_key=FLContextKey.TASK_RESULT, format=ExchangeFormat.PYTORCH
        )

        # Setup: store tensors using task_id as the key
        peer_name = "test_client"
        task_id = "test_task_weight_diff"
        receiver.tensors[task_id] = random_torch_tensors

        # Setup FL context
        mock_fl_context.get_peer_context().get_identity_name.return_value = peer_name
        # Mock get_prop for FLContextKey.TASK_ID specifically
        setup_mock_get_prop_with_task_id(mock_fl_context, task_id, sample_shareable_with_weight_diff_dxo)

        # Call set_ctx_with_tensors
        receiver.set_ctx_with_tensors(mock_fl_context)

        # Verify the DXO was updated
        call_args = mock_fl_context.set_prop.call_args
        updated_shareable = call_args.args[1]
        dxo_data = updated_shareable["DXO"]["data"]

        # Should contain the received tensors
        assert len(dxo_data) == len(random_torch_tensors)
        for name, tensor in random_torch_tensors.items():
            assert name in dxo_data
            assert torch.allclose(dxo_data[name], tensor)

    def test_set_ctx_with_tensors_nested_structure(
        self, mock_streamable_engine, mock_fl_context, sample_shareable_with_dxo, sample_nested_tensors
    ):
        """Test set_ctx_with_tensors with nested tensor structure."""
        receiver = TensorReceiver(
            engine=mock_streamable_engine, ctx_prop_key=FLContextKey.TASK_DATA, format=ExchangeFormat.PYTORCH
        )

        # Setup: store nested tensors using task_id as the key
        peer_name = "test_client"
        task_id = "test_task_nested_set"
        receiver.tensors[task_id] = sample_nested_tensors

        # Setup FL context
        mock_fl_context.get_peer_context().get_identity_name.return_value = peer_name
        # Mock get_prop for FLContextKey.TASK_ID specifically
        setup_mock_get_prop_with_task_id(mock_fl_context, task_id, sample_shareable_with_dxo)

        # Call set_ctx_with_tensors
        receiver.set_ctx_with_tensors(mock_fl_context)

        # Verify the DXO was updated with nested structure
        call_args = mock_fl_context.set_prop.call_args
        updated_shareable = call_args.args[1]
        dxo_data = updated_shareable["DXO"]["data"]

        assert "encoder" in dxo_data
        assert "decoder" in dxo_data

        # Verify nested tensors
        for root_key in ["encoder", "decoder"]:
            for name, tensor in sample_nested_tensors[root_key].items():
                assert name in dxo_data[root_key]
                assert torch.allclose(dxo_data[root_key][name], tensor)

    def test_set_ctx_with_tensors_no_tensors_for_peer(
        self, mock_streamable_engine, mock_fl_context, sample_shareable_with_dxo
    ):
        """Test set_ctx_with_tensors when no tensors are stored for task_id."""
        receiver = TensorReceiver(
            engine=mock_streamable_engine, ctx_prop_key=FLContextKey.TASK_DATA, format=ExchangeFormat.PYTORCH
        )

        # Setup FL context without storing any tensors for this task_id
        peer_name = "test_client"
        task_id = "test_task_no_tensors"
        mock_fl_context.get_peer_context().get_identity_name.return_value = peer_name
        # Mock get_prop to return task_id
        mock_fl_context.get_prop.return_value = task_id

        # Should log warning and return early (not raise an exception)
        receiver.set_ctx_with_tensors(mock_fl_context)

        # Verify that set_prop was not called since the method returned early
        mock_fl_context.set_prop.assert_not_called()

        # Verify that get_prop was called only once for TASK_ID (not for shareable)
        assert mock_fl_context.get_prop.call_count == 1
        mock_fl_context.get_prop.assert_called_once_with(FLContextKey.TASK_ID, None)

    def test_set_ctx_with_tensors_no_shareable(self, mock_streamable_engine, mock_fl_context, random_torch_tensors):
        """Test set_ctx_with_tensors when no shareable is found in context."""
        receiver = TensorReceiver(
            engine=mock_streamable_engine, ctx_prop_key=FLContextKey.TASK_DATA, format=ExchangeFormat.PYTORCH
        )

        # Setup: store tensors but no shareable in context
        peer_name = "test_client"
        task_id = "test_task_no_shareable"
        receiver.tensors[task_id] = random_torch_tensors

        mock_fl_context.get_peer_context().get_identity_name.return_value = peer_name
        # Mock get_prop to return task_id first, then None for shareable
        setup_mock_get_prop_with_task_id(mock_fl_context, task_id, None)

        # Should raise RuntimeError
        with pytest.raises(RuntimeError, match="No shareable found in FLContext"):
            receiver.set_ctx_with_tensors(mock_fl_context)

    def test_set_ctx_with_tensors_no_dxo(self, mock_streamable_engine, mock_fl_context, random_torch_tensors):
        """Test set_ctx_with_tensors when shareable contains no DXO."""
        receiver = TensorReceiver(
            engine=mock_streamable_engine, ctx_prop_key=FLContextKey.TASK_DATA, format=ExchangeFormat.PYTORCH
        )

        # Setup: store tensors and empty shareable
        peer_name = "test_client"
        task_id = "test_task_no_dxo"
        receiver.tensors[task_id] = random_torch_tensors

        empty_shareable = Shareable()  # No DXO key
        mock_fl_context.get_peer_context().get_identity_name.return_value = peer_name
        # Mock get_prop to return task_id first, then empty shareable
        setup_mock_get_prop_with_task_id(mock_fl_context, task_id, empty_shareable)

        # Should raise RuntimeError
        with pytest.raises(RuntimeError, match="No DXO found in shareable"):
            receiver.set_ctx_with_tensors(mock_fl_context)

    def test_set_ctx_with_tensors_invalid_data_kind(
        self, mock_streamable_engine, mock_fl_context, random_torch_tensors
    ):
        """Test set_ctx_with_tensors with invalid DXO data kind."""
        receiver = TensorReceiver(
            engine=mock_streamable_engine, ctx_prop_key=FLContextKey.TASK_DATA, format=ExchangeFormat.PYTORCH
        )

        # Setup: store tensors and shareable with invalid data kind
        peer_name = "test_client"
        task_id = "test_task_invalid_kind"
        receiver.tensors[task_id] = random_torch_tensors

        invalid_shareable = Shareable()
        invalid_shareable["DXO"] = {"kind": DataKind.METRICS, "data": {}}  # Invalid data kind

        mock_fl_context.get_peer_context().get_identity_name.return_value = peer_name
        # Mock get_prop to return task_id first, then invalid shareable
        setup_mock_get_prop_with_task_id(mock_fl_context, task_id, invalid_shareable)

        # Should raise RuntimeError
        with pytest.raises(RuntimeError, match="Task data kind is not WEIGHTS or WEIGHT_DIFF"):
            receiver.set_ctx_with_tensors(mock_fl_context)

    def test_multiple_peers_tensors(self, mock_streamable_engine, mock_fl_context, random_torch_tensors):
        """Test handling tensors from multiple tasks."""
        receiver = TensorReceiver(
            engine=mock_streamable_engine, ctx_prop_key=FLContextKey.TASK_DATA, format=ExchangeFormat.PYTORCH
        )

        # Simulate receiving tensors from multiple tasks
        task_id_1 = "test_task_multi_1"
        task_id_2 = "test_task_multi_2"
        task_1_tensors = {k: v for k, v in list(random_torch_tensors.items())[:5]}
        task_2_tensors = {k: v for k, v in list(random_torch_tensors.items())[5:]}

        # Store tensors for both tasks
        receiver.tensors[task_id_1] = task_1_tensors
        receiver.tensors[task_id_2] = task_2_tensors

        assert len(receiver.tensors) == 2
        assert task_id_1 in receiver.tensors
        assert task_id_2 in receiver.tensors

        # Verify each task's tensors are stored correctly
        assert receiver.tensors[task_id_1] == task_1_tensors
        assert receiver.tensors[task_id_2] == task_2_tensors

    def test_tensor_cleanup_after_use(
        self, mock_streamable_engine, mock_fl_context, sample_shareable_with_dxo, random_torch_tensors
    ):
        """Test that tensors are cleaned up after being used in set_ctx_with_tensors."""
        receiver = TensorReceiver(
            engine=mock_streamable_engine, ctx_prop_key=FLContextKey.TASK_DATA, format=ExchangeFormat.PYTORCH
        )

        # Setup: store tensors for multiple tasks
        task_id_1 = "test_task_cleanup_1"
        task_id_2 = "test_task_cleanup_2"
        receiver.tensors[task_id_1] = random_torch_tensors
        receiver.tensors[task_id_2] = {k: v * 2 for k, v in random_torch_tensors.items()}

        # Use tensors from task_id_1
        peer_name = "test_client"
        mock_fl_context.get_peer_context().get_identity_name.return_value = peer_name
        # Mock get_prop for FLContextKey.TASK_ID specifically
        setup_mock_get_prop_with_task_id(mock_fl_context, task_id_1, sample_shareable_with_dxo)

        receiver.set_ctx_with_tensors(mock_fl_context)

        # Verify task_id_1 tensors were removed but task_id_2 tensors remain
        assert task_id_1 not in receiver.tensors
        assert task_id_2 in receiver.tensors
        assert len(receiver.tensors) == 1

    @patch("nvflare.app_opt.tensor_stream.receiver.get_topic_for_ctx_prop_key")
    def test_integration_receive_and_set_workflow(
        self, mock_get_topic, mock_streamable_engine, mock_fl_context, sample_shareable_with_dxo, random_torch_tensors
    ):
        """Integration test for complete receive and set workflow."""
        mock_get_topic.return_value = TensorTopics.TASK_DATA

        # Create receiver
        receiver = TensorReceiver(
            engine=mock_streamable_engine, ctx_prop_key=FLContextKey.TASK_DATA, format=ExchangeFormat.PYTORCH
        )

        peer_name = "integration_client"
        task_id = "test_task_integration"

        # Step 1: Simulate receiving tensors (callback called by streaming engine)
        mock_fl_context.get_peer_context().get_identity_name.return_value = peer_name
        mock_fl_context.set_custom_prop("task_id", task_id)
        mock_fl_context.set_custom_prop(SAFE_TENSORS_PROP_KEY, random_torch_tensors)

        receiver._save_tensors_cb(True, mock_fl_context)

        # Verify tensors were stored using task_id as key
        assert task_id in receiver.tensors
        assert receiver.tensors[task_id] == random_torch_tensors

        # Step 2: Set context with received tensors
        # Mock get_prop to return task_id first, then shareable
        setup_mock_get_prop_with_task_id(mock_fl_context, task_id, sample_shareable_with_dxo)

        receiver.set_ctx_with_tensors(mock_fl_context)

        # Verify final state
        call_args = mock_fl_context.set_prop.call_args
        updated_shareable = call_args.args[1]
        dxo_data = updated_shareable["DXO"]["data"]

        # Verify all tensors made it through the complete workflow
        assert len(dxo_data) == len(random_torch_tensors)
        for name, original_tensor in random_torch_tensors.items():
            assert name in dxo_data
            assert torch.allclose(dxo_data[name], original_tensor)

        # Verify cleanup
        assert task_id not in receiver.tensors
