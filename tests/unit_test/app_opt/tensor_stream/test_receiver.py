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
from nvflare.app_opt.tensor_stream.consumer import TorchTensorsConsumerFactory
from nvflare.app_opt.tensor_stream.receiver import TensorReceiver
from nvflare.app_opt.tensor_stream.types import SAFE_TENSORS_PROP_KEY, TENSORS_CHANNEL, TensorTopics


class TestTensorReceiver:
    """Test cases for TensorReceiver class."""

    @pytest.mark.parametrize(
        "ctx_prop_key,format_type,channel,expected_topic,expected_channel",
        [
            (FLContextKey.TASK_DATA, "torch", "custom_channel", TensorTopics.TASK_DATA, "custom_channel"),
            (FLContextKey.TASK_RESULT, "numpy", None, TensorTopics.TASK_RESULT, TENSORS_CHANNEL),  # Default channel
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
        assert isinstance(call_args.kwargs["factory"], TorchTensorsConsumerFactory)
        assert call_args.kwargs["stream_done_cb"] == receiver._save_tensors_cb

        mock_get_topic.assert_called_once_with(ctx_prop_key)

    def test_save_tensors_cb_success(self, mock_streamable_engine, mock_fl_context, random_torch_tensors):
        """Test _save_tensors_cb with successful tensor reception."""
        receiver = TensorReceiver(engine=mock_streamable_engine, ctx_prop_key=FLContextKey.TASK_DATA, format="torch")

        # Setup FL context with received tensors
        peer_name = "test_peer"
        mock_fl_context.get_peer_context().get_identity_name.return_value = peer_name
        mock_fl_context.set_custom_prop(SAFE_TENSORS_PROP_KEY, random_torch_tensors)

        # Call the callback with success=True
        receiver._save_tensors_cb(True, mock_fl_context)

        # Verify tensors were stored
        assert peer_name in receiver.tensors
        assert receiver.tensors[peer_name] == random_torch_tensors

    def test_save_tensors_cb_failure(self, mock_streamable_engine, mock_fl_context):
        """Test _save_tensors_cb with failed tensor reception."""
        receiver = TensorReceiver(engine=mock_streamable_engine, ctx_prop_key=FLContextKey.TASK_DATA, format="torch")

        peer_name = "test_client"
        mock_fl_context.get_peer_context().get_identity_name.return_value = peer_name

        # Call the callback with success=False
        receiver._save_tensors_cb(False, mock_fl_context)

        # Verify no tensors were stored
        assert peer_name not in receiver.tensors
        assert len(receiver.tensors) == 0

    def test_save_tensors_cb_no_tensors(self, mock_streamable_engine, mock_fl_context):
        """Test _save_tensors_cb when no tensors are found in context."""
        receiver = TensorReceiver(engine=mock_streamable_engine, ctx_prop_key=FLContextKey.TASK_DATA, format="torch")

        peer_name = "test_client"
        mock_fl_context.get_peer_context().get_identity_name.return_value = peer_name
        mock_fl_context.set_custom_prop(SAFE_TENSORS_PROP_KEY, None)

        # Call the callback with success=True but no tensors
        receiver._save_tensors_cb(True, mock_fl_context)

        # Verify no tensors were stored
        assert peer_name not in receiver.tensors
        assert len(receiver.tensors) == 0

    def test_save_tensors_cb_nested_tensors(self, mock_streamable_engine, mock_fl_context, sample_nested_tensors):
        """Test _save_tensors_cb with nested tensor structure."""
        receiver = TensorReceiver(engine=mock_streamable_engine, ctx_prop_key=FLContextKey.TASK_DATA, format="torch")

        peer_name = "test_client"
        mock_fl_context.get_peer_context().get_identity_name.return_value = peer_name
        mock_fl_context.set_custom_prop(SAFE_TENSORS_PROP_KEY, sample_nested_tensors)

        # Call the callback
        receiver._save_tensors_cb(True, mock_fl_context)

        # Verify nested tensors were stored correctly
        assert peer_name in receiver.tensors
        assert receiver.tensors[peer_name] == sample_nested_tensors
        assert "encoder" in receiver.tensors[peer_name]
        assert "decoder" in receiver.tensors[peer_name]

    def test_set_ctx_with_tensors_torch_format(
        self, mock_streamable_engine, mock_fl_context, sample_shareable_with_dxo, random_torch_tensors
    ):
        """Test set_ctx_with_tensors with torch format."""
        receiver = TensorReceiver(engine=mock_streamable_engine, ctx_prop_key=FLContextKey.TASK_DATA, format="torch")

        # Setup: store tensors from peer
        peer_name = "test_client"
        receiver.tensors[peer_name] = random_torch_tensors

        # Setup FL context
        mock_fl_context.get_peer_context().get_identity_name.return_value = peer_name
        mock_fl_context.get_prop.return_value = sample_shareable_with_dxo

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
        assert peer_name not in receiver.tensors

    def test_set_ctx_with_tensors_numpy_format(
        self, mock_streamable_engine, mock_fl_context, sample_shareable_with_dxo, random_torch_tensors
    ):
        """Test set_ctx_with_tensors with numpy format conversion."""
        receiver = TensorReceiver(
            engine=mock_streamable_engine, ctx_prop_key=FLContextKey.TASK_DATA, format="numpy"  # Different format
        )

        # Setup: store tensors from peer
        peer_name = "test_client"
        receiver.tensors[peer_name] = random_torch_tensors

        # Setup FL context
        mock_fl_context.get_peer_context().get_identity_name.return_value = peer_name
        mock_fl_context.get_prop.return_value = sample_shareable_with_dxo

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
        receiver = TensorReceiver(engine=mock_streamable_engine, ctx_prop_key=FLContextKey.TASK_RESULT, format="torch")

        # Setup: store tensors from peer
        peer_name = "test_client"
        receiver.tensors[peer_name] = random_torch_tensors

        # Setup FL context
        mock_fl_context.get_peer_context().get_identity_name.return_value = peer_name
        mock_fl_context.get_prop.return_value = sample_shareable_with_weight_diff_dxo

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
        receiver = TensorReceiver(engine=mock_streamable_engine, ctx_prop_key=FLContextKey.TASK_DATA, format="torch")

        # Setup: store nested tensors from peer
        peer_name = "test_client"
        receiver.tensors[peer_name] = sample_nested_tensors

        # Setup FL context
        mock_fl_context.get_peer_context().get_identity_name.return_value = peer_name
        mock_fl_context.get_prop.return_value = sample_shareable_with_dxo

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
        """Test set_ctx_with_tensors when no tensors are stored for peer."""
        receiver = TensorReceiver(engine=mock_streamable_engine, ctx_prop_key=FLContextKey.TASK_DATA, format="torch")

        # Setup FL context without storing any tensors
        peer_name = "test_client"
        mock_fl_context.get_peer_context().get_identity_name.return_value = peer_name
        mock_fl_context.get_prop.return_value = sample_shareable_with_dxo

        # Should raise RuntimeError
        with pytest.raises(RuntimeError, match="No tensors found for peer test_client"):
            receiver.set_ctx_with_tensors(mock_fl_context)

    def test_set_ctx_with_tensors_no_shareable(self, mock_streamable_engine, mock_fl_context, random_torch_tensors):
        """Test set_ctx_with_tensors when no shareable is found in context."""
        receiver = TensorReceiver(engine=mock_streamable_engine, ctx_prop_key=FLContextKey.TASK_DATA, format="torch")

        # Setup: store tensors but no shareable in context
        peer_name = "test_client"
        receiver.tensors[peer_name] = random_torch_tensors

        mock_fl_context.get_peer_context().get_identity_name.return_value = peer_name
        mock_fl_context.get_prop.return_value = None  # No shareable

        # Should raise RuntimeError
        with pytest.raises(RuntimeError, match="No shareable found in FLContext"):
            receiver.set_ctx_with_tensors(mock_fl_context)

    def test_set_ctx_with_tensors_no_dxo(self, mock_streamable_engine, mock_fl_context, random_torch_tensors):
        """Test set_ctx_with_tensors when shareable contains no DXO."""
        receiver = TensorReceiver(engine=mock_streamable_engine, ctx_prop_key=FLContextKey.TASK_DATA, format="torch")

        # Setup: store tensors and empty shareable
        peer_name = "test_client"
        receiver.tensors[peer_name] = random_torch_tensors

        empty_shareable = Shareable()  # No DXO key
        mock_fl_context.get_peer_context().get_identity_name.return_value = peer_name
        mock_fl_context.get_prop.return_value = empty_shareable

        # Should raise RuntimeError
        with pytest.raises(RuntimeError, match="No DXO found in shareable"):
            receiver.set_ctx_with_tensors(mock_fl_context)

    def test_set_ctx_with_tensors_invalid_data_kind(
        self, mock_streamable_engine, mock_fl_context, random_torch_tensors
    ):
        """Test set_ctx_with_tensors with invalid DXO data kind."""
        receiver = TensorReceiver(engine=mock_streamable_engine, ctx_prop_key=FLContextKey.TASK_DATA, format="torch")

        # Setup: store tensors and shareable with invalid data kind
        peer_name = "test_client"
        receiver.tensors[peer_name] = random_torch_tensors

        invalid_shareable = Shareable()
        invalid_shareable["DXO"] = {"kind": DataKind.METRICS, "data": {}}  # Invalid data kind

        mock_fl_context.get_peer_context().get_identity_name.return_value = peer_name
        mock_fl_context.get_prop.return_value = invalid_shareable

        # Should raise RuntimeError
        with pytest.raises(RuntimeError, match="Task data kind is not WEIGHTS or WEIGHT_DIFF"):
            receiver.set_ctx_with_tensors(mock_fl_context)

    def test_multiple_peers_tensors(self, mock_streamable_engine, mock_fl_context, random_torch_tensors):
        """Test handling tensors from multiple peers."""
        receiver = TensorReceiver(engine=mock_streamable_engine, ctx_prop_key=FLContextKey.TASK_DATA, format="torch")

        # Simulate receiving tensors from multiple peers
        peer1_tensors = {k: v for k, v in list(random_torch_tensors.items())[:5]}
        peer2_tensors = {k: v for k, v in list(random_torch_tensors.items())[5:]}

        # Store tensors for both peers
        receiver.tensors["client1"] = peer1_tensors
        receiver.tensors["client2"] = peer2_tensors

        assert len(receiver.tensors) == 2
        assert "client1" in receiver.tensors
        assert "client2" in receiver.tensors

        # Verify each peer's tensors are stored correctly
        assert receiver.tensors["client1"] == peer1_tensors
        assert receiver.tensors["client2"] == peer2_tensors

    def test_tensor_cleanup_after_use(
        self, mock_streamable_engine, mock_fl_context, sample_shareable_with_dxo, random_torch_tensors
    ):
        """Test that tensors are cleaned up after being used in set_ctx_with_tensors."""
        receiver = TensorReceiver(engine=mock_streamable_engine, ctx_prop_key=FLContextKey.TASK_DATA, format="torch")

        # Setup: store tensors for multiple peers
        receiver.tensors["client1"] = random_torch_tensors
        receiver.tensors["client2"] = {k: v * 2 for k, v in random_torch_tensors.items()}

        # Use tensors from client1
        mock_fl_context.get_peer_context().get_identity_name.return_value = "client1"
        mock_fl_context.get_prop.return_value = sample_shareable_with_dxo

        receiver.set_ctx_with_tensors(mock_fl_context)

        # Verify client1 tensors were removed but client2 tensors remain
        assert "client1" not in receiver.tensors
        assert "client2" in receiver.tensors
        assert len(receiver.tensors) == 1

    @patch("nvflare.app_opt.tensor_stream.receiver.get_topic_for_ctx_prop_key")
    def test_integration_receive_and_set_workflow(
        self, mock_get_topic, mock_streamable_engine, mock_fl_context, sample_shareable_with_dxo, random_torch_tensors
    ):
        """Integration test for complete receive and set workflow."""
        mock_get_topic.return_value = TensorTopics.TASK_DATA

        # Create receiver
        receiver = TensorReceiver(engine=mock_streamable_engine, ctx_prop_key=FLContextKey.TASK_DATA, format="torch")

        peer_name = "integration_client"

        # Step 1: Simulate receiving tensors (callback called by streaming engine)
        mock_fl_context.get_peer_context().get_identity_name.return_value = peer_name
        mock_fl_context.set_custom_prop(SAFE_TENSORS_PROP_KEY, random_torch_tensors)

        receiver._save_tensors_cb(True, mock_fl_context)

        # Verify tensors were stored
        assert peer_name in receiver.tensors
        assert receiver.tensors[peer_name] == random_torch_tensors

        # Step 2: Set context with received tensors
        mock_fl_context.get_prop.return_value = sample_shareable_with_dxo

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
        assert peer_name not in receiver.tensors
