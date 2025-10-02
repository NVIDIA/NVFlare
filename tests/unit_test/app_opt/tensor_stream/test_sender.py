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

import numpy as np
import pytest
import torch

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.job_def import SERVER_SITE_NAME
from nvflare.app_opt.tensor_stream.sender import TensorSender
from nvflare.app_opt.tensor_stream.types import TENSORS_CHANNEL, TensorTopics
from nvflare.client.config import ExchangeFormat


class TestTensorSender:
    """Test cases for TensorSender class."""

    @pytest.mark.parametrize(
        "channel,expected_channel", [("custom_channel", "custom_channel"), (None, TENSORS_CHANNEL)]  # Default channel
    )
    def test_init(self, mock_streamable_engine, channel, expected_channel):
        """Test TensorSender initialization with various parameters."""
        ctx_prop_key = FLContextKey.TASK_DATA
        root_keys = ["encoder", "decoder"]

        if channel is None:
            sender = TensorSender(engine=mock_streamable_engine, ctx_prop_key=ctx_prop_key, root_keys=root_keys)
        else:
            sender = TensorSender(
                engine=mock_streamable_engine, ctx_prop_key=ctx_prop_key, root_keys=root_keys, channel=channel
            )

        assert sender.engine == mock_streamable_engine
        assert sender.ctx_prop_key == ctx_prop_key
        assert sender.root_keys == root_keys
        assert sender.channel == expected_channel
        assert sender.logger is not None

    @patch("nvflare.app_opt.tensor_stream.sender.get_targets_for_ctx_and_prop_key")
    @patch("nvflare.app_opt.tensor_stream.sender.get_topic_for_ctx_prop_key")
    def test_send_with_weights(
        self, mock_get_topic, mock_get_targets, mock_streamable_engine, mock_fl_context, sample_dxo_weights
    ):
        """Test sending tensors with WEIGHTS data kind."""
        # Setup mocks
        mock_get_targets.return_value = ["client1"]
        mock_get_topic.return_value = TensorTopics.TASK_DATA

        # Setup FL context with DXO data
        shareable = sample_dxo_weights.to_shareable()
        mock_fl_context.get_prop.return_value = shareable

        # Create sender
        sender = TensorSender(
            engine=mock_streamable_engine,
            ctx_prop_key=FLContextKey.TASK_DATA,
            root_keys=[""],  # Empty string for top-level tensors
        )

        # Send tensors
        entry_timeout = 5.0
        sender.send(mock_fl_context, entry_timeout)

        # Verify engine.stream_objects was called
        mock_streamable_engine.stream_objects.assert_called_once()
        call_args = mock_streamable_engine.stream_objects.call_args

        assert call_args.kwargs["channel"] == TENSORS_CHANNEL
        assert call_args.kwargs["topic"] == TensorTopics.TASK_DATA
        assert call_args.kwargs["targets"] == ["client1"]
        assert call_args.kwargs["fl_ctx"] == mock_fl_context
        assert call_args.kwargs["optional"] is False
        assert call_args.kwargs["secure"] is False

        # Verify producer is TorchTensorsProducer
        producer = call_args.kwargs["producer"]
        assert producer.__class__.__name__ == "TorchTensorsProducer"
        assert producer.entry_timeout == entry_timeout
        assert producer.root_key == ""

    @patch("nvflare.app_opt.tensor_stream.sender.get_targets_for_ctx_and_prop_key")
    @patch("nvflare.app_opt.tensor_stream.sender.get_topic_for_ctx_prop_key")
    def test_send_with_nested_weights(
        self, mock_get_topic, mock_get_targets, mock_streamable_engine, mock_fl_context, sample_dxo_nested_weights
    ):
        """Test sending tensors with nested structure (multiple root keys)."""
        # Setup mocks
        mock_get_targets.return_value = ["client1"]
        mock_get_topic.return_value = TensorTopics.TASK_DATA

        # Setup FL context with nested DXO data
        shareable = sample_dxo_nested_weights.to_shareable()
        mock_fl_context.get_prop.return_value = shareable

        # Create sender with multiple root keys
        root_keys = ["encoder", "decoder"]
        sender = TensorSender(engine=mock_streamable_engine, ctx_prop_key=FLContextKey.TASK_DATA, root_keys=root_keys)

        # Send tensors
        entry_timeout = 3.0
        sender.send(mock_fl_context, entry_timeout)

        # Verify engine.stream_objects was called twice (once per root key)
        assert mock_streamable_engine.stream_objects.call_count == 2

        # Check both calls
        calls = mock_streamable_engine.stream_objects.call_args_list
        for i, call in enumerate(calls):
            producer = call.kwargs["producer"]
            assert producer.__class__.__name__ == "TorchTensorsProducer"
            assert producer.entry_timeout == entry_timeout
            assert producer.root_key == root_keys[i]

    @patch("nvflare.app_opt.tensor_stream.sender.get_targets_for_ctx_and_prop_key")
    @patch("nvflare.app_opt.tensor_stream.sender.get_topic_for_ctx_prop_key")
    def test_send_with_weight_diff(
        self, mock_get_topic, mock_get_targets, mock_streamable_engine, mock_fl_context, sample_dxo_weight_diff
    ):
        """Test sending tensors with WEIGHT_DIFF data kind."""
        # Setup mocks
        mock_get_targets.return_value = [SERVER_SITE_NAME]
        mock_get_topic.return_value = TensorTopics.TASK_RESULT

        # Setup FL context with weight diff DXO data
        shareable = sample_dxo_weight_diff.to_shareable()
        mock_fl_context.get_prop.return_value = shareable

        # Create sender
        sender = TensorSender(engine=mock_streamable_engine, ctx_prop_key=FLContextKey.TASK_RESULT, root_keys=[""])

        # Send tensors
        entry_timeout = 2.0
        sender.send(mock_fl_context, entry_timeout)

        # Verify engine.stream_objects was called
        mock_streamable_engine.stream_objects.assert_called_once()
        call_args = mock_streamable_engine.stream_objects.call_args

        assert call_args.kwargs["targets"] == [SERVER_SITE_NAME]
        assert call_args.kwargs["topic"] == TensorTopics.TASK_RESULT

        producer = call_args.kwargs["producer"]
        assert producer.entry_timeout == entry_timeout

    def test_get_dxo_from_ctx_no_data(self, mock_streamable_engine, mock_fl_context):
        """Test _get_dxo_from_ctx when no data is available."""
        mock_fl_context.get_prop.return_value = None

        sender = TensorSender(engine=mock_streamable_engine, ctx_prop_key=FLContextKey.TASK_DATA, root_keys=[""])

        result = sender._get_dxo_from_ctx(mock_fl_context)
        assert result is None

    def test_get_dxo_from_ctx_wrong_data_kind(self, mock_streamable_engine, mock_fl_context):
        """Test _get_dxo_from_ctx with unsupported data kind."""
        # Create DXO with unsupported data kind
        dxo = DXO(data_kind=DataKind.METRICS, data={"accuracy": 0.95})
        shareable = dxo.to_shareable()
        mock_fl_context.get_prop.return_value = shareable

        sender = TensorSender(engine=mock_streamable_engine, ctx_prop_key=FLContextKey.TASK_DATA, root_keys=[""])

        result = sender._get_dxo_from_ctx(mock_fl_context)
        assert result is None

    @pytest.mark.parametrize(
        "tensor_format, root_key, test_data",
        [
            (ExchangeFormat.PYTORCH, "", {"layer1.weight": torch.randn(2, 3), "layer1.bias": torch.randn(2)}),
            (
                ExchangeFormat.NUMPY,
                "",
                {
                    "layer1.weight": np.random.randn(2, 3).astype(np.float32),
                    "layer1.bias": np.random.randn(2).astype(np.float32),
                },
            ),
            (
                ExchangeFormat.PYTORCH,
                "encoder",
                {
                    "encoder": {"layer1.weight": torch.randn(2, 3), "layer1.bias": torch.randn(2)},
                },
            ),
            (
                ExchangeFormat.NUMPY,
                "encoder",
                {
                    "encoder": {
                        "layer1.weight": np.random.randn(2, 3).astype(np.float32),
                        "layer1.bias": np.random.randn(2).astype(np.float32),
                    },
                },
            ),
        ],
    )
    def test_get_tensors_from_dxo_success_cases(self, mock_streamable_engine, tensor_format, root_key, test_data):
        """Test _get_tensors_from_dxo with various successful scenarios and tensor formats."""
        sender = TensorSender(
            engine=mock_streamable_engine,
            ctx_prop_key=FLContextKey.TASK_DATA,
            format=tensor_format,
            root_keys=[""],
        )

        dxo_weights = DXO(data_kind=DataKind.WEIGHTS, data=test_data)
        tensors = sender._get_tensors_from_dxo(dxo_weights, key=root_key)
        for name, tensor in tensors.items():
            # Always check if returned value is torch.Tensor
            assert isinstance(tensor, torch.Tensor)
            if root_key:
                original_data = test_data[root_key][name]
            else:
                original_data = test_data[name]
            if tensor_format == ExchangeFormat.NUMPY:
                original_data = torch.from_numpy(original_data)

            # For torch format, should be the same tensor
            assert tensor.shape == original_data.shape
            assert tensor.dtype == original_data.dtype
            assert torch.allclose(tensor, original_data)

    @pytest.mark.parametrize(
        "dxo_data,key,expected_error,error_message",
        [
            ({}, "", ValueError, "No tensor data found on the context shareable"),  # Empty data
            (
                {"valid_key": {"tensor": torch.randn(2, 2)}},
                "nonexistent_key",
                ValueError,
                "No tensor data found on the context shareable",
            ),  # Non-existent key
            (
                {"invalid_tensor": "this_is_a_string"},
                "",
                ValueError,
                "Expected torch.Tensor for key 'invalid_tensor', but got <class 'str'>",
            ),  # Unsupported type
        ],
    )
    def test_get_tensors_from_dxo_error_cases(
        self, mock_streamable_engine, dxo_data, key, expected_error, error_message
    ):
        """Test _get_tensors_from_dxo error handling."""
        dxo = DXO(data_kind=DataKind.WEIGHTS, data=dxo_data)
        sender = TensorSender(engine=mock_streamable_engine, ctx_prop_key=FLContextKey.TASK_DATA, root_keys=[""])

        with pytest.raises(expected_error, match=error_message):
            sender._get_tensors_from_dxo(dxo, key=key)

    def test_numpy_to_torch_conversion(self, mock_streamable_engine):
        """Test conversion from numpy arrays to torch tensors with different dtypes."""
        numpy_data = {
            "float32_array": np.random.randn(3, 4).astype(np.float32),
            "float64_array": np.random.randn(2, 5).astype(np.float64),
            "int32_array": np.random.randint(0, 100, (2, 3)).astype(np.int32),
            "int64_array": np.random.randint(0, 100, (4, 2)).astype(np.int64),
        }

        dxo = DXO(data_kind=DataKind.WEIGHTS, data=numpy_data)

        sender = TensorSender(
            engine=mock_streamable_engine,
            ctx_prop_key=FLContextKey.TASK_DATA,
            format=ExchangeFormat.NUMPY,
            root_keys=[""],
        )

        tensors = sender._get_tensors_from_dxo(dxo, key="")

        for name, tensor in tensors.items():
            assert isinstance(tensor, torch.Tensor)
            original_array = numpy_data[name]

            # Verify shape and dtype are preserved
            assert tensor.shape == torch.Size(original_array.shape)
            expected_dtype = getattr(torch, str(original_array.dtype).split(".")[-1])
            assert tensor.dtype == expected_dtype

            # Verify data is the same
            assert torch.allclose(tensor, torch.from_numpy(original_array))

    @patch("nvflare.app_opt.tensor_stream.sender.get_targets_for_ctx_and_prop_key")
    @patch("nvflare.app_opt.tensor_stream.sender.get_topic_for_ctx_prop_key")
    def test_send_integration_with_custom_channel(
        self, mock_get_topic, mock_get_targets, mock_streamable_engine, mock_fl_context, sample_dxo_weights
    ):
        """Test complete send workflow with custom channel."""
        # Setup mocks
        mock_get_targets.return_value = ["client1", "client2"]
        mock_get_topic.return_value = TensorTopics.TASK_DATA

        # Setup FL context
        shareable = sample_dxo_weights.to_shareable()
        mock_fl_context.get_prop.return_value = shareable

        # Create sender with custom channel
        custom_channel = "custom_tensor_channel"
        sender = TensorSender(
            engine=mock_streamable_engine, ctx_prop_key=FLContextKey.TASK_DATA, root_keys=[""], channel=custom_channel
        )

        # Send tensors
        sender.send(mock_fl_context, entry_timeout=1.0)

        # Verify correct channel was used
        call_args = mock_streamable_engine.stream_objects.call_args
        assert call_args.kwargs["channel"] == custom_channel
        assert call_args.kwargs["targets"] == ["client1", "client2"]

    @patch("nvflare.app_opt.tensor_stream.sender.get_targets_for_ctx_and_prop_key")
    @patch("nvflare.app_opt.tensor_stream.sender.get_topic_for_ctx_prop_key")
    def test_send_with_no_tensors_in_root_key(
        self, mock_get_topic, mock_get_targets, mock_streamable_engine, mock_fl_context, sample_dxo_nested_weights
    ):
        """Test send behavior when a root key has no tensors."""
        # Setup mocks
        mock_get_targets.return_value = ["client1"]
        mock_get_topic.return_value = TensorTopics.TASK_DATA

        # Setup FL context
        shareable = sample_dxo_nested_weights.to_shareable()
        mock_fl_context.get_prop.return_value = shareable

        # Create sender with root key that doesn't exist in the DXO
        sender = TensorSender(
            engine=mock_streamable_engine, ctx_prop_key=FLContextKey.TASK_DATA, root_keys=["nonexistent_key"]
        )

        # Send should raise ValueError for nonexistent key
        with pytest.raises(ValueError, match="No tensor data found on the context shareable"):
            sender.send(mock_fl_context, entry_timeout=1.0)
