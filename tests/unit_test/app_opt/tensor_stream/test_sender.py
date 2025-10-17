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
        format = ExchangeFormat.PYTORCH
        tasks = ["train", "validate"]

        if channel is None:
            sender = TensorSender(
                engine=mock_streamable_engine,
                ctx_prop_key=ctx_prop_key,
                format=format,
                tasks=tasks,
            )
        else:
            sender = TensorSender(
                engine=mock_streamable_engine,
                ctx_prop_key=ctx_prop_key,
                format=format,
                tasks=tasks,
                channel=channel,
            )

        assert sender.engine == mock_streamable_engine
        assert sender.ctx_prop_key == ctx_prop_key
        assert sender.format == format
        assert sender.tasks == tasks
        assert sender.channel == expected_channel
        assert sender.logger is not None

    @pytest.mark.parametrize(
        "dxo_fixture,ctx_prop_key,tasks,expected_topic,expected_targets",
        [
            ("sample_dxo_weights", FLContextKey.TASK_DATA, ["train"], TensorTopics.TASK_DATA, ["client1"]),
            (
                "sample_dxo_weight_diff",
                FLContextKey.TASK_RESULT,
                ["submit_model"],
                TensorTopics.TASK_RESULT,
                [SERVER_SITE_NAME],
            ),
            ("sample_dxo_nested_weights", FLContextKey.TASK_DATA, ["train"], TensorTopics.TASK_DATA, ["client1"]),
        ],
    )
    @patch("nvflare.app_opt.tensor_stream.sender.get_targets_for_ctx_and_prop_key")
    @patch("nvflare.app_opt.tensor_stream.sender.get_topic_for_ctx_prop_key")
    @patch("nvflare.app_opt.tensor_stream.sender.get_dxo_from_ctx")
    def test_send_with_different_data_kinds(
        self,
        mock_get_dxo,
        mock_get_topic,
        mock_get_targets,
        mock_streamable_engine,
        mock_fl_context,
        dxo_fixture,
        ctx_prop_key,
        tasks,
        expected_topic,
        expected_targets,
        request,
    ):
        """Test sending tensors with different data kinds and structures."""
        # Get the DXO fixture dynamically
        sample_dxo = request.getfixturevalue(dxo_fixture)

        # Setup mocks
        mock_get_targets.return_value = expected_targets
        mock_get_topic.return_value = expected_topic
        mock_get_dxo.return_value = sample_dxo

        # Setup FL context with task_id
        task_id = "test_task_123"
        mock_fl_context.get_prop.return_value = task_id

        # Create sender
        sender = TensorSender(
            engine=mock_streamable_engine,
            ctx_prop_key=ctx_prop_key,
            format=ExchangeFormat.PYTORCH,
            tasks=tasks,
        )

        # Step 1: Store tensors first
        sender.store_tensors(mock_fl_context)

        # Verify tensors were stored
        assert task_id in sender.task_params
        assert sender.task_params[task_id] == sample_dxo.data

        # Step 2: Send tensors
        entry_timeout = 5.0
        sender.send(mock_fl_context, entry_timeout)

        # Verify utility functions were called
        mock_get_dxo.assert_called_once_with(mock_fl_context, ctx_prop_key, tasks)
        mock_get_targets.assert_called_once_with(mock_fl_context, ctx_prop_key)
        mock_get_topic.assert_called_once_with(ctx_prop_key)

        # Verify engine.stream_objects was called
        assert mock_streamable_engine.stream_objects.call_count == 1

        # Verify call parameters
        call_args = mock_streamable_engine.stream_objects.call_args
        assert call_args.kwargs["channel"] == TENSORS_CHANNEL
        assert call_args.kwargs["topic"] == expected_topic
        assert call_args.kwargs["targets"] == expected_targets
        assert call_args.kwargs["fl_ctx"] == mock_fl_context
        assert call_args.kwargs["optional"] is False
        assert call_args.kwargs["secure"] is False

        # Verify producer
        producer = call_args.kwargs["producer"]
        assert producer.__class__.__name__ == "TensorProducer"
        assert producer.entry_timeout == entry_timeout

        # Verify tensors were removed after sending
        assert task_id not in sender.task_params

    @pytest.mark.parametrize(
        "error_message",
        [
            "No task found in FLContext",
            "Skipping task, data kind is not WEIGHTS or WEIGHT_DIFF",
        ],
    )
    def test_store_tensors_error_conditions(self, mock_streamable_engine, mock_fl_context, error_message):
        """Test store_tensors returns False when encountering various error conditions."""
        with patch("nvflare.app_opt.tensor_stream.sender.get_dxo_from_ctx") as mock_get_dxo:
            mock_get_dxo.side_effect = ValueError(error_message)

            # Setup task_id
            task_id = "test_task_error"
            mock_fl_context.get_prop.return_value = task_id

            sender = TensorSender(
                engine=mock_streamable_engine,
                ctx_prop_key=FLContextKey.TASK_DATA,
                format=ExchangeFormat.PYTORCH,
                tasks=["train"],
            )

            result = sender.store_tensors(mock_fl_context)
            assert result is False

            # Verify no tensors were stored
            assert task_id not in sender.task_params

    def test_send_without_store_raises_error(self, mock_streamable_engine, mock_fl_context):
        """Test send raises ValueError when tensors haven't been stored."""
        task_id = "test_task_missing"
        mock_fl_context.get_prop.return_value = task_id

        sender = TensorSender(
            engine=mock_streamable_engine,
            ctx_prop_key=FLContextKey.TASK_DATA,
            format=ExchangeFormat.PYTORCH,
            tasks=["train"],
        )

        # Try to send without storing first - should raise ValueError
        with pytest.raises(ValueError, match="No tensors stored for peer"):
            sender.send(mock_fl_context, entry_timeout=1.0)

    @pytest.mark.parametrize(
        "channel,expected_channel,targets",
        [
            ("custom_tensor_channel", "custom_tensor_channel", ["client1", "client2"]),
            (None, TENSORS_CHANNEL, ["client1"]),
        ],
    )
    @patch("nvflare.app_opt.tensor_stream.sender.get_targets_for_ctx_and_prop_key")
    @patch("nvflare.app_opt.tensor_stream.sender.get_topic_for_ctx_prop_key")
    @patch("nvflare.app_opt.tensor_stream.sender.get_dxo_from_ctx")
    def test_send_with_custom_channel(
        self,
        mock_get_dxo,
        mock_get_topic,
        mock_get_targets,
        mock_streamable_engine,
        mock_fl_context,
        sample_dxo_weights,
        channel,
        expected_channel,
        targets,
    ):
        """Test sending tensors with custom channel configuration."""
        # Setup mocks
        mock_get_targets.return_value = targets
        mock_get_topic.return_value = TensorTopics.TASK_DATA
        mock_get_dxo.return_value = sample_dxo_weights

        # Setup FL context with task_id
        task_id = "test_task_channel"
        mock_fl_context.get_prop.return_value = task_id

        # Create sender with optional custom channel
        kwargs = {
            "engine": mock_streamable_engine,
            "ctx_prop_key": FLContextKey.TASK_DATA,
            "format": ExchangeFormat.PYTORCH,
            "tasks": ["train"],
        }
        if channel is not None:
            kwargs["channel"] = channel
        sender = TensorSender(**kwargs)

        # Step 1: Store tensors
        sender.store_tensors(mock_fl_context)

        # Step 2: Send tensors
        sender.send(mock_fl_context, entry_timeout=1.0)

        # Verify correct channel and targets were used
        call_args = mock_streamable_engine.stream_objects.call_args
        assert call_args.kwargs["channel"] == expected_channel
        assert call_args.kwargs["targets"] == targets

    @pytest.mark.parametrize(
        "dxo_fixture,expected_tensor_count",
        [
            ("sample_dxo_nested_weights", 2),  # Nested structure has 2 root keys
            ("sample_dxo_weights", 10),  # Flat structure has 10 tensors
        ],
    )
    @patch("nvflare.app_opt.tensor_stream.sender.get_targets_for_ctx_and_prop_key")
    @patch("nvflare.app_opt.tensor_stream.sender.get_topic_for_ctx_prop_key")
    @patch("nvflare.app_opt.tensor_stream.sender.get_dxo_from_ctx")
    def test_send_with_different_data_structures(
        self,
        mock_get_dxo,
        mock_get_topic,
        mock_get_targets,
        mock_streamable_engine,
        mock_fl_context,
        dxo_fixture,
        expected_tensor_count,
        request,
    ):
        """Test sending tensors with different data structures (flat vs nested)."""
        # Get the DXO fixture dynamically
        sample_dxo = request.getfixturevalue(dxo_fixture)

        # Setup mocks
        mock_get_targets.return_value = ["client1"]
        mock_get_topic.return_value = TensorTopics.TASK_DATA
        mock_get_dxo.return_value = sample_dxo

        # Setup FL context with task_id
        task_id = "test_task_structure"
        mock_fl_context.get_prop.return_value = task_id

        sender = TensorSender(
            engine=mock_streamable_engine,
            ctx_prop_key=FLContextKey.TASK_DATA,
            format=ExchangeFormat.PYTORCH,
            tasks=["train"],
        )

        # Step 1: Store tensors
        sender.store_tensors(mock_fl_context)

        # Verify stored data matches expected structure
        assert task_id in sender.task_params
        assert len(sender.task_params[task_id]) == expected_tensor_count

        # Step 2: Send tensors
        sender.send(mock_fl_context, entry_timeout=3.0)

        # Verify stream_objects was called once (sends all params as one producer)
        assert mock_streamable_engine.stream_objects.call_count == 1

        # Verify producer has the correct task_id and keys
        call_args = mock_streamable_engine.stream_objects.call_args
        producer = call_args.kwargs["producer"]
        assert producer.__class__.__name__ == "TensorProducer"
        assert producer.task_id == task_id
        # Verify the producer has the correct tensor keys
        expected_keys = list(sample_dxo.data.keys())
        assert set(producer.tensors_keys) == set(expected_keys)
