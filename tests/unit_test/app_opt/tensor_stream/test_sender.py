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
from nvflare.apis.fl_constant import FLContextKey, ReservedKey
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
        assert sender.root_keys == []  # Should start empty and be auto-detected
        assert sender.format == format
        assert sender.tasks == tasks
        assert sender.channel == expected_channel
        assert sender.logger is not None

    @pytest.mark.parametrize(
        "dxo_fixture,ctx_prop_key,tasks,expected_topic,expected_targets,root_key_count",
        [
            ("sample_dxo_weights", FLContextKey.TASK_DATA, ["train"], TensorTopics.TASK_DATA, ["client1"], 1),
            (
                "sample_dxo_weight_diff",
                FLContextKey.TASK_RESULT,
                ["submit_model"],
                TensorTopics.TASK_RESULT,
                [SERVER_SITE_NAME],
                1,
            ),
            ("sample_dxo_nested_weights", FLContextKey.TASK_DATA, ["train"], TensorTopics.TASK_DATA, ["client1"], 2),
        ],
    )
    @patch("nvflare.app_opt.tensor_stream.sender.get_targets_for_ctx_and_prop_key")
    @patch("nvflare.app_opt.tensor_stream.sender.get_topic_for_ctx_prop_key")
    @patch("nvflare.app_opt.tensor_stream.sender.get_dxo_from_ctx")
    @patch("nvflare.app_opt.tensor_stream.sender.get_tensors_from_dxo")
    def test_send_with_different_data_kinds(
        self,
        mock_get_tensors,
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
        root_key_count,
        request,
    ):
        """Test sending tensors with different data kinds and structures."""
        # Get the DXO fixture dynamically
        sample_dxo = request.getfixturevalue(dxo_fixture)

        # Setup mocks
        mock_get_targets.return_value = expected_targets
        mock_get_topic.return_value = expected_topic
        mock_get_dxo.return_value = sample_dxo

        # Setup mock for nested vs flat data
        if root_key_count > 1:
            # For nested data, mock different calls for each key
            def mock_tensors_side_effect(dxo, key, format):
                if key in sample_dxo.data:
                    return sample_dxo.data[key]
                return {}

            mock_get_tensors.side_effect = mock_tensors_side_effect
        else:
            # For flat data, return the data directly
            mock_get_tensors.return_value = sample_dxo.data

        # Setup FL context with DXO data
        shareable = sample_dxo.to_shareable()
        shareable.set_header(ReservedKey.TASK_NAME, tasks[0])
        mock_fl_context.get_prop.return_value = shareable

        # Create sender
        sender = TensorSender(
            engine=mock_streamable_engine,
            ctx_prop_key=ctx_prop_key,
            format=ExchangeFormat.PYTORCH,
            tasks=tasks,
        )

        # Send tensors
        entry_timeout = 5.0
        result = sender.send(mock_fl_context, entry_timeout)

        # Verify return value
        assert result is True

        # Verify utility functions were called
        mock_get_dxo.assert_called_once_with(mock_fl_context, ctx_prop_key, tasks)
        assert mock_get_tensors.call_count == root_key_count

        # Verify engine.stream_objects was called correct number of times
        assert mock_streamable_engine.stream_objects.call_count == root_key_count

        # Verify common call parameters
        calls = mock_streamable_engine.stream_objects.call_args_list
        for call in calls:
            assert call.kwargs["channel"] == TENSORS_CHANNEL
            assert call.kwargs["topic"] == expected_topic
            assert call.kwargs["targets"] == expected_targets
            assert call.kwargs["fl_ctx"] == mock_fl_context
            assert call.kwargs["optional"] is False
            assert call.kwargs["secure"] is False

            # Verify producer
            producer = call.kwargs["producer"]
            assert producer.__class__.__name__ == "TensorProducer"
            assert producer.entry_timeout == entry_timeout

    @pytest.mark.parametrize(
        "error_message",
        [
            "No task found in FLContext",
            "Skipping task, data kind is not WEIGHTS or WEIGHT_DIFF",
        ],
    )
    def test_send_error_conditions_return_false(self, mock_streamable_engine, mock_fl_context, error_message):
        """Test send returns False when encountering various error conditions."""
        with patch("nvflare.app_opt.tensor_stream.sender.get_dxo_from_ctx") as mock_get_dxo:
            mock_get_dxo.side_effect = ValueError(error_message)

            sender = TensorSender(
                engine=mock_streamable_engine,
                ctx_prop_key=FLContextKey.TASK_DATA,
                format=ExchangeFormat.PYTORCH,
                tasks=["train"],
            )

            result = sender.send(mock_fl_context, entry_timeout=1.0)
            assert result is False

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
    @patch("nvflare.app_opt.tensor_stream.sender.get_tensors_from_dxo")
    def test_send_with_custom_channel(
        self,
        mock_get_tensors,
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
        mock_get_tensors.return_value = sample_dxo_weights.data

        # Setup FL context
        shareable = sample_dxo_weights.to_shareable()
        shareable.set_header(ReservedKey.TASK_NAME, "train")
        mock_fl_context.get_prop.return_value = shareable

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

        # Send tensors
        result = sender.send(mock_fl_context, entry_timeout=1.0)

        # Verify return value
        assert result is True

        # Verify correct channel and targets were used
        call_args = mock_streamable_engine.stream_objects.call_args
        assert call_args.kwargs["channel"] == expected_channel
        assert call_args.kwargs["targets"] == targets

    @pytest.mark.parametrize(
        "dxo_fixture,expected_root_keys",
        [
            ("sample_dxo_nested_weights", ["encoder", "decoder"]),  # Nested structure
            ("sample_dxo_weights", [""]),  # Flat structure
        ],
    )
    @patch("nvflare.app_opt.tensor_stream.sender.get_targets_for_ctx_and_prop_key")
    @patch("nvflare.app_opt.tensor_stream.sender.get_topic_for_ctx_prop_key")
    @patch("nvflare.app_opt.tensor_stream.sender.get_dxo_from_ctx")
    @patch("nvflare.app_opt.tensor_stream.sender.get_tensors_from_dxo")
    def test_send_auto_detect_root_keys(
        self,
        mock_get_tensors,
        mock_get_dxo,
        mock_get_topic,
        mock_get_targets,
        mock_streamable_engine,
        mock_fl_context,
        dxo_fixture,
        expected_root_keys,
        request,
    ):
        """Test automatic detection of root keys for different data structures."""
        # Get the DXO fixture dynamically
        sample_dxo = request.getfixturevalue(dxo_fixture)

        # Setup mocks
        mock_get_targets.return_value = ["client1"]
        mock_get_topic.return_value = TensorTopics.TASK_DATA
        mock_get_dxo.return_value = sample_dxo

        # Setup mock for nested vs flat data
        if len(expected_root_keys) > 1:
            # For nested data, mock different calls for each key
            def mock_tensors_side_effect(dxo, key, format):
                if key in sample_dxo.data:
                    return sample_dxo.data[key]
                return {}

            mock_get_tensors.side_effect = mock_tensors_side_effect
        else:
            # For flat data, return the data directly
            mock_get_tensors.return_value = sample_dxo.data

        # Setup FL context
        shareable = sample_dxo.to_shareable()
        shareable.set_header(ReservedKey.TASK_NAME, "train")
        mock_fl_context.get_prop.return_value = shareable

        sender = TensorSender(
            engine=mock_streamable_engine,
            ctx_prop_key=FLContextKey.TASK_DATA,
            format=ExchangeFormat.PYTORCH,
            tasks=["train"],
        )

        # Send tensors
        result = sender.send(mock_fl_context, entry_timeout=3.0)

        # Verify return value
        assert result is True

        # Verify that root_keys were auto-detected correctly
        if "" in expected_root_keys:
            assert sender.root_keys == expected_root_keys
        else:
            assert set(sender.root_keys) == set(expected_root_keys)

        # Verify correct number of calls
        assert mock_get_tensors.call_count == len(expected_root_keys)
        assert mock_streamable_engine.stream_objects.call_count == len(expected_root_keys)

        # Check producers have correct root_keys
        calls = mock_streamable_engine.stream_objects.call_args_list
        producer_keys = [call.kwargs["producer"].root_key for call in calls]
        if "" in expected_root_keys:
            assert producer_keys == expected_root_keys
        else:
            assert set(producer_keys) == set(expected_root_keys)
