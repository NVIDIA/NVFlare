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

from unittest.mock import Mock

import pytest

from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.job_def import SERVER_SITE_NAME
from nvflare.app_opt.tensor_stream.types import TensorTopics
from nvflare.app_opt.tensor_stream.utils import (
    clean_task_data,
    clean_task_result,
    get_targets_for_ctx_and_prop_key,
    get_topic_for_ctx_prop_key,
)


class TestCleanTaskData:
    """Test cases for clean_task_data function."""

    def test_clean_task_data_success(self, mock_fl_context, sample_shareable_with_dxo):
        """Test successful cleaning of task data."""
        # Setup task data with non-empty DXO data
        task_data = sample_shareable_with_dxo.copy()
        task_data["DXO"]["data"] = {"model": "some_large_tensor_data"}
        mock_fl_context.get_prop.return_value = task_data

        # Clean the task data
        clean_task_data(mock_fl_context)

        # Verify get_prop was called with correct key
        mock_fl_context.get_prop.assert_called_once_with(FLContextKey.TASK_DATA)

        # Verify set_prop was called with cleaned data
        mock_fl_context.set_prop.assert_called_once()
        call_args = mock_fl_context.set_prop.call_args

        assert call_args[1]["private"] is True
        assert call_args[1]["sticky"] is False

        # Verify the data was cleaned (set to empty dict)
        cleaned_task_data = call_args[1]["value"]
        assert cleaned_task_data["DXO"]["data"] == {}

        # Verify other parts of the shareable remain unchanged
        assert cleaned_task_data["DXO"]["kind"] == sample_shareable_with_dxo["DXO"]["kind"]

    @pytest.mark.parametrize(
        "initial_data", [{}, None, {"model": "data"}]  # Already empty  # None data  # Has data to clean
    )
    def test_clean_task_data_edge_cases(self, mock_fl_context, sample_shareable_with_dxo, initial_data):
        """Test cleaning task data with various initial states."""
        task_data = sample_shareable_with_dxo.copy()
        task_data["DXO"]["data"] = initial_data
        mock_fl_context.get_prop.return_value = task_data

        # Clean the task data
        clean_task_data(mock_fl_context)

        # Verify data was set to empty dict
        call_args = mock_fl_context.set_prop.call_args
        cleaned_task_data = call_args[1]["value"]
        assert cleaned_task_data["DXO"]["data"] == {}

    def test_clean_task_data_complex_structure(self, mock_fl_context):
        """Test cleaning task data with complex nested structure."""
        # Setup complex task data
        complex_task_data = {
            "DXO": {
                "data_kind": "WEIGHTS",
                "data": {
                    "encoder": {"weight": "large_tensor", "bias": "another_tensor"},
                    "decoder": {"layers": {"0": {"weight": "tensor_data"}}},
                    "metadata": {"shape_info": "preserved_data"},
                },
                "meta": {"round": 1},
            }
        }
        mock_fl_context.get_prop.return_value = complex_task_data

        # Clean the task data
        clean_task_data(mock_fl_context)

        # Verify complex data was cleaned but structure preserved
        call_args = mock_fl_context.set_prop.call_args
        cleaned_task_data = call_args[1]["value"]

        assert cleaned_task_data["DXO"]["data"] == {}
        assert cleaned_task_data["DXO"]["data_kind"] == "WEIGHTS"
        assert cleaned_task_data["DXO"]["meta"]["round"] == 1


class TestCleanTaskResult:
    """Test cases for clean_task_result function."""

    def test_clean_task_result_success(self, mock_fl_context, sample_shareable_with_dxo):
        """Test successful cleaning of task result."""
        # Setup task result with non-empty DXO data
        task_result = sample_shareable_with_dxo.copy()
        task_result["DXO"]["data"] = {"updated_model": "large_gradient_data"}
        mock_fl_context.get_prop.return_value = task_result

        # Clean the task result
        clean_task_result(mock_fl_context)

        # Verify get_prop was called with correct key
        mock_fl_context.get_prop.assert_called_once_with(FLContextKey.TASK_RESULT)

        # Verify set_prop was called with cleaned data
        mock_fl_context.set_prop.assert_called_once()
        call_args = mock_fl_context.set_prop.call_args

        assert call_args[1]["private"] is True
        assert call_args[1]["sticky"] is False

        # Verify the data was cleaned
        cleaned_task_result = call_args[1]["value"]
        assert cleaned_task_result["DXO"]["data"] == {}

    def test_clean_task_result_preserves_metadata(self, mock_fl_context):
        """Test that cleaning preserves all metadata except data."""
        # Setup task result with metadata
        task_result = {
            "DXO": {
                "data_kind": "METRICS",
                "data": {"accuracy": 0.95, "loss": 0.05},
                "meta": {"client_name": "client_1", "round_number": 5, "training_time": 120.5},
            },
            "peer_props": {"site_name": "client_1"},
        }
        mock_fl_context.get_prop.return_value = task_result

        # Clean the task result
        clean_task_result(mock_fl_context)

        # Verify metadata is preserved
        call_args = mock_fl_context.set_prop.call_args
        cleaned_task_result = call_args[1]["value"]

        assert cleaned_task_result["DXO"]["data"] == {}
        assert cleaned_task_result["DXO"]["data_kind"] == "METRICS"
        assert cleaned_task_result["DXO"]["meta"]["client_name"] == "client_1"
        assert cleaned_task_result["DXO"]["meta"]["round_number"] == 5
        assert cleaned_task_result["peer_props"]["site_name"] == "client_1"

    def test_clean_task_result_idempotent(self, mock_fl_context, sample_shareable_with_dxo):
        """Test that multiple clean calls are idempotent."""
        task_result = sample_shareable_with_dxo.copy()
        task_result["DXO"]["data"] = {"model": "data"}
        mock_fl_context.get_prop.return_value = task_result

        # First clean
        clean_task_result(mock_fl_context)

        # Verify second clean with already empty data also works
        mock_fl_context.reset_mock()
        task_result["DXO"]["data"] = {}
        mock_fl_context.get_prop.return_value = task_result

        clean_task_result(mock_fl_context)

        call_args = mock_fl_context.set_prop.call_args
        cleaned_task_result = call_args[1]["value"]
        assert cleaned_task_result["DXO"]["data"] == {}


class TestGetTopicForCtxPropKey:
    """Test cases for get_topic_for_ctx_prop_key function."""

    def test_task_data_topic(self):
        """Test getting topic for TASK_DATA key."""
        topic = get_topic_for_ctx_prop_key(FLContextKey.TASK_DATA)
        assert topic == TensorTopics.TASK_DATA

    def test_task_result_topic(self):
        """Test getting topic for TASK_RESULT key."""
        topic = get_topic_for_ctx_prop_key(FLContextKey.TASK_RESULT)
        assert topic == TensorTopics.TASK_RESULT

    def test_unsupported_key_raises_error(self):
        """Test that unsupported context property key raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported context property key"):
            get_topic_for_ctx_prop_key("UNSUPPORTED_KEY")

    @pytest.mark.parametrize("invalid_key", [None, "", "INVALID", "task_data"])
    def test_invalid_keys_raise_error(self, invalid_key):
        """Test that invalid keys raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported context property key"):
            get_topic_for_ctx_prop_key(invalid_key)


class TestGetTargetsForCtxAndPropKey:
    """Test cases for get_targets_for_ctx_and_prop_key function."""

    def test_task_data_targets_peer_identity(self, mock_fl_context):
        """Test getting targets for TASK_DATA returns peer identity."""
        # Setup mock peer context
        peer_identity = "client_123"
        mock_peer_context = Mock()
        mock_peer_context.get_identity_name.return_value = peer_identity
        mock_fl_context.get_peer_context.return_value = mock_peer_context

        # Get targets
        targets = get_targets_for_ctx_and_prop_key(mock_fl_context, FLContextKey.TASK_DATA)

        # Verify result
        assert targets == [peer_identity]
        mock_fl_context.get_peer_context.assert_called_once()
        mock_peer_context.get_identity_name.assert_called_once()

    def test_task_result_targets_server(self, mock_fl_context):
        """Test getting targets for TASK_RESULT returns server name."""
        targets = get_targets_for_ctx_and_prop_key(mock_fl_context, FLContextKey.TASK_RESULT)

        # Verify result
        assert targets == [SERVER_SITE_NAME]
        # Verify peer context is not accessed for task result
        mock_fl_context.get_peer_context.assert_not_called()

    @pytest.mark.parametrize("invalid_key", [None, "", "INVALID"])
    def test_invalid_keys_raise_error(self, mock_fl_context, invalid_key):
        """Test that invalid keys raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported context property key"):
            get_targets_for_ctx_and_prop_key(mock_fl_context, invalid_key)

    def test_peer_context_exceptions_propagate(self, mock_fl_context):
        """Test that peer context exceptions are properly propagated."""
        # Test peer context access failure
        mock_fl_context.get_peer_context.side_effect = Exception("Peer context error")
        with pytest.raises(Exception, match="Peer context error"):
            get_targets_for_ctx_and_prop_key(mock_fl_context, FLContextKey.TASK_DATA)

        # Reset side_effect and test identity name failure
        mock_fl_context.get_peer_context.side_effect = None
        mock_peer_context = Mock()
        mock_peer_context.get_identity_name.side_effect = Exception("Identity error")
        mock_fl_context.get_peer_context.return_value = mock_peer_context

        with pytest.raises(Exception, match="Identity error"):
            get_targets_for_ctx_and_prop_key(mock_fl_context, FLContextKey.TASK_DATA)

    def test_return_type_consistency(self, mock_fl_context):
        """Test that both functions return lists for consistency."""
        # Setup for TASK_DATA
        mock_peer_context = Mock()
        mock_peer_context.get_identity_name.return_value = "client_1"
        mock_fl_context.get_peer_context.return_value = mock_peer_context

        # Test TASK_DATA returns list
        task_data_targets = get_targets_for_ctx_and_prop_key(mock_fl_context, FLContextKey.TASK_DATA)
        assert isinstance(task_data_targets, list)
        assert len(task_data_targets) == 1

        # Test TASK_RESULT returns list
        task_result_targets = get_targets_for_ctx_and_prop_key(mock_fl_context, FLContextKey.TASK_RESULT)
        assert isinstance(task_result_targets, list)
        assert len(task_result_targets) == 1


class TestUtilsIntegration:
    """Integration tests for utils functions working together."""

    def test_topic_and_targets_consistency(self, mock_fl_context):
        """Test that topic and targets functions work consistently for the same keys."""
        # Setup peer context
        peer_identity = "integration_client"
        mock_peer_context = Mock()
        mock_peer_context.get_identity_name.return_value = peer_identity
        mock_fl_context.get_peer_context.return_value = mock_peer_context

        # Test TASK_DATA consistency
        task_data_topic = get_topic_for_ctx_prop_key(FLContextKey.TASK_DATA)
        task_data_targets = get_targets_for_ctx_and_prop_key(mock_fl_context, FLContextKey.TASK_DATA)

        assert task_data_topic == TensorTopics.TASK_DATA
        assert task_data_targets == [peer_identity]

        # Test TASK_RESULT consistency
        task_result_topic = get_topic_for_ctx_prop_key(FLContextKey.TASK_RESULT)
        task_result_targets = get_targets_for_ctx_and_prop_key(mock_fl_context, FLContextKey.TASK_RESULT)

        assert task_result_topic == TensorTopics.TASK_RESULT
        assert task_result_targets == [SERVER_SITE_NAME]
