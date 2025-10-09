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

import numpy as np
import pytest
import torch

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.job_def import SERVER_SITE_NAME
from nvflare.app_opt.tensor_stream.types import TensorTopics
from nvflare.app_opt.tensor_stream.utils import (
    clean_task_data,
    clean_task_result,
    get_dxo_from_ctx,
    get_targets_for_ctx_and_prop_key,
    get_topic_for_ctx_prop_key,
    to_numpy_recursive,
    to_torch_recursive,
    validate_and_prepare_tensors,
    validate_numpy_dict_params_recursive,
    validate_torch_dict_params_recursive,
)
from nvflare.client.config import ExchangeFormat


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


class TestToNumpyRecursive:
    """Test cases for to_numpy_recursive function."""

    def test_convert_single_tensor(self):
        """Test converting a single torch tensor to numpy."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = to_numpy_recursive(tensor)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))

    def test_convert_dict_of_tensors(self, random_torch_tensors):
        """Test converting a dictionary of torch tensors to numpy arrays."""
        result = to_numpy_recursive(random_torch_tensors)

        assert isinstance(result, dict)
        assert len(result) == len(random_torch_tensors)

        for key, value in result.items():
            assert isinstance(value, np.ndarray)
            np.testing.assert_array_equal(value, random_torch_tensors[key].numpy())

    def test_convert_nested_dict_tensors(self, sample_nested_tensors):
        """Test converting nested dictionary of torch tensors to numpy arrays."""
        result = to_numpy_recursive(sample_nested_tensors)

        assert isinstance(result, dict)
        assert "encoder" in result and "decoder" in result

        for section_name, section_tensors in result.items():
            assert isinstance(section_tensors, dict)
            for key, value in section_tensors.items():
                assert isinstance(value, np.ndarray)
                np.testing.assert_array_equal(value, sample_nested_tensors[section_name][key].numpy())

    def test_unsupported_object_raises_error(self):
        """Test that unsupported objects raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported object type"):
            to_numpy_recursive("invalid_string")

        with pytest.raises(ValueError, match="Unsupported object type"):
            to_numpy_recursive(123)

    def test_mixed_dict_with_invalid_value(self):
        """Test dict containing non-tensor values raises error."""
        mixed_dict = {"tensor": torch.tensor([1.0, 2.0]), "invalid": "not_a_tensor"}
        with pytest.raises(ValueError, match="Unsupported object type"):
            to_numpy_recursive(mixed_dict)


class TestToTorchRecursive:
    """Test cases for to_torch_recursive function."""

    def test_convert_single_array(self):
        """Test converting a single numpy array to torch tensor."""
        array = np.array([1.0, 2.0, 3.0])
        result = to_torch_recursive(array)

        assert isinstance(result, torch.Tensor)
        torch.testing.assert_close(result, torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))

    def test_convert_dict_of_arrays(self):
        """Test converting a dictionary of numpy arrays to torch tensors."""
        arrays = {"weight": np.array([[1.0, 2.0], [3.0, 4.0]]), "bias": np.array([0.1, 0.2])}
        result = to_torch_recursive(arrays)

        assert isinstance(result, dict)
        assert len(result) == 2

        for key, value in result.items():
            assert isinstance(value, torch.Tensor)
            torch.testing.assert_close(value, torch.from_numpy(arrays[key]))

    def test_convert_nested_dict_arrays(self):
        """Test converting nested dictionary of numpy arrays to torch tensors."""
        nested_arrays = {
            "encoder": {"weight": np.array([[1.0, 2.0]]), "bias": np.array([0.1])},
            "decoder": {"weight": np.array([[3.0, 4.0]]), "bias": np.array([0.2])},
        }
        result = to_torch_recursive(nested_arrays)

        assert isinstance(result, dict)
        assert "encoder" in result and "decoder" in result

        for section_name, section_arrays in result.items():
            assert isinstance(section_arrays, dict)
            for key, value in section_arrays.items():
                assert isinstance(value, torch.Tensor)
                expected = torch.from_numpy(nested_arrays[section_name][key])
                torch.testing.assert_close(value, expected)

    def test_convert_with_device(self):
        """Test converting with specified device."""
        array = np.array([1.0, 2.0, 3.0])
        result = to_torch_recursive(array, device=torch.device("cpu"))

        assert isinstance(result, torch.Tensor)
        assert result.device == torch.device("cpu")
        torch.testing.assert_close(result, torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))

    def test_unsupported_object_raises_error(self):
        """Test that unsupported objects raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported object type"):
            to_torch_recursive("invalid_string")

        with pytest.raises(ValueError, match="Unsupported object type"):
            to_torch_recursive(123)

    def test_mixed_dict_with_invalid_value(self):
        """Test dict containing non-array values raises error."""
        mixed_dict = {"array": np.array([1.0, 2.0]), "invalid": "not_an_array"}
        with pytest.raises(ValueError, match="Unsupported object type"):
            to_torch_recursive(mixed_dict)


class TestValidateTorchDictParamsRecursive:
    """Test cases for validate_torch_dict_params_recursive function."""

    def test_valid_torch_dict(self, random_torch_tensors):
        """Test validation of valid torch tensor dictionary."""
        # Should not raise any exception
        validate_torch_dict_params_recursive(random_torch_tensors)

    def test_valid_nested_torch_dict(self, sample_nested_tensors):
        """Test validation of valid nested torch tensor dictionary."""
        # Should not raise any exception
        validate_torch_dict_params_recursive(sample_nested_tensors)

    def test_non_dict_raises_error(self):
        """Test that non-dictionary input raises ValueError."""
        with pytest.raises(ValueError, match="Expected a dictionary"):
            validate_torch_dict_params_recursive("not_a_dict")

        with pytest.raises(ValueError, match="Expected a dictionary"):
            validate_torch_dict_params_recursive(torch.tensor([1.0, 2.0]))

    def test_dict_with_non_tensor_raises_error(self):
        """Test that dictionary with non-tensor values raises ValueError."""
        invalid_dict = {"tensor": torch.tensor([1.0, 2.0]), "invalid": "not_a_tensor"}
        with pytest.raises(ValueError, match="Expected torch.Tensor for key 'invalid'"):
            validate_torch_dict_params_recursive(invalid_dict)

    def test_nested_dict_with_non_tensor_raises_error(self):
        """Test that nested dictionary with non-tensor values raises ValueError."""
        invalid_nested_dict = {
            "valid_section": {"tensor": torch.tensor([1.0, 2.0])},
            "invalid_section": {
                "tensor": torch.tensor([3.0, 4.0]),
                "invalid": np.array([1.0, 2.0]),  # numpy array instead of torch tensor
            },
        }
        with pytest.raises(ValueError, match="Expected torch.Tensor for key 'invalid'"):
            validate_torch_dict_params_recursive(invalid_nested_dict)

    def test_empty_dict_is_valid(self):
        """Test that empty dictionary is considered valid."""
        # Should not raise any exception
        validate_torch_dict_params_recursive({})

    def test_deeply_nested_dict(self):
        """Test validation of deeply nested dictionary."""
        deeply_nested = {"level1": {"level2": {"tensor": torch.tensor([1.0, 2.0])}}}
        # Should not raise any exception
        validate_torch_dict_params_recursive(deeply_nested)


class TestValidateNumpyDictParamsRecursive:
    """Test cases for validate_numpy_dict_params_recursive function."""

    def test_valid_numpy_dict(self):
        """Test validation of valid numpy array dictionary."""
        numpy_dict = {"weight": np.array([[1.0, 2.0], [3.0, 4.0]]), "bias": np.array([0.1, 0.2])}
        # Should not raise any exception
        validate_numpy_dict_params_recursive(numpy_dict)

    def test_valid_nested_numpy_dict(self):
        """Test validation of valid nested numpy array dictionary."""
        nested_numpy_dict = {
            "encoder": {"weight": np.array([[1.0, 2.0]]), "bias": np.array([0.1])},
            "decoder": {"weight": np.array([[3.0, 4.0]]), "bias": np.array([0.2])},
        }
        # Should not raise any exception
        validate_numpy_dict_params_recursive(nested_numpy_dict)

    def test_non_dict_raises_error(self):
        """Test that non-dictionary input raises ValueError."""
        with pytest.raises(ValueError, match="Expected a dictionary"):
            validate_numpy_dict_params_recursive("not_a_dict")

        with pytest.raises(ValueError, match="Expected a dictionary"):
            validate_numpy_dict_params_recursive(np.array([1.0, 2.0]))

    def test_dict_with_non_array_raises_error(self):
        """Test that dictionary with non-array values raises ValueError."""
        invalid_dict = {"array": np.array([1.0, 2.0]), "invalid": "not_an_array"}
        with pytest.raises(ValueError, match="Expected np.ndarray for key 'invalid'"):
            validate_numpy_dict_params_recursive(invalid_dict)

    def test_dict_with_torch_tensor_raises_error(self):
        """Test that dictionary with torch tensor values raises ValueError."""
        invalid_dict = {
            "array": np.array([1.0, 2.0]),
            "tensor": torch.tensor([3.0, 4.0]),  # torch tensor instead of numpy array
        }
        with pytest.raises(ValueError, match="Expected np.ndarray for key 'tensor'"):
            validate_numpy_dict_params_recursive(invalid_dict)

    def test_empty_dict_is_valid(self):
        """Test that empty dictionary is considered valid."""
        # Should not raise any exception
        validate_numpy_dict_params_recursive({})


class TestGetDxoFromCtx:
    """Test cases for get_dxo_from_ctx function."""

    def test_get_dxo_success(self, mock_fl_context):
        """Test successful DXO extraction from context."""
        # Setup task shareable
        dxo = DXO(data_kind=DataKind.WEIGHTS, data={"model": torch.tensor([1.0, 2.0])})
        task_shareable = dxo.to_shareable()

        # Mock FLContext to return task name and shareable separately
        def mock_get_prop(key):
            if key == FLContextKey.TASK_NAME:
                return "train"
            elif key == FLContextKey.TASK_DATA:
                return task_shareable
            return None

        mock_fl_context.get_prop.side_effect = mock_get_prop

        # Test extraction
        result_dxo = get_dxo_from_ctx(mock_fl_context, FLContextKey.TASK_DATA, ["train", "validate"])

        assert isinstance(result_dxo, DXO)
        assert result_dxo.data_kind == DataKind.WEIGHTS
        # Verify both calls were made
        expected_calls = [
            mock_fl_context.get_prop.call_args_list[0][0][0],  # First call arg
            mock_fl_context.get_prop.call_args_list[1][0][0],  # Second call arg
        ]
        assert FLContextKey.TASK_NAME in expected_calls
        assert FLContextKey.TASK_DATA in expected_calls

    def test_get_dxo_weight_diff(self, mock_fl_context):
        """Test successful DXO extraction with WEIGHT_DIFF data kind."""
        # Setup task shareable with WEIGHT_DIFF
        dxo = DXO(data_kind=DataKind.WEIGHT_DIFF, data={"diff": torch.tensor([0.1, 0.2])})
        task_shareable = dxo.to_shareable()

        # Mock FLContext to return task name and shareable separately
        def mock_get_prop(key):
            if key == FLContextKey.TASK_NAME:
                return "aggregate"
            elif key == FLContextKey.TASK_RESULT:
                return task_shareable
            return None

        mock_fl_context.get_prop.side_effect = mock_get_prop

        # Test extraction
        result_dxo = get_dxo_from_ctx(mock_fl_context, FLContextKey.TASK_RESULT, ["aggregate"])

        assert isinstance(result_dxo, DXO)
        assert result_dxo.data_kind == DataKind.WEIGHT_DIFF

    def test_no_task_in_context_raises_error(self, mock_fl_context):
        """Test that missing task in context raises ValueError."""

        # Mock FLContext to return task name but no shareable
        def mock_get_prop(key):
            if key == FLContextKey.TASK_NAME:
                return "train"
            elif key == FLContextKey.TASK_DATA:
                return None
            return None

        mock_fl_context.get_prop.side_effect = mock_get_prop

        with pytest.raises(ValueError, match="No task found in FLContext"):
            get_dxo_from_ctx(mock_fl_context, FLContextKey.TASK_DATA, ["train"])

    def test_no_task_name_raises_error(self, mock_fl_context):
        """Test that missing task name raises ValueError."""

        # Mock FLContext to return no task name
        def mock_get_prop(key):
            if key == FLContextKey.TASK_NAME:
                return None  # No task name found
            return None

        mock_fl_context.get_prop.side_effect = mock_get_prop

        with pytest.raises(ValueError, match="No task name found in FLContext"):
            get_dxo_from_ctx(mock_fl_context, FLContextKey.TASK_DATA, ["train"])

    def test_invalid_task_name_raises_error(self, mock_fl_context):
        """Test that invalid task name raises ValueError."""

        # Mock FLContext to return invalid task name
        def mock_get_prop(key):
            if key == FLContextKey.TASK_NAME:
                return "invalid_task"  # Invalid task name
            return None

        mock_fl_context.get_prop.side_effect = mock_get_prop

        with pytest.raises(ValueError, match="Task name 'invalid_task' not part of configured tasks"):
            get_dxo_from_ctx(mock_fl_context, FLContextKey.TASK_DATA, ["train", "validate"])

    def test_invalid_data_kind_raises_error(self, mock_fl_context):
        """Test that invalid data kind raises ValueError."""
        dxo = DXO(data_kind=DataKind.METRICS, data={"accuracy": 0.95})  # Invalid data kind
        task_shareable = dxo.to_shareable()

        # Mock FLContext to return task name and shareable separately
        def mock_get_prop(key):
            if key == FLContextKey.TASK_NAME:
                return "train"
            elif key == FLContextKey.TASK_DATA:
                return task_shareable
            return None

        mock_fl_context.get_prop.side_effect = mock_get_prop

        with pytest.raises(ValueError, match="Skipping task, data kind is not WEIGHTS or WEIGHT_DIFF"):
            get_dxo_from_ctx(mock_fl_context, FLContextKey.TASK_DATA, ["train"])

    @pytest.mark.parametrize("task_name", ["train", "validate", "aggregate", "submit_model"])
    def test_various_valid_task_names(self, mock_fl_context, task_name):
        """Test various valid task names."""
        dxo = DXO(data_kind=DataKind.WEIGHTS, data={"model": torch.tensor([1.0])})
        task_shareable = dxo.to_shareable()

        # Mock FLContext to return task name and shareable separately
        def mock_get_prop(key):
            if key == FLContextKey.TASK_NAME:
                return task_name
            elif key == FLContextKey.TASK_DATA:
                return task_shareable
            return None

        mock_fl_context.get_prop.side_effect = mock_get_prop

        result_dxo = get_dxo_from_ctx(
            mock_fl_context, FLContextKey.TASK_DATA, ["train", "validate", "aggregate", "submit_model"]
        )

        assert isinstance(result_dxo, DXO)
        assert result_dxo.data_kind == DataKind.WEIGHTS


class TestGetTensorsFromDxo:
    """Test cases for get_tensors_from_dxo function."""

    def test_get_tensors_pytorch_format(self, sample_dxo_weights):
        """Test extracting tensors from DXO in PyTorch format."""
        result = validate_and_prepare_tensors(sample_dxo_weights, "", ExchangeFormat.PYTORCH)

        assert isinstance(result, dict)
        for key, value in result.items():
            assert isinstance(value, torch.Tensor)
            assert key in sample_dxo_weights.data

    def test_get_tensors_numpy_format(self):
        """Test extracting tensors from DXO in NumPy format."""
        # Create DXO with numpy arrays
        numpy_data = {"weight": np.array([[1.0, 2.0], [3.0, 4.0]]), "bias": np.array([0.1, 0.2])}
        dxo = DXO(data_kind=DataKind.WEIGHTS, data=numpy_data)

        result = validate_and_prepare_tensors(dxo, "", ExchangeFormat.NUMPY)

        assert isinstance(result, dict)
        for key, value in result.items():
            assert isinstance(value, torch.Tensor)
            torch.testing.assert_close(value, torch.from_numpy(numpy_data[key]))

    def test_get_tensors_with_key(self, sample_dxo_nested_weights):
        """Test extracting tensors with specific key."""
        result = validate_and_prepare_tensors(sample_dxo_nested_weights, "encoder", ExchangeFormat.PYTORCH)

        assert isinstance(result, dict)
        for key, value in result.items():
            assert isinstance(value, torch.Tensor)
            assert key in sample_dxo_nested_weights.data["encoder"]

    def test_get_tensors_empty_key_uses_all_data(self, sample_dxo_weights):
        """Test that empty key extracts all data."""
        result = validate_and_prepare_tensors(sample_dxo_weights, "", ExchangeFormat.PYTORCH)

        assert len(result) == len(sample_dxo_weights.data)
        for key in sample_dxo_weights.data.keys():
            assert key in result

    def test_no_data_raises_error(self):
        """Test that missing data raises ValueError."""
        dxo = DXO(data_kind=DataKind.WEIGHTS, data={})

        with pytest.raises(ValueError, match="No tensor data found on the context shareable"):
            validate_and_prepare_tensors(dxo, "", ExchangeFormat.PYTORCH)

    def test_missing_key_raises_error(self, sample_dxo_weights):
        """Test that missing key raises ValueError."""
        with pytest.raises(ValueError, match="No tensor data found on the context shareable. Key='missing_key'"):
            validate_and_prepare_tensors(sample_dxo_weights, "missing_key", ExchangeFormat.PYTORCH)

    def test_non_dict_data_raises_error(self):
        """Test that non-dictionary data raises ValueError."""
        # DXO creation with string data should fail during construction
        with pytest.raises(ValueError, match="invalid DXO: invalid data"):
            dxo = DXO(data_kind=DataKind.WEIGHTS, data="not_a_dict")

    def test_get_tensors_with_non_dict_data(self, sample_dxo_weights):
        """Test that get_tensors_from_dxo handles non-dict data properly."""
        # Create a valid DXO first
        dxo = sample_dxo_weights
        # Then manually corrupt the data to test error handling
        dxo.data = "not_a_dict"

        with pytest.raises(ValueError, match="Expected tensor data to be a dict"):
            validate_and_prepare_tensors(dxo, "", ExchangeFormat.PYTORCH)

    def test_unsupported_format_raises_error(self, sample_dxo_weights):
        """Test that unsupported format raises TypeError."""
        # Create a fake format that doesn't exist
        fake_format = "UNSUPPORTED_FORMAT"

        with pytest.raises(TypeError, match="Unsupported tensor data type"):
            validate_and_prepare_tensors(sample_dxo_weights, "", fake_format)

    def test_pytorch_format_validation_error(self):
        """Test that PyTorch format with invalid data raises error."""
        # Create DXO with mixed data types
        mixed_data = {"tensor": torch.tensor([1.0, 2.0]), "invalid": "not_a_tensor"}
        dxo = DXO(data_kind=DataKind.WEIGHTS, data=mixed_data)

        with pytest.raises(ValueError, match="Expected torch.Tensor for key 'invalid'"):
            validate_and_prepare_tensors(dxo, "", ExchangeFormat.PYTORCH)

    def test_numpy_format_validation_error(self):
        """Test that NumPy format with invalid data raises error."""
        # Create DXO with mixed data types
        mixed_data = {"array": np.array([1.0, 2.0]), "invalid": "not_an_array"}
        dxo = DXO(data_kind=DataKind.WEIGHTS, data=mixed_data)

        with pytest.raises(ValueError, match="Expected np.ndarray for key 'invalid'"):
            validate_and_prepare_tensors(dxo, "", ExchangeFormat.NUMPY)

    def test_nested_data_with_key(self):
        """Test extracting nested data with specific key."""
        nested_data = {
            "encoder": {"weight": torch.tensor([[1.0, 2.0]]), "bias": torch.tensor([0.1])},
            "decoder": {"weight": torch.tensor([[3.0, 4.0]]), "bias": torch.tensor([0.2])},
        }
        dxo = DXO(data_kind=DataKind.WEIGHTS, data=nested_data)

        result = validate_and_prepare_tensors(dxo, "encoder", ExchangeFormat.PYTORCH)

        assert len(result) == 2
        assert "weight" in result and "bias" in result
        torch.testing.assert_close(result["weight"], nested_data["encoder"]["weight"])
        torch.testing.assert_close(result["bias"], nested_data["encoder"]["bias"])


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
