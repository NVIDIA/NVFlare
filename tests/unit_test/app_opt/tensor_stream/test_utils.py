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
    chunk_tensors_from_params,
    clean_task_data,
    clean_task_result,
    copy_non_tensor_params,
    get_dxo_from_ctx,
    get_targets_for_ctx_and_prop_key,
    get_topic_for_ctx_prop_key,
    merge_params_dicts,
    to_numpy_recursive,
    update_params_with_tensors,
)


class TestCleanTaskData:
    """Test cases for clean_task_data function."""

    def test_clean_task_data_removes_tensors(self, mock_fl_context):
        """Test successful cleaning of task data - removes tensors but keeps non-tensor params."""
        # Setup task data with tensors and non-tensor data
        task_data = {
            "DXO": {
                "data_kind": "WEIGHTS",
                "data": {
                    "model": torch.tensor([1.0, 2.0]),
                    "metadata": {"shape_info": "preserved_data"},
                    "config": {"learning_rate": 0.01},
                },
                "meta": {"round": 1},
            }
        }
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

        # Verify the tensors were removed but non-tensor params preserved
        cleaned_task_data = call_args[1]["value"]
        cleaned_data = cleaned_task_data["DXO"]["data"]
        assert "model" not in cleaned_data
        assert cleaned_data["metadata"]["shape_info"] == "preserved_data"
        assert cleaned_data["config"]["learning_rate"] == 0.01

    def test_clean_task_data_with_nested_tensors(self, mock_fl_context):
        """Test cleaning task data with nested tensor structure."""
        task_data = {
            "DXO": {
                "data": {
                    "encoder": {
                        "weight": torch.tensor([1.0, 2.0]),
                        "bias": torch.tensor([0.1]),
                        "metadata": "keep_this",
                    },
                    "decoder": {"weight": np.array([3.0, 4.0]), "info": "decoder_info"},
                    "config": {"learning_rate": 0.01},
                }
            }
        }
        mock_fl_context.get_prop.return_value = task_data

        # Clean the task data
        clean_task_data(mock_fl_context)

        # Verify tensors removed but structure and non-tensor params preserved
        call_args = mock_fl_context.set_prop.call_args
        cleaned_task_data = call_args[1]["value"]
        cleaned_data = cleaned_task_data["DXO"]["data"]

        assert "weight" not in cleaned_data["encoder"]
        assert "bias" not in cleaned_data["encoder"]
        assert cleaned_data["encoder"]["metadata"] == "keep_this"
        assert "weight" not in cleaned_data["decoder"]
        assert cleaned_data["decoder"]["info"] == "decoder_info"
        assert cleaned_data["config"]["learning_rate"] == 0.01

    def test_clean_task_data_empty_after_removal(self, mock_fl_context):
        """Test that data is empty dict when only tensors existed."""
        task_data = {
            "DXO": {
                "data": {
                    "weight": torch.tensor([1.0, 2.0]),
                    "bias": np.array([0.1]),
                }
            }
        }
        mock_fl_context.get_prop.return_value = task_data

        clean_task_data(mock_fl_context)

        call_args = mock_fl_context.set_prop.call_args
        cleaned_task_data = call_args[1]["value"]
        cleaned_data = cleaned_task_data["DXO"]["data"]

        assert cleaned_data == {}
        assert "weight" not in cleaned_data
        assert "bias" not in cleaned_data

    def test_clean_task_data_preserves_dxo_structure(self, mock_fl_context):
        """Test that DXO structure outside of 'data' is preserved."""
        task_data = {
            "DXO": {
                "data_kind": "WEIGHTS",
                "data": {
                    "model": torch.tensor([1.0, 2.0]),
                    "config": {"lr": 0.01},
                },
                "meta": {"round": 5, "client": "client_1"},
            }
        }
        mock_fl_context.get_prop.return_value = task_data

        clean_task_data(mock_fl_context)

        call_args = mock_fl_context.set_prop.call_args
        cleaned_task_data = call_args[1]["value"]

        # Verify DXO structure is preserved
        assert "DXO" in cleaned_task_data
        assert "data_kind" in cleaned_task_data["DXO"]
        assert cleaned_task_data["DXO"]["data_kind"] == "WEIGHTS"
        assert "meta" in cleaned_task_data["DXO"]
        assert cleaned_task_data["DXO"]["meta"]["round"] == 5
        assert cleaned_task_data["DXO"]["meta"]["client"] == "client_1"

        # Verify data was cleaned
        cleaned_data = cleaned_task_data["DXO"]["data"]
        assert "model" not in cleaned_data
        assert cleaned_data["config"]["lr"] == 0.01

    def test_clean_task_data_deeply_nested_structure(self, mock_fl_context):
        """Test cleaning with deeply nested structure."""
        task_data = {
            "DXO": {
                "data": {
                    "level1": {
                        "level2": {
                            "tensor": torch.tensor([1.0]),
                            "level3": {
                                "value": "keep",
                                "tensor_deep": np.array([2.0]),
                            },
                        },
                        "config": "config_value",
                    }
                }
            }
        }
        mock_fl_context.get_prop.return_value = task_data

        clean_task_data(mock_fl_context)

        call_args = mock_fl_context.set_prop.call_args
        cleaned_task_data = call_args[1]["value"]
        cleaned_data = cleaned_task_data["DXO"]["data"]

        assert "tensor" not in cleaned_data["level1"]["level2"]
        assert "tensor_deep" not in cleaned_data["level1"]["level2"]["level3"]
        assert cleaned_data["level1"]["level2"]["level3"]["value"] == "keep"
        assert cleaned_data["level1"]["config"] == "config_value"


class TestCleanTaskResult:
    """Test cases for clean_task_result function."""

    def test_clean_task_result_removes_tensors(self, mock_fl_context):
        """Test successful cleaning of task result - removes tensors but keeps non-tensor params."""
        # Setup task result with tensors and non-tensor data
        task_result = {
            "DXO": {
                "data": {
                    "updated_model": torch.tensor([1.0, 2.0]),
                    "metrics": {"accuracy": 0.95, "loss": 0.05},
                }
            }
        }
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

        # Verify the tensors were removed but non-tensor params preserved
        cleaned_task_result = call_args[1]["value"]
        cleaned_data = cleaned_task_result["DXO"]["data"]
        assert "updated_model" not in cleaned_data
        assert cleaned_data["metrics"]["accuracy"] == 0.95
        assert cleaned_data["metrics"]["loss"] == 0.05

    def test_clean_task_result_with_nested_tensors(self, mock_fl_context):
        """Test cleaning task result with nested tensor structure."""
        task_result = {
            "DXO": {
                "data": {
                    "model_updates": {
                        "weight_diff": torch.tensor([0.1, 0.2]),
                        "statistics": {"mean": 0.15, "std": 0.05},
                    },
                    "gradients": {
                        "layer1": np.array([0.01, 0.02]),
                        "layer2": torch.tensor([0.03, 0.04]),
                    },
                    "metadata": {"epoch": 10, "client_id": "client_1"},
                }
            }
        }
        mock_fl_context.get_prop.return_value = task_result

        clean_task_result(mock_fl_context)

        call_args = mock_fl_context.set_prop.call_args
        cleaned_task_result = call_args[1]["value"]
        cleaned_data = cleaned_task_result["DXO"]["data"]

        # Verify tensors removed
        assert "weight_diff" not in cleaned_data["model_updates"]
        assert "gradients" not in cleaned_data

        # Verify non-tensor data preserved
        assert cleaned_data["model_updates"]["statistics"]["mean"] == 0.15
        assert cleaned_data["model_updates"]["statistics"]["std"] == 0.05
        assert cleaned_data["metadata"]["epoch"] == 10
        assert cleaned_data["metadata"]["client_id"] == "client_1"

    def test_clean_task_result_empty_after_removal(self, mock_fl_context):
        """Test that result data is empty dict when only tensors existed."""
        task_result = {
            "DXO": {
                "data": {
                    "weight_diff": torch.tensor([1.0, 2.0]),
                    "gradients": np.array([0.1, 0.2]),
                }
            }
        }
        mock_fl_context.get_prop.return_value = task_result

        clean_task_result(mock_fl_context)

        call_args = mock_fl_context.set_prop.call_args
        cleaned_task_result = call_args[1]["value"]
        cleaned_data = cleaned_task_result["DXO"]["data"]

        assert cleaned_data == {}
        assert "weight_diff" not in cleaned_data
        assert "gradients" not in cleaned_data

    def test_clean_task_result_preserves_dxo_structure(self, mock_fl_context):
        """Test that DXO structure outside of 'data' is preserved."""
        task_result = {
            "DXO": {
                "data_kind": "WEIGHT_DIFF",
                "data": {
                    "diff": torch.tensor([0.1, 0.2]),
                    "info": {"num_samples": 100},
                },
                "meta": {"aggregation": "weighted", "round": 3},
            }
        }
        mock_fl_context.get_prop.return_value = task_result

        clean_task_result(mock_fl_context)

        call_args = mock_fl_context.set_prop.call_args
        cleaned_task_result = call_args[1]["value"]

        # Verify DXO structure is preserved
        assert "DXO" in cleaned_task_result
        assert "data_kind" in cleaned_task_result["DXO"]
        assert cleaned_task_result["DXO"]["data_kind"] == "WEIGHT_DIFF"
        assert "meta" in cleaned_task_result["DXO"]
        assert cleaned_task_result["DXO"]["meta"]["aggregation"] == "weighted"
        assert cleaned_task_result["DXO"]["meta"]["round"] == 3

        # Verify data was cleaned
        cleaned_data = cleaned_task_result["DXO"]["data"]
        assert "diff" not in cleaned_data
        assert cleaned_data["info"]["num_samples"] == 100

    def test_clean_task_result_with_mixed_types(self, mock_fl_context):
        """Test cleaning with various non-tensor data types."""
        task_result = {
            "DXO": {
                "data": {
                    "tensor": torch.tensor([1.0, 2.0]),
                    "int_val": 42,
                    "float_val": 3.14,
                    "str_val": "test_string",
                    "bool_val": True,
                    "list_val": [1, 2, 3],
                    "none_val": None,
                    "nested": {
                        "array": np.array([1, 2]),
                        "dict_val": {"key": "value"},
                    },
                }
            }
        }
        mock_fl_context.get_prop.return_value = task_result

        clean_task_result(mock_fl_context)

        call_args = mock_fl_context.set_prop.call_args
        cleaned_task_result = call_args[1]["value"]
        cleaned_data = cleaned_task_result["DXO"]["data"]

        # Verify tensors removed
        assert "tensor" not in cleaned_data
        assert "array" not in cleaned_data["nested"]

        # Verify all other types preserved
        assert cleaned_data["int_val"] == 42
        assert cleaned_data["float_val"] == 3.14
        assert cleaned_data["str_val"] == "test_string"
        assert cleaned_data["bool_val"] is True
        assert cleaned_data["list_val"] == [1, 2, 3]
        assert cleaned_data["none_val"] is None
        assert cleaned_data["nested"]["dict_val"]["key"] == "value"


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


class TestCopyNonTensorParams:
    """Test cases for copy_non_tensor_params function."""

    def test_copy_excludes_tensors(self):
        """Test that torch tensors are excluded from the copy."""
        params = {
            "tensor": torch.tensor([1.0, 2.0]),
            "metadata": {"shape": "preserved"},
            "config": {"lr": 0.01},
        }
        result = copy_non_tensor_params(params)

        assert "tensor" not in result
        assert result["metadata"]["shape"] == "preserved"
        assert result["config"]["lr"] == 0.01

    def test_copy_excludes_numpy_arrays(self):
        """Test that numpy arrays are excluded from the copy."""
        params = {
            "array": np.array([1.0, 2.0]),
            "metadata": {"info": "keep"},
        }
        result = copy_non_tensor_params(params)

        assert "array" not in result
        assert result["metadata"]["info"] == "keep"

    def test_copy_nested_structure(self):
        """Test copying nested dictionary structure."""
        params = {
            "encoder": {
                "weight": torch.tensor([1.0, 2.0]),
                "config": {"type": "linear"},
            },
            "decoder": {
                "bias": np.array([0.1]),
                "metadata": "info",
            },
        }
        result = copy_non_tensor_params(params)

        assert "weight" not in result["encoder"]
        assert result["encoder"]["config"]["type"] == "linear"
        assert "bias" not in result["decoder"]
        assert result["decoder"]["metadata"] == "info"

    def test_empty_dict_when_only_tensors(self):
        """Test that result is empty when params contain only tensors."""
        params = {
            "tensor1": torch.tensor([1.0]),
            "tensor2": np.array([2.0]),
        }
        result = copy_non_tensor_params(params)

        assert result == {}

    def test_preserves_various_types(self):
        """Test that various non-tensor types are preserved."""
        params = {
            "int_val": 42,
            "float_val": 3.14,
            "str_val": "text",
            "list_val": [1, 2, 3],
            "bool_val": True,
            "tensor": torch.tensor([1.0]),
        }
        result = copy_non_tensor_params(params)

        assert result["int_val"] == 42
        assert result["float_val"] == 3.14
        assert result["str_val"] == "text"
        assert result["list_val"] == [1, 2, 3]
        assert result["bool_val"] is True
        assert "tensor" not in result


class TestChunkTensorsFromParams:
    """Test cases for chunk_tensors_from_params function."""

    def test_chunk_flat_tensors(self):
        """Test chunking flat dictionary of tensors."""
        params = {
            "weight1": torch.tensor([1.0, 2.0]),
            "weight2": torch.tensor([3.0, 4.0]),
            "bias": torch.tensor([0.1]),
        }
        chunks = list(chunk_tensors_from_params(params, chunk_size=2))

        assert len(chunks) == 2  # 3 tensors with chunk_size=2 gives 2 chunks
        parent_keys, tensors = chunks[0]
        assert parent_keys == ()  # Empty tuple for flat structure
        assert len(tensors) == 2

    def test_chunk_nested_tensors(self):
        """Test chunking nested dictionary of tensors."""
        params = {
            "encoder": {
                "weight": torch.tensor([1.0, 2.0]),
                "bias": torch.tensor([0.1]),
            },
            "decoder": {
                "weight": torch.tensor([3.0, 4.0]),
            },
        }
        chunks = list(chunk_tensors_from_params(params, chunk_size=10))

        assert len(chunks) == 2  # One for encoder, one for decoder
        for parent_keys, tensors in chunks:
            assert len(parent_keys) == 1
            assert parent_keys[0] in ["encoder", "decoder"]

    def test_chunk_with_numpy_arrays(self):
        """Test chunking with numpy arrays converted to tensors."""
        params = {
            "tensor": torch.tensor([1.0, 2.0]),
            "array": np.array([3.0, 4.0]),
        }
        chunks = list(chunk_tensors_from_params(params, chunk_size=10))

        assert len(chunks) == 1
        parent_keys, tensors = chunks[0]
        assert len(tensors) == 2
        assert all(isinstance(t, torch.Tensor) for t in tensors.values())

    def test_chunk_size_none_no_splitting(self):
        """Test that chunk_size=None doesn't split tensors."""
        params = {
            "w1": torch.tensor([1.0]),
            "w2": torch.tensor([2.0]),
            "w3": torch.tensor([3.0]),
        }
        chunks = list(chunk_tensors_from_params(params, chunk_size=None))

        assert len(chunks) == 1
        parent_keys, tensors = chunks[0]
        assert len(tensors) == 3

    def test_invalid_chunk_size_raises_error(self):
        """Test that invalid chunk_size raises ValueError."""
        params = {"weight": torch.tensor([1.0])}
        with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
            list(chunk_tensors_from_params(params, chunk_size=0))

        with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
            list(chunk_tensors_from_params(params, chunk_size=-1))


class TestUpdateParamsWithTensors:
    """Test cases for update_params_with_tensors function."""

    def test_update_flat_params(self):
        """Test updating flat params dictionary."""
        params = {}
        tensors = {
            "weight": torch.tensor([1.0, 2.0]),
            "bias": torch.tensor([0.1]),
        }
        update_params_with_tensors(params, [], tensors)

        assert "weight" in params
        assert "bias" in params
        torch.testing.assert_close(params["weight"], tensors["weight"])
        torch.testing.assert_close(params["bias"], tensors["bias"])

    def test_update_nested_params(self):
        """Test updating nested params dictionary."""
        params = {}
        tensors = {
            "weight": torch.tensor([1.0, 2.0]),
            "bias": torch.tensor([0.1]),
        }
        update_params_with_tensors(params, ["encoder"], tensors)

        assert "encoder" in params
        assert "weight" in params["encoder"]
        assert "bias" in params["encoder"]
        torch.testing.assert_close(params["encoder"]["weight"], tensors["weight"])

    def test_update_with_to_ndarray(self):
        """Test updating with conversion to numpy arrays."""
        params = {}
        tensors = {
            "weight": torch.tensor([1.0, 2.0]),
        }
        update_params_with_tensors(params, [], tensors, to_ndarray=True)

        assert isinstance(params["weight"], np.ndarray)
        np.testing.assert_array_equal(params["weight"], np.array([1.0, 2.0]))

    def test_update_existing_dict(self):
        """Test updating existing dictionary structure."""
        params = {"encoder": {"existing": "value"}}
        tensors = {"weight": torch.tensor([1.0, 2.0])}
        update_params_with_tensors(params, ["encoder"], tensors)

        assert params["encoder"]["existing"] == "value"
        assert "weight" in params["encoder"]

    def test_invalid_path_raises_error(self):
        """Test that invalid path raises ValueError."""
        params = {"encoder": "not_a_dict"}
        tensors = {"weight": torch.tensor([1.0])}

        with pytest.raises(ValueError, match="Expected dict at key"):
            update_params_with_tensors(params, ["encoder", "nested"], tensors)


class TestMergeParamsDicts:
    """Test cases for merge_params_dicts function."""

    def test_merge_flat_dicts(self):
        """Test merging flat dictionaries."""
        base = {"weight1": torch.tensor([1.0, 2.0])}
        new = {"weight2": torch.tensor([3.0, 4.0])}
        result = merge_params_dicts(base, new)

        assert "weight1" in result
        assert "weight2" in result
        torch.testing.assert_close(result["weight1"], base["weight1"])
        torch.testing.assert_close(result["weight2"], new["weight2"])

    def test_merge_overwrites_values(self):
        """Test that new values overwrite base values."""
        base = {"weight": torch.tensor([1.0, 2.0])}
        new = {"weight": torch.tensor([3.0, 4.0])}
        result = merge_params_dicts(base, new)

        torch.testing.assert_close(result["weight"], new["weight"])

    def test_merge_nested_dicts(self):
        """Test merging nested dictionaries."""
        base = {"encoder": {"weight": torch.tensor([1.0])}}
        new = {"encoder": {"bias": torch.tensor([0.1])}}
        result = merge_params_dicts(base, new)

        assert "weight" in result["encoder"]
        assert "bias" in result["encoder"]

    def test_merge_with_to_ndarray(self):
        """Test merging with conversion to numpy."""
        base = {}
        new = {"weight": torch.tensor([1.0, 2.0])}
        result = merge_params_dicts(base, new, to_ndarray=True)

        assert isinstance(result["weight"], np.ndarray)
        np.testing.assert_array_equal(result["weight"], np.array([1.0, 2.0]))

    def test_merge_preserves_base_dict(self):
        """Test that merging modifies base dict in place."""
        base = {"weight1": torch.tensor([1.0])}
        new = {"weight2": torch.tensor([2.0])}
        result = merge_params_dicts(base, new)

        assert result is base  # Same object
        assert "weight2" in base


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

    def test_chunk_and_update_workflow(self):
        """Test chunking tensors and updating params workflow."""
        # Create initial params
        params = {
            "encoder": {"weight": torch.tensor([1.0, 2.0]), "bias": torch.tensor([0.1])},
            "decoder": {"weight": torch.tensor([3.0, 4.0])},
        }

        # Collect tensors and reconstruct
        reconstructed = {}
        for parent_keys, tensors in chunk_tensors_from_params(params, chunk_size=10):
            update_params_with_tensors(reconstructed, list(parent_keys), tensors)

        # Verify reconstruction
        assert "encoder" in reconstructed
        assert "decoder" in reconstructed
        torch.testing.assert_close(reconstructed["encoder"]["weight"], params["encoder"]["weight"])
        torch.testing.assert_close(reconstructed["decoder"]["weight"], params["decoder"]["weight"])

    def test_clean_and_copy_workflow(self, mock_fl_context):
        """Test cleaning task data preserves non-tensor params."""
        task_data = {
            "DXO": {
                "data": {
                    "model": torch.tensor([1.0, 2.0]),
                    "metadata": {"epoch": 5},
                }
            }
        }
        mock_fl_context.get_prop.return_value = task_data

        # Clean the task data
        clean_task_data(mock_fl_context)

        # Verify cleaned data has only non-tensor params
        call_args = mock_fl_context.set_prop.call_args
        cleaned_task_data = call_args[1]["value"]
        cleaned_data = cleaned_task_data["DXO"]["data"]
        assert "model" not in cleaned_data
        assert cleaned_data["metadata"]["epoch"] == 5
