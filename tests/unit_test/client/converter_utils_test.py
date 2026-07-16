# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from unittest.mock import MagicMock

import numpy as np
import pytest

from nvflare.client.config import ExchangeFormat, normalize_exchange_format, validate_format_pair
from nvflare.client.converter_utils import convert_params


def test_identity_and_raw_declarations_are_no_ops():
    params = {"w": np.asarray([1.0])}

    assert convert_params(params, ExchangeFormat.NUMPY, ExchangeFormat.NUMPY, {}) is params
    assert convert_params(params, ExchangeFormat.RAW, ExchangeFormat.NUMPY, {}) is params
    assert convert_params(params, ExchangeFormat.NUMPY, ExchangeFormat.RAW, {}) is params


def test_declarations_are_validated_without_payload_inference():
    assert normalize_exchange_format("pytorch", "format") == ExchangeFormat.PYTORCH
    with pytest.raises(ValueError, match="invalid format"):
        normalize_exchange_format("unknown", "format")
    with pytest.raises(ValueError, match="unsupported parameter format conversion"):
        validate_format_pair(ExchangeFormat.PYTORCH, ExchangeFormat.KERAS_LAYER_WEIGHTS)


def test_pytorch_round_trip_preserves_tensor_shape_and_local_non_tensor_state():
    torch = pytest.importorskip("torch")
    state = {}
    logger = MagicMock()
    native = {
        "w": torch.arange(6, dtype=torch.float32, requires_grad=True).reshape(2, 3),
        "local_metadata": "not aggregated",
    }

    server = convert_params(native, ExchangeFormat.PYTORCH, ExchangeFormat.NUMPY, state, logger)

    assert list(server) == ["w"]
    assert isinstance(server["w"], np.ndarray)
    assert tuple(server["w"].shape) == (2, 3)
    logger.warning.assert_called_once()

    # Simulate an aggregator that flattened the tensor before the next task. The caller-owned
    # state restores both its native shape and the trainer-local non-tensor value.
    restored = convert_params({"w": server["w"].reshape(-1)}, ExchangeFormat.NUMPY, ExchangeFormat.PYTORCH, state)
    assert tuple(restored["w"].shape) == (2, 3)
    assert torch.equal(restored["w"], native["w"])
    assert restored["local_metadata"] == "not aggregated"


@pytest.mark.parametrize("second_round_has_metadata", [False, True])
def test_pytorch_round_trip_replaces_stale_non_tensor_state(second_round_has_metadata):
    torch = pytest.importorskip("torch")
    state = {}
    first_round = {
        "w": torch.tensor([1.0]),
        "local_metadata": "not aggregated",
    }
    convert_params(first_round, ExchangeFormat.PYTORCH, ExchangeFormat.NUMPY, state)

    second_round = {"w": torch.tensor([2.0])}
    if second_round_has_metadata:
        second_round["local_metadata"] = torch.tensor([3.0])
    convert_params(second_round, ExchangeFormat.PYTORCH, ExchangeFormat.NUMPY, state)

    restored = convert_params({"w": np.asarray([4.0])}, ExchangeFormat.NUMPY, ExchangeFormat.PYTORCH, state)
    assert "local_metadata" not in restored


def test_pytorch_conversion_requires_parameter_dict():
    pytest.importorskip("torch")
    with pytest.raises(TypeError, match="parameter dict"):
        convert_params([np.asarray([1.0])], ExchangeFormat.NUMPY, ExchangeFormat.PYTORCH, {})


def test_keras_layer_weight_round_trip():
    native = {"dense": [np.asarray([[1.0, 2.0]]), np.asarray([3.0])]}

    server = convert_params(native, ExchangeFormat.KERAS_LAYER_WEIGHTS, ExchangeFormat.NUMPY, {})
    restored = convert_params(server, ExchangeFormat.NUMPY, ExchangeFormat.KERAS_LAYER_WEIGHTS, {})

    assert list(restored) == ["dense"]
    np.testing.assert_array_equal(restored["dense"][0], native["dense"][0])
    np.testing.assert_array_equal(restored["dense"][1], native["dense"][1])
