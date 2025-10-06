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

import random
from typing import Dict
from unittest.mock import Mock

import pytest
import torch

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.streaming import StreamableEngine, StreamContext


@pytest.fixture
def random_torch_tensors() -> Dict[str, torch.Tensor]:
    """Pytest fixture to generate a dictionary with multiple random torch tensors.

    Returns:
        Dict[str, torch.Tensor]: Dictionary containing tensors with various shapes and data types.
    """
    # Set seed for reproducible tests
    torch.manual_seed(42)
    random.seed(42)

    tensors = {}

    # Generate tensors with different shapes and types
    tensor_configs = [
        ("layer1.weight", (128, 64), torch.float32),
        ("layer1.bias", (128,), torch.float32),
        ("layer2.weight", (64, 32), torch.float32),
        ("layer2.bias", (64,), torch.float32),
        ("embedding.weight", (1000, 256), torch.float32),
        ("output.weight", (10, 256), torch.float32),
        ("output.bias", (10,), torch.float32),
        ("batch_norm.running_mean", (128,), torch.float32),
        ("batch_norm.running_var", (128,), torch.float32),
        ("conv1.weight", (32, 3, 3, 3), torch.float32),
    ]

    for name, shape, dtype in tensor_configs:
        if "weight" in name:
            # Use normal distribution for weights
            tensors[name] = torch.randn(*shape, dtype=dtype)
        elif "bias" in name:
            # Use small random values for biases
            tensors[name] = torch.randn(*shape, dtype=dtype) * 0.1
        elif "running_mean" in name:
            # Running mean should be close to zero
            tensors[name] = torch.randn(*shape, dtype=dtype) * 0.01
        elif "running_var" in name:
            # Running variance should be close to one
            tensors[name] = torch.ones(*shape, dtype=dtype) + torch.randn(*shape, dtype=dtype) * 0.1
        else:
            # Default random tensor
            tensors[name] = torch.randn(*shape, dtype=dtype)

    return tensors


@pytest.fixture
def mock_fl_context():
    """Mock FL context for testing."""
    fl_ctx = Mock(spec=FLContext)
    peer_ctx = Mock(spec=FLContext)
    peer_ctx.get_identity_name.return_value = "test_peer"
    fl_ctx.get_peer_context.return_value = peer_ctx
    fl_ctx.get_identity_name.return_value = "test_client"

    # Track custom properties set on the context
    fl_ctx._custom_props = {}

    def set_custom_prop(key, value):
        fl_ctx._custom_props[key] = value

    def get_custom_prop(key, default=None):
        return fl_ctx._custom_props.get(key, default)

    fl_ctx.set_custom_prop = set_custom_prop
    fl_ctx.get_custom_prop = get_custom_prop

    return fl_ctx


@pytest.fixture
def mock_stream_context():
    """Mock stream context for testing."""
    return Mock(spec=StreamContext)


@pytest.fixture
def mock_streamable_engine():
    """Mock StreamableEngine for testing."""
    engine = Mock(spec=StreamableEngine)
    engine.get_clients = Mock(return_value=["client1", "client2", "client3"])  # Default 3 clients
    engine.register_stream_processing = Mock()
    engine.fire_event = Mock()
    return engine


# Alias for backward compatibility
@pytest.fixture
def mock_engine_with_clients(mock_streamable_engine):
    """Alias for mock_streamable_engine."""
    return mock_streamable_engine


@pytest.fixture
def sample_nested_tensors(random_torch_tensors):
    """Create nested tensor structure similar to what consumer would produce."""
    return {
        "encoder": {k: v for k, v in list(random_torch_tensors.items())[:5]},
        "decoder": {k: v for k, v in list(random_torch_tensors.items())[5:]},
    }


@pytest.fixture
def sample_dxo_weights(random_torch_tensors) -> DXO:
    """Create a sample DXO with WEIGHTS data kind."""
    dxo = DXO(data_kind=DataKind.WEIGHTS, data=random_torch_tensors)
    return dxo


@pytest.fixture
def sample_dxo_nested_weights(random_torch_tensors) -> DXO:
    """Create a sample DXO with nested weights structure."""
    nested_data = {
        "encoder": {k: v for k, v in list(random_torch_tensors.items())[:5]},
        "decoder": {k: v for k, v in list(random_torch_tensors.items())[5:]},
    }
    dxo = DXO(data_kind=DataKind.WEIGHTS, data=nested_data)
    return dxo


@pytest.fixture
def sample_dxo_weight_diff(random_torch_tensors) -> DXO:
    """Create a sample DXO with WEIGHT_DIFF data kind."""
    # Create small weight differences
    diff_tensors = {k: v * 0.1 for k, v in random_torch_tensors.items()}
    dxo = DXO(data_kind=DataKind.WEIGHT_DIFF, data=diff_tensors)
    return dxo


@pytest.fixture
def sample_dxo_with_numpy(random_torch_tensors) -> DXO:
    """Create a sample DXO with mixed torch tensors and numpy arrays."""
    mixed_data = {}
    for i, (k, v) in enumerate(random_torch_tensors.items()):
        if i % 2 == 0:  # Keep half as torch tensors
            mixed_data[k] = v
        else:  # Convert half to numpy arrays
            mixed_data[k] = v.numpy()

    dxo = DXO(data_kind=DataKind.WEIGHTS, data=mixed_data)
    return dxo


@pytest.fixture
def sample_shareable_with_dxo():
    """Create a sample Shareable containing a DXO structure."""
    shareable = Shareable()
    shareable["DXO"] = {"kind": DataKind.WEIGHTS, "data": {}}  # Empty data, will be populated by receiver
    return shareable


@pytest.fixture
def sample_shareable_with_weight_diff_dxo():
    """Create a sample Shareable containing a WEIGHT_DIFF DXO structure."""
    shareable = Shareable()
    shareable["DXO"] = {
        "kind": DataKind.WEIGHT_DIFF,
        "data": {"existing_param": torch.randn(2, 2)},  # Some existing data
    }
    return shareable
