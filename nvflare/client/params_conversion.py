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

"""Parameter representation adaptation at the trainer-side Client API boundary.

The job declares both representations in ``TASK_EXCHANGE``.  These functions honor that
declaration without inspecting payload values to guess a framework.  State needed to round-trip
PyTorch state dictionaries (tensor shapes and non-tensor entries) is an ordinary caller-owned
dictionary, so this module does not recreate the legacy ``ParamsConverter`` component hierarchy.
"""

from typing import Any, MutableMapping, Optional

import numpy as np

from nvflare.client.config import ExchangeFormat
from nvflare.fuel.utils.import_utils import optional_import

_TENSOR_SHAPES = "tensor_shapes"
_EXCLUDE_VARS = "exclude_vars"


def normalize_exchange_format(value, name: str) -> ExchangeFormat:
    """Return a validated ``ExchangeFormat`` value for a config declaration."""

    try:
        return ExchangeFormat(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"invalid {name} {value!r}: must be one of {list(ExchangeFormat)}") from e


def validate_format_pair(params_exchange_format, server_expected_format) -> None:
    """Validate a declared trainer/server representation pair without importing a framework."""

    client_format = normalize_exchange_format(params_exchange_format, "params_exchange_format")
    server_format = normalize_exchange_format(server_expected_format, "server_expected_format")
    if client_format == server_format or ExchangeFormat.RAW in (client_format, server_format):
        return
    supported = {
        frozenset((ExchangeFormat.NUMPY, ExchangeFormat.PYTORCH)),
        frozenset((ExchangeFormat.NUMPY, ExchangeFormat.KERAS_LAYER_WEIGHTS)),
    }
    if frozenset((client_format, server_format)) not in supported:
        raise ValueError(f"unsupported parameter format conversion: {server_format.value} <-> {client_format.value}")


def convert_params(
    params: Any,
    source_format,
    target_format,
    state: MutableMapping[str, Any],
    logger: Optional[Any] = None,
) -> Any:
    """Convert parameters according to an explicit source/target declaration.

    ``RAW`` is an explicit no-adaptation declaration.  Framework packages are imported only
    when their declared conversion is actually executed in the trainer process.
    """

    if params is None:
        return None
    source = normalize_exchange_format(source_format, "source_format")
    target = normalize_exchange_format(target_format, "target_format")
    validate_format_pair(source, target)
    if source == target or ExchangeFormat.RAW in (source, target):
        return params

    if source == ExchangeFormat.NUMPY and target == ExchangeFormat.PYTORCH:
        return _numpy_to_pytorch(params, state)
    if source == ExchangeFormat.PYTORCH and target == ExchangeFormat.NUMPY:
        return _pytorch_to_numpy(params, state, logger)
    if source == ExchangeFormat.NUMPY and target == ExchangeFormat.KERAS_LAYER_WEIGHTS:
        from nvflare.app_opt.tf.utils import unflat_layer_weights_dict

        return unflat_layer_weights_dict(params)
    if source == ExchangeFormat.KERAS_LAYER_WEIGHTS and target == ExchangeFormat.NUMPY:
        from nvflare.app_opt.tf.utils import flat_layer_weights_dict

        return flat_layer_weights_dict(params)

    # validate_format_pair() keeps this defensive branch unreachable.
    raise ValueError(f"unsupported parameter format conversion: {source.value} -> {target.value}")


def _load_torch():
    torch, ok = optional_import(module="torch")
    if not ok:
        raise RuntimeError("PyTorch is required for the declared PyTorch parameter exchange format")
    return torch


def _numpy_to_pytorch(params: Any, state: MutableMapping[str, Any]) -> Any:
    if not isinstance(params, dict):
        raise TypeError(f"NumPy-to-PyTorch conversion expects a parameter dict, got {type(params)}")
    torch = _load_torch()
    tensor_shapes = state.get(_TENSOR_SHAPES) or {}
    converted = {
        key: torch.as_tensor(np.reshape(value, tensor_shapes[key])) if key in tensor_shapes else torch.as_tensor(value)
        for key, value in params.items()
    }
    converted.update(state.get(_EXCLUDE_VARS) or {})
    return converted


def _pytorch_to_numpy(params: Any, state: MutableMapping[str, Any], logger: Optional[Any]) -> Any:
    if not isinstance(params, dict):
        raise TypeError(f"PyTorch-to-NumPy conversion expects a parameter dict, got {type(params)}")
    torch = _load_torch()
    converted = {}
    tensor_shapes = {}
    exclude_vars = {}
    for key, value in params.items():
        if isinstance(value, torch.Tensor):
            try:
                converted[key] = value.detach().cpu().numpy()
            except Exception as e:
                raise ValueError(
                    "failed to convert a PyTorch tensor to NumPy; use a NumPy-supported dtype "
                    f"or declare a native PyTorch server exchange format: {e}"
                ) from e
            tensor_shapes[key] = value.shape
        else:
            exclude_vars[key] = value

    # This state is session-scoped. Replace both maps on every outgoing conversion so
    # entries removed or changed between rounds cannot be restored from stale state.
    state[_TENSOR_SHAPES] = tensor_shapes
    state[_EXCLUDE_VARS] = exclude_vars
    if exclude_vars:
        if logger is not None:
            logger.warning(
                f"{len(exclude_vars)} vars excluded because they are not tensors: {list(exclude_vars.keys())}"
            )
    return converted
