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

from typing import Any, Callable, MutableMapping, Optional

from nvflare.client.config import ExchangeFormat, normalize_exchange_format
from nvflare.fuel.utils.import_utils import optional_import

_ADAPTER_SPECS = {
    (ExchangeFormat.NUMPY, ExchangeFormat.PYTORCH): (
        "_numpy_to_pytorch",
        ExchangeFormat.PYTORCH,
    ),
    (ExchangeFormat.PYTORCH, ExchangeFormat.NUMPY): (
        "_pytorch_to_numpy",
        ExchangeFormat.PYTORCH,
    ),
    (ExchangeFormat.NUMPY, ExchangeFormat.KERAS_LAYER_WEIGHTS): (
        "_numpy_to_keras_layer_weights",
        ExchangeFormat.KERAS_LAYER_WEIGHTS,
    ),
    (ExchangeFormat.KERAS_LAYER_WEIGHTS, ExchangeFormat.NUMPY): (
        "_keras_layer_weights_to_numpy",
        ExchangeFormat.KERAS_LAYER_WEIGHTS,
    ),
}


def validate_format_pair(source_format, target_format) -> None:
    """Validate that both directions of a declared format pair are supported."""

    source = normalize_exchange_format(source_format, "source_format")
    target = normalize_exchange_format(target_format, "target_format")
    if source == target or ExchangeFormat.RAW in (source, target):
        return
    if (source, target) not in _ADAPTER_SPECS or (target, source) not in _ADAPTER_SPECS:
        raise ValueError(f"unsupported parameter format conversion: {source.value} <-> {target.value}")


def _load_dependency(module: str, name: str, format_name: str):
    dependency, ok = optional_import(module=module, name=name)
    if not ok:
        raise RuntimeError(f"Can't import {name} for {format_name} exchange format")
    return dependency


def _numpy_to_pytorch(params, state: MutableMapping[str, Any], logger=None):
    np = _load_dependency("numpy", "", ExchangeFormat.PYTORCH)
    torch = _load_dependency("torch", "", ExchangeFormat.PYTORCH)
    tensor_shapes = state.get("tensor_shapes")
    exclude_vars = state.get("exclude_vars")

    if tensor_shapes:
        result = {
            key: (
                torch.as_tensor(np.reshape(value, tensor_shapes[key]))
                if key in tensor_shapes
                else torch.as_tensor(value)
            )
            for key, value in params.items()
        }
    else:
        result = {key: torch.as_tensor(value) for key, value in params.items()}
    if exclude_vars:
        result.update(exclude_vars)
    return result


def _pytorch_to_numpy(params, state: MutableMapping[str, Any], logger=None):
    torch = _load_dependency("torch", "", ExchangeFormat.PYTORCH)
    result = {}
    tensor_shapes = {}
    exclude_vars = {}
    for key, value in params.items():
        if isinstance(value, torch.Tensor):
            try:
                result[key] = value.detach().cpu().numpy()
            except Exception as e:
                raise ValueError(
                    "failed to convert a PyTorch tensor to NumPy; use a NumPy-supported dtype "
                    f"or declare a native PyTorch server exchange format: {e}"
                ) from e
            tensor_shapes[key] = value.shape
        else:
            exclude_vars[key] = value

    state["tensor_shapes"] = tensor_shapes
    state["exclude_vars"] = exclude_vars
    if exclude_vars and logger is not None:
        logger.warning(f"{len(exclude_vars)} vars excluded as they were non-tensor type: {list(exclude_vars.keys())}")
    return result


def _numpy_to_keras_layer_weights(params, state: MutableMapping[str, Any], logger=None):
    from nvflare.app_opt.tf.utils import unflat_layer_weights_dict

    return unflat_layer_weights_dict(params)


def _keras_layer_weights_to_numpy(params, state: MutableMapping[str, Any], logger=None):
    from nvflare.app_opt.tf.utils import flat_layer_weights_dict

    return flat_layer_weights_dict(params)


def _get_adapter(source_format, target_format) -> Callable:
    adapter_name, _ = _ADAPTER_SPECS[(source_format, target_format)]
    return globals()[adapter_name]


def convert_params(
    params: Any,
    source_format,
    target_format,
    state: MutableMapping[str, Any],
    logger: Optional[Any] = None,
) -> Any:
    """Adapt trainer parameters according to an explicit source/target declaration."""

    if params is None:
        return None
    source = normalize_exchange_format(source_format, "source_format")
    target = normalize_exchange_format(target_format, "target_format")
    validate_format_pair(source, target)
    if source == target or ExchangeFormat.RAW in (source, target):
        return params

    if ExchangeFormat.PYTORCH in (source, target) and not isinstance(params, dict):
        raise TypeError(f"PyTorch parameter conversion expects a parameter dict, got {type(params)}")
    if ExchangeFormat.KERAS_LAYER_WEIGHTS in (source, target) and not isinstance(params, dict):
        raise TypeError(f"Keras layer-weight conversion expects a parameter dict, got {type(params)}")

    return _get_adapter(source, target)(params, state, logger)
