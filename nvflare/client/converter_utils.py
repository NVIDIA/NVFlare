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

from typing import Any, MutableMapping, Optional, Tuple

from nvflare.app_common.abstract.params_converter import ParamsConverter
from nvflare.client.config import ExchangeFormat, normalize_exchange_format, validate_format_pair
from nvflare.fuel.utils.import_utils import optional_import


class _ConverterContext:
    """Minimal state context for ParamsConverters running in a trainer process."""

    def __init__(self, props: Optional[MutableMapping[str, Any]] = None):
        self._props = props if props is not None else {}

    def get_prop(self, key: str, default=None):
        return self._props.get(key, default)

    def set_prop(self, key: str, value: Any, private: Optional[bool] = None, sticky: Optional[bool] = None):
        _ = private
        _ = sticky
        self._props[key] = value


_CONVERTER_SPECS = {
    (ExchangeFormat.NUMPY, ExchangeFormat.PYTORCH): (
        "nvflare.app_opt.pt.numpy_params_converter",
        "NumpyToPTParamsConverter",
        ExchangeFormat.PYTORCH,
    ),
    (ExchangeFormat.PYTORCH, ExchangeFormat.NUMPY): (
        "nvflare.app_opt.pt.numpy_params_converter",
        "PTToNumpyParamsConverter",
        ExchangeFormat.PYTORCH,
    ),
    (ExchangeFormat.NUMPY, ExchangeFormat.KERAS_LAYER_WEIGHTS): (
        "nvflare.app_opt.tf.params_converter",
        "NumpyToKerasModelParamsConverter",
        ExchangeFormat.KERAS_LAYER_WEIGHTS,
    ),
    (ExchangeFormat.KERAS_LAYER_WEIGHTS, ExchangeFormat.NUMPY): (
        "nvflare.app_opt.tf.params_converter",
        "KerasModelToNumpyParamsConverter",
        ExchangeFormat.KERAS_LAYER_WEIGHTS,
    ),
}


def _load_converter(module: str, name: str, format_name: str):
    converter_cls, ok = optional_import(module=module, name=name)
    if not ok:
        raise RuntimeError(f"Can't import {name} for {format_name} exchange format")
    return converter_cls


def _create_converter(source_format, target_format, supported_tasks=None) -> Optional[ParamsConverter]:
    spec = _CONVERTER_SPECS.get((source_format, target_format))
    if spec is None:
        return None
    module, name, format_name = spec
    converter_cls = _load_converter(module=module, name=name, format_name=format_name)
    return converter_cls(supported_tasks)


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

    converter = _create_converter(source, target)
    if converter is None:
        # validate_format_pair() keeps this defensive branch unreachable.
        raise ValueError(f"unsupported parameter format conversion: {source.value} -> {target.value}")
    if logger is not None:
        converter.logger = logger
    return converter.convert(params, _ConverterContext(state))


def create_default_params_converters(
    server_expected_format: str,
    params_exchange_format: str,
    train_task_name: str,
    eval_task_name: str,
    submit_model_task_name: str,
) -> Tuple[Optional[ParamsConverter], Optional[ParamsConverter]]:
    """Create default from/to NVFlare converters for common Client API formats."""
    # Older client_api_config files can leave exchange_format empty.
    if server_expected_format != ExchangeFormat.NUMPY:
        return None, None
    if params_exchange_format not in (ExchangeFormat.PYTORCH, ExchangeFormat.KERAS_LAYER_WEIGHTS):
        return None, None
    server_format = ExchangeFormat.NUMPY
    client_format = ExchangeFormat(params_exchange_format)
    return (
        _create_converter(server_format, client_format, [train_task_name, eval_task_name]),
        _create_converter(client_format, server_format, [train_task_name, submit_model_task_name]),
    )
