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

from typing import Optional, Tuple

from nvflare.app_common.abstract.params_converter import ParamsConverter
from nvflare.client.config import ExchangeFormat
from nvflare.fuel.utils.import_utils import optional_import


def _load_converter(module: str, name: str, format_name: str):
    converter_cls, ok = optional_import(module=module, name=name)
    if not ok:
        raise RuntimeError(f"Can't import {name} for {format_name} exchange format")
    return converter_cls


def create_default_params_converters(
    server_expected_format: str,
    params_exchange_format: str,
    train_task_name: str,
    eval_task_name: str,
    submit_model_task_name: str,
) -> Tuple[Optional[ParamsConverter], Optional[ParamsConverter]]:
    """Create default from/to NVFlare converters for common Client API formats."""
    if server_expected_format != ExchangeFormat.NUMPY:
        return None, None

    if params_exchange_format == ExchangeFormat.PYTORCH:
        numpy_to_pt = _load_converter(
            module="nvflare.app_opt.pt.numpy_params_converter",
            name="NumpyToPTParamsConverter",
            format_name=ExchangeFormat.PYTORCH,
        )
        pt_to_numpy = _load_converter(
            module="nvflare.app_opt.pt.numpy_params_converter",
            name="PTToNumpyParamsConverter",
            format_name=ExchangeFormat.PYTORCH,
        )
        return (
            numpy_to_pt([train_task_name, eval_task_name]),
            pt_to_numpy([train_task_name, submit_model_task_name]),
        )

    if params_exchange_format == ExchangeFormat.KERAS_LAYER_WEIGHTS:
        numpy_to_keras = _load_converter(
            module="nvflare.app_opt.tf.params_converter",
            name="NumpyToKerasModelParamsConverter",
            format_name=ExchangeFormat.KERAS_LAYER_WEIGHTS,
        )
        keras_to_numpy = _load_converter(
            module="nvflare.app_opt.tf.params_converter",
            name="KerasModelToNumpyParamsConverter",
            format_name=ExchangeFormat.KERAS_LAYER_WEIGHTS,
        )
        return (
            numpy_to_keras([train_task_name, eval_task_name]),
            keras_to_numpy([train_task_name, submit_model_task_name]),
        )

    return None, None
