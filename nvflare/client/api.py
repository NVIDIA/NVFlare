# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import copy
import inspect
import json
import os
from typing import Any, Callable, Dict, Optional, Tuple, Union

from nvflare.app_common.model_exchange.file_pipe_model_exchanger import FilePipeModelExchanger
from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.import_utils import optional_import

from .config import ClientConfig
from .constants import ModelExchangeFormat
from .model_cache import Cache
from .utils import numerical_params_diff

PROCESS_CACHE: Dict[int, Cache] = {}


# TODO: some other helper methods:
#   - get_total_rounds()
#   - get_job_id()


def _check_param_diff_func(params_diff_func: Callable):
    num_of_args = len(inspect.getfullargspec(params_diff_func).args)
    if num_of_args != 2:
        raise RuntimeError("params_diff_func need to have signature `params_diff_func(original model, new model)`")


def init(
    config: Union[str, Dict] = "config/config_exchange.json",
    params_diff_func: Optional[Callable] = numerical_params_diff,
):
    """Initializes NVFlare Client API environment.

    Args:
        config (str or dict): configuration file or config dictionary.
        params_diff_func (Callable): a function to calculate the params difference if the params_type is "DIFF"
            default is "nvflare.client.utils.numerical_params_diff"

    Note that params_diff_func signature:
        params_diff_func(original model, new model) -> model difference
    """
    pid = os.getpid()
    if pid in PROCESS_CACHE:
        raise RuntimeError("Can't call init twice.")

    if isinstance(config, str):
        if not os.path.exists(config):
            raise RuntimeError(f"Missing config file {config}.")

        with open(config, "r") as f:
            config_dict = json.load(f)
    elif isinstance(config, dict):
        config_dict = config
    else:
        raise ValueError("config should be either a string or dictionary.")

    if params_diff_func is not None:
        _check_param_diff_func(params_diff_func)

    client_config = ClientConfig(config=config_dict)
    if client_config.get_exchange_format() == ModelExchangeFormat.PYTORCH:
        tensor_decomposer, ok = optional_import(module="nvflare.app_opt.pt.decomposers", name="TensorDecomposer")
        if ok:
            fobs.register(tensor_decomposer)
        else:
            raise RuntimeError(f"Can't import TensorDecomposer for format: {ModelExchangeFormat.PYTORCH}")

    mdx = FilePipeModelExchanger(data_exchange_path=client_config.get_exchange_path())
    PROCESS_CACHE[pid] = Cache(mdx, client_config, params_diff_func)


def receive_model() -> Tuple[Any, Dict]:
    """Receives model from NVFlare side.

    Returns:
        A tuple of model, metadata received.
    """
    pid = os.getpid()
    if pid not in PROCESS_CACHE:
        raise RuntimeError("needs to call init method first")
    cache = PROCESS_CACHE[pid]
    model = copy.deepcopy(cache.input_model.params)
    meta = cache.meta
    return model, meta


def submit_metrics(metrics: Dict, meta: Optional[Dict] = None) -> None:
    """Submits metrics.

    Args:
        metrics (Dict): metrics to be submitted.
        meta: the metadata to be submitted along with the metrics.
    """
    pid = os.getpid()
    if pid not in PROCESS_CACHE:
        raise RuntimeError("needs to call init method first")
    cache = PROCESS_CACHE[pid]
    if meta is not None:
        cache.meta.update(meta)
    cache.metrics = metrics


def submit_model(model: Any, meta: Optional[Dict] = None) -> None:
    """Submits model.

    Args:
        model: model to be submitted.
        meta: the metadata to be submitted along with the model.
    """
    pid = os.getpid()
    if pid not in PROCESS_CACHE:
        raise RuntimeError("needs to call init method first")
    cache = PROCESS_CACHE[pid]
    if meta is not None:
        cache.meta.update(meta)

    cache.output_params = model


def send_model() -> None:
    """Sends the model to NVFlare side."""
    pid = os.getpid()
    if pid not in PROCESS_CACHE:
        raise RuntimeError("needs to call init method first")
    cache = PROCESS_CACHE[pid]
    if cache.output_params is None:
        raise RuntimeError("no model to send to NVFlare side.")

    fl_model = cache.construct_fl_model(params=cache.output_params)
    cache.model_exchanger.submit_model(model=fl_model)
    cache.model_exchanger.finalize(close_pipe=False)
    PROCESS_CACHE.pop(pid)


def get_sys_meta() -> Dict:
    """Gets NVFlare system metadata.

    Returns:
       A dict of system metadata.
    """
    pid = os.getpid()
    if pid not in PROCESS_CACHE:
        raise RuntimeError("needs to call init method first")
    cache = PROCESS_CACHE[pid]
    return cache.sys_meta
