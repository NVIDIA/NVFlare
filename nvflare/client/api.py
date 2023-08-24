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

import os
from typing import Dict, Union

from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.model_exchange.file_pipe_model_exchanger import FilePipeModelExchanger
from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.import_utils import optional_import

from .config import ClientConfig, from_file
from .constants import CONFIG_EXCHANGE, ModelExchangeFormat
from .model_cache import Cache
from .utils import DIFF_FUNCS

PROCESS_CACHE: Dict[int, Cache] = {}

# TODO: some other helper methods:
#   - get_total_rounds()
#   - get_job_id()


def init(config: Union[str, Dict] = f"config/{CONFIG_EXCHANGE}"):
    """Initializes NVFlare Client API environment.

    Args:
        config (str or dict): configuration file or config dictionary.
    """
    pid = os.getpid()
    if pid in PROCESS_CACHE:
        raise RuntimeError("Can't call init twice.")

    if isinstance(config, str):
        client_config = from_file(config_file=config)
    elif isinstance(config, dict):
        client_config = ClientConfig(config=config)
    else:
        raise ValueError("config should be either a string or dictionary.")

    if client_config.get_exchange_format() == ModelExchangeFormat.PYTORCH:
        tensor_decomposer, ok = optional_import(module="nvflare.app_opt.pt.decomposers", name="TensorDecomposer")
        if ok:
            fobs.register(tensor_decomposer)
        else:
            raise RuntimeError(f"Can't import TensorDecomposer for format: {ModelExchangeFormat.PYTORCH}")

    mdx = FilePipeModelExchanger(data_exchange_path=client_config.get_exchange_path())
    PROCESS_CACHE[pid] = Cache(mdx, client_config)


def receive() -> FLModel:
    """Receives model from NVFlare side.

    Returns:
        A tuple of model, metadata received.
    """
    pid = os.getpid()
    if pid not in PROCESS_CACHE:
        raise RuntimeError("needs to call init method first")
    cache = PROCESS_CACHE[pid]
    cache.receive()
    return cache.input_model


def send(fl_model: FLModel, clear=True) -> None:
    """Sends the model to NVFlare side."""
    pid = os.getpid()
    if pid not in PROCESS_CACHE:
        raise RuntimeError("needs to call init method first")

    cache = PROCESS_CACHE[pid]
    cache.send(model=fl_model)
    if clear:
        cache.model_exchanger.finalize(close_pipe=False)
        PROCESS_CACHE.pop(pid)


def clear():
    pid = os.getpid()
    if pid not in PROCESS_CACHE:
        raise RuntimeError("needs to call init method first")
    cache = PROCESS_CACHE[pid]
    cache.model_exchanger.finalize(close_pipe=False)
    PROCESS_CACHE.pop(pid)


def system_info() -> Dict:
    """Gets NVFlare system information.

    Returns:
       A dict of system information.
    """
    pid = os.getpid()
    if pid not in PROCESS_CACHE:
        raise RuntimeError("needs to call init method first")
    cache = PROCESS_CACHE[pid]
    return cache.sys_info


def params_diff(original: Dict, new: Dict) -> Dict:
    pid = os.getpid()
    if pid not in PROCESS_CACHE:
        raise RuntimeError("needs to call init method first")
    cache = PROCESS_CACHE[pid]
    diff_func = DIFF_FUNCS.get(cache.config.get_exchange_format(), None)
    if diff_func is None:
        raise RuntimeError("no default params diff function")
    return diff_func(original, new)


def get_config() -> Dict:
    pid = os.getpid()
    if pid not in PROCESS_CACHE:
        raise RuntimeError("needs to call init method first")
    cache = PROCESS_CACHE[pid]
    return cache.config.config
