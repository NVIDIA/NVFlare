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
from typing import Dict, Optional, Union

from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.model_exchange.constants import ModelExchangeFormat
from nvflare.app_common.model_exchange.file_pipe_model_exchanger import FilePipeModelExchanger
from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.import_utils import optional_import

from .config import ClientConfig, ConfigKey, from_file
from .constants import CONFIG_EXCHANGE
from .model_registry import ModelRegistry
from .utils import DIFF_FUNCS

PROCESS_MODEL_REGISTRY: Dict[int, ModelRegistry] = {}


def init(config: Union[str, Dict] = f"config/{CONFIG_EXCHANGE}", rank: Optional[str] = None):
    """Initializes NVFlare Client API environment.

    Args:
        config (str or dict): configuration file or config dictionary.
        rank (str): local rank of the process. It is only useful when the training script has multiple worker processes. (for example multi GPU)
    """
    if rank is None:
        rank = os.environ.get("RANK", "0")
    pid = os.getpid()
    if pid in PROCESS_MODEL_REGISTRY:
        raise RuntimeError("Can't call init twice.")

    if isinstance(config, str):
        client_config = from_file(config_file=config)
    elif isinstance(config, dict):
        client_config = ClientConfig(config=config)
    else:
        raise ValueError("config should be either a string or dictionary.")

    mdx = None
    if rank == "0":
        if client_config.get_exchange_format() == ModelExchangeFormat.PYTORCH:
            tensor_decomposer, ok = optional_import(module="nvflare.app_opt.pt.decomposers", name="TensorDecomposer")
            if ok:
                fobs.register(tensor_decomposer)
            else:
                raise RuntimeError(f"Can't import TensorDecomposer for format: {ModelExchangeFormat.PYTORCH}")

        # TODO: make Pipe configurable
        mdx = FilePipeModelExchanger(data_exchange_path=client_config.get_exchange_path())

    PROCESS_MODEL_REGISTRY[pid] = ModelRegistry(client_config, rank, mdx)


def _get_model_registry() -> Optional[ModelRegistry]:
    pid = os.getpid()
    if pid not in PROCESS_MODEL_REGISTRY:
        raise RuntimeError("needs to call init method first")
    return PROCESS_MODEL_REGISTRY[pid]


def receive() -> FLModel:
    """Receives model from NVFlare side.

    Returns:
        An FLModel received.
    """
    model_registry = _get_model_registry()
    return model_registry.get_model()


def send(fl_model: FLModel, clear_registry: bool = True) -> None:
    """Sends the model to NVFlare side.

    Args:
        clear_registry (bool): To clear the registry or not.
    """
    model_registry = _get_model_registry()
    model_registry.send(model=fl_model)
    if clear_registry:
        clear()


def clear():
    """Clears the model registry."""
    model_registry = _get_model_registry()
    model_registry.clear()


def system_info() -> Dict:
    """Gets NVFlare system information.

    System information will be available after a valid FLModel is received.
    It does not retrieve information actively.

    Returns:
       A dict of system information.
    """
    model_registry = _get_model_registry()
    return model_registry.get_sys_info()


def params_diff(original: Dict, new: Dict) -> Dict:
    model_registry = _get_model_registry()
    diff_func = DIFF_FUNCS.get(model_registry.config.get_exchange_format(), None)
    if diff_func is None:
        raise RuntimeError("no default params diff function")
    return diff_func(original, new)


def get_config() -> Dict:
    model_registry = _get_model_registry()
    return model_registry.config.config


def get_job_id() -> str:
    sys_info = system_info()
    return sys_info.get(ConfigKey.JOB_ID, "")


def get_total_rounds() -> int:
    sys_info = system_info()
    return sys_info.get(ConfigKey.TOTAL_ROUNDS, 0)


def get_site_name() -> str:
    sys_info = system_info()
    return sys_info.get(ConfigKey.SITE_NAME, "")


def receive_global_model():
    """Yields model received from NVFlare server."""
    sys_info = system_info()
    total_rounds = sys_info["total_rounds"]
    is_last_round = False
    launch_once = get_config().get("launch_once", True)

    while True:
        input_model = receive()
        current_round = input_model.current_round
        yield input_model
        is_last_round = current_round == total_rounds - 1
        if launch_once and is_last_round:
            break
        elif not launch_once:
            break
        clear()
