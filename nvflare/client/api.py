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
from nvflare.app_common.data_exchange.constants import ExchangeFormat
from nvflare.app_common.data_exchange.data_exchanger import DataExchangeException, DataExchanger
from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.import_utils import optional_import
from nvflare.fuel.utils.pipe.file_pipe import FilePipe

from .config import ClientConfig, ConfigKey, from_file
from .constants import CONFIG_EXCHANGE
from .model_registry import ModelRegistry

PROCESS_MODEL_REGISTRY = None


def init(config: Union[str, Dict] = f"config/{CONFIG_EXCHANGE}", rank: Optional[str] = None):
    """Initializes NVFlare Client API environment.

    Args:
        config (str or dict): configuration file or config dictionary.
        rank (str): local rank of the process.
            It is only useful when the training script has multiple worker processes. (for example multi GPU)

    Example:
        ``init(config="./config.json")``
    """
    global PROCESS_MODEL_REGISTRY  # Declare PROCESS_MODEL_REGISTRY as global

    try:
        if rank is None:
            rank = os.environ.get("RANK", "0")

        if PROCESS_MODEL_REGISTRY:
            raise RuntimeError("Can't call init twice.")

        if isinstance(config, str):
            client_config = from_file(config_file=config)
        elif isinstance(config, dict):
            client_config = ClientConfig(config=config)
        else:
            raise ValueError("config should be either a string or dictionary.")

        dx = None
        if rank == "0":
            if client_config.get_exchange_format() == ExchangeFormat.PYTORCH:
                tensor_decomposer, ok = optional_import(
                    module="nvflare.app_opt.pt.decomposers", name="TensorDecomposer"
                )
                if ok:
                    fobs.register(tensor_decomposer)
                else:
                    raise RuntimeError(f"Can't import TensorDecomposer for format: {ExchangeFormat.PYTORCH}")

            pipe_args = client_config.get_pipe_args()
            if client_config.get_pipe_class() == "FilePipe":
                pipe = FilePipe(**pipe_args)
            else:
                raise RuntimeError(f"Pipe class {client_config.get_pipe_class()} is not supported.")

            dx = DataExchanger(
                supported_topics=client_config.get_supported_topics(),
                pipe=pipe,
                pipe_name=client_config.get_pipe_name(),
            )

        PROCESS_MODEL_REGISTRY = ModelRegistry(client_config, rank, dx)
    except Exception as e:
        print(f"Exception {e} happens in flare.init()")


def _get_model_registry() -> ModelRegistry:
    if PROCESS_MODEL_REGISTRY is None:
        raise RuntimeError("needs to call init method first")
    return PROCESS_MODEL_REGISTRY


def receive(timeout: Optional[float] = None) -> Optional[FLModel]:
    """Receives model from NVFlare side.

    Args:
        timeout (float, optional): timeout to receive an FLModel

    Returns:
        An FLModel received.

    Example:
        ``input_model = receive()``
    """
    model_registry = _get_model_registry()
    return model_registry.get_model(timeout)


def send(fl_model: FLModel, clear_registry: bool = True) -> None:
    """Sends the model to NVFlare side.

    Args:
        fl_model (FLModel): FLModel to be sent.
        clear_registry (bool): whether to clear the model registry after send

    Example:
        ``send(model=FLModel(...))``
    """
    model_registry = _get_model_registry()
    model_registry.submit_model(model=fl_model)
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

    Example:
        ``sys_info = system_info()``
    """
    model_registry = _get_model_registry()
    return model_registry.get_sys_info()


def get_job_id() -> str:
    """Gets the NVFlare job id.

    Returns:
        The id of the job.
    """
    sys_info = system_info()
    return sys_info.get(ConfigKey.JOB_ID, "")


def get_total_rounds() -> int:
    """Gets the total_rounds of the job.

    Returns:
        The total_rounds of the job.
    """
    sys_info = system_info()
    return sys_info.get(ConfigKey.TOTAL_ROUNDS, 0)


def get_identity() -> str:
    """Gets the identity of the site that this script is running on from NVFlare.

    Returns:
        The identity of the site that this script is running on.
    """
    sys_info = system_info()
    return sys_info.get(ConfigKey.IDENTITY, "")


def is_running() -> bool:
    """Checks if NVFlare job is running.

    Returns:
        If NVFlare job is running.
    """
    try:
        receive()
        return True
    except DataExchangeException:
        return False


def is_train() -> bool:
    """Checks if the task received from NVFlare is a training task.

    Returns:
        If the task is a training task.
    """
    model_registry = _get_model_registry()
    if model_registry.rank != "0":
        raise RuntimeError("only rank 0 can call is_train!")
    return model_registry.task_name == model_registry.config.config[ConfigKey.TRAIN_TASK_NAME]


def is_evaluate() -> bool:
    """Checks if the task received from NVFlare is an evaluation task.

    Returns:
        If the task is an evaluation task.
    """
    model_registry = _get_model_registry()
    if model_registry.rank != "0":
        raise RuntimeError("only rank 0 can call is_evaluate!")
    return model_registry.task_name == model_registry.config.config[ConfigKey.EVAL_TASK_NAME]


def is_submit_model() -> bool:
    """Checks if the task received from NVFlare is a submit_model task.

    Returns:
        If the task is a submit_model task.
    """
    model_registry = _get_model_registry()
    if model_registry.rank != "0":
        raise RuntimeError("only rank 0 can call is_submit_model!")
    return model_registry.task_name == model_registry.config.config[ConfigKey.SUBMIT_MODEL_TASK_NAME]
