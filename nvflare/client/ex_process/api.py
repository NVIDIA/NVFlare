# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import importlib
import os
from typing import Any, Dict, Optional, Tuple

from nvflare.apis.analytix import AnalyticsDataType
from nvflare.apis.fl_constant import FLMetaKey
from nvflare.apis.utils.analytix_utils import create_analytic_dxo
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.client.api_spec import APISpec
from nvflare.client.config import ClientConfig, ConfigKey, ExchangeFormat, from_file
from nvflare.client.constants import CLIENT_API_CONFIG
from nvflare.client.flare_agent import FlareAgentException
from nvflare.client.flare_agent_with_fl_model import FlareAgentWithFLModel
from nvflare.client.model_registry import ModelRegistry
from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.import_utils import optional_import
from nvflare.fuel.utils.obj_utils import get_logger
from nvflare.fuel.utils.pipe.pipe import Pipe


def _create_client_config(config: str) -> ClientConfig:
    if isinstance(config, str):
        client_config = from_file(config_file=config)
    else:
        raise ValueError(f"config should be a string but got: {type(config)}")
    return client_config


def _create_pipe_using_config(client_config: ClientConfig, section: str) -> Tuple[Pipe, str]:
    pipe_class_name = client_config.get_pipe_class(section)
    module_name, _, class_name = pipe_class_name.rpartition(".")
    module = importlib.import_module(module_name)
    pipe_class = getattr(module, class_name)

    pipe_args = client_config.get_pipe_args(section)
    pipe = pipe_class(**pipe_args)
    pipe_channel_name = client_config.get_pipe_channel_name(section)
    return pipe, pipe_channel_name


def _register_tensor_decomposer():
    tensor_decomposer, ok = optional_import(module="nvflare.app_opt.pt.decomposers", name="TensorDecomposer")
    if ok:
        fobs.register(tensor_decomposer)
    else:
        raise RuntimeError(f"Can't import TensorDecomposer for format: {ExchangeFormat.PYTORCH}")


class ExProcessClientAPI(APISpec):
    def __init__(self):
        self.process_model_registry = None
        self.logger = get_logger(self)

    def get_model_registry(self) -> ModelRegistry:
        """Gets the ModelRegistry."""
        if self.process_model_registry is None:
            raise RuntimeError("needs to call init method first")
        return self.process_model_registry

    def init(self, rank: Optional[str] = None):
        """Initializes NVFlare Client API environment.

        Args:
            rank (str): local rank of the process.
                It is only useful when the training script has multiple worker processes. (for example multi GPU)
        """

        if rank is None:
            rank = os.environ.get("RANK", "0")

        if self.process_model_registry:
            self.logger.warning("Warning: called init() more than once. The subsequence calls are ignored")
            return

        config_file = f"config/{CLIENT_API_CONFIG}"
        client_config = _create_client_config(config=config_file)

        flare_agent = None
        try:
            if rank == "0":
                if client_config.get_exchange_format() == ExchangeFormat.PYTORCH:
                    _register_tensor_decomposer()

                pipe, task_channel_name = None, ""
                if ConfigKey.TASK_EXCHANGE in client_config.config:
                    pipe, task_channel_name = _create_pipe_using_config(
                        client_config=client_config, section=ConfigKey.TASK_EXCHANGE
                    )
                metric_pipe, metric_channel_name = None, ""
                if ConfigKey.METRICS_EXCHANGE in client_config.config:
                    metric_pipe, metric_channel_name = _create_pipe_using_config(
                        client_config=client_config, section=ConfigKey.METRICS_EXCHANGE
                    )

                flare_agent = FlareAgentWithFLModel(
                    pipe=pipe,
                    task_channel_name=task_channel_name,
                    metric_pipe=metric_pipe,
                    metric_channel_name=metric_channel_name,
                    heartbeat_timeout=client_config.get_heartbeat_timeout(),
                )
                flare_agent.start()

            self.process_model_registry = ModelRegistry(client_config, rank, flare_agent)
        except Exception as e:
            self.logger.error(f"flare.init failed: {e}")
            raise e

    def receive(self, timeout: Optional[float] = None) -> Optional[FLModel]:
        """Receives model from NVFlare side.

        Returns:
            An FLModel received.
        """
        model_registry = self.get_model_registry()
        return model_registry.get_model(timeout)

    def send(self, model: FLModel, clear_cache: bool = True) -> None:
        """Sends the model to Controller side.
        Args:
            model (FLModel): Sends a FLModel object.
            clear_cache (bool): To clear the cache or not.
        """
        model_registry = self.get_model_registry()
        model_registry.submit_model(model=model)
        if clear_cache:
            self.clear()

    def system_info(self) -> Dict:
        """Gets NVFlare system information.

        System information will be available after a valid FLModel is received.
        It does not retrieve information actively.

        Returns:
           A dict of system information.
        """
        model_registry = self.get_model_registry()
        return model_registry.get_sys_info()

    def get_config(self) -> Dict:
        model_registry = self.get_model_registry()
        return model_registry.config.config

    def get_job_id(self) -> str:
        sys_info = self.system_info()
        return sys_info.get(FLMetaKey.JOB_ID, "")

    def get_site_name(self) -> str:
        sys_info = self.system_info()
        return sys_info.get(FLMetaKey.SITE_NAME, "")

    def get_task_name(self) -> str:
        model_registry = self.get_model_registry()
        if model_registry.rank != "0":
            raise RuntimeError("only rank 0 can call get_task_name!")
        return model_registry.get_task().task_name

    def is_running(self) -> bool:
        try:
            self.receive()
            return True
        except FlareAgentException:
            return False

    def is_train(self) -> bool:
        model_registry = self.get_model_registry()
        if model_registry.rank != "0":
            raise RuntimeError("only rank 0 can call is_train!")
        return model_registry.task_name == model_registry.config.get_train_task()

    def is_evaluate(self) -> bool:
        model_registry = self.get_model_registry()
        if model_registry.rank != "0":
            raise RuntimeError("only rank 0 can call is_evaluate!")
        return model_registry.task_name == model_registry.config.get_eval_task()

    def is_submit_model(self) -> bool:
        model_registry = self.get_model_registry()
        if model_registry.rank != "0":
            raise RuntimeError("only rank 0 can call is_submit_model!")
        return model_registry.task_name == model_registry.config.get_submit_model_task()

    def log(self, key: str, value: Any, data_type: AnalyticsDataType, **kwargs):
        model_registry = self.get_model_registry()
        if model_registry.rank != "0":
            raise RuntimeError("only rank 0 can call log!")

        flare_agent = model_registry.flare_agent
        dxo = create_analytic_dxo(tag=key, value=value, data_type=data_type, **kwargs)
        flare_agent.log(dxo)

    def clear(self):
        """Clears the model registry."""
        model_registry = self.get_model_registry()
        model_registry.clear()
