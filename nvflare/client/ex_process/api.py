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
from nvflare.apis.fl_constant import ConnPropKey, FLMetaKey, WorkspaceConstants
from nvflare.apis.utils.analytix_utils import create_analytic_dxo
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.client.api_spec import APISpec
from nvflare.client.config import ClientConfig, ConfigKey, ExchangeFormat, from_file
from nvflare.client.converter_utils import create_default_params_converters
from nvflare.client.flare_agent import FlareAgentException
from nvflare.client.flare_agent_with_fl_model import FlareAgentWithFLModel
from nvflare.client.model_registry import ModelRegistry
from nvflare.fuel.data_event.utils import set_scope_property
from nvflare.fuel.utils.config_factory import ConfigFactory
from nvflare.fuel.utils.fobs import fobs
from nvflare.fuel.utils.import_utils import optional_import
from nvflare.fuel.utils.log_utils import apply_log_config, get_obj_logger
from nvflare.fuel.utils.mem_utils import log_rss
from nvflare.fuel.utils.pipe.pipe import Pipe

_ROTATING_HANDLER_CLASSES = {
    "logging.handlers.RotatingFileHandler",
    "logging.handlers.TimedRotatingFileHandler",
}
_ROTATING_ONLY_KEYS = {"maxBytes", "backupCount", "when", "interval", "utc", "atTime"}


def _downgrade_rotating_handlers(dict_config: dict) -> None:
    """Replace rotating file handlers with plain FileHandler in subprocess log config.

    Both the CJ and the subprocess write to the same log files.  Only the CJ
    should trigger rotation; RotatingFileHandler is not process-safe and two
    processes rotating the same file concurrently can corrupt it.  The subprocess
    uses plain FileHandler (append-only, no rotation) so the CJ remains the sole
    rotation manager.
    """
    for handler_cfg in dict_config.get("handlers", {}).values():
        if handler_cfg.get("class") in _ROTATING_HANDLER_CLASSES:
            handler_cfg["class"] = "logging.FileHandler"
            for key in _ROTATING_ONLY_KEYS:
                handler_cfg.pop(key, None)


def _create_client_config(config: str) -> ClientConfig:
    if isinstance(config, str):
        client_config = from_file(config_file=config)
    else:
        raise ValueError(f"config should be a string but got: {type(config)}")

    site_name = client_config.get_site_name()

    root_conn_props = client_config.get_root_conn_props()
    if root_conn_props:
        set_scope_property(site_name, ConnPropKey.ROOT_CONN_PROPS, root_conn_props)

    cp_conn_props = client_config.get_cp_conn_props()
    if cp_conn_props:
        set_scope_property(site_name, ConnPropKey.CP_CONN_PROPS, cp_conn_props)

    relay_conn_props = client_config.get_relay_conn_props()
    if relay_conn_props:
        set_scope_property(site_name, ConnPropKey.RELAY_CONN_PROPS, relay_conn_props)

    # get message auth info and put them into Databus for CellPipe to use
    auth_token = client_config.get_auth_token()
    signature = client_config.get_auth_token_signature()
    set_scope_property(scope_name=site_name, key=FLMetaKey.AUTH_TOKEN, value=auth_token)
    set_scope_property(scope_name=site_name, key=FLMetaKey.AUTH_TOKEN_SIGNATURE, value=signature)

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
    def __init__(self, config_file: str):
        super().__init__()  # Initialize memory management from base class

        self.model_registry = None
        self.logger = get_obj_logger(self)
        self.receive_called = False
        self.config_file = config_file
        self.flare_agent = None
        # Memory settings will be read from config in init()

    def _configure_subprocess_logging(self, client_config: ClientConfig) -> None:
        """Configure Python logging in the subprocess using the site's log config file.

        Uses ConfigFactory.load_config() so all supported variants (.json, .conf,
        .yml, .default) are found automatically — the hardcoded `.json` suffix is
        not assumed.  RotatingFileHandler entries in the config are downgraded to
        plain FileHandler before applying: both the CJ and the subprocess share the
        same log files, and only the CJ should trigger rotation (RotatingFileHandler
        is not process-safe).  consoleHandler output reaches stdout, where
        SubprocessLauncher routes it to the terminal or wraps it with logger.info()
        for raw print() lines from user training scripts.
        """
        try:
            task_exchange = client_config.config.get(ConfigKey.TASK_EXCHANGE, {})
            pipe_args = task_exchange.get(ConfigKey.PIPE, {}).get(ConfigKey.ARG, {})
            workspace_dir = pipe_args.get("workspace_dir", "")
            if not workspace_dir:
                return

            local_dir = os.path.join(workspace_dir, "local")
            conf = ConfigFactory.load_config(WorkspaceConstants.LOGGING_CONFIG, search_dirs=[local_dir])
            if not conf:
                return

            dict_config = conf.to_dict()
            _downgrade_rotating_handlers(dict_config)
            apply_log_config(dict_config, workspace_dir)
        except Exception as e:
            # Logging setup failure must never crash the training script.
            self.logger.warning(f"Unable to configure subprocess logging: {e}")

    def get_model_registry(self) -> ModelRegistry:
        """Gets the ModelRegistry."""
        if self.model_registry is None:
            raise RuntimeError("needs to call init method first")
        return self.model_registry

    def init(self, rank: Optional[str] = None):
        """Initializes NVFlare Client API environment.

        Args:
            rank (str): local rank of the process.
                It is only useful when the training script has multiple worker processes. (for example multi GPU)
        """

        if rank is None:
            rank = os.environ.get("RANK", "0")

        if self.model_registry:
            self.logger.warning("Warning: called init() more than once. The subsequence calls are ignored")
            return

        client_config = _create_client_config(config=self.config_file)

        # Configure logging for the subprocess using the site's log_config.json.
        # Without this the subprocess Python logging is unconfigured — logger.info()
        # is silently dropped. With it, all NVFlare loggers write to sys.stdout
        # (captured by SubprocessLauncher) and to the site's log.txt file.
        self._configure_subprocess_logging(client_config)

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
                    # Enable per-message PASS_THROUGH on the subprocess-side CellPipe
                    # (reverse path: subprocess → CJ → FL Server).  Every result message
                    # sent from the subprocess will carry MessageHeaderKey.PASS_THROUGH=True
                    # so CJ's Adapter.call() builds a per-call decode context with
                    # PASS_THROUGH=True → LazyDownloadRef at CJ.  CJ never materialises
                    # the trained tensors; the server downloads directly from the subprocess.
                    from nvflare.fuel.utils.pipe.cell_pipe import CellPipe as _CellPipe

                    if isinstance(pipe, _CellPipe):
                        pipe.pass_through_on_send = True
                        self.logger.info("PASS_THROUGH enabled on subprocess CellPipe (reverse path)")
                metric_pipe, metric_channel_name = None, ""
                if ConfigKey.METRICS_EXCHANGE in client_config.config:
                    metric_pipe, metric_channel_name = _create_pipe_using_config(
                        client_config=client_config, section=ConfigKey.METRICS_EXCHANGE
                    )
                from_nvflare_converter, to_nvflare_converter = create_default_params_converters(
                    server_expected_format=client_config.get_server_expected_format(),
                    params_exchange_format=client_config.get_exchange_format(),
                    train_task_name=client_config.get_train_task(),
                    eval_task_name=client_config.get_eval_task(),
                    submit_model_task_name=client_config.get_submit_model_task(),
                )

                flare_agent = FlareAgentWithFLModel(
                    pipe=pipe,
                    task_channel_name=task_channel_name,
                    metric_pipe=metric_pipe,
                    metric_channel_name=metric_channel_name,
                    heartbeat_timeout=client_config.get_heartbeat_timeout(),
                    submit_result_timeout=client_config.get_submit_result_timeout(),
                    max_resends=client_config.get_max_resends(),
                    download_complete_timeout=client_config.get_download_complete_timeout(),
                    launch_once=client_config.get_launch_once(),
                    from_nvflare_converter=from_nvflare_converter,
                    to_nvflare_converter=to_nvflare_converter,
                )
                flare_agent.start()

            self.model_registry = ModelRegistry(client_config, rank, flare_agent)
            self.flare_agent = flare_agent

            # Read memory management settings from config (with env var override)
            task_exchange = client_config.config.get(ConfigKey.TASK_EXCHANGE, {})
            config_gc_rounds = task_exchange.get(ConfigKey.MEMORY_GC_ROUNDS, 0)
            config_cuda_cache = task_exchange.get(ConfigKey.CUDA_EMPTY_CACHE, False)

            # Environment variables override config values.
            self._memory_gc_rounds = int(os.environ.get("NVFLARE_CLIENT_MEMORY_GC_ROUNDS", str(config_gc_rounds)))
            self._cuda_empty_cache = (
                os.environ.get("NVFLARE_CUDA_EMPTY_CACHE", str(config_cuda_cache)).lower() == "true"
            )

            if self._memory_gc_rounds > 0:
                self.logger.info(f"Memory management enabled: cleanup every {self._memory_gc_rounds} round(s)")
        except Exception as e:
            self.logger.error(f"flare.init failed: {e}")
            raise e

    def receive(self, timeout: Optional[float] = None) -> Optional[FLModel]:
        result = self.__receive()
        self.receive_called = True
        if result is not None:
            self._mem_round = result.current_round
            self._mem_site = self.get_site_name()
            log_rss(f"CA s={self._mem_site} r={result.current_round} recv")
        return result

    def __receive(self, timeout: Optional[float] = None) -> Optional[FLModel]:
        model_registry = self.get_model_registry()
        return model_registry.get_model(timeout)

    def send(self, model: FLModel, clear_cache: bool = True) -> None:
        model_registry = self.get_model_registry()
        if not self.receive_called:
            raise RuntimeError('"receive" needs to be called before sending model!')
        model_registry.submit_model(model=model)
        if clear_cache:
            # Serialization is complete. Release the sent model's params and the
            # received model's params — both are dead weight after flare.send().
            # NOTE: model.params and input_model.params will be None after this.
            model_registry.release_params(model)
            self.clear()

        self._maybe_cleanup_memory()
        log_rss(f"CA s={getattr(self, '_mem_site', '?')} r={getattr(self, '_mem_round', None)} send")

    def system_info(self) -> Dict:
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
            self.__receive()
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
        model_registry = self.get_model_registry()
        model_registry.clear()
        self.receive_called = False

    def shutdown(self):
        if self.flare_agent:
            self.flare_agent.stop()
