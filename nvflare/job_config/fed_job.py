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

from typing import Any, List

import torch.nn as nn  # TODO: How to handle pytorch dependency?

from nvflare.apis.executor import Executor
from nvflare.apis.filter import Filter
from nvflare.apis.impl.controller import Controller
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.app_common.abstract.learnable_persistor import LearnablePersistor
from nvflare.app_common.abstract.shareable_generator import ShareableGenerator
from nvflare.app_common.executors.script_executor import ScriptExecutor
from nvflare.app_common.widgets.external_configurator import ExternalConfigurator
from nvflare.app_common.widgets.intime_model_selector import IntimeModelSelector
from nvflare.app_common.widgets.metric_relay import MetricRelay
from nvflare.app_common.widgets.validation_json_generator import ValidationJsonGenerator
from nvflare.app_opt.pt import PTFileModelPersistor
from nvflare.app_opt.pt.file_model_locator import PTFileModelLocator
from nvflare.app_opt.tracking.tb.tb_receiver import TBAnalyticsReceiver
from nvflare.fuel.utils.constants import Mode
from nvflare.fuel.utils.pipe.file_pipe import FilePipe
from nvflare.job_config.fed_app_config import ClientAppConfig, FedAppConfig, ServerAppConfig
from nvflare.job_config.fed_job_config import FedJobConfig


class FilterType:
    TASK_RESULT = "_TASK_RESULT_FILTER_TYPE_"
    TASK_DATA = "_TASK_DATA_FILTER_TYPE_"


class FedApp:
    def __init__(self):
        """FedApp handles `ClientAppConfig` and `ServerAppConfig` and allows setting task result or task data filters.

        Args:
        """
        self.app = None  # Union[ClientAppConfig, ServerAppConfig]
        self._used_ids = []

    def get_app_config(self):
        return self.app

    def add_task_result_filter(self, tasks: List[str], task_filter: Filter):
        self.app.add_task_result_filter(tasks, task_filter)

    def add_task_data_filter(self, tasks: List[str], task_filter: Filter):
        self.app.add_task_data_filter(tasks, task_filter)

    def add_component(self, component, id=None):
        if id is None:
            id = "component"
        self.app.add_component(self._check_id(id), component)

    def set_persistor(self, model: nn.Module):  # TODO: support other persistors
        component = PTFileModelPersistor(model=model)
        self.app.add_component("persistor", component)

        component = PTFileModelLocator(pt_persistor_id="persistor")
        self.app.add_component("model_locator", component)

    def _check_id(self, id: str = "") -> str:
        if id not in self._used_ids:
            self._used_ids.append(id)
        else:
            cnt = 0
            _id = f"{id}_{cnt}"
            while _id in self._used_ids:
                cnt += 1
            id = f"{id}_{cnt}"
            self._used_ids.append(id)
        return id


class FedJob:
    def __init__(self, name="fed_job", min_clients=1, mandatory_clients=None, key_metric="accuracy") -> None:
        """FedJob allows users to generate job configurations in a Pythonic way.
        The `to()` routine allows users to send different components to either the server or clients.

        Args:
            job_name: the name of the NVFlare job
            min_clients: the minimum number of clients for the job
            mandatory_clients: mandatory clients to run the job (optional)
            key_metric: Metric used to determine if the model is globally best.
                if metrics are a `dict`, `key_metric` can select the metric used for global model selection.
                Defaults to "accuracy".
        """
        self.job_name = name
        self.key_metric = key_metric
        self.clients = []
        self.job: FedJobConfig = FedJobConfig(
            job_name=self.job_name, min_clients=min_clients, mandatory_clients=mandatory_clients
        )
        self._deploy_map = {}
        self._deployed = False
        self._gpus = {}

    def to(
        self, obj: Any, target: str, tasks: List[str] = None, gpu: int = None, filter_type: FilterType = None, id=None
    ):
        """assign an `obj` to a target (server or clients).
        The obj will be given a default `id` if non is provided based on its type.

        Returns:

        """
        if isinstance(obj, Controller):
            if target not in self._deploy_map:
                self._deploy_map[target] = ControllerApp(key_metric=self.key_metric)
            self._deploy_map[target].add_controller(obj, id)
        elif isinstance(obj, Executor):
            if target not in self._deploy_map:
                if isinstance(obj, ScriptExecutor):
                    external_scripts = [obj._task_script_path]
                else:
                    external_scripts = None
                self._deploy_map[target] = ExecutorApp(external_scripts=external_scripts)
                self.clients.append(target)
                if gpu is not None:
                    if target not in self._gpus:  # GPU can only be selected once per client.
                        self._gpus[target] = str(gpu)
                    else:
                        print(f"{target} already set to use GPU {self._gpus[target]}. Ignoring gpu={gpu}.")
            self._deploy_map[target].add_executor(obj, tasks=tasks)
        else:  # handle objects that are not Controller or Executor type
            if target not in self._deploy_map:
                raise ValueError(
                    f"{target} doesn't have a `Controller` or `Executor`. Deploy one first before adding components!"
                )

            if isinstance(obj, nn.Module):  # if model, set a persistor
                self._deploy_map[target].set_persistor(obj)
            elif isinstance(obj, Filter):  # handle filters
                if filter_type == FilterType.TASK_RESULT:
                    self._deploy_map[target].add_task_result_filter(tasks, obj)
                elif filter_type == FilterType.TASK_DATA:
                    self._deploy_map[target].add_task_data_filter(tasks, obj)
                else:
                    raise ValueError(
                        f"Provided a filter for {target} without specifying valid `filter_type`. Select from `FilterType.TASK_RESULT` or `FilterType.TASK_DATA`."
                    )
            else:  # handle other types
                if id is None:  # handle built-in types and set ids
                    if isinstance(obj, Aggregator):
                        id = "aggregator"
                    elif isinstance(obj, LearnablePersistor):
                        id = "persistor"
                    elif isinstance(obj, ShareableGenerator):
                        id = "shareable_generator"
                self._deploy_map[target].add_component(obj, id)

    def _deploy(self, app: FedApp, target: str):
        if not isinstance(app, FedApp):
            raise ValueError(f"App needs to be of type `FedApp` but was type {type(app)}")

        client_server_config = app.get_app_config()
        if isinstance(client_server_config, ClientAppConfig):
            app_config = FedAppConfig(server_app=None, client_app=client_server_config)
            app_name = f"app_{target}"
        elif isinstance(client_server_config, ServerAppConfig):
            app_config = FedAppConfig(server_app=client_server_config, client_app=None)
            app_name = "app_server"
        else:
            raise ValueError(
                f"App needs to be of type `ClientAppConfig` or `ServerAppConfig` but was type {type(client_server_config)}"
            )

        self.job.add_fed_app(app_name, app_config)
        self.job.set_site_app(target, app_name)

    def _run_deploy(self):
        if not self._deployed:
            for target in self._deploy_map:
                self._deploy(self._deploy_map[target], target)

            self._deployed = True

    def export_job(self, job_root):
        self._run_deploy()
        self.job.generate_job_config(job_root)

    def simulator_run(self, workspace, threads: int = None):
        self._run_deploy()

        n_clients = len(self.clients)
        if threads is None:
            threads = n_clients

        self.job.simulator_run(
            workspace,
            clients=",".join(self.clients),
            n_clients=n_clients,
            threads=threads,
            gpu=",".join([self._gpus[client] for client in self._gpus.keys()]),
        )


class ExecutorApp(FedApp):
    def __init__(self, external_scripts: List = None):
        """Wrapper around `ClientAppConfig`.

        Args:
            external_scripts: List of external scripts that need to be deployed to the client. Defaults to None.
        """
        super().__init__()
        self.external_scripts = external_scripts
        self._create_client_app()

    def add_executor(self, executor, tasks=None):
        if tasks is None:
            tasks = ["*"]  # Add executor for any task by default
        self.app.add_executor(tasks, executor)

    def _create_client_app(self):
        self.app = ClientAppConfig()

        component = FilePipe(  # TODO: support CellPipe, causes type error for passing secure_mode = "{SECURE_MODE}"
            mode=Mode.PASSIVE,
            root_path="{WORKSPACE}/{JOB_ID}/{SITE_NAME}",
        )
        self.app.add_component("metrics_pipe", component)

        component = MetricRelay(pipe_id="metrics_pipe", event_type="fed.analytix_log_stats", read_interval=0.1)
        self.app.add_component("metric_relay", component)

        component = ExternalConfigurator(component_ids=["metric_relay"])
        self.app.add_component("config_preparer", component)

        if self.external_scripts is not None:
            for _script in self.external_scripts:
                self.app.add_ext_script(_script)


class ControllerApp(FedApp):
    """Wrapper around `ServerAppConfig`.

    Args:
    """

    def __init__(self, key_metric="accuracy"):
        super().__init__()
        self.key_metric = key_metric
        self._create_server_app()

    def add_controller(self, controller, id=None):
        if id is None:
            id = "controller"
        self.app.add_workflow(self._check_id(id), controller)

    def _create_server_app(self):
        self.app: ServerAppConfig = ServerAppConfig()

        component = ValidationJsonGenerator()
        self.app.add_component("json_generator", component)

        if self.key_metric:
            component = IntimeModelSelector(key_metric=self.key_metric)
            self.app.add_component("model_selector", component)

        # TODO: make different tracking receivers configurable
        component = TBAnalyticsReceiver(events=["fed.analytix_log_stats"])
        self.app.add_component("receiver", component)
