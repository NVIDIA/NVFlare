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

import re
import uuid
from typing import Any, List, Union

from nvflare.apis.executor import Executor
from nvflare.apis.filter import Filter
from nvflare.apis.impl.controller import Controller
from nvflare.apis.job_def import ALL_SITES, SERVER_SITE_NAME
from nvflare.app_common.executors.script_executor import ScriptExecutor
from nvflare.app_common.widgets.convert_to_fed_event import ConvertToFedEvent
from nvflare.app_common.widgets.intime_model_selector import IntimeModelSelector
from nvflare.app_common.widgets.validation_json_generator import ValidationJsonGenerator
from nvflare.fuel.utils.class_utils import get_component_init_parameters
from nvflare.fuel.utils.import_utils import optional_import
from nvflare.fuel.utils.validation_utils import check_positive_int
from nvflare.job_config.fed_app_config import ClientAppConfig, FedAppConfig, ServerAppConfig
from nvflare.job_config.fed_job_config import FedJobConfig

torch, torch_ok = optional_import(module="torch")
if torch_ok:
    import torch.nn as nn

    from nvflare.app_opt.pt import PTFileModelPersistor
    from nvflare.app_opt.pt.file_model_locator import PTFileModelLocator

tf, tf_ok = optional_import(module="tensorflow")
if tf_ok:
    from nvflare.app_opt.tf.model_persistor import TFModelPersistor

tb, tb_ok = optional_import(module="tensorboard")
if torch_ok and tb_ok:
    from nvflare.app_opt.tracking.tb.tb_receiver import TBAnalyticsReceiver

SPECIAL_CHARACTERS = '"!@#$%^&*()+?=,<>/'


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
        self.app.add_component(self._gen_tracked_id(id), component)

    def _generate_id(self, id: str = "") -> str:
        if id not in self._used_ids:
            return id
        else:
            while id in self._used_ids:
                # increase integer counts in id
                cnt = re.search(r"\d+", id)
                if cnt:
                    cnt = cnt.group()
                    id = id.replace(cnt, str(int(cnt) + 1))
                else:
                    id = id + "1"
        return id

    def _gen_tracked_id(self, id: str = "") -> str:
        id = self._generate_id(id)
        self._used_ids.append(id)
        return id

    def add_external_scripts(self, external_scripts: List):
        """Register external scripts to include them in custom directory.

        Args:
            external_scripts: List of external scripts that need to be deployed to the client/server.
        """
        for _script in external_scripts:
            self.app.add_ext_script(_script)


class ExecutorApp(FedApp):
    def __init__(self):
        """Wrapper around `ClientAppConfig`."""
        super().__init__()
        self._create_client_app()

    def add_executor(self, executor, tasks=None):
        if tasks is None:
            tasks = ["*"]  # Add executor for any task by default
        self.app.add_executor(tasks, executor)

    def _create_client_app(self):
        self.app = ClientAppConfig()

        component = ConvertToFedEvent(events_to_convert=["analytix_log_stats"], fed_event_prefix="fed.")
        self.app.add_component("event_to_fed", component)


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
        self.app.add_workflow(self._gen_tracked_id(id), controller)

    def _create_server_app(self):
        self.app: ServerAppConfig = ServerAppConfig()

        component = ValidationJsonGenerator()
        self.app.add_component("json_generator", component)

        if self.key_metric:
            component = IntimeModelSelector(key_metric=self.key_metric)
            self.app.add_component("model_selector", component)

        # TODO: make different tracking receivers configurable
        if torch_ok and tb_ok:
            component = TBAnalyticsReceiver(events=["fed.analytix_log_stats"])
            self.app.add_component("receiver", component)


class FedJob:
    def __init__(self, name="fed_job", min_clients=1, mandatory_clients=None, key_metric="accuracy") -> None:
        """FedJob allows users to generate job configurations in a Pythonic way.
        The `to()` routine allows users to send different components to either the server or clients.

        Args:
            name: the name of the NVFlare job
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
        self._components = {}

    def to(
        self,
        obj: Any,
        target: str,
        tasks: List[str] = None,
        gpu: Union[int, List[int]] = None,
        filter_type: FilterType = None,
        id=None,
    ):
        """assign an object to a target (server or clients).

        Args:
            obj: The object to be assigned. The obj will be given a default `id` if non is provided based on its type.
            target: The target location of th object. Can be "server" or a client name, e.g. "site-1".
            tasks: In case object is an `Executor`, optional list of tasks the executor should handle.
                Defaults to `None`. If `None`, all tasks will be handled using `[*]`.
            gpu: GPU index or list of GPU indices used for simulating the run on that target.
            filter_type: The type of filter used. Either `FilterType.TASK_RESULT` or `FilterType.TASK_DATA`.
            id: Optional user-defined id for the object. Defaults to `None` and ID will automatically be assigned.

        Returns:

        """
        self._validate_target(target)

        if isinstance(obj, Controller):
            if target not in self._deploy_map:
                self._deploy_map[target] = ControllerApp(key_metric=self.key_metric)
            self._deploy_map[target].add_controller(obj, id)
        elif isinstance(obj, Executor):
            if target not in self._deploy_map:
                self._deploy_map[target] = ExecutorApp()
            if isinstance(obj, ScriptExecutor):
                external_scripts = [obj._task_script_path]
                self._deploy_map[target].add_external_scripts(external_scripts)
            if target not in self.clients:
                self.clients.append(target)
            if gpu is not None:
                if target not in self._gpus:  # GPU can only be selected once per client.
                    self._gpus[target] = str(gpu)
                else:
                    print(f"{target} already set to use GPU {self._gpus[target]}. Ignoring gpu={gpu}.")
            self._deploy_map[target].add_executor(obj, tasks=tasks)
        elif isinstance(obj, str):  # treat the str type object as external script
            if target not in self._deploy_map:
                raise ValueError(
                    f"{target} doesn't have a `Controller` or `Executor`. Deploy one first before adding external script!"
                )

            self._deploy_map[target].add_external_scripts([obj])
        else:  # handle objects that are not Controller or Executor type
            if target not in self._deploy_map:
                raise ValueError(
                    f"{target} doesn't have a `Controller` or `Executor`. Deploy one first before adding components!"
                )

            if isinstance(obj, Filter):  # handle filters
                if filter_type == FilterType.TASK_RESULT:
                    self._deploy_map[target].add_task_result_filter(tasks, obj)
                elif filter_type == FilterType.TASK_DATA:
                    self._deploy_map[target].add_task_data_filter(tasks, obj)
                else:
                    raise ValueError(
                        f"Provided a filter for {target} without specifying a valid `filter_type`. "
                        f"Select from `FilterType.TASK_RESULT` or `FilterType.TASK_DATA`."
                    )
            # else assume a model is being set
            else:  # TODO: handle other persistors
                added_model = False
                # Check different models framework types and add corresponding persistor
                if torch_ok:
                    if isinstance(obj, nn.Module):  # if model, create a PT persistor
                        component = PTFileModelPersistor(model=obj)
                        self._deploy_map[target].app.add_component("persistor", component)

                        component = PTFileModelLocator(pt_persistor_id="persistor")
                        self._deploy_map[target].app.add_component("model_locator", component)
                        added_model = True
                elif tf_ok:
                    if isinstance(obj, tf.keras.Model):  # if model, create a TF persistor
                        component = TFModelPersistor(model=obj)
                        self._deploy_map[target].app.add_component("persistor", component)
                        added_model = True

                if not added_model:  # if it wasn't a model, add as component
                    self._deploy_map[target].add_component(obj, id)

        # add any other components the object might have referenced via id
        if self._components:
            self._add_referenced_components(obj, target)

    def to_server(
        self,
        obj: Any,
        filter_type: FilterType = None,
        id=None,
    ):
        """assign an object to the server.

        Args:
            obj: The object to be assigned. The obj will be given a default `id` if non is provided based on its type.
            filter_type: The type of filter used. Either `FilterType.TASK_RESULT` or `FilterType.TASK_DATA`.
            id: Optional user-defined id for the object. Defaults to `None` and ID will automatically be assigned.

        Returns:

        """
        if isinstance(obj, Executor):
            raise ValueError("Use `job.to(executor, <client_name>)` or `job.to_clients(executor)` for Executors.")

        self.to(obj=obj, target=SERVER_SITE_NAME, filter_type=filter_type, id=id)

    def to_clients(
        self,
        obj: Any,
        tasks: List[str] = None,
        filter_type: FilterType = None,
        id=None,
    ):
        """assign an object to all clients.

        Args:
            obj (Any): Object to be deployed.
            tasks: In case object is an `Executor`, optional list of tasks the executor should handle.
                Defaults to `None`. If `None`, all tasks will be handled using `[*]`.
            filter_type: The type of filter used. Either `FilterType.TASK_RESULT` or `FilterType.TASK_DATA`.
            id: Optional user-defined id for the object. Defaults to `None` and ID will automatically be assigned.

        Returns:

        """
        if isinstance(obj, Controller):
            raise ValueError('Use `job.to(controller, "server")` or `job.to_server(controller)` for Controllers.')

        self.to(obj=obj, target=ALL_SITES, tasks=tasks, filter_type=filter_type, id=id)

    def as_id(self, obj: Any):
        id = str(uuid.uuid4())
        self._components[id] = obj
        return id

    def _add_referenced_components(self, base_component, target):
        """Adds any other components the object might have referenced via id"""
        # Check all arguments for ids referenced with .as_id()
        if hasattr(base_component, "__dict__"):
            parameters = get_component_init_parameters(base_component)
            attrs = base_component.__dict__
            for param in parameters:
                attr_key = param if param in attrs.keys() else "_" + param
                if attr_key in attrs.keys():
                    base_id = attrs[attr_key]
                    if isinstance(base_id, str):  # could be id
                        if base_id in self._components:
                            self._deploy_map[target].add_component(self._components[base_id], base_id)
                            # add any components referenced by this component
                            self._add_referenced_components(self._components[base_id], target)
                            # remove already added components from tracked components
                            self._components.pop(base_id)

    def _set_site_app(self, app: FedApp, target: str):
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

    def _set_all_app(self, client_app: ExecutorApp, server_app: ControllerApp):
        if not isinstance(client_app, ExecutorApp):
            raise ValueError(f"`client_app` needs to be of type `ExecutorApp` but was type {type(client_app)}")
        if not isinstance(server_app, ControllerApp):
            raise ValueError(f"`server_app` needs to be of type `ControllerApp` but was type {type(server_app)}")

        client_config = client_app.get_app_config()
        server_config = server_app.get_app_config()

        app_config = FedAppConfig(server_app=server_config, client_app=client_config)
        app_name = "app"

        self.job.add_fed_app(app_name, app_config)
        self.job.set_site_app(ALL_SITES, app_name)

    def _set_all_apps(self):
        if not self._deployed:
            if ALL_SITES in self._deploy_map:
                if SERVER_SITE_NAME not in self._deploy_map:
                    raise ValueError('Missing server components! Deploy using `to(obj, "server") or `to_server(obj)`')
                self._set_all_app(client_app=self._deploy_map[ALL_SITES], server_app=self._deploy_map[SERVER_SITE_NAME])
            else:
                for target in self._deploy_map:
                    self._set_site_app(self._deploy_map[target], target)

            self._deployed = True

    def export_job(self, job_root):
        self._set_all_apps()
        self.job.generate_job_config(job_root)

    def simulator_run(self, workspace, n_clients: int = None, threads: int = None):
        self._set_all_apps()

        if ALL_SITES in self.clients and not n_clients:
            raise ValueError("Clients were not specified using to(). Please provide the number of clients to simulate.")
        elif ALL_SITES in self.clients and n_clients:
            check_positive_int("n_clients", n_clients)
            self.clients = [f"site-{i}" for i in range(1, n_clients + 1)]
        elif self.clients and n_clients:
            raise ValueError("You already specified clients using `to()`. Don't use `n_clients` in simulator_run.")

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

    def _validate_target(self, target):
        if not target:
            raise ValueError("Must provide a valid target name")

        if any(c in SPECIAL_CHARACTERS for c in target) and target != ALL_SITES:
            raise ValueError(f"target {target} name contains invalid character")
        pass
