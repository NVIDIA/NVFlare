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
import os.path
import re
import uuid
from typing import Any, List, Optional

from nvflare.apis.executor import Executor
from nvflare.apis.filter import Filter
from nvflare.apis.impl.controller import Controller
from nvflare.apis.job_def import ALL_SITES, SERVER_SITE_NAME
from nvflare.fuel.utils.class_utils import get_component_init_parameters
from nvflare.fuel.utils.validation_utils import check_positive_int
from nvflare.job_config.fed_app_config import ClientAppConfig, FedAppConfig, ServerAppConfig
from nvflare.job_config.fed_job_config import FedJobConfig

from .defs import FilterType, JobTargetType

SPECIAL_CHARACTERS = '"!@#$%^&*()+?=,<>/'

_ADD_TO_JOB_METHOD_NAME = "add_to_fed_job"


class FedApp:
    def __init__(self):
        """FedApp handles `ClientAppConfig` and `ServerAppConfig` and allows setting task result or task data filters."""
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
        final_id = self.generate_tracked_id(id)
        self.app.add_component(final_id, component)
        return final_id

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

    def generate_tracked_id(self, id: str = "") -> str:
        id = self._generate_id(id)
        self._used_ids.append(id)
        return id

    def add_external_script(self, ext_script: str):
        """Register external script to include them in custom directory.

        Args:
            ext_script: List of external scripts that need to be deployed to the client/server.
        """
        self.app.add_ext_script(ext_script)

    def add_external_dir(self, ext_dir: str):
        """Register external folder to include them in custom directory.

        Args:
            ext_dir: external folder that need to be deployed to the client/server.
        """
        self.app.add_ext_dir(ext_dir)

    def _add_resource(self, resource: str):
        if not isinstance(resource, str):
            raise ValueError(f"cannot add resource: resource must be a str but got {type(resource)}")
        elif os.path.isdir(resource):
            self.add_external_dir(resource)
        elif os.path.isfile(resource):
            self.add_external_script(resource)
        else:
            raise ValueError(f"cannot add resource: invalid resource {resource}: it must be either a directory or file")

    def add_resources(self, resources: List[str]):
        """Add resources to the job. To be used by job component programmer.

        Args:
            resources:

        Returns:

        """
        for r in resources:
            self._add_resource(r)


class JobCtx:
    def __init__(self, obj: Any, target: str, comp_id: str):
        self.obj = obj
        self.target = target
        self.comp_id = comp_id


class ClientApp(FedApp):
    def __init__(self):
        """Wrapper around `ClientAppConfig`."""
        super().__init__()
        self.app = ClientAppConfig()

    def add_executor(self, executor: Executor, tasks=None):
        if not tasks:
            tasks = ["*"]  # Add executor for any task by default
        self.app.add_executor(tasks, executor)


class ServerApp(FedApp):
    """Wrapper around `ServerAppConfig`."""

    def __init__(self):
        super().__init__()
        self.app: ServerAppConfig = ServerAppConfig()

    def add_controller(self, controller: Controller, id=None):
        if not id:
            id = "controller"
        self.app.add_workflow(self.generate_tracked_id(id), controller)


class FedJob:
    def __init__(
        self,
        name: str = "fed_job",
        min_clients: int = 1,
        mandatory_clients: Optional[List[str]] = None,
    ) -> None:
        """FedJob allows users to generate job configurations in a Pythonic way.
        The `to()` routine allows users to send different components to either the server or clients.

        Args:
            name: the name of the NVFlare job
            min_clients: the minimum number of clients for the job
            mandatory_clients: mandatory clients to run the job (optional)

        """
        self.name = name
        self.clients = []
        self.job: FedJobConfig = FedJobConfig(
            job_name=self.name, min_clients=min_clients, mandatory_clients=mandatory_clients
        )
        self._deploy_map = {}
        self._deployed = False
        self._components = {}

    def set_up_client(self, target: str):
        """Setup routine called by FedJob when first sending object to a client target.

        Args:
            target: the target to perform setup.

        Returns:

        """
        pass

    def _add_server_app(self, obj: ServerApp, target: str):
        self._deploy_map[target] = obj

    def _add_client_app(self, obj: ClientApp, target: str):
        self._deploy_map[target] = obj
        if target not in self.clients:
            self.clients.append(target)

    def to(
        self,
        obj: Any,
        target: str,
        id=None,
        **kwargs,
    ) -> Any:
        """Assign an object to the target. For end users.

        Args:
            obj: the object to be assigned
            target: the target that the object is assigned to
            id: the id of the object
            **kwargs: additional args to be passed to the object's add_to_fed_job method.

        If the obj provides the add_to_fed_job method, it will be called with the kwargs.
        This method must follow this signature:

            add_to_fed_job(job, ctx, ...)

            job: this is the job (self)
            ctx: this is the JobCtx that keeps contextual info of this call.

        The add_to_fed_job function is usually implemented in FL component classes.
        When implementing this function, you should not use anything in the ctx; instead, you should use
        the "add_xxx" methods of the "job" object: add_component, add_resources, add_filter, add_executor, etc.

        Returns:
            result of add_to_job_method if called, or id of added component

        """
        if not obj:
            raise ValueError("cannot add empty object to job")

        if isinstance(obj, (ClientApp, ServerApp)):
            raise ValueError("adding (ClientApp, ServerApp) is not allowed")

        self._validate_target(target)

        target_type = JobTargetType.get_target_type(target)
        app = self._deploy_map.get(target)
        if not app:
            if target_type == JobTargetType.SERVER:
                app = ServerApp()
                self._add_server_app(app, target)
            else:
                app = ClientApp()
                self._add_client_app(app, target)
                self.set_up_client(target)

        if isinstance(obj, str):  # treat the str type object as external script
            if os.path.isdir(obj):
                app.add_external_dir(obj)
            else:
                app.add_external_script(obj)
            return None

        get_target_type_method = getattr(obj, "get_job_target_type", None)
        if get_target_type_method is not None:
            expected_target_type = get_target_type_method()
            if expected_target_type != target_type:
                if target_type == JobTargetType.SERVER:
                    raise ValueError(f"this object can only be assigned to server, but tried to assign to {target}")
                else:
                    raise ValueError(f"this object can only be assigned to client, but tried to assign to {target}")

        add_to_job_method = getattr(obj, _ADD_TO_JOB_METHOD_NAME, None)
        if add_to_job_method is not None:
            ctx = JobCtx(obj, target, id)
            result = add_to_job_method(self, ctx, **kwargs)
        else:
            # basic object
            result = app.add_component(obj, id)

        # add any other components the object might have referenced via id
        if self._components:
            self._add_referenced_components(obj, target)
        return result

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

    def _get_app(self, ctx: JobCtx):
        app = self._deploy_map.get(ctx.target)
        if not app:
            target_type = JobTargetType.get_target_type(ctx.target)
            if target_type == JobTargetType.CLIENT:
                app_type = "a ClientApp"
            else:
                app_type = "a ServerApp"
            raise RuntimeError(f"No app found for target '{ctx.target}' - missing {app_type}")
        return app

    def add_component(self, comp_id: str, obj: Any, ctx: JobCtx):
        """Add a component to the job. To be used by job component programmer.

        Args:
            comp_id: component id
            obj: component to be added to job.
            ctx: JobCtx for contextual information.

        Returns:
            final id assigned to component.

        """
        app = self._get_app(ctx)
        if not comp_id:
            comp_id = ctx.comp_id
        final_id = app.add_component(obj, comp_id)
        if self._components:
            self._add_referenced_components(obj, ctx.target)
        return final_id

    def add_controller(self, obj: Controller, ctx: JobCtx):
        """Add a Controller object to the job. To be used by controller programmer.

        Args:
            obj: Controller to be added to job.
            ctx: JobCtx for contextual information.

        Returns:

        """
        target_type = JobTargetType.get_target_type(ctx.target)
        app = self._get_app(ctx)
        if target_type != JobTargetType.SERVER:  # add client-side controllers as components
            app.add_component(obj, ctx.comp_id)
        else:
            app.add_controller(obj, ctx.comp_id)

    def add_executor(self, obj: Executor, tasks: List[str], ctx: JobCtx):
        """Add an executor to the job. To be used by executor programmer.

        Args:
            obj: Executor to be added to job.
            tasks: List of tasks that should be handled. If `None`, all tasks will be handled using `[*]`.
            ctx: JobCtx for contextual information.

        Returns:

        """
        app = self._get_app(ctx)
        app.add_executor(obj, tasks=tasks)

    def add_filter(self, obj: Filter, filter_type: str, tasks, ctx: JobCtx):
        """Add a filter to the job. To be used by filter programmer.

        Args:
            obj: Filter to be added to job.
            filter_type: The type of filter used. Either `FilterType.TASK_RESULT` or `FilterType.TASK_DATA`.
            tasks: List of tasks that Filter applies to.
            ctx: JobCtx for contextual information.

        Returns:

        """
        app = self._get_app(ctx)
        if filter_type == FilterType.TASK_RESULT:
            app.add_task_result_filter(tasks, obj)
        elif filter_type == FilterType.TASK_DATA:
            app.add_task_data_filter(tasks, obj)
        else:
            raise ValueError(
                f"Provided a filter for {ctx.target} without specifying a valid `filter_type`. "
                f"Select from `FilterType.TASK_RESULT` or `FilterType.TASK_DATA`."
            )

    def add_resources(self, resources: List[str], ctx: JobCtx):
        """Add resources to the job. To be used by job component programmer.

        Args:
            resources: List of filenames or directories to be added to job.
            ctx: JobCtx for contextual information.

        Returns:

        """
        app = self._get_app(ctx)
        app.add_resources(resources)

    def to_server(
        self,
        obj: Any,
        id=None,
        **kwargs,
    ):
        """assign an object to the server. For end users.

        Args:
            obj: The object to be assigned. The obj will be given a default `id` if none is provided based on its type.
            id: Optional user-defined id for the object. Defaults to `None` and ID will automatically be assigned.
            **kwargs: additional args to be passed to the object's add_to_fed_job method.

        Returns:
            result of add_to_job_method if called, or id of added component

        """
        if isinstance(obj, Executor):
            raise ValueError("Use `job.to(executor, <client_name>)` or `job.to_clients(executor)` for Executors.")

        return self.to(obj=obj, target=SERVER_SITE_NAME, id=id, **kwargs)

    def to_clients(
        self,
        obj: Any,
        id=None,
        **kwargs,
    ):
        """assign an object to all clients. For end users.

        Args:
            obj (Any): Object to be deployed.
            id: Optional user-defined id for the object. Defaults to `None` and ID will automatically be assigned.
            **kwargs: additional args to be passed to the object's add_to_fed_job method.

        Returns:
            result of add_to_job_method if called, or id of added component

        """
        if isinstance(obj, Controller):
            raise ValueError('Use `job.to(controller, "server")` or `job.to_server(controller)` for Controllers.')

        return self.to(obj=obj, target=ALL_SITES, id=id, **kwargs)

    def _validate_target(self, target):
        if not target:
            raise ValueError("Must provide a valid target name")

        if any(c in SPECIAL_CHARACTERS for c in target) and target != ALL_SITES:
            raise ValueError(f"target {target} name contains invalid character")

    def _set_all_app(self, client_app: ClientApp, server_app: ServerApp):
        if not isinstance(client_app, ClientApp):
            raise ValueError(f"`client_app` needs to be of type `ClientApp` but was type {type(client_app)}")
        if not isinstance(server_app, ServerApp):
            raise ValueError(f"`server_app` needs to be of type `ServerApp` but was type {type(server_app)}")

        client_config = client_app.get_app_config()
        server_config = server_app.get_app_config()

        app_config = FedAppConfig(server_app=server_config, client_app=client_config)
        app_name = "app"

        self.job.add_fed_app(app_name, app_config)
        self.job.set_site_app(ALL_SITES, app_name)

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

    def export_job(self, job_root: str):
        """Export job config to `job_root` directory with name `self.job_name`.
        For end users.

        Args:
            job_root: directory to export job configuration.

        Returns:

        """
        self._set_all_apps()
        self.job.generate_job_config(job_root)

    def simulator_run(self, workspace: str, n_clients: int = None, threads: int = None, gpu: str = None):
        """Run the job with the simulator with the `workspace` using `clients` and `threads`.
        For end users.

        Args:
            workspace: workspace directory for job.
            n_clients: number of clients.
            threads: number of threads.
            gpu: gpu assignments for simulating clients, comma separated

        Returns:

        """
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
            gpu=gpu,
        )

    def as_id(self, obj: Any) -> str:
        """Generate and return uuid for `obj`. For end users.
        If this id is referenced by another added object, this `obj` will also be added as a component.
        """
        cid = str(uuid.uuid4())
        self._components[cid] = obj
        return cid

    @staticmethod
    def check_kwargs(args_to_check: dict, args_expected: dict):
        """Check kwargs for arguments. Raise Error if required arg is missing, or unexpected arg is given.

        Args:
            args_to_check (dict): kwargs dictionary to check.
            args_expected (dict): dictionary of argument name to boolean of whether argument is required (True) or optional (False).

        """
        if not args_expected and not args_to_check:
            return

        if args_to_check and not args_expected:
            raise ValueError(f"received args {list(args_to_check.keys())}, but no args expected")

        args_info = {}
        for k, required in args_expected.items():
            args_info[k] = "required" if required else "optional"

        # see whether required args are present
        for k, required in args_expected.items():
            if required and (not args_to_check or k not in args_to_check):
                raise ValueError(f"Missing required arg '{k}'. " f"Supported args: {args_info}")

        # see whether we got unexpected args
        if args_to_check:
            for k in args_to_check.keys():
                if k not in args_expected:
                    raise ValueError(f"Received unexpected arg '{k}'. " f"Supported args: {args_info}")


def has_add_to_job_method(obj: Any) -> bool:
    add_to_job_method = getattr(obj, _ADD_TO_JOB_METHOD_NAME, None)
    return add_to_job_method is not None and callable(add_to_job_method)
