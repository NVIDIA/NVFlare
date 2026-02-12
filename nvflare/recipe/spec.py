# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

from nvflare.apis.filter import Filter
from nvflare.app_common.widgets.decomposer_reg import DecomposerRegister
from nvflare.fuel.utils.fobs import Decomposer
from nvflare.job_config.api import FedJob
from nvflare.job_config.defs import FilterType


class ExecEnv(ABC):

    def __init__(self, extra: dict = None):
        """Constructor of ExecEnv

        Args:
            extra: a dict of extra properties
        """
        if not extra:
            extra = {}
        if not isinstance(extra, dict):
            raise ValueError(f"extra must be dict but got {type(extra)}")
        self.extra = extra

    def get_extra_prop(self, prop_name: str, default=None):
        """Get the specified extra property.

        Args:
            prop_name: name of the property
            default: the default value to return if the named property does not exist.

        Returns: value of the property or the default

        """
        return self.extra.get(prop_name, default)

    @abstractmethod
    def deploy(self, job: FedJob) -> str:
        """Deploy a FedJob and return an execution response.

        Args:
            job: The FedJob to deploy.

        Returns:
            str: The job ID.
        """
        pass

    @abstractmethod
    def get_job_status(self, job_id: str) -> Optional[str]:
        """Get the status of a job.

        Args:
            job_id: The job ID to check status for.

        Returns:
            Optional[str]: The status of the job, or None if not supported.
        """
        pass

    @abstractmethod
    def abort_job(self, job_id: str) -> None:
        """Abort a running job.

        Args:
            job_id: The job ID to abort.
        """
        pass

    @abstractmethod
    def get_job_result(self, job_id: str, timeout: float = 0.0) -> Optional[str]:
        """Get the result workspace of a job.

        Args:
            job_id: The job ID to get results for.
            timeout: The timeout for the job to complete. Defaults to 0.0 (no timeout).

        Returns:
            Optional[str]: The result workspace path if job completed, None if still running or stopped early.
        """
        pass

    def stop(self, clean_up: bool = False) -> None:
        """Stop the execution environment and optionally clean up resources.

        This method is called after job execution to ensure proper cleanup.
        Default implementation is a no-op. Override in subclasses that need cleanup.

        Args:
            clean_up: If True, remove workspace and temporary files after stopping.
                      If False, only stop running processes but preserve workspace.
                      Defaults to False.
        """
        pass


class Recipe(ABC):

    def __init__(self, job: FedJob):
        """This is base class of a recipe. Recipes are implemented by jobs.
        A concrete recipe must provide the job for recipe implementation.

        Args:
            job: the job that implements the recipe.
        """
        self.job = job

    def process_env(self, env: ExecEnv):
        """Process environment-specific configuration.

        Subclasses can override to add environment-specific processing.
        Script validation is handled by each ExecEnv subclass in deploy().
        """
        pass

    def _get_existing_client_sites(self) -> List[str]:
        """Get list of existing per-site client apps (excluding ALL_SITES and server).

        This helper method checks if there are already per-site client configurations
        in the deploy map. If so, new client-side objects should be added to each
        specific site to preserve the per-site structure and avoid creating a shared
        ALL_SITES app that would override per-site configurations.

        Note: This method uses the private attribute job._deploy_map because FedJob
        has no public API to enumerate per-site deploy targets; we need it to decide
        whether to add client-side objects per site or to a shared ALL_SITES app.

        Returns:
            List of existing client site names, or empty list if none exist (or
            _deploy_map is unavailable).
        """
        from nvflare.apis.job_def import ALL_SITES, SERVER_SITE_NAME
        from nvflare.job_config.defs import JobTargetType

        deploy_map = getattr(self.job, "_deploy_map", {})
        return [
            target
            for target in deploy_map.keys()
            if target not in [ALL_SITES, SERVER_SITE_NAME]
            and JobTargetType.get_target_type(target) == JobTargetType.CLIENT
        ]

    def _add_to_client_apps(self, obj, clients: Optional[List[str]] = None, **kwargs):
        """Add an object to client apps, preserving existing per-site structure.

        Args:
            obj: Object to add to clients.
            clients: Optional list of specific client names. If None, applies to all clients.
            **kwargs: Extra options forwarded to `job.to()`/`job.to_clients()`.
        """
        if clients is None:
            existing_client_sites = self._get_existing_client_sites()
            if existing_client_sites:
                for site in existing_client_sites:
                    self.job.to(obj, site, **kwargs)
            else:
                self.job.to_clients(obj, **kwargs)
        else:
            for client in clients:
                self.job.to(obj, client, **kwargs)

    def add_client_input_filter(
        self, filter: Filter, tasks: Optional[List[str]] = None, clients: Optional[List[str]] = None
    ):
        """Add a filter to clients for incoming tasks from the server.

        Args:
            filter: the filter to be added
            tasks: tasks that the filter applies to
            clients: client names to add, if None, all clients will be added.

        Returns: None

        """
        self._add_to_client_apps(filter, clients=clients, filter_type=FilterType.TASK_DATA, tasks=tasks)

    def add_client_output_filter(
        self, filter: Filter, tasks: Optional[List[str]] = None, clients: Optional[List[str]] = None
    ):
        """Add a filter to clients for outgoing result to server.

        Args:
            filter: the filter to be added
            tasks: tasks that the filter applies to
            clients: client names to add, if None, all clients will be added.

        Returns: None

        """
        self._add_to_client_apps(filter, clients=clients, filter_type=FilterType.TASK_RESULT, tasks=tasks)

    def add_client_config(self, config: Dict, clients: Optional[List[str]] = None):
        """Add top-level configuration parameters to config_fed_client.json.

        Args:
            config: Dictionary of configuration parameters to add.
            clients: Optional list of specific client names. If None, applies to all clients.

        Raises:
            TypeError: If config is not a dictionary.
        """
        if not isinstance(config, dict):
            raise TypeError(f"config must be a dict, got {type(config).__name__}")

        self._add_to_client_apps(config, clients=clients)

    def add_client_file(self, file_path: str, clients: Optional[List[str]] = None):
        """Add a file or directory to client apps.

        The file will be added to the client's custom directory and bundled with the job.
        Can be a script, configuration file, or any resource needed by clients.

        Args:
            file_path: Path to the file or directory to add to clients.
            clients: Optional list of specific client names. If None, applies to all clients.

        Raises:
            TypeError: If file_path is not a string.

        Example:
            # Add a wrapper script to all clients
            recipe.add_client_file("client_wrapper.sh")

            # Add a script to specific clients
            recipe.add_client_file("custom_script.py", clients=["site1", "site2"])
        """
        if not isinstance(file_path, str):
            raise TypeError(f"file_path must be a str, got {type(file_path).__name__}")

        self._add_to_client_apps(file_path, clients=clients)

    def add_server_output_filter(self, filter: Filter, tasks: Optional[List[str]] = None):
        """Add a filter to the server for outgoing tasks to clients.

        Args:
            filter: the filter to be added
            tasks: tasks that the filter applies to

        Returns: None

        """
        self.job.to_server(filter, filter_type=FilterType.TASK_DATA, tasks=tasks)

    def add_server_input_filter(self, filter: Filter, tasks: Optional[List[str]] = None):
        """Add a filter to server for incoming task result from clients. .

        Args:
            filter: the filter to be added
            tasks: tasks that the filter applies to

        Returns: None

        """
        self.job.to_server(filter, filter_type=FilterType.TASK_RESULT, tasks=tasks)

    def add_server_config(self, config: Dict):
        """Add top-level configuration parameters to config_fed_server.json.

        Args:
            config: Dictionary of configuration parameters to add.

        Raises:
            TypeError: If config is not a dictionary.
        """
        if not isinstance(config, dict):
            raise TypeError(f"config must be a dict, got {type(config).__name__}")

        self.job.to_server(config)

    def add_server_file(self, file_path: str):
        """Add a file or directory to server app.

        The file will be added to the server's custom directory and bundled with the job.
        Can be a script, configuration file, or any resource needed by the server.

        Args:
            file_path: Path to the file or directory to add to server.

        Raises:
            TypeError: If file_path is not a string.

        Example:
            # Add a wrapper script to server
            recipe.add_server_file("server_wrapper.sh")
        """
        if not isinstance(file_path, str):
            raise TypeError(f"file_path must be a str, got {type(file_path).__name__}")

        self.job.to_server(file_path)

    @staticmethod
    def _get_full_class_name(obj):
        """
        Returns the fully qualified name of an object.
        """
        cls = type(obj)
        module = cls.__module__
        qualname = cls.__qualname__
        if module == "builtins":  # For built-in types like int, str, etc.
            return qualname
        return f"{module}.{qualname}"

    def add_decomposers(self, decomposers: List[Union[str, Decomposer]]):
        """Add decomposers to the job

        Args:
            decomposers: spec of decomposers. Can be class names or Decomposer objects

        Returns: None

        """
        if not decomposers:
            return

        class_names = []
        for d in decomposers:
            if isinstance(d, str):
                # class name
                class_names.append(d)
            elif isinstance(d, Decomposer):
                class_names.append(self._get_full_class_name(d))

        reg = DecomposerRegister(class_names)
        self.job.to_server(reg, id="decomposer_reg")

        self._add_to_client_apps(reg, id="decomposer_reg")

    def export(
        self,
        job_dir: str,
        server_exec_params: dict = None,
        client_exec_params: dict = None,
        env: ExecEnv = None,
    ):
        """Export the recipe to a job definition.

        Args:
            job_dir: directory where the job will be exported to.
            server_exec_params: execution params for the server
            client_exec_params: execution params for clients
            env: the environment that the exported job will be running in

        Returns: None

        """
        if server_exec_params:
            self.job.to_server(server_exec_params)

        if client_exec_params:
            self._add_to_client_apps(client_exec_params)

        if env:
            self.process_env(env)

        self.job.export_job(job_dir)

    def execute(self, env: ExecEnv, server_exec_params: dict = None, client_exec_params: dict = None) -> "Run":
        """Execute the recipe in a specified execution environment.

        Args:
            env: the execution environment
            server_exec_params: execution params for the server
            client_exec_params: execution params for clients

        Returns: Run to get job ID and execution results

        """
        if server_exec_params:
            self.job.to_server(server_exec_params)

        if client_exec_params:
            self._add_to_client_apps(client_exec_params)

        self.process_env(env)
        job_id = env.deploy(self.job)
        from nvflare.recipe.run import Run

        run = Run(env, job_id)
        return run
