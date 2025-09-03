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
from enum import Enum
from typing import List, Optional

from nvflare.apis.filter import Filter
from nvflare.job_config.api import FedJob
from nvflare.job_config.defs import FilterType
from nvflare.recipe.run import Run


class ExecEnvType(str, Enum):
    SIM = "sim"
    POC = "poc"
    PROD = "prod"


class ExecEnv(ABC):

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
    def get_env_info(self) -> dict:
        """Get the environment information.

        Returns:
            dict: The environment information.
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
        if clients is None:
            self.job.to_clients(filter, filter_type=FilterType.TASK_DATA, tasks=tasks)
        else:
            for client in clients:
                self.job.to(filter, client, filter_type=FilterType.TASK_DATA, tasks=tasks)

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
        if clients is None:
            self.job.to_clients(filter, filter_type=FilterType.TASK_RESULT, tasks=tasks)
        else:
            for client in clients:
                self.job.to(filter, client, filter_type=FilterType.TASK_RESULT, tasks=tasks)

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

    def export(self, job_dir: str, server_exec_params: dict = None, client_exec_params: dict = None):
        """Export the recipe to a job definition.

        Args:
            job_dir: directory where the job will be exported to.
            server_exec_params: execution params for the server
            client_exec_params: execution params for clients

        Returns: None

        """
        if server_exec_params:
            self.job.to_server(server_exec_params)

        if client_exec_params:
            self.job.to_clients(client_exec_params)

        self.job.export_job(job_dir)

    def execute(self, env: ExecEnv, server_exec_params: dict = None, client_exec_params: dict = None) -> Run:
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
            self.job.to_clients(client_exec_params)

        job_id = env.deploy(self.job)
        run = Run(env.get_env_info(), job_id)
        return run
