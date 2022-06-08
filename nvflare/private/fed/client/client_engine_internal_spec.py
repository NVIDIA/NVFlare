# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.apis.client_engine_spec import ClientEngineSpec, TaskAssignment
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.workspace import Workspace


class ClientEngineInternalSpec(ClientEngineSpec, ABC):
    """The ClientEngineInternalSpec defines the ClientEngine APIs running in the parent process."""

    def get_task_assignment(self, fl_ctx: FLContext) -> TaskAssignment:
        pass

    def send_task_result(self, result: Shareable, fl_ctx: FLContext) -> bool:
        pass

    def get_workspace(self) -> Workspace:
        pass

    def get_all_components(self) -> dict:
        pass

    @abstractmethod
    def get_engine_status(self):
        pass

    @abstractmethod
    def get_client_name(self) -> str:
        """Get the ClientEngine client_name.

        Returns: the client_name

        """
        pass

    @abstractmethod
    def deploy_app(self, app_name: str, job_id: str, client_name: str, app_data) -> str:
        """Deploys the app to specified run.

        Args:
            app_name: FL_app name
            job_id: job that the app is to be deployed to
            client_name: name of the client
            app_data: zip data of the app

        Returns:
            A string message.
        """
        pass

    @abstractmethod
    def start_app(
        self,
        job_id: str,
        allocated_resource: dict = None,
        token: str = None,
        resource_consumer=None,
        resource_manager=None,
    ) -> str:
        """Starts the app for the specified run.

        Args:
            job_id: job_id
            allocated_resource: allocated resource
            token: token
            resource_consumer: resource consumer
            resource_manager: resource manager

        Returns:
            A string message.
        """
        pass

    @abstractmethod
    def abort_app(self, job_id: str) -> str:
        """Aborts the app execution for the specified run.

        Returns:
            A string message.
        """
        pass

    @abstractmethod
    def abort_task(self, job_id: str) -> str:
        """Abort the client current executing task.

        Returns:
            A string message.
        """
        pass

    @abstractmethod
    def delete_run(self, job_id: str) -> str:
        """Deletes the specified run.

        Args:
            job_id: job_id

        Returns:
            A string message.
        """
        pass

    @abstractmethod
    def shutdown(self) -> str:
        """Shuts down the FL client.

        Returns:
            A string message.
        """
        pass

    @abstractmethod
    def restart(self) -> str:
        """Restarts the FL client.

        Returns:
            A string message.
        """
        pass

    @abstractmethod
    def get_all_job_ids(self) -> []:
        """Get all the client job_id.

        Returns: list of all the job_id

        """
        pass
