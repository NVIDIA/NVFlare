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

from abc import abstractmethod
from typing import Dict, List, Optional, Tuple

from .client import Client
from .engine_spec import EngineSpec
from .fl_context import FLContext
from .fl_snapshot import RunSnapshot


class ServerEngineSpec(EngineSpec):

    @abstractmethod
    def get_clients(self) -> List[Client]:
        pass

    @abstractmethod
    def sync_clients_from_main_process(self):
        """To fetch the participating clients from the main parent process

        Returns: clients

        """
        pass

    @abstractmethod
    def update_job_run_status(self):
        """To update the job run status to parent process."""
        pass

    @abstractmethod
    def validate_clients(self, client_names: List[str]) -> Tuple[List[Client], List[str]]:
        """Validate specified client names.

        Args:
            client_names: list of names to be validated

        Returns: a list of validate clients  and a list of invalid client names

        """
        pass

    @abstractmethod
    def persist_components(self, fl_ctx: FLContext, completed: bool):
        """To persist the FL running components

        Args:
            fl_ctx: FLContext
            completed: flag to indicate where the run is complete

        Returns:

        """
        pass

    @abstractmethod
    def restore_components(self, snapshot: RunSnapshot, fl_ctx: FLContext):
        """To restore the FL components from the saved snapshot

        Args:
            snapshot: RunSnapshot
            fl_ctx: FLContext

        Returns:

        """
        pass

    @abstractmethod
    def start_client_job(self, job_id, client_sites):
        """To send the start client run commands to the clients

        Args:
            client_sites: client sites
            job_id: job_id

        Returns:

        """
        pass

    @abstractmethod
    def check_client_resources(
        self, job_id: str, resource_reqs: Dict[str, dict]
    ) -> Dict[str, Tuple[bool, Optional[str]]]:
        """Sends the check_client_resources requests to the clients.

        Args:
            job_id: ID of the job
            resource_reqs: A dict of {client_name: resource requirements dict}

        Returns:
            A dict of {client_name: client_check_result}.
                client_check_result is a tuple of (is_resource_enough, token);
                is_resource_enough is a bool indicates whether there is enough resources;
                token is for resource reservation / cancellation for this check request.
        """
        pass

    @abstractmethod
    def cancel_client_resources(
        self, resource_check_results: Dict[str, Tuple[bool, str]], resource_reqs: Dict[str, dict]
    ):
        """Cancels the request resources for the job.

        Args:
            resource_check_results: A dict of {client_name: client_check_result}
                where client_check_result is a tuple of (is_resource_enough, resource reserve token if any)
            resource_reqs: A dict of {client_name: resource requirements dict}
        """
        pass

    @abstractmethod
    def get_client_name_from_token(self, token: str) -> str:
        """Gets client name from a client login token."""
        pass
