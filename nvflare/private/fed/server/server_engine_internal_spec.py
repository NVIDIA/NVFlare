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

import time
from abc import ABC, abstractmethod

from nvflare.apis.client import Client
from nvflare.apis.fl_constant import MachineStatus
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def_manager_spec import JobDefManagerSpec
from nvflare.apis.server_engine_spec import ServerEngineSpec
from nvflare.apis.shareable import Shareable

from .job_runner import JobRunner
from .run_manager import RunInfo, RunManager
from .server_json_config import ServerJsonConfigurator


class EngineInfo(object):
    def __init__(self):
        """Engine information."""
        self.start_time = time.time()
        self.status = MachineStatus.STOPPED

        self.app_names = {}


class ServerEngineInternalSpec(ServerEngineSpec, ABC):
    def get_engine_info(self) -> EngineInfo:
        """Get general info of the engine."""
        pass

    def get_run_info(self) -> RunInfo:
        pass

    @abstractmethod
    def get_staging_path_of_app(self, app_name: str) -> str:
        """Get the staging path of the app waiting to be deployed.

        Args:
            app_name (str): application name

        Returns:
            The app's folder path or empty string if the app doesn't exist
        """
        pass

    @abstractmethod
    def deploy_app_to_server(self, job_id: str, app_name: str, app_staging_path: str) -> str:
        """Deploy the specified app to the server.

        Copy the app folder tree from staging area to the server's RUN area

        Args:
            job_id: job id of the app to be deployed
            app_name: name of the app to be deployed
            app_staging_path: the full path to the app folder in staging area

        Returns:
            An error message. An empty string if successful.
        """
        pass

    @abstractmethod
    def get_app_data(self, app_name: str) -> (str, object):
        """Get data for deploying the app.

        Args:
            app_name: name of the app

        Returns:
            An error message. An empty string if successful.
        """
        pass

    @abstractmethod
    def get_app_run_info(self, job_id) -> RunInfo:
        """Get the app RunInfo from the child process.

        Returns:
            App RunInfo

        """
        pass

    @abstractmethod
    def delete_job_id(self, job_id: str) -> str:
        """Delete specified RUN.

        The Engine must do status check before the run can be deleted.
        Args:
            job_id: job id

        Returns:
            An error message. An empty string if successful.
        """
        pass

    @abstractmethod
    def start_app_on_server(self, job_idber: str, job_id: str = None, job_clients=None, snapshot=None) -> str:
        """Start the FL app on Server.

        Returns:
            An error message. An empty string if successful.
        """
        pass

    @abstractmethod
    def check_app_start_readiness(self, job_id: str) -> str:
        """Check whether the app is ready to start.

        Returns:
            An error message. An empty string if successful.
        """
        pass

    @abstractmethod
    def abort_app_on_clients(self, clients: [str]):
        """Abort the application on the specified clients."""
        pass

    @abstractmethod
    def abort_app_on_server(self, job_id: str):
        """Abort the application on the server."""
        pass

    @abstractmethod
    def shutdown_server(self) -> str:
        """Shutdown the server.

        The engine should not exit right away.
        It should set its status to STOPPING, and set up a timer (in a different thread),
        and return from this call right away (if other restart conditions are met).
        When the timer fires, it exits.
        This would give the caller to process the feedback or clean up (e.g. admin cmd response).

        Returns:
            An error message. An empty string if successful.
        """
        pass

    @abstractmethod
    def remove_clients(self, clients: [str]) -> str:
        """Remove specified clients.

        Args:
            clients: clients to be removed

        Returns:
             An error message. An empty string if successful.
        """
        pass

    @abstractmethod
    def restart_server(self) -> str:
        """Restart the server.

        The engine should not exit right away.
        See shutdown_server.

        Returns:
             An error message. An empty string if successful.
        """
        pass

    @abstractmethod
    def set_run_manager(self, run_manager: RunManager):
        """Set the RunManager for server.

        Args:
            run_manager: A RunManager object
        """
        pass

    @abstractmethod
    def set_job_runner(self, job_runner: JobRunner, job_manager: JobDefManagerSpec):
        """Set the JobRunner for server.

        Args:
            job_runner: A JobRunner object
            job_manager: A JobDefManagerSpec object
        """
        pass

    @abstractmethod
    def set_configurator(self, conf: ServerJsonConfigurator):
        """Set the configurator for server.

        Args:
            conf: A ServerJsonConfigurator object
        """
        pass

    @abstractmethod
    def build_component(self, config_dict):
        """Build a component from the config_dict.

        Args:
            config_dict: configuration.
        """
        pass

    @abstractmethod
    def get_client_name_from_token(self, token: str) -> str:
        """Get the registered client name from communication token.

        Args:
            token: communication token

        Returns:
            Client name
        """
        pass

    @abstractmethod
    def get_client_from_name(self, client_name: str) -> Client:
        """Get the registered client token from client_name.

        Args:
            client_name: client name

        Returns: registered client

        """
        pass

    @abstractmethod
    def get_job_clients(self, client_sites) -> {}:
        """To get the participating clients for the job

        Args:
            client_sites: clients with the dispatching info

        Returns:

        """
        pass

    @abstractmethod
    def ask_to_stop(self):
        """Ask the engine to stop the current run."""
        pass

    @abstractmethod
    def aux_send(self, targets: [], topic: str, request: Shareable, timeout: float, fl_ctx: FLContext) -> dict:
        """Send a request to client(s) via the auxiliary channel.

        Args:
            targets: list of Client or client names
            topic: topic of the request
            request: request to be sent
            timeout: number of secs to wait for replies
            fl_ctx: FL context

        Returns:
             A dict of replies: client_name => Shareable

        NOTE: when a reply is received, the peer_ctx props must be set into the PEER_PROPS header
        of the reply Shareable.

        If a reply is not received from a client, do not put it into the reply dict.
        """
        pass

    @abstractmethod
    def show_stats(self, job_id):
        """Show_stats of the server.

        Args:
            job_id: current job_id

        Returns:
            Component stats of the server

        """
        pass

    @abstractmethod
    def get_errors(self, job_id):
        """Get the errors of the server components.

        Args:
            job_id: current job_id

        Returns:
            Server components errors.

        """
        pass
