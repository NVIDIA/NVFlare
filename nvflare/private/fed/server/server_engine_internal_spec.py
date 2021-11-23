# Copyright (c) 2021, NVIDIA CORPORATION.
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
from abc import ABC

from nvflare.apis.fl_constant import MachineStatus
from nvflare.apis.fl_context import FLContext
from nvflare.apis.server_engine_spec import ServerEngineSpec
from nvflare.apis.shareable import Shareable
from .run_manager import RunManager, RunInfo
from .server_json_config import ServerJsonConfigurator


class EngineInfo(object):
    def __init__(self):
        self.start_time = time.time()
        self.status = MachineStatus.STOPPED

        self.app_name = "?"


class ServerEngineInternalSpec(ServerEngineSpec, ABC):

    # methods for internal use
    def get_engine_info(self) -> EngineInfo:
        """
        Get general info of the Engine

        Returns:

        """
        pass

    def get_run_info(self) -> RunInfo:
        pass

    def get_staging_path_of_app(self, app_name: str) -> str:
        """
        Get the staging path of the app waiting to be deployed

        Args:
            app_name:

        Returns: the app's folder path or empty string if the app doesn't exist

        """
        pass

    def deploy_app_to_server(self, app_name: str, app_staging_path: str) -> str:
        """
        Deploy the specified app to the server.
        Copy the app folder tree from staging area to the server's RUN area

        Args:
            app_name: name of the app to be deployed
            app_staging_path: the full path to the app folder in staging area

        Returns: error message if couldn't deploy, or empty string if successful

        """
        pass

    def prepare_deploy_app_to_client(self, app_name: str, app_staging_path: str, client_name: str) -> str:
        """
        Prepare to deploy the specified app to the specified client name.
        Copy the app folder tree from staging area to the client's RUN area on Server
        Args:
            app_name: name of the app
            app_staging_path: the full path to the app folder in staging area
            client_name: name of the client

        Returns: error message if couldn't deploy, or empty string if successful

        """
        pass

    def get_app_data(self, app_name: str) -> (str, object):
        """
        Get data for deploying the app

        Args:
            client_name:

        Returns: error msg and data object.

        """
        pass

    def get_run_number(self) -> str:
        """
        Get the current run_number of the engine
        Returns:run_number

        """
        pass

    def set_run_number(self, run_num: int) -> str:
        """
        Set the run number for the next RUN.

        Args:
            run_num:

        Returns: error if any

        """
        pass

    def delete_run_number(self, run_num: int) -> str:
        """
        Delete specified RUN.

        The Engine must do status check before the run can be deleted.
        Args:
            run_num:

        Returns: error if any

        """
        pass

    def start_app_on_server(self) -> str:
        """
        Start the FL app on Server.

        Returns: error if any

        """
        pass

    def check_app_start_readiness(self) -> str:
        """
        Check whether the app is ready to start.

        Returns: error if any

        """
        pass

    def abort_app_on_clients(self, clients: [str]):
        """
        Clean up the specified client for aborting

        Returns:

        """
        pass

    def abort_app_on_server(self):
        """
        Clean up the app on Server

        Returns:

        """
        pass

    def shutdown_server(self) -> str:
        """
        Shutdown the FL Server

        Returns: error if any

        The engine should not exit right away.
        It should set its status to STOPPING, and set up a timer (in a different thread),
        and return from this call right away (if other restart conditions are met).
        When the timer fires, it exits.
        This would give the caller to process the feedback or clean up (e.g. admin cmd response).

        """
        pass

    def remove_clients(self, clients: [str]) -> str:
        """
        Remove specified clients

        Args:
            clients:

        Returns: error if any

        """
        pass

    def restart_server(self) -> str:
        """
        Restart the FL Server

        Returns: error if any

        The engine should not exit right away.
        See shutdown_server.

        """
        pass

    def set_run_manager(self, run_manager: RunManager):
        """
        set the RunManager for FL server
        Args:
            run_manager:

        Returns:

        """
        pass

    def set_configurator(self, conf: ServerJsonConfigurator):
        """
        set the ServerJsonConfigurator for FL server
        Args:
            conf:

        Returns:

        """
        pass

    def build_component(self, config_dict):
        """
        Build a component from the config_dict.
        Args:
            config_dict:

        Returns:

        """
        pass

    def get_client_name_from_token(self, token: str) -> str:
        """
        Get the registered client name from communication token.
        Args:
            token:

        Returns: client name

        """
        pass

    def ask_to_stop(self):
        """
        Ask the engine to stop the current run.
        Returns:

        """
        pass

    def aux_send(
            self,
            targets: [],
            topic: str,
            request: Shareable,
            timeout: float,
            fl_ctx: FLContext
    ) -> dict:
        """
        Send a request to client(s) via the aux channel

        Args:
            targets: list of Client or client names
            topic: topic of the request
            request: request to be sent
            timeout: number of secs to wait for replies
            fl_ctx: FL context

        Returns: a MessageSendStatus and a dict of replies: client_name => Shareable

        NOTE: when a reply is received, the peer_ctx props must be set into the PEER_PROPS header
        of the reply Shareable.

        If a reply is not received from a client, do not put it into the reply dict.

        """
        pass
