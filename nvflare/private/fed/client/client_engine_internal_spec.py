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

from nvflare.apis.client_engine_spec import ClientEngineSpec


class ClientEngineInternalSpec(ClientEngineSpec):
    """
    The ClientEngineInternalSpec defines the ClientEngine APIs running in the parent process.
    """
    def get_engine_status(self):
        pass

    # def get_current_run_info(self) -> ClientRunInfo:
    #     """
    #     Return info of the current run.
    #
    #     Returns: current run info, or None if app not running.
    #
    #     """
    #     pass

    def get_client_name(self) -> str:
        """
        Get the ClientEngine client_name
        Returns: the client_name

        """
        pass

    def deploy_app(self, app_name: str, run_num: int, client_name: str, app_data) -> str:
        """
        Deploy the app to specified run

        Args:
            app_name: FL_app name
            run_num: run that the app is to be deployed to
            client_name: name of the client
            app_data: zip data of the app

        Returns: error if any

        """
        pass

    def start_app(self, run_number: int) -> str:
        """
        Start the app for the specified run

        Args:
            run_number:

        Returns: error if any

        """
        pass

    def abort_app(self, run_number: int) -> str:
        """
        Abort the app execution in current run.

        Returns: error if any

        """
        pass

    def abort_task(self, run_number: int) -> str:
        """
        Abort the client current executing task.

        Returns: error if any

        """
        pass

    def delete_run(self, run_num: int) -> str:
        """
        Delete the specified run

        Args:
            run_num:

        Returns: error if any

        """
        pass

    def shutdown(self) -> str:
        """
        Shutdown the FL client

        Returns: error if any

        """
        pass

    def restart(self) -> str:
        """
        Restart the FL client.

        Returns: error if any

        """
        pass

    def set_run_number(self, run_num: int) -> str:
        """
        Set the client run_number

        Returns: error if any

        """
        pass

    # def aux_send(self,
    #              topic: str,
    #              request: Shareable,
    #              timeout: float,
    #              fl_ctx: FLContext) -> Shareable:
    #     """
    #     Send the request to the Server.
    #
    #     If reply is received, make sure to set peer_ctx into the reply shareable!
    #
    #     Args:
    #         topic: topic of the request
    #         request: request to be sent
    #         timeout: number of secs to wait for reply. 0 means fire-and-forget.
    #         fl_ctx: fl context
    #
    #     Returns: a reply.
    #
    #     """
    #     pass
