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

import os
import time

from nvflare.fuel.hci.client.fl_admin_api import FLAdminAPI
from nvflare.fuel.hci.client.fl_admin_api_constants import FLDetailKey
from nvflare.fuel.hci.client.fl_admin_api_spec import TargetType

from nvflare.fuel.hci.client.fl_admin_api_runner import FLAdminAPIRunner, api_command_wrapper, wait_until_clients_greater_than_cb

class CustomFLAdminAPIRunner(FLAdminAPIRunner):
    def __init__(
        self,
        host,
        port,
        username,
        admin_dir,
        poc=False,
        debug=False,
    ):        
        super().__init__(
            host,
            port,
            username,
            admin_dir,
            poc=poc,
            debug=debug,
            )

    def run(
        self,
        run_number,
        app,
        restart_all_first=False,
        min_clients=2,
        timeout=2000,
        shutdown_on_error=False,
        shutdown_at_end=False,
    ):
        """An example script to start a specified app(app folder must be transferred and deployed already).
        Prints the command to be executed first so it is easy to follow along as the commands run.

        Args:
            run_number: run number to use
            app: app to upload, deploy, and start
            min_clients: minimum number of clients to have registered on the server after restart all before proceeding (default: 2)
            timeout: number of seconds to wait in the wait_until_client_status command before the app is aborted in this example (default: 2000)
            restart_all_first: whether to restart all before the run in order to have clean environment
            shutdown_on_error: whether to shut down all if there is an error on app startup
            shutdown_at_end: whether to shut down all at the end of script

        """
        try:
            if restart_all_first:
                print(f'api.restart("{TargetType.ALL}")')
                api_command_wrapper(self.api.restart(TargetType.ALL))
                self.attempt_login_until_success()
                print("api.wait_until_server_status(callback=wait_until_clients_greater_than_cb)")
                api_command_wrapper(
                    self.api.wait_until_server_status(
                        callback=wait_until_clients_greater_than_cb, min_clients=min_clients
                    )
                )
            print("api.check_status(TargetType.SERVER)")
            api_command_wrapper(self.api.check_status(TargetType.SERVER))
            print(f"api.set_run_number({run_number})")
            api_command_wrapper(self.api.set_run_number(run_number))
            print("api.check_status(TargetType.CLIENT)")
            api_command_wrapper(self.api.check_status(TargetType.CLIENT))
            print("api.start_app(TargetType.ALL)")
            api_command_wrapper(self.api.start_app(TargetType.ALL))
            time.sleep(1)
            print("api.check_status(TargetType.SERVER)")
            reply = api_command_wrapper(self.api.check_status(TargetType.SERVER))
            if shutdown_on_error and reply["details"][FLDetailKey.SERVER_ENGINE_STATUS] == "stopped":
                print("Server startup failed! Shutdown all...")
                api_command_wrapper(self.api.shutdown(TargetType.ALL))
                return
            # The following wait_until can be put into a loop that has other behavior other than aborting after the
            # timeout is reached for actual apps. This code is just a demonstration of running an app expected to stop
            # before the timeout.
            print("api.wait_until_client_status()")
            wait_result = api_command_wrapper(self.api.wait_until_client_status(timeout=timeout))
            if wait_result.get("details"):
                if wait_result.get("details").get("message"):
                    if wait_result.get("details").get("message") == "Waited until timeout.":
                        print(
                            "aborting because waited until timeout and there are clients that have still not stopped."
                        )
                        print("api.abort(TargetType.ALL)")
                        api_command_wrapper(self.api.abort(TargetType.ALL))
            print("api.check_status(TargetType.SERVER)")
            api_command_wrapper(self.api.check_status(TargetType.SERVER))
            # now server engine status should be stopped
            time.sleep(10)  # wait for clients to stop in case they take longer than server to stop
            print("api.check_status(TargetType.CLIENT)")
            api_command_wrapper(self.api.check_status(TargetType.CLIENT))
            if shutdown_at_end:
                print("api.shutdown(TargetType.ALL)")
                api_command_wrapper(self.api.shutdown(TargetType.ALL))
        except RuntimeError as e:
            print(f"There was an exception: {e}")
            if shutdown_on_error:
                print("Attempting shutdown all...")
                try:
                    api_command_wrapper(self.api.shutdown(TargetType.ALL))
                except RuntimeError as e:
                    print(f"There was an exception while attempting shutdown all: {e}")

    def transfer(
        self,
        run_number,
        app,
        restart_all_first=False,
        min_clients=2,
        timeout=2000,
        shutdown_on_error=False,
        shutdown_at_end=False,
    ):
        """An example script to upload and deploy app(app folder must be in upload_dir already).
        Prints the command to be executed first so it is easy to follow along as the commands run.

        Args:
            run_number: run number to use
            app: app to upload, deploy, and start
            min_clients: minimum number of clients to have registered on the server after restart all before proceeding (default: 2)
            timeout: number of seconds to wait in the wait_until_client_status command before the app is aborted in this example (default: 2000)
            restart_all_first: whether to restart all before the run in order to have clean environment
            shutdown_on_error: whether to shut down all if there is an error on app startup
            shutdown_at_end: whether to shut down all at the end of script

        """
        try:
            if restart_all_first:
                print(f'api.restart("{TargetType.ALL}")')
                api_command_wrapper(self.api.restart(TargetType.ALL))
                self.attempt_login_until_success()
                print("api.wait_until_server_status(callback=wait_until_clients_greater_than_cb)")
                api_command_wrapper(
                    self.api.wait_until_server_status(
                        callback=wait_until_clients_greater_than_cb, min_clients=min_clients
                    )
                )
            print("api.check_status(TargetType.SERVER)")
            api_command_wrapper(self.api.check_status(TargetType.SERVER))
            print(f"api.set_run_number({run_number})")
            api_command_wrapper(self.api.set_run_number(run_number))
            print(f'api.upload_app("{app}")')
            api_command_wrapper(self.api.upload_app(app))
            print(f'api.deploy_app("{app}", TargetType.ALL)')
            api_command_wrapper(self.api.deploy_app(app, TargetType.ALL))
            print("api.check_status(TargetType.CLIENT)")
            api_command_wrapper(self.api.check_status(TargetType.CLIENT))
        except RuntimeError as e:
            print(f"There was an exception: {e}")
            if shutdown_on_error:
                print("Attempting shutdown all...")
                try:
                    api_command_wrapper(self.api.shutdown(TargetType.ALL))
                except RuntimeError as e:
                    print(f"There was an exception while attempting shutdown all: {e}")

    def attempt_login_until_success(self):
        logged_in = False
        while not logged_in:
            reply = None
            try:
                if self.poc:
                    reply = self.api.login_with_password("admin", "admin")
                else:
                    print("api.login()")
                    reply = api_command_wrapper(self.api.login(self.username))
            except RuntimeError:
                time.sleep(10)
                pass
            if reply and reply["status"] == "SUCCESS":
                logged_in = True
        return True
