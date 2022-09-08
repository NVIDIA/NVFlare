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

import os
import time

from nvflare.fuel.common.excepts import ConfigError
from nvflare.fuel.hci.client.fl_admin_api import FLAdminAPI
from nvflare.fuel.hci.client.fl_admin_api_spec import TargetType
from nvflare.private.fed.app.fl_conf import FLAdminClientStarterConfigurator


def api_command_wrapper(api_command_result):
    """Prints the result of the command and raises RuntimeError to interrupt command sequence if there is an error.

    Args:
        api_command_result: result of the api command

    """
    print(api_command_result)
    if not api_command_result["status"] == "SUCCESS":
        raise RuntimeError("command was not successful!")

    return api_command_result


class FLAdminAPIRunner:
    def __init__(
        self,
        username,
        admin_dir,
        poc=False,
        debug=False,
    ):
        """Initializes and logs into an FLAdminAPI instance.

        The default locations for certs, keys, and directories are used.

        Args:
            username: string of username to log in with
            admin_dir: string of root admin dir containing the startup dir
            poc: whether to run in poc mode without SSL certs
            debug: whether to turn on debug mode
        """
        assert isinstance(username, str), "username must be str"
        self.username = username
        assert isinstance(admin_dir, str), "admin_dir must be str"
        if poc:
            self.poc = True
        else:
            self.poc = False
        if debug:
            debug = True

        try:
            os.chdir(admin_dir)
            workspace = os.path.join(admin_dir, "startup")
            conf = FLAdminClientStarterConfigurator(app_root=workspace, admin_config_file_name="fed_admin.json")
            conf.configure()
        except ConfigError as ex:
            print("ConfigError:", str(ex))

        try:
            admin_config = conf.config_data["admin"]
        except KeyError:
            print("Missing admin section in fed_admin configuration.")

        ca_cert = admin_config.get("ca_cert", "")
        client_cert = admin_config.get("client_cert", "")
        client_key = admin_config.get("client_key", "")

        if admin_config.get("with_ssl"):
            if len(ca_cert) <= 0:
                print("missing CA Cert file name field ca_cert in fed_admin configuration")
                return

            if len(client_cert) <= 0:
                print("missing Client Cert file name field client_cert in fed_admin configuration")
                return

            if len(client_key) <= 0:
                print("missing Client Key file name field client_key in fed_admin configuration")
                return
        else:
            ca_cert = None
            client_key = None
            client_cert = None

        upload_dir = admin_config.get("upload_dir")
        download_dir = admin_config.get("download_dir")
        if not os.path.isdir(download_dir):
            os.makedirs(download_dir, exist_ok=True)

        assert os.path.isdir(admin_dir), f"admin directory does not exist at {admin_dir}"
        if not self.poc:
            assert os.path.isfile(ca_cert), f"rootCA.pem does not exist at {ca_cert}"
            assert os.path.isfile(client_cert), f"client.crt does not exist at {client_cert}"
            assert os.path.isfile(client_key), f"client.key does not exist at {client_key}"

        # Connect with admin client
        self.api = FLAdminAPI(
            ca_cert=ca_cert,
            client_cert=client_cert,
            client_key=client_key,
            upload_dir=upload_dir,
            download_dir=download_dir,
            overseer_agent=conf.overseer_agent,
            user_name=username,
            poc=self.poc,
            debug=debug,
        )

        # wait for admin to login
        _t_warning_start = time.time()
        while not self.api.server_sess_active:
            time.sleep(0.5)
            if time.time() - _t_warning_start > 10:
                print("Admin is taking a long time to log in to the server...")
                print("Make sure the server is up and available, and all configurations are correct.")
                _t_warning_start = time.time()

    def run(
        self,
        job_folder_name,
    ):
        """An example script to upload, deploy, and start a specified app.

        Note that the app folder must be in upload_dir already. Prints the command to be executed first so it is easy
        to follow along as the commands run.

        Args:
            job_folder_name: name of job folder to submit, either relative to the upload_dir specified in the fed_admin.json config, or absolute path

        """
        try:
            print("api.check_status(TargetType.SERVER)")
            api_command_wrapper(self.api.check_status(TargetType.SERVER))
            print(f'api.submit_job("{job_folder_name}")')
            api_command_wrapper(self.api.submit_job(job_folder_name))
            time.sleep(1)
            print("api.check_status(TargetType.SERVER)")
            api_command_wrapper(self.api.check_status(TargetType.SERVER))
            # The following wait_until can be put into a loop that has other behavior other than waiting until clients
            # are in a status of stopped. For this code, the app is expected to stop, or this may not end.
            print("api.wait_until_client_status()")
            wait_result = api_command_wrapper(self.api.wait_until_client_status())
            print(wait_result)
            print("api.check_status(TargetType.SERVER)")
            api_command_wrapper(self.api.check_status(TargetType.SERVER))
            # now server engine status should be stopped
            time.sleep(10)  # wait for clients to stop in case they take longer than server to stop
            print("api.check_status(TargetType.CLIENT)")
            api_command_wrapper(self.api.check_status(TargetType.CLIENT))
        except RuntimeError as e:
            print(f"There was an exception: {e}")
