# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.apis.workspace import Workspace
from nvflare.fuel.common.excepts import ConfigError
from nvflare.fuel.hci.client.api_spec import AdminConfigKey, UidSource
from nvflare.fuel.hci.client.config import secure_load_admin_config
from nvflare.fuel.hci.client.fl_admin_api import FLAdminAPI
from nvflare.fuel.hci.client.fl_admin_api_spec import TargetType
from nvflare.security.logging import secure_format_exception


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
            debug: whether to turn on debug mode
        """
        assert isinstance(username, str), "username must be str"
        self.username = username
        assert isinstance(admin_dir, str), "admin_dir must be str"

        if debug:
            debug = True

        try:
            os.chdir(admin_dir)
            workspace = Workspace(root_dir=admin_dir)
            conf = secure_load_admin_config(workspace)
        except ConfigError as e:
            print(f"ConfigError: {secure_format_exception(e)}")
            return

        admin_config = conf.get_admin_config()
        if not admin_config:
            print(f"Missing '{AdminConfigKey.ADMIN}' section in fed_admin configuration.")
            return

        upload_dir = admin_config.get(AdminConfigKey.UPLOAD_DIR)
        download_dir = admin_config.get(AdminConfigKey.DOWNLOAD_DIR)
        if download_dir and not os.path.isdir(download_dir):
            os.makedirs(download_dir)

        # Connect with admin client
        if poc:
            admin_config[AdminConfigKey.UID_SOURCE] = UidSource.CERT

        self.api = FLAdminAPI(
            admin_config=admin_config,
            upload_dir=upload_dir,
            download_dir=download_dir,
            user_name=username,
            debug=debug,
        )

        self.api.connect(timeout=5.0)
        self.api.login()

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
            print(f"There was an exception: {secure_format_exception(e)}")

    def close(self):
        self.api.logout()
