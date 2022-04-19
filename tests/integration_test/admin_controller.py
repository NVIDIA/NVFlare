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

import logging
import time

from nvflare.fuel.hci.client.api_status import APIStatus
from nvflare.fuel.hci.client.fl_admin_api import FLAdminAPI
from nvflare.fuel.hci.client.fl_admin_api_constants import FLDetailKey
from nvflare.fuel.hci.client.fl_admin_api_spec import TargetType


class AdminController:
    def __init__(self, jobs_root_dir, poll_period=10):
        """
        This class runs an app on a given server and clients.
        """
        super().__init__()

        self.jobs_root_dir = jobs_root_dir
        self.poll_period = poll_period

        self.admin_api: FLAdminAPI = FLAdminAPI(
            host="localhost",
            port=8003,
            upload_dir=self.jobs_root_dir,
            download_dir=self.jobs_root_dir,
            poc=True,
            debug=False,
            user_name="admin",
            password="admin",
        )

        self.logger = logging.getLogger(self.__class__.__name__)
        self.job_id = None

    def initialize(self):
        success = False

        try:
            response = None
            timeout = 100
            start_time = time.time()
            while time.time() - start_time <= timeout:
                response = self.admin_api.login_with_password(username="admin", password="admin")
                if response["status"] == APIStatus.SUCCESS:
                    success = True
                    break
                time.sleep(1.0)

            if not success:
                details = response.get("details") if response else "No details"
                raise ValueError(f"Login to admin api failed: {details}")
            else:
                print("Admin successfully logged into server.")
        except Exception as e:
            print(f"Exception in logging in to admin: {e.__str__()}")
        return success

    def get_run_data(self):
        run_data = {"job_id": self.job_id, "jobs_root_dir": self.jobs_root_dir}

        return run_data

    def ensure_clients_started(self, num_clients):
        if not self.admin_api:
            return False

        timeout = 1000
        start_time = time.time()
        clients_up = False
        while not clients_up:
            if time.time() - start_time > timeout:
                raise ValueError(f"Clients could not be started in {timeout} seconds.")

            response = self.admin_api.check_status(target_type=TargetType.CLIENT)
            if response["status"] == APIStatus.SUCCESS:
                # print(f"check client status response {response}")
                if "details" not in response:  # clients not ready....
                    # client response would be: {'status': <APIStatus.SUCCESS: 'SUCCESS'>, 'raw': {'time': '2021-10-29 00:09:06.220615', 'data': [{'type': 'error', 'data': 'no clients available'}], 'status': <APIStatus.SUCCESS: 'SUCCESS'>}}
                    # How can the user know if clients are ready or not.....
                    continue
                for row in response["details"]["client_statuses"][1:]:
                    if row[3] != "not started":
                        continue
                # wait for all clients to come up
                if len(response["details"]["client_statuses"]) < num_clients + 1:
                    continue
                clients_up = True
                print("All clients are up.")
            time.sleep(1.0)

        return clients_up

    def server_status(self):
        if not self.admin_api:
            return ""

        response = self.admin_api.check_status(target_type=TargetType.SERVER)
        if response["status"] == APIStatus.SUCCESS:
            if "details" in response:
                return response["details"]
        return ""

    def client_status(self):
        if not self.admin_api:
            return ""

        response = self.admin_api.check_status(target_type=TargetType.CLIENT)
        if response["status"] == APIStatus.SUCCESS:
            if "details" in response:
                return response["details"]
        return ""

    def submit_job(self, job_name) -> bool:
        if not self.admin_api:
            return False

        response = self.admin_api.upload_job(job_name)
        if response["status"] != APIStatus.SUCCESS:
            raise RuntimeError(f"upload_job failed: {response}")
        self.job_id = response["details"]["job_id"]

        return True

    def wait_for_job_done(self):
        # TODO:: Is it possible to get the training log after training is done?
        training_done = False
        while not training_done:
            time.sleep(self.poll_period)
            response = self.admin_api.check_status(target_type=TargetType.SERVER)
            if response["status"] != APIStatus.SUCCESS:
                raise RuntimeError(f"check_status failed: {response}")
            if response["details"][FLDetailKey.SERVER_ENGINE_STATUS] == "stopped":
                response = self.admin_api.check_status(target_type=TargetType.CLIENT)
                if response["status"] != APIStatus.SUCCESS:
                    raise RuntimeError(f"check_status failed: {response}")
                for row in response["details"]["client_statuses"]:
                    if row[3] != "stopped":
                        continue
                training_done = True

    def finalize(self):
        self.admin_api.shutdown(target_type=TargetType.ALL)
