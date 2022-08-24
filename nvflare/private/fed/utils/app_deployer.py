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
import shutil

from nvflare.apis.workspace import Workspace
from nvflare.fuel.hci.zip_utils import unzip_all_from_bytes
from .app_authz import AppAuthzService


class AppDeployer(object):

    def __init__(
            self,
            submitter_name: str,
            submitter_org: str,
            submitter_role: str,
            workspace: Workspace,
            job_id: str,
            app_name: str,
            app_data
    ):
        self.submitter_name = submitter_name
        self.submitter_org = submitter_org
        self.submitter_role = submitter_role
        self.app_name = app_name
        self.workspace = workspace
        self.job_id = job_id
        self.app_data = app_data

    def deploy(self) -> str:
        """
        Try to deploy the app.

        Returns: error message if any

        """
        try:
            run_dir = self.workspace.get_run_dir(self.job_id)
            app_path = self.workspace.get_app_dir(self.job_id)
            app_file = os.path.join(run_dir, "fl_app.txt")

            if os.path.exists(run_dir):
                shutil.rmtree(run_dir)

            if not os.path.exists(app_path):
                os.makedirs(app_path)

            unzip_all_from_bytes(self.app_data, app_path)

            with open(app_file, "wt") as f:
                f.write(f"{self.app_name}")

            authorized, err = AppAuthzService.authorize(
                app_path=app_path,
                submitter_name=self.submitter_name,
                submitter_org=self.submitter_org,
                submitter_role=self.submitter_role
            )
            if err:
                return err

            if not authorized:
                return "not authorized"

        except BaseException as ex:
            return "exception {} when deploying app {}".format(ex, self.app_name)
