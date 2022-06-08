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

from nvflare.apis.fl_constant import WorkspaceConstants


class Workspace:
    def __init__(self, root_dir: str, name: str, config_folder: str):
        """Define a workspace.

        NOTE::

            Workspace folder structure:

                Workspace ROOT
                    startup (optional)
                        provisioned content
                    run_1
                        config (required)
                            configurations
                        custom (optional)
                            custom python code
                        other_folder (app defined)

        Args:
            root_dir: root directory of the workspace
            name: name of the workspace
            config_folder: where to find required config inside an app
        """
        self.root_dir = root_dir
        self.name = name
        self.config_folder = config_folder

    def get_startup_kit_dir(self) -> str:
        return os.path.join(self.root_dir, "startup")

    def get_root_dir(self) -> str:
        return self.root_dir

    def get_run_dir(self, job_id: str) -> str:
        return os.path.join(self.root_dir, WorkspaceConstants.WORKSPACE_PREFIX + str(job_id))

    def get_app_dir(self, job_id: str) -> str:
        return os.path.join(self.get_run_dir(job_id), WorkspaceConstants.APP_PREFIX + self.name)

    def get_app_config_dir(self, job_id: str) -> str:
        return os.path.join(self.get_app_dir(job_id), self.config_folder)

    def get_app_custom_dir(self, job_id: str) -> str:
        return os.path.join(self.get_app_dir(job_id), "custom")
