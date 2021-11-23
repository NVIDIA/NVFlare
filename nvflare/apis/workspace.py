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

"""
Workspace folder structure:

WSROOT
    startup (optional)
        provisioned content
    runs
        1
            config (required)
                config_fed_client.json
                config_fed_server.json
                ...
            custom (optional)
                custom python code
            whatever (app defined)

"""

import os


class Workspace(object):
    def __init__(self, root_dir: str, name: str, config_folder: str):
        self.root_dir = root_dir
        self.name = name
        self.config_folder = config_folder

    def get_startup_kit_dir(self) -> str:
        return os.path.join(self.root_dir, "startup")

    def get_root_dir(self) -> str:
        return self.root_dir

    def get_run_dir(self, run_num: int) -> str:
        return os.path.join(self.root_dir, "run_" + str(run_num))

    def get_app_dir(self, run_num: int) -> str:
        return os.path.join(self.get_run_dir(run_num), "app_" + self.name)

    def get_app_config_dir(self, run_num) -> str:
        return os.path.join(self.get_app_dir(run_num), self.config_folder)

    def get_app_custom_dir(self, run_num) -> str:
        return os.path.join(self.get_app_dir(run_num), "custom")
