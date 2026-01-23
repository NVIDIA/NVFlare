# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import os.path

from nvflare.collab.api.workspace import Workspace


class SimWorkspace(Workspace):

    def __init__(self, root_dir: str, experiment_name: str, exp_id: str, site_name: str):
        super().__init__()
        if not isinstance(root_dir, str):
            raise ValueError(f"root_dir must be str but got {type(root_dir)}")

        if not isinstance(exp_id, str):
            raise ValueError(f"exp_id must be str but got {type(exp_id)}")

        if not isinstance(experiment_name, str):
            raise ValueError(f"experiment_name must be str but got {type(experiment_name)}")

        if not isinstance(site_name, str):
            raise ValueError(f"site_name must be str but got {type(site_name)}")

        self.root_dir = root_dir
        self.site_name = site_name
        self.exp_name = experiment_name
        self.exp_dir = os.path.join(root_dir, experiment_name, exp_id)
        self.work_dir = os.path.join(self.exp_dir, site_name)
        os.makedirs(self.work_dir, exist_ok=True)

    def get_root_dir(self) -> str:
        return self.root_dir

    def get_work_dir(self) -> str:
        return self.work_dir

    def get_experiment_dir(self) -> str:
        return self.exp_dir
