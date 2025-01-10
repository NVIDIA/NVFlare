# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import pytest

from nvflare.lighter.provision import prepare_project


class TestProvision:
    def test_prepare_project(self):
        project_config = {"api_version": 2}
        with pytest.raises(ValueError, match="API version expected 3 but found 2"):
            prepare_project(project_dict=project_config)

        project_config = {
            "api_version": 3,
            "name": "mytest",
            "description": "test",
            "participants": [
                {"type": "server", "name": "server1", "org": "org"},
                {"type": "server", "name": "server2", "org": "org"},
                {"type": "server", "name": "server3", "org": "org"},
            ],
        }

        with pytest.raises(ValueError, match=".* already has a server defined"):
            prepare_project(project_dict=project_config)
