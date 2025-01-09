# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import tempfile

import pytest

from nvflare.job_config.base_app_config import BaseAppConfig


class TestBaseAppConfig:
    def setup_method(self, method):
        self.app_config = BaseAppConfig()

    def test_add_relative_script(self):
        cwd = os.getcwd()
        with tempfile.NamedTemporaryFile(dir=cwd, suffix=".py") as temp_file:
            script = os.path.basename(temp_file.name)
            self.app_config.add_ext_script(script)
            assert script in self.app_config.ext_scripts

    def test_add_ext_script(self):
        script = "/scripts/sample.py"
        self.app_config.add_ext_script(script)
        assert script in self.app_config.ext_scripts

    def test_add_ext_script_error(self):
        script = "scripts/sample.py"
        with pytest.raises(Exception):
            self.app_config.add_ext_script(script)
