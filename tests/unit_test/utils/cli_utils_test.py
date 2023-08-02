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
from pathlib import Path
from unittest.mock import patch
from pyhocon import ConfigFactory as CF

from nvflare.utils.cli_utils import get_hidden_nvflare_dir, get_home_dir, get_hidden_nvflare_config_path, \
    find_startup_kit_location


class TestCLIUtils:

    def test_get_hidden_nvflare_dir(self):
        hidden_dir = get_hidden_nvflare_dir()
        assert str(hidden_dir) == str(Path.home() / ".nvflare")

    def test_get_hidden_nvflare_config_path(self):
        assert get_hidden_nvflare_config_path(str(get_hidden_nvflare_dir())) == \
               str(Path.home() / ".nvflare/config.conf")

    def test_find_startup_kit_location(self):
        with patch("nvflare.utils.cli_utils.load_config") as mock2:
            conf = CF.parse_string(f"""
                startup_kit {{
                    path = "/tmp/nvflare/poc/example_project/prod_00"
                }}
            """)
            mock2.return_value = conf
            assert '/tmp/nvflare/poc/example_project/prod_00' == find_startup_kit_location()

