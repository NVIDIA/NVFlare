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

import pytest
from pyhocon import ConfigFactory as CF

from nvflare.utils.cli_utils import (
    append_if_not_in_list,
    create_startup_kit_config,
    get_hidden_nvflare_config_path,
    get_hidden_nvflare_dir,
)


class TestCLIUtils:
    def test_get_hidden_nvflare_dir(self):
        hidden_dir = get_hidden_nvflare_dir()
        assert str(hidden_dir) == str(Path.home() / ".nvflare")

    def test_get_hidden_nvflare_config_path(self):
        assert get_hidden_nvflare_config_path(str(get_hidden_nvflare_dir())) == str(
            Path.home() / ".nvflare/config.conf"
        )

    def test_create_startup_kit_config(self):
        with patch("nvflare.utils.cli_utils.check_startup_dir", side_effect=None) as mock:
            mock.return_value = ""
            with patch("os.path.isdir", side_effect=None) as mock1:
                mock1.return_value = True
                prev_conf = CF.parse_string(
                    """
                        poc_workspace {
                            path = "/tmp/nvflare/poc"
                        }
                    """
                )
                config = create_startup_kit_config(
                    nvflare_config=prev_conf, startup_kit_dir="/tmp/nvflare/poc/example_project/prod_00"
                )

                assert "/tmp/nvflare/poc" == config.get("poc_workspace.path")
                assert "/tmp/nvflare/poc/example_project/prod_00" == config.get("startup_kit.path")

                config = create_startup_kit_config(nvflare_config=prev_conf, startup_kit_dir="")

                assert config.get("startup_kit.path", None) is None

    @pytest.mark.parametrize(
        "inputs, result", [(([], "a"), ["a"]), ((["a"], "a"), ["a"]), ((["a", "b"], "b"), ["a", "b"])]
    )
    def test_append_if_not_in_list(self, inputs, result):
        arr, item = inputs
        assert result == append_if_not_in_list(arr, item)
