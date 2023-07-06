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
from unittest.mock import MagicMock

import pytest

from nvflare.fuel.utils.config import ConfigFormat, ConfigLoader
from nvflare.fuel.utils.config_factory import ConfigFactory


class TestConfigFactory:
    @pytest.mark.parametrize(
        "init_file_path, file_basename",
        [
            ["fed_client_config.json", "fed_client_config"],
            ["fed_client_config.json.default", "fed_client_config"],
            ["/abc/efg/acb@abc.com/fed_client_config.json", "fed_client_config"],
            ["/abc/efg/acb@abc.com/fed_client_config.json.default", "fed_client_config"],
            ["./abc/fed_client_config.json.default", "fed_client_config"],
        ],
    )
    def test_get_file_basename(self, init_file_path, file_basename):
        assert file_basename == ConfigFactory.get_file_basename(init_file_path)

    def test_fmt_to_loader(self):
        fmt2loaders = ConfigFactory._fmt2Loader
        ext2fmts = ConfigFormat.config_ext_formats()

        config_dicts = dict(
            example="1",
            version=2,
            a=dict(a=2, b=3),
            databases=dict(
                mysql=dict(url="mysql://xxx", user="admin"), postgres=dict(url="postgres://xxx", user="root")
            ),
            arrs=[1, 2, 5, 6],
        )

        # note this tests assume Pyhocon and OmegaConf both are installed.
        # to test only Pyhocon installed (but OmegaConf not installed) or
        # OmegaConf installed (but Pyhocon Not installed)
        # One can simply filter one config format out. The code should still work without asking for other dependency

        for ext in ext2fmts:
            fmt = ext2fmts[ext]
            loader: ConfigLoader = fmt2loaders[fmt]
            assert isinstance(loader, ConfigLoader)
            assert loader.get_format() == fmt
            config = loader.load_config_from_dict(config_dicts)
            assert config.to_dict() == config_dicts

    def _walk_result(self, search_dir):
        if search_dir == ".":
            return [(".", [], ["fed_client_config.conf", "fed_client_config.yml"])]

        elif search_dir == "/tmp/nvflare":

            return [
                ("/tmp/nvflare", ["site-1", "xyz"], []),
                ("/tmp/nvflare/site-1", [], ["fed_client_config.conf"]),
                ("/tmp/nvflare/xyz", [], ["hello.txt"]),
            ]
        else:
            return []

    @pytest.mark.parametrize(
        "init_file_path, search_dirs, expected_loc, expected_fmt",
        [
            ["fed_client_config.json", ["."], "./fed_client_config.conf", ConfigFormat.PYHOCON],
            [
                "fed_client_config.json",
                ["/tmp/nvflare"],
                "/tmp/nvflare/site-1/fed_client_config.conf",
                ConfigFormat.PYHOCON,
            ],
        ],
    )
    def test_config_search(self, init_file_path, search_dirs, expected_loc, expected_fmt):
        import os

        fn = os.walk
        os.walk = MagicMock(side_effect=self._walk_result)
        fmt, location = ConfigFactory.search_config_format(init_file_path, search_dirs)
        assert fmt == expected_fmt
        assert location == expected_loc
        # restore function to avoid unexpected side-effect
        os.walk = fn

    @pytest.mark.parametrize(
        "init_file_path, expected, namelist",
        [
            [
                "config_fed_client.json",
                True,
                [
                    "user_email_match/meta.json",
                    "user_email_match/app/",
                    "user_email_match/app/custom/",
                    "user_email_match/app/config/",
                    "user_email_match/app/custom/local_psi.py",
                    "user_email_match/app/config/config_fed_client.conf",
                    "user_email_match/app/config/config_fed_server.conf",
                    "user_email_match/meta.json",
                    "user_email_match/app/",
                    "user_email_match/app/custom/",
                    "user_email_match/app/config/",
                    "user_email_match/app/custom/local_psi.py",
                    "user_email_match/app/config/config_fed_client.conf",
                    "user_email_match/app/config/config_fed_server.conf",
                    "user_email_match/meta.json",
                    "user_email_match/app/",
                    "user_email_match/app/custom/",
                    "user_email_match/app/config/",
                    "user_email_match/app/custom/local_psi.py",
                    "user_email_match/app/config/config_fed_client.conf",
                    "user_email_match/app/config/config_fed_server.conf" "user_email_match/meta.json",
                    "user_email_match/app/",
                    "user_email_match/app/custom/",
                    "user_email_match/app/config/",
                    "user_email_match/app/custom/local_psi.py",
                    "user_email_match/app/config/config_fed_client.conf",
                    "user_email_match/app/config/config_fed_server.conf",
                ],
            ]
        ],
    )
    def test_match_config(self, init_file_path, expected, namelist):
        def match(parent, config_path: str):
            import os

            full_path = os.path.join("user_email_match/app/config", config_path)
            return full_path in namelist

        assert expected == ConfigFactory.match_config(None, init_file_path, match)
