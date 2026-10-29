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
            ["fed_client_config", "fed_client_config"],
            ["/abc/efg/fed_client_config", "fed_client_config"],
            ["comm_config", "comm_config"],
        ],
    )
    def test_get_file_basename(self, init_file_path, file_basename):
        assert file_basename == ConfigFactory.get_file_basename(init_file_path)

    def test_fmt_to_loader(self):
        fmt2loaders = ConfigFactory._fmt2Loader

        config_dicts = dict(
            example="1",
            version=2,
            a=dict(a=2, b=3),
            databases=dict(
                mysql=dict(url="mysql://xxx", user="admin"), postgres=dict(url="postgres://xxx", user="root")
            ),
            arrs=[1, 2, 5, 6],
        )

        for fmt, loader in fmt2loaders.items():
            assert isinstance(loader, ConfigLoader)
            assert loader.get_format() == fmt
            config = loader.load_config_from_dict(config_dicts)
            assert config.to_dict() == config_dicts

    @pytest.mark.parametrize("suffix", [".yml", ".yaml", ".yml.default", ".yaml.default"])
    def test_yaml_config_overrides_json_defaults(self, tmp_path, suffix):
        (tmp_path / f"resources{suffix}").write_text("setting: custom\n")
        if not suffix.endswith(".default"):
            (tmp_path / "resources.json.default").write_text('{"setting": "default"}')

        config = ConfigFactory.load_config("resources.json", [str(tmp_path)])
        assert config.to_dict() == {"setting": "custom"}
        assert config.get_location() == str(tmp_path / f"resources{suffix}")

    @pytest.mark.parametrize(
        "source",
        [
            "value: [unterminated",
            "value: !include private.yaml",
            "value: !!python/object/apply:os.getenv [PRIVATE_VALUE]",
        ],
    )
    def test_yaml_rejects_invalid_syntax_and_object_constructors(self, source):
        with pytest.raises(ValueError, match="Invalid YAML configuration"):
            ConfigFactory.get_config_loader(ConfigFormat.OMEGACONF).load_config_from_str(source)

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
            ["fed_client_config.json", ["."], "./fed_client_config.conf", ConfigFormat.JSON],
            [
                "fed_client_config.json",
                ["/tmp/nvflare"],
                "/tmp/nvflare/site-1/fed_client_config.conf",
                ConfigFormat.JSON,
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
