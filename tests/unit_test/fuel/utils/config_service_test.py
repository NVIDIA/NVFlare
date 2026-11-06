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
import json
import os
from argparse import Namespace

import pytest

from nvflare.fuel.utils.config_service import ConfigService


class TestConfigService:
    def test_refresh_config_path_keeps_registered_state(self, tmp_path, monkeypatch):
        previous = tmp_path / "previous"
        site_local = tmp_path / "local"
        previous.mkdir()
        site_local.mkdir()
        for name, value in (
            ("_sections", {}),
            ("_config_path", []),
            ("_cmd_args", None),
            ("_var_dict", None),
            ("_var_values", {}),
        ):
            monkeypatch.setattr(ConfigService, name, value)

        ConfigService.initialize(
            section_files={},
            config_path=[str(previous)],
            parsed_args=Namespace(command_option="command"),
            var_dict={"custom_option": "override"},
        )
        ConfigService.add_section("existing", {"marker": "kept"})
        assert ConfigService.get_int_var("max_message_size", default=1024) == 1024

        ConfigService.refresh_config_path([str(site_local), str(previous)])

        assert ConfigService._config_path == [str(site_local), str(previous)]
        assert ConfigService.get_section("existing") == {"marker": "kept"}
        assert ConfigService.get_str_var("command_option") == "command"
        assert ConfigService.get_str_var("custom_option") == "override"
        assert ConfigService.get_int_var("max_message_size", conf={"max_message_size": 2048}) == 2048

    @pytest.mark.parametrize(
        "config, var_name, value",
        [
            [{"x1": 1}, "x1", 1],
            [{"x1": 2.5}, "x1", 2],
            [{"x1": "100"}, "x1", 100],
        ],
    )
    def test_get_int(self, config, var_name, value):
        ConfigService.reset()
        v = ConfigService.get_int_var(var_name, conf=config)
        assert v == value

    @pytest.mark.parametrize(
        "config, var_name, value",
        [
            [{}, "x1", 100],
        ],
    )
    def test_get_int_default(self, config, var_name, value):
        ConfigService.reset()
        v = ConfigService.get_int_var(var_name, conf=config, default=value)
        assert v == value

    @pytest.mark.parametrize(
        "config, var_name",
        [
            [{"x1": "abc"}, "x1"],
        ],
    )
    def test_get_int_error(self, config, var_name):
        ConfigService.reset()
        with pytest.raises(ValueError):
            ConfigService.get_int_var(var_name, conf=config)

    @pytest.mark.parametrize(
        "config, var_name, value",
        [
            [{"x1": 1}, "x1", 1.0],
            [{"x1": 2.4}, "x1", 2.4],
            [{"x1": "100.5"}, "x1", 100.5],
        ],
    )
    def test_get_float(self, config, var_name, value):
        ConfigService.reset()
        v = ConfigService.get_float_var(var_name, conf=config)
        assert v == value

    @pytest.mark.parametrize(
        "config, var_name, value",
        [
            [{}, "x1", 100.0],
        ],
    )
    def test_get_float_default(self, config, var_name, value):
        ConfigService.reset()
        v = ConfigService.get_float_var(var_name, conf=config, default=value)
        assert v == value

    @pytest.mark.parametrize(
        "config, var_name",
        [
            [{"x1": "abc"}, "x1"],
        ],
    )
    def test_get_float_error(self, config, var_name):
        ConfigService.reset()
        with pytest.raises(ValueError):
            ConfigService.get_float_var(var_name, conf=config)

    @pytest.mark.parametrize(
        "config, var_name, value",
        [
            [{"x1": 1}, "x1", "1"],
            [{"x1": "test"}, "x1", "test"],
            [{"x1": 100.5}, "x1", "100.5"],
        ],
    )
    def test_get_str(self, config, var_name, value):
        ConfigService.reset()
        v = ConfigService.get_str_var(var_name, conf=config)
        assert v == value

    @pytest.mark.parametrize(
        "config, var_name, value",
        [
            [{}, "x1", "test"],
        ],
    )
    def test_get_str_default(self, config, var_name, value):
        ConfigService.reset()
        v = ConfigService.get_str_var(var_name, conf=config, default=value)
        assert v == value

    @pytest.mark.parametrize(
        "config, var_name, value",
        [
            [{"x1": {"a": 1, "b": 2}}, "x1", {"a": 1, "b": 2}],
            [{"x2": '{"a": 1, "b": 2}'}, "x2", {"a": 1, "b": 2}],
        ],
    )
    def test_get_dict(self, config, var_name, value):
        ConfigService.reset()
        v = ConfigService.get_dict_var(var_name, conf=config)
        assert v == value

    @pytest.mark.parametrize(
        "config, var_name, value",
        [
            [{}, "x1", {"a": 1, "b": 2}],
            [{}, "x1", None],
        ],
    )
    def test_get_dict_default(self, config, var_name, value):
        ConfigService.reset()
        v = ConfigService.get_dict_var(var_name, conf=config, default=value)
        assert v == value

    @pytest.mark.parametrize(
        "config, var_name",
        [
            [{"x1": "abc"}, "x1"],
            [{"x1": 1}, "x1"],
            [{"x1": True}, "x1"],
        ],
    )
    def test_get_dict_error(self, config, var_name):
        ConfigService.reset()
        with pytest.raises(ValueError):
            ConfigService.get_dict_var(var_name, conf=config)

    @pytest.mark.parametrize(
        "config, var_name, value",
        [
            [{"x": 1}, "x", True],
            [{"x": -2}, "x", True],
            [{"x": 0}, "x", False],
            [{"x": "y"}, "x", True],
            [{"x": "Y"}, "x", True],
            [{"x": "yes"}, "x", True],
            [{"x": "Yes"}, "x", True],
            [{"x": "t"}, "x", True],
            [{"x": "true"}, "x", True],
            [{"x": "True"}, "x", True],
            [{"x": "y"}, "x", True],
            [{"x": "1"}, "x", True],
            [{"x": "f"}, "x", False],
            [{"x": "false"}, "x", False],
            [{"x": "anythingelse"}, "x", False],
        ],
    )
    def test_get_bool(self, config, var_name, value):
        ConfigService.reset()
        v = ConfigService.get_bool_var(var_name, conf=config)
        assert v == value

    @pytest.mark.parametrize(
        "config, var_name, value",
        [
            [{}, "x1", True],
            [{}, "x1", False],
            [{}, "x1", None],
        ],
    )
    def test_get_bool_default(self, config, var_name, value):
        ConfigService.reset()
        v = ConfigService.get_bool_var(var_name, conf=config, default=value)
        assert v == value

    @pytest.mark.parametrize(
        "config, var_name",
        [
            [{"x1": 3.45}, "x1"],
            [{"x1": []}, "x1"],
            [{"x1": {}}, "x1"],
        ],
    )
    def test_get_bool_error(self, config, var_name):
        ConfigService.reset()
        with pytest.raises(ValueError):
            ConfigService.get_bool_var(var_name, conf=config)

    @pytest.mark.parametrize(
        "config, var_name, value",
        [
            [{}, "x", 1],
            [{}, "x", 2],
            [{}, "x", 100],
        ],
    )
    def test_get_int_from_env(self, config, var_name, value):
        ConfigService.reset()
        env = os.environ
        env_var_name = f"NVFLARE_{var_name}".upper()
        env[env_var_name] = str(value)
        v = ConfigService.get_int_var(var_name, conf=config)
        assert v == value


def _write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")


class TestLoadConfigDict:
    def test_only_json_extensions_in_priority_order(self):
        assert ConfigService.CONFIG_EXTENSIONS == (".json", ".json.default")

    def test_json_preferred_over_default_in_same_dir(self, tmp_path):
        _write(tmp_path / "x.json.default", {"v": "default"})
        _write(tmp_path / "x.json", {"v": "json"})

        assert ConfigService.load_config_dict("x.json.default", [str(tmp_path)]) == {"v": "json"}

    def test_nested_json_preferred_over_top_level_default(self, tmp_path):
        _write(tmp_path / "x.json.default", "{not json")
        _write(tmp_path / "nested" / "x.json", {"v": "nested"})

        assert ConfigService.load_config_dict("x.json", [str(tmp_path)]) == {"v": "nested"}

    def test_default_used_when_no_json(self, tmp_path):
        _write(tmp_path / "x.json.default", {"v": "default"})

        assert ConfigService.load_config_dict("x.json", [str(tmp_path)]) == {"v": "default"}

    def test_search_dirs_searched_in_order(self, tmp_path):
        _write(tmp_path / "a" / "x.json.default", {"v": "a"})
        _write(tmp_path / "b" / "x.json", {"v": "b"})

        assert ConfigService.load_config_dict("x.json", [str(tmp_path / "a"), str(tmp_path / "b")]) == {"v": "a"}

    def test_defaults_to_parent_dir_of_file_path(self, tmp_path):
        _write(tmp_path / "nested" / "x.json.default", {"v": 1})

        assert ConfigService.load_config_dict(str(tmp_path / "x.json")) == {"v": 1}
        assert ConfigService.load_config_dict(str(tmp_path / "x")) == {"v": 1}

    def test_hocon_and_yaml_files_are_ignored(self, tmp_path):
        _write(tmp_path / "x.conf", "v = 1")
        _write(tmp_path / "x.yml", "v: 1")

        assert ConfigService.load_config_dict("x.json", [str(tmp_path)], raise_exception=False) is None
        with pytest.raises(FileNotFoundError, match=r"x\[\.json\|\.json\.default\]"):
            ConfigService.load_config_dict("x.json", [str(tmp_path)])

    def test_malformed_config_raises(self, tmp_path):
        _write(tmp_path / "x.json", "{not json")

        with pytest.raises(json.JSONDecodeError):
            ConfigService.load_config_dict("x.json", [str(tmp_path)], raise_exception=False)

    def test_load_configuration_searches_config_path(self, tmp_path, monkeypatch):
        _write(tmp_path / "site" / "x.json", {"v": "site"})
        monkeypatch.setattr(ConfigService, "_config_path", [str(tmp_path)])

        assert ConfigService.load_configuration("x.json") == {"v": "site"}
        assert ConfigService.load_configuration("missing.json") is None
