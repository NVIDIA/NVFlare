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

import pytest

from nvflare.fuel.utils.config_service import ConfigService


class TestConfigService:
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
