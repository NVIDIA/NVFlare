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
import json

import pytest

from nvflare.fuel.utils.config import ConfigFormat
from nvflare.fuel.utils.json_config_loader import JsonConfigLoader


class TestJsonConfig:

    def return_dict(self, file_name):
        if file_name == "test.json":
            return {
                "a": {
                    "a1": 1,
                    "a2": 2,
                },
                "b": 1,
                "c": "hi",
                "d": [1,2]
            }
        else:
            return {
                "a": {
                    "a1": 2,
                    "a2": 4,
                },
                "b": 2,
                "c": "hello",
                "d": [2,4]
            }

    def test_json_loader(self):
        loader = JsonConfigLoader()
        loader._from_file = self.return_dict
        dicts = {
            "a": {
                "a1": 200,
            },
            "c": "hello",
            "d": [200,400, 500],
            "e1": "Yes",
            "e2": "True",
            "e3": "NO",
        }

        config = loader.load_config("test.json", "default_test.json", dicts)
        assert config is not None
        assert config.get_config("a").get_int("a1") == 200
        assert config.get_int("b") == 1
        assert config.get_str("c") == "hello"
        assert config.get_list("d") == [200,400,500]
        assert config.get_str("e1") == "Yes"
        assert config.get_str("e2") == "True"
        assert config.get_str("e3") == "NO"
        assert config.get_str("e4") is None
        assert config.get_bool("e1") is True
        assert config.get_bool("e2") is True
        assert config.get_bool("e3") is False

        assert config.get_format() == ConfigFormat.JSON

        with pytest.raises(Exception):
            assert config.get_str("d") == [200,400,500]

        with pytest.raises(Exception):
            assert config.get_str("b") == 1

        with pytest.raises(Exception):
            assert config.get_int("a") == 1

        config = loader.load_config_from_dict(dicts)
        assert config is not None
        assert config.get_conf() == dicts
        assert config.get_format() == ConfigFormat.JSON

        config = loader.load_config_from_str(json.dumps(dicts))
        assert config is not None
        assert config.get_conf() == dicts
        assert config.get_format() == ConfigFormat.JSON




