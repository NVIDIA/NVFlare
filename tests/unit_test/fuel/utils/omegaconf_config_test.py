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

import yaml

from nvflare.fuel.utils.config import ConfigFormat
from nvflare.fuel_opt.utils.omegaconf_loader import OmegaConfLoader


class TestOmegaConfConfig:
    def return_conf(self, file_name):
        if file_name == "test.yml":
            x = """
                a :
                    a1 : 1
                    a2 : 2
                b :  1
                c : "hi"
                d : 
                    - 1
                    - 2 
                """
        else:
            x = """
                a:
                    a1 : 2
                    a2 :  4
                b : 2
                c : "hello"
                d : 
                    - 2
                    - 4
            """

        return yaml.safe_load(x)

    def test_config_loader(self):
        loader = OmegaConfLoader()
        assert loader.get_format() == ConfigFormat.OMEGACONF
        loader._from_file = self.return_conf
        dicts = {
            "a": {
                "a1": 200,
            },
            "c": "hello",
            "d": [200, 400, 500],
            "e1": "Yes",
            "e2": "True",
            "e3": "NO",
        }

        config = loader.load_config("test.yml")
        assert config.get_format() == ConfigFormat.OMEGACONF
        conf = config.get_native_conf()
        print(config.to_dict())
        assert config is not None
        assert conf["a"]["a1"] == 1
        assert conf["b"] == 1
        assert conf["c"] == "hi"
        assert conf["d"] == [1, 2]
        assert conf.get("e4", None) is None

        a_dict = conf["a"]
        assert a_dict == {"a1": 1, "a2": 2}

    def test_load_config_from_dict(self):
        loader = OmegaConfLoader()
        assert loader.get_format() == ConfigFormat.OMEGACONF
        dicts = {
            "a": {
                "a1": 200,
            },
            "c": "hello",
            "d": [200, 400, 500],
            "e1": "Yes",
            "e2": "True",
            "e3": "NO",
        }

        config = loader.load_config_from_dict(dicts)
        assert config.get_format() == ConfigFormat.OMEGACONF
        assert config is not None
        assert config.to_dict() == dicts

    def test_load_config_from_str(self, monkeypatch):
        monkeypatch.setenv("NVFLARE_YAML_CANARY", "synthetic-private-value")
        loader = OmegaConfLoader()
        assert loader.get_format() == ConfigFormat.OMEGACONF
        dicts = {
            "a": {
                "a1": 200,
            },
            "c": "hello",
            "d": [200, 400, 500],
            "e1": "Yes",
            "e2": "True",
            "e3": "NO",
        }

        dicts.update(copy="${a}", password="${oc.env:NVFLARE_YAML_CANARY}", items=[True, None, "on"])
        config = loader.load_config_from_dict(dicts)
        config = loader.load_config_from_str("# YAML comment\n" + config.to_str())
        assert config is not None
        assert config.to_dict() == dicts
        assert config.to_dict(resolve=False) == dicts
        assert config.get_format() == ConfigFormat.OMEGACONF
        assert loader.load_config_from_str(config.to_str({"value": "${a}"})).to_dict() == {"value": "${a}"}
