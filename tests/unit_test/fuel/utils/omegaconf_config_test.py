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
from omegaconf import OmegaConf

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

        return OmegaConf.create(x)

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
        assert conf.a.a1 == 1
        with pytest.raises(Exception):
            assert conf["a.a1"] == 1
        assert conf["a"]["a1"] == 1
        assert conf.b == 1
        assert conf.c == "hi"
        assert conf.d == [1, 2]
        assert conf.get("e4", None) is None

        a_dict = OmegaConf.to_container(conf.a, resolve=True)
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

    def test_load_config_from_str(self):
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
        config = loader.load_config_from_str(config.to_str())
        assert config is not None
        assert config.to_dict() == dicts
        assert config.get_format() == ConfigFormat.OMEGACONF
