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
from pyhocon import ConfigFactory as CF

from nvflare.fuel.utils.config import ConfigFormat
from nvflare.fuel_opt.utils.pyhocon_loader import PyhoconConfig, PyhoconLoader


class TestPyHoconConfig:
    def return_conf(self, file_name):
        if file_name == "test.conf":
            x = """config {
                            a {
                                a1 = 1
                                a2 = 2
                            }
                            b =  1
                            c = hi
                            d = [1,2]
            } """
        else:
            x = """config {
                        a =  {
                            a1 =  2
                            a2 =  4
                        }
                        b = 2
                        c = hello
                        d = [2,4]
            } """

        return CF.parse_string(x)

    def test_config_loader(self):
        loader = PyhoconLoader()
        assert loader.get_format() == ConfigFormat.PYHOCON
        loader._from_file = self.return_conf
        dicts = {
            "config": {
                "a": {
                    "a1": 200,
                },
                "c": "hello",
                "d": [200, 400, 500],
                "e1": "Yes",
                "e2": "True",
                "e3": "NO",
            }
        }

        config = loader.load_config("test.conf")
        assert config.get_format() == ConfigFormat.PYHOCON
        conf = config.get_native_conf()
        print("conf=", conf)
        conf = conf.get_config("config")
        assert config is not None
        assert conf.get_config("a").get_int("a1") == 1
        assert conf.get_int("a.a1") == 1
        assert conf.get_int("b") == 1
        assert conf.get_string("c") == "hi"
        assert conf.get_list("d") == [1, 2]
        assert conf.get_string("e4", None) is None

        with pytest.raises(Exception):
            assert conf.get_string("d") == [1, 2]

        with pytest.raises(Exception):
            assert conf.get_string("d") == 1

        assert PyhoconConfig(CF.from_dict(conf.get("a"))).to_dict() == {"a1": 1, "a2": 2}
        with pytest.raises(Exception):
            assert conf.get_int("a") == 1

    def test_load_config_from_dict(self):
        loader = PyhoconLoader()
        assert loader.get_format() == ConfigFormat.PYHOCON
        dicts = {
            "config": {
                "a": {
                    "a1": 200,
                },
                "c": "hello",
                "d": [200, 400, 500],
                "e1": "Yes",
                "e2": "True",
                "e3": "NO",
            }
        }

        config = loader.load_config_from_dict(dicts)
        assert config.get_format() == ConfigFormat.PYHOCON
        assert config is not None
        assert config.to_dict() == dicts
        assert config.get_format() == ConfigFormat.PYHOCON

    def test_load_config_from_str(self):
        loader = PyhoconLoader()
        assert loader.get_format() == ConfigFormat.PYHOCON
        dicts = {
            "config": {
                "a": {
                    "a1": 200,
                },
                "c": "hello",
                "d": [200, 400, 500],
                "e1": "Yes",
                "e2": "True",
                "e3": "NO",
            }
        }

        config = loader.load_config_from_dict(dicts)
        config = loader.load_config_from_str(config.to_str())
        assert config is not None
        assert config.to_dict() == dicts
        assert config.get_format() == ConfigFormat.PYHOCON
