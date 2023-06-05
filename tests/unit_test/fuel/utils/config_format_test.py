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
from typing import List

from nvflare.fuel.utils.config import ConfigFormat


class TestConfigFormat:

    def test_config_exts(self):
        exts = ConfigFormat._config_format_extensions()
        assert (exts.get(ConfigFormat.JSON.name) == [".json", ".json.default"])
        assert (exts.get(ConfigFormat.PYHOCON.name) == [".conf", ".conf.default"])
        assert (exts.get(ConfigFormat.OMEGACONF.name) == [".yml", ".yml.default"])

    def test_get_exts(self):
        exts = ConfigFormat.get_extensions(ConfigFormat.JSON.name)
        assert exts == [".json", ".json.default"]

        exts = ConfigFormat.get_extensions(ConfigFormat.PYHOCON.name)
        assert exts == [".conf", ".conf.default"]

        exts = ConfigFormat.get_extensions(ConfigFormat.OMEGACONF.name)
        assert exts == [".yml", ".yml.default"]

    def test_config_exts2(self):
        config_exts = ConfigFormat.config_exts()
        assert config_exts == ".json|.json.default|.conf|.conf.default|.yml|.yml.default"

    def test_ordered_lookup_exts(self):
        xs: List[(ConfigFormat, str)] = ConfigFormat.ordered_search_extensions()
        assert (len(xs) == 6)
        exts = xs[:2]
        for fmt, f in exts:
            assert(fmt == ConfigFormat.JSON)
            assert(f.startswith(".json"))

        exts = xs[2:4]
        for fmt, f in exts:
            assert(fmt == ConfigFormat.PYHOCON)
            assert(f.startswith(".conf"))

        exts = xs[4:6]
        for fmt, f in exts:
            assert(fmt == ConfigFormat.OMEGACONF)
            assert(f.startswith(".yml"))
