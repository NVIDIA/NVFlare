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

from nvflare.fuel.utils.config import ConfigFormat


class TestConfigFormat:
    def test_only_json_is_a_config_format(self):
        assert list(ConfigFormat.config_ext_formats().items()) == [
            (".json", ConfigFormat.JSON),
            (".json.default", ConfigFormat.JSON),
        ]
        assert ConfigFormat.extensions(target_fmt=ConfigFormat.JSON) == [".json", ".json.default"]
