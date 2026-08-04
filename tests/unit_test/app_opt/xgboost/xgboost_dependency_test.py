# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from configparser import ConfigParser
from pathlib import Path

import pytest


@pytest.mark.parametrize("extra", ["app_opt", "app_opt_cpu"])
def test_xgboost_dependency_excludes_release_without_column_split_support(extra):
    config = ConfigParser()
    config.read(Path(__file__).resolve().parents[4] / "setup.cfg")

    requirements = config["options.extras_require"][extra].splitlines()

    assert "xgboost<3.4" in requirements
