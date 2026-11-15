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


"""The deployment contracts require Linux/GNU tools, with no live services."""

import os
import sys
from pathlib import Path

import pytest


def pytest_collection_modifyitems(items):
    root = Path(__file__).parent
    for item in items:
        if root not in item.path.parents:
            continue
        item.add_marker(pytest.mark.timeout(180))
        if sys.platform != "linux":
            item.add_marker(pytest.mark.skip(reason="CoCo CLI contracts require Linux/GNU userland"))


@pytest.fixture(autouse=True)
def use_test_interpreter(monkeypatch):
    # Shell helpers must use the same dependencies as pytest. Each fake-tool
    # fixture can still prepend its own bin directory afterwards.
    monkeypatch.setenv("PATH", str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""))
