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
import pytest

import nvflare.client as flare
from nvflare.client import api


class TestPackageExports:
    """The public Client API surface named by the rank contract must be importable from the package."""

    @pytest.mark.parametrize(
        "name",
        [
            "init",
            "receive",
            "send",
            "log",
            "shutdown",
            "is_running",
            "get_task_name",
            "is_train",
            "is_evaluate",
            "is_submit_model",
        ],
    )
    def test_control_api_is_the_api_function(self, name):
        # assert identity, not mere presence: a refactor that rebinds one of these
        # names to a different object (or leaves a transitively-imported submodule
        # attribute shadowing it) must fail here.
        assert hasattr(flare, name), f"nvflare.client must export {name}"
        assert getattr(flare, name) is getattr(api, name), f"nvflare.client.{name} must be nvflare.client.api.{name}"
