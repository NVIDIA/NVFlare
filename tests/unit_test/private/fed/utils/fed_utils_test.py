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

from typing import List, Union

import pytest

from nvflare.apis.client import Client
from nvflare.private.fed.utils.fed_utils import get_target_names

TARGET_NAMES_TEST_CASES = [
    (
        [Client("site-1", None), Client("site-2", None), Client("site-3", None)],
        ["site-1", "site-2", "site-3"],
    ),
    (
        ["site-1", "site-2", "site-3"],
        ["site-1", "site-2", "site-3"],
    ),
    (
        [Client("site-1", None), "site-2", Client("site-3", None)],
        ["site-1", "site-2", "site-3"],
    ),
    (
        [Client("", None), "", Client("site-3", None)],
        ["site-3"],
    ),
]


class TestFedUtils:
    @pytest.mark.parametrize("targets, target_names", TARGET_NAMES_TEST_CASES)
    def test_get_target_names(self, targets: List[Union[str, Client]], target_names: List[str]):
        assert get_target_names(targets) == target_names
