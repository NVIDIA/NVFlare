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
"""DIRECTORY-WIDE fixtures: the autouse fixture below applies to EVERY test in
tests/unit_test/fuel/f3/streaming/, including tests that never touch confirms.
A new test here never exercises the real ConfigService read of the confirm
kill-switch; patch _receiver_confirm_cached explicitly if you need the legacy
(confirm-off) path or the real config read."""

from unittest.mock import patch

import pytest

from nvflare.fuel.f3.streaming import download_service as ds_module


@pytest.fixture(autouse=True)
def confirm_switch_on():
    """Pin the receiver-confirm kill-switch ON for all streaming tests.

    Individual tests patch it OFF explicitly; legacy-path tests are unaffected (their
    requests carry no capability key, so the switch is never consulted for them).
    """
    with patch.object(ds_module, "_receiver_confirm_cached", True):
        yield
