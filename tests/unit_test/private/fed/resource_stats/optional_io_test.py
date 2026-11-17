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

import threading
from unittest.mock import Mock

from nvflare.private.fed.resource_stats.optional_io import optional_io
from tests.resource_stats_io_test_utils import stalled_resource_io
from tests.timing_utils import join_thread


def test_blocked_io_returns_default_and_skips_new_operations_until_finished():
    operation = Mock()
    result = []
    with stalled_resource_io(operation, "read", value="late") as (entered, threads):
        caller = threads.start(lambda: result.append(optional_io(operation.read, default="missing")))
        assert entered.wait(10)
        join_thread(caller)
        assert result == ["missing"]
        assert optional_io(operation.next, default="busy") == "busy"
        operation.next.assert_not_called()
    assert result == ["missing"]
    assert optional_io(lambda: threading.current_thread().daemon) is True
    assert optional_io(lambda: 0, default=1) == 0


def test_io_exception_uses_default_and_releases_admission():
    assert optional_io(Mock(side_effect=OSError("unavailable")), default="missing") == "missing"
    assert optional_io(lambda: "ready") == "ready"
