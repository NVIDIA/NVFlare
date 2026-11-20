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

"""Shared resource reporting fixtures."""

import pytest

from nvflare.private.fed.resource_stats.measurements import participant_measurements
from tests.resource_stats_report_test_utils import _participant_summary


@pytest.fixture
def make_handoff():
    def make(*, complete=False):
        record = participant_measurements(_participant_summary())
        if not complete:
            for container, names in (
                (record, ("resource_capacity", "resource_time")),
                (record["resource_utilization"], ("cpu_consumed", "message_traffic")),
            ):
                for name in names:
                    container[name] = {"status": "unavailable"}
                    record["issues"].append({"metric": name, "code": "observation_incomplete"})
            record["issues"].sort(key=lambda issue: issue["metric"])
        return record

    return make


@pytest.fixture
def usage_reader():
    return lambda *values: iter(tuple(float(item) for item in value) for value in values).__next__
