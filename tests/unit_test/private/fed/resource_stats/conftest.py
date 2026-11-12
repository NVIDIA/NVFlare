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

from nvflare.private.fed.resource_stats import collector, handoff


@pytest.fixture(autouse=True)
def direct_filesystem_observations(monkeypatch):
    def observe(run_dir):
        return {
            "workspace_filesystem": collector.observe_workspace_filesystem(run_dir),
            "retained_content": collector.observe_retained_content(run_dir),
        }

    monkeypatch.setattr(collector, "observe_filesystem", observe)


@pytest.fixture
def make_handoff():
    def make(*, complete=False):
        record = {
            "duration_seconds": 10,
            "internal_version": handoff.INTERNAL_HANDOFF_VERSION,
            "kind": handoff.INTERNAL_HANDOFF_KIND,
            "resource_capacity": {"cpu": {"units": 2}, "memory": {"bytes": 4096}, "gpu": {"count": 0}},
            "resource_time": {
                "cpu": {"unit_seconds": 20},
                "memory": {"byte_seconds": 40960},
                "gpu": {"instance_seconds": 0},
            },
            "workspace_filesystem": {"capacity_bytes": 1024},
            "resource_utilization": {
                "cpu_consumed": {"cpu_seconds": 2},
                "retained_content": {"bytes": 128},
                "message_traffic": {"sent_bytes": 512, "received_bytes": 128},
            },
            "issues": [],
        }
        if not complete:
            for container, names in (
                (record, ("resource_capacity", "resource_time")),
                (record["resource_utilization"], ("cpu_consumed", "retained_content", "message_traffic")),
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
