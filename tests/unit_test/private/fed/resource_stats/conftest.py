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

"""Keep filesystem policy tests inside one interpreter."""

import pytest

from nvflare.private.fed.resource_stats import collector, filesystem_probe


@pytest.fixture(autouse=True)
def direct_filesystem_observations(monkeypatch):
    def observe(run_dir):
        return {
            "workspace_filesystem": collector.observe_workspace_filesystem(run_dir),
            "retained_content": collector.observe_retained_content(run_dir),
        }

    monkeypatch.setattr(filesystem_probe, "observe_filesystem_isolated", observe)
