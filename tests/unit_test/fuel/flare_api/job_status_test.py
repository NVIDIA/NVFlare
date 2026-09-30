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

from nvflare.fuel.flare_api.job_status import is_terminal_job_status, job_status_outcome


@pytest.mark.parametrize(
    "status,outcome",
    [
        ("FINISHED:COMPLETED", "completed"),
        ("FINISHED_OK", "completed"),
        ("FINISHED:CAN_NOT_SCHEDULE", "not_scheduled"),
        ("FINISHED:EXECUTION_EXCEPTION", "failed"),
        ("FINISHED:ABORTED", "aborted"),
        ("ABORTED", "aborted"),
        ("RUNNING", None),
        (None, None),
    ],
)
def test_job_status_outcome(status, outcome):
    assert job_status_outcome(status) == outcome
    assert is_terminal_job_status(status) is (outcome is not None)
