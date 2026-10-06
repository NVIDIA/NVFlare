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

from nvflare.apis.task_launcher_spec import (
    TaskExecutionPhase,
    TaskExecutionStatus,
    TaskHandleSpec,
    TaskLauncherSpec,
    TaskLaunchRequest,
    TaskResourceRequest,
    TaskSettlementError,
)


@pytest.mark.parametrize("field, value", [("memory_mb", 0), ("memory_mb", True), ("gpu_count", -1), ("gpu_count", 0.5)])
def test_resource_request_rejects_invalid_memory_and_gpu(field, value):
    with pytest.raises(ValueError, match=field):
        TaskResourceRequest(**{field: value})


@pytest.mark.parametrize(
    "changes, error",
    [
        ({"argv": []}, ValueError),
        ({"argv": [""]}, ValueError),
        ({"argv": [42]}, ValueError),
        ({"argv": ["python", "nul\x00argument"]}, ValueError),
        ({"environment": {"BAD=NAME": "value"}}, ValueError),
        ({"environment": {"NAME": 1}}, ValueError),
        ({"environment": {"": "value"}}, ValueError),
        ({"environment": {"NAME": "nul\x00value"}}, ValueError),
        ({"environment": {"nul\x00name": "value"}}, ValueError),
        ({"cwd": ""}, ValueError),
        ({"cwd": "nul\x00directory"}, ValueError),
        ({"cwd": 42}, ValueError),
        ({"resources": {}}, TypeError),
    ],
)
def test_launch_request_rejects_invalid_process_inputs(changes, error):
    values = dict(job_id="job", site_name="site", task_id="task", attempt_id="attempt", argv=["python"])
    with pytest.raises(error):
        TaskLaunchRequest(**(values | changes))


def test_settlement_error_preserves_observed_status():
    status = TaskExecutionStatus(TaskExecutionPhase.TERMINAL, exit_code=0, settled=False)
    error = TaskSettlementError("still alive", status)
    assert str(error) == "still alive"
    assert error.status is status
    assert not status.succeeded


@pytest.mark.parametrize("contract", [TaskHandleSpec, TaskLauncherSpec])
def test_incomplete_backend_cannot_be_instantiated(contract):
    with pytest.raises(TypeError, match="abstract"):
        contract()


@pytest.mark.parametrize("name", ["request", "execution_id", "poll", "cancel", "wait_for_settlement"])
def test_handle_contract_has_no_fallback_implementation(name):
    member = getattr(TaskHandleSpec, name)
    implementation = member.fget if isinstance(member, property) else member
    with pytest.raises(NotImplementedError):
        implementation(None)


def test_launcher_contract_has_no_fallback_implementation():
    with pytest.raises(NotImplementedError):
        TaskLauncherSpec.launch_task(None, None)
