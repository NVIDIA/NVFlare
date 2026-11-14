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

"""In-process Slurm CP host dispatch assertion with the scheduler boundary mocked."""

from unittest.mock import Mock

from nvflare.apis.task_launcher_spec import TaskLaunchRequest
from nvflare.app_opt.job_launcher.slurm.task_launcher import WORKER_MODULE, _SlurmTaskBackend


def assert_physical_task_dispatch(handle, root, monkeypatch):
    backend = handle.owner.backend
    assert type(backend) is _SlurmTaskBackend
    request = TaskLaunchRequest(
        "job-1",
        "site-1",
        "task-1",
        "attempt-1",
        ("/shared/python", "-m", WORKER_MODULE, "--bootstrap", str(root / "bootstrap.json"), "--parent_pid", "1"),
        cwd=str(root),
    )
    # Bootstrap validation has its own tests. This regression guards the CP
    # dispatch method: a CJ proxy cannot reach the native manager boundary.
    monkeypatch.setattr(backend, "_materialize_staged", Mock(return_value=request))
    backend.manager._initialized = True
    boundary = Mock(return_value=object())
    monkeypatch.setattr(backend.manager, "launch_task", boundary)
    result = backend.launch_task(request)
    boundary.assert_called_once_with(request, backend._binding.resources)
    assert result is boundary.return_value
