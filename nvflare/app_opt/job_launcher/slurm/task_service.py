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

"""Slurm-specific CP ownership and shared-workspace policy for task control."""

import os

from nvflare.apis.job_launcher_spec import JobReturnCode
from nvflare.apis.task_launcher_spec import TaskLauncherError
from nvflare.private.fed.task_worker.task_service import CPRemoteTaskHandle, CPTaskClient, CPTaskService
from nvflare.private.fed.task_worker.task_service import _Job as _BaseJob

ENV_TASK_OWNER = "NVFLARE_SLURM_TASK_OWNER"
ENV_TASK_TOKEN = "NVFLARE_SLURM_TASK_TOKEN"
_CHANNEL = "slurm_task_control"


def _within(root, path):
    if not isinstance(path, str) or not os.path.isabs(path):
        raise ValueError("task paths must be absolute")
    if os.path.commonpath((root, os.path.realpath(path))) != root:
        raise ValueError("task path is outside the owning job directory")


class _Job(_BaseJob):
    pending_id = "slurm:pending"
    unit_name = "allocation"
    cancel_after_lost_launch = True

    def _cj_is_dead(self, handle):
        return handle.poll() != JobReturnCode.UNKNOWN


class SlurmTaskService(CPTaskService):
    channel = _CHANNEL
    backend_name = "Slurm"
    service_key = "slurm"
    job_class = _Job
    allow_complete = True
    closed_unknown_cancel_ok = True
    unfinished_task_error = "job already owns an unfinished task"

    def _check_path(self, job, path):
        _within(job.backend._binding.run_dir, path)


class SlurmRemoteTaskHandle(CPRemoteTaskHandle):
    pending_id = "slurm:pending"

    def result_received(self):
        self.observe(self.client.result_received(self.request.task_id, self.request.attempt_id))


class SlurmTaskClient(CPTaskClient):
    channel = "slurm_task_control"
    backend_name = "Slurm"
    owner_env = ENV_TASK_OWNER
    token_env = ENV_TASK_TOKEN
    handle_class = SlurmRemoteTaskHandle

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pending_completions = set()

    def _before_launch(self):
        # A lost completion reply must never allow another physical launch.
        for task_id, attempt_id in tuple(self._pending_completions):
            try:
                self.result_received(task_id, attempt_id)
            except Exception as error:
                raise TaskLauncherError("CP completion acknowledgement unavailable; no new task submitted") from error

    def result_received(self, task_id, attempt_id):
        identity = (task_id, attempt_id)
        self._pending_completions.add(identity)
        result = self.call("complete", task_id=task_id, attempt_id=attempt_id)
        self._pending_completions.discard(identity)
        return result
