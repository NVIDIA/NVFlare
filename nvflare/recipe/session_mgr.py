# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import os
import tempfile
from typing import Dict, Optional

from nvflare.fuel.flare_api.api_spec import MonitorReturnCode
from nvflare.fuel.flare_api.flare_api import Session, new_secure_session
from nvflare.job_config.api import FedJob


def _job_monitor_callback(session: Session, job_id: str, job_meta, *cb_args, **cb_kwargs) -> bool:
    """Shared callback to print job meta during monitoring."""
    # cb_run_counter is a dictionary that is passed to the callback and is used to keep track of the number of times the callback has been called
    if cb_kwargs["cb_run_counter"]["count"] == 0:
        print("Job ID: ", job_id)
        print("Job Meta: ", job_meta)

    if job_meta["status"] == "RUNNING":
        print(".", end="")
    else:
        print("\n" + str(job_meta))

    cb_kwargs["cb_run_counter"]["count"] += 1
    return True


class SessionManager:
    """Centralized session management for POC and Production environments.

    Handles all session operations including job submission, monitoring, and lifecycle management.
    Implements session caching to avoid multiple login/logout cycles.
    """

    def __init__(self, session_params: Dict[str, any]):
        self.session_params = session_params

    def _get_session(self):
        """Context manager that provides a session, with optional caching."""
        sess = new_secure_session(**self.session_params)
        return sess

    def submit_job(self, job: FedJob) -> str:
        """Submit a job and return job ID."""
        sess = self._get_session()
        with tempfile.TemporaryDirectory() as temp_dir:
            job.export_job(temp_dir)
            job_path = os.path.join(temp_dir, job.name)
            job_id = sess.submit_job(job_path)
            sess.close()
            print(f"Submitted job '{job.name}' with ID: {job_id}")
            return job_id

    def get_job_status(self, job_id: str) -> Optional[str]:
        """Get the status of the job."""
        sess = self._get_session()
        status = sess.get_job_status(job_id)
        sess.close()
        return status

    def abort_job(self, job_id: str) -> None:
        """Abort the running job."""
        sess = self._get_session()
        msg = sess.abort_job(job_id)
        print(f"Job {job_id} aborted successfully with message: {msg}")
        sess.close()

    def get_job_result(self, job_id: str, timeout: float = 0.0) -> Optional[str]:
        """Get the result workspace of the job."""
        sess = self._get_session()
        cb_run_counter = {"count": 0}
        rc = sess.monitor_job(job_id, timeout=timeout, cb=_job_monitor_callback, cb_run_counter=cb_run_counter)
        print(f"job monitor done: {rc=}")
        if rc == MonitorReturnCode.JOB_FINISHED:
            result = sess.download_job_result(job_id)
            sess.close()
            return result
        elif rc == MonitorReturnCode.TIMEOUT:
            print(
                f"Job {job_id} did not complete within {timeout} seconds. "
                "Job is still running. Try calling get_result() again with a longer timeout."
            )
            sess.close()
            return None
        elif rc == MonitorReturnCode.ENDED_BY_CB:
            print(
                "Job monitoring was stopped early by callback. "
                "Result may not be available yet. Check job status and try again."
            )
            sess.close()
            return None
        else:
            raise RuntimeError(f"Unexpected monitor return code: {rc}")
