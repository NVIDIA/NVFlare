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
import time
from typing import Dict, Optional

from nvflare.fuel.flare_api.api_spec import MonitorReturnCode
from nvflare.fuel.flare_api.flare_api import Session, new_secure_session
from nvflare.fuel.utils.job_secret_scanner import warn_on_potential_secrets_in_job_dir
from nvflare.fuel.utils.log_utils import get_module_logger
from nvflare.job_config.api import FedJob


def _job_monitor_callback(session: Session, job_id: str, job_meta, *cb_args, **cb_kwargs) -> bool:
    """Show status changes and a periodic waiting message using existing job metadata."""
    state = cb_kwargs["cb_run_counter"]
    now = time.monotonic()
    status = job_meta["status"]
    if state["count"] == 0:
        state["started"] = now
        print(f"Job ID: {job_id}", flush=True)
    if state["count"] == 0 or status != state.get("status") or now - state["last_report"] >= 15:
        print(f"Job status: {status} ({now - state['started']:.0f}s monitored)", flush=True)
        state["last_report"] = now
        state["status"] = status
        get_module_logger().debug("Job metadata: %s", job_meta)
    state["count"] += 1
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
        with tempfile.TemporaryDirectory() as temp_dir:
            job.export_job(temp_dir)
            warn_on_potential_secrets_in_job_dir(temp_dir, job_name=job.name)
            job_path = os.path.join(temp_dir, job.name)
            sess = self._get_session()
            try:
                job_id = sess.submit_job(job_path)
            finally:
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
        if rc == MonitorReturnCode.JOB_FINISHED:
            print("Downloading job results...", flush=True)
            result = sess.download_job_result(job_id)
            sess.close()
            return result
        elif rc == MonitorReturnCode.TIMEOUT:
            print(f"Monitoring job {job_id} timed out after {timeout} seconds. No results were downloaded.")
            sess.close()
            return None
        elif rc == MonitorReturnCode.ENDED_BY_CB:
            print("Job monitoring was stopped early by callback. No results were downloaded.")
            sess.close()
            return None
        else:
            raise RuntimeError(f"Unexpected monitor return code: {rc}")
