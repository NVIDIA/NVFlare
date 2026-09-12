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

import hashlib
import json
import logging
import os
import tempfile
import time
from typing import Dict, Optional

from nvflare.apis.job_def import RunStatus
from nvflare.fuel.flare_api.api_spec import MonitorReturnCode
from nvflare.fuel.flare_api.flare_api import Session, new_secure_session
from nvflare.fuel.utils.job_secret_scanner import warn_on_potential_secrets_in_job_dir
from nvflare.fuel.utils.log_utils import FL_LOG_LEVEL, LogMode, ProgressFormatter, get_module_logger
from nvflare.job_config.api import FedJob
from nvflare.recipe._failure_summary import collect_client_errors


def _show_job_progress(session, job_id, state):
    """Replay a bounded tail of existing server logs with bounded deduplication."""
    try:
        response = session.get_job_logs(job_id, target="server", log_file_name="log.json", tail_lines=200)
        logs = response.get("logs", {})
        formatter = ProgressFormatter()
        recent = set()
        # The existing API caps the transfer at 5 MiB. Bound parsing and retained
        # state independently: only the server's last 64 KiB / 200 lines.
        text = logs.get("server", "").encode("utf-8")[-65536:].decode("utf-8", errors="ignore")
        for line in text.splitlines()[-200:]:
            digest = hashlib.sha256(line.encode()).digest()
            try:
                record = json.loads(line)
            except (ValueError, TypeError):
                continue  # An in-flight final line may be incomplete; retry on the next callback.
            if not isinstance(record, dict):
                continue
            name = record.get("fullName", "")
            level = record.get("levelname", "INFO")
            if not isinstance(name, str) or not isinstance(level, str):
                continue
            if not name.endswith(".progress") and level not in ("WARNING", "ERROR", "CRITICAL"):
                continue
            already_shown = digest in state["seen"] or digest in recent
            recent.add(digest)
            if already_shown:
                continue
            message = record.get("message", "")
            context = record.get("fl_ctx", "")
            if isinstance(context, str) and context:
                message = f"{context}: {message}"
            log_record = logging.LogRecord(name, getattr(logging, level, logging.INFO), "", 0, message, (), None)
            print(formatter.format(log_record), flush=True)
        state["seen"] = recent
    except Exception as ex:
        if not state.get("warned"):
            print("Live progress could not be retrieved. Detailed logs remain on the server.", flush=True)
            state["warned"] = True
        get_module_logger().debug("Could not retrieve progress for %s: %s", job_id, ex)


def _job_monitor_callback(session: Session, job_id: str, job_meta, *cb_args, **cb_kwargs) -> bool:
    """Show status changes, with the existing server progress log when requested."""
    state = cb_kwargs["cb_run_counter"]
    now = time.monotonic()
    status = job_meta["status"]
    if state["count"] == 0:
        state["started"] = now
        print(f"Job ID: {job_id}", flush=True)
    changed = state["count"] == 0 or status != state.get("status")
    if "progress" in state and (changed or now - state["last_progress"] >= 5):
        _show_job_progress(session, job_id, state["progress"])
        state["last_progress"] = now
    if changed:
        print(f"Job status: {status} ({now - state['started']:.0f}s monitored)", flush=True)
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
        if os.environ.get(FL_LOG_LEVEL, LogMode.CONCISE) == LogMode.CONCISE:
            cb_run_counter["progress"] = {"seen": set()}
        rc = sess.monitor_job(job_id, timeout=timeout, cb=_job_monitor_callback, cb_run_counter=cb_run_counter)
        if rc == MonitorReturnCode.JOB_FINISHED:
            print("Downloading job results...", flush=True)
            result = sess.download_job_result(job_id)
            if result and cb_run_counter.get("status") not in (None, RunStatus.FINISHED_COMPLETED.value):
                try:
                    collect_client_errors(sess, job_id, result)
                except Exception as ex:
                    get_module_logger().debug("Could not retrieve client error logs for %s: %s", job_id, ex)
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
