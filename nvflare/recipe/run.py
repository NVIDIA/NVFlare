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
import threading
import time
from typing import Optional

from nvflare.fuel.flare_api.flare_api import _job_status_outcome
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.recipe._failure_summary import failure_summary
from nvflare.recipe._run_summary import result_summary, summary_header
from nvflare.recipe.spec import ExecEnv


class Run:
    """Represents a running or completed job execution.

    Provides methods to get job status, results, and abort the job.
    Caches status and result after the execution environment is stopped.

    This class is thread-safe. All state-changing operations are protected by a lock.
    """

    def __init__(self, exec_env: ExecEnv, job_id: str):
        """Initialize a Run instance.

        Args:
            exec_env: The execution environment managing this job.
            job_id: The unique identifier for the job.

        Raises:
            ValueError: If exec_env is None or job_id is empty.
        """
        if exec_env is None:
            raise ValueError("exec_env cannot be None")
        if not job_id or not isinstance(job_id, str):
            raise ValueError("job_id must be a non-empty string")

        self.exec_env = exec_env
        self.job_id = job_id
        self._lock = threading.Lock()
        self._stopped = False
        self._cached_status: Optional[str] = None
        self._cached_result: Optional[str] = None
        self.logger = get_obj_logger(self)
        self._started_at = time.monotonic()

    def get_job_id(self) -> str:
        """Get the job ID.

        Returns:
            str: The job ID.
        """
        return self.job_id

    def get_status(self) -> Optional[str]:
        """Get the status of the run.

        Returns:
            Optional[str]: The status of the run, or None if not available or on error.
        """
        with self._lock:
            if self._stopped:
                return self._cached_status
            try:
                return self.exec_env.get_job_status(self.job_id)
            except Exception as e:
                self.logger.warning(f"Failed to get job status: {e}")
                return None

    def get_result(self, timeout: float = 0.0, clean_up: bool = True) -> Optional[str]:
        """Get the result workspace of the run.

        Waits for job to complete, caches status, then stops execution environment.

        Args:
            timeout (float, optional): Timeout for job completion. Defaults to 0.0 (no timeout).
            clean_up (bool, optional): Whether to remove the execution-environment workspace
                (e.g. the POC workspace) when stopping. Defaults to True, preserving the
                existing "each run is independent" behavior. Pass ``clean_up=False`` to keep
                the workspace on disk after the run so server/client log files (including
                the per-service ``poc_console.log`` introduced in #4500) remain available
                for debugging or test assertions.

        Returns:
            Optional[str]: Result workspace path, or None if job not finished or on error.
            The path may be removed by the time this method returns when ``clean_up=True``.
        """
        with self._lock:
            if self._stopped:
                return self._cached_result

            result = None
            try:
                result = self.exec_env.get_job_result(self.job_id, timeout=timeout)
                self._cached_result = result
            except Exception as e:
                self.logger.warning(f"Failed to get job result: {e}")
                self._cached_result = None

            try:
                self._cached_status = self.exec_env.get_job_status(self.job_id)
            except Exception as e:
                self.logger.warning(f"Failed to get job status: {e}")
                self._cached_status = None

            report = ""
            failure_report = ""
            outcome = _job_status_outcome(self._cached_status)
            if outcome == "failed":
                failure_report = failure_summary(result)
            if result and os.path.isdir(result):
                try:
                    report = result_summary(result)
                except Exception as e:
                    self.logger.debug("Could not summarize result artifacts: %s", e)
            elapsed = time.monotonic() - self._started_at
            try:
                self.exec_env.stop(clean_up=clean_up)
            except Exception as e:
                self.logger.warning(f"Failed to stop execution environment: {e}")
            finally:
                self._stopped = True

            status_label = {"completed": "✓ Completed", "failed": "✗ Failed"}.get(
                outcome, self._cached_status or "Status unavailable"
            )
            print(summary_header(status_label, elapsed), flush=True)
            if failure_report:
                print(f"\n  Status    {self._cached_status}", flush=True)
                print(failure_report, flush=True)
            if report:
                print(report, flush=True)

            if result:
                if os.path.isdir(result):
                    print(f"  Results   {result}", flush=True)
                else:
                    print(
                        f"Result workspace is not available locally: {result}. "
                        "To retain an environment's workspace, use get_result(clean_up=False) when running the job.",
                        flush=True,
                    )
            else:
                print(
                    "No result workspace was returned. See the status and preceding messages for details.", flush=True
                )

            return result

    def abort(self) -> None:
        """Abort the running job.

        This is a no-op if the execution environment has already been stopped
        (e.g., after get_result() was called). Errors are logged but not raised.
        """
        with self._lock:
            if self._stopped:
                return
            try:
                self.exec_env.abort_job(self.job_id)
            except Exception as e:
                self.logger.warning(f"Failed to abort job: {e}")
