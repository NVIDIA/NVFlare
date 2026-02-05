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

import threading
from typing import Optional

from nvflare.fuel.utils.log_utils import get_obj_logger
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
        # Lock released before exec_env call for performance. If another thread
        # stops the env concurrently, the try/except handles it gracefully.
        try:
            return self.exec_env.get_job_status(self.job_id)
        except Exception as e:
            self.logger.warning(f"Failed to get job status: {e}")
            return None

    def get_result(self, timeout: float = 0.0) -> Optional[str]:
        """Get the result workspace of the run.

        Waits for job to complete, caches status, then stops execution environment.

        Args:
            timeout (float, optional): Timeout for job completion. Defaults to 0.0 (no timeout).

        Returns:
            Optional[str]: Result workspace path, or None if job not finished or on error.
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

            try:
                self.exec_env.stop(clean_up=True)
            except Exception as e:
                self.logger.warning(f"Failed to stop execution environment: {e}")
            finally:
                self._stopped = True

            return result

    def abort(self) -> None:
        """Abort the running job.

        This is a no-op if the execution environment has already been stopped
        (e.g., after get_result() was called). Errors are logged but not raised.
        """
        with self._lock:
            if self._stopped:
                return
        # Lock released before exec_env call for performance. If another thread
        # stops the env concurrently, the try/except handles it gracefully.
        try:
            self.exec_env.abort_job(self.job_id)
        except Exception as e:
            self.logger.warning(f"Failed to abort job: {e}")
