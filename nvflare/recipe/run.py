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

from typing import Any, Dict, Optional

from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.recipe.spec import ExecEnv


class Run:
    def __init__(self, exec_env: ExecEnv, job_id: str):
        self.exec_env = exec_env
        self.job_id = job_id
        self._stopped = False
        self._cached_status: Optional[str] = None
        self._cached_result: Optional[Dict[str, Any]] = None
        self.logger = get_obj_logger(self)

    def get_job_id(self) -> str:
        return self.job_id

    def get_status(self) -> Optional[str]:
        """Get the status of the run.

        Returns:
            Optional[str]: The status of the run, or None if not available.
        """
        if self._stopped:
            return self._cached_status
        return self.exec_env.get_job_status(self.job_id)

    def get_result(self, timeout: float = 0.0) -> Optional[Dict[str, Any]]:
        """Get the result of the run.

        Waits for job to complete, reads result content, caches status, then stops POC.

        Args:
            timeout (float, optional): Timeout for job completion. Defaults to 0.0 (no timeout).

        Returns:
            Optional[Dict[str, Any]]: Result content as dict, or None if job not finished.
        """
        if self._stopped is True:
            return self._cached_result

        result = None
        try:
            result = self.exec_env.get_job_result(self.job_id, timeout=timeout)
            self._cached_result = result
            self._cached_status = self.exec_env.get_job_status(self.job_id)
        except Exception:
            self._cached_status = None
            self._cached_result = None
        finally:
            try:
                self.exec_env.stop(clean_up=True)
                self._stopped = True
            except TypeError:
                self.logger.warning("Unable to stop or cleanup execution environment")

        return result

    def abort(self):
        """Abort the running job."""
        if not self._stopped:
            self.exec_env.abort_job(self.job_id)
