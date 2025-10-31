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

from typing import Optional

from nvflare.recipe.spec import ExecEnv


class Run:
    def __init__(self, exec_env: ExecEnv, job_id: str):
        self.exec_env = exec_env
        self.job_id = job_id

    def get_job_id(self) -> str:
        return self.job_id

    def get_status(self) -> Optional[str]:
        """Get the status of the run.

        Returns:
            Optional[str]: The status of the run, or None if called in a simulation environment.
        """
        return self.exec_env.get_job_status(self.job_id)

    def get_result(self, timeout: float = 0.0) -> Optional[str]:
        """Get the result workspace of the run.

        Args:
            timeout (float, optional): The timeout for the job to complete.
                Defaults to 0.0, means never timeout.

        Returns:
            Optional[str]: The result workspace path if job completed, None if still running or stopped early.
        """
        return self.exec_env.get_job_result(self.job_id, timeout=timeout)

    def abort(self):
        """Abort the running job."""
        self.exec_env.abort_job(self.job_id)
