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
from contextlib import contextmanager
from typing import Generator

from nvflare.fuel.flare_api.api_spec import MonitorReturnCode
from nvflare.fuel.flare_api.flare_api import basic_cb_with_print, new_secure_session


class Run:
    def __init__(self, env_info: dict, job_id: str):
        self.env_info = env_info
        self.job_id = job_id
        self.handlers = {
            "sim": self._get_sim_result,
            "poc": self._get_prod_result,
            "prod": self._get_prod_result,
        }

    def get_job_id(self) -> str:
        return self.job_id

    def _is_sim_env(self) -> bool:
        """Check if this is a simulation environment."""
        return self.env_info.get("env_type") == "sim"

    def _get_session_params(self) -> dict:
        """Get session parameters from env_info."""
        return {
            "startup_kit_location": self.env_info.get("startup_kit_location"),
            "username": self.env_info.get("username"),
            "timeout": self.env_info.get("login_timeout", 10),
        }

    @contextmanager
    def _secure_session(self) -> Generator:
        """Context manager for secure session handling."""
        if self._is_sim_env():
            raise RuntimeError("Secure session not needed for simulation environment")

        sess = None
        try:
            sess = new_secure_session(**self._get_session_params())
            yield sess
        except Exception as e:
            raise RuntimeError(f"Failed to create/use session: {e}")
        finally:
            if sess:
                sess.close()

    def get_status(self) -> str:
        """Get the status of the run.

        Returns:
            str: The status of the run.
        """
        if self._is_sim_env():
            print(
                "get_status will always return completed in a simulation environment, please check the log inside the workspace returned by get_result()"
            )
            return "completed"

        with self._secure_session() as sess:
            return sess.get_job_status(self.job_id)

    def _get_sim_result(self, **kwargs) -> str:
        workspace_root = self.env_info.get("workspace_root")
        if workspace_root is None:
            raise RuntimeError("Simulation workspace_root is None - SimEnv may not be properly initialized")
        return os.path.join(workspace_root, self.job_id)

    def _get_prod_result(self, timeout: float = 0.0) -> str:
        with self._secure_session() as sess:
            cb_run_counter = {"count": 0}
            rc = sess.monitor_job(self.job_id, timeout=timeout, cb=basic_cb_with_print, cb_run_counter=cb_run_counter)
            print(f"job monitor done: {rc=}")
            if rc == MonitorReturnCode.JOB_FINISHED:
                return sess.download_job_result(self.job_id)
            else:
                raise RuntimeError(f"Monitor job failed: {rc}")

    def get_result(self, timeout: float = 0.0) -> str:
        """Get the result workspace of the run.

        Args:
            timeout (float, optional): The timeout for the job to complete.
                Defaults to 0.0, means never timeout.

        Returns:
            str: The result workspace of the run.
        """
        env_type = self.env_info.get("env_type")
        return self.handlers[env_type](timeout=timeout)

    def abort(self):
        """Abort the running job."""
        if self._is_sim_env():
            print("abort is not supported in a simulation environment, it will always run to completion.")
            return

        with self._secure_session() as sess:
            sess.abort_job(self.job_id)
