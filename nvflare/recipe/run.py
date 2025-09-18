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
from typing import Generator, Optional

from nvflare.fuel.flare_api.api_spec import MonitorReturnCode
from nvflare.fuel.flare_api.flare_api import Session, new_secure_session


def _cb_with_print(session: Session, job_id: str, job_meta, *cb_args, **cb_kwargs) -> bool:
    """Callback to print job meta."""
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
        sess = None
        try:
            sess = new_secure_session(**self._get_session_params())
            yield sess
        except Exception as e:
            raise RuntimeError(f"Failed to create/use session: {e}")
        finally:
            if sess:
                sess.close()

    def get_status(self) -> Optional[str]:
        """Get the status of the run.

        Returns:
            Optional[str]: The status of the run, or None if called in a simulation environment.
        """
        if self._is_sim_env():
            print(
                f"Note, get_status returns None in SimEnv. The simulation logs can be found at {self._get_sim_result()}"
            )
            return None

        with self._secure_session() as sess:
            return sess.get_job_status(self.job_id)

    def _get_sim_result(self, **kwargs) -> str:
        workspace_root = self.env_info.get("workspace_root")
        if workspace_root is None:
            raise RuntimeError("Simulation workspace_root is None - SimEnv may not be properly initialized")
        return os.path.join(workspace_root, self.job_id)

    def _get_prod_result(self, timeout: float = 0.0) -> Optional[str]:
        with self._secure_session() as sess:
            cb_run_counter = {"count": 0}
            rc = sess.monitor_job(self.job_id, timeout=timeout, cb=_cb_with_print, cb_run_counter=cb_run_counter)
            print(f"job monitor done: {rc=}")
            if rc == MonitorReturnCode.JOB_FINISHED:
                return sess.download_job_result(self.job_id)
            elif rc == MonitorReturnCode.TIMEOUT:
                print(
                    f"Job {self.job_id} did not complete within {timeout} seconds. "
                    "Job is still running. Try calling get_result() again with a longer timeout."
                )
                return None
            elif rc == MonitorReturnCode.ENDED_BY_CB:
                print(
                    "Job monitoring was stopped early by callback. "
                    "Result may not be available yet. Check job status and try again."
                )
                return None
            else:
                raise RuntimeError(f"Unexpected monitor return code: {rc}")

    def get_result(self, timeout: float = 0.0) -> Optional[str]:
        """Get the result workspace of the run.

        Args:
            timeout (float, optional): The timeout for the job to complete.
                Defaults to 0.0, means never timeout.

        Returns:
            Optional[str]: The result workspace path if job completed, None if still running or stopped early.
        """
        env_type = self.env_info.get("env_type")
        return self.handlers[env_type](timeout=timeout)

    def abort(self):
        """Abort the running job."""
        if self._is_sim_env():
            print("abort is not supported in a simulation environment, it will always run to completion.")
            return

        try:
            with self._secure_session() as sess:
                msg = sess.abort_job(self.job_id)
                print(f"Job {self.job_id} aborted successfully with message: {msg}")
        except Exception as e:
            print(f"Failed to abort job {self.job_id}: {e}")
