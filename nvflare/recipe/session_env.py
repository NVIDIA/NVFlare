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

from abc import abstractmethod
from typing import Dict, Optional

from nvflare.fuel.flare_api.api_spec import MonitorReturnCode
from nvflare.fuel.flare_api.flare_api import Session, new_secure_session

from .spec import ExecEnv


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


class SessionEnv(ExecEnv):
    """Base class for execution environments that use sessions for job management.

    This class provides common functionality for POC and Production environments
    that need to manage jobs via sessions to the Flare API.
    """

    @abstractmethod
    def _get_session_params(self) -> Dict[str, any]:
        """Get the session parameters for creating a session.

        Returns:
            Dict containing session parameters like username, startup_kit_location, timeout
        """
        pass

    def get_job_status(self, job_id: str) -> Optional[str]:
        """Get the status of the job."""
        sess = None
        try:
            sess = new_secure_session(**self._get_session_params())
            return sess.get_job_status(job_id)
        except Exception as e:
            raise RuntimeError(f"Failed to get job status: {e}")
        finally:
            if sess:
                sess.close()

    def abort_job(self, job_id: str) -> None:
        """Abort the running job."""
        sess = None
        try:
            sess = new_secure_session(**self._get_session_params())
            msg = sess.abort_job(job_id)
            print(f"Job {job_id} aborted successfully with message: {msg}")
        except Exception as e:
            print(f"Failed to abort job {job_id}: {e}")
        finally:
            if sess:
                sess.close()

    def get_job_result(self, job_id: str, timeout: float = 0.0) -> Optional[str]:
        """Get the result workspace of the job."""
        sess = None
        try:
            sess = new_secure_session(**self._get_session_params())
            cb_run_counter = {"count": 0}
            rc = sess.monitor_job(job_id, timeout=timeout, cb=_job_monitor_callback, cb_run_counter=cb_run_counter)
            print(f"job monitor done: {rc=}")
            if rc == MonitorReturnCode.JOB_FINISHED:
                return sess.download_job_result(job_id)
            elif rc == MonitorReturnCode.TIMEOUT:
                print(
                    f"Job {job_id} did not complete within {timeout} seconds. "
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
        except Exception as e:
            raise RuntimeError(f"Failed to get job result: {e}")
        finally:
            if sess:
                sess.close()
