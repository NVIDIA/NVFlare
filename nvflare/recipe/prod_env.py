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

import os.path
import tempfile
from typing import Optional

from pydantic import BaseModel, PositiveFloat, conint, model_validator

from nvflare.fuel.flare_api.flare_api import Session, new_secure_session
from nvflare.job_config.api import FedJob

from .spec import ExecEnv

DEFAULT_ADMIN_USER = "admin@nvidia.com"


def status_monitor_cb(session: Session, job_id: str, job_meta, *cb_args, **cb_kwargs) -> bool:
    if job_meta["status"] == "RUNNING":
        if cb_kwargs["cb_run_counter"]["count"] < 3 or cb_kwargs["cb_run_counter"]["count"] % 15 == 0:
            print(job_meta)
        else:
            # avoid printing job_meta repeatedly to save space on the screen and not overwhelm the user
            print(".", end="")
    else:
        print("\n" + str(job_meta))

    cb_kwargs["cb_run_counter"]["count"] += 1
    return True


# Internal — not part of the public API
class _ProdEnvValidator(BaseModel):
    startup_kit_dir: str
    login_timeout: PositiveFloat = 5.0
    monitor_job_duration: Optional[conint(ge=0)] = None  # must be zero or positive if specified

    @model_validator(mode="after")
    def check_startup_kit_dir_exists(self) -> "_ProdEnvValidator":
        if not os.path.exists(self.startup_kit_dir):
            raise ValueError(f"startup_kit_dir path does not exist: {self.startup_kit_dir}")
        return self


class ProdEnv(ExecEnv):
    def __init__(
        self,
        startup_kit_dir: str,
        login_timeout: float = 5.0,
        monitor_job_duration: Optional[int] = None,
    ):
        """Production execution environment for submitting and monitoring NVFlare jobs.

        This environment uses the startup kit of an NVFlare deployment to submit jobs via the Flare API.

        Args:
            startup_kit_dir (str): Path to the admin's startup kit directory.
            login_timeout (float): Timeout (in seconds) for logging into the Flare API session. Must be > 0.
            monitor_job_duration (int, optional): Duration (in seconds) to monitor job execution.
                If None, monitoring is skipped. If 0, will wait for the job to complete. Must be >= 0.
        """
        v = _ProdEnvValidator(
            startup_kit_dir=startup_kit_dir,
            login_timeout=login_timeout,
            monitor_job_duration=monitor_job_duration,
        )

        self.startup_kit_dir = v.startup_kit_dir
        self.login_timeout = v.login_timeout
        self.monitor_job_duration = v.monitor_job_duration
        self.admin_user = os.path.basename(startup_kit_dir)

    def deploy(self, job: FedJob):
        sess = None
        try:
            sess = new_secure_session(
                username=self.admin_user, startup_kit_location=self.startup_kit_dir, timeout=self.login_timeout
            )
            with tempfile.TemporaryDirectory() as temp_dir:
                job.export_job(temp_dir)
                job_path = os.path.join(temp_dir, job.name)
                job_id = sess.submit_job(job_path)
                print(f"Submitted job '{job.name}' with ID: {job_id}")

            if self.monitor_job_duration is not None:
                rc = sess.monitor_job(job_id, cb=status_monitor_cb, timeout=self.monitor_job_duration)
                print(f"job monitor done: {rc=}")

            return job_id
        except Exception as e:
            raise RuntimeError(f"Failed to submit/monitor job via Flare API: {e}")
        finally:
            if sess:
                sess.close()

    def get_env_info(self) -> dict:
        return {
            "env_type": "prod",
            "startup_kit_dir": self.startup_kit_dir,
            "login_timeout": self.login_timeout,
            "monitor_job_duration": self.monitor_job_duration,
            "admin_user": self.admin_user,
        }
