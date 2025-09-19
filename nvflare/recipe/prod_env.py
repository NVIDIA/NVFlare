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
from typing import Optional

from pydantic import BaseModel, PositiveFloat, model_validator

from nvflare.job_config.api import FedJob
from nvflare.recipe.spec import ExecEnv

from .session_mgr import SessionManager

DEFAULT_ADMIN_USER = "admin@nvidia.com"


# Internal — not part of the public API
class _ProdEnvValidator(BaseModel):
    startup_kit_location: str
    login_timeout: PositiveFloat = 5.0
    username: str = DEFAULT_ADMIN_USER

    @model_validator(mode="after")
    def check_startup_kit_location_exists(self) -> "_ProdEnvValidator":
        if not os.path.exists(self.startup_kit_location):
            raise ValueError(f"startup_kit_location path does not exist: {self.startup_kit_location}")
        return self


class ProdEnv(ExecEnv):
    def __init__(
        self,
        startup_kit_location: str,
        login_timeout: float = 5.0,
        username: str = DEFAULT_ADMIN_USER,
        extra: dict = None,
    ):
        """Production execution environment for submitting and monitoring NVFlare jobs.

        This environment uses the startup kit of an NVFlare deployment to submit jobs via the Flare API.

        Args:
            startup_kit_location (str): Path to the admin's startup kit directory.
            login_timeout (float): Timeout (in seconds) for logging into the Flare API session. Must be > 0.
            username (str): Username to log in with.
            extra: extra env info.
        """
        super().__init__(extra)

        v = _ProdEnvValidator(
            startup_kit_location=startup_kit_location,
            login_timeout=login_timeout,
            username=username,
        )

        self.startup_kit_location = v.startup_kit_location
        self.login_timeout = v.login_timeout
        self.username = v.username
        self._session_manager = None  # Lazy initialization

    def get_job_status(self, job_id: str) -> Optional[str]:
        return self._get_session_manager().get_job_status(job_id)

    def abort_job(self, job_id: str) -> None:
        self._get_session_manager().abort_job(job_id)

    def get_job_result(self, job_id: str, timeout: float = 0.0) -> Optional[str]:
        return self._get_session_manager().get_job_result(job_id, timeout)

    def deploy(self, job: FedJob):
        """Deploy a job using SessionManager."""
        try:
            return self._get_session_manager().submit_job(job)
        except Exception as e:
            raise RuntimeError(f"Failed to submit job via Flare API: {e}")

    def _get_session_manager(self):
        """Get or create SessionManager with lazy initialization."""
        if self._session_manager is None:
            session_params = {
                "username": self.username,
                "startup_kit_location": self.startup_kit_location,
                "timeout": self.login_timeout,
            }
            self._session_manager = SessionManager(session_params)
        return self._session_manager
