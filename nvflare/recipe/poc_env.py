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
import shutil
import time
from typing import Optional

from pydantic import BaseModel, conint, model_validator

from nvflare.job_config.api import FedJob
from nvflare.recipe.spec import ExecEnv
from nvflare.tool.poc.poc_commands import (
    _clean_poc,
    _start_poc,
    _stop_poc,
    get_poc_workspace,
    get_prod_dir,
    is_poc_running,
    prepare_poc_provision,
    setup_service_config,
)
from nvflare.tool.poc.service_constants import FlareServiceConstants as SC

from .session_mgr import SessionManager

STOP_POC_TIMEOUT = 10
SERVICE_START_TIMEOUT = 3
DEFAULT_ADMIN_USER = "admin@nvidia.com"


# Internal — not part of the public API
class _PocEnvValidator(BaseModel):
    num_clients: Optional[conint(gt=0)] = None
    clients: Optional[list[str]] = None
    gpu_ids: Optional[list[int]] = None
    use_he: bool = False
    docker_image: Optional[str] = None
    project_conf_path: str = ""
    username: str = DEFAULT_ADMIN_USER

    @model_validator(mode="after")
    def check_client_configuration(self):
        # Check if clients list is empty
        if self.clients is not None and len(self.clients) == 0:
            raise ValueError("clients list cannot be empty")

        # Check if both num_clients and clients are specified and inconsistent
        if self.clients is not None and self.num_clients > 0 and len(self.clients) != self.num_clients:
            raise ValueError(
                f"Inconsistent: num_clients={self.num_clients} but clients list has {len(self.clients)} entries"
            )

        # Check if num_clients is valid when clients is None
        if self.clients is None and self.num_clients <= 0:
            raise ValueError("num_clients must be greater than 0")

        return self


class PocEnv(ExecEnv):
    """Proof of Concept execution environment for local testing and development.

    This environment sets up a POC deployment on a single machine with multiple
    processes representing the server, clients, and admin console.
    """

    def __init__(
        self,
        *,
        num_clients: Optional[int] = 2,
        clients: Optional[list[str]] = None,
        gpu_ids: Optional[list[int]] = None,
        use_he: bool = False,
        docker_image: str = None,
        project_conf_path: str = "",
        username: str = DEFAULT_ADMIN_USER,
        extra: dict = None,
    ):
        """Initialize POC execution environment.

        Args:
            num_clients (int, optional): Number of clients to use in POC mode. Defaults to 2.
            clients (list[str], optional): List of client names. If None, will generate site-1, site-2, etc. Defaults to None.
                If specified, number_of_clients argument will be ignored.
            gpu_ids (list[int], optional): List of GPU IDs to assign to clients. If None, uses CPU only. Defaults to None.
            use_he (bool, optional): Whether to use HE. Defaults to False.
            docker_image (str, optional): Docker image to use for POC. Defaults to None.
            project_conf_path (str, optional): Path to the project configuration file. Defaults to "".
                If specified, 'number_of_clients','clients' and 'docker' specific options will be ignored.
            username (str, optional): Admin user. Defaults to "admin@nvidia.com".
            extra: extra env info.
        """
        super().__init__(extra)

        v = _PocEnvValidator(
            num_clients=num_clients,
            clients=clients,
            gpu_ids=gpu_ids,
            use_he=use_he,
            docker_image=docker_image,
            project_conf_path=project_conf_path,
            username=username,
        )

        self.clients = v.clients
        self.num_clients = len(v.clients) if v.clients is not None else v.num_clients
        self.poc_workspace = get_poc_workspace()
        self.gpu_ids = v.gpu_ids or []
        self.use_he = v.use_he
        self.project_conf_path = v.project_conf_path
        self.docker_image = v.docker_image
        self.username = v.username
        self._session_manager = None  # Lazy initialization

    def deploy(self, job: FedJob):
        """Deploy a FedJob to the POC environment.

        Args:
            job (FedJob): The FedJob to deploy.

        Returns:
            str: Job ID or deployment result.
        """
        if self._check_poc_running():
            self.stop(clean_poc=True)

        print("Preparing and starting fresh POC services...")
        prepare_poc_provision(
            clients=self.clients or [],  # Empty list if None, let prepare_clients generate
            number_of_clients=self.num_clients,
            workspace=self.poc_workspace,
            docker_image=self.docker_image,
            use_he=self.use_he,
            project_conf_path=self.project_conf_path,
            examples_dir=None,
        )

        _start_poc(
            poc_workspace=self.poc_workspace,
            gpu_ids=self.gpu_ids,
            excluded=[self.username],
            services_list=[],
        )
        print("POC services started successfully")

        # Give services time to start up
        time.sleep(SERVICE_START_TIMEOUT)

        # Submit job using SessionManager
        return self._get_session_manager().submit_job(job)

    def _check_poc_running(self) -> bool:
        try:
            project_config, service_config = setup_service_config(self.poc_workspace)
        except Exception as e:
            # POC workspace is not initialized yet, so we don't need to stop and clean it
            return False

        if not is_poc_running(self.poc_workspace, service_config, project_config):
            return False

        return True

    def stop(self, clean_poc: bool = False):
        """Try to stop and clean existing POC.

        Args:
            clean_poc (bool, optional): Whether to clean the POC workspace. Defaults to False.
        """
        project_config, service_config = setup_service_config(self.poc_workspace)

        try:
            print("Stopping existing POC services...")
            _stop_poc(
                poc_workspace=self.poc_workspace,
                excluded=[self.username],  # Exclude admin console (consistent with start)
                services_list=[],
            )
            count = 0
            poc_running = True
            while count < STOP_POC_TIMEOUT:
                if not is_poc_running(self.poc_workspace, service_config, project_config):
                    poc_running = False
                    break
                time.sleep(1)
                count += 1

            if clean_poc:
                if poc_running:
                    print(
                        f"Warning: POC still running after {STOP_POC_TIMEOUT} seconds, cannot clean workspace. Skipping cleanup."
                    )
                else:
                    _clean_poc(self.poc_workspace)
        except Exception as e:
            print(f"Warning: Failed to stop and clean existing POC: {e}")
        print(f"Removing POC workspace: {self.poc_workspace}")
        shutil.rmtree(self.poc_workspace, ignore_errors=True)

    def get_job_status(self, job_id: str) -> Optional[str]:
        return self._get_session_manager().get_job_status(job_id)

    def abort_job(self, job_id: str) -> None:
        self._get_session_manager().abort_job(job_id)

    def get_job_result(self, job_id: str, timeout: float = 0.0) -> Optional[str]:
        return self._get_session_manager().get_job_result(job_id, timeout)

    def _get_admin_startup_kit_path(self) -> str:
        """Get the path to the admin startup kit for POC.

        Returns:
            str: Path to admin startup kit directory.
        """
        try:
            project_config, service_config = setup_service_config(self.poc_workspace)
            project_name = project_config.get("name")
            prod_dir = get_prod_dir(self.poc_workspace, project_name)

            # POC admin directory structure: {workspace}/{project_name}/prod_00/admin@nvidia.com
            project_admin_dir = service_config.get(SC.FLARE_PROJ_ADMIN, SC.FLARE_PROJ_ADMIN)
            admin_dir = os.path.join(prod_dir, project_admin_dir)

            if not os.path.exists(admin_dir):
                raise RuntimeError(f"Admin startup kit not found at: {admin_dir}")

            return admin_dir

        except Exception as e:
            raise RuntimeError(f"Failed to locate admin startup kit: {e}")

    def _get_session_manager(self):
        """Get or create SessionManager with lazy initialization."""
        if self._session_manager is None:
            session_params = {
                "username": self.username,
                "startup_kit_location": self._get_admin_startup_kit_path(),
                "timeout": self.get_extra_prop("login_timeout", 10),
            }
            self._session_manager = SessionManager(session_params)
        return self._session_manager
