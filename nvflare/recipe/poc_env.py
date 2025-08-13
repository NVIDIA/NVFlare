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
import tempfile
import time
from typing import List, Optional

from nvflare.fuel.flare_api.flare_api import new_secure_session
from nvflare.job_config.api import FedJob
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

from .spec import ExecEnv

STOP_POC_TIMEOUT = 10
DEFAULT_ADMIN_USER = "admin@nvidia.com"


class POCEnv(ExecEnv):
    """Proof of Concept execution environment for local testing and development.

    This environment sets up a POC deployment on a single machine with multiple
    processes representing the server, clients, and admin console.
    """

    def __init__(
        self,
        *,
        num_clients: int = 2,
        clients: Optional[List[str]] = None,
        gpu_ids: Optional[List[int]] = None,
        auto_stop: bool = True,
        monitor_duration: int = 0,
        use_he: bool = False,
        docker_image: str = None,
        project_conf_path: str = "",
    ):
        """Initialize POC execution environment.

        Args:
            num_clients (int, optional): Number of clients to use in POC mode. Defaults to 2.
            clients (List[str], optional): List of client names. If None, will generate site-1, site-2, etc. Defaults to None.
                If specified, number_of_clients argument will be ignored.
            gpu_ids (List[int], optional): List of GPU IDs to assign to clients. If None, uses CPU only. Defaults to None.
            auto_stop (bool, optional): Whether to automatically stop POC services after job completion. Defaults to True.
            monitor_duration (int, optional): Duration to monitor job execution (in seconds). 0 means wait until completion, negative means no monitoring. Defaults to 0.
            use_he (bool, optional): Whether to use HE. Defaults to False.
            docker_image (str, optional): Docker image to use for POC. Defaults to None.
            project_conf_path (str, optional): Path to the project configuration file. Defaults to "".
                If specified, 'number_of_clients','clients' and 'docker' specific options will be ignored.
        """
        # Validate client configuration
        if clients is not None:
            if len(clients) == 0:
                raise ValueError("clients list cannot be empty")
            if num_clients > 0 and len(clients) != num_clients:
                raise ValueError(f"Inconsistent: num_clients={num_clients} but clients list has {len(clients)} entries")
        else:
            if num_clients <= 0:
                raise ValueError("num_clients must be greater than 0")

        self.clients = clients
        self.num_clients = len(clients) if clients is not None else num_clients
        self.poc_workspace = get_poc_workspace()
        self.gpu_ids = gpu_ids or []
        self.auto_stop = auto_stop
        self.monitor_duration = monitor_duration
        self.use_he = use_he
        self.project_conf_path = project_conf_path
        self.docker_image = docker_image

    def _try_to_stop_and_clean_existing_poc(self):
        """Try to stop and clean existing POC if it is running."""
        try:
            project_config, service_config = setup_service_config(self.poc_workspace)
        except Exception as e:
            # POC workspace is not initialized yet, so we don't need to stop and clean it
            pass

        try:
            if is_poc_running(self.poc_workspace, service_config, project_config):
                print("POC services already running, stopping and cleaning to ensure fresh environment...")
                self._stop_and_clean_poc()
        except Exception as e:
            print(f"Warning: Failed to stop and clean existing POC: {e}")
        print(f"Removing POC workspace: {self.poc_workspace}")
        shutil.rmtree(self.poc_workspace, ignore_errors=True)

    def deploy(self, job: FedJob):
        """Deploy a FedJob to the POC environment.

        Args:
            job (FedJob): The FedJob to deploy.

        Returns:
            str: Job ID or deployment result.
        """
        try:
            self._try_to_stop_and_clean_existing_poc()

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
                excluded=[DEFAULT_ADMIN_USER],
                services_list=[],
            )
            print("POC services started successfully")

            # Give services time to start up
            time.sleep(3)

            # Submit job using Flare API like ProdEnv
            with tempfile.TemporaryDirectory() as temp_dir:
                job.export_job(temp_dir)
                job_path = os.path.join(temp_dir, job.name)

                job_id = self._submit_and_monitor_job(job_path, job.name)

                return job_id

        except Exception as e:
            print(f"Error deploying job to POC environment: {e}")
            raise
        finally:
            # Stop and clean if auto_stop is enabled (we always start our own POC)
            if self.auto_stop:
                self._stop_and_clean_poc()

    def _stop_and_clean_poc(self):
        """Stop POC services and clean workspace with proper wait logic."""
        try:
            project_config, service_config = setup_service_config(self.poc_workspace)

            _stop_poc(
                poc_workspace=self.poc_workspace,
                excluded=[DEFAULT_ADMIN_USER],  # Exclude admin console (consistent with start)
                services_list=[],
            )

            # Wait for services to stop before cleaning
            for _ in range(STOP_POC_TIMEOUT):
                if not is_poc_running(self.poc_workspace, service_config, project_config):
                    break
                time.sleep(1)
            else:
                print(
                    f"Warning: POC still running after {STOP_POC_TIMEOUT} seconds, cannot clean workspace. Skipping cleanup."
                )
                return

            _clean_poc(self.poc_workspace)

        except Exception as e:
            print(f"Warning: Failed to stop and clean POC: {e}")

    def _submit_and_monitor_job(self, job_path: str, job_name: str) -> str:
        """Submit and monitor job via Flare API using a single session.

        Args:
            job_path: Path to the exported job directory.
            job_name: Name of the job for logging.

        Returns:
            str: Job ID returned by the system.
        """
        try:
            # Get the admin startup kit path for POC
            admin_dir = self._get_admin_startup_kit_path()

            # Create secure session with POC admin (reuse for both submit and monitor)
            sess = new_secure_session(
                username=DEFAULT_ADMIN_USER,  # Default POC admin user
                startup_kit_location=admin_dir,
            )

            try:
                # Submit the job
                job_id = sess.submit_job(job_path)
                print(f"Submitted job '{job_name}' with ID: {job_id}")

                # Monitor job based on duration setting
                if self.monitor_duration >= 0:
                    if self.monitor_duration == 0:
                        print("Monitoring job until completion...")
                    else:
                        print(f"Monitoring job for {self.monitor_duration} seconds...")

                    result = sess.monitor_job(job_id, timeout=self.monitor_duration)
                    print(f"Job monitoring completed: {result}")
                else:
                    print("Job submitted, not monitoring (monitor_duration < 0)")

                return job_id

            finally:
                sess.close()

        except Exception as e:
            raise RuntimeError(f"Failed to submit/monitor job via Flare API: {e}")

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
