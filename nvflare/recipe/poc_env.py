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
    is_poc_running,
    prepare_poc_provision,
    setup_service_config,
    validate_poc_workspace,
)

from .spec import ExecEnv


class POCEnv(ExecEnv):
    """Proof of Concept execution environment for local testing and development.

    This environment sets up a POC deployment on a single machine with multiple
    processes representing the server, clients, and admin console.
    """

    def __init__(
        self,
        poc_workspace: Optional[str] = None,
        num_clients: int = 2,
        gpu_ids: Optional[List[int]] = None,
        auto_stop: bool = True,
        auto_clean: bool = True,
        monitor_duration: int = 0,
    ):
        """Initialize POC execution environment.

        Args:
            poc_workspace: POC workspace directory. If None, uses default from environment.
            num_clients: Number of clients to use in POC mode.
            gpu_ids: List of GPU IDs to assign to clients. If None, uses CPU only.
            auto_stop: Whether to automatically stop POC services after job completion.
            auto_clean: Whether to automatically clean POC workspace after job completion (implies auto_stop).
            monitor_duration: Duration to monitor job execution (in seconds). 0 means wait until completion, negative means no monitoring.
        """
        self.poc_workspace = poc_workspace or get_poc_workspace()
        self.num_clients = num_clients
        self.gpu_ids = gpu_ids or []
        self.auto_stop = auto_stop
        self.auto_clean = auto_clean
        self.monitor_duration = monitor_duration
        self._poc_started_by_us = False

    def deploy(self, job: FedJob):
        """Deploy a FedJob to the POC environment.

        Args:
            job: The FedJob to deploy.

        Returns:
            str: Job ID or deployment result.
        """
        try:
            # Check if POC is running, prepare and start if needed
            if not self.is_running():
                print("POC services not running")
                # Only prepare workspace if POC is not running
                self._ensure_poc_workspace()
                print("Starting POC services...")
                self._start_poc_services()
                self._poc_started_by_us = True
            else:
                print("POC services already running")

            # Submit job using Flare API like ProdEnv
            with tempfile.TemporaryDirectory() as temp_dir:
                job.export_job(temp_dir)
                job_path = os.path.join(temp_dir, job.name)

                # Submit job via Flare API
                job_id = self._submit_job_via_api(job_path)
                print(f"Submitted job '{job.name}' with ID: {job_id}")

                # Monitor job based on duration setting
                if self.monitor_duration >= 0:
                    if self.monitor_duration == 0:
                        print("Monitoring job until completion...")
                    else:
                        print(f"Monitoring job for {self.monitor_duration} seconds...")
                    self._monitor_job(job_id, self.monitor_duration)
                else:
                    print("Job submitted, not monitoring (monitor_duration < 0)")

                return job_id

        except Exception as e:
            print(f"Error deploying job to POC environment: {e}")
            raise
        finally:
            # Handle cleanup if we started the services
            if self._poc_started_by_us:
                # Stop services if auto_stop OR auto_clean (cleaning requires stopping)
                if self.auto_stop or self.auto_clean:
                    self._stop_poc_services()

                # Clean workspace if requested
                if self.auto_clean:
                    # Wait for services to shut down with timeout
                    timeout = 10  # Maximum 10 seconds
                    for i in range(timeout):
                        if not self.is_running():
                            self._clean_poc_workspace()
                            break
                        time.sleep(1)
                    else:
                        print(
                            f"Warning: POC still running after {timeout} seconds, cannot clean workspace. Skipping cleanup."
                        )

    def _ensure_poc_workspace(self):
        """Ensure POC workspace exists and is properly configured."""
        workspace_exists = os.path.exists(self.poc_workspace)
        workspace_valid = False

        if workspace_exists:
            try:
                project_config, service_config = setup_service_config(self.poc_workspace)
                validate_poc_workspace(self.poc_workspace, service_config, project_config)
                workspace_valid = True
            except Exception:
                # Workspace exists but is invalid
                workspace_valid = False

        if not workspace_exists or not workspace_valid:
            print(f"POC workspace not ready, preparing it at: {self.poc_workspace}")
            self._prepare_poc_workspace()

    def _prepare_poc_workspace(self):
        """Prepare POC workspace by running poc prepare (non-interactive)."""
        try:
            # Generate client names if not provided
            clients = [f"site-{i + 1}" for i in range(self.num_clients)]

            # Use non-interactive prepare_poc_provision instead of _prepare_poc
            prepare_poc_provision(
                clients=clients,
                number_of_clients=self.num_clients,
                workspace=self.poc_workspace,
                docker_image=None,  # Use default
                use_he=False,  # No homomorphic encryption by default
                project_conf_path="",  # Use default project config
                examples_dir=None,  # No examples directory
            )
            print(f"POC workspace prepared successfully at: {self.poc_workspace}")

        except Exception as e:
            raise RuntimeError(f"Failed to prepare POC workspace: {e}")

    def _start_poc_services(self):
        """Start POC services (server, clients) excluding admin console."""
        try:
            print("Starting POC services (excluding admin console)...")
            _start_poc(
                poc_workspace=self.poc_workspace,
                gpu_ids=self.gpu_ids,
                excluded=["admin@nvidia.com"],  # Exclude admin console like --ex admin@nvidia.com
                services_list=[],
            )
            print("POC services started successfully")

            # Give services time to start up
            time.sleep(3)

        except Exception as e:
            raise RuntimeError(f"Failed to start POC services: {e}")

    def _stop_poc_services(self):
        """Stop POC services (excluding admin console)."""
        try:
            print("Stopping POC services...")
            _stop_poc(
                poc_workspace=self.poc_workspace,
                excluded=["admin@nvidia.com"],  # Exclude admin console (consistent with start)
                services_list=[],
            )
            self._poc_started_by_us = False
            print("POC services stopped successfully")

        except Exception as e:
            print(f"Warning: Failed to stop POC services cleanly: {e}")

    def _clean_poc_workspace(self):
        """Clean up POC workspace by removing all generated files."""
        try:
            print(f"Cleaning POC workspace: {self.poc_workspace}")
            _clean_poc(self.poc_workspace)
            print("POC workspace cleaned successfully")

        except Exception as e:
            print(f"Warning: Failed to clean POC workspace: {e}")

    def _submit_job_via_api(self, job_path: str) -> str:
        """Submit job via Flare API to POC environment.

        Args:
            job_path: Path to the exported job directory.

        Returns:
            str: Job ID returned by the system.
        """
        try:
            # Get the admin startup kit path for POC
            admin_dir = self._get_admin_startup_kit_path()

            # Create secure session with POC admin
            sess = new_secure_session(
                username="admin@nvidia.com",  # Default POC admin user
                startup_kit_location=admin_dir,
            )

            try:
                # Submit the job
                job_id = sess.submit_job(job_path)
                return job_id
            finally:
                sess.close()

        except Exception as e:
            raise RuntimeError(f"Failed to submit job via Flare API: {e}")

    def _monitor_job(self, job_id: str, duration: int):
        """Monitor job progress for specified duration.

        Args:
            job_id: The job ID to monitor.
            duration: Duration to monitor in seconds.
        """
        try:
            admin_dir = self._get_admin_startup_kit_path()
            sess = new_secure_session(
                username="admin@nvidia.com",
                startup_kit_location=admin_dir,
            )

            try:
                # Monitor job with timeout
                result = sess.monitor_job(job_id, timeout=duration)
                print(f"Job monitoring completed: {result}")
            finally:
                sess.close()

        except Exception as e:
            print(f"Warning: Failed to monitor job: {e}")

    def _get_admin_startup_kit_path(self) -> str:
        """Get the path to the admin startup kit for POC.

        Returns:
            str: Path to admin startup kit directory.
        """
        try:
            project_config, service_config = setup_service_config(self.poc_workspace)
            project_name = project_config.get("name", "example_project")

            # POC admin directory structure: {workspace}/{project_name}/prod_00/admin@nvidia.com
            admin_dir = os.path.join(self.poc_workspace, project_name, "prod_00", "admin@nvidia.com")

            if not os.path.exists(admin_dir):
                raise RuntimeError(f"Admin startup kit not found at: {admin_dir}")

            return admin_dir

        except Exception as e:
            raise RuntimeError(f"Failed to locate admin startup kit: {e}")

    def start_services(self):
        """Manually start POC services."""
        if not self.is_running():
            self._start_poc_services()
            self._poc_started_by_us = True

    def stop_services(self):
        """Manually stop POC services."""
        if self.is_running():
            self._stop_poc_services()

    def clean_workspace(self):
        """Manually clean POC workspace."""
        self._clean_poc_workspace()

    def is_running(self) -> bool:
        """Check if POC services are currently running.

        Returns:
            bool: True if POC services are running, False otherwise.
        """
        try:
            project_config, service_config = setup_service_config(self.poc_workspace)
            return is_poc_running(self.poc_workspace, service_config, project_config)
        except Exception:
            return False
