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
import threading
import time
import uuid
from typing import Optional

from pydantic import BaseModel, conint, model_validator

from nvflare.apis.job_def import DEFAULT_STUDY
from nvflare.apis.utils.format_check import name_check
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.job_config.api import FedJob
from nvflare.recipe.spec import ExecEnv
from nvflare.recipe.utils import collect_non_local_scripts
from nvflare.tool.poc.poc_commands import (
    POC_START_READY_TIMEOUT,
    _clean_poc,
    _start_poc,
    _stop_poc,
    _wait_for_poc_system_ready,
    get_poc_workspace,
    get_prod_dir,
    is_poc_running,
    prepare_poc_provision,
    setup_service_config,
)
from nvflare.tool.poc.service_constants import FlareServiceConstants as SC

from .session_mgr import SessionManager

STOP_POC_TIMEOUT = 10
DEFAULT_ADMIN_USER = "admin@nvidia.com"
_WORKSPACE_BACKUP_PREFIX = ".nvflare-recipe-backup-"


# Internal — not part of the public API
class _PocEnvValidator(BaseModel):
    num_clients: Optional[conint(gt=0)] = None
    clients: Optional[list[str]] = None
    gpu_ids: Optional[list[int]] = None
    use_he: bool = False
    docker_image: Optional[str] = None
    project_conf_path: str = ""
    username: str = DEFAULT_ADMIN_USER
    study: str = DEFAULT_STUDY

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
        if self.clients is None and (self.num_clients is None or self.num_clients <= 0):
            raise ValueError("num_clients must be greater than 0")

        if name_check(self.study, "study")[0]:
            raise ValueError(
                f"study name '{self.study}' contains unsupported characters. Use only lowercase letters, numbers, and hyphens."
            )

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
        docker_image: Optional[str] = None,
        project_conf_path: str = "",
        username: str = DEFAULT_ADMIN_USER,
        study: str = DEFAULT_STUDY,
        extra: Optional[dict] = None,
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
            study (str, optional): Study name to tag submitted jobs. Defaults to "default".
            extra: extra env info.
        """
        super().__init__(extra)
        self.logger = get_obj_logger(self)

        v = _PocEnvValidator(
            num_clients=num_clients,
            clients=clients,
            gpu_ids=gpu_ids,
            use_he=use_he,
            docker_image=docker_image,
            project_conf_path=project_conf_path,
            username=username,
            study=study,
        )

        self.clients = v.clients
        self.num_clients = len(v.clients) if v.clients is not None else v.num_clients
        self.poc_workspace = get_poc_workspace()
        self.gpu_ids = v.gpu_ids or []
        self.use_he = v.use_he
        self.project_conf_path = v.project_conf_path
        self.docker_image = v.docker_image
        self.username = v.username
        self.study = v.study
        self._session_manager = None  # Lazy initialization
        self._session_manager_lock = threading.Lock()
        self._deployment_started = False
        self._workspace_owned = False

    @property
    def deployment_started(self) -> bool:
        """Whether the latest deploy call passed preflight and began POC preparation."""
        return self._deployment_started

    @property
    def workspace_owned(self) -> bool:
        """Whether the latest deploy created or replaced the active POC workspace."""
        return self._workspace_owned

    def _backup_existing_workspace(self) -> Optional[str]:
        """Move a retained workspace aside before provisioning mutates its path."""
        if not os.path.exists(self.poc_workspace):
            return None
        backup = f"{self.poc_workspace}{_WORKSPACE_BACKUP_PREFIX}{uuid.uuid4().hex}"
        os.replace(self.poc_workspace, backup)
        return backup

    def _restore_existing_workspace(self, backup: str) -> None:
        """Discard a partial replacement and atomically restore the retained workspace."""
        if os.path.isdir(self.poc_workspace) and not os.path.islink(self.poc_workspace):
            shutil.rmtree(self.poc_workspace)
        elif os.path.lexists(self.poc_workspace):
            os.remove(self.poc_workspace)
        os.replace(backup, self.poc_workspace)

    def deploy(self, job: FedJob) -> str:
        """Deploy a FedJob to the POC environment.

        Args:
            job (FedJob): The FedJob to deploy.

        Returns:
            str: Job ID.

        Raises:
            ValueError: If scripts do not exist locally.
        """
        # Reset before non-mutating preflight so callers can distinguish a
        # rejected job from an invocation that owns a new POC lifecycle.
        self._deployment_started = False
        self._workspace_owned = False

        # Validate scripts exist locally for POC
        non_local_scripts = collect_non_local_scripts(job)
        if non_local_scripts:
            raise ValueError(
                f"The following scripts do not exist locally: {non_local_scripts}. "
                f"For PocEnv, all scripts must be present on the local machine."
            )

        if self._check_poc_running():
            # Keep the stopped workspace until fresh provisioning succeeds, so
            # a failed replacement can restore its prior logs and results.
            self.stop(clean_up=False)
            if self._check_poc_running():
                raise RuntimeError("Existing POC services could not be stopped")

        self._deployment_started = True
        workspace_backup = self._backup_existing_workspace()
        self.logger.info("Preparing and starting fresh POC services...")
        try:
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
            project_config, service_config = setup_service_config(self.poc_workspace)
            if not _wait_for_poc_system_ready(
                self.poc_workspace,
                project_config,
                service_config,
                services_list=[],
                excluded=[self.username],
                timeout_in_sec=POC_START_READY_TIMEOUT,
            ):
                raise RuntimeError("POC services were started but no server or clients were selected for readiness")
        except BaseException:
            if self._check_poc_running():
                self.stop(clean_up=False)
                if self._check_poc_running():
                    self._workspace_owned = True
                    backup_note = (
                        f"the retained workspace backup remains at {workspace_backup}"
                        if workspace_backup
                        else "there was no retained workspace to restore"
                    )
                    raise RuntimeError(
                        f"POC deployment failed and partially started services could not be stopped; " f"{backup_note}"
                    )
            if workspace_backup:
                try:
                    self._restore_existing_workspace(workspace_backup)
                except Exception as restore_error:
                    # The active path contains only this invocation's partial
                    # replacement. Let the caller clean it, but retain the
                    # backup for manual recovery.
                    self._workspace_owned = True
                    raise RuntimeError(
                        f"POC deployment failed and the retained workspace could not be restored from {workspace_backup}"
                    ) from restore_error
                self._workspace_owned = False
            else:
                self._workspace_owned = True
            raise
        else:
            self._workspace_owned = True
            if workspace_backup:
                shutil.rmtree(workspace_backup, ignore_errors=True)
        self.logger.info("POC services started successfully")

        # Submit job using SessionManager
        return self._get_session_manager().submit_job(job)

    def _check_poc_running(self) -> bool:
        """Check if POC services are currently running.

        Returns:
            bool: True if POC is running, False otherwise.
        """
        try:
            project_config, service_config = setup_service_config(self.poc_workspace)
        except Exception:
            # POC workspace is not initialized yet, so we don't need to stop and clean it
            return False

        if not is_poc_running(self.poc_workspace, service_config, project_config):
            return False

        return True

    def stop(self, clean_up: bool = False) -> None:
        """Try to stop and clean existing POC.

        This method is idempotent - safe to call multiple times.

        Args:
            clean_up (bool, optional): Whether to clean the POC workspace. Defaults to False.
        """
        # Check if already stopped (idempotent)
        if not self._check_poc_running():
            # POC already stopped or workspace doesn't exist
            if clean_up and os.path.exists(self.poc_workspace):
                self.logger.info(f"Removing POC workspace: {self.poc_workspace}")
                shutil.rmtree(self.poc_workspace, ignore_errors=True)
            self._session_manager = None  # Clear stale session manager
            return

        try:
            project_config, service_config = setup_service_config(self.poc_workspace)
            self.logger.info("Stopping existing POC services...")
            _stop_poc(
                poc_workspace=self.poc_workspace,
                excluded=[self.username],  # Exclude admin console (consistent with start)
                services_list=[],
            )
            count = 0
            poc_running = True
            while count < STOP_POC_TIMEOUT:
                try:
                    if not is_poc_running(self.poc_workspace, service_config, project_config):
                        poc_running = False
                        break
                except Exception:
                    poc_running = False
                    break
                time.sleep(1)
                count += 1

            if clean_up:
                if poc_running:
                    self.logger.warning(
                        f"POC still running after {STOP_POC_TIMEOUT} seconds, cannot clean workspace. Skipping cleanup."
                    )
                else:
                    try:
                        _clean_poc(self.poc_workspace)
                    except Exception as e:
                        self.logger.debug(f"Failed to clean POC: {e}")
        except Exception as e:
            self.logger.warning(f"Failed to stop and clean existing POC: {e}")
        finally:
            self._session_manager = None  # Clear stale session manager

    def get_job_status(self, job_id: str) -> Optional[str]:
        """Get the status of a job.

        Args:
            job_id: The job ID to check status for.

        Returns:
            Optional[str]: The status of the job, or None if not available.
        """
        return self._get_session_manager().get_job_status(job_id)

    def abort_job(self, job_id: str) -> None:
        """Abort a running job.

        Args:
            job_id: The job ID to abort.
        """
        self._get_session_manager().abort_job(job_id)

    def get_job_result(self, job_id: str, timeout: float = 0.0) -> Optional[str]:
        """Get the result workspace of a job.

        Args:
            job_id: The job ID to get results for.
            timeout: The timeout for the job to complete. Defaults to 0.0 (no timeout).

        Returns:
            Optional[str]: The result workspace path if job completed, None otherwise.
        """
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
            raise RuntimeError(f"Failed to locate admin startup kit: {e}") from e

    def _get_session_manager(self) -> SessionManager:
        """Get or create SessionManager with lazy initialization (thread-safe)."""
        with self._session_manager_lock:
            if self._session_manager is None:
                session_params = {
                    "username": self.username,
                    "startup_kit_location": self._get_admin_startup_kit_path(),
                    "timeout": self.get_extra_prop("login_timeout", 10),
                    "study": self.study,
                }
                self._session_manager = SessionManager(session_params)
            return self._session_manager
