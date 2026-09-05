# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

import errno
import fcntl
import os
import shutil
import stat
import subprocess
import tempfile
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
    _docker_cli_env,
    _is_live_pid_file,
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
POC_READY_POLL_INTERVAL = 0.2
POC_READY_STABLE_INTERVAL = 2.0
DEFAULT_ADMIN_USER = "admin@nvidia.com"
_RECIPE_WORKSPACE_SUFFIX = ".recipe-"


def _recipe_runtime_lock_path() -> str:
    """Return the per-user host lock that serializes Recipe POC runtimes."""
    return os.path.join(tempfile.gettempdir(), f".nvflare-recipe-poc-{os.geteuid()}.lock")


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
    processes representing the server, clients, and admin console. Each deployment
    uses a new Recipe-owned workspace beside the configured CLI POC workspace.
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
        # The configured POC path belongs to the reusable CLI workflow. Recipe
        # executions use unique sibling workspaces so they never replace or
        # restore user-retained POC state.
        self._poc_workspace_root = os.path.normpath(get_poc_workspace())
        self.poc_workspace = self._new_poc_workspace()
        self.gpu_ids = v.gpu_ids or []
        self.use_he = v.use_he
        self.project_conf_path = v.project_conf_path
        self.docker_image = v.docker_image
        self.username = v.username
        self.study = v.study
        self._session_manager = None  # Lazy initialization
        self._session_manager_lock = threading.Lock()
        self._workspace_used = False
        self._runtime_lock_file = None
        self._deployment_lock = threading.Lock()

    def _new_poc_workspace(self) -> str:
        """Return a unique Recipe-owned workspace beside the configured POC path."""
        return f"{self._poc_workspace_root}{_RECIPE_WORKSPACE_SUFFIX}{uuid.uuid4().hex}"

    def _is_recipe_workspace(self, workspace: str) -> bool:
        """Return whether the path has this environment's exact sibling-and-UUID form."""
        root = os.path.abspath(self._poc_workspace_root)
        candidate = os.path.abspath(workspace)
        if os.path.dirname(candidate) != os.path.dirname(root):
            return False
        prefix = f"{os.path.basename(root)}{_RECIPE_WORKSPACE_SUFFIX}"
        candidate_name = os.path.basename(candidate)
        if not candidate_name.startswith(prefix):
            return False
        identifier = candidate_name[len(prefix) :]
        return len(identifier) == 32 and all(c in "0123456789abcdef" for c in identifier.lower())

    def _clean_up_failed_deployment(self) -> None:
        """Stop a failed deployment and verify its per-run workspace was cleaned."""
        if not self._is_recipe_workspace(self.poc_workspace):
            raise RuntimeError(f"refusing to clean unmanaged POC workspace {self.poc_workspace}")
        # deploy() already holds the instance lifecycle guard. Use the private
        # implementation so failure cleanup cannot deadlock on that guard.
        self._stop(clean_up=True)
        if self._check_poc_running():
            raise RuntimeError("POC services remain running")
        if os.path.exists(self.poc_workspace):
            raise RuntimeError("the per-run POC workspace could not be removed")

    def _is_poc_workspace_running(self, workspace: str) -> bool:
        """Return whether any managed service is running in a POC workspace."""
        try:
            project_config, service_config = setup_service_config(workspace)
        except Exception:
            return False
        return bool(self._running_services(project_config, service_config, workspace))

    def _acquire_runtime_lock(self) -> None:
        """Claim the host's single Recipe-managed POC runtime slot."""
        if self._runtime_lock_file is not None:
            return

        lock_path = _recipe_runtime_lock_path()
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(lock_path, flags, 0o600)
        try:
            lock_stat = os.fstat(fd)
            if not stat.S_ISREG(lock_stat.st_mode) or lock_stat.st_uid != os.geteuid():
                raise RuntimeError(f"Refusing to use unsafe Recipe POC runtime lock {lock_path}")
            os.fchmod(fd, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as e:
                if e.errno in (errno.EACCES, errno.EAGAIN):
                    raise RuntimeError(
                        "Another Recipe PocEnv deployment is active on this host. "
                        "Stop it before starting this deployment."
                    ) from e
                raise
            lock_file = os.fdopen(fd, "r+")
            fd = None
            previous_workspace = lock_file.read().strip()
            if previous_workspace and self._is_poc_workspace_running(previous_workspace):
                raise RuntimeError(
                    f"A prior Recipe PocEnv deployment is still active at {previous_workspace}. "
                    "Stop its services and remove the workspace manually before starting another deployment."
                )
            self._runtime_lock_file = lock_file
        except BaseException:
            if fd is not None:
                os.close(fd)
            else:
                lock_file.close()
            raise

    def _record_runtime_workspace(self) -> None:
        """Durably record the workspace whose services own the runtime slot."""
        lock_file = self._runtime_lock_file
        if lock_file is None:
            raise RuntimeError("Recipe POC runtime lock is not held")
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(os.path.abspath(self.poc_workspace))
        lock_file.flush()
        os.fsync(lock_file.fileno())

    def _release_runtime_lock(self, clear_workspace: bool = False) -> None:
        """Release this environment's Recipe POC runtime slot, if held."""
        lock_file = self._runtime_lock_file
        if lock_file is None:
            return
        self._runtime_lock_file = None
        try:
            if clear_workspace:
                lock_file.seek(0)
                lock_file.truncate()
                lock_file.flush()
                os.fsync(lock_file.fileno())
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            lock_file.close()

    @staticmethod
    def _is_docker_service_running(service_name: str) -> bool:
        """Return whether Docker reports the named POC container as running."""
        try:
            result = subprocess.run(
                ["docker", "inspect", "--format", "{{.State.Running}}", service_name],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
                env=_docker_cli_env(),
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RuntimeError(f"Could not determine Docker POC service state for {service_name!r}") from error

        state = result.stdout.strip().lower()
        if result.returncode == 0 and state in ("true", "false"):
            return state == "true"

        error_message = (getattr(result, "stderr", "") or result.stdout).strip()
        if result.returncode != 0 and any(
            marker in error_message.lower() for marker in ("no such object", "no such container")
        ):
            return False
        detail = f": {error_message}" if error_message else ""
        raise RuntimeError(
            f"Could not determine Docker POC service state for {service_name!r} "
            f"(docker inspect exited {result.returncode}){detail}"
        )

    @staticmethod
    def _running_services(project_config: dict, service_config: dict, poc_workspace: str) -> list[str]:
        """Return managed POC services whose local process is still alive."""
        service_names = [service_config[SC.FLARE_SERVER], *service_config.get(SC.FLARE_CLIENTS, [])]
        if service_config.get(SC.IS_DOCKER_RUN):
            return [name for name in service_names if PocEnv._is_docker_service_running(name)]

        project_name = project_config.get("name")
        prod_dir = get_prod_dir(poc_workspace, project_name)
        running = []
        for service_name in service_names:
            service_dir = os.path.join(prod_dir, service_name)
            if _is_live_pid_file(os.path.join(service_dir, "pid.fl")) or _is_live_pid_file(
                os.path.join(service_dir, "daemon_pid.fl")
            ):
                running.append(service_name)
        return running

    def _wait_for_services_ready(self, project_config: dict, service_config: dict) -> None:
        """Wait until every managed service remains alive through a stabilization interval."""
        expected_services = [service_config[SC.FLARE_SERVER], *service_config.get(SC.FLARE_CLIENTS, [])]
        if not expected_services:
            raise RuntimeError("POC provisioning did not configure a server or clients")

        deadline = time.monotonic() + POC_START_READY_TIMEOUT
        all_running_since = None
        running_services = []
        while time.monotonic() < deadline:
            running_services = self._running_services(project_config, service_config, self.poc_workspace)
            if set(running_services) == set(expected_services):
                if all_running_since is None:
                    all_running_since = time.monotonic()
                elif time.monotonic() - all_running_since >= POC_READY_STABLE_INTERVAL:
                    return
            else:
                all_running_since = None
            time.sleep(POC_READY_POLL_INTERVAL)

        missing_services = sorted(set(expected_services) - set(running_services))
        raise RuntimeError(
            f"POC services did not remain healthy within {POC_START_READY_TIMEOUT} seconds; "
            f"not running: {', '.join(missing_services)}"
        )

    def deploy(self, job: FedJob) -> str:
        """Deploy a FedJob to the POC environment.

        Args:
            job (FedJob): The FedJob to deploy.

        Returns:
            str: Job ID.

        Raises:
            ValueError: If scripts do not exist locally.
        """
        if not self._deployment_lock.acquire(blocking=False):
            raise RuntimeError("This PocEnv already has a deployment in progress")
        try:
            return self._deploy(job)
        finally:
            self._deployment_lock.release()

    def _deploy(self, job: FedJob) -> str:
        """Perform one deployment while the instance deployment guard is held."""
        # Validate scripts exist locally for POC
        non_local_scripts = collect_non_local_scripts(job)
        if non_local_scripts:
            raise ValueError(
                f"The following scripts do not exist locally: {non_local_scripts}. "
                f"For PocEnv, all scripts must be present on the local machine."
            )

        if self._workspace_used and self._check_poc_running():
            raise RuntimeError("This PocEnv already has a running deployment; stop it before deploying another job")

        # Recipe POC workspaces have isolated files but share host ports and
        # Docker participant names. The process-held lock closes the race
        # between checking those resources and starting services. It is
        # released only after this deployment's services have stopped.
        self._acquire_runtime_lock()
        try:
            if self._is_poc_workspace_running(self._poc_workspace_root):
                raise RuntimeError(
                    f"The configured CLI POC deployment is running at {self._poc_workspace_root}. "
                    "Stop it with 'nvflare poc stop' before starting a Recipe PocEnv deployment."
                )
        except BaseException:
            self._release_runtime_lock()
            raise

        if self._workspace_used:
            self.poc_workspace = self._new_poc_workspace()
            self._session_manager = None
        try:
            self._record_runtime_workspace()
        except BaseException:
            self._release_runtime_lock()
            raise
        self._workspace_used = True

        self.logger.info(f"Preparing and starting POC services in new workspace: {self.poc_workspace}")
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
            self._wait_for_services_ready(project_config, service_config)
            if not _wait_for_poc_system_ready(
                self.poc_workspace,
                project_config,
                service_config,
                services_list=[],
                excluded=[self.username],
                timeout_in_sec=POC_START_READY_TIMEOUT,
            ):
                raise RuntimeError("POC services were started but no server or clients were selected for readiness")
            # Successful submission also proves that the admin connection is
            # ready after the process/container and client-registration checks.
            job_id = self._get_session_manager().submit_job(job)
        except BaseException as deployment_error:
            # This path is unique to the current Recipe execution, so failure
            # cleanup cannot delete a retained CLI workspace or a prior run.
            try:
                self._clean_up_failed_deployment()
            except Exception as cleanup_error:
                raise RuntimeError(
                    f"POC deployment failed ({deployment_error}); cleanup could not be completed safely "
                    f"for the per-run workspace {self.poc_workspace}: {cleanup_error}. Stop any remaining POC "
                    "services and remove this workspace manually."
                ) from cleanup_error
            raise
        self.logger.info("POC services started successfully")
        return job_id

    def _check_poc_running(self) -> bool:
        """Check if POC services are currently running.

        Returns:
            bool: True if POC is running, False otherwise.
        """
        return self._is_poc_workspace_running(self.poc_workspace)

    def stop(self, clean_up: bool = False) -> None:
        """Try to stop and clean existing POC.

        This method is idempotent - safe to call multiple times.

        Args:
            clean_up (bool, optional): Whether to clean the POC workspace. Defaults to False.
        """
        # Wait for an in-progress deployment to finish before inspecting
        # services or clearing its durable runtime record.
        with self._deployment_lock:
            self._stop(clean_up)

    def _stop(self, clean_up: bool = False) -> None:
        """Stop POC while the caller holds the instance lifecycle guard."""
        # Check if already stopped (idempotent)
        if not self._check_poc_running():
            # POC already stopped or workspace doesn't exist
            if clean_up and os.path.exists(self.poc_workspace):
                self.logger.info(f"Removing POC workspace: {self.poc_workspace}")
                shutil.rmtree(self.poc_workspace, ignore_errors=True)
            self._session_manager = None  # Clear stale session manager
            self._release_runtime_lock(clear_workspace=True)
            return

        try:
            project_config, service_config = setup_service_config(self.poc_workspace)
            self.logger.info("Stopping existing POC services...")
            # Prefer the coordinated server shutdown while it is reachable. If
            # the server exited during startup, stop any surviving local client
            # processes directly so the per-run workspace can be cleaned safely.
            services_list = []
            if service_config.get(SC.IS_DOCKER_RUN) or not is_poc_running(
                self.poc_workspace, service_config, project_config
            ):
                services_list = self._running_services(project_config, service_config, self.poc_workspace)
            _stop_poc(
                poc_workspace=self.poc_workspace,
                excluded=[self.username],  # Exclude admin console (consistent with start)
                services_list=services_list,
            )
            count = 0
            poc_running = True
            poc_state_error = None
            while count < STOP_POC_TIMEOUT:
                try:
                    if not self._running_services(project_config, service_config, self.poc_workspace):
                        poc_running = False
                        break
                except Exception as state_error:
                    poc_state_error = state_error
                    self.logger.warning(f"Could not verify whether POC services stopped: {state_error}")
                    # Preserve the workspace when service state is unknown. It
                    # contains the configuration needed for manual cleanup.
                    poc_running = True
                    break
                time.sleep(1)
                count += 1

            if clean_up:
                if poc_running:
                    reason = (
                        f"service state could not be verified ({poc_state_error})"
                        if poc_state_error
                        else f"services are still running after {STOP_POC_TIMEOUT} seconds"
                    )
                    self.logger.warning(
                        f"POC {reason}; preserving workspace {self.poc_workspace}. "
                        "Stop any remaining services and remove it manually."
                    )
                else:
                    try:
                        _clean_poc(self.poc_workspace)
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to clean POC workspace {self.poc_workspace}: {e}. Remove it manually."
                        )
            if not poc_running:
                self._release_runtime_lock(clear_workspace=True)
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
