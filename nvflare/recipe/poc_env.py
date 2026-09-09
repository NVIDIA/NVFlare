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

import os
import shutil
import subprocess
import threading
import time
import uuid
from typing import Optional

from pydantic import BaseModel, conint, model_validator

from nvflare.apis.job_def import DEFAULT_STUDY
from nvflare.apis.utils.format_check import name_check
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.job_config.api import FedJob
from nvflare.lighter.utils import load_yaml
from nvflare.recipe.spec import ExecEnv
from nvflare.recipe.utils import collect_non_local_scripts
from nvflare.tool.poc.poc_commands import (
    POC_DEFAULT_ADMIN_PORT,
    POC_DEFAULT_FED_LEARN_PORT,
    POC_START_READY_TIMEOUT,
    _build_poc_port_preflight,
    _docker_cli_env,
    _get_docker_endpoint,
    _is_live_pid_file,
    _start_poc,
    _stop_poc,
    _wait_for_poc_system_ready,
    get_poc_workspace,
    get_prod_dir,
    is_docker_run,
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
_RECIPE_DOCKER_CONTAINER_NAMES = "_recipe_docker_container_names"


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
        self._deployment_started = False
        self._services_may_have_started = False
        self._docker_container_names = {}
        self._docker_network_name = None
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

    def _remove_recipe_workspace(self) -> None:
        """Remove only this environment's Recipe-owned workspace."""
        if not self._is_recipe_workspace(self.poc_workspace):
            raise RuntimeError(f"refusing to remove unmanaged POC workspace {self.poc_workspace}")
        shutil.rmtree(self.poc_workspace)

    def _configure_docker_identities(self, service_config: dict) -> None:
        """Assign unique per-workspace Docker identities without changing FL identities."""
        self._docker_container_names = {}
        self._docker_network_name = None
        if not service_config.get(SC.IS_DOCKER_RUN):
            return
        if not self._is_recipe_workspace(self.poc_workspace):
            raise RuntimeError(f"cannot create Recipe Docker names for unmanaged workspace {self.poc_workspace}")

        deployment_id = os.path.basename(self.poc_workspace).rsplit(_RECIPE_WORKSPACE_SUFFIX, 1)[1]
        service_names = [service_config[SC.FLARE_SERVER], *service_config.get(SC.FLARE_CLIENTS, [])]
        self._docker_container_names = {
            service_name: f"nvflare-recipe-{deployment_id}-{uuid.uuid5(uuid.NAMESPACE_OID, service_name).hex}"
            for service_name in service_names
        }
        self._docker_network_name = f"nvflare-recipe-{deployment_id}"

    def _remove_docker_network(self, deadline: Optional[float] = None) -> None:
        """Remove the network, allowing time for auto-removed job containers to detach."""
        if not self._docker_network_name:
            return
        if deadline is None:
            deadline = time.monotonic() + STOP_POC_TIMEOUT
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"POC shutdown deadline expired before removing Docker network {self._docker_network_name!r}"
            )
        docker_env = _docker_cli_env()
        error_message = None
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                detail = f": {error_message}" if error_message else ""
                raise RuntimeError(
                    f"could not remove Docker network {self._docker_network_name!r} "
                    f"before the POC shutdown deadline{detail}"
                )
            try:
                result = subprocess.run(
                    ["docker", "network", "rm", self._docker_network_name],
                    capture_output=True,
                    text=True,
                    timeout=min(5, remaining),
                    check=False,
                    env=docker_env,
                )
            except (OSError, subprocess.TimeoutExpired) as error:
                raise RuntimeError(f"could not remove Docker network {self._docker_network_name!r}") from error

            error_message = (getattr(result, "stderr", "") or result.stdout).strip()
            if result.returncode == 0 or any(
                marker in error_message.lower() for marker in ("no such network", "not found")
            ):
                self._docker_network_name = None
                return
            if "active endpoints" not in error_message.lower():
                detail = f": {error_message}" if error_message else ""
                raise RuntimeError(
                    f"could not remove Docker network {self._docker_network_name!r} "
                    f"(docker network rm exited {result.returncode}){detail}"
                )
            # Parent shutdown can finish before --rm job containers release their
            # network endpoints. Let Docker finish without disconnecting containers
            # or removing their workspace while they may still be using it.
            time.sleep(min(POC_READY_POLL_INTERVAL, max(0, deadline - time.monotonic())))

    def _with_docker_container_names(self, workspace: str, service_config: dict) -> dict:
        """Attach this deployment's Docker-name mapping to current-workspace service state."""
        if (
            self._docker_container_names
            and os.path.abspath(workspace) == os.path.abspath(self.poc_workspace)
            and service_config.get(SC.IS_DOCKER_RUN)
        ):
            return {**service_config, _RECIPE_DOCKER_CONTAINER_NAMES: self._docker_container_names}
        return service_config

    def _raise_unknown_service_state(self, workspace: str, error: Exception) -> None:
        """Raise recovery guidance when a Recipe workspace cannot be inspected."""
        raise RuntimeError(
            f"Could not determine service state for the Recipe PocEnv workspace {workspace}: "
            f"{error}. Stop any remaining services, then remove the workspace manually."
        ) from error

    def _clean_up_failed_deployment(self) -> None:
        """Stop a failed deployment and verify its per-run workspace was cleaned."""
        if not self._is_recipe_workspace(self.poc_workspace):
            raise RuntimeError(f"refusing to clean unmanaged POC workspace {self.poc_workspace}")
        if not self._services_may_have_started:
            if os.path.exists(self.poc_workspace):
                self._remove_recipe_workspace()
            self._session_manager = None
            return
        # deploy() already holds the instance lifecycle guard. Use the private
        # implementation so failure cleanup cannot deadlock on that guard.
        self._stop(clean_up=True)
        if self._check_poc_running():
            raise RuntimeError("POC services remain running")
        if os.path.exists(self.poc_workspace):
            raise RuntimeError("the per-run POC workspace could not be removed")

    def _is_poc_workspace_running(self, workspace: str, fail_if_unknown: bool = False) -> bool:
        """Return whether any managed service is running in a POC workspace."""
        try:
            project_config, service_config = setup_service_config(workspace)
        except Exception as e:
            if fail_if_unknown:
                self._raise_unknown_service_state(workspace, e)
            return False
        service_config = self._with_docker_container_names(workspace, service_config)
        try:
            return bool(self._running_services(project_config, service_config, workspace))
        except Exception as e:
            if fail_if_unknown:
                self._raise_unknown_service_state(workspace, e)
            raise

    @staticmethod
    def _get_docker_service_state(service_name: str) -> Optional[bool]:
        """Return a Docker POC container's running state, or None if it does not exist."""
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
            return None
        detail = f": {error_message}" if error_message else ""
        raise RuntimeError(
            f"Could not determine Docker POC service state for {service_name!r} "
            f"(docker inspect exited {result.returncode}){detail}"
        )

    @staticmethod
    def _is_docker_service_running(service_name: str) -> bool:
        """Return whether Docker reports the named POC container as running."""
        return PocEnv._get_docker_service_state(service_name) is True

    @staticmethod
    def _docker_daemon_is_local() -> bool:
        """Return whether Docker startup targets the local Unix-socket daemon."""
        endpoint = _get_docker_endpoint(_docker_cli_env())
        return not endpoint or endpoint.startswith("unix://")

    @staticmethod
    def _ensure_ports_available(project_config: dict, is_docker: bool) -> None:
        """Reject configured ports already owned by another local deployment."""
        # A loopback bind is authoritative only when Docker uses the local
        # Unix-socket daemon. Remote-daemon conflicts fail at Docker startup.
        if not is_docker or PocEnv._docker_daemon_is_local():
            port_conflicts = _build_poc_port_preflight(project_config).get("conflicts", [])
            if port_conflicts:
                details = "; ".join(conflict.get("message", str(conflict)) for conflict in port_conflicts)
                raise RuntimeError(
                    f"POC service port preflight failed: {details}. Stop the process or other Recipe PocEnv using "
                    "the configured port(s) before starting this PocEnv; 'nvflare poc stop' only stops the "
                    "configured CLI deployment."
                )

    @staticmethod
    def _ensure_shared_resources_available(project_config: dict, service_config: dict) -> None:
        """Reject ports or Docker names already owned by another POC deployment."""
        is_docker = service_config.get(SC.IS_DOCKER_RUN)
        PocEnv._ensure_ports_available(project_config, is_docker)

        if is_docker:
            service_names = [service_config[SC.FLARE_SERVER], *service_config.get(SC.FLARE_CLIENTS, [])]
            container_names = service_config.get(_RECIPE_DOCKER_CONTAINER_NAMES, {})
            existing_services = [
                container_names.get(service_name, service_name)
                for service_name in service_names
                if PocEnv._get_docker_service_state(container_names.get(service_name, service_name)) is not None
            ]
            if existing_services:
                raise RuntimeError(
                    "Docker POC participant container name(s) already exist: "
                    f"{', '.join(existing_services)}. Stop and remove them before starting this PocEnv."
                )

    def _preflight_ports_before_provision(self) -> None:
        """Check intended ports before provisioning consumes this one-shot environment."""
        if self.project_conf_path:
            project_config = load_yaml(self.project_conf_path)
            is_docker = is_docker_run(project_config)
        else:
            project_config = {
                "participants": [
                    {
                        "name": "server",
                        "type": "server",
                        "fed_learn_port": POC_DEFAULT_FED_LEARN_PORT,
                        "admin_port": POC_DEFAULT_ADMIN_PORT,
                    }
                ]
            }
            is_docker = bool(self.docker_image)
        self._ensure_ports_available(project_config, is_docker)

    @staticmethod
    def _running_services(project_config: dict, service_config: dict, poc_workspace: str) -> list[str]:
        """Return managed POC services whose local process is still alive."""
        service_names = [service_config[SC.FLARE_SERVER], *service_config.get(SC.FLARE_CLIENTS, [])]
        if service_config.get(SC.IS_DOCKER_RUN):
            container_names = service_config.get(_RECIPE_DOCKER_CONTAINER_NAMES, {})
            return [
                name for name in service_names if PocEnv._is_docker_service_running(container_names.get(name, name))
            ]

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
        if self._deployment_started:
            raise RuntimeError("This PocEnv has already been used; create a new PocEnv for another deployment")

        # Validate scripts exist locally for POC
        non_local_scripts = collect_non_local_scripts(job)
        if non_local_scripts:
            raise ValueError(
                f"The following scripts do not exist locally: {non_local_scripts}. "
                f"For PocEnv, all scripts must be present on the local machine."
            )

        if self._is_poc_workspace_running(self._poc_workspace_root):
            raise RuntimeError(
                f"The configured CLI POC deployment is running at {self._poc_workspace_root}. "
                "Stop it with 'nvflare poc stop' before starting a Recipe PocEnv deployment."
            )
        self._preflight_ports_before_provision()
        self.logger.info(f"Preparing and starting POC services in new workspace: {self.poc_workspace}")
        try:
            # A PocEnv owns one provisioning lifecycle. Mark it consumed at
            # the point provisioning begins, even if this attempt later fails.
            self._deployment_started = True
            prepare_poc_provision(
                clients=self.clients or [],  # Empty list if None, let prepare_clients generate
                number_of_clients=self.num_clients,
                workspace=self.poc_workspace,
                docker_image=self.docker_image,
                use_he=self.use_he,
                project_conf_path=self.project_conf_path,
                examples_dir=None,
            )
            project_config, service_config = setup_service_config(self.poc_workspace)
            self._configure_docker_identities(service_config)
            service_config = self._with_docker_container_names(self.poc_workspace, service_config)
            # Recipe workspaces isolate files, while POC servers still bind
            # configured ports. Docker deployments additionally get unique
            # container and network identities so cleanup cannot cross runs.
            self._ensure_shared_resources_available(project_config, service_config)
            # Startup can leave some services running even if it raises. From
            # this point onward, missing service metadata is an unknown state
            # and cleanup must preserve the workspace.
            self._services_may_have_started = True
            start_args = {
                "poc_workspace": self.poc_workspace,
                "gpu_ids": self.gpu_ids,
                "excluded": [self.username],
                "services_list": [],
            }
            if self._docker_container_names:
                start_args["docker_container_names"] = self._docker_container_names
                start_args["docker_network_name"] = self._docker_network_name
            _start_poc(
                **start_args,
            )
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
        return self._is_poc_workspace_running(self.poc_workspace, fail_if_unknown=self._services_may_have_started)

    def stop(self, clean_up: bool = False) -> None:
        """Try to stop and clean existing POC.

        This method is idempotent - safe to call multiple times.

        Args:
            clean_up (bool, optional): Whether to clean the POC workspace. Defaults to False.
        """
        # Wait for an in-progress deployment to finish before inspecting its
        # services or removing its workspace.
        with self._deployment_lock:
            self._stop(clean_up)

    def _stop(self, clean_up: bool = False) -> None:
        """Stop POC while the caller holds the instance lifecycle guard."""
        try:
            try:
                poc_running = self._check_poc_running()
            except Exception as state_error:
                # State can be unknown when provisioned metadata is damaged.
                # Still make the normal stop attempt, but do not remove the
                # workspace or reset the uncertainty flag without verification.
                self.logger.warning(f"Could not determine whether POC services are running: {state_error}")
                stop_args = {
                    "poc_workspace": self.poc_workspace,
                    "excluded": [self.username],
                    "services_list": [],
                }
                if self._docker_container_names:
                    stop_args["docker_container_names"] = self._docker_container_names
                try:
                    _stop_poc(**stop_args)
                except Exception as stop_error:
                    self.logger.warning(f"Could not stop POC services with unknown state: {stop_error}")
                if clean_up:
                    self.logger.warning(
                        f"POC service state could not be verified; preserving workspace {self.poc_workspace}. "
                        "Stop any remaining services and remove it manually."
                    )
                return

            # Check if already stopped (idempotent)
            if not poc_running:
                # POC already stopped or workspace doesn't exist
                self._remove_docker_network()
                self._services_may_have_started = False
                if clean_up and os.path.exists(self.poc_workspace):
                    self.logger.info(f"Removing POC workspace: {self.poc_workspace}")
                    try:
                        self._remove_recipe_workspace()
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to clean POC workspace {self.poc_workspace}: {e}. Remove it manually."
                        )
                return

            project_config, service_config = setup_service_config(self.poc_workspace)
            service_config = self._with_docker_container_names(self.poc_workspace, service_config)
            self.logger.info("Stopping existing POC services...")
            # Prefer the coordinated server shutdown while it is reachable. If
            # the server exited during startup, stop any surviving local client
            # processes directly so the per-run workspace can be cleaned safely.
            services_list = []
            if service_config.get(SC.IS_DOCKER_RUN) or not is_poc_running(
                self.poc_workspace, service_config, project_config
            ):
                services_list = self._running_services(project_config, service_config, self.poc_workspace)
            stop_args = {
                "poc_workspace": self.poc_workspace,
                "excluded": [self.username],  # Exclude admin console (consistent with start)
                "services_list": services_list,
            }
            if self._docker_container_names:
                stop_args["docker_container_names"] = self._docker_container_names
            _stop_poc(
                **stop_args,
            )
            # Service exit and network teardown share one wait budget after the
            # shutdown command; slow endpoint detachment must not add another one.
            deadline = time.monotonic() + STOP_POC_TIMEOUT
            poc_running = True
            poc_state_error = None
            while time.monotonic() < deadline:
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
                time.sleep(min(1, max(0, deadline - time.monotonic())))

            if poc_running:
                if clean_up:
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
                self._remove_docker_network(deadline=deadline)
                self._services_may_have_started = False
                if clean_up:
                    try:
                        self._remove_recipe_workspace()
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to clean POC workspace {self.poc_workspace}: {e}. Remove it manually."
                        )
        except Exception as e:
            self.logger.warning(f"Failed to stop and clean existing POC: {e}")
            if clean_up:
                self.logger.warning(
                    f"POC cleanup could not be completed; preserving workspace {self.poc_workspace}. "
                    "Stop any remaining services or job containers, then retry PocEnv.stop(clean_up=True)."
                )
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
