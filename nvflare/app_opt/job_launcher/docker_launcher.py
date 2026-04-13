# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import logging
import os
import re
import time
from abc import abstractmethod

try:
    import docker.errors

    import docker

    _DOCKER_AVAILABLE = True
except ImportError:
    _DOCKER_AVAILABLE = False

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, JobConstants
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import JobMetaKey, get_job_meta_study
from nvflare.apis.job_launcher_spec import JobHandleSpec, JobLauncherSpec, JobProcessArgs, JobReturnCode, add_launcher
from nvflare.apis.workspace import Workspace
from nvflare.utils.job_launcher_utils import (
    extract_container_kwargs,
    extract_job_image,
    get_client_job_args,
    get_server_job_args,
)


# Docker container status strings
class DockerStatus:
    CREATED = "created"
    RESTARTING = "restarting"
    RUNNING = "running"
    PAUSED = "paused"
    EXITED = "exited"
    DEAD = "dead"


TERMINAL_STATUSES = {DockerStatus.EXITED, DockerStatus.DEAD}


def _sanitize_container_name(name: str) -> str:
    """Sanitize a string to a valid Docker container name.

    Docker container names allow alphanumeric, hyphens, underscores, and dots.
    """
    name = name.lower()
    name = re.sub(r"[^a-z0-9\-_.]", "-", name)
    name = name.strip("-")
    return name or "nvflare-job"


def _exit_code_to_return_code(exit_code: int) -> JobReturnCode:
    if exit_code == 0:
        return JobReturnCode.SUCCESS
    elif exit_code == JobReturnCode.ABORTED:
        return JobReturnCode.ABORTED
    else:
        return JobReturnCode.EXECUTION_ERROR


class DockerJobHandle(JobHandleSpec):
    """Handle for a running Docker container job.

    Modeled on K8sJobHandle: once the container reaches a terminal state,
    terminal_state is set and all subsequent poll()/wait() calls return
    immediately without querying Docker.
    """

    def __init__(
        self,
        container_id: str,
        container_name: str,
        docker_client,
        timeout: int = 30,
        shutdown_timeout: int = 60,
    ):
        super().__init__()
        self.container_id = container_id
        self.container_name = container_name
        self.docker_client = docker_client
        self.timeout = timeout
        self.shutdown_timeout = shutdown_timeout
        self.terminal_state: JobReturnCode = None  # set once, never cleared
        self.logger = logging.getLogger(self.__class__.__name__)

    def _get_container(self):
        """Query Docker for the current container object.

        Returns None if not found (sets terminal_state) or on API error.
        """
        try:
            return self.docker_client.containers.get(self.container_id)
        except docker.errors.NotFound:
            self.logger.info(f"container {self.container_name} not found; assuming terminated")
            if self.terminal_state is None:
                self.terminal_state = JobReturnCode.ABORTED
            return None
        except docker.errors.APIError as e:
            self.logger.warning(f"error querying container {self.container_name}: {e}")
            return None
        except Exception as e:
            self.logger.warning(f"unexpected error querying container {self.container_name}: {e}")
            return None

    def _resolve_terminal_return_code(self, container) -> JobReturnCode:
        """Get the final JobReturnCode from a terminal container using exit code."""
        if container.status == DockerStatus.DEAD:
            return JobReturnCode.ABORTED
        # EXITED: read actual exit code from container attrs
        exit_code = container.attrs.get("State", {}).get("ExitCode", 1)
        return _exit_code_to_return_code(exit_code)

    def _remove_container(self):
        """Remove the container after it has reached a terminal state."""
        try:
            container = self.docker_client.containers.get(self.container_id)
            container.remove(force=True)
            self.logger.debug(f"removed container {self.container_name}")
        except docker.errors.NotFound:
            pass  # already gone
        except docker.errors.APIError as e:
            self.logger.warning(f"error removing container {self.container_name}: {e}")

    def poll(self) -> JobReturnCode:
        """Non-blocking status check. Returns UNKNOWN while still running."""
        if self.terminal_state is not None:
            return self.terminal_state
        container = self._get_container()
        if container is None:
            return self.terminal_state if self.terminal_state is not None else JobReturnCode.UNKNOWN
        if container.status in TERMINAL_STATUSES:
            rc = self._resolve_terminal_return_code(container)
            self.terminal_state = rc
            self._remove_container()
            return rc
        return JobReturnCode.UNKNOWN

    def wait(self):
        """Block until the container reaches a terminal state.

        Uses shutdown_timeout as a workaround for containers that hang during NVFlare
        shutdown due to non-daemon threads (conn_mgr/frame_mgr) that never fully exit.
        After shutdown_timeout seconds the container is force-terminated.
        """
        start = time.time()
        while True:
            if self.terminal_state is not None:
                return
            container = self._get_container()
            if container is None:
                return
            if container.status in TERMINAL_STATUSES:
                self.terminal_state = self._resolve_terminal_return_code(container)
                self._remove_container()
                return
            if time.time() - start > self.shutdown_timeout:
                self.logger.warning(
                    f"container {self.container_name} did not exit within {self.shutdown_timeout}s; "
                    f"force-terminating (workaround for NVFlare thread shutdown hang)"
                )
                self.terminate()
                return
            time.sleep(1)

    def terminate(self):
        """Stop and remove the container. Always sets terminal_state."""
        try:
            container = self.docker_client.containers.get(self.container_id)
            container.stop(timeout=0)
            container.remove(force=True)
        except docker.errors.NotFound:
            self.logger.info(f"container {self.container_name} not found during termination; assuming terminated")
        except docker.errors.APIError as e:
            self.logger.error(f"error terminating container {self.container_name}: {e}")
        except Exception as e:
            self.logger.error(f"unexpected error terminating container {self.container_name}: {e}")
        finally:
            # Always set terminal_state so poll()/wait() return immediately
            if self.terminal_state is None:
                self.terminal_state = JobReturnCode.ABORTED

    def enter_states(self, states_to_enter: list) -> bool:
        """Poll until the container enters one of the target states.

        Returns True if the target state was reached, False otherwise
        (timeout, stuck, or terminal state reached before target).
        """
        starting_time = time.time()
        if not isinstance(states_to_enter, (list, tuple)):
            states_to_enter = [states_to_enter]

        while True:
            if self.terminal_state is not None:
                return False

            container = self._get_container()
            if container is None:
                return False

            status = container.status

            if status in states_to_enter:
                return True

            if status in TERMINAL_STATUSES:
                self.terminal_state = self._resolve_terminal_return_code(container)
                self._remove_container()
                return False

            if self.timeout is not None and time.time() - starting_time > self.timeout:
                self.logger.warning(f"container {self.container_name} timed out waiting for {states_to_enter}")
                self.terminate()
                return False

            time.sleep(1)


def _job_args_dict(job_args: dict, arg_names: list) -> dict:
    """Extract a {flag: value} dict from JOB_PROCESS_ARGS for the given arg names."""
    result = {}
    for name in arg_names:
        e = job_args.get(name)
        if not e:
            continue
        flag, value = e
        result[flag] = value
    return result


class DockerJobLauncher(JobLauncherSpec):
    """Launches NVFlare job processes as Docker containers.

    SP/CP runs as a container started by start_docker.sh (site admin).
    SJ/CJ containers are started dynamically per job by this launcher.

    Assumptions:
    - Docker network already exists (created by start_docker.sh or site admin).
    - Workspace is a host directory bind-mounted into all containers at /var/tmp/nvflare/workspace.
    - SP/CP container name is known and reachable via Docker DNS on the network.
    - parent_url is derived at runtime from the site name and the port in JOB_PROCESS_ARGS.
    """

    WORKSPACE_MOUNT = "/var/tmp/nvflare/workspace"
    DATA_MOUNT = "/var/tmp/nvflare/data"
    STUDY_DATA_PATH_FILE = "local/study_data.json"

    def __init__(
        self,
        workspace: str = None,
        network: str = "nvflare-network",
        python_path: str = "/usr/local/bin/python",
        timeout: int = 30,
        shutdown_timeout: int = 60,
        extra_container_kwargs: dict = None,
    ):
        """
        Args:
            workspace: host path to the NVFlare workspace directory (bind-mounted into job containers
                       at /var/tmp/nvflare/workspace). If not provided, reads from NVFL_DOCKER_WORKSPACE
                       environment variable. Must be the HOST path because it is passed directly to the
                       Docker daemon as a volume bind source.
            network: Docker network name. Must already exist.
            python_path: Python executable path inside the job container.
            timeout: max seconds to wait for container to reach RUNNING state (default 30).
            shutdown_timeout: max seconds in wait() before force-terminating after job completes.
                              Workaround for NVFlare thread shutdown hang (conn_mgr/frame_mgr)
                              that prevents container from exiting cleanly (default 60).
            extra_container_kwargs: site-level default docker run kwargs passed to all job containers
                                    on this site. Job-level container_kwargs in deploy_map take precedence
                                    on conflict. Keys use Docker SDK naming (underscores, not hyphens).
                                    Example: {"shm_size": "8g", "ipc_mode": "host"}
                                    Note: "volumes", "network", "environment", "command", "name", "detach", "user", "working_dir"
                                    are controlled by the launcher and cannot be overridden here.
        """
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

        if not workspace:
            workspace = os.environ.get("NVFL_DOCKER_WORKSPACE")

        self.workspace = workspace
        self.network = network
        self.python_path = python_path
        self.timeout = timeout
        self.shutdown_timeout = shutdown_timeout
        extra_container_kwargs = extra_container_kwargs or {}
        _RESERVED_KWARGS = {"volumes", "network", "environment", "command", "name", "detach", "user", "working_dir"}
        reserved_used = _RESERVED_KWARGS & set(extra_container_kwargs.keys())
        if reserved_used:
            raise ValueError(
                f"extra_container_kwargs must not contain reserved keys: {sorted(reserved_used)}. "
                f"These are controlled by the launcher."
            )
        self.extra_container_kwargs = extra_container_kwargs

        self._docker_client = None

    def _get_docker_client(self):
        if self._docker_client is None:
            if not _DOCKER_AVAILABLE:
                raise RuntimeError("docker SDK not installed; install it with: pip install docker")
            try:
                client = docker.from_env()
                client.ping()
            except Exception as e:
                raise RuntimeError(f"cannot connect to Docker daemon: {e}")
            try:
                client.networks.get(self.network)
            except docker.errors.NotFound:
                raise RuntimeError(
                    f"Docker network '{self.network}' does not exist. "
                    f"Create it with: docker network create {self.network}"
                )
            except docker.errors.APIError as e:
                raise RuntimeError(f"error checking Docker network '{self.network}': {e}")
            self._docker_client = client
        return self._docker_client

    def launch_job(self, job_meta: dict, fl_ctx: FLContext) -> JobHandleSpec:
        job_id = job_meta.get(JobConstants.JOB_ID)
        if not job_id:
            raise RuntimeError("missing JOB_ID in job_meta")

        job_args = fl_ctx.get_prop(FLContextKey.JOB_PROCESS_ARGS)
        if not job_args:
            raise RuntimeError(f"missing {FLContextKey.JOB_PROCESS_ARGS} in FLContext")

        exe_module_entry = job_args.get(JobProcessArgs.EXE_MODULE)
        if not exe_module_entry:
            raise RuntimeError(f"missing {JobProcessArgs.EXE_MODULE} in JOB_PROCESS_ARGS")
        _, exe_module = exe_module_entry

        job_image = extract_job_image(job_meta, fl_ctx.get_identity_name())
        site_name = fl_ctx.get_identity_name()
        container_name = _sanitize_container_name(f"{site_name}-{job_id}")

        workspace = self.workspace
        if not workspace:
            raise ValueError(
                "workspace must be set to the host path of the NVFlare workspace directory, "
                "or set the NVFL_DOCKER_WORKSPACE environment variable"
            )

        # Derive parent_url at runtime: site name (= container name on Docker DNS) + port
        # from the original PARENT_URL in job_args. This avoids baking parent_url into
        # resources.json at provision time.
        if JobProcessArgs.PARENT_URL in job_args:
            flag, original_url = job_args[JobProcessArgs.PARENT_URL]
            port = original_url.rsplit(":", 1)[-1]
            parent_url = f"tcp://{site_name}:{port}"
            job_args = dict(job_args)
            job_args[JobProcessArgs.PARENT_URL] = (flag, parent_url)

        module_args = self.get_module_args(job_args)
        module_args_list = []
        for flag, value in module_args.items():
            if value is not None:
                module_args_list.extend([flag, str(value)])

        # Append --set options (same as K8s launcher)
        args = fl_ctx.get_prop(FLContextKey.ARGS)
        set_list = args.set if args is not None and getattr(args, "set", None) is not None else None
        if set_list:
            module_args_list.extend(["--set"] + set_list)

        command = [self.python_path, "-u", "-m", exe_module] + module_args_list

        # PYTHONPATH: translate app_custom_folder host path to container-internal path
        # so custom Python code in the job app is importable inside the container.
        # USER: some libraries (e.g. torch._dynamo) call getpass.getuser() which falls back to
        # pwd.getpwuid(os.getuid()). When the container runs as a host UID not in /etc/passwd,
        # this raises KeyError. Setting USER satisfies the env-var fast path in getpass.getuser().
        # Pass USER and HOME so libraries that call getpass.getuser() or os.path.expanduser("~")
        # don't fall back to pwd.getpwuid() — which fails when the host UID has no /etc/passwd entry.
        environment = {
            "USER": os.environ.get("USER", "nvflare"),
            "HOME": os.environ.get("HOME", "/tmp"),
        }
        workspace_obj: Workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
        if workspace_obj is not None:
            python_paths = []
            app_custom_folder = workspace_obj.get_app_custom_dir(job_id)
            if app_custom_folder:
                python_paths.append(app_custom_folder.replace(workspace, self.WORKSPACE_MOUNT, 1))
            site_custom_folder = workspace_obj.get_site_custom_dir()
            if site_custom_folder and os.path.isdir(site_custom_folder):
                python_paths.append(site_custom_folder.replace(workspace, self.WORKSPACE_MOUNT, 1))
            if python_paths:
                environment["PYTHONPATH"] = os.pathsep.join(python_paths)

        # container_kwargs: per-job from deploy_map, merged with site-level defaults from resources.json.
        # Job-level takes precedence on conflict.
        job_container_kwargs = extract_container_kwargs(job_meta, site_name)
        merged_container_kwargs = {**self.extra_container_kwargs, **job_container_kwargs}

        # GPU: translate resource_spec.num_of_gpus → device_requests (consistent with K8s/process launchers).
        # Explicit device_requests in container_kwargs takes precedence (fine-grain override).
        resource_spec = job_meta.get(JobMetaKey.RESOURCE_SPEC.value, {}) or {}
        num_gpus = (resource_spec.get(site_name) or {}).get("num_of_gpus", 0)
        if num_gpus:
            merged_container_kwargs.setdefault(
                "device_requests", [{"Count": num_gpus, "Capabilities": [["gpu"]]}]
            )

        # Volumes: always mount workspace; optionally mount study data if local/study_data.json exists
        volumes = {
            workspace: {"bind": self.WORKSPACE_MOUNT, "mode": "rw"},
        }
        # Read study data map from workspace/local/study_data.json (host path).
        # Maps study name → host data path; study name comes from meta.json "study" field.
        study_data_file = os.path.join(workspace, self.STUDY_DATA_PATH_FILE)
        if os.path.isfile(study_data_file):
            try:
                import json as _json
                with open(study_data_file) as f:
                    study_data_map = _json.load(f)
                study = get_job_meta_study(job_meta)
                data_host_path = study_data_map.get(study) if study else None
                if data_host_path:
                    volumes[data_host_path] = {"bind": self.DATA_MOUNT, "mode": "ro"}
                    self.logger.info(f"mounting study '{study}' data from {data_host_path} -> {self.DATA_MOUNT}")
            except Exception as e:
                self.logger.warning(f"failed to read {study_data_file}: {e}")

        self.logger.info(f"launching job {job_id} as container {container_name} using image {job_image}")

        docker_client = self._get_docker_client()
        try:
            container = docker_client.containers.run(
                job_image,
                command=command,
                name=container_name,
                network=self.network,
                detach=True,
                environment=environment if environment else None,
                volumes=volumes,
                working_dir=self.WORKSPACE_MOUNT,
                # Run as the same user as SP/CP so job-written files are accessible to SP/CP
                # (e.g. cross_val_results.json written by SJ must be readable/deletable by SP).
                # Never pass Docker socket to job containers.
                user=f"{os.getuid()}:{os.getgid()}",
                **merged_container_kwargs,
            )
        except docker.errors.ImageNotFound:
            raise RuntimeError(f"image '{job_image}' not found for job {job_id}")
        except docker.errors.APIError as e:
            raise RuntimeError(f"error creating container for job {job_id}: {e}")

        job_handle = DockerJobHandle(
            container_id=container.id,
            container_name=container_name,
            docker_client=docker_client,
            timeout=self.timeout,
            shutdown_timeout=self.shutdown_timeout,
        )

        try:
            if not job_handle.enter_states([DockerStatus.RUNNING]):
                self.logger.warning(f"container {container_name} did not reach RUNNING state for job {job_id}")
        except BaseException:
            job_handle.terminate()
            raise

        # Always return a handle — caller detects failure via poll()
        return job_handle

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.BEFORE_JOB_LAUNCH:
            job_meta = fl_ctx.get_prop(FLContextKey.JOB_META)
            job_image = extract_job_image(job_meta, fl_ctx.get_identity_name())
            if job_image:
                add_launcher(self, fl_ctx)

    @abstractmethod
    def get_module_args(self, job_args: dict) -> dict:
        """Return a {flag: value} dict of args to pass to the job module.

        Args:
            job_args: JOB_PROCESS_ARGS dict from FLContext (with PARENT_URL already overridden).

        Returns:
            dict of {flag: value} pairs to append after '-u -m <module>' in the container command.
        """
        pass


class ClientDockerJobLauncher(DockerJobLauncher):
    def get_module_args(self, job_args: dict) -> dict:
        return _job_args_dict(job_args, get_client_job_args(include_exe_module=False, include_set_options=False))


class ServerDockerJobLauncher(DockerJobLauncher):
    def get_module_args(self, job_args: dict) -> dict:
        return _job_args_dict(job_args, get_server_job_args(include_exe_module=False, include_set_options=False))
