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
import posixpath
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
from nvflare.apis.fl_constant import FLContextKey, JobConstants, WorkspaceConstants
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import JobMetaKey
from nvflare.apis.job_launcher_spec import JobHandleSpec, JobLauncherSpec, JobProcessArgs, JobReturnCode, add_launcher
from nvflare.apis.workspace import Workspace
from nvflare.app_opt.job_launcher.study_data import (
    load_study_data_file,
    resolve_study_dataset_mounts,
    should_mount_study_data,
)
from nvflare.utils.job_launcher_utils import get_client_job_args, get_job_launcher_spec, get_server_job_args


# Docker container status strings
class DockerStatus:
    CREATED = "created"
    RESTARTING = "restarting"
    RUNNING = "running"
    PAUSED = "paused"
    EXITED = "exited"
    DEAD = "dead"


TERMINAL_STATUSES = {DockerStatus.EXITED, DockerStatus.DEAD}
# Docker tmpfs mounts are commonly owned by root; use sticky world-writable mode so the
# non-root job-container user can initialize ephemeral top-level workspace directories.
_WORKSPACE_TMPFS_MODE = 0o1777
_RESERVED_WORKSPACE_CHILD_NAMES = {
    WorkspaceConstants.STARTUP_FOLDER_NAME,
    WorkspaceConstants.SITE_FOLDER_NAME,
}


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


def _safe_workspace_child_path(workspace: str, child_name: str, allow_reserved: bool = False) -> str:
    """Return a host workspace child path, rejecting paths that escape workspace."""
    child_name = str(child_name)
    normalized_child_name = os.path.normpath(child_name)
    # normpath catches traversal spellings; the separator checks catch already-normalized nested paths.
    if (
        not child_name
        or os.path.isabs(child_name)
        or normalized_child_name != child_name
        or normalized_child_name in ("", ".", "..")
        or os.sep in normalized_child_name
        or (os.altsep and os.altsep in normalized_child_name)
    ):
        raise RuntimeError(f"job workspace path must be a single workspace child: {child_name}")
    if not allow_reserved and normalized_child_name in _RESERVED_WORKSPACE_CHILD_NAMES:
        raise RuntimeError(f"job workspace path uses reserved workspace name: {child_name}")

    child_path = os.path.normpath(os.path.join(workspace, child_name))
    workspace_real = os.path.realpath(workspace)
    child_real = os.path.realpath(child_path)
    if os.path.commonpath([workspace_real, child_real]) != workspace_real:
        raise RuntimeError(f"job workspace path escapes workspace: {child_name}")
    if os.path.islink(child_path):
        raise RuntimeError(f"workspace child path must not be a symlink: {child_name}")
    return child_path


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
    ):
        super().__init__()
        self.container_id = container_id
        self.container_name = container_name
        self.docker_client = docker_client
        self.timeout = timeout
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
        """Block until the container reaches a terminal state."""
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
    - Job containers get an isolated workspace view at /var/tmp/nvflare/workspace:
      the root is writable ephemeral tmpfs, startup/local are read-only, and only the current
      job workspace is read-write and persistent on the host.
    - SP/CP container name is known and reachable via Docker DNS on the network.
    - parent_url is derived at runtime from the site name and the port in JOB_PROCESS_ARGS.
    """

    WORKSPACE_MOUNT = "/var/tmp/nvflare/workspace"
    STUDY_DATA_PATH_FILE = "local/study_data.yaml"

    DEFAULT_PYTHON_PATH = "/usr/local/bin/python"

    def __init__(
        self,
        workspace: str = None,
        network: str = "nvflare-network",
        python_path: str = None,
        timeout: int = 30,
        default_job_container_kwargs: dict = None,
        default_job_env: dict = None,
        default_python_path: str = None,
    ):
        """
        Args:
            workspace: host path to the NVFlare workspace directory. Job containers receive an isolated
                       workspace view: startup/local are mounted read-only, and the current job workspace
                       is mounted read-write at /var/tmp/nvflare/workspace/<job_id>. If not provided,
                       reads from NVFL_DOCKER_WORKSPACE environment variable. Must be the HOST path
                       because it is passed directly to the Docker daemon as a volume bind source.
            network: Docker network name. Must already exist.
            python_path: Deprecated alias for default_python_path.
            timeout: max seconds to wait for container to reach RUNNING state (default 30).
            default_job_container_kwargs: site-level default docker run kwargs applied to every job
                                          container launched by this site. Job-level resource_spec[site][docker]
                                          takes precedence on conflict. Keys use Docker SDK naming
                                          (underscores, not hyphens).
                                          Example: {"shm_size": "8g", "ipc_mode": "host"}
                                          Note: "volumes", "mounts", "network", "environment", "command",
                                          "name", "detach", "user", "working_dir" are controlled by the launcher
                                          and cannot be overridden here.
            default_job_env: site-level default environment variables injected into every job
                             container launched by this site. Useful for site/runtime-specific
                             settings such as NCCL workarounds. Launcher-controlled variables
                             like USER, HOME, and PYTHONPATH still take precedence.
            default_python_path: Default Python executable path inside job containers. Jobs can override
                                 it with launcher_spec[site]["docker"]["python_path"].
        """
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

        if not workspace:
            workspace = os.environ.get("NVFL_DOCKER_WORKSPACE")

        self.workspace = workspace
        self.network = network
        self.default_python_path = default_python_path if default_python_path is not None else python_path
        if self.default_python_path is None:
            self.default_python_path = self.DEFAULT_PYTHON_PATH
        if not isinstance(self.default_python_path, str) or not self.default_python_path:
            raise ValueError("default_python_path must be a non-empty string")
        self.timeout = timeout
        default_job_container_kwargs = default_job_container_kwargs or {}
        _RESERVED_KWARGS = {
            "volumes",
            "mounts",
            "network",
            "environment",
            "command",
            "name",
            "detach",
            "user",
            "working_dir",
        }
        reserved_used = _RESERVED_KWARGS & set(default_job_container_kwargs.keys())
        if reserved_used:
            raise ValueError(
                f"default_job_container_kwargs must not contain reserved keys: {sorted(reserved_used)}. "
                f"These are controlled by the launcher."
            )
        self.default_job_container_kwargs = default_job_container_kwargs
        self.default_job_env = default_job_env or {}

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

        site_name = fl_ctx.get_identity_name()
        docker_spec = get_job_launcher_spec(job_meta, site_name, "docker")
        job_image = docker_spec.get("image")
        container_name = _sanitize_container_name(f"{site_name}-{job_id}")
        if not job_image:
            raise RuntimeError(
                f"DockerJobLauncher is configured for site '{site_name}' but no job image "
                f"was specified in meta.json for this site. "
                f"Set launcher_spec['{site_name}']['docker']['image'] (preferred), "
                f"launcher_spec['default']['docker']['image'] (shared default), "
                f"or resource_spec['{site_name}']['docker']['image'] (legacy)."
            )

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

        python_path = docker_spec.get("python_path", self.default_python_path)
        if not isinstance(python_path, str) or not python_path:
            raise RuntimeError(f"launcher_spec['{site_name}']['docker']['python_path'] must be a non-empty string")
        command = [python_path, "-u", "-m", exe_module] + module_args_list

        # PYTHONPATH: translate app_custom_folder host path to container-internal path
        # so custom Python code in the job app is importable inside the container.
        # USER: some libraries (e.g. torch._dynamo) call getpass.getuser() which falls back to
        # pwd.getpwuid(os.getuid()). When the container runs as a host UID not in /etc/passwd,
        # this raises KeyError. Setting USER satisfies the env-var fast path in getpass.getuser().
        # Pass USER and HOME so libraries that call getpass.getuser() or os.path.expanduser("~")
        # don't fall back to pwd.getpwuid() — which fails when the host UID has no /etc/passwd entry.
        environment = {
            **self.default_job_env,
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

        # Docker launcher spec: per-job Docker settings (image, shm_size, ipc_mode, ...) live in
        # launcher_spec[site][docker]. Falls back to nested resource_spec[site][docker] for
        # backward compatibility. num_of_gpus falls back to flat resource_spec[site] (Option 4).
        # Site-level defaults (default_job_container_kwargs) are merged in; job-level takes precedence on conflict.
        _site_rs = (job_meta.get(JobMetaKey.RESOURCE_SPEC.value) or {}).get(site_name) or {}
        _flat_gpus = 0 if any(k in _site_rs for k in ("process", "docker", "k8s")) else _site_rs.get("num_of_gpus", 0)
        num_gpus = docker_spec["num_of_gpus"] if "num_of_gpus" in docker_spec else _flat_gpus
        _RESERVED_KWARGS = {
            "volumes",
            "mounts",
            "network",
            "environment",
            "command",
            "name",
            "detach",
            "user",
            "working_dir",
        }
        _NON_CONTAINER_KEYS = {"num_of_gpus", "image", "python_path"} | _RESERVED_KWARGS
        reserved_in_spec = _RESERVED_KWARGS & set(docker_spec.keys())
        if reserved_in_spec:
            self.logger.warning(
                f"job {job_id}: launcher_spec['{site_name}']['docker'] contains reserved keys "
                f"{sorted(reserved_in_spec)} — ignored (controlled by the launcher)"
            )
        job_container_kwargs = {k: v for k, v in docker_spec.items() if k not in _NON_CONTAINER_KEYS}
        merged_container_kwargs = {**self.default_job_container_kwargs, **job_container_kwargs}

        # GPU precedence:
        # 1. explicit job-level device_requests in docker_spec
        # 2. job-level num_of_gpus translated to device_requests
        # 3. site-level default device_requests from default_job_container_kwargs
        #
        # This preserves the documented rule that job-level resource_spec takes precedence
        # over site-level defaults, while still allowing fine-grained device_requests overrides.
        if num_gpus and "device_requests" not in job_container_kwargs:
            merged_container_kwargs["device_requests"] = [{"Count": num_gpus, "Capabilities": [["gpu"]]}]

        # Give the job an isolated workspace view. The root tmpfs must be writable by the non-root
        # container user because server job startup may create ephemeral storage dirs such as
        # snapshot-storage and jobs-storage.
        # startup/local are read-only, and only this job's workspace is read-write and persistent on the host.
        job_workspace_name = WorkspaceConstants.WORKSPACE_PREFIX + str(job_id)
        host_job_workspace = _safe_workspace_child_path(workspace, job_workspace_name)
        host_startup_dir = _safe_workspace_child_path(
            workspace, WorkspaceConstants.STARTUP_FOLDER_NAME, allow_reserved=True
        )
        host_local_dir = _safe_workspace_child_path(workspace, WorkspaceConstants.SITE_FOLDER_NAME, allow_reserved=True)
        container_job_workspace = posixpath.join(self.WORKSPACE_MOUNT, job_workspace_name)
        container_startup_dir = posixpath.join(self.WORKSPACE_MOUNT, WorkspaceConstants.STARTUP_FOLDER_NAME)
        container_local_dir = posixpath.join(self.WORKSPACE_MOUNT, WorkspaceConstants.SITE_FOLDER_NAME)
        data_mounts = []

        # Read study data map from workspace/local/study_data.yaml.
        # Must use WORKSPACE_MOUNT (container-internal path) for the file read because launch_job
        # runs inside the SP/CP container. The host path (workspace) does not exist in the container
        # filesystem. The Docker volume source must remain the host path for the daemon API.
        # Maps study -> dataset -> {source, mode}; source is a host path for Docker.
        # Each dataset is mounted at /data/<study>/<dataset>.
        study_data_file = os.path.join(self.WORKSPACE_MOUNT, self.STUDY_DATA_PATH_FILE)
        study = job_meta.get(JobMetaKey.STUDY.value)
        if should_mount_study_data(study):
            study_data_map = load_study_data_file(study_data_file, logger=self.logger)
            data_mounts = resolve_study_dataset_mounts(study_data_map, study, study_data_file, logger=self.logger)
            for dataset_mount in data_mounts:
                self.logger.info(
                    "mounting study '%s' dataset '%s' from %s -> %s",
                    study,
                    dataset_mount.dataset,
                    dataset_mount.source,
                    dataset_mount.mount_path,
                )

        self.logger.info(f"launching job {job_id} as container {container_name} using image {job_image}")

        docker_client = self._get_docker_client()
        try:
            mounts = [
                docker.types.Mount(
                    target=self.WORKSPACE_MOUNT,
                    source=None,
                    type="tmpfs",
                    read_only=False,
                    tmpfs_mode=_WORKSPACE_TMPFS_MODE,
                ),
                docker.types.Mount(
                    target=container_startup_dir,
                    source=host_startup_dir,
                    type="bind",
                    read_only=True,
                ),
                docker.types.Mount(target=container_local_dir, source=host_local_dir, type="bind", read_only=True),
                docker.types.Mount(
                    target=container_job_workspace,
                    source=host_job_workspace,
                    type="bind",
                    read_only=False,
                ),
            ]
            for dataset_mount in data_mounts:
                mounts.append(
                    docker.types.Mount(
                        target=dataset_mount.mount_path,
                        source=dataset_mount.source,
                        type="bind",
                        read_only=dataset_mount.read_only,
                    )
                )
            container = docker_client.containers.run(
                job_image,
                command=command,
                name=container_name,
                network=self.network,
                detach=True,
                environment=environment if environment else None,
                mounts=mounts,
                working_dir=container_job_workspace,
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
