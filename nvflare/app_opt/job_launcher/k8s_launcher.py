from __future__ import annotations

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
import base64
import copy
import logging
import os
import re
import socket
import threading
import time
from abc import abstractmethod
from enum import Enum

import yaml

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, JobConstants
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import JobMetaKey
from nvflare.apis.job_launcher_spec import JobHandleSpec, JobLauncherSpec, JobProcessArgs, JobReturnCode, add_launcher
from nvflare.app_opt.job_launcher.workspace_http_server import ENV_WORKSPACE_URL, WorkspaceHTTPServer
from nvflare.utils.job_launcher_utils import (
    extract_job_image,
    get_client_job_args,
    get_launcher_resource_spec,
    get_server_job_args,
)


class JobState(Enum):
    STARTING = "starting"
    RUNNING = "running"
    TERMINATED = "terminated"
    SUCCEEDED = "succeeded"
    UNKNOWN = "unknown"


class PodPhase(Enum):
    PENDING = "Pending"
    RUNNING = "Running"
    SUCCEEDED = "Succeeded"
    FAILED = "Failed"
    UNKNOWN = "Unknown"


POD_STATE_MAPPING = {
    PodPhase.PENDING.value: JobState.STARTING,
    PodPhase.RUNNING.value: JobState.RUNNING,
    PodPhase.SUCCEEDED.value: JobState.SUCCEEDED,
    PodPhase.FAILED.value: JobState.TERMINATED,
    PodPhase.UNKNOWN.value: JobState.UNKNOWN,
}

JOB_RETURN_CODE_MAPPING = {
    JobState.SUCCEEDED: JobReturnCode.SUCCESS,
    JobState.STARTING: JobReturnCode.UNKNOWN,
    JobState.RUNNING: JobReturnCode.UNKNOWN,
    JobState.TERMINATED: JobReturnCode.ABORTED,
    JobState.UNKNOWN: JobReturnCode.UNKNOWN,
}

DEFAULT_CONTAINER_ARGS_MODULE_ARGS_DICT = {
    "-m": None,
    "-w": None,
    "-t": None,
    "-d": None,
    "-n": None,
    "-c": None,
    "-p": None,
    "-g": None,
    "-scheme": None,
    "-s": None,
}

DEFAULT_NAMESPACE = "default"
DEFAULT_PENDING_TIMEOUT = 120
DEFAULT_PYTHON_PATH = "/usr/local/bin/python"


DATA_PVC_VOLUME_NAME = "nvfldata"
WORKSPACE_MOUNT_PATH = "/var/tmp/nvflare/workspace"
DEFAULT_EPHEMERAL_STORAGE = "1Gi"

# Files actually read from startup/ by the job pod at runtime. Others in
# startup/ are dropped to shrink the Secret. local/ is bundled whole with each
# job workspace so job resource files and local custom code keep working.
_STARTUP_KEEP_SUFFIXES = (".crt", ".key", ".pem", ".json")


def _keep_startup_file(fname: str) -> bool:
    return fname.endswith(_STARTUP_KEEP_SUFFIXES)


def _get_workspace_server_tls_files(startup_dir: str) -> tuple[str, str]:
    candidates = [
        ("server.crt", "server.key"),
        ("client.crt", "client.key"),
    ]
    for cert_name, key_name in candidates:
        cert_file = os.path.join(startup_dir, cert_name)
        key_file = os.path.join(startup_dir, key_name)
        if os.path.exists(cert_file) and os.path.exists(key_file):
            return cert_file, key_file
    raise RuntimeError(f"workspace HTTPS server requires cert/key files in startup dir: {startup_dir}")


def uuid4_to_rfc1123(uuid_str: str) -> str:
    name = uuid_str.lower()
    # Strip any chars that aren't alphanumeric or hyphen
    name = re.sub(r"[^a-z0-9-]", "", name)
    # Prefix with a letter if it starts with a digit
    if name and name[0].isdigit():
        name = "j" + name
    # Kubernetes label limit: 63 chars; strip trailing hyphens after truncation
    # (truncation can expose a hyphen that was interior before slicing)
    return name[:63].rstrip("-")


class K8sJobHandle(JobHandleSpec):
    def __init__(
        self,
        job_id: str,
        api_instance,
        job_config: dict,
        namespace=DEFAULT_NAMESPACE,
        timeout=None,
        pending_timeout=DEFAULT_PENDING_TIMEOUT,
        python_path=DEFAULT_PYTHON_PATH,
        workspace_server=None,
        workspace_token: str = "",
    ):
        super().__init__()
        self.job_id = job_id
        self.timeout = timeout
        self.terminal_state = None
        self.workspace_server = workspace_server
        self.workspace_token = workspace_token
        self.api_instance = api_instance
        self.namespace = namespace
        self.pod_manifest = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {"name": None},  # set by job_config['name']
            "spec": {
                "containers": None,  # link to container_list
                "volumes": None,  # link to volume_list
                "restartPolicy": "Never",
            },
        }
        self.volume_list = []

        self.container_list = [
            {
                "image": None,
                "name": None,
                "command": [python_path],
                "args": None,  # args_list + args_dict + args_sets
                "volumeMounts": None,  # volume_mount_list
                "imagePullPolicy": "Always",
            }
        ]
        command = job_config.get("command")
        if not command:
            raise ValueError("job_config must contain a non-empty 'command' key")
        self.container_args_python_args_list = ["-u", "-m", command]
        self.container_volume_mount_list = []
        self._make_manifest(job_config)
        self._stuck_count = 0
        self._max_stuck_count = self.timeout if self.timeout is not None else pending_timeout
        self.logger = logging.getLogger(self.__class__.__name__)

    def _make_manifest(self, job_config):
        self.container_volume_mount_list.extend(job_config.get("volume_mount_list", []))
        set_list = job_config.get("set_list")
        if not set_list:
            self.container_args_module_args_sets = list()
        else:
            self.container_args_module_args_sets = ["--set"] + set_list
        if job_config.get("module_args") is None:
            self.container_args_module_args_dict = DEFAULT_CONTAINER_ARGS_MODULE_ARGS_DICT.copy()
        else:
            self.container_args_module_args_dict = job_config.get("module_args")
        self.container_args_module_args_dict_as_list = list()
        for k, v in self.container_args_module_args_dict.items():
            if v is None:
                continue
            self.container_args_module_args_dict_as_list.append(k)
            self.container_args_module_args_dict_as_list.append(str(v))
        self.volume_list.extend(job_config.get("volume_list", []))
        self.pod_manifest["metadata"]["name"] = job_config.get("name")
        self.pod_manifest["spec"]["containers"] = self.container_list
        self.pod_manifest["spec"]["volumes"] = self.volume_list
        security_context = job_config.get("security_context")
        if security_context:
            self.pod_manifest["spec"]["securityContext"] = security_context

        image = job_config.get("image")
        if not image:
            raise ValueError("job_config must contain a non-empty 'image' key")
        self.container_list[0]["image"] = image
        self.container_list[0]["name"] = job_config.get("container_name", "nvflare_job")
        self.container_list[0]["args"] = (
            self.container_args_python_args_list
            + self.container_args_module_args_dict_as_list
            + self.container_args_module_args_sets
        )
        self.container_list[0]["volumeMounts"] = self.container_volume_mount_list
        if job_config.get("resources"):
            self.container_list[0]["resources"] = job_config["resources"]
        env_vars = {k: v for k, v in job_config.get("env", {}).items() if str(v)}
        if env_vars:
            self.container_list[0]["env"] = [{"name": k, "value": str(v)} for k, v in env_vars.items()]

    def get_manifest(self):
        return copy.deepcopy(self.pod_manifest)

    def enter_states(self, job_states_to_enter: list):
        starting_time = time.time()
        if not isinstance(job_states_to_enter, (list, tuple)):
            job_states_to_enter = [job_states_to_enter]
        if not all([isinstance(js, JobState) for js in job_states_to_enter]):
            raise ValueError(f"expect job_states_to_enter with valid values, but get {job_states_to_enter}")
        while True:
            pod_phase = self._query_phase()
            if self._stuck_in_pending(pod_phase):
                self.terminate()
                return False
            job_state = POD_STATE_MAPPING.get(pod_phase, JobState.UNKNOWN)
            if job_state in job_states_to_enter:
                return True
            elif pod_phase in [PodPhase.FAILED.value, PodPhase.SUCCEEDED.value]:  # terminal state
                self.terminal_state = POD_STATE_MAPPING.get(pod_phase, JobState.UNKNOWN)
                self._remove_workspace_job()
                return False
            elif self.timeout is not None and time.time() - starting_time > self.timeout:
                self.terminate()
                return False
            time.sleep(1)

    def _remove_workspace_job(self) -> None:
        if self.workspace_server and self.workspace_token:
            self.workspace_server.remove_job(self.workspace_token)
            self.workspace_token = ""

    def terminate(self):
        from kubernetes.client.rest import ApiException

        try:
            self.api_instance.delete_namespaced_pod(name=self.job_id, namespace=self.namespace, grace_period_seconds=0)
            self.terminal_state = JobState.TERMINATED
        except ApiException as e:
            if getattr(e, "status", None) == 404:
                self.logger.info(f"job {self.job_id} pod not found during termination; assuming terminated")
            else:
                self.logger.error(f"failed to terminate job {self.job_id}: {e}")
            self.terminal_state = JobState.TERMINATED
        except Exception as e:
            self.logger.error(f"unexpected error terminating job {self.job_id}: {e}")
            self.terminal_state = JobState.TERMINATED
        self._remove_workspace_job()
        return None

    def poll(self):
        if self.terminal_state is not None:
            return JOB_RETURN_CODE_MAPPING.get(self.terminal_state)
        job_state = self._query_state()
        if self.terminal_state is not None:
            return JOB_RETURN_CODE_MAPPING.get(self.terminal_state)
        if job_state in (JobState.SUCCEEDED, JobState.TERMINATED):
            self.terminal_state = job_state
            self._remove_workspace_job()
        return JOB_RETURN_CODE_MAPPING.get(job_state, JobReturnCode.UNKNOWN)

    def _query_phase(self):
        from kubernetes.client.rest import ApiException

        try:
            resp = self.api_instance.read_namespaced_pod(name=self.job_id, namespace=self.namespace)
        except ApiException as e:
            if getattr(e, "status", None) == 404:
                self.logger.info(f"job {self.job_id} pod not found during querying; assuming terminated")
                self.terminal_state = JobState.TERMINATED
                self._remove_workspace_job()
            else:
                self.logger.warning(f"failed to query pod phase {self.job_id}: {e}")
            return PodPhase.UNKNOWN.value
        except Exception as e:
            self.logger.warning(f"unexpected error querying pod phase {self.job_id}: {e}")
            return PodPhase.UNKNOWN.value
        return resp.status.phase

    def _query_state(self):
        pod_phase = self._query_phase()
        return POD_STATE_MAPPING.get(pod_phase, JobState.UNKNOWN)

    def _stuck_in_pending(self, current_phase):
        if current_phase == PodPhase.PENDING.value:
            self._stuck_count += 1
            if self._max_stuck_count is not None and self._stuck_count >= self._max_stuck_count:
                return True
        else:
            self._stuck_count = 0
        return False

    def wait(self):
        while True:
            if self.terminal_state is not None:
                return
            job_state = self._query_state()
            if job_state in (JobState.SUCCEEDED, JobState.TERMINATED):
                self.terminal_state = job_state  # persist so poll() stays accurate
                self._remove_workspace_job()
                return
            time.sleep(1)


class K8sJobLauncher(JobLauncherSpec):
    def __init__(
        self,
        config_file_path: str,
        study_data_pvc_file_path: str,
        timeout=None,
        namespace=DEFAULT_NAMESPACE,
        pending_timeout=DEFAULT_PENDING_TIMEOUT,
        python_path=DEFAULT_PYTHON_PATH,
        security_context: dict = None,
    ):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config_file_path = config_file_path
        self.study_data_pvc_file_path = study_data_pvc_file_path
        self.timeout = timeout
        self.namespace = namespace
        self.pending_timeout = pending_timeout
        self.python_path = python_path
        self.security_context = security_context
        self.study_data_pvc_dict = None
        self.default_data_pvc = None
        self.core_v1 = None
        self._ws_server: WorkspaceHTTPServer | None = None
        self._ws_server_lock = threading.Lock()

    def _ensure_startup_secret(self, site_name: str, startup_dir: str) -> str:
        """Create or update a k8s Secret containing the site startup kit.

        Returns the Secret name.
        """
        from kubernetes.client.rest import ApiException

        secret_name = f"nvflare-startup-{site_name.lower().replace('_', '-')}"
        data = {}
        if os.path.isdir(startup_dir):
            for fname in os.listdir(startup_dir):
                if not _keep_startup_file(fname):
                    continue
                fpath = os.path.join(startup_dir, fname)
                if os.path.isfile(fpath):
                    with open(fpath, "rb") as f:
                        data[fname] = base64.b64encode(f.read()).decode()

        secret_body = {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {"name": secret_name, "namespace": self.namespace},
            "type": "Opaque",
            "data": data,
        }
        try:
            self.core_v1.read_namespaced_secret(name=secret_name, namespace=self.namespace)
            self.core_v1.replace_namespaced_secret(name=secret_name, namespace=self.namespace, body=secret_body)
            self.logger.debug("Updated startup Secret %s", secret_name)
        except ApiException as e:
            if getattr(e, "status", None) == 404:
                self.core_v1.create_namespaced_secret(namespace=self.namespace, body=secret_body)
                self.logger.debug("Created startup Secret %s", secret_name)
            else:
                raise
        return secret_name

    def launch_job(self, job_meta: dict, fl_ctx: FLContext) -> JobHandleSpec:
        if self.default_data_pvc is None:
            with open(self.study_data_pvc_file_path, "rt") as f:
                study_data_pvc_dict = yaml.safe_load(f)
            if not study_data_pvc_dict:
                raise ValueError(
                    f"study_data_pvc_file_path '{self.study_data_pvc_file_path}' is empty or contains no PVC entries."
                )
            # study_data_pvc_file_path file is
            # a yaml file with this format
            # study_name_1: data_pvc_1
            # study_name_2: data_pvc_2
            # ...
            # ...
            # default: default_data_pvc
            # currently, support one pvc and always mount to /var/tmp/nvflare/data
            if not isinstance(study_data_pvc_dict, dict):
                raise ValueError(
                    f"file at study_data_pvc_file_path '{self.study_data_pvc_file_path}' does not contain a dictionary."
                )
            default_data_pvc = study_data_pvc_dict.get("default")
            if default_data_pvc is None:
                raise ValueError(f"No default PVC found in '{self.study_data_pvc_file_path}'.")
            self.default_data_pvc = default_data_pvc
            self.study_data_pvc_dict = study_data_pvc_dict
        if self.core_v1 is None:
            from kubernetes import config
            from kubernetes.client import Configuration
            from kubernetes.client.api import core_v1_api

            try:
                if self.config_file_path:
                    config.load_kube_config(self.config_file_path)
                else:
                    config.load_incluster_config()
                c = Configuration().get_default_copy()
            except AttributeError:
                c = Configuration()
                c.assert_hostname = False
            Configuration.set_default(c)
            self.core_v1 = core_v1_api.CoreV1Api()
        site_name = fl_ctx.get_identity_name()
        raw_job_id = job_meta.get(JobConstants.JOB_ID)
        if not raw_job_id:
            raise RuntimeError(f"missing {JobConstants.JOB_ID} in job_meta")
        job_id = uuid4_to_rfc1123(raw_job_id)
        workspace_obj = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
        if workspace_obj is None:
            raise RuntimeError(f"missing {FLContextKey.WORKSPACE_OBJECT} in FLContext")
        app_custom_folder = workspace_obj.get_app_custom_dir(raw_job_id)
        args = fl_ctx.get_prop(FLContextKey.ARGS)
        if args is None:
            raise RuntimeError(f"missing {FLContextKey.ARGS} in FLContext")
        job_image = extract_job_image(job_meta, site_name)
        site_resources = get_launcher_resource_spec(job_meta, site_name, "k8s")
        study = job_meta.get(JobMetaKey.STUDY.value)
        job_resource = site_resources.get("num_of_gpus", None)
        job_args = fl_ctx.get_prop(FLContextKey.JOB_PROCESS_ARGS)
        if not job_args:
            raise RuntimeError(f"missing {FLContextKey.JOB_PROCESS_ARGS} in FLContext")

        exe_module_entry = job_args.get(JobProcessArgs.EXE_MODULE)
        if not exe_module_entry:
            raise RuntimeError(f"missing {JobProcessArgs.EXE_MODULE} in {FLContextKey.JOB_PROCESS_ARGS}")
        _, job_cmd = exe_module_entry
        data_pvc = self.study_data_pvc_dict.get(study, self.default_data_pvc)

        env = {}
        if app_custom_folder:
            env["PYTHONPATH"] = app_custom_folder

        workspace_root = args.workspace
        startup_dir = workspace_obj.get_startup_kit_dir()

        # Start the shared HTTPS server once; reuse it for all subsequent jobs.
        # Double-checked lock so concurrent first-launches cannot start two servers.
        if self._ws_server is None:
            with self._ws_server_lock:
                if self._ws_server is None:
                    cert_file, key_file = _get_workspace_server_tls_files(startup_dir)
                    srv = WorkspaceHTTPServer(cert_file=cert_file, key_file=key_file)
                    srv.start(workspace_root=workspace_root)
                    self._ws_server = srv

        # Register this job; get its unguessable URL token.
        url_token = self._ws_server.add_job(raw_job_id, workspace_root)

        # Determine the IP that job pods use to reach this process.
        # In k8s, socket.gethostname() returns the pod name which resolves
        # to the pod IP -- this works without any extra Helm configuration.
        parent_host = socket.gethostbyname(socket.gethostname())

        startup_secret_name = self._ensure_startup_secret(site_name, startup_dir)

        env[ENV_WORKSPACE_URL] = self._ws_server.get_url(parent_host, url_token)

        volume_list = [
            {"name": "workspace-job", "emptyDir": {"sizeLimit": DEFAULT_EPHEMERAL_STORAGE}},
            {"name": "startup-kit", "secret": {"secretName": startup_secret_name}},
            {"name": DATA_PVC_VOLUME_NAME, "persistentVolumeClaim": {"claimName": data_pvc}},
        ]
        volume_mount_list = [
            {"name": "workspace-job", "mountPath": WORKSPACE_MOUNT_PATH},
            {"name": "startup-kit", "mountPath": f"{WORKSPACE_MOUNT_PATH}/startup", "readOnly": True},
            {"name": DATA_PVC_VOLUME_NAME, "mountPath": "/var/tmp/nvflare/data", "readOnly": True},
        ]

        job_config = {
            "name": job_id,
            "image": job_image,
            "container_name": f"container-{job_id}",
            "command": job_cmd,
            "volume_mount_list": volume_mount_list,
            "volume_list": volume_list,
            "module_args": self.get_module_args(job_id, fl_ctx),
            "env": env,
        }
        if args is not None and getattr(args, "set", None) is not None:
            job_config.update({"set_list": args.set})
        resources = {
            "requests": {"ephemeral-storage": DEFAULT_EPHEMERAL_STORAGE},
            "limits": {"ephemeral-storage": DEFAULT_EPHEMERAL_STORAGE},
        }
        if job_resource:
            resources["limits"]["nvidia.com/gpu"] = job_resource
        job_config["resources"] = resources
        if self.security_context:
            job_config["security_context"] = self.security_context
        job_handle = K8sJobHandle(
            job_id,
            self.core_v1,
            job_config,
            namespace=self.namespace,
            timeout=self.timeout,
            pending_timeout=self.pending_timeout,
            python_path=self.python_path,
            workspace_server=self._ws_server,
            workspace_token=url_token,
        )
        pod_manifest = job_handle.get_manifest()
        self.logger.debug(f"launch job with k8s_launcher. {pod_manifest=}")
        try:
            self.core_v1.create_namespaced_pod(body=pod_manifest, namespace=self.namespace)
        except Exception as e:
            self.logger.error(f"failed to launch job {job_id}: {e}")
            job_handle.terminal_state = JobState.TERMINATED
            if self._ws_server is not None:
                self._ws_server.remove_job(url_token)
            return job_handle
        try:
            entered_running = job_handle.enter_states([JobState.RUNNING])
        except BaseException:
            job_handle.terminate()
            raise
        if not entered_running:
            self.logger.warning(f"unable to enter running phase {job_id}")
        return job_handle

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.BEFORE_JOB_LAUNCH:
            job_meta = fl_ctx.get_prop(FLContextKey.JOB_META)
            job_image = extract_job_image(job_meta, fl_ctx.get_identity_name())
            if job_image:
                add_launcher(self, fl_ctx)

    @abstractmethod
    def get_module_args(self, job_id, fl_ctx: FLContext):
        """To get the args to run the launcher

        Args:
            job_id: run job_id
            fl_ctx: FLContext

        Returns:

        """
        pass


def _job_args_dict(job_args: dict, arg_names: list) -> dict:
    result = {}
    for name in arg_names:
        e = job_args.get(name)
        if e is None:
            continue

        n, v = e
        result[n] = v
    return result


class ClientK8sJobLauncher(K8sJobLauncher):
    def get_module_args(self, _job_id, fl_ctx: FLContext):
        job_args = fl_ctx.get_prop(FLContextKey.JOB_PROCESS_ARGS)
        if not job_args:
            raise RuntimeError(f"missing {FLContextKey.JOB_PROCESS_ARGS} in FLContext")

        return _job_args_dict(job_args, get_client_job_args(False, False))


class ServerK8sJobLauncher(K8sJobLauncher):
    def get_module_args(self, _job_id, fl_ctx: FLContext):
        job_args = fl_ctx.get_prop(FLContextKey.JOB_PROCESS_ARGS)
        if not job_args:
            raise RuntimeError(f"missing {FLContextKey.JOB_PROCESS_ARGS} in FLContext")

        return _job_args_dict(job_args, get_server_job_args(False, False))
