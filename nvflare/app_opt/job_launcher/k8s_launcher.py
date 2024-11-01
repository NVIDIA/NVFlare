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
import time
from abc import abstractmethod
from enum import Enum

from kubernetes import config
from kubernetes.client import Configuration
from kubernetes.client.api import core_v1_api
from kubernetes.client.rest import ApiException

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, JobConstants
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_launcher_spec import JobHandleSpec, JobLauncherSpec, JobReturnCode, add_launcher
from nvflare.apis.workspace import Workspace
from nvflare.private.fed.utils.fed_utils import extract_job_image


class JobState(Enum):
    STARTING = "starting"
    RUNNING = "running"
    TERMINATED = "terminated"
    SUCCEEDED = "succeeded"
    UNKNOWN = "unknown"


POD_STATE_MAPPING = {
    "Pending": JobState.STARTING,
    "Running": JobState.RUNNING,
    "Succeeded": JobState.SUCCEEDED,
    "Failed": JobState.TERMINATED,
    "Unknown": JobState.UNKNOWN,
}

JOB_RETURN_CODE_MAPPING = {
    JobState.SUCCEEDED: JobReturnCode.SUCCESS,
    JobState.STARTING: JobReturnCode.UNKNOWN,
    JobState.RUNNING: JobReturnCode.UNKNOWN,
    JobState.TERMINATED: JobReturnCode.ABORTED,
    JobState.UNKNOWN: JobReturnCode.UNKNOWN,
}


class K8sJobHandle(JobHandleSpec):
    def __init__(self, job_id: str, api_instance: core_v1_api, job_config: dict, namespace="default", timeout=None):
        super().__init__()
        self.job_id = job_id
        self.timeout = timeout

        self.api_instance = api_instance
        self.namespace = namespace
        self.pod_manifest = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {"name": None},  # set by job_config['name']
            "spec": {
                "containers": None,  # link to container_list
                "volumes": None,  # link to volume_list
                "restartPolicy": "OnFailure",
            },
        }
        self.volume_list = [{"name": None, "hostPath": {"path": None, "type": "Directory"}}]
        self.container_list = [
            {
                "image": None,
                "name": None,
                "command": ["/usr/local/bin/python"],
                "args": None,  # args_list + args_dict + args_sets
                "volumeMounts": None,  # volume_mount_list
                "imagePullPolicy": "Always",
            }
        ]
        self.container_args_python_args_list = ["-u", "-m", job_config.get("command")]
        self.container_args_module_args_dict = {
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
        self.container_volume_mount_list = [
            {
                "name": None,
                "mountPath": None,
            }
        ]
        self._make_manifest(job_config)

    def _make_manifest(self, job_config):
        self.container_volume_mount_list = job_config.get(
            "volume_mount_list", [{"name": "workspace-nvflare", "mountPath": "/workspace/nvflare"}]
        )
        set_list = job_config.get("set_list")
        if set_list is None:
            self.container_args_module_args_sets = list()
        else:
            self.container_args_module_args_sets = ["--set"] + set_list
        self.container_args_module_args_dict = job_config.get(
            "module_args",
            {
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
            },
        )
        self.container_args_module_args_dict_as_list = list()
        for k, v in self.container_args_module_args_dict.items():
            self.container_args_module_args_dict_as_list.append(k)
            self.container_args_module_args_dict_as_list.append(v)
        self.volume_list = job_config.get(
            "volume_list", [{"name": None, "hostPath": {"path": None, "type": "Directory"}}]
        )

        self.pod_manifest["metadata"]["name"] = job_config.get("name")
        self.pod_manifest["spec"]["containers"] = self.container_list
        self.pod_manifest["spec"]["volumes"] = self.volume_list

        self.container_list[0]["image"] = job_config.get("image", "nvflare/nvflare:2.5.0")
        self.container_list[0]["name"] = job_config.get("container_name", "nvflare_job")
        self.container_list[0]["args"] = (
            self.container_args_python_args_list
            + self.container_args_module_args_dict_as_list
            + self.container_args_module_args_sets
        )
        self.container_list[0]["volumeMounts"] = self.container_volume_mount_list

    def get_manifest(self):
        return self.pod_manifest

    def enter_states(self, job_states_to_enter: list, timeout=None):
        starting_time = time.time()
        if not isinstance(job_states_to_enter, (list, tuple)):
            job_states_to_enter = [job_states_to_enter]
        if not all([isinstance(js, JobState)] for js in job_states_to_enter):
            raise ValueError(f"expect job_states_to_enter with valid values, but get {job_states_to_enter}")
        while True:
            job_state = self._query_state()
            if job_state in job_states_to_enter:
                return True
            elif timeout is not None and time.time() - starting_time > timeout:
                return False
            time.sleep(1)

    def terminate(self):
        resp = self.api_instance.delete_namespaced_pod(
            name=self.job_id, namespace=self.namespace, grace_period_seconds=0
        )
        return self.enter_states([JobState.TERMINATED], timeout=self.timeout)

    def poll(self):
        job_state = self._query_state()
        return JOB_RETURN_CODE_MAPPING.get(job_state, JobReturnCode.UNKNOWN)

    def _query_state(self):
        try:
            resp = self.api_instance.read_namespaced_pod(name=self.job_id, namespace=self.namespace)
        except ApiException as e:
            return JobState.UNKNOWN
        return POD_STATE_MAPPING.get(resp.status.phase, JobState.UNKNOWN)

    def wait(self):
        self.enter_states([JobState.SUCCEEDED, JobState.TERMINATED])


class K8sJobLauncher(JobLauncherSpec):
    def __init__(
        self,
        config_file_path,
        root_hostpath: str,
        workspace: str,
        mount_path: str,
        timeout=None,
        namespace="default",
    ):
        super().__init__()

        self.root_hostpath = root_hostpath
        self.workspace = workspace
        self.mount_path = mount_path
        self.timeout = timeout

        config.load_kube_config(config_file_path)
        try:
            c = Configuration().get_default_copy()
        except AttributeError:
            c = Configuration()
            c.assert_hostname = False
        Configuration.set_default(c)
        self.core_v1 = core_v1_api.CoreV1Api()
        self.namespace = namespace

        self.job_handle = None
        self.logger = logging.getLogger(self.__class__.__name__)

    def launch_job(self, job_meta: dict, fl_ctx: FLContext) -> JobHandleSpec:

        job_id = job_meta.get(JobConstants.JOB_ID)
        args = fl_ctx.get_prop(FLContextKey.ARGS)
        job_image = extract_job_image(job_meta, fl_ctx.get_identity_name())
        self.logger.info(f"launch job use image: {job_image}")
        job_config = {
            "name": job_id,
            "image": job_image,
            "container_name": f"container-{job_id}",
            "command": self.get_command(),
            "volume_mount_list": [{"name": self.workspace, "mountPath": self.mount_path}],
            "volume_list": [{"name": self.workspace, "hostPath": {"path": self.root_hostpath, "type": "Directory"}}],
            "module_args": self.get_module_args(job_id, fl_ctx),
            "set_list": self.get_set_list(args, fl_ctx),
        }

        self.logger.info(f"launch job with k8s_launcher. Job_id:{job_id}")

        job_handle = K8sJobHandle(job_id, self.core_v1, job_config, namespace=self.namespace, timeout=self.timeout)
        try:
            self.core_v1.create_namespaced_pod(body=job_handle.get_manifest(), namespace=self.namespace)
            if job_handle.enter_states([JobState.RUNNING], timeout=self.timeout):
                return job_handle
            else:
                job_handle.terminate()
                return None
        except ApiException as e:
            job_handle.terminate()
            return None

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.GET_JOB_LAUNCHER:
            job_meta = fl_ctx.get_prop(FLContextKey.JOB_META)
            job_image = extract_job_image(job_meta, fl_ctx.get_identity_name())
            if job_image:
                add_launcher(self, fl_ctx)

    @abstractmethod
    def get_command(self):
        """To get the run command of the launcher

        Returns: the command for the launcher process

        """
        pass

    @abstractmethod
    def get_module_args(self, job_id, fl_ctx: FLContext):
        """To get the args to run the launcher

        Args:
            job_id: run job_id
            fl_ctx: FLContext

        Returns:

        """
        pass

    @abstractmethod
    def get_set_list(self, args, fl_ctx: FLContext):
        """To get the command set_list

        Args:
            args: command args
            fl_ctx: FLContext

        Returns: set_list command options

        """
        pass


class ClientK8sJobLauncher(K8sJobLauncher):
    def get_command(self):
        return "nvflare.private.fed.app.client.worker_process"

    def get_module_args(self, job_id, fl_ctx: FLContext):
        workspace_obj: Workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
        args = fl_ctx.get_prop(FLContextKey.ARGS)
        client = fl_ctx.get_prop(FLContextKey.SITE_OBJ)
        server_config = fl_ctx.get_prop(FLContextKey.SERVER_CONFIG)
        if not server_config:
            raise RuntimeError(f"missing {FLContextKey.SERVER_CONFIG} in FL context")
        service = server_config[0].get("service", {})
        if not isinstance(service, dict):
            raise RuntimeError(f"expect server config data to be dict but got {type(service)}")
        self.logger.info(f"K8sJobLauncher start to launch job: {job_id} for client: {client.client_name}")

        return {
            "-m": args.workspace,
            "-w": (workspace_obj.get_startup_kit_dir()),
            "-t": client.token,
            "-d": client.ssid,
            "-n": job_id,
            "-c": client.client_name,
            "-p": str(client.cell.get_internal_listener_url()),
            "-g": service.get("target"),
            "-scheme": service.get("scheme", "grpc"),
            "-s": "fed_client.json",
        }

    def get_set_list(self, args, fl_ctx: FLContext):
        args.set.append("print_conf=True")
        return args.set


class ServerK8sJobLauncher(K8sJobLauncher):
    def get_command(self):
        return "nvflare.private.fed.app.server.runner_process"

    def get_module_args(self, job_id, fl_ctx: FLContext):
        workspace_obj: Workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
        args = fl_ctx.get_prop(FLContextKey.ARGS)
        server = fl_ctx.get_prop(FLContextKey.SITE_OBJ)

        return {
            "-m": args.workspace,
            "-s": "fed_server.json",
            "-r": workspace_obj.get_app_dir(),
            "-n": str(job_id),
            "-p": str(server.cell.get_internal_listener_url()),
            "-u": str(server.cell.get_root_url_for_child()),
            "--host": str(server.server_state.host),
            "--port": str(server.server_state.service_port),
            "--ssid": str(server.server_state.ssid),
            "--ha_mode": str(server.ha_mode),
        }

    def get_set_list(self, args, fl_ctx: FLContext):
        restore_snapshot = fl_ctx.get_prop(FLContextKey.SNAPSHOT, False)
        args.set.append("print_conf=True")
        args.set.append("restore_snapshot=" + str(restore_snapshot))
        return args.set
