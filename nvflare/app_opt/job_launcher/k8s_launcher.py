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

import yaml
from kubernetes import config
from kubernetes.client import Configuration
from kubernetes.client.api import core_v1_api
from kubernetes.client.rest import ApiException

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, JobConstants
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import JobMetaKey
from nvflare.apis.job_launcher_spec import JobHandleSpec, JobLauncherSpec, JobProcessArgs, JobReturnCode, add_launcher
from nvflare.utils.job_launcher_utils import extract_job_image, get_client_job_args, get_server_job_args


class JobState(Enum):
    STARTING = "starting"
    RUNNING = "running"
    TERMINATED = "terminated"
    SUCCEEDED = "succeeded"
    UNKNOWN = "unknown"


class POD_Phase(Enum):
    PENDING = "Pending"
    RUNNING = "Running"
    SUCCEEDED = "Succeeded"
    FAILED = "Failed"
    TERMINATED = "Terminated"
    UNKNOWN = "Unknown"


POD_STATE_MAPPING = {
    POD_Phase.PENDING.value: JobState.STARTING,
    POD_Phase.RUNNING.value: JobState.RUNNING,
    POD_Phase.SUCCEEDED.value: JobState.SUCCEEDED,
    POD_Phase.FAILED.value: JobState.TERMINATED,
    POD_Phase.TERMINATED.value: JobState.TERMINATED,
    POD_Phase.UNKNOWN.value: JobState.UNKNOWN,
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


class PV_NAME(Enum):
    WORKSPACE = "nvflws"
    DATA = "nvfldata"
    ETC = "nvfletc"


VOLUME_MOUNT_LIST = [
    {"name": PV_NAME.WORKSPACE.value, "mountPath": "/var/tmp/nvflare/workspace"},
    {"name": PV_NAME.DATA.value, "mountPath": "/var/tmp/nvflare/data"},
    {"name": PV_NAME.ETC.value, "mountPath": "/var/tmp/nvflare/etc"},
]


class K8sJobHandle(JobHandleSpec):
    def __init__(self, job_id: str, api_instance: core_v1_api, job_config: dict, namespace="default", timeout=None):
        super().__init__()
        self.job_id = job_id
        self.timeout = timeout
        self.terminal_state = None
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
                "resources": None,
                "command": ["/usr/local/bin/python"],
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
        self._last_phase = None
        self._stuck_count = -10
        self._max_stuck_count = self.timeout
        self.logger = logging.getLogger(self.__class__.__name__)

    def _make_manifest(self, job_config):
        self.container_volume_mount_list.extend(job_config.get("volume_mount_list", []))
        set_list = job_config.get("set_list")
        if set_list is None:
            self.container_args_module_args_sets = list()
        else:
            self.container_args_module_args_sets = ["--set"] + set_list
        if job_config.get("module_args") is None:
            self.container_args_module_args_dict = DEFAULT_CONTAINER_ARGS_MODULE_ARGS_DICT
        else:
            self.container_args_module_args_dict = job_config.get("module_args")
        self.container_args_module_args_dict_as_list = list()
        for k, v in self.container_args_module_args_dict.items():
            self.container_args_module_args_dict_as_list.append(k)
            self.container_args_module_args_dict_as_list.append(v)
        self.volume_list.extend(job_config.get("volume_list", []))
        self.pod_manifest["metadata"]["name"] = job_config.get("name")
        self.pod_manifest["spec"]["containers"] = self.container_list
        self.pod_manifest["spec"]["volumes"] = self.volume_list

        self.container_list[0]["image"] = job_config.get("image", "nvflare/nvflare:2.8.0")
        self.container_list[0]["name"] = job_config.get("container_name", "nvflare_job")
        self.container_list[0]["args"] = (
            self.container_args_python_args_list
            + self.container_args_module_args_dict_as_list
            + self.container_args_module_args_sets
        )
        self.container_list[0]["volumeMounts"] = self.container_volume_mount_list
        if job_config.get("resources", {}).get("limits", {}).get("nvidia.com/gpu") is not None:
            self.container_list[0]["resources"] = job_config.get("resources")

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
        self.enter_states([JobState.TERMINATED], timeout=self.timeout)
        self.terminal_state = JobState.TERMINATED
        return None

    def poll(self):
        if self.terminal_state is not None:
            return self.terminal_state
        job_state = self._query_state()
        return JOB_RETURN_CODE_MAPPING.get(job_state, JobReturnCode.UNKNOWN)

    def _query_state(self):
        try:
            resp = self.api_instance.read_namespaced_pod(name=self.job_id, namespace=self.namespace)
        except ApiException as e:
            return JobState.UNKNOWN
        if self._stuck(resp.status.phase):
            self.terminate()
            return JobState.TERMINATED
        return POD_STATE_MAPPING.get(resp.status.phase, JobState.UNKNOWN)

    def _stuck(self, current_phase):
        if self._max_stuck_count is None:
            return False
        if current_phase == POD_Phase.PENDING.value:
            self._stuck_count += 1
            if self._stuck_count > self._max_stuck_count:
                return True
        return False

    def wait(self):
        while True:
            if self.terminal_state is not None:
                return
            job_state = self._query_state()
            if job_state in (JobState.SUCCEEDED, JobState.TERMINATED):
                return
            time.sleep(1)


class K8sJobLauncher(JobLauncherSpec):
    def __init__(
        self,
        config_file_path: str,
        workspace_pvc: str,
        etc_pvc: str,
        data_pvc_file_path: str,
        timeout=None,
        namespace="default",
    ):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

        self.workspace_pvc = workspace_pvc
        self.etc_pvc = etc_pvc
        self.data_pvc_file_path = data_pvc_file_path
        self.timeout = timeout
        self.namespace = namespace
        with open(data_pvc_file_path, "rt") as f:
            data_pvc_dict = yaml.safe_load(f)
        if not data_pvc_dict:
            raise ValueError(f"data_pvc_file_path '{data_pvc_file_path}' is empty or contains no PVC entries.")
        # data_pvc_dict will be pvc: mountPath
        # currently, support one pvc and always mount to /var/tmp/nvflare/data
        # ie, ignore the mountPath in data_pvc_dict
        self.data_pvc = list(data_pvc_dict.keys())[0]

        config.load_kube_config(config_file_path)
        try:
            c = Configuration().get_default_copy()
        except AttributeError:
            c = Configuration()
            c.assert_hostname = False
        Configuration.set_default(c)
        self.core_v1 = core_v1_api.CoreV1Api()
        self.job_handle = None

    def launch_job(self, job_meta: dict, fl_ctx: FLContext) -> JobHandleSpec:
        site_name = fl_ctx.get_identity_name()
        job_id = job_meta.get(JobConstants.JOB_ID)
        args = fl_ctx.get_prop(FLContextKey.ARGS)
        job_image = extract_job_image(job_meta, site_name)
        site_resources = job_meta.get(JobMetaKey.RESOURCE_SPEC.value, {}).get(site_name, {})
        job_resource = site_resources.get("num_of_gpus", None)

        job_args = fl_ctx.get_prop(FLContextKey.JOB_PROCESS_ARGS)
        if not job_args:
            raise RuntimeError(f"missing {FLContextKey.JOB_PROCESS_ARGS} in FLContext")

        _, job_cmd = job_args[JobProcessArgs.EXE_MODULE]
        job_config = {
            "name": job_id,
            "image": job_image,
            "container_name": f"container-{job_id}",
            "command": job_cmd,
            "volume_mount_list": VOLUME_MOUNT_LIST,
            "volume_list": [
                {"name": PV_NAME.WORKSPACE.value, "persistentVolumeClaim": {"claimName": self.workspace_pvc}},
                {"name": PV_NAME.DATA.value, "persistentVolumeClaim": {"claimName": self.data_pvc}},
                {"name": PV_NAME.ETC.value, "persistentVolumeClaim": {"claimName": self.etc_pvc}},
            ],
            "module_args": self.get_module_args(job_id, fl_ctx),
            "set_list": args.set,
            "resources": {"limits": {"nvidia.com/gpu": job_resource}},
        }

        job_handle = K8sJobHandle(job_id, self.core_v1, job_config, namespace=self.namespace, timeout=self.timeout)
        pod_manifest = job_handle.get_manifest()
        self.logger.info(f"launch job with k8s_launcher. {pod_manifest=}")
        try:
            self.core_v1.create_namespaced_pod(body=pod_manifest, namespace=self.namespace)
            job_handle.enter_states([JobState.RUNNING], timeout=self.timeout)
            return job_handle
        except ApiException as e:
            job_handle.terminate()
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
        if not e:
            continue

        n, v = e
        result[n] = v
    return result


class ClientK8sJobLauncher(K8sJobLauncher):
    def get_module_args(self, job_id, fl_ctx: FLContext):
        job_args = fl_ctx.get_prop(FLContextKey.JOB_PROCESS_ARGS)
        if not job_args:
            raise RuntimeError(f"missing {FLContextKey.JOB_PROCESS_ARGS} in FLContext")

        return _job_args_dict(job_args, get_client_job_args(False, False))


class ServerK8sJobLauncher(K8sJobLauncher):
    def get_module_args(self, job_id, fl_ctx: FLContext):
        job_args = fl_ctx.get_prop(FLContextKey.JOB_PROCESS_ARGS)
        if not job_args:
            raise RuntimeError(f"missing {FLContextKey.JOB_PROCESS_ARGS} in FLContext")

        return _job_args_dict(job_args, get_server_job_args(False, False))
