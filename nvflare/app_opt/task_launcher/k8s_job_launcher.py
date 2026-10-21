# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Kubernetes CP profile for a CPU CJ with explicitly durable task storage."""

import hashlib
import json
import os
import re

from nvflare.apis.fl_constant import FLContextKey, JobConstants
from nvflare.app_opt.job_launcher.k8s_launcher import ClientK8sJobLauncher
from nvflare.private.fed.client.task_execution_policy import GPU_ENVIRONMENT_NAMES
from nvflare.utils.job_launcher_utils import get_resource_manager_spec


def _dns_name(name, value):
    if not isinstance(value, str) or len(value) > 63 or not re.fullmatch(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?", value):
        raise ValueError(f"{name} must be a DNS label")
    return value


def task_workspace_subpath(job_id, site_name):
    """Separate each site's job workspace without embedding untrusted path components."""
    digest = hashlib.sha256(json.dumps([job_id, site_name]).encode()).hexdigest()
    return f"nvflare-tasks/{digest}"


def task_security_context(uid, gid):
    for name, value in (("run_as_user", uid), ("run_as_group", gid)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be an explicit positive numeric ID shared by CJ and workers")
    return {"runAsUser": uid, "runAsGroup": gid, "runAsNonRoot": True, "fsGroup": gid}


def reject_job_gpu_reservation(meta, site_name):
    if get_resource_manager_spec(meta, site_name).get("num_of_gpus", 0):
        raise ValueError(
            "Kubernetes task GPUs require native launcher_spec k8s num_of_gpus; "
            "whole-job CP GPU reservations are not supported by this task profile"
        )


class ClientK8sTaskJobLauncher(ClientK8sJobLauncher):
    """Site-selected profile; the application's sole opt-in remains execution_lifetime.

    Uses the resident launcher's workspace download, startup Secret and job
    credential transport. Only task-lifetime CJs receive the PVC/CPU profile.
    Shared runtime selection validates this matching site-owned profile.
    """

    def __init__(
        self,
        config_file_path,
        task_workspace_pvc: str,
        task_workspace_pvc_root: str,
        task_cj_service_account: str,
        run_as_user: int,
        run_as_group: int,
        **kwargs,
    ):
        super().__init__(config_file_path=config_file_path, **kwargs)
        self.task_workspace_pvc = _dns_name("task_workspace_pvc", task_workspace_pvc)
        if not isinstance(task_workspace_pvc_root, str) or not os.path.isabs(task_workspace_pvc_root):
            raise ValueError("task_workspace_pvc_root must name the CP's explicit mount of the existing PVC")
        self.task_workspace_pvc_root = os.path.realpath(task_workspace_pvc_root)
        self.task_cj_service_account = _dns_name("task_cj_service_account", task_cj_service_account)
        self.task_security_context = task_security_context(run_as_user, run_as_group)

    def launch_job(self, job_meta, fl_ctx):
        workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
        if workspace is not None:
            job_id = job_meta.get(JobConstants.JOB_ID)
            with open(workspace.get_client_app_config_file_path(job_id)) as stream:
                lifetime = json.load(stream).get("execution_lifetime", "job")
            if lifetime == "task":
                reject_job_gpu_reservation(job_meta, fl_ctx.get_identity_name())
        return super().launch_job(job_meta, fl_ctx)

    def prepare_job_manifest(self, manifest, job_id, site_name, workspace):
        with open(workspace.get_client_app_config_file_path(job_id)) as stream:
            client_config = json.load(stream)
        lifetime = client_config.get("execution_lifetime", "job")
        if lifetime == "job":
            return
        if lifetime != "task":
            raise ValueError("execution_lifetime must be 'job' or 'task'")
        spec = manifest["spec"]
        if (
            len(spec["containers"]) != 1
            or spec.get("initContainers")
            or spec.get("ephemeralContainers")
            or any(spec.get(name) for name in ("hostPID", "hostIPC", "hostNetwork"))
        ):
            raise ValueError("the Kubernetes task CJ profile requires one isolated container")
        volumes = [volume for volume in spec["volumes"] if volume["name"] == "workspace-job"]
        if len(volumes) != 1:
            raise ValueError("task CJ requires the resident workspace-job volume")
        volumes[0].clear()
        volumes[0].update(name="workspace-job", persistentVolumeClaim={"claimName": self.task_workspace_pvc})
        container = spec["containers"][0]
        mounts = [mount for mount in container["volumeMounts"] if mount["name"] == "workspace-job"]
        if len(mounts) != 1 or mounts[0]["mountPath"] != self.workspace_mount_path:
            raise ValueError("task CJ workspace mount differs from the configured durable workspace")
        mounts[0]["subPath"] = self._prepare_workspace_subpath(job_id, site_name)
        for amounts in container.get("resources", {}).values():
            for resource in list(amounts):
                if "gpu" in resource.lower():
                    del amounts[resource]
        env = [item for item in container.get("env", []) if item["name"] not in GPU_ENVIRONMENT_NAMES]
        env.extend(
            {"name": name, "value": "none" if name == "NVIDIA_VISIBLE_DEVICES" else ""}
            for name in sorted(GPU_ENVIRONMENT_NAMES)
        )
        container["env"] = env
        container["securityContext"] = {
            "runAsUser": self.task_security_context["runAsUser"],
            "runAsGroup": self.task_security_context["runAsGroup"],
            "runAsNonRoot": True,
            "allowPrivilegeEscalation": False,
            "capabilities": {"drop": ["ALL"]},
        }
        spec["securityContext"] = dict(self.task_security_context)
        spec["serviceAccountName"] = self.task_cj_service_account
        spec["automountServiceAccountToken"] = True

    def _prepare_workspace_subpath(self, job_id, site_name):
        subpath = task_workspace_subpath(job_id, site_name)
        # CP explicitly mounts the same PVC. Create a fresh, owned job subtree
        # instead of relying on kubelet-created subPath directory permissions.
        # Never reuse retained data after a CJ/CP restart or failed launch.
        from nvflare.app_opt.task_launcher.k8s_launcher import check_posix_storage

        root = self.task_workspace_pvc_root
        if not os.path.isdir(root):
            raise ValueError("the task workspace PVC must already be mounted in CP")
        check_posix_storage(root)
        parent = os.path.join(root, "nvflare-tasks")
        os.makedirs(parent, mode=0o700, exist_ok=True)
        if os.path.islink(parent) or os.path.realpath(parent) != parent:
            raise ValueError("task workspace parent must be a real directory on the configured PVC")
        directory = os.path.join(root, subpath)
        os.mkdir(directory, mode=0o700)
        info = os.stat(directory)
        if info.st_uid != self.task_security_context["runAsUser"]:
            raise ValueError("CP, CJ and task workers must use the same nonroot numeric UID")
        return subpath
