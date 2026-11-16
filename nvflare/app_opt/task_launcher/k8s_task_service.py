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

"""Kubernetes-specific CP ownership and workspace policy for task control."""

from nvflare.app_opt.job_launcher.workspace_cell_transfer import ENV_WORKSPACE_OWNER_FQCN, ENV_WORKSPACE_TRANSFER_TOKEN
from nvflare.private.fed.task_worker.task_service import CPRemoteTaskHandle, CPTaskClient, CPTaskService
from nvflare.private.fed.task_worker.task_service import _Job as _BaseJob

_MAX_CANCEL_FENCES = 128


class _Job(_BaseJob):
    pending_id = "k8s:pending"
    unit_name = "Pod"

    def attach_cj(self, name, uid):
        if not isinstance(uid, str) or not uid:
            raise ValueError("CP task owner requires the created CJ UID")
        super().attach_cj((name, uid))

    def _cj_is_dead(self, handle):
        name, uid = handle
        try:
            pod = self.backend.core_v1.read_namespaced_pod(
                name=name,
                namespace=self.backend.namespace,
                _request_timeout=self.backend.request_timeout,
            )
            return (
                pod.metadata.uid != uid
                or pod.status.phase in ("Succeeded", "Failed")
                or bool(pod.metadata.deletion_timestamp)
            )
        except Exception as error:
            if getattr(error, "status", None) == 404:
                return True
            raise


class K8sTaskService(CPTaskService):
    channel = "k8s_task_control"
    backend_name = "Kubernetes"
    service_key = "k8s"
    job_class = _Job

    @property
    def max_cancel_fences(self):
        return _MAX_CANCEL_FENCES

    def _check_path(self, job, path):
        from nvflare.app_opt.task_launcher.k8s_launcher import _within

        _within(job.backend.workspace_root, path)


class K8sRemoteTaskHandle(CPRemoteTaskHandle):
    pending_id = "k8s:pending"


class K8sTaskClient(CPTaskClient):
    channel = "k8s_task_control"
    backend_name = "Kubernetes"
    owner_env = ENV_WORKSPACE_OWNER_FQCN
    token_env = ENV_WORKSPACE_TRANSFER_TOKEN
    handle_class = K8sRemoteTaskHandle
