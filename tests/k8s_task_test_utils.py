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

"""In-memory Kubernetes API/Cell doubles shared by task policy and lifecycle tests."""

import copy
from types import SimpleNamespace as NS
from unittest.mock import Mock

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.task_launcher_spec import TaskLaunchRequest
from nvflare.app_opt.job_launcher.workspace_cell_transfer import ENV_WORKSPACE_OWNER_FQCN, ENV_WORKSPACE_TRANSFER_TOKEN
from nvflare.app_opt.task_launcher.k8s_launcher import WORKER_MODULE, K8sTaskBackend
from nvflare.app_opt.task_launcher.k8s_task_service import K8sTaskService
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.private.fed.task_worker.artifacts import FileTaskArtifactStore
from nvflare.private.fed.task_worker.protocol import TaskAttemptIdentity, WorkerBootstrap, write_bootstrap


class ApiError(Exception):
    def __init__(self, status):
        self.status = status


class PodApi:
    """A deterministic API model; no claim of runtime or scheduler qualification."""

    def __init__(self):
        self.pod = None
        self.calls = []
        self.deleted = False
        self.create_error = None
        self.read_error = None
        self.patch_error = None
        self.on_create = None
        self.on_delete = None

    def create_namespaced_pod(self, *, namespace, body, **kwargs):
        self.manifest = copy.deepcopy(body)
        if self.create_error:
            raise self.create_error
        self.pod = NS(
            metadata=NS(**body["metadata"], uid="uid-1", resource_version="1", deletion_timestamp=None),
            spec=NS(node_name=None, containers=[NS(name="nvflare-task")], init_containers=[], ephemeral_containers=[]),
            status=NS(
                phase="Pending",
                reason=None,
                conditions=[],
                container_statuses=[],
                init_container_statuses=[],
                ephemeral_container_statuses=[],
            ),
        )
        self.calls.append("create")
        if self.on_create:
            self.on_create()
        return copy.deepcopy(self.pod)

    def read_namespaced_pod(self, **kwargs):
        self.calls.append("read")
        if self.read_error:
            raise self.read_error
        if self.pod is None:
            raise ApiError(404)
        return copy.deepcopy(self.pod)

    def delete_namespaced_pod(self, *, body, **kwargs):
        assert body["preconditions"]["uid"] == self.pod.metadata.uid
        assert body["gracePeriodSeconds"] > 0
        self.calls.append("delete")
        self.pod.metadata.deletion_timestamp = "deleting"
        self.pod.metadata.resource_version = str(int(self.pod.metadata.resource_version) + 1)
        if self.on_delete:
            self.on_delete()
        return NS(status="Success")

    def patch_namespaced_pod(self, *, body, **kwargs):
        self.calls.append("patch")
        if self.patch_error:
            raise self.patch_error
        assert body[:3] == [
            {"op": "test", "path": "/metadata/uid", "value": self.pod.metadata.uid},
            {"op": "test", "path": "/metadata/resourceVersion", "value": self.pod.metadata.resource_version},
            {"op": "test", "path": "/metadata/finalizers", "value": self.pod.metadata.finalizers},
        ]
        self.pod.metadata.finalizers = body[3]["value"]
        if not self.pod.metadata.finalizers and self.pod.metadata.deletion_timestamp:
            self.pod = None
            self.deleted = True

    def read_namespaced_pod_log(self, *, limit_bytes, **kwargs):
        assert limit_bytes == 65536
        self.calls.append("logs")
        return "task output\n"

    def running(self):
        self.pod.spec.node_name = "node-1"
        self.pod.status.phase = "Running"
        self.pod.status.container_statuses = [
            NS(name="nvflare-task", state=NS(running=NS(), terminated=None, waiting=None))
        ]

    def finish(self, *, exit_code=0, reason=None, sandbox_down=True):
        self.pod.spec.node_name = "node-1"
        self.pod.status.phase = "Failed" if reason or exit_code else "Succeeded"
        self.pod.status.reason = reason
        self.pod.status.conditions = [NS(type="PodReadyToStartContainers", status="False" if sandbox_down else "True")]
        self.pod.status.container_statuses = [
            NS(name="nvflare-task", state=NS(terminated=NS(exit_code=exit_code), running=None, waiting=None))
        ]


def _bind_to_cj(launcher, request, meta=None):
    context = Mock()
    context.get_job_id.return_value = request.job_id
    context.get_identity_name.return_value = request.site_name
    context.get_prop.side_effect = lambda key, default=None: meta if key == FLContextKey.JOB_META else default
    launcher.handle_event(EventType.START_RUN, context)
    return context


def make_task_setup(tmp_path):
    identity = TaskAttemptIdentity("job", "site-1", "task", "compute", "attempt")
    store = FileTaskArtifactStore(str(tmp_path / "artifacts"))
    store.create_attempt(identity)
    bootstrap = WorkerBootstrap(identity, store.root_dir, str(tmp_path), {"path": "unused.Compute"})
    path = store.bootstrap_path(identity)
    write_bootstrap(path, bootstrap)
    request = TaskLaunchRequest(
        identity.job_id,
        identity.site_name,
        identity.task_id,
        identity.attempt_id,
        ("/cj/python", "-m", WORKER_MODULE, "--bootstrap", path, "--parent_pid", "456"),
        cwd=str(tmp_path),
    )
    launcher = K8sTaskBackend(
        startup_secret_name="federation-a.site-1",
        namespace="site",
        image="site/image@sha256:trusted",
        workspace_root=str(tmp_path),
        workspace_pvc="existing-pvc",
        worker_service_account="task-no-api",
        cleanup_timeout=5,
        poll_interval=0.001,
        cpu_cores=1.5,
        memory_mb=512,
        gpu_count=1,
        environment={"LANG": "", "MY_SETTING": "explicit"},
    )
    _bind_to_cj(launcher, request)
    api = launcher.core_v1 = PodApi()
    return launcher, api, request, store, bootstrap


class LoopbackCell:
    def __init__(self, service):
        self.service = service
        self.origin = "site-parent.job"
        self.drop_reply = False
        self.decode_pass_through_topics = set()

    def send_request(self, *, request, **kwargs):
        request.set_header(MessageHeaderKey.ORIGIN, self.origin)
        reply = self.service._request(request)
        if self.drop_reply:
            raise TimeoutError("reply lost after CP accepted the request")
        return reply


def wire_client(monkeypatch):
    cp = NS(get_fqcn=lambda: "site-parent", register_request_cb=Mock())
    service = K8sTaskService(cp)
    monkeypatch.setenv(ENV_WORKSPACE_OWNER_FQCN, "site-parent")
    monkeypatch.setenv(ENV_WORKSPACE_TRANSFER_TOKEN, "job-token")
    return LoopbackCell(service), service
