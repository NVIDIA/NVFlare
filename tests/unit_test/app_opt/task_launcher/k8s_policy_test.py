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

import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, ReservedKey, TaskResultReceipt
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.apis.workspace import Workspace
from nvflare.app_opt.job_launcher.k8s_launcher import ClientK8sJobLauncher
from nvflare.app_opt.task_launcher.k8s_launcher import K8sTaskBackend, K8sTaskLauncher
from nvflare.private.fed.client.task_execution_policy import TaskExecutionPolicy
from nvflare.private.fed.client.task_worker_executor import TaskWorkerExecutor
from nvflare.private.fed.task_worker.artifacts import FileTaskArtifactStore
from nvflare.private.fed.task_worker.protocol import read_bootstrap
from tests.k8s_task_test_utils import PodApi, wire_client


@pytest.fixture
def profile(tmp_path):
    args = {
        "config_file_path": None,
        "namespace": "site",
        "job_workspace_pvc": "existing-pvc",
        "job_workspace_pvc_root": "/cp/explicit-pvc-mount",
        "workspace_mount_path": str(tmp_path),
        "run_as_user": 1000,
        "run_as_group": 1000,
    }
    config = {
        "components": [{"path": "nvflare.app_opt.job_launcher.k8s_launcher.ClientK8sJobLauncher", "args": args}],
        "task_execution": {
            "launcher": {
                "path": "nvflare.app_opt.task_launcher.k8s_launcher.K8sTaskLauncher",
                "args": {
                    "namespace": "site",
                    "image": "site/worker",
                    "workspace_root": str(tmp_path),
                    "workspace_pvc": "existing-pvc",
                    "worker_service_account": "task-no-api",
                    "gpu_count": 2,
                },
            },
        },
    }
    launcher = K8sTaskLauncher("site", "site/worker", str(tmp_path), "existing-pvc", "task-no-api")
    return launcher, config


def test_cj_short_name_requests_its_unregistered_full_path(profile):
    launcher, config = profile
    component = config["components"][0]
    component.pop("path")
    component["name"] = "ClientK8sJobLauncher"
    with pytest.raises(ValueError, match="not registered by short name; specify path=.*ClientK8sJobLauncher"):
        launcher.validate_site_config(config)


def test_matching_site_profile_validates_without_backend_access(profile):
    launcher, config = profile
    launcher.validate_site_config(config)
    assert launcher.core_v1 is None


@pytest.mark.parametrize("gpu_count", [0, 1])
def test_canonical_selection_through_cj_execution_routes_cpu_gpu_and_resends_without_compute(
    profile, tmp_path, isolated_fobs, gpu_count, monkeypatch
):
    _, config = profile
    (tmp_path / "local").mkdir()
    (tmp_path / "startup").mkdir()
    (tmp_path / "local" / "resources.json").write_text(json.dumps(config))
    workspace = Workspace(str(tmp_path), site_name="site-1")
    policy = TaskExecutionPolicy.from_workspace(workspace)
    assert type(policy.launcher) is K8sTaskLauncher
    assert policy.worker_environment() == {}
    context = FLContext()

    cell, service = wire_client(monkeypatch)
    cell.decode_pass_through_topics = set()
    # This test models Pods/CP with an in-memory transport; isolate the actual
    # live metrics link, tested separately with real Cell and mTLS sessions.
    from nvflare.app_opt.job_launcher.k8s_launcher import ENV_CJ_POD_IP
    from nvflare.private.fed.task_worker import cell_session

    monkeypatch.setenv(ENV_CJ_POD_IP, "10.42.0.7")
    listen = Mock()
    session = Mock()
    session.bootstrap_fields.return_value = {
        "connect_url": "tcp://10.42.0.7:12345",
        "worker_fqcn": "site-parent.job.task_worker_test",
    }
    monkeypatch.setattr(cell_session, "listen", listen)
    monkeypatch.setattr(cell_session, "CellWorkerSession", Mock(return_value=session))
    context.set_prop(ReservedKey.ENGINE, SimpleNamespace(get_cell=lambda: cell), private=True, sticky=True)
    for key, value in (
        (ReservedKey.RUN_NUM, "job"),
        (ReservedKey.IDENTITY_NAME, "site-1"),
        (FLContextKey.WORKSPACE_OBJECT, workspace),
        (FLContextKey.TASK_ID, "task"),
        (FLContextKey.TASK_NAME, "train"),
        (FLContextKey.TASK_ATTEMPT_ID, "attempt"),
        (
            FLContextKey.JOB_META,
            {"launcher_spec": {"default": {"k8s": {"num_of_gpus": gpu_count, "cpu": "500m", "memory": "512Mi"}}}},
        ),
    ):
        context.set_prop(key, value, private=True, sticky=True)
    api = PodApi()
    backend = policy.launcher.create_backend(
        api,
        str(tmp_path),
        context.get_prop(FLContextKey.JOB_META),
        "job",
        "site-1",
        startup_secret_name="federation-a.site-1",
    )
    backend.poll_interval = 0.001
    owner = service.add_job(backend, "job", "site-1", "job-token")
    adapter = TaskWorkerExecutor({"path": "nvflare.app_common.np.np_trainer.NPTrainer"}, (), policy)

    def complete():
        argv = api.manifest["spec"]["containers"][0]["args"]
        bootstrap = read_bootstrap(argv[argv.index("--bootstrap") + 1])
        assert bootstrap.cell_session["connect_url"] == "tcp://10.42.0.7:12345"
        FileTaskArtifactStore(bootstrap.artifact_root).commit_result(bootstrap.identity, Shareable({"gpu": gpu_count}))
        api.finish()

    api.on_create = complete
    adapter.handle_event(EventType.START_RUN, context)
    listen.assert_called_once_with(cell, "10.42.0.7")
    try:
        result = adapter.execute("train", Shareable(), context, Signal())
        assert result["gpu"] == gpu_count
        assert api.deleted
        assert api.manifest["spec"]["volumes"][1]["secret"]["secretName"] == "federation-a.site-1"
        requests = api.manifest["spec"]["containers"][0]["resources"]["requests"]
        assert requests["cpu"] == "500m"
        assert requests["memory"] == "512Mi"
        assert requests.get("nvidia.com/gpu") == ("1" if gpu_count else None)
        assert adapter.execute("train", Shareable(), context, Signal()) == result
        assert api.calls.count("create") == 1
        context.set_prop(FLContextKey.TASK_RESULT, result, private=True, sticky=False)
        context.set_prop(FLContextKey.TASK_RESULT_RECEIPT, TaskResultReceipt.RECEIVED, private=True, sticky=False)
        adapter.handle_event(EventType.AFTER_SEND_TASK_RESULT, context)
    finally:
        adapter.handle_event(EventType.END_RUN, context)
        owner.release()


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "server",
        "process",
        "multiple",
        "namespace",
        "pvc",
        "workspace",
        "uid",
        "gid",
        "cp-mount",
        "environment",
    ],
)
def test_mixed_or_inconsistent_site_profiles_fail_before_backend_access(profile, tmp_path, monkeypatch, fault):
    launcher, config = profile
    config = copy.deepcopy(config)
    cp = config["components"][0]
    if fault == "missing":
        config["components"] = []
    elif fault == "server":
        cp["path"] = "nvflare.app_opt.job_launcher.k8s_launcher.ServerK8sJobLauncher"
    elif fault == "process":
        cp["path"] = "nvflare.app_common.job_launcher.client_process_launcher.ClientProcessJobLauncher"
    elif fault == "multiple":
        config["components"].append(copy.deepcopy(cp))
    elif fault == "namespace":
        cp["args"]["namespace"] = "other"
    elif fault == "pvc":
        cp["args"]["job_workspace_pvc"] = "other"
    elif fault == "workspace":
        cp["args"]["workspace_mount_path"] = "/other"
    elif fault in ("uid", "gid"):
        cp["args"]["run_as_user" if fault == "uid" else "run_as_group"] = 2000
    elif fault == "cp-mount":
        cp["args"].pop("job_workspace_pvc_root")
    elif fault == "environment":
        config["task_execution"]["environment_variables"] = ["PYTHONPATH"]
    (tmp_path / "local").mkdir()
    (tmp_path / "startup").mkdir()
    (tmp_path / "local" / "resources.json").write_text(json.dumps(config))
    api = Mock(side_effect=AssertionError("invalid site profile must fail before Kubernetes access"))
    monkeypatch.setattr(K8sTaskBackend, "_api", api)
    with pytest.raises(ValueError):
        TaskExecutionPolicy.from_workspace(Workspace(str(tmp_path), site_name="site-1"))
    api.assert_not_called()


def test_kubernetes_site_cannot_default_to_a_local_task_launcher(profile, tmp_path):
    _, config = profile
    config["task_execution"] = {}
    (tmp_path / "local").mkdir()
    (tmp_path / "startup").mkdir()
    (tmp_path / "local" / "resources.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="explicit site Kubernetes or Slurm"):
        TaskExecutionPolicy.from_workspace(Workspace(str(tmp_path), site_name="site-1"))


def test_positive_whole_job_gpu_reservation_is_rejected_before_cj_or_task_launch(profile, tmp_path):
    launcher, config = profile
    meta = {"job_id": "job", "resource_spec": {"@default": {"num_of_gpus": 1}}}
    context = Mock()
    context.get_identity_name.return_value = "site-1"
    context.get_job_id.return_value = "job"
    context.get_prop.return_value = meta
    with pytest.raises(ValueError, match="portable GPUs reserve the whole job"):
        launcher.handle_event(EventType.START_RUN, context)
    assert launcher._job_binding is None
    assert launcher.core_v1 is None

    path = tmp_path / "client.json"
    path.write_text(json.dumps({"execution_lifetime": "task"}))
    context.get_prop.return_value = SimpleNamespace(get_client_app_config_file_path=lambda _: str(path))
    cp_launcher = ClientK8sJobLauncher(**config["components"][0]["args"])
    with pytest.raises(ValueError, match="portable GPUs reserve the whole job"):
        cp_launcher.launch_job(meta, context)
    assert cp_launcher.core_v1 is None


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "10000"])
def test_attempt_history_limit_requires_a_positive_integer(tmp_path, limit):
    with pytest.raises(ValueError, match="max_attempts must be a positive integer"):
        K8sTaskLauncher("site", "site/worker", str(tmp_path), "pvc", "no-api", max_attempts=limit)


def test_cp_backend_keeps_the_site_attempt_history_limit(profile):
    launcher, _ = profile
    launcher.max_attempts = 123
    backend = launcher.create_backend(Mock(), "/cp/jobs/job", {}, "job", "site-1", startup_secret_name="site-startup")
    assert backend.max_attempts == 123


@pytest.mark.parametrize("pod_ip", ["10.42.0.7", "192.168.1.10"])
def test_worker_listener_uses_cj_downward_api_address(profile, monkeypatch, pod_ip):
    from nvflare.app_opt.job_launcher.k8s_launcher import ENV_CJ_POD_IP

    launcher, _ = profile
    monkeypatch.setenv(ENV_CJ_POD_IP, pod_ip)
    assert launcher.get_worker_listener_host() == pod_ip
    assert launcher.core_v1 is None


@pytest.mark.parametrize(
    "pod_ip",
    [None, "", "localhost", "127.0.0.1", "0.0.0.0", "224.0.0.1", "169.254.1.1", "::1", "fd00::1", "10.42.0.7:1234"],
)
def test_worker_listener_rejects_unusable_cj_addresses(profile, monkeypatch, pod_ip):
    from nvflare.app_opt.job_launcher.k8s_launcher import ENV_CJ_POD_IP

    launcher, _ = profile
    if pod_ip is None:
        monkeypatch.delenv(ENV_CJ_POD_IP, raising=False)
    else:
        monkeypatch.setenv(ENV_CJ_POD_IP, pod_ip)
    with pytest.raises(ValueError, match=ENV_CJ_POD_IP):
        launcher.get_worker_listener_host()
