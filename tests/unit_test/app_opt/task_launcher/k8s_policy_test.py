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
from nvflare.app_opt.task_launcher.k8s_job_launcher import ClientK8sTaskJobLauncher
from nvflare.app_opt.task_launcher.k8s_launcher import K8sTaskLauncher
from nvflare.private.fed.client.task_execution_policy import TaskExecutionPolicy
from nvflare.private.fed.client.task_worker_executor import TaskWorkerExecutor
from nvflare.private.fed.task_worker.artifacts import FileTaskArtifactStore
from nvflare.private.fed.task_worker.protocol import read_bootstrap
from tests.unit_test.app_opt.task_launcher.k8s_launcher_test import PodApi


@pytest.fixture
def profile(tmp_path):
    args = {
        "config_file_path": None,
        "namespace": "site",
        "task_workspace_pvc": "existing-pvc",
        "task_workspace_pvc_root": "/cp/explicit-pvc-mount",
        "workspace_mount_path": str(tmp_path),
        "task_cj_service_account": "cj-launch-authority",
        "run_as_user": 1000,
        "run_as_group": 1000,
    }
    config = {
        "components": [
            {"path": "nvflare.app_opt.task_launcher.k8s_job_launcher.ClientK8sTaskJobLauncher", "args": args}
        ],
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


def test_matching_site_profile_validates_without_backend_access(profile):
    launcher, config = profile
    launcher.validate_site_config(config)
    assert launcher.core_v1 is None


@pytest.mark.parametrize("gpu_count", [0, 1])
def test_canonical_selection_through_cj_execution_routes_cpu_gpu_and_resends_without_compute(
    profile, tmp_path, isolated_fobs, gpu_count
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
    api = policy.launcher.core_v1 = PodApi()
    adapter = TaskWorkerExecutor({"path": "nvflare.app_common.np.np_trainer.NPTrainer"}, (), policy)

    def complete():
        bootstrap = read_bootstrap(api.manifest["spec"]["containers"][0]["args"][-1])
        FileTaskArtifactStore(bootstrap.artifact_root).commit_result(bootstrap.identity, Shareable({"gpu": gpu_count}))
        api.finish()

    api.on_create = complete
    adapter.handle_event(EventType.START_RUN, context)
    try:
        result = adapter.execute("train", Shareable(), context, Signal())
        assert result["gpu"] == gpu_count
        assert api.deleted
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


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "resident",
        "process",
        "multiple",
        "namespace",
        "pvc",
        "workspace",
        "uid",
        "gid",
        "accounts",
        "cp-mount",
        "environment",
    ],
)
def test_mixed_or_inconsistent_site_profiles_fail_before_backend_access(profile, fault):
    launcher, config = profile
    config = copy.deepcopy(config)
    cp = config["components"][0]
    if fault == "missing":
        config["components"] = []
    elif fault == "resident":
        cp["path"] = "nvflare.app_opt.job_launcher.k8s_launcher.ClientK8sJobLauncher"
    elif fault == "process":
        cp["path"] = "nvflare.app_common.job_launcher.client_process_launcher.ClientProcessJobLauncher"
    elif fault == "multiple":
        config["components"].append(copy.deepcopy(cp))
    elif fault == "namespace":
        cp["args"]["namespace"] = "other"
    elif fault == "pvc":
        cp["args"]["task_workspace_pvc"] = "other"
    elif fault == "workspace":
        cp["args"]["workspace_mount_path"] = "/other"
    elif fault in ("uid", "gid"):
        cp["args"]["run_as_user" if fault == "uid" else "run_as_group"] = 2000
    elif fault == "accounts":
        cp["args"]["task_cj_service_account"] = launcher.worker_service_account
    elif fault == "cp-mount":
        cp["args"].pop("task_workspace_pvc_root")
    elif fault == "environment":
        config["task_execution"]["environment_variables"] = ["PYTHONPATH"]
    with pytest.raises(ValueError):
        launcher.validate_site_config(config)
    assert launcher.core_v1 is None


def test_positive_whole_job_gpu_reservation_is_rejected_before_cj_or_task_launch(profile, tmp_path):
    launcher, config = profile
    meta = {"job_id": "job", "resource_spec": {"@default": {"num_of_gpus": 1}}}
    context = Mock()
    context.get_identity_name.return_value = "site-1"
    context.get_job_id.return_value = "job"
    context.get_prop.return_value = meta
    with pytest.raises(ValueError, match="whole-job CP GPU reservations"):
        launcher.handle_event(EventType.START_RUN, context)
    assert launcher._job_binding is None
    assert launcher.core_v1 is None

    path = tmp_path / "client.json"
    path.write_text(json.dumps({"execution_lifetime": "task"}))
    context.get_prop.return_value = SimpleNamespace(get_client_app_config_file_path=lambda _: str(path))
    cp_launcher = ClientK8sTaskJobLauncher(**config["components"][0]["args"])
    with pytest.raises(ValueError, match="whole-job CP GPU reservations"):
        cp_launcher.launch_job(meta, context)
    assert cp_launcher.core_v1 is None
