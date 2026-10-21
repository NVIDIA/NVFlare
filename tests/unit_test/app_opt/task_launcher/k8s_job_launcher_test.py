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
import os
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import pytest

from nvflare.apis.fl_constant import FLContextKey, JobConstants, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_launcher_spec import JobProcessArgs
from nvflare.apis.workspace import Workspace
from nvflare.app_opt.task_launcher.k8s_job_launcher import (
    ClientK8sTaskJobLauncher,
    task_security_context,
    task_workspace_subpath,
)

JOB = "4df684c0-d73d-4e7e-83d6-e4e8d0171532"
SITE = "site-1"


def _profile(pvc_root):
    return ClientK8sTaskJobLauncher(
        config_file_path=None,
        namespace="site",
        task_workspace_pvc="existing-pvc",
        task_workspace_pvc_root=str(pvc_root),
        task_cj_service_account="cj-task-authority",
        run_as_user=max(os.getuid(), 1),
        run_as_group=max(os.getgid(), 1),
    )


@pytest.mark.parametrize("lifetime", ["job", "task"])
def test_cp_submission_preserves_resident_or_selects_cpu_durable_cj(tmp_path, lifetime):
    (tmp_path / "startup").mkdir()
    (tmp_path / "local").mkdir()
    config = tmp_path / JOB / f"app_{SITE}" / "config"
    config.mkdir(parents=True)
    (config / "config_fed_client.json").write_text(json.dumps({"execution_lifetime": lifetime}))
    workspace = Workspace(str(tmp_path), site_name=SITE)
    fl_ctx = FLContext()
    fl_ctx.set_prop(ReservedKey.IDENTITY_NAME, SITE, private=False, sticky=True)
    fl_ctx.set_prop(ReservedKey.ENGINE, NS(cell=Mock()), private=True, sticky=False)
    fl_ctx.set_prop(FLContextKey.SECURE_MODE, False, private=True, sticky=False)
    fl_ctx.set_prop(FLContextKey.WORKSPACE_OBJECT, workspace, private=True, sticky=False)
    fl_ctx.set_prop(FLContextKey.ARGS, NS(workspace=str(tmp_path), set=[]), private=True, sticky=False)
    fl_ctx.set_prop(
        FLContextKey.JOB_PROCESS_ARGS,
        {
            JobProcessArgs.EXE_MODULE: ("-m", "nvflare.private.fed.app.client.worker_process"),
            JobProcessArgs.WORKSPACE: ("-w", "/var/tmp/nvflare/workspace"),
            JobProcessArgs.JOB_ID: ("-n", JOB),
        },
        private=True,
        sticky=False,
    )
    pvc_root = tmp_path / "mounted-pvc"
    pvc_root.mkdir()
    launcher = _profile(pvc_root)
    launcher._prepare_workspace_subpath = Mock(return_value=task_workspace_subpath(JOB, SITE))
    api = launcher.core_v1 = Mock()
    api.read_namespaced_pod.return_value = NS(status=NS(phase="Running"))
    launcher._ensure_startup_secret = Mock(return_value="existing-startup-secret")
    launcher._ensure_job_credential_secret = Mock(return_value="cj-credentials")
    launcher._own_credential_secret = Mock()
    meta = {JobConstants.JOB_ID: JOB, "launcher_spec": {SITE: {"k8s": {"image": "site/image", "num_of_gpus": 2}}}}
    with patch("nvflare.app_opt.job_launcher.k8s_launcher.WorkspaceTransferManager") as transfer:
        transfer.get_or_create.return_value.owner_fqcn = "site-parent"
        transfer.get_or_create.return_value.add_job.return_value = "transfer-token"
        handle = launcher.launch_job(meta, fl_ctx)
    manifest = api.create_namespaced_pod.call_args.kwargs["body"]
    assert handle.get_manifest() == manifest
    spec = manifest["spec"]
    container = spec["containers"][0]
    workspace_volume = next(volume for volume in spec["volumes"] if volume["name"] == "workspace-job")
    workspace_mount = next(mount for mount in container["volumeMounts"] if mount["name"] == "workspace-job")
    if lifetime == "job":
        assert "emptyDir" in workspace_volume
        assert container["resources"]["requests"]["nvidia.com/gpu"] == "2"
        assert "serviceAccountName" not in spec
    else:
        assert workspace_volume == {"name": "workspace-job", "persistentVolumeClaim": {"claimName": "existing-pvc"}}
        assert workspace_mount["subPath"] == task_workspace_subpath(JOB, SITE)
        assert "nvidia.com/gpu" not in container["resources"]["requests"]
        assert "nvidia.com/gpu" not in container["resources"]["limits"]
        assert spec["serviceAccountName"] == "cj-task-authority"
        assert spec["automountServiceAccountToken"] is True
        assert spec["securityContext"] == task_security_context(max(os.getuid(), 1), max(os.getgid(), 1))
        assert container["securityContext"]["allowPrivilegeEscalation"] is False
        environment = {item["name"]: item.get("value") for item in container["env"]}
        assert environment["CUDA_VISIBLE_DEVICES"] == ""
        assert environment["NVIDIA_VISIBLE_DEVICES"] == "none"
    # Existing credential ownership and workspace-transfer mechanisms are retained.
    launcher._own_credential_secret.assert_called_once()
    transfer.get_or_create.return_value.add_job.assert_called_once_with(JOB, str(tmp_path))
    if lifetime == "task":
        launcher._prepare_workspace_subpath.assert_called_once_with(JOB, SITE)
    else:
        launcher._prepare_workspace_subpath.assert_not_called()


@pytest.mark.parametrize("fault", ["sidecar", "host", "missing-volume", "wrong-mount", "unknown-lifetime"])
def test_unsupported_cj_profiles_fail_before_submission(tmp_path, fault):
    path = tmp_path / "client.json"
    path.write_text(json.dumps({"execution_lifetime": "other" if fault == "unknown-lifetime" else "task"}))
    workspace = NS(get_client_app_config_file_path=lambda _: str(path))
    manifest = {
        "spec": {
            "volumes": [{"name": "workspace-job", "emptyDir": {}}],
            "containers": [
                {"name": "cj", "volumeMounts": [{"name": "workspace-job", "mountPath": "/var/tmp/nvflare/workspace"}]}
            ],
        }
    }
    if fault == "sidecar":
        manifest["spec"]["containers"].append({"name": "sidecar"})
    elif fault == "host":
        manifest["spec"]["hostPID"] = True
    elif fault == "missing-volume":
        manifest["spec"]["volumes"] = []
    elif fault == "wrong-mount":
        manifest["spec"]["containers"][0]["volumeMounts"][0]["mountPath"] = "/other"
    with pytest.raises(ValueError):
        _profile(tmp_path).prepare_job_manifest(copy.deepcopy(manifest), JOB, SITE, workspace)


def test_subpaths_are_site_and_job_scoped_without_raw_path_components():
    paths = {task_workspace_subpath(job, site) for job in (JOB, "../other") for site in (SITE, "site/other")}
    assert len(paths) == 4
    assert all(path.startswith("nvflare-tasks/") and ".." not in path for path in paths)


@pytest.mark.parametrize("uid,gid", [(0, 1000), (1000, 0), (True, 1000), (1000, "1000")])
def test_profile_requires_numeric_nonroot_filesystem_identity(uid, gid):
    with pytest.raises(ValueError, match="explicit positive numeric ID"):
        task_security_context(uid, gid)


def test_cp_creates_fresh_owned_subpath_without_reusing_retained_workspace(tmp_path):
    launcher = _profile(tmp_path)
    if os.getuid() == 0:
        with pytest.raises(ValueError, match="same nonroot numeric UID"):
            launcher._prepare_workspace_subpath(JOB, SITE)
        return
    subpath = launcher._prepare_workspace_subpath(JOB, SITE)
    directory = tmp_path / subpath
    assert directory.is_dir()
    assert directory.stat().st_uid == os.getuid()
    (directory / "retained-result").write_bytes(b"do not discard")
    with pytest.raises(FileExistsError):
        launcher._prepare_workspace_subpath(JOB, SITE)
    assert (directory / "retained-result").read_bytes() == b"do not discard"


def test_cp_rejects_unmounted_pvc_or_replaced_parent(tmp_path):
    launcher = _profile(tmp_path / "unmounted")
    with pytest.raises(ValueError, match="already be mounted"):
        launcher._prepare_workspace_subpath(JOB, SITE)
    (tmp_path / "outside").mkdir()
    (tmp_path / "nvflare-tasks").symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="real directory"):
        _profile(tmp_path)._prepare_workspace_subpath(JOB, SITE)
