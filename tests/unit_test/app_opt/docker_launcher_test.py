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

from unittest.mock import Mock

from nvflare.apis.fl_constant import FLContextKey, JobConstants, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import JobMetaKey
from nvflare.app_opt.job_launcher.docker_launcher import DockerJobHandle, DockerJobLauncher


class _DummyWorkspace:
    def get_app_custom_dir(self, job_id):
        return ""


class _DummyDockerLauncher(DockerJobLauncher):
    def get_command(self, job_meta, fl_ctx) -> (str, str):
        return "test-container", "python worker.py"


def _make_fl_ctx():
    fl_ctx = FLContext()
    fl_ctx.set_prop(FLContextKey.WORKSPACE_OBJECT, _DummyWorkspace(), private=True, sticky=False)
    fl_ctx.set_prop(ReservedKey.IDENTITY_NAME, "server", private=True, sticky=False)
    return fl_ctx


def _make_job_meta(project=""):
    job_meta = {
        JobConstants.JOB_ID: "job-1",
        JobMetaKey.DEPLOY_MAP.value: {
            "app": [
                {
                    JobConstants.SITES: ["server"],
                    JobConstants.JOB_IMAGE: "nvflare:test",
                }
            ]
        },
    }
    if project:
        job_meta[JobMetaKey.PROJECT.value] = project
    return job_meta


def test_launch_job_returns_none_when_workspace_env_missing(monkeypatch):
    launcher = _DummyDockerLauncher()
    fl_ctx = _make_fl_ctx()
    job_meta = _make_job_meta(project="cancer-research")
    docker_from_env = Mock()

    monkeypatch.delenv("NVFL_DOCKER_WORKSPACE", raising=False)
    monkeypatch.setattr("nvflare.app_opt.job_launcher.docker_launcher.docker.from_env", docker_from_env)

    handle = launcher.launch_job(job_meta, fl_ctx)

    assert handle is None
    docker_from_env.assert_not_called()


def test_launch_job_returns_none_when_project_workspace_missing(monkeypatch, tmp_path):
    launcher = _DummyDockerLauncher()
    fl_ctx = _make_fl_ctx()
    job_meta = _make_job_meta(project="cancer-research")
    docker_from_env = Mock()

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    monkeypatch.setenv("NVFL_DOCKER_WORKSPACE", str(workspace_root))
    monkeypatch.setattr("nvflare.app_opt.job_launcher.docker_launcher.docker.from_env", docker_from_env)

    handle = launcher.launch_job(job_meta, fl_ctx)

    assert handle is None
    docker_from_env.assert_not_called()


def test_launch_job_uses_project_workspace_when_present(monkeypatch, tmp_path):
    launcher = _DummyDockerLauncher()
    fl_ctx = _make_fl_ctx()
    job_meta = _make_job_meta(project="cancer-research")

    workspace_root = tmp_path / "workspace"
    project_workspace = workspace_root / "cancer-research"
    project_workspace.mkdir(parents=True)

    fake_container = Mock()
    fake_container.id = "container-id"
    fake_client = Mock()
    fake_client.containers.run.return_value = fake_container

    monkeypatch.setenv("NVFL_DOCKER_WORKSPACE", str(workspace_root))
    monkeypatch.setattr("nvflare.app_opt.job_launcher.docker_launcher.docker.from_env", Mock(return_value=fake_client))
    monkeypatch.setattr(DockerJobHandle, "enter_states", Mock(return_value=True))

    handle = launcher.launch_job(job_meta, fl_ctx)

    assert isinstance(handle, DockerJobHandle)
    fake_client.containers.run.assert_called_once()
    assert fake_client.containers.run.call_args.kwargs["volumes"] == {
        str(project_workspace): {
            "bind": launcher.mount_path,
            "mode": "rw",
        }
    }
