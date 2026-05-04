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

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Stub out the 'docker' package so the module can be imported without Docker
# ---------------------------------------------------------------------------
_docker_mock = ModuleType("docker")
_docker_errors = ModuleType("docker.errors")
_docker_types = ModuleType("docker.types")


class _NotFound(Exception):
    pass


class _APIError(Exception):
    pass


class _ImageNotFound(_NotFound):
    pass


_docker_errors.NotFound = _NotFound
_docker_errors.APIError = _APIError
_docker_errors.ImageNotFound = _ImageNotFound
_docker_mock.errors = _docker_errors
_docker_mock.types = _docker_types
_docker_types.DeviceRequest = MagicMock


class _Mount(dict):
    def __init__(self, target, source, type="volume", read_only=False, **kwargs):
        super().__init__(Target=target, Source=source, Type=type, ReadOnly=read_only, **kwargs)


_docker_types.Mount = _Mount

for _mod_name, _mod_obj in [
    ("docker", _docker_mock),
    ("docker.errors", _docker_errors),
    ("docker.types", _docker_types),
]:
    sys.modules[_mod_name] = _mod_obj

# Patch docker.from_env and docker.errors at import time so DockerJobLauncher.__init__
# doesn't actually try to connect to the Docker daemon.
_docker_mock.from_env = MagicMock

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, JobConstants
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import JobMetaKey
from nvflare.apis.job_launcher_spec import JobProcessArgs, JobReturnCode
from nvflare.app_opt.job_launcher.docker_launcher import (
    ClientDockerJobLauncher,
    DockerJobHandle,
    DockerJobLauncher,
    ServerDockerJobLauncher,
    _exit_code_to_return_code,
    _safe_workspace_child_path,
    _sanitize_container_name,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_docker_client():
    dc = MagicMock()
    dc.ping.return_value = True
    dc.networks.get.return_value = MagicMock()
    return dc


def _make_launcher(cls=None, workspace="/ws", **kwargs):
    """Create a launcher with a pre-wired mock Docker client (no real Docker needed)."""
    if cls is None:
        cls = ClientDockerJobLauncher

    dc = _make_docker_client()
    with patch("docker.from_env", return_value=dc):
        launcher = cls.__new__(cls)
        DockerJobLauncher.__init__(launcher, workspace=workspace, **kwargs)
    launcher._docker_client = dc
    return launcher


def _make_handle(container_id="cid-1", container_name="test-container", docker_client=None, **kwargs):
    if docker_client is None:
        docker_client = _make_docker_client()
    return DockerJobHandle(
        container_id=container_id, container_name=container_name, docker_client=docker_client, **kwargs
    )


def _make_container(status="running", exit_code=0):
    c = MagicMock()
    c.status = status
    c.attrs = {"State": {"ExitCode": exit_code}}
    return c


def _mounts_by_target(mounts):
    return {m["Target"]: m for m in mounts}


# ---------------------------------------------------------------------------
# _sanitize_container_name
# ---------------------------------------------------------------------------


class TestSanitizeContainerName:
    def test_lowercase(self):
        assert _sanitize_container_name("MyContainer") == "mycontainer"

    def test_replaces_invalid_chars(self):
        assert _sanitize_container_name("foo/bar:baz") == "foo-bar-baz"

    def test_strips_leading_hyphens(self):
        assert _sanitize_container_name("-foo") == "foo"

    def test_fallback_on_empty(self):
        assert _sanitize_container_name("") == "nvflare-job"

    def test_allows_dots_underscores(self):
        assert _sanitize_container_name("foo.bar_baz") == "foo.bar_baz"


# ---------------------------------------------------------------------------
# _exit_code_to_return_code
# ---------------------------------------------------------------------------


class TestExitCodeToReturnCode:
    def test_zero_is_success(self):
        assert _exit_code_to_return_code(0) == JobReturnCode.SUCCESS

    def test_aborted_code_is_aborted(self):
        assert _exit_code_to_return_code(JobReturnCode.ABORTED) == JobReturnCode.ABORTED

    def test_nonzero_is_execution_error(self):
        assert _exit_code_to_return_code(1) == JobReturnCode.EXECUTION_ERROR
        assert _exit_code_to_return_code(127) == JobReturnCode.EXECUTION_ERROR


# ---------------------------------------------------------------------------
# _safe_workspace_child_path
# ---------------------------------------------------------------------------


class TestSafeWorkspaceChildPath:
    def test_returns_child_under_workspace(self):
        assert _safe_workspace_child_path("/workspace", "job-1") == "/workspace/job-1"

    def test_allows_reserved_workspace_name_when_requested(self):
        assert _safe_workspace_child_path("/workspace", "startup", allow_reserved=True) == "/workspace/startup"
        assert _safe_workspace_child_path("/workspace", "local", allow_reserved=True) == "/workspace/local"

    def test_rejects_path_escape(self):
        with pytest.raises(RuntimeError, match="single workspace child"):
            _safe_workspace_child_path("/workspace", "../other")

    def test_rejects_nested_child(self):
        with pytest.raises(RuntimeError, match="single workspace child"):
            _safe_workspace_child_path("/workspace", "job-1/../job-2")

    def test_rejects_reserved_workspace_name(self):
        with pytest.raises(RuntimeError, match="reserved workspace name"):
            _safe_workspace_child_path("/workspace", "local")

    def test_rejects_child_symlink(self, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        target = workspace / "job-2"
        target.mkdir()
        child = workspace / "job-1"
        try:
            child.symlink_to(target, target_is_directory=True)
        except (NotImplementedError, OSError):
            pytest.skip("symlinks are not supported on this filesystem")

        with pytest.raises(RuntimeError, match="must not be a symlink"):
            _safe_workspace_child_path(str(workspace), "job-1")


# ---------------------------------------------------------------------------
# DockerJobHandle — terminal_state pattern
# ---------------------------------------------------------------------------


class TestDockerJobHandleTerminalState:
    def test_poll_returns_unknown_while_running(self):
        dc = _make_docker_client()
        dc.containers.get.return_value = _make_container("running")
        h = _make_handle(docker_client=dc)
        assert h.poll() == JobReturnCode.UNKNOWN

    def test_poll_returns_terminal_state_on_exited_success(self):
        dc = _make_docker_client()
        dc.containers.get.return_value = _make_container("exited", exit_code=0)
        h = _make_handle(docker_client=dc)
        rc = h.poll()
        assert rc == JobReturnCode.SUCCESS
        assert h.terminal_state == JobReturnCode.SUCCESS

    def test_poll_returns_execution_error_on_nonzero_exit(self):
        dc = _make_docker_client()
        dc.containers.get.return_value = _make_container("exited", exit_code=1)
        h = _make_handle(docker_client=dc)
        rc = h.poll()
        assert rc == JobReturnCode.EXECUTION_ERROR

    def test_poll_returns_aborted_on_dead(self):
        dc = _make_docker_client()
        dc.containers.get.return_value = _make_container("dead")
        h = _make_handle(docker_client=dc)
        rc = h.poll()
        assert rc == JobReturnCode.ABORTED

    def test_poll_does_not_query_docker_after_terminal(self):
        """Once terminal_state is set, poll() must return immediately without Docker calls."""
        dc = _make_docker_client()
        h = _make_handle(docker_client=dc)
        h.terminal_state = JobReturnCode.SUCCESS
        rc = h.poll()
        assert rc == JobReturnCode.SUCCESS
        dc.containers.get.assert_not_called()

    def test_poll_container_not_found_sets_aborted(self):
        dc = _make_docker_client()
        dc.containers.get.side_effect = _NotFound()
        h = _make_handle(docker_client=dc)
        # first call sets terminal_state via _get_container returning None
        rc = h.poll()
        assert h.terminal_state == JobReturnCode.ABORTED

    def test_terminal_state_set_once_never_cleared(self):
        dc = _make_docker_client()
        # First call: exited successfully
        dc.containers.get.return_value = _make_container("exited", exit_code=0)
        h = _make_handle(docker_client=dc)
        h.poll()
        assert h.terminal_state == JobReturnCode.SUCCESS
        # Change mock to return dead — terminal_state must not change
        dc.containers.get.return_value = _make_container("dead")
        rc = h.poll()
        assert rc == JobReturnCode.SUCCESS
        assert h.terminal_state == JobReturnCode.SUCCESS

# ---------------------------------------------------------------------------
# DockerJobHandle — terminate
# ---------------------------------------------------------------------------


class TestDockerJobHandleTerminate:
    def test_terminate_sets_terminal_state(self):
        dc = _make_docker_client()
        h = _make_handle(docker_client=dc)
        h.terminate()
        assert h.terminal_state == JobReturnCode.ABORTED

    def test_terminate_sets_terminal_state_even_on_api_error(self):
        dc = _make_docker_client()
        dc.containers.get.side_effect = _APIError("oops")
        h = _make_handle(docker_client=dc)
        h.terminate()
        assert h.terminal_state == JobReturnCode.ABORTED

    def test_terminate_container_not_found_still_sets_aborted(self):
        dc = _make_docker_client()
        dc.containers.get.side_effect = _NotFound()
        h = _make_handle(docker_client=dc)
        h.terminate()
        assert h.terminal_state == JobReturnCode.ABORTED

    def test_terminate_does_not_overwrite_existing_terminal_state(self):
        """If already terminated (e.g. SUCCESS), a second terminate() must not change it."""
        dc = _make_docker_client()
        dc.containers.get.side_effect = _NotFound()
        h = _make_handle(docker_client=dc)
        h.terminal_state = JobReturnCode.SUCCESS
        h.terminate()
        assert h.terminal_state == JobReturnCode.SUCCESS


# ---------------------------------------------------------------------------
# DockerJobHandle — wait
# ---------------------------------------------------------------------------


class TestDockerJobHandleWait:
    def test_wait_returns_immediately_if_terminal(self):
        h = _make_handle()
        h.terminal_state = JobReturnCode.SUCCESS
        h.wait()  # must not block

    def test_wait_sets_terminal_on_exited(self):
        dc = _make_docker_client()
        dc.containers.get.return_value = _make_container("exited", exit_code=0)
        h = _make_handle(docker_client=dc)
        h.wait()
        assert h.terminal_state == JobReturnCode.SUCCESS


# ---------------------------------------------------------------------------
# DockerJobHandle — enter_states / stuck detection
# ---------------------------------------------------------------------------


class TestDockerJobHandleEnterStates:
    def test_returns_true_when_state_reached(self):
        dc = _make_docker_client()
        dc.containers.get.return_value = _make_container("running")
        h = _make_handle(docker_client=dc)
        result = h.enter_states(["running"])
        assert result is True

    def test_returns_false_when_terminal_before_state(self):
        dc = _make_docker_client()
        dc.containers.get.return_value = _make_container("exited", exit_code=1)
        h = _make_handle(docker_client=dc)
        result = h.enter_states(["running"])
        assert result is False
        assert h.terminal_state == JobReturnCode.EXECUTION_ERROR

    def test_timeout_in_created_terminates_and_returns_false(self):
        dc = _make_docker_client()
        dc.containers.get.return_value = _make_container("created")
        h = _make_handle(docker_client=dc, timeout=1)
        result = h.enter_states(["running"])
        assert result is False
        assert h.terminal_state == JobReturnCode.ABORTED

    def test_returns_false_if_already_terminal(self):
        h = _make_handle()
        h.terminal_state = JobReturnCode.ABORTED
        result = h.enter_states(["running"])
        assert result is False


# ---------------------------------------------------------------------------
# DockerJobLauncher — init validation
# ---------------------------------------------------------------------------


class TestDockerJobLauncherInit:
    def test_raises_if_workspace_empty_and_no_env(self):
        """workspace is validated lazily in launch_job, not __init__."""
        with patch.dict("os.environ", {}, clear=True):
            launcher = ClientDockerJobLauncher.__new__(ClientDockerJobLauncher)
            DockerJobLauncher.__init__(launcher, workspace=None)
            launcher._docker_client = _make_docker_client()
            fl_ctx, _ = _make_fl_ctx()
            with pytest.raises(ValueError, match="workspace"):
                launcher.launch_job(_make_job_meta(), fl_ctx)

    def test_workspace_read_from_env_if_not_provided(self):
        dc = _make_docker_client()
        with patch("docker.from_env", return_value=dc):
            with patch.dict("os.environ", {"NVFL_DOCKER_WORKSPACE": "/host/ws"}):
                launcher = ClientDockerJobLauncher.__new__(ClientDockerJobLauncher)
                DockerJobLauncher.__init__(launcher, workspace=None)
        assert launcher.workspace == "/host/ws"

    def test_raises_if_default_job_container_kwargs_contains_reserved_key(self):
        for reserved in ("volumes", "mounts", "network", "environment", "command", "name", "detach"):
            with pytest.raises(ValueError, match="reserved"):
                _make_launcher(default_job_container_kwargs={reserved: "anything"})

    def test_default_job_container_kwargs_non_reserved_accepted(self):
        launcher = _make_launcher(default_job_container_kwargs={"shm_size": "2g", "ipc_mode": "host"})
        assert launcher.default_job_container_kwargs == {"shm_size": "2g", "ipc_mode": "host"}

    def test_default_job_env_accepted(self):
        launcher = _make_launcher(default_job_env={"NCCL_P2P_DISABLE": "1"})
        assert launcher.default_job_env == {"NCCL_P2P_DISABLE": "1"}

    def test_raises_if_docker_not_reachable(self):
        """Docker connectivity is validated lazily in _get_docker_client."""
        dc = _make_docker_client()
        dc.ping.side_effect = Exception("connection refused")
        launcher = ClientDockerJobLauncher.__new__(ClientDockerJobLauncher)
        DockerJobLauncher.__init__(launcher, workspace="/ws")
        with patch("docker.from_env", return_value=dc):
            with pytest.raises(RuntimeError, match="cannot connect"):
                launcher._get_docker_client()

    def test_raises_if_network_not_found(self):
        """Network existence is validated lazily in _get_docker_client."""
        dc = _make_docker_client()
        dc.networks.get.side_effect = _NotFound()
        launcher = ClientDockerJobLauncher.__new__(ClientDockerJobLauncher)
        DockerJobLauncher.__init__(launcher, workspace="/ws")
        with patch("docker.from_env", return_value=dc):
            with pytest.raises(RuntimeError, match="does not exist"):
                launcher._get_docker_client()


# ---------------------------------------------------------------------------
# DockerJobLauncher — launch_job
# ---------------------------------------------------------------------------


def _make_fl_ctx(
    job_id="job-1",
    exe_module="nvflare.private.fed.app.client.worker_process",
    parent_url="localhost:8002",
    identity_name="site-1",
    workspace_path="/ws",
    set_list=None,
    num_of_gpus=None,
):
    fl_ctx = MagicMock(spec=FLContext)
    fl_ctx.get_identity_name.return_value = identity_name

    # JOB_PROCESS_ARGS
    job_args = {
        JobProcessArgs.EXE_MODULE: ("", exe_module),
        JobProcessArgs.PARENT_URL: ("-u", parent_url),
        JobProcessArgs.WORKSPACE: ("-w", workspace_path),
        JobProcessArgs.STARTUP_DIR: ("-s", workspace_path + "/startup"),
    }
    fl_ctx.get_prop.side_effect = lambda key, *a, **kw: {
        FLContextKey.JOB_PROCESS_ARGS: job_args,
        FLContextKey.WORKSPACE_OBJECT: None,
        FLContextKey.ARGS: None,
    }.get(key)

    return fl_ctx, job_args


def _make_job_meta(job_id="job-1", site_name="site-1", docker_spec=None, resource_spec=None, study=None):
    meta = {
        JobConstants.JOB_ID: job_id,
        "deploy_map": {"app": [site_name]},
    }
    if study is not None:
        meta[JobMetaKey.STUDY.value] = study
    if resource_spec is not None:
        meta[JobMetaKey.RESOURCE_SPEC.value] = resource_spec
    else:
        spec = {"image": "nvflare/nvflare:test"}
        if docker_spec is not None:
            spec.update(docker_spec)
        meta["launcher_spec"] = {site_name: {"docker": spec}}
    return meta


class TestDockerJobLauncherLaunchJob:
    def test_launch_raises_if_image_missing_for_configured_site(self):
        launcher = _make_launcher()
        fl_ctx, _ = _make_fl_ctx(identity_name="site-1")
        job_meta = _make_job_meta(site_name="site-1", resource_spec={"site-1": {"docker": {}}})

        with pytest.raises(RuntimeError, match="no job image was specified"):
            launcher.launch_job(job_meta, fl_ctx)

    def test_launch_returns_handle(self):
        launcher = _make_launcher()
        dc = launcher._docker_client
        container = MagicMock()
        container.id = "abc123"
        container.status = "running"
        dc.containers.run.return_value = container
        dc.containers.get.return_value = _make_container("running")

        fl_ctx, _ = _make_fl_ctx()
        handle = launcher.launch_job(_make_job_meta(), fl_ctx)

        assert handle is not None
        assert isinstance(handle, DockerJobHandle)

    def test_launch_overrides_parent_url(self):
        """Launcher must derive parent_url from site name + port; localhost must not reach job container."""
        launcher = _make_launcher()
        dc = launcher._docker_client
        container = MagicMock()
        container.id = "abc123"
        container.status = "running"
        dc.containers.run.return_value = container
        dc.containers.get.return_value = _make_container("running")

        fl_ctx, job_args = _make_fl_ctx(identity_name="site-1", parent_url="tcp://localhost:8004")
        launcher.launch_job(_make_job_meta(), fl_ctx)

        call_kwargs = dc.containers.run.call_args
        command = call_kwargs[1]["command"] if call_kwargs[1] else call_kwargs[0][1]
        command_str = " ".join(str(a) for a in command)
        # Must replace localhost with container name (site name)
        assert "localhost" not in command_str
        assert "tcp://site-1:8004" in command_str

    def test_launch_raises_on_missing_job_id(self):
        launcher = _make_launcher()
        fl_ctx, _ = _make_fl_ctx()
        with pytest.raises(RuntimeError, match="JOB_ID"):
            launcher.launch_job({}, fl_ctx)

    def test_launch_raises_on_missing_job_process_args(self):
        launcher = _make_launcher()
        fl_ctx = MagicMock(spec=FLContext)
        fl_ctx.get_prop.return_value = None
        fl_ctx.get_identity_name.return_value = "site-1"
        with pytest.raises(RuntimeError):
            launcher.launch_job(_make_job_meta(), fl_ctx)

    def test_launch_workspace_bind_mounted(self):
        launcher = _make_launcher(workspace="/host/workspace")
        dc = launcher._docker_client
        container = MagicMock()
        container.id = "abc123"
        dc.containers.run.return_value = container
        dc.containers.get.return_value = _make_container("running")

        fl_ctx, _ = _make_fl_ctx()
        launcher.launch_job(_make_job_meta(), fl_ctx)

        call_kwargs = dc.containers.run.call_args[1]
        mounts = call_kwargs["mounts"]
        assert mounts[0]["Target"] == "/var/tmp/nvflare/workspace"
        assert mounts[1]["Target"] == "/var/tmp/nvflare/workspace/startup"
        assert mounts[2]["Target"] == "/var/tmp/nvflare/workspace/local"
        assert mounts[3]["Target"] == "/var/tmp/nvflare/workspace/job-1"

        mounts_by_target = _mounts_by_target(call_kwargs["mounts"])
        assert mounts_by_target["/var/tmp/nvflare/workspace"] == {
            "Target": "/var/tmp/nvflare/workspace",
            "Source": None,
            "Type": "tmpfs",
            "ReadOnly": False,
            "tmpfs_mode": 0o555,
        }
        assert mounts_by_target["/var/tmp/nvflare/workspace/startup"] == {
            "Target": "/var/tmp/nvflare/workspace/startup",
            "Source": "/host/workspace/startup",
            "Type": "bind",
            "ReadOnly": True,
        }
        assert mounts_by_target["/var/tmp/nvflare/workspace/local"] == {
            "Target": "/var/tmp/nvflare/workspace/local",
            "Source": "/host/workspace/local",
            "Type": "bind",
            "ReadOnly": True,
        }
        assert mounts_by_target["/var/tmp/nvflare/workspace/job-1"] == {
            "Target": "/var/tmp/nvflare/workspace/job-1",
            "Source": "/host/workspace/job-1",
            "Type": "bind",
            "ReadOnly": False,
        }

    def test_launch_rejects_job_workspace_path_escape(self):
        launcher = _make_launcher(workspace="/host/workspace")
        dc = launcher._docker_client

        fl_ctx, _ = _make_fl_ctx()
        with pytest.raises(RuntimeError, match="single workspace child"):
            launcher.launch_job(_make_job_meta(job_id="../other"), fl_ctx)

        dc.containers.run.assert_not_called()

    def test_launch_study_data_mounts_nested_datasets(self):
        launcher = _make_launcher()
        dc = launcher._docker_client
        container = MagicMock()
        container.id = "abc123"
        dc.containers.run.return_value = container
        dc.containers.get.return_value = _make_container("running")
        study_data = {
            "study-a": {
                "training": {"source": "/data/train", "mode": "ro"},
                "output": {"source": "/data/out", "mode": "rw"},
            }
        }

        fl_ctx, _ = _make_fl_ctx()
        with patch(
            "nvflare.app_opt.job_launcher.docker_launcher.load_study_data_file", return_value=study_data
        ) as mock_load:
            with patch("nvflare.app_opt.job_launcher.docker_launcher.os.path.exists", return_value=True):
                launcher.launch_job(_make_job_meta(study="study-a"), fl_ctx)

        mock_load.assert_called_once_with("/var/tmp/nvflare/workspace/local/study_data.yaml")
        call_kwargs = dc.containers.run.call_args[1]
        assert call_kwargs["command"] == [
            "/usr/local/bin/python",
            "-u",
            "-m",
            "nvflare.private.fed.app.client.worker_process",
            "-w",
            "/ws",
            "-s",
            "/ws/startup",
            "-u",
            "tcp://site-1:8002",
        ]
        assert call_kwargs["working_dir"] == "/var/tmp/nvflare/workspace"

        mounts_by_target = _mounts_by_target(call_kwargs["mounts"])
        assert mounts_by_target["/data/study-a/training"] == {
            "Target": "/data/study-a/training",
            "Source": "/data/train",
            "Type": "bind",
            "ReadOnly": True,
        }
        assert mounts_by_target["/data/study-a/output"] == {
            "Target": "/data/study-a/output",
            "Source": "/data/out",
            "Type": "bind",
            "ReadOnly": False,
        }

    def test_launch_study_data_mounts_same_source_to_multiple_targets(self):
        launcher = _make_launcher()
        dc = launcher._docker_client
        container = MagicMock()
        container.id = "abc123"
        dc.containers.run.return_value = container
        dc.containers.get.return_value = _make_container("running")
        study_data = {
            "study-a": {
                "training": {"source": "/data/shared", "mode": "ro"},
                "validation": {"source": "/data/shared", "mode": "ro"},
            }
        }

        fl_ctx, _ = _make_fl_ctx()
        with patch("nvflare.app_opt.job_launcher.docker_launcher.load_study_data_file", return_value=study_data):
            launcher.launch_job(_make_job_meta(study="study-a"), fl_ctx)

        mounts_by_target = _mounts_by_target(dc.containers.run.call_args[1]["mounts"])
        assert mounts_by_target["/data/study-a/training"]["Source"] == "/data/shared"
        assert mounts_by_target["/data/study-a/validation"]["Source"] == "/data/shared"

    def test_launch_study_data_host_source_is_not_prechecked_from_parent_container(self):
        launcher = _make_launcher()
        dc = launcher._docker_client
        container = MagicMock()
        container.id = "abc123"
        dc.containers.run.return_value = container
        dc.containers.get.return_value = _make_container("running")
        study_data = {"study-a": {"training": {"source": "/host/not-mounted-in-parent", "mode": "ro"}}}

        fl_ctx, _ = _make_fl_ctx()
        with patch("nvflare.app_opt.job_launcher.docker_launcher.load_study_data_file", return_value=study_data):
            with patch(
                "nvflare.app_opt.job_launcher.docker_launcher.os.path.exists",
                side_effect=AssertionError("host source should be left for Docker to validate"),
            ):
                launcher.launch_job(_make_job_meta(study="study-a"), fl_ctx)

        mounts_by_target = _mounts_by_target(dc.containers.run.call_args[1]["mounts"])
        assert mounts_by_target["/data/study-a/training"] == {
            "Target": "/data/study-a/training",
            "Source": "/host/not-mounted-in-parent",
            "Type": "bind",
            "ReadOnly": True,
        }

    def test_launch_default_study_without_mapping_does_not_mount_data(self):
        launcher = _make_launcher()
        dc = launcher._docker_client
        container = MagicMock()
        container.id = "abc123"
        dc.containers.run.return_value = container
        dc.containers.get.return_value = _make_container("running")

        fl_ctx, _ = _make_fl_ctx()
        with patch("nvflare.app_opt.job_launcher.docker_launcher.load_study_data_file", return_value={}) as mock_load:
            launcher.launch_job(_make_job_meta(study="default"), fl_ctx)

        mock_load.assert_called_once_with("/var/tmp/nvflare/workspace/local/study_data.yaml")
        mounts_by_target = _mounts_by_target(dc.containers.run.call_args[1]["mounts"])
        assert set(mounts_by_target) == {
            "/var/tmp/nvflare/workspace",
            "/var/tmp/nvflare/workspace/startup",
            "/var/tmp/nvflare/workspace/local",
            "/var/tmp/nvflare/workspace/job-1",
        }

    def test_launch_default_study_mounts_default_mapping_when_present(self):
        launcher = _make_launcher()
        dc = launcher._docker_client
        container = MagicMock()
        container.id = "abc123"
        dc.containers.run.return_value = container
        dc.containers.get.return_value = _make_container("running")
        study_data = {"default": {"training": {"source": "/data/default-train", "mode": "ro"}}}

        fl_ctx, _ = _make_fl_ctx()
        with patch("nvflare.app_opt.job_launcher.docker_launcher.load_study_data_file", return_value=study_data):
            with patch("nvflare.app_opt.job_launcher.docker_launcher.os.path.exists", return_value=True):
                launcher.launch_job(_make_job_meta(study="default"), fl_ctx)

        mounts_by_target = _mounts_by_target(dc.containers.run.call_args[1]["mounts"])
        assert mounts_by_target["/data/default/training"] == {
            "Target": "/data/default/training",
            "Source": "/data/default-train",
            "Type": "bind",
            "ReadOnly": True,
        }

    def test_launch_omits_data_mount_when_study_mapping_is_missing(self):
        launcher = _make_launcher()
        dc = launcher._docker_client
        container = MagicMock()
        container.id = "abc123"
        dc.containers.run.return_value = container
        dc.containers.get.return_value = _make_container("running")
        study_data = {"other-study": {"training": {"source": "/data/train", "mode": "ro"}}}

        fl_ctx, _ = _make_fl_ctx()
        with patch("nvflare.app_opt.job_launcher.docker_launcher.load_study_data_file", return_value=study_data):
            launcher.launch_job(_make_job_meta(study="study-a"), fl_ctx)

        mounts_by_target = _mounts_by_target(dc.containers.run.call_args[1]["mounts"])
        assert set(mounts_by_target) == {
            "/var/tmp/nvflare/workspace",
            "/var/tmp/nvflare/workspace/startup",
            "/var/tmp/nvflare/workspace/local",
            "/var/tmp/nvflare/workspace/job-1",
        }

    def test_launch_no_docker_socket_in_job_container(self):
        """Job containers must never receive the Docker socket."""
        launcher = _make_launcher()
        dc = launcher._docker_client
        container = MagicMock()
        container.id = "abc123"
        dc.containers.run.return_value = container
        dc.containers.get.return_value = _make_container("running")

        fl_ctx, _ = _make_fl_ctx()
        launcher.launch_job(_make_job_meta(), fl_ctx)

        call_kwargs = dc.containers.run.call_args[1]
        mounts = call_kwargs.get("mounts", [])
        assert all(m["Source"] != "/var/run/docker.sock" for m in mounts)

    def test_launch_merges_default_job_env(self):
        launcher = _make_launcher(default_job_env={"NCCL_P2P_DISABLE": "1"})
        dc = launcher._docker_client
        container = MagicMock()
        container.id = "abc123"
        dc.containers.run.return_value = container
        dc.containers.get.return_value = _make_container("running")

        fl_ctx, _ = _make_fl_ctx()
        launcher.launch_job(_make_job_meta(), fl_ctx)

        call_kwargs = dc.containers.run.call_args[1]
        environment = call_kwargs["environment"]
        assert environment["NCCL_P2P_DISABLE"] == "1"
        assert "USER" in environment
        assert "HOME" in environment

    def test_launcher_env_overrides_default_job_env(self):
        launcher = _make_launcher(default_job_env={"USER": "wrong", "HOME": "/bad/home"})
        dc = launcher._docker_client
        container = MagicMock()
        container.id = "abc123"
        dc.containers.run.return_value = container
        dc.containers.get.return_value = _make_container("running")

        with patch.dict("os.environ", {"USER": "actual-user", "HOME": "/real/home"}, clear=False):
            fl_ctx, _ = _make_fl_ctx()
            launcher.launch_job(_make_job_meta(), fl_ctx)

        call_kwargs = dc.containers.run.call_args[1]
        environment = call_kwargs["environment"]
        assert environment["USER"] == "actual-user"
        assert environment["HOME"] == "/real/home"

    def test_launch_python_path_from_launcher_spec_overrides_default(self):
        launcher = _make_launcher(default_python_path="/usr/bin/python")
        dc = launcher._docker_client
        container = MagicMock()
        container.id = "abc123"
        dc.containers.run.return_value = container
        dc.containers.get.return_value = _make_container("running")

        fl_ctx, _ = _make_fl_ctx(identity_name="site-1")
        job_meta = _make_job_meta(site_name="site-1", docker_spec={"python_path": "/opt/conda/bin/python"})
        launcher.launch_job(job_meta, fl_ctx)

        call_kwargs = dc.containers.run.call_args[1]
        assert call_kwargs["command"][0] == "/opt/conda/bin/python"
        assert "python_path" not in call_kwargs

    def test_launch_gpu_via_resource_spec_num_of_gpus(self):
        """num_of_gpus in resource_spec.docker is translated to device_requests for the job container."""
        launcher = _make_launcher()
        dc = launcher._docker_client
        container = MagicMock()
        container.id = "abc123"
        dc.containers.run.return_value = container
        dc.containers.get.return_value = _make_container("running")

        fl_ctx, _ = _make_fl_ctx(identity_name="site-1")
        job_meta = _make_job_meta(site_name="site-1", docker_spec={"num_of_gpus": 2})
        launcher.launch_job(job_meta, fl_ctx)

        call_kwargs = dc.containers.run.call_args[1]
        device_requests = call_kwargs.get("device_requests")
        assert device_requests == [{"Count": 2, "Capabilities": [["gpu"]]}]

    def test_launch_docker_spec_device_requests_overrides_num_of_gpus(self):
        """Explicit device_requests in docker_spec takes precedence over num_of_gpus."""
        launcher = _make_launcher()
        dc = launcher._docker_client
        container = MagicMock()
        container.id = "abc123"
        dc.containers.run.return_value = container
        dc.containers.get.return_value = _make_container("running")

        fl_ctx, _ = _make_fl_ctx(identity_name="site-1")
        explicit_dr = [{"Count": 4, "Capabilities": [["gpu"]]}]
        job_meta = _make_job_meta(
            site_name="site-1",
            docker_spec={"num_of_gpus": 1, "device_requests": explicit_dr},
        )
        launcher.launch_job(job_meta, fl_ctx)

        call_kwargs = dc.containers.run.call_args[1]
        assert call_kwargs.get("device_requests") == explicit_dr

    def test_launch_num_of_gpus_overrides_default_device_requests(self):
        """Job-level num_of_gpus must override site-level default device_requests."""
        launcher = _make_launcher(
            default_job_container_kwargs={"device_requests": [{"Count": 1, "Capabilities": [["gpu"]]}]}
        )
        dc = launcher._docker_client
        container = MagicMock()
        container.id = "abc123"
        dc.containers.run.return_value = container
        dc.containers.get.return_value = _make_container("running")

        fl_ctx, _ = _make_fl_ctx(identity_name="site-1")
        job_meta = _make_job_meta(site_name="site-1", docker_spec={"num_of_gpus": 2})
        launcher.launch_job(job_meta, fl_ctx)

        call_kwargs = dc.containers.run.call_args[1]
        assert call_kwargs.get("device_requests") == [{"Count": 2, "Capabilities": [["gpu"]]}]

    def test_launch_num_of_gpus_mixed_with_mode_keys_ignored(self):
        """num_of_gpus at the site level alongside mode keys (nested format) is treated as nested —
        the site-level num_of_gpus is not used as a flat fallback."""
        launcher = _make_launcher()
        dc = launcher._docker_client
        container = MagicMock()
        container.id = "abc123"
        dc.containers.run.return_value = container
        dc.containers.get.return_value = _make_container("running")

        fl_ctx, _ = _make_fl_ctx(identity_name="site-1")
        job_meta = _make_job_meta(
            site_name="site-1",
            resource_spec={"site-1": {"docker": {"image": "nvflare:test"}, "num_of_gpus": 2}},
        )
        launcher.launch_job(job_meta, fl_ctx)

        call_kwargs = dc.containers.run.call_args[1]
        assert call_kwargs.get("device_requests") is None

    def test_launch_num_of_gpus_from_flat_resource_spec(self):
        """Option 4: num_of_gpus in flat resource_spec[site] is used by Docker launcher."""
        launcher = _make_launcher()
        dc = launcher._docker_client
        container = MagicMock()
        container.id = "abc123"
        dc.containers.run.return_value = container
        dc.containers.get.return_value = _make_container("running")

        fl_ctx, _ = _make_fl_ctx(identity_name="site-1")
        job_meta = _make_job_meta(site_name="site-1")
        job_meta[JobMetaKey.RESOURCE_SPEC.value] = {"site-1": {"num_of_gpus": 2}}
        launcher.launch_job(job_meta, fl_ctx)

        call_kwargs = dc.containers.run.call_args[1]
        assert call_kwargs.get("device_requests") == [{"Count": 2, "Capabilities": [["gpu"]]}]

    def test_launch_image_from_launcher_spec_default(self):
        """launcher_spec 'default' key applies to all sites that have no explicit entry."""
        launcher = _make_launcher()
        dc = launcher._docker_client
        container = MagicMock()
        container.id = "abc123"
        dc.containers.run.return_value = container
        dc.containers.get.return_value = _make_container("running")

        fl_ctx, _ = _make_fl_ctx(identity_name="site-1")
        job_meta = {
            JobConstants.JOB_ID: "job-1",
            JobMetaKey.JOB_LAUNCHER_SPEC.value: {
                "default": {"docker": {"image": "default/img:v1"}},
            },
        }
        launcher.launch_job(job_meta, fl_ctx)

        assert dc.containers.run.call_args[0][0] == "default/img:v1"

    def test_launch_no_container_kwargs_no_extra_keys(self):
        launcher = _make_launcher()
        dc = launcher._docker_client
        container = MagicMock()
        container.id = "abc123"
        dc.containers.run.return_value = container
        dc.containers.get.return_value = _make_container("running")

        fl_ctx, _ = _make_fl_ctx()
        launcher.launch_job(_make_job_meta(), fl_ctx)

        call_kwargs = dc.containers.run.call_args[1]
        assert call_kwargs.get("device_requests") is None

    def test_launch_always_returns_handle_even_if_not_running(self):
        """Launcher must always return a handle — caller detects failure via poll()."""
        launcher = _make_launcher()
        dc = launcher._docker_client
        container = MagicMock()
        container.id = "abc123"
        dc.containers.run.return_value = container
        # Container never reaches RUNNING (stuck in created → terminated)
        dc.containers.get.return_value = _make_container("exited", exit_code=1)

        fl_ctx, _ = _make_fl_ctx()
        handle = launcher.launch_job(_make_job_meta(), fl_ctx)

        assert handle is not None

    def test_launch_image_not_found_raises(self):
        launcher = _make_launcher()
        dc = launcher._docker_client
        dc.containers.run.side_effect = _ImageNotFound("no such image")

        fl_ctx, _ = _make_fl_ctx()
        with pytest.raises(RuntimeError, match="not found"):
            launcher.launch_job(_make_job_meta(), fl_ctx)


# ---------------------------------------------------------------------------
# DockerJobLauncher — container kwargs merge (site-level defaults + job-level docker_spec)
# ---------------------------------------------------------------------------


class TestContainerKwargsMerge:
    def _run_launch(self, launcher, job_meta):
        dc = launcher._docker_client
        container = MagicMock()
        container.id = "abc123"
        dc.containers.run.return_value = container
        dc.containers.get.return_value = _make_container("running")
        fl_ctx, _ = _make_fl_ctx(identity_name="site-1")
        launcher.launch_job(job_meta, fl_ctx)
        return dc.containers.run.call_args[1]

    def test_site_kwargs_passed_when_no_job_kwargs(self):
        launcher = _make_launcher(default_job_container_kwargs={"ipc_mode": "host"})
        call_kwargs = self._run_launch(launcher, _make_job_meta())
        assert call_kwargs.get("ipc_mode") == "host"

    def test_job_kwargs_from_docker_spec_passed(self):
        launcher = _make_launcher()
        job_meta = _make_job_meta(docker_spec={"shm_size": "8g"})
        call_kwargs = self._run_launch(launcher, job_meta)
        assert call_kwargs.get("shm_size") == "8g"

    def test_job_and_site_kwargs_merged(self):
        launcher = _make_launcher(default_job_container_kwargs={"ipc_mode": "host"})
        job_meta = _make_job_meta(docker_spec={"shm_size": "8g"})
        call_kwargs = self._run_launch(launcher, job_meta)
        assert call_kwargs.get("ipc_mode") == "host"
        assert call_kwargs.get("shm_size") == "8g"

    def test_job_kwargs_override_site_kwargs_on_conflict(self):
        launcher = _make_launcher(default_job_container_kwargs={"shm_size": "2g"})
        job_meta = _make_job_meta(docker_spec={"shm_size": "8g"})
        call_kwargs = self._run_launch(launcher, job_meta)
        assert call_kwargs.get("shm_size") == "8g"

    def test_no_kwargs_results_in_no_extra_keys(self):
        launcher = _make_launcher()
        call_kwargs = self._run_launch(launcher, _make_job_meta())
        assert "shm_size" not in call_kwargs
        assert "ipc_mode" not in call_kwargs


# ---------------------------------------------------------------------------
# DockerJobLauncher — image allowlist
# ---------------------------------------------------------------------------


class TestHandleEvent:
    def test_handle_event_always_adds_launcher(self):
        launcher = _make_launcher()
        fl_ctx, _ = _make_fl_ctx()

        with patch("nvflare.app_opt.job_launcher.docker_launcher.add_launcher") as mock_add:
            launcher.handle_event(EventType.BEFORE_JOB_LAUNCH, fl_ctx)
            mock_add.assert_called_once()


# ---------------------------------------------------------------------------
# ClientDockerJobLauncher / ServerDockerJobLauncher — get_module_args
# ---------------------------------------------------------------------------


class TestGetModuleArgs:
    def test_client_launcher_is_concrete(self):
        """ClientDockerJobLauncher must implement get_module_args (not abstract)."""
        launcher = _make_launcher(cls=ClientDockerJobLauncher)
        result = launcher.get_module_args({})
        assert isinstance(result, dict)

    def test_server_launcher_is_concrete(self):
        launcher = _make_launcher(cls=ServerDockerJobLauncher)
        result = launcher.get_module_args({})
        assert isinstance(result, dict)
