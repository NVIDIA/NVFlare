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


def _make_launcher(cls=None, workspace="/ws", parent_url="nvflare-cp:8002", **kwargs):
    """Create a launcher with a pre-wired mock Docker client (no real Docker needed)."""
    if cls is None:
        cls = ClientDockerJobLauncher

    dc = _make_docker_client()
    with patch("docker.from_env", return_value=dc):
        launcher = cls.__new__(cls)
        DockerJobLauncher.__init__(launcher, workspace=workspace, parent_url=parent_url, **kwargs)
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

    def test_stuck_in_created_terminates_and_returns_false(self):
        dc = _make_docker_client()
        dc.containers.get.return_value = _make_container("created")
        h = _make_handle(docker_client=dc, pending_timeout=3)
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
            DockerJobLauncher.__init__(launcher, workspace=None, parent_url="cp:8002")
            launcher._docker_client = _make_docker_client()
            fl_ctx, _ = _make_fl_ctx()
            with patch("nvflare.app_opt.job_launcher.docker_launcher.extract_job_image", return_value="nvflare:test"):
                with pytest.raises(ValueError, match="workspace"):
                    launcher.launch_job(_make_job_meta(), fl_ctx)

    def test_workspace_read_from_env_if_not_provided(self):
        dc = _make_docker_client()
        with patch("docker.from_env", return_value=dc):
            with patch.dict("os.environ", {"NVFL_DOCKER_WORKSPACE": "/host/ws"}):
                launcher = ClientDockerJobLauncher.__new__(ClientDockerJobLauncher)
                DockerJobLauncher.__init__(launcher, workspace=None, parent_url="cp:8002")
        assert launcher.workspace == "/host/ws"

    def test_raises_if_parent_url_empty(self):
        with pytest.raises(ValueError, match="parent_url"):
            _make_launcher(workspace="/ws", parent_url="")

    def test_raises_if_extra_container_kwargs_contains_reserved_key(self):
        for reserved in ("volumes", "network", "device_requests", "environment", "command", "name", "detach"):
            with pytest.raises(ValueError, match="reserved"):
                _make_launcher(extra_container_kwargs={reserved: "anything"})

    def test_extra_container_kwargs_non_reserved_accepted(self):
        launcher = _make_launcher(extra_container_kwargs={"shm_size": "2g", "ipc_mode": "host"})
        assert launcher.extra_container_kwargs == {"shm_size": "2g", "ipc_mode": "host"}

    def test_raises_if_docker_not_reachable(self):
        """Docker connectivity is validated lazily in _get_docker_client."""
        dc = _make_docker_client()
        dc.ping.side_effect = Exception("connection refused")
        launcher = ClientDockerJobLauncher.__new__(ClientDockerJobLauncher)
        DockerJobLauncher.__init__(launcher, workspace="/ws", parent_url="cp:8002")
        with patch("docker.from_env", return_value=dc):
            with pytest.raises(RuntimeError, match="cannot connect"):
                launcher._get_docker_client()

    def test_raises_if_network_not_found(self):
        """Network existence is validated lazily in _get_docker_client."""
        dc = _make_docker_client()
        dc.networks.get.side_effect = _NotFound()
        launcher = ClientDockerJobLauncher.__new__(ClientDockerJobLauncher)
        DockerJobLauncher.__init__(launcher, workspace="/ws", parent_url="cp:8002")
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


def _make_job_meta(job_id="job-1", job_image="nvflare/nvflare:test", site_name="site-1", num_of_gpus=None):
    meta = {JobConstants.JOB_ID: job_id, "job_image": job_image}
    if num_of_gpus is not None:
        meta[JobMetaKey.RESOURCE_SPEC.value] = {site_name: {"num_of_gpus": num_of_gpus}}
    return meta


class TestDockerJobLauncherLaunchJob:
    def test_launch_returns_handle(self):
        launcher = _make_launcher()
        dc = launcher._docker_client
        container = MagicMock()
        container.id = "abc123"
        container.status = "running"
        dc.containers.run.return_value = container
        dc.containers.get.return_value = _make_container("running")

        fl_ctx, _ = _make_fl_ctx()
        with patch("nvflare.app_opt.job_launcher.docker_launcher.extract_job_image", return_value="nvflare:test"):
            handle = launcher.launch_job(_make_job_meta(), fl_ctx)

        assert handle is not None
        assert isinstance(handle, DockerJobHandle)

    def test_launch_overrides_parent_url(self):
        launcher = _make_launcher(parent_url="nvflare-cp:8002")
        dc = launcher._docker_client
        container = MagicMock()
        container.id = "abc123"
        container.status = "running"
        dc.containers.run.return_value = container
        dc.containers.get.return_value = _make_container("running")

        fl_ctx, job_args = _make_fl_ctx(parent_url="localhost:8002")
        with patch("nvflare.app_opt.job_launcher.docker_launcher.extract_job_image", return_value="nvflare:test"):
            launcher.launch_job(_make_job_meta(), fl_ctx)

        # The launcher must NOT pass job_args' original PARENT_URL to containers.run —
        # it should pass self.parent_url ("nvflare-cp:8002") instead.
        call_kwargs = dc.containers.run.call_args
        command = call_kwargs[1]["command"] if call_kwargs[1] else call_kwargs[0][1]
        # The command list must not contain localhost:8002
        assert "localhost:8002" not in " ".join(str(a) for a in command)

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
        with patch("nvflare.app_opt.job_launcher.docker_launcher.extract_job_image", return_value="nvflare:test"):
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
        with patch("nvflare.app_opt.job_launcher.docker_launcher.extract_job_image", return_value="nvflare:test"):
            launcher.launch_job(_make_job_meta(), fl_ctx)

        call_kwargs = dc.containers.run.call_args[1]
        volumes = call_kwargs["volumes"]
        assert "/host/workspace" in volumes
        assert volumes["/host/workspace"]["mode"] == "rw"

    def test_launch_no_docker_socket_in_job_container(self):
        """Job containers must never receive the Docker socket."""
        launcher = _make_launcher()
        dc = launcher._docker_client
        container = MagicMock()
        container.id = "abc123"
        dc.containers.run.return_value = container
        dc.containers.get.return_value = _make_container("running")

        fl_ctx, _ = _make_fl_ctx()
        with patch("nvflare.app_opt.job_launcher.docker_launcher.extract_job_image", return_value="nvflare:test"):
            launcher.launch_job(_make_job_meta(), fl_ctx)

        call_kwargs = dc.containers.run.call_args[1]
        volumes = call_kwargs.get("volumes", {})
        assert "/var/run/docker.sock" not in volumes

    def test_launch_gpu_device_request_added(self):
        launcher = _make_launcher()
        dc = launcher._docker_client
        container = MagicMock()
        container.id = "abc123"
        dc.containers.run.return_value = container
        dc.containers.get.return_value = _make_container("running")

        fl_ctx, _ = _make_fl_ctx(identity_name="site-1")
        job_meta = _make_job_meta(site_name="site-1", num_of_gpus=2)
        with patch("nvflare.app_opt.job_launcher.docker_launcher.extract_job_image", return_value="nvflare:test"):
            launcher.launch_job(job_meta, fl_ctx)

        call_kwargs = dc.containers.run.call_args[1]
        assert call_kwargs.get("device_requests") is not None

    def test_launch_no_gpu_no_device_requests(self):
        launcher = _make_launcher()
        dc = launcher._docker_client
        container = MagicMock()
        container.id = "abc123"
        dc.containers.run.return_value = container
        dc.containers.get.return_value = _make_container("running")

        fl_ctx, _ = _make_fl_ctx()
        with patch("nvflare.app_opt.job_launcher.docker_launcher.extract_job_image", return_value="nvflare:test"):
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
        with patch("nvflare.app_opt.job_launcher.docker_launcher.extract_job_image", return_value="nvflare:test"):
            handle = launcher.launch_job(_make_job_meta(), fl_ctx)

        assert handle is not None

    def test_launch_image_not_found_raises(self):
        launcher = _make_launcher()
        dc = launcher._docker_client
        dc.containers.run.side_effect = _ImageNotFound("no such image")

        fl_ctx, _ = _make_fl_ctx()
        with patch("nvflare.app_opt.job_launcher.docker_launcher.extract_job_image", return_value="nvflare:test"):
            with pytest.raises(RuntimeError, match="not found"):
                launcher.launch_job(_make_job_meta(), fl_ctx)


# ---------------------------------------------------------------------------
# DockerJobLauncher — image allowlist
# ---------------------------------------------------------------------------


class TestImageAllowlist:
    def test_no_allowlist_permits_any_image(self):
        launcher = _make_launcher()
        assert launcher._is_image_allowed("anything/goes:latest") is True

    def test_matching_prefix_is_allowed(self):
        launcher = _make_launcher()
        launcher.allowed_image_prefixes = ["myregistry.corp.com/", "nvflare/nvflare:"]
        assert launcher._is_image_allowed("myregistry.corp.com/myimage:1.0") is True
        assert launcher._is_image_allowed("nvflare/nvflare:2.5.0") is True

    def test_non_matching_prefix_is_blocked(self):
        launcher = _make_launcher()
        launcher.allowed_image_prefixes = ["myregistry.corp.com/"]
        assert launcher._is_image_allowed("docker.io/evilimage:latest") is False

    def test_handle_event_raises_on_disallowed_image(self):
        launcher = _make_launcher()
        launcher.allowed_image_prefixes = ["trusted.registry.com/"]
        fl_ctx, _ = _make_fl_ctx()
        fl_ctx.get_prop.side_effect = lambda key, *a, **kw: {FLContextKey.JOB_META: _make_job_meta()}.get(key)

        with patch(
            "nvflare.app_opt.job_launcher.docker_launcher.extract_job_image",
            return_value="untrusted.io/badimage:latest",
        ):
            with pytest.raises(RuntimeError, match="not permitted"):
                launcher.handle_event(EventType.BEFORE_JOB_LAUNCH, fl_ctx)

    def test_handle_event_allows_matching_image(self):
        launcher = _make_launcher()
        launcher.allowed_image_prefixes = ["trusted.registry.com/"]
        fl_ctx, _ = _make_fl_ctx()
        fl_ctx.get_prop.side_effect = lambda key, *a, **kw: {FLContextKey.JOB_META: _make_job_meta()}.get(key)

        with patch(
            "nvflare.app_opt.job_launcher.docker_launcher.extract_job_image",
            return_value="trusted.registry.com/nvflare:2.5",
        ):
            with patch("nvflare.app_opt.job_launcher.docker_launcher.add_launcher") as mock_add:
                launcher.handle_event(EventType.BEFORE_JOB_LAUNCH, fl_ctx)
                mock_add.assert_called_once()

    def test_handle_event_skips_if_no_image(self):
        """Jobs without a job_image don't trigger the launcher at all."""
        launcher = _make_launcher()
        launcher.allowed_image_prefixes = ["trusted.registry.com/"]
        fl_ctx, _ = _make_fl_ctx()
        fl_ctx.get_prop.side_effect = lambda key, *a, **kw: {FLContextKey.JOB_META: {}}.get(key)

        with patch("nvflare.app_opt.job_launcher.docker_launcher.extract_job_image", return_value=None):
            with patch("nvflare.app_opt.job_launcher.docker_launcher.add_launcher") as mock_add:
                launcher.handle_event(EventType.BEFORE_JOB_LAUNCH, fl_ctx)
                mock_add.assert_not_called()


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
