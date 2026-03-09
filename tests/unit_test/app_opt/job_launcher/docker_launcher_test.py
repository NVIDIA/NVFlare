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

from unittest.mock import Mock, patch

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, JobConstants, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import JobMetaKey
from nvflare.apis.job_launcher_spec import JobReturnCode
from nvflare.app_opt.job_launcher.docker_launcher import (
    DOCKER_STATE,
    JOB_RETURN_CODE_MAPPING,
    DockerJobHandle,
    DockerJobLauncher,
)


# ---------------------------------------------------------------------------
# Constants and mappings
# ---------------------------------------------------------------------------
class TestDockerState:
    def test_state_values(self):
        assert DOCKER_STATE.CREATED == "created"
        assert DOCKER_STATE.RESTARTING == "restarting"
        assert DOCKER_STATE.RUNNING == "running"
        assert DOCKER_STATE.PAUSED == "paused"
        assert DOCKER_STATE.EXITED == "exited"
        assert DOCKER_STATE.DEAD == "dead"


class TestDockerJobReturnCodeMapping:
    def test_running_maps_to_unknown(self):
        assert JOB_RETURN_CODE_MAPPING[DOCKER_STATE.RUNNING] == JobReturnCode.UNKNOWN

    def test_exited_maps_to_success(self):
        assert JOB_RETURN_CODE_MAPPING[DOCKER_STATE.EXITED] == JobReturnCode.SUCCESS

    def test_dead_maps_to_aborted(self):
        assert JOB_RETURN_CODE_MAPPING[DOCKER_STATE.DEAD] == JobReturnCode.ABORTED

    def test_created_maps_to_unknown(self):
        assert JOB_RETURN_CODE_MAPPING[DOCKER_STATE.CREATED] == JobReturnCode.UNKNOWN

    def test_paused_maps_to_unknown(self):
        assert JOB_RETURN_CODE_MAPPING[DOCKER_STATE.PAUSED] == JobReturnCode.UNKNOWN

    def test_restarting_maps_to_unknown(self):
        assert JOB_RETURN_CODE_MAPPING[DOCKER_STATE.RESTARTING] == JobReturnCode.UNKNOWN


# ---------------------------------------------------------------------------
# DockerJobHandle
# ---------------------------------------------------------------------------
class TestDockerJobHandle:
    def test_init_defaults(self):
        handle = DockerJobHandle()
        assert handle.container is None
        assert handle.timeout is None

    def test_init_with_timeout(self):
        handle = DockerJobHandle(timeout=30)
        assert handle.container is None
        assert handle.timeout == 30

    def test_set_container(self):
        handle = DockerJobHandle()
        container = Mock()
        handle._set_container(container)
        assert handle.container is container

    def test_terminate_stops_container(self):
        handle = DockerJobHandle()
        container = Mock()
        handle._set_container(container)
        handle.terminate()
        container.stop.assert_called_once()

    def test_terminate_noop_when_no_container(self):
        handle = DockerJobHandle()
        handle.terminate()

    # -- poll -----------------------------------------------------------------
    @patch.object(DockerJobHandle, "_get_container")
    def test_poll_running_returns_unknown(self, mock_get):
        container = Mock()
        container.status = DOCKER_STATE.RUNNING
        mock_get.return_value = container
        handle = DockerJobHandle()
        assert handle.poll() == JobReturnCode.UNKNOWN

    @patch.object(DockerJobHandle, "_get_container")
    def test_poll_exited_removes_and_returns_success(self, mock_get):
        container = Mock()
        container.status = DOCKER_STATE.EXITED
        mock_get.return_value = container
        handle = DockerJobHandle()
        result = handle.poll()
        container.remove.assert_called_once_with(force=True)
        assert result == JobReturnCode.SUCCESS

    @patch.object(DockerJobHandle, "_get_container")
    def test_poll_dead_removes_and_returns_aborted(self, mock_get):
        container = Mock()
        container.status = DOCKER_STATE.DEAD
        mock_get.return_value = container
        handle = DockerJobHandle()
        result = handle.poll()
        container.remove.assert_called_once_with(force=True)
        assert result == JobReturnCode.ABORTED

    @patch.object(DockerJobHandle, "_get_container")
    def test_poll_returns_none_when_container_gone(self, mock_get):
        mock_get.return_value = None
        handle = DockerJobHandle()
        assert handle.poll() is None

    @patch.object(DockerJobHandle, "_get_container")
    def test_poll_unknown_status_returns_unknown(self, mock_get):
        container = Mock()
        container.status = "something_unexpected"
        mock_get.return_value = container
        handle = DockerJobHandle()
        assert handle.poll() == JobReturnCode.UNKNOWN

    # -- wait -----------------------------------------------------------------
    @patch.object(DockerJobHandle, "enter_states")
    def test_wait_calls_enter_states(self, mock_enter):
        handle = DockerJobHandle(timeout=10)
        handle._set_container(Mock())
        handle.wait()
        mock_enter.assert_called_once_with([DOCKER_STATE.EXITED, DOCKER_STATE.DEAD], 10)

    def test_wait_noop_when_no_container(self):
        handle = DockerJobHandle()
        handle.wait()

    # -- enter_states ---------------------------------------------------------
    @patch.object(DockerJobHandle, "_get_container")
    def test_enter_states_returns_true_when_state_matches(self, mock_get):
        container = Mock()
        container.status = DOCKER_STATE.RUNNING
        mock_get.return_value = container
        handle = DockerJobHandle()
        assert handle.enter_states([DOCKER_STATE.RUNNING]) is True

    @patch.object(DockerJobHandle, "_get_container")
    def test_enter_states_returns_false_when_container_gone(self, mock_get):
        mock_get.return_value = None
        handle = DockerJobHandle()
        assert handle.enter_states([DOCKER_STATE.RUNNING]) is False

    @patch.object(DockerJobHandle, "_get_container")
    def test_enter_states_returns_false_on_timeout(self, mock_get):
        container = Mock()
        container.status = DOCKER_STATE.CREATED
        mock_get.return_value = container
        handle = DockerJobHandle()
        assert handle.enter_states([DOCKER_STATE.RUNNING], timeout=0) is False

    @patch.object(DockerJobHandle, "_get_container")
    def test_enter_states_wraps_single_state(self, mock_get):
        container = Mock()
        container.status = DOCKER_STATE.EXITED
        mock_get.return_value = container
        handle = DockerJobHandle()
        assert handle.enter_states(DOCKER_STATE.EXITED) is True

    # -- _get_container -------------------------------------------------------
    @patch("nvflare.app_opt.job_launcher.docker_launcher.docker")
    def test_get_container_returns_container(self, mock_docker):
        orig_container = Mock()
        orig_container.id = "abc123"
        refreshed = Mock()
        mock_docker.from_env.return_value.containers.get.return_value = refreshed

        handle = DockerJobHandle()
        handle._set_container(orig_container)
        result = handle._get_container()
        assert result is refreshed
        mock_docker.from_env.return_value.containers.get.assert_called_once_with("abc123")

    @patch("nvflare.app_opt.job_launcher.docker_launcher.docker")
    def test_get_container_returns_none_on_exception(self, mock_docker):
        orig_container = Mock()
        orig_container.id = "abc123"
        mock_docker.from_env.side_effect = Exception("connection error")

        handle = DockerJobHandle()
        handle._set_container(orig_container)
        assert handle._get_container() is None


# ---------------------------------------------------------------------------
# DockerJobLauncher
# ---------------------------------------------------------------------------
def _make_fl_ctx_for_docker_launch():
    fl_ctx = FLContext()
    fl_ctx.set_prop(ReservedKey.IDENTITY_NAME, "client-1", private=False, sticky=True)
    workspace_obj = Mock()
    workspace_obj.get_app_custom_dir.return_value = "/custom/dir"
    fl_ctx.set_prop(FLContextKey.WORKSPACE_OBJECT, workspace_obj, private=True, sticky=False)
    fl_ctx.set_prop(
        FLContextKey.JOB_PROCESS_ARGS,
        {
            "exe_module": ("-m", "nvflare.private.fed.app.client.worker_process"),
            "workspace": ("-w", "/workspace"),
        },
        private=True,
        sticky=False,
    )
    return fl_ctx


def _make_docker_job_meta(image="nvflare/nvflare:test", job_id="job-123"):
    return {
        JobConstants.JOB_ID: job_id,
        JobMetaKey.DEPLOY_MAP.value: {"app": [{"sites": ["client-1"], "image": image}]},
    }


class TestDockerJobLauncher:
    def test_init_defaults(self):
        launcher = DockerJobLauncher()
        assert launcher.mount_path == "/workspace"
        assert launcher.network == "nvflare-network"
        assert launcher.timeout is None

    def test_init_custom(self):
        launcher = DockerJobLauncher(mount_path="/custom", network="my-net", timeout=120)
        assert launcher.mount_path == "/custom"
        assert launcher.network == "my-net"
        assert launcher.timeout == 120

    # -- handle_event ---------------------------------------------------------
    def test_handle_event_adds_launcher_when_image_present(self):
        from nvflare.app_opt.job_launcher.docker_launcher import ClientDockerJobLauncher

        launcher = ClientDockerJobLauncher()
        fl_ctx = FLContext()
        job_meta = {JobMetaKey.DEPLOY_MAP.value: {"app": [{"sites": ["client-1"], "image": "nvflare/custom:latest"}]}}
        fl_ctx.set_prop(FLContextKey.JOB_META, job_meta, private=True, sticky=False)
        fl_ctx.set_prop(ReservedKey.IDENTITY_NAME, "client-1", private=False, sticky=True)

        launcher.handle_event(EventType.BEFORE_JOB_LAUNCH, fl_ctx)

        launchers = fl_ctx.get_prop(FLContextKey.JOB_LAUNCHER)
        assert launchers is not None
        assert launcher in launchers

    def test_handle_event_skips_when_no_image(self):
        from nvflare.app_opt.job_launcher.docker_launcher import ClientDockerJobLauncher

        launcher = ClientDockerJobLauncher()
        fl_ctx = FLContext()
        job_meta = {JobMetaKey.DEPLOY_MAP.value: {}}
        fl_ctx.set_prop(FLContextKey.JOB_META, job_meta, private=True, sticky=False)
        fl_ctx.set_prop(ReservedKey.IDENTITY_NAME, "client-1", private=False, sticky=True)

        launcher.handle_event(EventType.BEFORE_JOB_LAUNCH, fl_ctx)
        assert fl_ctx.get_prop(FLContextKey.JOB_LAUNCHER) is None

    def test_handle_event_ignores_other_events(self):
        from nvflare.app_opt.job_launcher.docker_launcher import ClientDockerJobLauncher

        launcher = ClientDockerJobLauncher()
        fl_ctx = FLContext()
        launcher.handle_event(EventType.SYSTEM_START, fl_ctx)
        assert fl_ctx.get_prop(FLContextKey.JOB_LAUNCHER) is None

    # -- launch_job -----------------------------------------------------------
    @patch("nvflare.app_opt.job_launcher.docker_launcher.docker")
    @patch("nvflare.app_opt.job_launcher.docker_launcher.os")
    def test_launch_job_success(self, mock_os, mock_docker):
        from nvflare.app_opt.job_launcher.docker_launcher import ClientDockerJobLauncher

        mock_os.environ.get.return_value = "/docker/workspace"
        container = Mock()
        container.status = DOCKER_STATE.RUNNING
        mock_docker.from_env.return_value.containers.run.return_value = container

        launcher = ClientDockerJobLauncher(timeout=5)
        fl_ctx = _make_fl_ctx_for_docker_launch()
        job_meta = _make_docker_job_meta()

        with patch.object(DockerJobHandle, "enter_states", return_value=True):
            handle = launcher.launch_job(job_meta, fl_ctx)

        assert handle is not None
        mock_docker.from_env.return_value.containers.run.assert_called_once()

    @patch("nvflare.app_opt.job_launcher.docker_launcher.docker")
    @patch("nvflare.app_opt.job_launcher.docker_launcher.os")
    def test_launch_job_returns_handle_on_enter_states_failure(self, mock_os, mock_docker):
        from nvflare.app_opt.job_launcher.docker_launcher import ClientDockerJobLauncher

        mock_os.environ.get.return_value = "/docker/workspace"
        container = Mock()
        mock_docker.from_env.return_value.containers.run.return_value = container

        launcher = ClientDockerJobLauncher(timeout=1)
        fl_ctx = _make_fl_ctx_for_docker_launch()
        job_meta = _make_docker_job_meta()

        with patch.object(DockerJobHandle, "enter_states", return_value=False):
            handle = launcher.launch_job(job_meta, fl_ctx)

        assert handle is not None

    @patch("nvflare.app_opt.job_launcher.docker_launcher.docker")
    @patch("nvflare.app_opt.job_launcher.docker_launcher.os")
    def test_launch_job_terminates_on_enter_states_failure(self, mock_os, mock_docker):
        from nvflare.app_opt.job_launcher.docker_launcher import ClientDockerJobLauncher

        mock_os.environ.get.return_value = "/docker/workspace"
        container = Mock()
        mock_docker.from_env.return_value.containers.run.return_value = container

        launcher = ClientDockerJobLauncher(timeout=1)
        fl_ctx = _make_fl_ctx_for_docker_launch()
        job_meta = _make_docker_job_meta()

        with patch.object(DockerJobHandle, "enter_states", return_value=False) as mock_enter:
            handle = launcher.launch_job(job_meta, fl_ctx)

        container.stop.assert_called_once()

    @patch("nvflare.app_opt.job_launcher.docker_launcher.docker")
    @patch("nvflare.app_opt.job_launcher.docker_launcher.os")
    def test_launch_job_returns_handle_on_enter_states_exception(self, mock_os, mock_docker):
        from nvflare.app_opt.job_launcher.docker_launcher import ClientDockerJobLauncher

        mock_os.environ.get.return_value = "/docker/workspace"
        container = Mock()
        mock_docker.from_env.return_value.containers.run.return_value = container

        launcher = ClientDockerJobLauncher(timeout=1)
        fl_ctx = _make_fl_ctx_for_docker_launch()
        job_meta = _make_docker_job_meta()

        with patch.object(DockerJobHandle, "enter_states", side_effect=RuntimeError("boom")):
            handle = launcher.launch_job(job_meta, fl_ctx)

        assert handle is not None
        container.stop.assert_called_once()

    @patch("nvflare.app_opt.job_launcher.docker_launcher.docker")
    @patch("nvflare.app_opt.job_launcher.docker_launcher.os")
    def test_launch_job_returns_handle_on_image_not_found(self, mock_os, mock_docker):
        import docker as docker_pkg
        from nvflare.app_opt.job_launcher.docker_launcher import ClientDockerJobLauncher

        mock_os.environ.get.return_value = "/docker/workspace"
        mock_docker.from_env.return_value.containers.run.side_effect = docker_pkg.errors.ImageNotFound("not found")
        mock_docker.errors = docker_pkg.errors

        launcher = ClientDockerJobLauncher()
        fl_ctx = _make_fl_ctx_for_docker_launch()
        job_meta = _make_docker_job_meta(image="bad/image:latest")

        handle = launcher.launch_job(job_meta, fl_ctx)
        assert handle is not None
        assert handle.container is None

    @patch("nvflare.app_opt.job_launcher.docker_launcher.docker")
    @patch("nvflare.app_opt.job_launcher.docker_launcher.os")
    def test_launch_job_returns_handle_on_api_error(self, mock_os, mock_docker):
        import docker as docker_pkg
        from nvflare.app_opt.job_launcher.docker_launcher import ClientDockerJobLauncher

        mock_os.environ.get.return_value = "/docker/workspace"
        mock_docker.from_env.return_value.containers.run.side_effect = docker_pkg.errors.APIError("api error")
        mock_docker.errors = docker_pkg.errors

        launcher = ClientDockerJobLauncher()
        fl_ctx = _make_fl_ctx_for_docker_launch()
        job_meta = _make_docker_job_meta()

        handle = launcher.launch_job(job_meta, fl_ctx)
        assert handle is not None
        assert handle.container is None

    @patch("nvflare.app_opt.job_launcher.docker_launcher.docker")
    @patch("nvflare.app_opt.job_launcher.docker_launcher.os")
    def test_launch_job_empty_custom_folder_uses_pythonpath_only(self, mock_os, mock_docker):
        from nvflare.app_opt.job_launcher.docker_launcher import ClientDockerJobLauncher

        mock_os.environ.get.return_value = "/docker/workspace"
        container = Mock()
        mock_docker.from_env.return_value.containers.run.return_value = container

        launcher = ClientDockerJobLauncher()
        fl_ctx = FLContext()
        fl_ctx.set_prop(ReservedKey.IDENTITY_NAME, "client-1", private=False, sticky=True)
        workspace_obj = Mock()
        workspace_obj.get_app_custom_dir.return_value = ""
        fl_ctx.set_prop(FLContextKey.WORKSPACE_OBJECT, workspace_obj, private=True, sticky=False)
        fl_ctx.set_prop(
            FLContextKey.JOB_PROCESS_ARGS,
            {
                "exe_module": ("-m", "worker"),
                "workspace": ("-w", "/workspace"),
            },
            private=True,
            sticky=False,
        )

        job_meta = _make_docker_job_meta()

        with patch.object(DockerJobHandle, "enter_states", return_value=True):
            handle = launcher.launch_job(job_meta, fl_ctx)

        call_kwargs = mock_docker.from_env.return_value.containers.run.call_args
        command_str = call_kwargs[1]["command"] if "command" in call_kwargs[1] else call_kwargs[0][1]
        assert "$PYTHONPATH" in command_str
        assert "/custom" not in command_str


# ---------------------------------------------------------------------------
# ClientDockerJobLauncher.get_command
# ---------------------------------------------------------------------------
class TestClientDockerJobLauncher:
    @patch("nvflare.app_opt.job_launcher.docker_launcher.generate_client_command")
    def test_get_command(self, mock_gen_cmd):
        from nvflare.app_opt.job_launcher.docker_launcher import ClientDockerJobLauncher

        mock_gen_cmd.return_value = "python -u -m worker_process -w /workspace"
        launcher = ClientDockerJobLauncher()
        fl_ctx = FLContext()
        fl_ctx.set_prop(ReservedKey.IDENTITY_NAME, "client-1", private=False, sticky=True)
        job_meta = {JobConstants.JOB_ID: "job-abc"}

        name, cmd = launcher.get_command(job_meta, fl_ctx)
        assert name == "client-1-job-abc"
        assert cmd == "python -u -m worker_process -w /workspace"


# ---------------------------------------------------------------------------
# ServerDockerJobLauncher.get_command
# ---------------------------------------------------------------------------
class TestServerDockerJobLauncher:
    @patch("nvflare.app_opt.job_launcher.docker_launcher.generate_server_command")
    def test_get_command(self, mock_gen_cmd):
        from nvflare.app_opt.job_launcher.docker_launcher import ServerDockerJobLauncher

        mock_gen_cmd.return_value = "python -u -m server_process -w /workspace"
        launcher = ServerDockerJobLauncher()
        fl_ctx = FLContext()
        job_meta = {JobConstants.JOB_ID: "job-xyz"}

        name, cmd = launcher.get_command(job_meta, fl_ctx)
        assert name == "server-job-xyz"
        assert cmd == "python -u -m server_process -w /workspace"
