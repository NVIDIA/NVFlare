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
from unittest.mock import MagicMock, Mock, patch

import pytest

_k8s_mock = ModuleType("kubernetes")
_k8s_client = ModuleType("kubernetes.client")
_k8s_config = ModuleType("kubernetes.config")
_k8s_rest = ModuleType("kubernetes.client.rest")
_k8s_api = ModuleType("kubernetes.client.api")
_k8s_core = ModuleType("kubernetes.client.api.core_v1_api")


class _FakeApiException(Exception):
    def __init__(self, status=None, reason=None, http_resp=None):
        self.status = status
        self.reason = reason


_k8s_rest.ApiException = _FakeApiException
_k8s_client.Configuration = MagicMock
_k8s_client.rest = _k8s_rest
_k8s_client.api = _k8s_api
_k8s_core.CoreV1Api = MagicMock
_k8s_api.core_v1_api = _k8s_core
_k8s_mock.config = _k8s_config
_k8s_mock.client = _k8s_client

for _mod_name, _mod_obj in [
    ("kubernetes", _k8s_mock),
    ("kubernetes.config", _k8s_config),
    ("kubernetes.client", _k8s_client),
    ("kubernetes.client.rest", _k8s_rest),
    ("kubernetes.client.api", _k8s_api),
    ("kubernetes.client.api.core_v1_api", _k8s_core),
]:
    sys.modules.setdefault(_mod_name, _mod_obj)

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import JobMetaKey
from nvflare.apis.job_launcher_spec import JobProcessArgs, JobReturnCode
from nvflare.app_opt.job_launcher.k8s_launcher import (
    DEFAULT_CONTAINER_ARGS_MODULE_ARGS_DICT,
    JOB_RETURN_CODE_MAPPING,
    POD_STATE_MAPPING,
    PV_NAME,
    VOLUME_MOUNT_LIST,
    JobState,
    K8sJobHandle,
    POD_Phase,
    _job_args_dict,
)


def _make_job_config(**overrides):
    cfg = {
        "name": "test-job-123",
        "image": "nvflare/nvflare:test",
        "container_name": "container-test-job-123",
        "command": "nvflare.private.fed.app.client.worker_process",
        "volume_mount_list": VOLUME_MOUNT_LIST,
        "volume_list": [
            {"name": PV_NAME.WORKSPACE.value, "persistentVolumeClaim": {"claimName": "ws-pvc"}},
            {"name": PV_NAME.DATA.value, "persistentVolumeClaim": {"claimName": "data-pvc"}},
            {"name": PV_NAME.ETC.value, "persistentVolumeClaim": {"claimName": "etc-pvc"}},
        ],
        "module_args": {"-m": "val_m", "-w": "val_w"},
        "set_list": ["key1=val1", "key2=val2"],
        "resources": {"limits": {"nvidia.com/gpu": 1}},
    }
    cfg.update(overrides)
    return cfg


def _make_api_instance():
    return MagicMock()


# ---------------------------------------------------------------------------
# Mapping tables
# ---------------------------------------------------------------------------
class TestPodStateMapping:
    def test_all_phases_mapped(self):
        for phase in POD_Phase:
            assert phase.value in POD_STATE_MAPPING

    def test_pending_maps_to_starting(self):
        assert POD_STATE_MAPPING[POD_Phase.PENDING.value] == JobState.STARTING

    def test_running_maps_to_running(self):
        assert POD_STATE_MAPPING[POD_Phase.RUNNING.value] == JobState.RUNNING

    def test_succeeded_maps_to_succeeded(self):
        assert POD_STATE_MAPPING[POD_Phase.SUCCEEDED.value] == JobState.SUCCEEDED

    def test_failed_maps_to_terminated(self):
        assert POD_STATE_MAPPING[POD_Phase.FAILED.value] == JobState.TERMINATED


class TestJobReturnCodeMapping:
    def test_all_job_states_mapped(self):
        for state in JobState:
            assert state in JOB_RETURN_CODE_MAPPING

    def test_succeeded_maps_to_success(self):
        assert JOB_RETURN_CODE_MAPPING[JobState.SUCCEEDED] == JobReturnCode.SUCCESS

    def test_terminated_maps_to_aborted(self):
        assert JOB_RETURN_CODE_MAPPING[JobState.TERMINATED] == JobReturnCode.ABORTED

    def test_running_maps_to_unknown(self):
        assert JOB_RETURN_CODE_MAPPING[JobState.RUNNING] == JobReturnCode.UNKNOWN


# ---------------------------------------------------------------------------
# K8sJobHandle
# ---------------------------------------------------------------------------
class TestK8sJobHandle:
    # -- construction ---------------------------------------------------------
    def test_init_raises_on_missing_command(self):
        cfg = _make_job_config()
        del cfg["command"]
        with pytest.raises(ValueError, match="command"):
            K8sJobHandle("job-1", _make_api_instance(), cfg)

    def test_init_raises_on_empty_command(self):
        cfg = _make_job_config(command="")
        with pytest.raises(ValueError, match="command"):
            K8sJobHandle("job-1", _make_api_instance(), cfg)

    def test_stuck_count_starts_at_zero(self):
        cfg = _make_job_config()
        handle = K8sJobHandle("job-1", _make_api_instance(), cfg, timeout=30)
        assert handle._stuck_count == 0

    def test_max_stuck_count_includes_grace_period(self):
        cfg = _make_job_config()
        handle = K8sJobHandle("job-1", _make_api_instance(), cfg, timeout=30)
        assert handle._max_stuck_count == 30 + handle._stuck_grace_period

    def test_max_stuck_count_is_none_with_no_timeout(self):
        cfg = _make_job_config()
        handle = K8sJobHandle("job-1", _make_api_instance(), cfg, timeout=None)
        assert handle._max_stuck_count is None

    # -- manifest -------------------------------------------------------------
    def test_manifest_metadata_name(self):
        cfg = _make_job_config()
        handle = K8sJobHandle("job-1", _make_api_instance(), cfg)
        assert handle.get_manifest()["metadata"]["name"] == "test-job-123"

    def test_manifest_container_image(self):
        cfg = _make_job_config()
        handle = K8sJobHandle("job-1", _make_api_instance(), cfg)
        container = handle.get_manifest()["spec"]["containers"][0]
        assert container["image"] == "nvflare/nvflare:test"

    def test_manifest_container_name(self):
        cfg = _make_job_config()
        handle = K8sJobHandle("job-1", _make_api_instance(), cfg)
        container = handle.get_manifest()["spec"]["containers"][0]
        assert container["name"] == "container-test-job-123"

    def test_manifest_default_image(self):
        cfg = _make_job_config()
        del cfg["image"]
        handle = K8sJobHandle("job-1", _make_api_instance(), cfg)
        container = handle.get_manifest()["spec"]["containers"][0]
        assert container["image"] == "nvflare/nvflare:2.8.0"

    def test_manifest_default_container_name(self):
        cfg = _make_job_config()
        del cfg["container_name"]
        handle = K8sJobHandle("job-1", _make_api_instance(), cfg)
        container = handle.get_manifest()["spec"]["containers"][0]
        assert container["name"] == "nvflare_job"

    def test_manifest_restart_policy(self):
        cfg = _make_job_config()
        handle = K8sJobHandle("job-1", _make_api_instance(), cfg)
        assert handle.get_manifest()["spec"]["restartPolicy"] == "Never"

    def test_manifest_volumes(self):
        cfg = _make_job_config()
        handle = K8sJobHandle("job-1", _make_api_instance(), cfg)
        volumes = handle.get_manifest()["spec"]["volumes"]
        assert len(volumes) == 3
        pvc_names = [v["persistentVolumeClaim"]["claimName"] for v in volumes]
        assert "ws-pvc" in pvc_names
        assert "data-pvc" in pvc_names
        assert "etc-pvc" in pvc_names

    def test_manifest_volume_mounts(self):
        cfg = _make_job_config()
        handle = K8sJobHandle("job-1", _make_api_instance(), cfg)
        container = handle.get_manifest()["spec"]["containers"][0]
        assert container["volumeMounts"] == VOLUME_MOUNT_LIST

    def test_manifest_args_contain_command(self):
        cfg = _make_job_config()
        handle = K8sJobHandle("job-1", _make_api_instance(), cfg)
        args = handle.get_manifest()["spec"]["containers"][0]["args"]
        assert "-u" in args
        assert "-m" in args
        assert "nvflare.private.fed.app.client.worker_process" in args

    def test_manifest_args_contain_module_args(self):
        cfg = _make_job_config()
        handle = K8sJobHandle("job-1", _make_api_instance(), cfg)
        args = handle.get_manifest()["spec"]["containers"][0]["args"]
        assert "val_m" in args
        assert "val_w" in args

    def test_manifest_args_contain_set_list(self):
        cfg = _make_job_config()
        handle = K8sJobHandle("job-1", _make_api_instance(), cfg)
        args = handle.get_manifest()["spec"]["containers"][0]["args"]
        assert "--set" in args
        assert "key1=val1" in args
        assert "key2=val2" in args

    def test_manifest_no_set_list(self):
        cfg = _make_job_config()
        cfg["set_list"] = None
        handle = K8sJobHandle("job-1", _make_api_instance(), cfg)
        args = handle.get_manifest()["spec"]["containers"][0]["args"]
        assert "--set" not in args

    def test_manifest_none_module_args_skipped(self):
        cfg = _make_job_config(module_args={"-a": "keep", "-b": None})
        handle = K8sJobHandle("job-1", _make_api_instance(), cfg)
        args = handle.get_manifest()["spec"]["containers"][0]["args"]
        assert "-a" in args
        assert "keep" in args
        assert "-b" not in args

    def test_manifest_default_module_args_copies_dict(self):
        cfg = _make_job_config()
        cfg["module_args"] = None
        handle = K8sJobHandle("job-1", _make_api_instance(), cfg)
        assert handle.container_args_module_args_dict is not DEFAULT_CONTAINER_ARGS_MODULE_ARGS_DICT
        assert handle.container_args_module_args_dict == DEFAULT_CONTAINER_ARGS_MODULE_ARGS_DICT

    def test_manifest_default_module_args_all_none_produces_empty_args_list(self):
        cfg = _make_job_config()
        cfg["module_args"] = None
        handle = K8sJobHandle("job-1", _make_api_instance(), cfg)
        assert handle.container_args_module_args_dict_as_list == []

    def test_manifest_gpu_resources(self):
        cfg = _make_job_config()
        handle = K8sJobHandle("job-1", _make_api_instance(), cfg)
        container = handle.get_manifest()["spec"]["containers"][0]
        assert container["resources"]["limits"]["nvidia.com/gpu"] == 1

    def test_manifest_no_gpu_resources(self):
        cfg = _make_job_config(resources={"limits": {"nvidia.com/gpu": None}})
        handle = K8sJobHandle("job-1", _make_api_instance(), cfg)
        container = handle.get_manifest()["spec"]["containers"][0]
        assert container["resources"] is None

    # -- poll -----------------------------------------------------------------
    def test_poll_returns_unknown_when_running(self):
        api = _make_api_instance()
        resp = Mock()
        resp.status.phase = POD_Phase.RUNNING.value
        api.read_namespaced_pod.return_value = resp
        handle = K8sJobHandle("job-1", api, _make_job_config())
        assert handle.poll() == JobReturnCode.UNKNOWN

    def test_poll_returns_success_when_succeeded(self):
        api = _make_api_instance()
        resp = Mock()
        resp.status.phase = POD_Phase.SUCCEEDED.value
        api.read_namespaced_pod.return_value = resp
        handle = K8sJobHandle("job-1", api, _make_job_config())
        assert handle.poll() == JobReturnCode.SUCCESS

    def test_poll_returns_aborted_when_failed(self):
        api = _make_api_instance()
        resp = Mock()
        resp.status.phase = POD_Phase.FAILED.value
        api.read_namespaced_pod.return_value = resp
        handle = K8sJobHandle("job-1", api, _make_job_config())
        assert handle.poll() == JobReturnCode.ABORTED

    def test_poll_uses_terminal_state_if_set(self):
        api = _make_api_instance()
        handle = K8sJobHandle("job-1", api, _make_job_config())
        handle.terminal_state = JobState.TERMINATED
        assert handle.poll() == JobReturnCode.ABORTED
        api.read_namespaced_pod.assert_not_called()

    # -- terminate ------------------------------------------------------------
    def test_terminate_deletes_pod_and_sets_terminated(self):
        api = _make_api_instance()
        handle = K8sJobHandle("job-1", api, _make_job_config())
        handle.terminate()
        api.delete_namespaced_pod.assert_called_once_with(name="job-1", namespace="default", grace_period_seconds=0)
        assert handle.terminal_state == JobState.TERMINATED

    def test_terminate_sets_terminated_on_404(self):
        api = _make_api_instance()
        api.delete_namespaced_pod.side_effect = _FakeApiException(status=404, reason="Not Found")
        handle = K8sJobHandle("job-1", api, _make_job_config())
        handle.terminate()
        assert handle.terminal_state == JobState.TERMINATED

    def test_terminate_does_not_set_state_on_non_404_error(self):
        api = _make_api_instance()
        api.delete_namespaced_pod.side_effect = _FakeApiException(status=500, reason="Internal")
        handle = K8sJobHandle("job-1", api, _make_job_config())
        handle.terminate()
        assert handle.terminal_state is None

    def test_terminate_custom_namespace(self):
        api = _make_api_instance()
        handle = K8sJobHandle("job-1", api, _make_job_config(), namespace="custom-ns")
        handle.terminate()
        api.delete_namespaced_pod.assert_called_once_with(name="job-1", namespace="custom-ns", grace_period_seconds=0)

    # -- _query_phase ---------------------------------------------------------
    def test_query_phase_returns_unknown_on_api_error(self):
        api = _make_api_instance()
        api.read_namespaced_pod.side_effect = _FakeApiException(status=500, reason="Error")
        handle = K8sJobHandle("job-1", api, _make_job_config())
        assert handle._query_phase() == POD_Phase.UNKNOWN.value

    # -- _stuck ---------------------------------------------------------------
    def test_stuck_returns_false_when_no_timeout_and_grace_only(self):
        api = _make_api_instance()
        handle = K8sJobHandle("job-1", api, _make_job_config(), timeout=None)
        assert handle._stuck(POD_Phase.PENDING.value) is False

    def test_stuck_returns_true_after_max_count_with_grace(self):
        api = _make_api_instance()
        handle = K8sJobHandle("job-1", api, _make_job_config(), timeout=5)
        handle._stuck_count = handle._max_stuck_count
        assert handle._stuck(POD_Phase.PENDING.value) is True

    def test_stuck_returns_false_for_non_pending(self):
        api = _make_api_instance()
        handle = K8sJobHandle("job-1", api, _make_job_config(), timeout=5)
        handle._stuck_count = 9999
        assert handle._stuck(POD_Phase.RUNNING.value) is False

    def test_stuck_increments_count_on_pending(self):
        api = _make_api_instance()
        handle = K8sJobHandle("job-1", api, _make_job_config(), timeout=100)
        initial = handle._stuck_count
        handle._stuck(POD_Phase.PENDING.value)
        assert handle._stuck_count == initial + 1

    # -- wait -----------------------------------------------------------------
    def test_wait_returns_immediately_if_terminal_state_set(self):
        api = _make_api_instance()
        handle = K8sJobHandle("job-1", api, _make_job_config())
        handle.terminal_state = JobState.TERMINATED
        handle.wait()
        api.read_namespaced_pod.assert_not_called()

    def test_wait_persists_succeeded_terminal_state(self):
        api = _make_api_instance()
        resp = Mock()
        resp.status.phase = POD_Phase.SUCCEEDED.value
        api.read_namespaced_pod.return_value = resp
        handle = K8sJobHandle("job-1", api, _make_job_config())
        handle.wait()
        assert handle.terminal_state == JobState.SUCCEEDED

    def test_wait_persists_terminated_terminal_state(self):
        api = _make_api_instance()
        resp = Mock()
        resp.status.phase = POD_Phase.FAILED.value
        api.read_namespaced_pod.return_value = resp
        handle = K8sJobHandle("job-1", api, _make_job_config())
        handle.wait()
        assert handle.terminal_state == JobState.TERMINATED

    def test_wait_poll_consistent_after_wait(self):
        api = _make_api_instance()
        resp = Mock()
        resp.status.phase = POD_Phase.SUCCEEDED.value
        api.read_namespaced_pod.return_value = resp
        handle = K8sJobHandle("job-1", api, _make_job_config())
        handle.wait()
        assert handle.poll() == JobReturnCode.SUCCESS

    # -- enter_states ---------------------------------------------------------
    def test_enter_states_returns_true_when_state_matches(self):
        api = _make_api_instance()
        resp = Mock()
        resp.status.phase = POD_Phase.RUNNING.value
        api.read_namespaced_pod.return_value = resp
        handle = K8sJobHandle("job-1", api, _make_job_config())
        assert handle.enter_states([JobState.RUNNING]) is True

    def test_enter_states_accepts_single_state(self):
        api = _make_api_instance()
        resp = Mock()
        resp.status.phase = POD_Phase.SUCCEEDED.value
        api.read_namespaced_pod.return_value = resp
        handle = K8sJobHandle("job-1", api, _make_job_config())
        assert handle.enter_states(JobState.SUCCEEDED) is True

    def test_enter_states_returns_false_on_timeout(self):
        api = _make_api_instance()
        resp = Mock()
        resp.status.phase = POD_Phase.PENDING.value
        api.read_namespaced_pod.return_value = resp
        handle = K8sJobHandle("job-1", api, _make_job_config(), timeout=None)
        assert handle.enter_states([JobState.RUNNING], timeout=0) is False

    def test_enter_states_raises_on_invalid_state(self):
        api = _make_api_instance()
        handle = K8sJobHandle("job-1", api, _make_job_config())
        with pytest.raises(ValueError, match="expect job_states_to_enter"):
            handle.enter_states(["not_a_state"])


# ---------------------------------------------------------------------------
# _job_args_dict helper
# ---------------------------------------------------------------------------
class TestJobArgsDict:
    def test_basic(self):
        job_args = {
            "workspace": ("-w", "/workspace"),
            "job_id": ("-j", "job-1"),
        }
        result = _job_args_dict(job_args, ["workspace", "job_id"])
        assert result == {"-w": "/workspace", "-j": "job-1"}

    def test_skips_missing_keys(self):
        job_args = {"workspace": ("-w", "/workspace")}
        result = _job_args_dict(job_args, ["workspace", "missing_key"])
        assert result == {"-w": "/workspace"}

    def test_empty_args(self):
        assert _job_args_dict({}, ["a", "b"]) == {}

    def test_empty_arg_names(self):
        assert _job_args_dict({"workspace": ("-w", "/workspace")}, []) == {}


# ---------------------------------------------------------------------------
# K8sJobLauncher handle_event
# ---------------------------------------------------------------------------
def _make_k8s_launcher_patches():
    return [
        patch("nvflare.app_opt.job_launcher.k8s_launcher.config"),
        patch("nvflare.app_opt.job_launcher.k8s_launcher.Configuration"),
        patch("nvflare.app_opt.job_launcher.k8s_launcher.core_v1_api"),
        patch("builtins.open", create=True),
        patch("nvflare.app_opt.job_launcher.k8s_launcher.yaml"),
    ]


def _enter_patches(patches):
    mocks = [p.start() for p in patches]
    return mocks


def _exit_patches(patches):
    for p in patches:
        p.stop()


def _setup_launcher(mock_yaml, mock_conf, launcher_cls):
    mock_yaml.safe_load.return_value = {"data-pvc": "/data"}
    mock_conf_instance = MagicMock()
    mock_conf.return_value = mock_conf_instance
    mock_conf.get_default_copy = Mock(return_value=mock_conf_instance)
    return launcher_cls(
        config_file_path="/fake/kube/config",
        workspace_pvc="ws-pvc",
        etc_pvc="etc-pvc",
        data_pvc_file_path="/fake/data_pvc.yaml",
    )


class TestK8sJobLauncherHandleEvent:
    def test_adds_launcher_when_image_present(self):
        from nvflare.app_opt.job_launcher.k8s_launcher import ClientK8sJobLauncher

        patches = _make_k8s_launcher_patches()
        mock_cfg, mock_conf, mock_core, mock_open, mock_yaml = _enter_patches(patches)
        try:
            launcher = _setup_launcher(mock_yaml, mock_conf, ClientK8sJobLauncher)
            fl_ctx = FLContext()
            job_meta = {JobMetaKey.DEPLOY_MAP.value: {"app": [{"sites": ["site-1"], "image": "nvflare/custom:latest"}]}}
            fl_ctx.set_prop(FLContextKey.JOB_META, job_meta, private=True, sticky=False)
            fl_ctx.set_prop(ReservedKey.IDENTITY_NAME, "site-1", private=False, sticky=True)

            launcher.handle_event(EventType.BEFORE_JOB_LAUNCH, fl_ctx)

            launchers = fl_ctx.get_prop(FLContextKey.JOB_LAUNCHER)
            assert launchers is not None
            assert launcher in launchers
        finally:
            _exit_patches(patches)

    def test_skips_when_no_image(self):
        from nvflare.app_opt.job_launcher.k8s_launcher import ClientK8sJobLauncher

        patches = _make_k8s_launcher_patches()
        mock_cfg, mock_conf, mock_core, mock_open, mock_yaml = _enter_patches(patches)
        try:
            launcher = _setup_launcher(mock_yaml, mock_conf, ClientK8sJobLauncher)
            fl_ctx = FLContext()
            job_meta = {JobMetaKey.DEPLOY_MAP.value: {}}
            fl_ctx.set_prop(FLContextKey.JOB_META, job_meta, private=True, sticky=False)
            fl_ctx.set_prop(ReservedKey.IDENTITY_NAME, "site-1", private=False, sticky=True)

            launcher.handle_event(EventType.BEFORE_JOB_LAUNCH, fl_ctx)

            assert fl_ctx.get_prop(FLContextKey.JOB_LAUNCHER) is None
        finally:
            _exit_patches(patches)


# ---------------------------------------------------------------------------
# K8sJobLauncher __init__
# ---------------------------------------------------------------------------
class TestK8sJobLauncherInit:
    def test_init_reads_data_pvc(self):
        from nvflare.app_opt.job_launcher.k8s_launcher import ClientK8sJobLauncher

        patches = _make_k8s_launcher_patches()
        mock_cfg, mock_conf, mock_core, mock_open, mock_yaml = _enter_patches(patches)
        try:
            mock_yaml.safe_load.return_value = {"my-data-pvc": "/mount/data"}
            mock_conf_instance = MagicMock()
            mock_conf.return_value = mock_conf_instance
            mock_conf.get_default_copy = Mock(return_value=mock_conf_instance)

            launcher = ClientK8sJobLauncher(
                config_file_path="/fake/kube/config",
                workspace_pvc="ws-pvc",
                etc_pvc="etc-pvc",
                data_pvc_file_path="/fake/data_pvc.yaml",
                timeout=60,
                namespace="test-ns",
            )

            assert launcher.workspace_pvc == "ws-pvc"
            assert launcher.etc_pvc == "etc-pvc"
            assert launcher.data_pvc == "my-data-pvc"
            assert launcher.timeout == 60
            assert launcher.namespace == "test-ns"
        finally:
            _exit_patches(patches)

    def test_init_raises_on_empty_pvc_file(self):
        from nvflare.app_opt.job_launcher.k8s_launcher import ClientK8sJobLauncher

        patches = _make_k8s_launcher_patches()
        mock_cfg, mock_conf, mock_core, mock_open, mock_yaml = _enter_patches(patches)
        try:
            mock_yaml.safe_load.return_value = {}
            mock_conf_instance = MagicMock()
            mock_conf.return_value = mock_conf_instance
            mock_conf.get_default_copy = Mock(return_value=mock_conf_instance)

            with pytest.raises(ValueError, match="empty"):
                ClientK8sJobLauncher(
                    config_file_path="/fake/kube/config",
                    workspace_pvc="ws-pvc",
                    etc_pvc="etc-pvc",
                    data_pvc_file_path="/fake/data_pvc.yaml",
                )
        finally:
            _exit_patches(patches)


# ---------------------------------------------------------------------------
# ClientK8sJobLauncher.get_module_args
# ---------------------------------------------------------------------------
class TestClientK8sJobLauncherGetModuleArgs:
    def test_returns_dict(self):
        from nvflare.app_opt.job_launcher.k8s_launcher import ClientK8sJobLauncher

        patches = _make_k8s_launcher_patches()
        mock_cfg, mock_conf, mock_core, mock_open, mock_yaml = _enter_patches(patches)
        try:
            launcher = _setup_launcher(mock_yaml, mock_conf, ClientK8sJobLauncher)
            fl_ctx = FLContext()
            job_args = {
                JobProcessArgs.WORKSPACE: ("-w", "/workspace"),
                JobProcessArgs.JOB_ID: ("-j", "job-1"),
            }
            fl_ctx.set_prop(FLContextKey.JOB_PROCESS_ARGS, job_args, private=True, sticky=False)

            result = launcher.get_module_args("job-1", fl_ctx)
            assert isinstance(result, dict)
            assert result.get("-w") == "/workspace"
        finally:
            _exit_patches(patches)

    def test_raises_when_no_args(self):
        from nvflare.app_opt.job_launcher.k8s_launcher import ClientK8sJobLauncher

        patches = _make_k8s_launcher_patches()
        mock_cfg, mock_conf, mock_core, mock_open, mock_yaml = _enter_patches(patches)
        try:
            launcher = _setup_launcher(mock_yaml, mock_conf, ClientK8sJobLauncher)
            fl_ctx = FLContext()
            with pytest.raises(RuntimeError, match="job_process_args"):
                launcher.get_module_args("job-1", fl_ctx)
        finally:
            _exit_patches(patches)


# ---------------------------------------------------------------------------
# ServerK8sJobLauncher.get_module_args
# ---------------------------------------------------------------------------
class TestServerK8sJobLauncherGetModuleArgs:
    def test_returns_dict(self):
        from nvflare.app_opt.job_launcher.k8s_launcher import ServerK8sJobLauncher

        patches = _make_k8s_launcher_patches()
        mock_cfg, mock_conf, mock_core, mock_open, mock_yaml = _enter_patches(patches)
        try:
            launcher = _setup_launcher(mock_yaml, mock_conf, ServerK8sJobLauncher)
            fl_ctx = FLContext()
            job_args = {
                JobProcessArgs.WORKSPACE: ("-w", "/workspace"),
                JobProcessArgs.JOB_ID: ("-j", "job-1"),
                JobProcessArgs.ROOT_URL: ("--root_url", "grpc://server:8003"),
            }
            fl_ctx.set_prop(FLContextKey.JOB_PROCESS_ARGS, job_args, private=True, sticky=False)

            result = launcher.get_module_args("job-1", fl_ctx)
            assert isinstance(result, dict)
            assert result.get("-w") == "/workspace"
        finally:
            _exit_patches(patches)

    def test_raises_when_no_args(self):
        from nvflare.app_opt.job_launcher.k8s_launcher import ServerK8sJobLauncher

        patches = _make_k8s_launcher_patches()
        mock_cfg, mock_conf, mock_core, mock_open, mock_yaml = _enter_patches(patches)
        try:
            launcher = _setup_launcher(mock_yaml, mock_conf, ServerK8sJobLauncher)
            fl_ctx = FLContext()
            with pytest.raises(RuntimeError, match="job_process_args"):
                launcher.get_module_args("job-1", fl_ctx)
        finally:
            _exit_patches(patches)
