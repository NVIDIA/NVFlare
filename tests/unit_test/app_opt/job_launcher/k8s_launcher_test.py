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
from nvflare.apis.fl_constant import FLContextKey, JobConstants, ReservedKey
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
    PodPhase,
    _job_args_dict,
    uuid4_to_rfc1123,
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


def _make_handle(job_id="job-1", api=None, cfg=None, **kwargs):
    if api is None:
        api = _make_api_instance()
    if cfg is None:
        cfg = _make_job_config()
    handle = K8sJobHandle(job_id, api, cfg, **kwargs)
    handle.job_id = job_id
    return handle


# ---------------------------------------------------------------------------
# uuid4_to_rfc1123
# ---------------------------------------------------------------------------
class TestUuid4ToRfc1123:
    def test_lowercase(self):
        assert uuid4_to_rfc1123("ABCD-1234") == "abcd-1234"

    def test_strips_invalid_chars(self):
        assert uuid4_to_rfc1123("abc_def.ghi") == "abcdefghi"

    def test_prefixes_leading_digit(self):
        result = uuid4_to_rfc1123("1234-abcd")
        assert result[0] == "j"
        assert result == "j1234-abcd"

    def test_strips_trailing_hyphens(self):
        assert uuid4_to_rfc1123("abc-") == "abc"

    def test_strips_trailing_hyphen_exposed_by_truncation(self):
        # 62 'a's followed by '-' followed by more chars: truncation exposes the hyphen
        name = "a" * 62 + "-" + "b" * 10
        result = uuid4_to_rfc1123(name)
        assert not result.endswith("-"), f"trailing hyphen in {result!r}"
        assert len(result) == 62

    def test_truncates_to_63_chars(self):
        long_str = "a" * 100
        assert len(uuid4_to_rfc1123(long_str)) == 63

    def test_typical_uuid_gets_prefixed(self):
        result = uuid4_to_rfc1123("550e8400-e29b-41d4-a716-446655440000")
        assert result == "j550e8400-e29b-41d4-a716-446655440000"

    def test_letter_leading_uuid_no_prefix(self):
        result = uuid4_to_rfc1123("abcd1234-e29b-41d4-a716-446655440000")
        assert result == "abcd1234-e29b-41d4-a716-446655440000"

    def test_empty_string(self):
        assert uuid4_to_rfc1123("") == ""


# ---------------------------------------------------------------------------
# Mapping tables
# ---------------------------------------------------------------------------
class TestPodStateMapping:
    def test_all_phases_mapped(self):
        for phase in PodPhase:
            assert phase.value in POD_STATE_MAPPING

    def test_pending_maps_to_starting(self):
        assert POD_STATE_MAPPING[PodPhase.PENDING.value] == JobState.STARTING

    def test_running_maps_to_running(self):
        assert POD_STATE_MAPPING[PodPhase.RUNNING.value] == JobState.RUNNING

    def test_succeeded_maps_to_succeeded(self):
        assert POD_STATE_MAPPING[PodPhase.SUCCEEDED.value] == JobState.SUCCEEDED

    def test_failed_maps_to_terminated(self):
        assert POD_STATE_MAPPING[PodPhase.FAILED.value] == JobState.TERMINATED


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

    def test_max_stuck_count_equals_timeout(self):
        cfg = _make_job_config()
        handle = K8sJobHandle("job-1", _make_api_instance(), cfg, timeout=30)
        assert handle._max_stuck_count == 30

    def test_max_stuck_count_uses_pending_timeout_when_no_timeout(self):
        cfg = _make_job_config()
        handle = K8sJobHandle("job-1", _make_api_instance(), cfg, timeout=None)
        assert handle._max_stuck_count == 30

    def test_max_stuck_count_uses_custom_pending_timeout(self):
        cfg = _make_job_config()
        handle = K8sJobHandle("job-1", _make_api_instance(), cfg, timeout=None, pending_timeout=60)
        assert handle._max_stuck_count == 60

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

    def test_manifest_raises_on_missing_image(self):
        cfg = _make_job_config()
        del cfg["image"]
        with pytest.raises(ValueError, match="image"):
            K8sJobHandle("job-1", _make_api_instance(), cfg)

    def test_manifest_raises_on_empty_image(self):
        cfg = _make_job_config(image="")
        with pytest.raises(ValueError, match="image"):
            K8sJobHandle("job-1", _make_api_instance(), cfg)

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

    def test_manifest_non_string_module_arg_values_are_stringified(self):
        cfg = _make_job_config(module_args={"-p": 8080, "-n": 42})
        handle = K8sJobHandle("job-1", _make_api_instance(), cfg)
        args = handle.get_manifest()["spec"]["containers"][0]["args"]
        assert "8080" in args
        assert "42" in args
        assert all(isinstance(a, str) for a in args), "all container args must be str"

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
        assert "resources" not in container

    def test_manifest_no_resources_key(self):
        cfg = _make_job_config()
        del cfg["resources"]
        handle = K8sJobHandle("job-1", _make_api_instance(), cfg)
        container = handle.get_manifest()["spec"]["containers"][0]
        assert "resources" not in container

    def test_get_manifest_returns_independent_copy(self):
        handle = K8sJobHandle("job-1", _make_api_instance(), _make_job_config())
        manifest = handle.get_manifest()
        # Mutate the returned copy at every mutable level
        manifest["metadata"]["name"] = "MUTATED"
        manifest["spec"]["containers"][0]["image"] = "MUTATED"
        manifest["spec"]["volumes"].clear()
        # Internal state must be unchanged
        internal = handle.pod_manifest
        assert internal["metadata"]["name"] == "test-job-123"
        assert internal["spec"]["containers"][0]["image"] == "nvflare/nvflare:test"
        assert len(internal["spec"]["volumes"]) > 0

    # -- poll -----------------------------------------------------------------
    def test_poll_returns_unknown_when_running(self):
        api = _make_api_instance()
        resp = Mock()
        resp.status.phase = PodPhase.RUNNING.value
        api.read_namespaced_pod.return_value = resp
        handle = _make_handle(api=api)
        assert handle.poll() == JobReturnCode.UNKNOWN

    def test_poll_returns_success_when_succeeded(self):
        api = _make_api_instance()
        resp = Mock()
        resp.status.phase = PodPhase.SUCCEEDED.value
        api.read_namespaced_pod.return_value = resp
        handle = _make_handle(api=api)
        assert handle.poll() == JobReturnCode.SUCCESS

    def test_poll_returns_aborted_when_failed(self):
        api = _make_api_instance()
        resp = Mock()
        resp.status.phase = PodPhase.FAILED.value
        api.read_namespaced_pod.return_value = resp
        handle = _make_handle(api=api)
        assert handle.poll() == JobReturnCode.ABORTED

    def test_poll_uses_terminal_state_if_set(self):
        api = _make_api_instance()
        handle = _make_handle(api=api)
        handle.terminal_state = JobState.TERMINATED
        assert handle.poll() == JobReturnCode.ABORTED
        api.read_namespaced_pod.assert_not_called()

    # -- terminate ------------------------------------------------------------
    def test_terminate_deletes_pod_and_sets_terminated(self):
        api = _make_api_instance()
        handle = _make_handle(api=api)
        handle.terminate()
        api.delete_namespaced_pod.assert_called_once_with(name="job-1", namespace="default", grace_period_seconds=0)
        assert handle.terminal_state == JobState.TERMINATED

    def test_terminate_sets_terminated_on_404(self):
        api = _make_api_instance()
        api.delete_namespaced_pod.side_effect = _FakeApiException(status=404, reason="Not Found")
        handle = _make_handle(api=api)
        handle.terminate()
        assert handle.terminal_state == JobState.TERMINATED

    def test_terminate_sets_terminated_on_non_404_api_error(self):
        api = _make_api_instance()
        api.delete_namespaced_pod.side_effect = _FakeApiException(status=500, reason="Internal")
        handle = _make_handle(api=api)
        handle.terminate()
        assert handle.terminal_state == JobState.TERMINATED

    def test_terminate_sets_terminated_on_network_error(self):
        api = _make_api_instance()
        api.delete_namespaced_pod.side_effect = ConnectionError("network unreachable")
        handle = _make_handle(api=api)
        handle.terminate()
        assert handle.terminal_state == JobState.TERMINATED

    def test_terminate_custom_namespace(self):
        api = _make_api_instance()
        handle = _make_handle(api=api, namespace="custom-ns")
        handle.terminate()
        api.delete_namespaced_pod.assert_called_once_with(name="job-1", namespace="custom-ns", grace_period_seconds=0)

    # -- _query_phase ---------------------------------------------------------
    def test_query_phase_returns_unknown_on_api_error(self):
        api = _make_api_instance()
        api.read_namespaced_pod.side_effect = _FakeApiException(status=500, reason="Error")
        handle = _make_handle(api=api)
        assert handle._query_phase() == PodPhase.UNKNOWN.value

    def test_query_phase_returns_unknown_on_generic_exception(self):
        api = _make_api_instance()
        api.read_namespaced_pod.side_effect = RuntimeError("connection lost")
        handle = _make_handle(api=api)
        assert handle._query_phase() == PodPhase.UNKNOWN.value

    # -- _stuck_in_pending ----------------------------------------------------
    def test_stuck_in_pending_returns_true_at_max_count(self):
        # With _stuck_count seeded to max-1, one more PENDING call increments to
        # exactly max, which should fire (>=, not >).
        api = _make_api_instance()
        handle = K8sJobHandle("job-1", api, _make_job_config(), timeout=5)
        handle._stuck_count = handle._max_stuck_count - 1
        assert handle._stuck_in_pending(PodPhase.PENDING.value) is True

    def test_stuck_in_pending_returns_false_one_before_max(self):
        # One iteration before the threshold must not fire.
        api = _make_api_instance()
        handle = K8sJobHandle("job-1", api, _make_job_config(), timeout=5)
        handle._stuck_count = handle._max_stuck_count - 2
        assert handle._stuck_in_pending(PodPhase.PENDING.value) is False

    def test_stuck_in_pending_returns_false_for_non_pending(self):
        api = _make_api_instance()
        handle = K8sJobHandle("job-1", api, _make_job_config(), timeout=5)
        handle._stuck_count = 9999
        assert handle._stuck_in_pending(PodPhase.RUNNING.value) is False

    def test_stuck_in_pending_resets_count_on_non_pending(self):
        api = _make_api_instance()
        handle = K8sJobHandle("job-1", api, _make_job_config(), timeout=5)
        handle._stuck_count = 3
        handle._stuck_in_pending(PodPhase.RUNNING.value)
        assert handle._stuck_count == 0

    def test_stuck_in_pending_increments_count(self):
        api = _make_api_instance()
        handle = K8sJobHandle("job-1", api, _make_job_config(), timeout=100)
        initial = handle._stuck_count
        handle._stuck_in_pending(PodPhase.PENDING.value)
        assert handle._stuck_count == initial + 1

    def test_stuck_in_pending_returns_false_when_under_max(self):
        api = _make_api_instance()
        handle = K8sJobHandle("job-1", api, _make_job_config(), timeout=None, pending_timeout=100)
        assert handle._stuck_in_pending(PodPhase.PENDING.value) is False

    def test_stuck_in_pending_never_fires_when_pending_timeout_none(self):
        # pending_timeout=None with timeout=None → _max_stuck_count=None → stuck detection disabled
        api = _make_api_instance()
        handle = K8sJobHandle("job-1", api, _make_job_config(), timeout=None, pending_timeout=None)
        assert handle._max_stuck_count is None
        # Drive _stuck_count very high — must not raise and must return False
        handle._stuck_count = 10_000
        assert handle._stuck_in_pending(PodPhase.PENDING.value) is False

    # -- wait -----------------------------------------------------------------
    def test_wait_returns_immediately_if_terminal_state_set(self):
        api = _make_api_instance()
        handle = _make_handle(api=api)
        handle.terminal_state = JobState.TERMINATED
        handle.wait()
        api.read_namespaced_pod.assert_not_called()

    def test_wait_persists_succeeded_terminal_state(self):
        api = _make_api_instance()
        resp = Mock()
        resp.status.phase = PodPhase.SUCCEEDED.value
        api.read_namespaced_pod.return_value = resp
        handle = _make_handle(api=api)
        handle.wait()
        assert handle.terminal_state == JobState.SUCCEEDED

    def test_wait_persists_terminated_terminal_state(self):
        api = _make_api_instance()
        resp = Mock()
        resp.status.phase = PodPhase.FAILED.value
        api.read_namespaced_pod.return_value = resp
        handle = _make_handle(api=api)
        handle.wait()
        assert handle.terminal_state == JobState.TERMINATED

    def test_wait_poll_consistent_after_wait(self):
        api = _make_api_instance()
        resp = Mock()
        resp.status.phase = PodPhase.SUCCEEDED.value
        api.read_namespaced_pod.return_value = resp
        handle = _make_handle(api=api)
        handle.wait()
        assert handle.poll() == JobReturnCode.SUCCESS

    # -- enter_states ---------------------------------------------------------
    def test_enter_states_returns_true_when_state_matches(self):
        api = _make_api_instance()
        resp = Mock()
        resp.status.phase = PodPhase.RUNNING.value
        api.read_namespaced_pod.return_value = resp
        handle = _make_handle(api=api)
        assert handle.enter_states([JobState.RUNNING]) is True

    def test_enter_states_accepts_single_state(self):
        api = _make_api_instance()
        resp = Mock()
        resp.status.phase = PodPhase.SUCCEEDED.value
        api.read_namespaced_pod.return_value = resp
        handle = _make_handle(api=api)
        assert handle.enter_states(JobState.SUCCEEDED) is True

    def test_enter_states_returns_false_on_timeout(self):
        api = _make_api_instance()
        resp = Mock()
        resp.status.phase = PodPhase.PENDING.value
        api.read_namespaced_pod.return_value = resp
        handle = _make_handle(api=api, timeout=0)
        assert handle.enter_states([JobState.RUNNING]) is False

    def test_enter_states_terminates_on_timeout(self):
        api = _make_api_instance()
        resp = Mock()
        resp.status.phase = PodPhase.PENDING.value
        api.read_namespaced_pod.return_value = resp
        handle = _make_handle(api=api, timeout=0)
        handle.enter_states([JobState.RUNNING])
        api.delete_namespaced_pod.assert_called_once()
        assert handle.terminal_state == JobState.TERMINATED

    def test_enter_states_returns_false_on_terminal_pod_phase(self):
        api = _make_api_instance()
        resp = Mock()
        resp.status.phase = PodPhase.FAILED.value
        api.read_namespaced_pod.return_value = resp
        handle = _make_handle(api=api)
        assert handle.enter_states([JobState.RUNNING]) is False
        assert handle.terminal_state == JobState.TERMINATED

    def test_enter_states_returns_false_on_succeeded_when_waiting_for_running(self):
        api = _make_api_instance()
        resp = Mock()
        resp.status.phase = PodPhase.SUCCEEDED.value
        api.read_namespaced_pod.return_value = resp
        handle = _make_handle(api=api)
        assert handle.enter_states([JobState.RUNNING]) is False
        assert handle.terminal_state == JobState.SUCCEEDED

    def test_enter_states_raises_on_invalid_state(self):
        handle = _make_handle()
        with pytest.raises(ValueError, match="expect job_states_to_enter"):
            handle.enter_states(["not_a_state"])

    # -- enter_states: wall-clock timeout branch ------------------------------
    # The pod is placed in UNKNOWN phase (non-pending so stuck detection does
    # not fire, non-terminal so the terminal-phase branch is skipped) and
    # time.time is mocked so the elapsed-time check fires on the first
    # iteration.  This is the branch that existing tests miss because they
    # use timeout=0 with a PENDING pod, which hits stuck detection instead.

    @patch("nvflare.app_opt.job_launcher.k8s_launcher.time")
    def test_enter_states_wall_clock_timeout_returns_false(self, mock_time):
        mock_time.time.side_effect = [0.0, 100.0]  # start=0, check=100 → 100 > timeout=10
        mock_time.sleep = Mock()
        api = _make_api_instance()
        resp = Mock()
        resp.status.phase = PodPhase.UNKNOWN.value
        api.read_namespaced_pod.return_value = resp
        handle = _make_handle(api=api, timeout=10)
        assert handle.enter_states([JobState.RUNNING]) is False

    @patch("nvflare.app_opt.job_launcher.k8s_launcher.time")
    def test_enter_states_wall_clock_timeout_calls_terminate_and_sets_terminal_state(self, mock_time):
        mock_time.time.side_effect = [0.0, 100.0]
        mock_time.sleep = Mock()
        api = _make_api_instance()
        resp = Mock()
        resp.status.phase = PodPhase.UNKNOWN.value
        api.read_namespaced_pod.return_value = resp
        handle = _make_handle(api=api, timeout=10)
        handle.enter_states([JobState.RUNNING])
        api.delete_namespaced_pod.assert_called_once()
        assert handle.terminal_state == JobState.TERMINATED

    @patch("nvflare.app_opt.job_launcher.k8s_launcher.time")
    def test_enter_states_wall_clock_not_fired_when_timeout_none(self, mock_time):
        # With timeout=None the wall-clock guard (self.timeout is not None) is
        # unconditionally False; the loop exits through the terminal-phase path.
        mock_time.time.return_value = 9999.0
        mock_time.sleep = Mock()
        api = _make_api_instance()
        resp = Mock()
        resp.status.phase = PodPhase.SUCCEEDED.value
        api.read_namespaced_pod.return_value = resp
        handle = _make_handle(api=api, timeout=None)
        handle.enter_states([JobState.RUNNING])
        api.delete_namespaced_pod.assert_not_called()

    @patch("nvflare.app_opt.job_launcher.k8s_launcher.time")
    def test_enter_states_wall_clock_not_fired_before_elapsed(self, mock_time):
        # First iteration: time not yet elapsed → wall-clock skipped, loop continues.
        # Second iteration: pod completes → exits via terminal-phase path, no terminate().
        mock_time.time.side_effect = [0.0, 0.5]  # start=0, first check=0.5 < timeout=10
        mock_time.sleep = Mock()
        api = _make_api_instance()
        resp_unknown = Mock()
        resp_unknown.status.phase = PodPhase.UNKNOWN.value
        resp_succeeded = Mock()
        resp_succeeded.status.phase = PodPhase.SUCCEEDED.value
        api.read_namespaced_pod.side_effect = [resp_unknown, resp_succeeded]
        handle = _make_handle(api=api, timeout=10)
        result = handle.enter_states([JobState.RUNNING])
        assert result is False
        api.delete_namespaced_pod.assert_not_called()
        assert handle.terminal_state == JobState.SUCCEEDED


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

    def test_init_raises_on_non_dict_pvc_file(self):
        from nvflare.app_opt.job_launcher.k8s_launcher import ClientK8sJobLauncher

        patches = _make_k8s_launcher_patches()
        mock_cfg, mock_conf, mock_core, mock_open, mock_yaml = _enter_patches(patches)
        try:
            mock_yaml.safe_load.return_value = "not-a-dict"
            mock_conf_instance = MagicMock()
            mock_conf.return_value = mock_conf_instance
            mock_conf.get_default_copy = Mock(return_value=mock_conf_instance)

            with pytest.raises(ValueError, match="dictionary"):
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


# ---------------------------------------------------------------------------
# K8sJobLauncher launch_job — integration-style happy path
# ---------------------------------------------------------------------------

_WORKER_MODULE = "nvflare.private.fed.app.client.worker_process"
_JOB_UUID = "550e8400-e29b-41d4-a716-446655440000"
_EXPECTED_JOB_ID = uuid4_to_rfc1123(_JOB_UUID)


def _make_launch_job_meta(site_name="site-1", image="nvflare/nvflare:latest", gpu=None):
    meta = {
        JobConstants.JOB_ID: _JOB_UUID,
        JobMetaKey.DEPLOY_MAP.value: {"app": [{"sites": [site_name], "image": image}]},
    }
    if gpu is not None:
        meta[JobMetaKey.RESOURCE_SPEC.value] = {site_name: {"num_of_gpus": gpu}}
    return meta


def _make_launch_fl_ctx(site_name="site-1", set_items=None):
    fl_ctx = FLContext()
    fl_ctx.set_prop(ReservedKey.IDENTITY_NAME, site_name, private=False, sticky=True)
    job_args = {
        JobProcessArgs.EXE_MODULE: ("-m", _WORKER_MODULE),
        JobProcessArgs.WORKSPACE: ("-w", "/var/tmp/nvflare/workspace"),
        JobProcessArgs.JOB_ID: ("-n", "job-abc"),
    }
    fl_ctx.set_prop(FLContextKey.JOB_PROCESS_ARGS, job_args, private=True, sticky=False)
    if set_items is not None:
        args_obj = Mock()
        args_obj.set = set_items
        fl_ctx.set_prop(FLContextKey.ARGS, args_obj, private=False, sticky=False)
    return fl_ctx


class TestK8sJobLauncherLaunchJob:
    """Integration-style tests that exercise the full launch_job() code path.

    The kubernetes API is mocked but the rest of the code — uuid sanitization,
    manifest construction, enter_states polling, and handle construction — runs
    for real.  read_namespaced_pod is primed to return Running immediately so
    enter_states returns True on the first iteration without sleeping.
    """

    def _setup(self, patches, namespace="test-ns"):
        from nvflare.app_opt.job_launcher.k8s_launcher import ClientK8sJobLauncher

        mock_cfg, mock_conf, mock_core_module, mock_open, mock_yaml = _enter_patches(patches)
        mock_yaml.safe_load.return_value = {"data-pvc": "/data"}
        mock_conf_instance = MagicMock()
        mock_conf.return_value = mock_conf_instance
        mock_conf.get_default_copy = Mock(return_value=mock_conf_instance)
        mock_api = MagicMock()
        mock_core_module.CoreV1Api.return_value = mock_api
        launcher = ClientK8sJobLauncher(
            config_file_path="/fake/kube/config",
            workspace_pvc="ws-pvc",
            etc_pvc="etc-pvc",
            data_pvc_file_path="/fake/data_pvc.yaml",
            namespace=namespace,
        )
        return launcher, mock_api

    def _prime_running(self, mock_api):
        resp = Mock()
        resp.status.phase = PodPhase.RUNNING.value
        mock_api.read_namespaced_pod.return_value = resp

    # -- return value ---------------------------------------------------------

    def test_returns_k8s_job_handle(self):
        from nvflare.app_opt.job_launcher.k8s_launcher import K8sJobHandle

        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches)
        self._prime_running(mock_api)
        try:
            handle = launcher.launch_job(_make_launch_job_meta(), _make_launch_fl_ctx())
            assert isinstance(handle, K8sJobHandle)
        finally:
            _exit_patches(patches)

    def test_terminal_state_is_none_after_clean_launch(self):
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches)
        self._prime_running(mock_api)
        try:
            handle = launcher.launch_job(_make_launch_job_meta(), _make_launch_fl_ctx())
            assert handle.terminal_state is None
        finally:
            _exit_patches(patches)

    # -- API call -------------------------------------------------------------

    def test_create_namespaced_pod_called_once(self):
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches)
        self._prime_running(mock_api)
        try:
            launcher.launch_job(_make_launch_job_meta(), _make_launch_fl_ctx())
            mock_api.create_namespaced_pod.assert_called_once()
        finally:
            _exit_patches(patches)

    def test_create_namespaced_pod_uses_launcher_namespace(self):
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches, namespace="prod-ns")
        self._prime_running(mock_api)
        try:
            launcher.launch_job(_make_launch_job_meta(), _make_launch_fl_ctx())
            assert mock_api.create_namespaced_pod.call_args.kwargs["namespace"] == "prod-ns"
        finally:
            _exit_patches(patches)

    # -- pod manifest: identity fields ----------------------------------------

    def test_pod_manifest_name_is_rfc1123_of_job_id(self):
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches)
        self._prime_running(mock_api)
        try:
            launcher.launch_job(_make_launch_job_meta(), _make_launch_fl_ctx())
            manifest = mock_api.create_namespaced_pod.call_args.kwargs["body"]
            assert manifest["metadata"]["name"] == _EXPECTED_JOB_ID
        finally:
            _exit_patches(patches)

    def test_pod_manifest_image_from_job_meta(self):
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches)
        self._prime_running(mock_api)
        try:
            launcher.launch_job(_make_launch_job_meta(image="myrepo/myimage:v2"), _make_launch_fl_ctx())
            manifest = mock_api.create_namespaced_pod.call_args.kwargs["body"]
            assert manifest["spec"]["containers"][0]["image"] == "myrepo/myimage:v2"
        finally:
            _exit_patches(patches)

    def test_pod_manifest_restart_policy_never(self):
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches)
        self._prime_running(mock_api)
        try:
            launcher.launch_job(_make_launch_job_meta(), _make_launch_fl_ctx())
            manifest = mock_api.create_namespaced_pod.call_args.kwargs["body"]
            assert manifest["spec"]["restartPolicy"] == "Never"
        finally:
            _exit_patches(patches)

    # -- pod manifest: container args -----------------------------------------

    def test_pod_manifest_container_args_contain_worker_module(self):
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches)
        self._prime_running(mock_api)
        try:
            launcher.launch_job(_make_launch_job_meta(), _make_launch_fl_ctx())
            manifest = mock_api.create_namespaced_pod.call_args.kwargs["body"]
            args = manifest["spec"]["containers"][0]["args"]
            assert "-u" in args
            assert "-m" in args
            assert _WORKER_MODULE in args
        finally:
            _exit_patches(patches)

    def test_pod_manifest_set_list_propagated_from_args(self):
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches)
        self._prime_running(mock_api)
        try:
            fl_ctx = _make_launch_fl_ctx(set_items=["lr=0.01", "epochs=5"])
            launcher.launch_job(_make_launch_job_meta(), fl_ctx)
            manifest = mock_api.create_namespaced_pod.call_args.kwargs["body"]
            args = manifest["spec"]["containers"][0]["args"]
            assert "--set" in args
            assert "lr=0.01" in args
            assert "epochs=5" in args
        finally:
            _exit_patches(patches)

    def test_pod_manifest_no_set_list_when_args_not_set(self):
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches)
        self._prime_running(mock_api)
        try:
            launcher.launch_job(_make_launch_job_meta(), _make_launch_fl_ctx())
            manifest = mock_api.create_namespaced_pod.call_args.kwargs["body"]
            args = manifest["spec"]["containers"][0]["args"]
            assert "--set" not in args
        finally:
            _exit_patches(patches)

    # -- pod manifest: volumes ------------------------------------------------

    def test_pod_manifest_pvcs_use_launcher_pvc_names(self):
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches)
        self._prime_running(mock_api)
        try:
            launcher.launch_job(_make_launch_job_meta(), _make_launch_fl_ctx())
            manifest = mock_api.create_namespaced_pod.call_args.kwargs["body"]
            claims = {v["name"]: v["persistentVolumeClaim"]["claimName"] for v in manifest["spec"]["volumes"]}
            assert claims[PV_NAME.WORKSPACE.value] == "ws-pvc"
            assert claims[PV_NAME.DATA.value] == "data-pvc"
            assert claims[PV_NAME.ETC.value] == "etc-pvc"
        finally:
            _exit_patches(patches)

    # -- pod manifest: GPU resources ------------------------------------------

    def test_pod_manifest_includes_gpu_limit_when_specified(self):
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches)
        self._prime_running(mock_api)
        try:
            launcher.launch_job(_make_launch_job_meta(gpu=2), _make_launch_fl_ctx())
            manifest = mock_api.create_namespaced_pod.call_args.kwargs["body"]
            assert manifest["spec"]["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] == 2
        finally:
            _exit_patches(patches)

    def test_pod_manifest_omits_gpu_limit_when_not_specified(self):
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches)
        self._prime_running(mock_api)
        try:
            launcher.launch_job(_make_launch_job_meta(), _make_launch_fl_ctx())
            manifest = mock_api.create_namespaced_pod.call_args.kwargs["body"]
            assert "resources" not in manifest["spec"]["containers"][0]
        finally:
            _exit_patches(patches)

    # -- create_namespaced_pod failure paths ----------------------------------

    def test_network_error_on_create_returns_handle_with_terminal_state(self):
        """Non-ApiException (e.g. network timeout) must not leave terminal_state=None."""
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches)
        mock_api.create_namespaced_pod.side_effect = ConnectionError("network unreachable")
        try:
            handle = launcher.launch_job(_make_launch_job_meta(), _make_launch_fl_ctx())
            assert isinstance(handle, K8sJobHandle)
            assert handle.terminal_state == JobState.TERMINATED
        finally:
            _exit_patches(patches)

    def test_network_error_on_create_does_not_call_terminate_api(self):
        """Pod was never created; no delete API call should be made."""
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches)
        mock_api.create_namespaced_pod.side_effect = ConnectionError("network unreachable")
        try:
            launcher.launch_job(_make_launch_job_meta(), _make_launch_fl_ctx())
            mock_api.delete_namespaced_pod.assert_not_called()
        finally:
            _exit_patches(patches)

    def test_network_error_on_create_does_not_call_enter_states(self):
        """enter_states must not be reached when pod creation fails."""
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches)
        mock_api.create_namespaced_pod.side_effect = ConnectionError("network unreachable")
        try:
            launcher.launch_job(_make_launch_job_meta(), _make_launch_fl_ctx())
            # read_namespaced_pod is the backing call for _query_phase / enter_states
            mock_api.read_namespaced_pod.assert_not_called()
        finally:
            _exit_patches(patches)
