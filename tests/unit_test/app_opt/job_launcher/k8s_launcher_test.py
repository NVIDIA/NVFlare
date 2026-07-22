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

import logging
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
_k8s_client.Configuration = MagicMock()  # instance so .set_default() is auto-mocked
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
    sys.modules[_mod_name] = _mod_obj

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, JobConstants, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import JobMetaKey
from nvflare.apis.job_launcher_spec import JobProcessArgs, JobReturnCode
from nvflare.app_opt.job_launcher.k8s_launcher import (
    JOB_RETURN_CODE_MAPPING,
    POD_STATE_MAPPING,
    WORKSPACE_MOUNT_PATH,
    JobState,
    K8sJobHandle,
    PodPhase,
    _job_args_dict,
    job_pod_name,
    site_name_to_rfc1123,
    study_dataset_volume_name,
    uuid4_to_rfc1123,
)
from nvflare.app_opt.job_launcher.workspace_cell_transfer import ENV_WORKSPACE_OWNER_FQCN, ENV_WORKSPACE_TRANSFER_TOKEN

_DEFAULT_DATA_VOLUME_NAME = study_dataset_volume_name("study-a", "training")

_DEFAULT_VOLUME_MOUNT_LIST = [
    {"name": "workspace-job", "mountPath": WORKSPACE_MOUNT_PATH},
    {"name": "startup-kit", "mountPath": f"{WORKSPACE_MOUNT_PATH}/startup", "readOnly": True},
    {"name": _DEFAULT_DATA_VOLUME_NAME, "mountPath": "/data/study-a/training", "readOnly": True},
]

_DEFAULT_VOLUME_LIST = [
    {"name": "workspace-job", "emptyDir": {}},
    {"name": "startup-kit", "secret": {"secretName": f"nvflare-startup-{site_name_to_rfc1123('site1')}"}},
    {"name": _DEFAULT_DATA_VOLUME_NAME, "persistentVolumeClaim": {"claimName": "data-pvc"}},
]


def _make_job_config(**overrides):
    cfg = {
        "name": "test-job-123",
        "image": "nvflare/nvflare:test",
        "container_name": "container-test-job-123",
        "command": "nvflare.private.fed.app.client.worker_process",
        "volume_mount_list": _DEFAULT_VOLUME_MOUNT_LIST,
        "volume_list": _DEFAULT_VOLUME_LIST,
        "module_args": {"-m": "val_m", "-w": "val_w"},
        "set_list": ["key1=val1", "key2=val2"],
        "resources": {"limits": {"nvidia.com/gpu": 1}},
    }
    cfg.update(overrides)
    return cfg


def _make_api_instance():
    return MagicMock()


def _make_condition(type_, status, reason=None, message=None):
    condition = Mock()
    condition.type = type_
    condition.status = status
    condition.reason = reason
    condition.message = message
    return condition


def _make_waiting_container_status(reason, message=None):
    waiting = Mock()
    waiting.reason = reason
    waiting.message = message
    state = Mock()
    state.waiting = waiting
    container_status = Mock()
    container_status.state = state
    return container_status


def _make_pod_response(phase=PodPhase.PENDING.value, conditions=None, node_name=None, container_statuses=None):
    resp = Mock()
    resp.status.phase = phase
    resp.status.conditions = conditions or []
    resp.status.node_name = node_name
    resp.status.container_statuses = container_statuses or []
    resp.status.init_container_statuses = []
    return resp


def _make_resource_pending_pod(resource="cpu"):
    return _make_pod_response(
        conditions=[
            _make_condition(
                "PodScheduled",
                "False",
                "Unschedulable",
                f"0/1 nodes are available: 1 Insufficient {resource}.",
            )
        ]
    )


def _make_container_creating_pod():
    return _make_pod_response(
        node_name="node-1",
        conditions=[_make_condition("PodScheduled", "True")],
        container_statuses=[_make_waiting_container_status("ContainerCreating")],
    )


def _make_event(reason, message=None, event_type="Warning", last_timestamp=None):
    event = Mock()
    event.reason = reason
    event.message = message
    event.type = event_type
    event.last_timestamp = last_timestamp
    return event


def _make_event_list(events):
    event_list = Mock()
    event_list.items = events
    return event_list


def _set_pod_events(api, events):
    event_list = _make_event_list(events)
    api.list_namespaced_event.return_value = event_list


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


class TestSiteNameToRfc1123:
    def test_site_name_collision_gets_distinct_suffixes(self):
        assert site_name_to_rfc1123("site#1") != site_name_to_rfc1123("site1")


class TestJobPodName:
    def test_includes_site_suffix(self):
        pod_name = job_pod_name("job-123", "site-1")
        assert pod_name == f"job-123-{site_name_to_rfc1123('site-1', max_length=20)}"

    def test_same_job_id_has_distinct_pods_for_different_sites(self):
        assert job_pod_name("job-123", "server") != job_pod_name("job-123", "client")

    def test_truncates_to_k8s_label_limit(self):
        pod_name = job_pod_name("j" + "a" * 100, "site-1")
        assert len(pod_name) == 63
        assert not pod_name.endswith("-")


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

    def test_pending_timeout_is_independent_from_launch_timeout(self):
        cfg = _make_job_config()
        handle = K8sJobHandle("job-1", _make_api_instance(), cfg, timeout=30)
        assert handle._pending_timeout_secs == 120
        assert handle.pending_timeout == 120

    def test_pending_timeout_secs_uses_pending_timeout_when_no_timeout(self):
        cfg = _make_job_config()
        handle = K8sJobHandle("job-1", _make_api_instance(), cfg, timeout=None)
        assert handle._pending_timeout_secs == 120

    def test_pending_timeout_secs_uses_custom_pending_timeout(self):
        cfg = _make_job_config()
        handle = K8sJobHandle("job-1", _make_api_instance(), cfg, timeout=None, pending_timeout=60)
        assert handle._pending_timeout_secs == 60

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

    def test_manifest_template_prefers_nvflare_job_over_unnamed_first_container(self):
        cfg = _make_job_config()
        del cfg["container_name"]
        pod_manifest_template = {
            "apiVersion": "v1",
            "kind": "Pod",
            "spec": {
                "containers": [
                    {"image": "sidecar:v1"},
                    {"name": "nvflare_job", "image": "template-image"},
                ]
            },
        }

        handle = K8sJobHandle("job-1", _make_api_instance(), cfg, pod_manifest_template=pod_manifest_template)

        containers = handle.get_manifest()["spec"]["containers"]
        assert containers[0] == {"image": "sidecar:v1"}
        assert containers[1]["name"] == "nvflare_job"
        assert containers[1]["image"] == "nvflare/nvflare:test"
        assert containers[1]["command"] == ["/usr/local/bin/python"]
        assert "nvflare.private.fed.app.client.worker_process" in containers[1]["args"]

    def test_manifest_restart_policy(self):
        cfg = _make_job_config()
        handle = K8sJobHandle("job-1", _make_api_instance(), cfg)
        assert handle.get_manifest()["spec"]["restartPolicy"] == "Never"

    def test_manifest_image_pull_secrets(self):
        cfg = _make_job_config(image_pull_secrets=["job-regcred", "site.registry.example.com"])
        handle = K8sJobHandle("job-1", _make_api_instance(), cfg)
        assert handle.get_manifest()["spec"]["imagePullSecrets"] == [
            {"name": "job-regcred"},
            {"name": "site.registry.example.com"},
        ]

    def test_manifest_volumes(self):
        cfg = _make_job_config()
        handle = K8sJobHandle("job-1", _make_api_instance(), cfg)
        volumes = handle.get_manifest()["spec"]["volumes"]
        assert len(volumes) == 3
        vol_map = {v["name"]: v for v in volumes}
        assert "emptyDir" in vol_map["workspace-job"]
        assert "secret" in vol_map["startup-kit"]
        assert vol_map[_DEFAULT_DATA_VOLUME_NAME]["persistentVolumeClaim"]["claimName"] == "data-pvc"

    def test_manifest_volume_mounts(self):
        cfg = _make_job_config()
        handle = K8sJobHandle("job-1", _make_api_instance(), cfg)
        container = handle.get_manifest()["spec"]["containers"][0]
        assert container["volumeMounts"] == _DEFAULT_VOLUME_MOUNT_LIST

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

    def test_manifest_missing_module_args_produces_empty_args_list(self):
        cfg = _make_job_config()
        cfg["module_args"] = None
        handle = K8sJobHandle("job-1", _make_api_instance(), cfg)
        assert handle.container_args_module_args_dict_as_list == []

    def test_manifest_gpu_resources(self):
        cfg = _make_job_config()
        handle = K8sJobHandle("job-1", _make_api_instance(), cfg)
        container = handle.get_manifest()["spec"]["containers"][0]
        assert container["resources"]["limits"]["nvidia.com/gpu"] == 1

    def test_manifest_resources_passed_through_unconditionally(self):
        cfg = _make_job_config(resources={"limits": {"cpu": "500m"}})

        handle = K8sJobHandle("job-1", _make_api_instance(), cfg)
        container = handle.get_manifest()["spec"]["containers"][0]
        assert container["resources"] == {"limits": {"cpu": "500m"}}

    def test_manifest_no_gpu_resources(self):
        cfg = _make_job_config(resources={"limits": {}})

        handle = K8sJobHandle("job-1", _make_api_instance(), cfg)
        container = handle.get_manifest()["spec"]["containers"][0]
        assert container["resources"] == {"limits": {}}

    def test_manifest_no_resources_key(self):
        cfg = _make_job_config()
        del cfg["resources"]
        handle = K8sJobHandle("job-1", _make_api_instance(), cfg)
        container = handle.get_manifest()["spec"]["containers"][0]
        assert "resources" not in container

    # -- PYTHONPATH env var ---------------------------------------------------

    def test_manifest_pythonpath_set_when_app_custom_folder_present(self):
        cfg = _make_job_config(env={"PYTHONPATH": "/custom/app"})
        handle = K8sJobHandle("job-1", _make_api_instance(), cfg)
        container = handle.get_manifest()["spec"]["containers"][0]
        assert "env" in container
        env_map = {e["name"]: e["value"] for e in container["env"]}
        assert env_map["PYTHONPATH"] == "/custom/app"

    def test_manifest_pythonpath_value_is_string(self):
        cfg = _make_job_config(env={"PYTHONPATH": "/my/custom"})
        handle = K8sJobHandle("job-1", _make_api_instance(), cfg)
        container = handle.get_manifest()["spec"]["containers"][0]
        env_map = {e["name"]: e["value"] for e in container["env"]}
        assert isinstance(env_map["PYTHONPATH"], str)

    def test_manifest_no_env_when_app_custom_folder_empty_string(self):
        cfg = _make_job_config(env={"PYTHONPATH": ""})
        handle = K8sJobHandle("job-1", _make_api_instance(), cfg)
        container = handle.get_manifest()["spec"]["containers"][0]
        assert "env" not in container

    def test_manifest_no_env_when_no_env_key(self):
        cfg = _make_job_config()
        handle = K8sJobHandle("job-1", _make_api_instance(), cfg)
        container = handle.get_manifest()["spec"]["containers"][0]
        assert "env" not in container

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

    def test_poll_removes_workspace_job_on_terminal_state(self):
        api = _make_api_instance()
        resp = Mock()
        resp.status.phase = PodPhase.SUCCEEDED.value
        api.read_namespaced_pod.return_value = resp
        ws_transfer = Mock()
        handle = _make_handle(api=api, workspace_transfer=ws_transfer, workspace_job_id="job-raw")
        assert handle.poll() == JobReturnCode.SUCCESS
        ws_transfer.remove_job.assert_called_once_with("job-raw")

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

    def test_terminate_uses_pod_name_when_supplied(self):
        api = _make_api_instance()
        handle = K8sJobHandle("job-1", api, _make_job_config(), pod_name="job-1-site-1")
        handle.terminate()
        api.delete_namespaced_pod.assert_called_once_with(
            name="job-1-site-1", namespace="default", grace_period_seconds=0
        )
        assert handle.job_id == "job-1"
        assert handle.pod_name == "job-1-site-1"

    # -- _query_phase ---------------------------------------------------------
    def test_query_phase_returns_none_on_api_error(self):
        api = _make_api_instance()
        api.read_namespaced_pod.side_effect = _FakeApiException(status=500, reason="Error")
        handle = _make_handle(api=api)
        assert handle._query_phase() is None

    def test_query_phase_returns_none_on_generic_exception(self):
        api = _make_api_instance()
        api.read_namespaced_pod.side_effect = RuntimeError("connection lost")
        handle = _make_handle(api=api)
        assert handle._query_phase() is None

    def test_query_phase_sets_terminal_state_on_not_found(self):
        api = _make_api_instance()
        api.read_namespaced_pod.side_effect = _FakeApiException(status=404, reason="Not Found")
        handle = _make_handle(api=api)

        assert handle._query_phase() == PodPhase.UNKNOWN.value
        assert handle.terminal_state == JobState.TERMINATED

    def test_query_phase_uses_pod_name_when_supplied(self):
        api = _make_api_instance()
        resp = Mock()
        resp.status.phase = PodPhase.RUNNING.value
        api.read_namespaced_pod.return_value = resp
        handle = K8sJobHandle("job-1", api, _make_job_config(), pod_name="job-1-site-1")
        assert handle._query_phase() == PodPhase.RUNNING.value
        api.read_namespaced_pod.assert_called_once_with(name="job-1-site-1", namespace="default")

    def test_query_phase_returns_none_when_phase_missing(self):
        api = _make_api_instance()
        resp = Mock()
        resp.status.phase = None
        api.read_namespaced_pod.return_value = resp
        handle = _make_handle(api=api)
        assert handle._query_phase() is None

    # -- _stuck_in_pending ----------------------------------------------------
    def test_stuck_in_pending_returns_true_at_pending_timeout(self):
        api = _make_api_instance()
        handle = K8sJobHandle("job-1", api, _make_job_config(), pending_timeout=5)
        assert handle._stuck_in_pending(PodPhase.PENDING.value, now=0) is False
        assert handle._stuck_in_pending(PodPhase.PENDING.value, now=5) is True

    def test_stuck_in_pending_returns_false_before_pending_timeout(self):
        api = _make_api_instance()
        handle = K8sJobHandle("job-1", api, _make_job_config(), pending_timeout=5)
        assert handle._stuck_in_pending(PodPhase.PENDING.value, now=0) is False
        assert handle._stuck_in_pending(PodPhase.PENDING.value, now=4.9) is False

    def test_stuck_in_pending_returns_true_immediately_when_pending_timeout_zero(self):
        api = _make_api_instance()
        handle = K8sJobHandle("job-1", api, _make_job_config(), pending_timeout=0)
        assert handle._stuck_in_pending(PodPhase.PENDING.value, now=0) is True

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

    def test_stuck_in_pending_ignores_unobserved_phase(self):
        api = _make_api_instance()
        handle = K8sJobHandle("job-1", api, _make_job_config(), timeout=None, pending_timeout=2)
        handle._pending_since = 0
        handle._stuck_count = 1

        assert handle._stuck_in_pending(None) is False
        assert handle._pending_since == 0
        assert handle._stuck_count == 1

    def test_stuck_in_pending_does_not_count_unknown(self):
        api = _make_api_instance()
        handle = K8sJobHandle("job-1", api, _make_job_config(), timeout=None, pending_timeout=2)

        assert handle._stuck_in_pending(PodPhase.UNKNOWN.value, now=0) is False
        assert handle._stuck_in_pending(PodPhase.UNKNOWN.value, now=2) is False

    def test_stuck_in_pending_returns_false_when_under_max(self):
        api = _make_api_instance()
        handle = K8sJobHandle("job-1", api, _make_job_config(), timeout=None, pending_timeout=100)
        assert handle._stuck_in_pending(PodPhase.PENDING.value) is False

    def test_stuck_in_pending_never_fires_when_pending_timeout_none(self):
        # pending_timeout=None with timeout=None disables resource-shortage stuck detection.
        api = _make_api_instance()
        handle = K8sJobHandle("job-1", api, _make_job_config(), timeout=None, pending_timeout=None)
        assert handle._pending_timeout_secs is None
        # Drive _stuck_count very high — must not raise and must return False
        handle._stuck_count = 10_000
        assert handle._stuck_in_pending(PodPhase.PENDING.value) is False
        assert handle._stuck_in_pending(PodPhase.UNKNOWN.value) is False

    # -- wait -----------------------------------------------------------------
    def test_wait_returns_immediately_if_terminal_state_set(self):
        api = _make_api_instance()
        handle = _make_handle(api=api)
        handle.terminal_state = JobState.TERMINATED
        handle.wait()
        api.read_namespaced_pod.assert_not_called()

    @patch("nvflare.app_opt.job_launcher.k8s_launcher.time.sleep")
    def test_wait_returns_without_sleep_when_query_marks_terminal(self, mock_sleep):
        api = _make_api_instance()
        api.read_namespaced_pod.side_effect = _FakeApiException(status=404, reason="Not Found")
        handle = _make_handle(api=api)

        handle.wait()

        assert handle.terminal_state == JobState.TERMINATED
        mock_sleep.assert_not_called()

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

    def test_wait_removes_workspace_job_on_terminal_state(self):
        api = _make_api_instance()
        resp = Mock()
        resp.status.phase = PodPhase.SUCCEEDED.value
        api.read_namespaced_pod.return_value = resp
        ws_transfer = Mock()
        handle = _make_handle(api=api, workspace_transfer=ws_transfer, workspace_job_id="job-raw")
        handle.wait()
        ws_transfer.remove_job.assert_called_once_with("job-raw")

    @patch("nvflare.app_opt.job_launcher.k8s_launcher.time")
    def test_wait_deletes_pending_pod_after_pending_timeout(self, mock_time):
        mock_time.time.side_effect = [0.0, 1.0]
        mock_time.sleep = Mock()
        api = _make_api_instance()
        api.read_namespaced_pod.return_value = _make_resource_pending_pod()
        handle = _make_handle(api=api, timeout=None, pending_timeout=1)

        handle.wait()

        assert api.read_namespaced_pod.call_count == 2
        api.delete_namespaced_pod.assert_called_once()
        assert handle.terminal_state == JobState.TERMINATED
        assert handle.poll() == JobReturnCode.EXCEPTION

    def test_terminate_removes_workspace_job(self):
        api = _make_api_instance()
        ws_transfer = Mock()
        handle = _make_handle(api=api, workspace_transfer=ws_transfer, workspace_job_id="job-raw")
        handle.terminate()
        ws_transfer.remove_job.assert_called_once_with("job-raw")

    def test_manual_terminate_poll_returns_aborted(self):
        api = _make_api_instance()
        handle = _make_handle(api=api)

        handle.terminate()

        assert handle.poll() == JobReturnCode.ABORTED

    def test_poll_honors_preserved_return_code_when_observed_state_becomes_terminal(self):
        api = _make_api_instance()
        resp = Mock()
        resp.status.phase = PodPhase.FAILED.value
        api.read_namespaced_pod.return_value = resp
        handle = _make_handle(api=api)
        handle.terminal_return_code = JobReturnCode.EXCEPTION

        assert handle.poll() == JobReturnCode.EXCEPTION
        assert handle.terminal_state == JobState.TERMINATED

    def test_poll_deletes_pending_pod_immediately_when_pending_timeout_zero(self):
        api = _make_api_instance()
        api.read_namespaced_pod.return_value = _make_resource_pending_pod()
        handle = _make_handle(api=api, pending_timeout=0)

        assert handle.poll() == JobReturnCode.EXCEPTION

        api.delete_namespaced_pod.assert_called_once()
        assert handle.terminal_state == JobState.TERMINATED

    def test_poll_terminates_on_image_pull_backoff(self):
        api = _make_api_instance()
        api.read_namespaced_pod.return_value = _make_pod_response(
            node_name="node-1",
            conditions=[_make_condition("PodScheduled", "True")],
            container_statuses=[
                _make_waiting_container_status(
                    "ImagePullBackOff",
                    "Back-off pulling image",
                )
            ],
        )
        handle = _make_handle(api=api, pending_timeout=120)

        assert handle.poll() == JobReturnCode.EXCEPTION

        api.delete_namespaced_pod.assert_called_once()
        assert handle.terminal_state == JobState.TERMINATED

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

    def test_enter_states_timeout_poll_returns_exception(self):
        api = _make_api_instance()
        resp = Mock()
        resp.status.phase = PodPhase.PENDING.value
        api.read_namespaced_pod.return_value = resp
        handle = _make_handle(api=api, timeout=0)
        handle.enter_states([JobState.RUNNING])
        assert handle.poll() == JobReturnCode.EXCEPTION

    @patch("nvflare.app_opt.job_launcher.k8s_launcher.time")
    def test_enter_states_deletes_pending_pod_after_pending_timeout(self, mock_time):
        mock_time.time.side_effect = [0.0, 0.0, 1.0, 2.0]
        mock_time.sleep = Mock()
        api = _make_api_instance()
        api.read_namespaced_pod.return_value = _make_resource_pending_pod()
        handle = _make_handle(
            api=api,
            timeout=None,
            pending_timeout=2,
            namespace="nvflare-test",
            pod_name="pending-resource-pod",
        )

        assert handle.enter_states([JobState.RUNNING]) is False

        assert api.read_namespaced_pod.call_count == 3
        api.delete_namespaced_pod.assert_called_once_with(
            name="pending-resource-pod",
            namespace="nvflare-test",
            grace_period_seconds=0,
        )
        assert handle.terminal_state == JobState.TERMINATED
        assert handle.poll() == JobReturnCode.EXCEPTION

    @patch("nvflare.app_opt.job_launcher.k8s_launcher.time")
    def test_enter_states_deletes_pending_pod_immediately_when_pending_timeout_zero(self, mock_time):
        mock_time.time.side_effect = [0.0, 0.0]
        mock_time.sleep = Mock()
        api = _make_api_instance()
        api.read_namespaced_pod.return_value = _make_resource_pending_pod()
        handle = _make_handle(api=api, timeout=None, pending_timeout=0)

        assert handle.enter_states([JobState.RUNNING]) is False

        api.read_namespaced_pod.assert_called_once()
        api.delete_namespaced_pod.assert_called_once()
        assert handle.terminal_state == JobState.TERMINATED
        assert handle.poll() == JobReturnCode.EXCEPTION

    @patch("nvflare.app_opt.job_launcher.k8s_launcher.time")
    def test_enter_states_uses_event_resource_shortage_for_pending_timeout(self, mock_time):
        mock_time.time.side_effect = [0.0, 0.0]
        mock_time.sleep = Mock()
        api = _make_api_instance()
        api.read_namespaced_pod.return_value = _make_pod_response()
        _set_pod_events(
            api,
            [_make_event("FailedScheduling", "0/1 nodes are available: 1 Insufficient nvidia.com/gpu.")],
        )
        handle = _make_handle(api=api, timeout=None, pending_timeout=0)

        assert handle.enter_states([JobState.RUNNING]) is False

        api.list_namespaced_event.assert_called_once_with(
            namespace="default",
            field_selector="involvedObject.name=job-1",
        )
        api.delete_namespaced_pod.assert_called_once()
        assert handle.poll() == JobReturnCode.EXCEPTION

    @patch("nvflare.app_opt.job_launcher.k8s_launcher.time")
    def test_enter_states_terminates_on_non_resource_scheduling_failure(self, mock_time):
        mock_time.time.side_effect = [0.0, 0.0]
        mock_time.sleep = Mock()
        api = _make_api_instance()
        api.read_namespaced_pod.return_value = _make_pod_response(
            conditions=[
                _make_condition(
                    "PodScheduled",
                    "False",
                    "Unschedulable",
                    "0/1 nodes are available: pod has unbound immediate PersistentVolumeClaims.",
                )
            ]
        )
        handle = _make_handle(api=api, timeout=None, pending_timeout=120)

        assert handle.enter_states([JobState.RUNNING]) is False

        api.read_namespaced_pod.assert_called_once()
        api.delete_namespaced_pod.assert_called_once()
        assert handle.poll() == JobReturnCode.EXCEPTION

    @patch("nvflare.app_opt.job_launcher.k8s_launcher.time")
    def test_enter_states_terminates_on_non_resource_failed_scheduling_event(self, mock_time):
        mock_time.time.side_effect = [0.0, 0.0]
        mock_time.sleep = Mock()
        api = _make_api_instance()
        api.read_namespaced_pod.return_value = _make_pod_response()
        _set_pod_events(
            api,
            [_make_event("FailedScheduling", "0/1 nodes are available: pod did not match node affinity.")],
        )
        handle = _make_handle(api=api, timeout=None, pending_timeout=120)

        assert handle.enter_states([JobState.RUNNING]) is False

        api.delete_namespaced_pod.assert_called_once()
        assert handle.poll() == JobReturnCode.EXCEPTION

    @patch("nvflare.app_opt.job_launcher.k8s_launcher.time")
    def test_enter_states_uses_latest_scheduling_event(self, mock_time):
        mock_time.time.side_effect = [0.0, 0.0]
        mock_time.sleep = Mock()
        api = _make_api_instance()
        api.read_namespaced_pod.return_value = _make_pod_response()
        _set_pod_events(
            api,
            [
                _make_event(
                    "FailedScheduling",
                    "0/1 nodes are available: 1 Insufficient cpu.",
                    last_timestamp="2026-01-01T00:00:00Z",
                ),
                _make_event(
                    "FailedScheduling",
                    "0/1 nodes are available: pod did not match node affinity.",
                    last_timestamp="2026-01-01T00:01:00Z",
                ),
            ],
        )
        handle = _make_handle(api=api, timeout=None, pending_timeout=None)

        assert handle.enter_states([JobState.RUNNING]) is False

        api.delete_namespaced_pod.assert_called_once()
        assert handle.poll() == JobReturnCode.EXCEPTION

    @patch("nvflare.app_opt.job_launcher.k8s_launcher.time")
    def test_enter_states_ignores_non_warning_failed_scheduling_event(self, mock_time):
        mock_time.time.side_effect = [0.0, 0.0, 1.0]
        mock_time.sleep = Mock()
        api = _make_api_instance()
        running_resp = Mock()
        running_resp.status.phase = PodPhase.RUNNING.value
        api.read_namespaced_pod.side_effect = [_make_pod_response(), running_resp]
        _set_pod_events(
            api,
            [_make_event("FailedScheduling", "non-warning scheduling event", event_type="Normal")],
        )
        handle = _make_handle(api=api, timeout=10, pending_timeout=0)

        assert handle.enter_states([JobState.RUNNING]) is True

        api.delete_namespaced_pod.assert_not_called()

    @patch("nvflare.app_opt.job_launcher.k8s_launcher.time")
    def test_enter_states_waits_on_unrecognized_container_waiting_reason(self, mock_time):
        mock_time.time.side_effect = [0.0, 0.0, 1.0]
        mock_time.sleep = Mock()
        api = _make_api_instance()
        unknown_waiting = _make_pod_response(
            node_name="node-1",
            conditions=[_make_condition("PodScheduled", "True")],
            container_statuses=[_make_waiting_container_status("ContainerStatusUnknown")],
        )
        running_resp = Mock()
        running_resp.status.phase = PodPhase.RUNNING.value
        api.read_namespaced_pod.side_effect = [unknown_waiting, running_resp]
        handle = _make_handle(api=api, timeout=10, pending_timeout=0)

        assert handle.enter_states([JobState.RUNNING]) is True

        api.delete_namespaced_pod.assert_not_called()

    @patch("nvflare.app_opt.job_launcher.k8s_launcher.time")
    def test_enter_states_terminates_on_recent_scheduled_warning_event(self, mock_time):
        mock_time.time.side_effect = [10.0, 10.0]
        mock_time.sleep = Mock()
        api = _make_api_instance()
        api.read_namespaced_pod.return_value = _make_container_creating_pod()
        _set_pod_events(
            api,
            [_make_event("FailedMount", "Unable to attach or mount volumes", last_timestamp=9.0)],
        )
        handle = _make_handle(api=api, timeout=None, pending_timeout=120)

        assert handle.enter_states([JobState.RUNNING]) is False

        api.delete_namespaced_pod.assert_called_once()
        assert handle.poll() == JobReturnCode.EXCEPTION

    @patch("nvflare.app_opt.job_launcher.k8s_launcher.time")
    def test_enter_states_terminates_on_persistent_scheduled_warning_event(self, mock_time):
        mock_time.time.side_effect = [100.0, 100.0]
        mock_time.sleep = Mock()
        api = _make_api_instance()
        api.read_namespaced_pod.return_value = _make_container_creating_pod()
        _set_pod_events(
            api,
            [_make_event("FailedMount", "Unable to attach or mount volumes", last_timestamp=80.0)],
        )
        handle = _make_handle(api=api, timeout=None, pending_timeout=120)

        assert handle.enter_states([JobState.RUNNING]) is False

        api.delete_namespaced_pod.assert_called_once()
        assert handle.poll() == JobReturnCode.EXCEPTION

    @patch("nvflare.app_opt.job_launcher.k8s_launcher.time")
    def test_enter_states_ignores_stale_scheduled_warning_event(self, mock_time):
        mock_time.time.side_effect = [120.0, 120.0, 121.0]
        mock_time.sleep = Mock()
        api = _make_api_instance()
        running_resp = Mock()
        running_resp.status.phase = PodPhase.RUNNING.value
        api.read_namespaced_pod.side_effect = [_make_container_creating_pod(), running_resp]
        _set_pod_events(
            api,
            [_make_event("FailedMount", "Unable to attach or mount volumes", last_timestamp=1.0)],
        )
        handle = _make_handle(api=api, timeout=20, pending_timeout=0)

        assert handle.enter_states([JobState.RUNNING]) is True

        api.delete_namespaced_pod.assert_not_called()

    @patch("nvflare.app_opt.job_launcher.k8s_launcher.time")
    def test_enter_states_ignores_event_api_failure_for_normal_startup_delay(self, mock_time):
        mock_time.time.side_effect = [0.0, 0.0, 1.0]
        mock_time.sleep = Mock()
        api = _make_api_instance()
        running_resp = Mock()
        running_resp.status.phase = PodPhase.RUNNING.value
        api.read_namespaced_pod.side_effect = [_make_container_creating_pod(), running_resp]
        api.list_namespaced_event.side_effect = _FakeApiException(status=500, reason="Internal")
        handle = _make_handle(api=api, timeout=10, pending_timeout=0)

        assert handle.enter_states([JobState.RUNNING]) is True

        api.delete_namespaced_pod.assert_not_called()

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

    def test_enter_states_returns_false_when_query_marks_terminal(self):
        api = _make_api_instance()
        api.read_namespaced_pod.side_effect = _FakeApiException(status=404, reason="Not Found")
        handle = _make_handle(api=api, timeout=None, pending_timeout=120)

        assert handle.enter_states([JobState.RUNNING]) is False
        assert handle.terminal_state == JobState.TERMINATED
        api.delete_namespaced_pod.assert_not_called()

    @patch("nvflare.app_opt.job_launcher.k8s_launcher.time")
    def test_enter_states_terminates_immediately_on_unknown_phase(self, mock_time):
        mock_time.time.side_effect = [0.0, 0.0]
        mock_time.sleep = Mock()
        api = _make_api_instance()
        api.read_namespaced_pod.return_value = _make_pod_response(phase=PodPhase.UNKNOWN.value)
        handle = _make_handle(api=api, timeout=None, pending_timeout=2)

        assert handle.enter_states([JobState.RUNNING]) is False
        assert handle.terminal_state == JobState.TERMINATED
        assert api.read_namespaced_pod.call_count == 1
        api.delete_namespaced_pod.assert_called_once()
        assert handle.poll() == JobReturnCode.EXCEPTION

    @patch("nvflare.app_opt.job_launcher.k8s_launcher.time.sleep")
    def test_enter_states_does_not_count_api_errors_as_pending(self, mock_sleep):
        api = _make_api_instance()
        pending_resp = _make_resource_pending_pod()
        running_resp = Mock()
        running_resp.status.phase = PodPhase.RUNNING.value
        api.read_namespaced_pod.side_effect = [
            pending_resp,
            _FakeApiException(status=500, reason="Internal"),
            running_resp,
        ]
        handle = _make_handle(api=api, timeout=None, pending_timeout=2)

        assert handle.enter_states([JobState.RUNNING]) is True
        assert handle._stuck_count == 0
        assert api.read_namespaced_pod.call_count == 3
        api.delete_namespaced_pod.assert_not_called()

    @patch("nvflare.app_opt.job_launcher.k8s_launcher.time")
    def test_enter_states_pauses_pending_timeout_clock_across_api_errors(self, mock_time):
        mock_time.time.side_effect = [0.0, 0.0, 1.0, 2.0, 3.0]
        mock_time.sleep = Mock()
        api = _make_api_instance()
        pending_resp = _make_resource_pending_pod()
        api.read_namespaced_pod.side_effect = [
            pending_resp,
            _FakeApiException(status=500, reason="Internal"),
            pending_resp,
            pending_resp,
        ]
        handle = _make_handle(api=api, timeout=None, pending_timeout=2)

        assert handle.enter_states([JobState.RUNNING]) is False

        assert api.read_namespaced_pod.call_count == 4
        api.delete_namespaced_pod.assert_called_once()
        assert handle.terminal_state == JobState.TERMINATED
        assert handle.poll() == JobReturnCode.EXCEPTION

    @patch("nvflare.app_opt.job_launcher.k8s_launcher.time")
    def test_enter_states_pauses_event_resource_pending_clock_across_event_api_errors(self, mock_time):
        mock_time.time.side_effect = [0.0, 0.0, 1.0, 5.0, 10.0, 11.0]
        mock_time.sleep = Mock()
        api = _make_api_instance()
        running_resp = Mock()
        running_resp.status.phase = PodPhase.RUNNING.value
        api.read_namespaced_pod.side_effect = [
            _make_pod_response(),
            _make_pod_response(),
            _make_pod_response(),
            _make_pod_response(),
            running_resp,
        ]
        api.list_namespaced_event.side_effect = [
            _make_event_list([_make_event("FailedScheduling", "0/1 nodes are available: 1 Insufficient cpu.")]),
            _FakeApiException(status=500, reason="Internal"),
            _FakeApiException(status=500, reason="Internal"),
            _make_event_list([_make_event("FailedScheduling", "0/1 nodes are available: 1 Insufficient cpu.")]),
        ]
        handle = _make_handle(api=api, timeout=None, pending_timeout=2)

        assert handle.enter_states([JobState.RUNNING]) is True

        assert api.read_namespaced_pod.call_count == 5
        api.delete_namespaced_pod.assert_not_called()

    def test_scheduled_pod_event_api_failure_resets_resource_pending_timer(self):
        api = _make_api_instance()
        api.list_namespaced_event.side_effect = _FakeApiException(status=500, reason="Internal")
        handle = _make_handle(api=api, timeout=None, pending_timeout=10)

        assert handle._handle_starting_pod(_make_resource_pending_pod(), PodPhase.PENDING.value, now=0) is False
        assert handle._pending_since == 0

        assert handle._handle_starting_pod(_make_container_creating_pod(), PodPhase.PENDING.value, now=5) is False

        assert handle._pending_since is None
        assert handle._pending_timer_paused_at is None
        api.delete_namespaced_pod.assert_not_called()

    @patch("nvflare.app_opt.job_launcher.k8s_launcher.time")
    def test_enter_states_expires_event_resource_pending_after_event_api_recovery(self, mock_time):
        mock_time.time.side_effect = [0.0, 0.0, 1.0, 5.0, 10.0, 11.0]
        mock_time.sleep = Mock()
        api = _make_api_instance()
        api.read_namespaced_pod.return_value = _make_pod_response()
        api.list_namespaced_event.side_effect = [
            _make_event_list([_make_event("FailedScheduling", "0/1 nodes are available: 1 Insufficient cpu.")]),
            _FakeApiException(status=500, reason="Internal"),
            _FakeApiException(status=500, reason="Internal"),
            _make_event_list([_make_event("FailedScheduling", "0/1 nodes are available: 1 Insufficient cpu.")]),
            _make_event_list([_make_event("FailedScheduling", "0/1 nodes are available: 1 Insufficient cpu.")]),
        ]
        handle = _make_handle(api=api, timeout=None, pending_timeout=2)

        assert handle.enter_states([JobState.RUNNING]) is False

        assert api.read_namespaced_pod.call_count == 5
        api.delete_namespaced_pod.assert_called_once()
        assert handle.terminal_state == JobState.TERMINATED
        assert handle.poll() == JobReturnCode.EXCEPTION

    def test_enter_states_raises_on_invalid_state(self):
        handle = _make_handle()
        with pytest.raises(ValueError, match="expect job_states_to_enter"):
            handle.enter_states(["not_a_state"])

    # -- enter_states: wall-clock timeout branch ------------------------------
    # The pod is scheduled but still creating its container. Pending-resource
    # detection does not fire, so these tests cover the wall-clock timeout branch.

    @patch("nvflare.app_opt.job_launcher.k8s_launcher.time")
    def test_enter_states_wall_clock_timeout_returns_false(self, mock_time):
        mock_time.time.side_effect = [0.0, 100.0]  # start=0, check=100 → 100 > timeout=10
        mock_time.sleep = Mock()
        api = _make_api_instance()
        api.read_namespaced_pod.return_value = _make_container_creating_pod()
        handle = _make_handle(api=api, timeout=10)
        assert handle.enter_states([JobState.RUNNING]) is False

    @patch("nvflare.app_opt.job_launcher.k8s_launcher.time")
    def test_enter_states_wall_clock_timeout_calls_terminate_and_sets_terminal_state(self, mock_time):
        mock_time.time.side_effect = [0.0, 100.0]
        mock_time.sleep = Mock()
        api = _make_api_instance()
        api.read_namespaced_pod.return_value = _make_container_creating_pod()
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
        mock_time.time.side_effect = [0.0, 0.5, 1.0]  # start=0, first check=0.5 < timeout=10
        mock_time.sleep = Mock()
        api = _make_api_instance()
        resp_creating = _make_container_creating_pod()
        resp_succeeded = Mock()
        resp_succeeded.status.phase = PodPhase.SUCCEEDED.value
        api.read_namespaced_pod.side_effect = [resp_creating, resp_succeeded]
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
def _make_k8s_launcher_patches(patch_open=True):
    # kubernetes is imported inside launch_job (not at module level), so we rely
    # on the sys.modules stubs injected at the top of this file.  The only
    # YAML dependency we need to intercept is in the shared study_data helper,
    # plus builtins.open for reading the study-data file. CoreV1Api on the
    # already-stubbed _k8s_core module is patched with patch.object so each
    # test gets a fresh, controllable mock and the original stub is restored.
    # WorkspaceTransferManager is patched so tests don't register real CellNet handlers.
    patches = [
        patch("nvflare.app_opt.job_launcher.study_data.yaml"),
        patch.object(_k8s_core, "CoreV1Api"),
        patch("nvflare.app_opt.job_launcher.k8s_launcher.WorkspaceTransferManager"),
    ]
    if patch_open:
        patches.insert(0, patch("builtins.open", create=True))
    return patches


def _enter_patches(patches):
    mocks = [p.start() for p in patches]
    return mocks


def _exit_patches(patches):
    for p in patches:
        p.stop()


def _setup_launcher(launcher_cls):
    # K8sJobLauncher.__init__ only stores parameters; kubernetes config and the
    # PVC file are loaded lazily inside launch_job, so no mocks are needed here.
    return launcher_cls(
        config_file_path="/fake/kube/config",
        study_data_pvc_file_path="/fake/study_data.yaml",
    )


class TestK8sJobLauncherHandleEvent:
    def test_always_adds_launcher_on_before_job_launch(self):
        from nvflare.app_opt.job_launcher.k8s_launcher import ClientK8sJobLauncher

        patches = _make_k8s_launcher_patches()
        _mock_open, _mock_yaml, _mock_core_cls, _mock_ws_cls, *_ = _enter_patches(patches)
        try:
            launcher = _setup_launcher(ClientK8sJobLauncher)
            fl_ctx = FLContext()
            job_meta = {JobMetaKey.JOB_LAUNCHER_SPEC.value: {"site-1": {"k8s": {"image": "nvflare/custom:v1"}}}}
            fl_ctx.set_prop(FLContextKey.JOB_META, job_meta, private=True, sticky=False)
            fl_ctx.set_prop(ReservedKey.IDENTITY_NAME, "site-1", private=False, sticky=True)

            launcher.handle_event(EventType.BEFORE_JOB_LAUNCH, fl_ctx)

            launchers = fl_ctx.get_prop(FLContextKey.JOB_LAUNCHER)
            assert launchers is not None
            assert launcher in launchers
        finally:
            _exit_patches(patches)

    def test_adds_launcher_even_without_image_in_meta(self):
        # Launcher selection is a site policy (resources.json), not a job decision.
        # K8sJobLauncher always registers; launch_job raises if no image is configured.
        from nvflare.app_opt.job_launcher.k8s_launcher import ClientK8sJobLauncher

        patches = _make_k8s_launcher_patches()
        _mock_open, _mock_yaml, _mock_core_cls, _mock_ws_cls, *_ = _enter_patches(patches)
        try:
            launcher = _setup_launcher(ClientK8sJobLauncher)
            fl_ctx = FLContext()
            job_meta = {}
            fl_ctx.set_prop(FLContextKey.JOB_META, job_meta, private=True, sticky=False)
            fl_ctx.set_prop(ReservedKey.IDENTITY_NAME, "site-1", private=False, sticky=True)

            launcher.handle_event(EventType.BEFORE_JOB_LAUNCH, fl_ctx)

            launchers = fl_ctx.get_prop(FLContextKey.JOB_LAUNCHER)
            assert launchers is not None
            assert launcher in launchers
        finally:
            _exit_patches(patches)


# ---------------------------------------------------------------------------
# K8sJobLauncher __init__
# ---------------------------------------------------------------------------
class TestK8sJobLauncherInit:
    def test_init_stores_parameters(self):
        # __init__ only stores params; kubernetes config and PVC file are loaded
        # lazily inside launch_job, so no mocks are needed.
        from nvflare.app_opt.job_launcher.k8s_launcher import ClientK8sJobLauncher

        launcher = ClientK8sJobLauncher(
            config_file_path="/fake/kube/config",
            study_data_pvc_file_path="/fake/study_data.yaml",
            timeout=60,
            namespace="test-ns",
            image_pull_secrets=["job-regcred"],
        )

        assert launcher.study_data_pvc_file_path == "/fake/study_data.yaml"
        assert launcher.timeout == 60
        assert launcher.namespace == "test-ns"
        assert launcher.workspace_mount_path == WORKSPACE_MOUNT_PATH
        assert launcher.image_pull_secrets == ["job-regcred"]
        # study_data.yaml is populated lazily
        assert launcher.study_data_pvc_dict is None

    def test_init_stores_custom_workspace_mount_path(self):
        from nvflare.app_opt.job_launcher.k8s_launcher import ClientK8sJobLauncher

        launcher = ClientK8sJobLauncher(
            config_file_path="/fake/kube/config",
            study_data_pvc_file_path="/fake/study_data.yaml",
            workspace_mount_path="/mnt/data/nvflare",
        )

        assert launcher.workspace_mount_path == "/mnt/data/nvflare"

    def test_init_rejects_empty_workspace_mount_path(self):
        from nvflare.app_opt.job_launcher.k8s_launcher import ClientK8sJobLauncher

        with pytest.raises(ValueError, match="workspace_mount_path"):
            ClientK8sJobLauncher(
                config_file_path="/fake/kube/config",
                study_data_pvc_file_path="/fake/study_data.yaml",
                workspace_mount_path="",
            )

    def test_init_rejects_non_string_workspace_mount_path(self):
        from nvflare.app_opt.job_launcher.k8s_launcher import ClientK8sJobLauncher

        with pytest.raises(ValueError, match="workspace_mount_path"):
            ClientK8sJobLauncher(
                config_file_path="/fake/kube/config",
                study_data_pvc_file_path="/fake/study_data.yaml",
                workspace_mount_path=1,
            )

    @pytest.mark.parametrize("study_data_pvc_file_path", ["", 1, False])
    def test_init_rejects_invalid_study_data_pvc_file_path(self, study_data_pvc_file_path):
        from nvflare.app_opt.job_launcher.k8s_launcher import ClientK8sJobLauncher

        with pytest.raises(ValueError, match="study_data_pvc_file_path"):
            ClientK8sJobLauncher(
                config_file_path="/fake/kube/config",
                study_data_pvc_file_path=study_data_pvc_file_path,
            )

    def test_init_rejects_empty_ephemeral_storage(self):
        from nvflare.app_opt.job_launcher.k8s_launcher import ClientK8sJobLauncher

        with pytest.raises(ValueError, match="ephemeral_storage"):
            ClientK8sJobLauncher(
                config_file_path="/fake/kube/config",
                study_data_pvc_file_path="/fake/study_data.yaml",
                ephemeral_storage="",
            )

    def test_init_rejects_non_string_ephemeral_storage(self):
        from nvflare.app_opt.job_launcher.k8s_launcher import ClientK8sJobLauncher

        with pytest.raises(ValueError, match="ephemeral_storage"):
            ClientK8sJobLauncher(
                config_file_path="/fake/kube/config",
                study_data_pvc_file_path="/fake/study_data.yaml",
                ephemeral_storage=1,
            )

    @pytest.mark.parametrize("pending_timeout", [-1, "1", True])
    def test_init_rejects_invalid_pending_timeout(self, pending_timeout):
        from nvflare.app_opt.job_launcher.k8s_launcher import ClientK8sJobLauncher

        with pytest.raises(ValueError, match="pending_timeout"):
            ClientK8sJobLauncher(
                config_file_path="/fake/kube/config",
                study_data_pvc_file_path="/fake/study_data.yaml",
                pending_timeout=pending_timeout,
            )

    @pytest.mark.parametrize(
        "image_pull_secrets",
        ["job-regcred", [""], ["  "], [7]],
    )
    def test_init_rejects_invalid_image_pull_secrets(self, image_pull_secrets):
        from nvflare.app_opt.job_launcher.k8s_launcher import ClientK8sJobLauncher

        with pytest.raises(ValueError, match="image_pull_secrets"):
            ClientK8sJobLauncher(
                config_file_path="/fake/kube/config",
                study_data_pvc_file_path="/fake/study_data.yaml",
                image_pull_secrets=image_pull_secrets,
            )


# ---------------------------------------------------------------------------
# ClientK8sJobLauncher.get_module_args
# ---------------------------------------------------------------------------
class TestClientK8sJobLauncherGetModuleArgs:
    def test_returns_dict(self):
        from nvflare.app_opt.job_launcher.k8s_launcher import ClientK8sJobLauncher

        patches = _make_k8s_launcher_patches()
        _mock_open, _mock_yaml, _mock_core_cls, _mock_ws_cls, *_ = _enter_patches(patches)
        try:
            launcher = _setup_launcher(ClientK8sJobLauncher)
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
        _mock_open, _mock_yaml, _mock_core_cls, _mock_ws_cls, *_ = _enter_patches(patches)
        try:
            launcher = _setup_launcher(ClientK8sJobLauncher)
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
        _mock_open, _mock_yaml, _mock_core_cls, _mock_ws_cls, *_ = _enter_patches(patches)
        try:
            launcher = _setup_launcher(ServerK8sJobLauncher)
            fl_ctx = FLContext()
            job_args = {
                JobProcessArgs.WORKSPACE: ("-w", "/workspace"),
                JobProcessArgs.JOB_ID: ("-j", "job-1"),
                JobProcessArgs.ROOT_URL: ("--root_url", "https://server:8003"),
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
        _mock_open, _mock_yaml, _mock_core_cls, _mock_ws_cls, *_ = _enter_patches(patches)
        try:
            launcher = _setup_launcher(ServerK8sJobLauncher)
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
_EXPECTED_POD_NAME = job_pod_name(_EXPECTED_JOB_ID, "site-1")


def _make_launch_job_meta(
    site_name="site-1", image="nvflare/nvflare:latest", gpu=None, study=None, ephemeral_storage=None
):
    k8s_spec = {"image": image}
    if gpu is not None:
        k8s_spec["num_of_gpus"] = gpu
    if ephemeral_storage is not None:
        k8s_spec["ephemeral_storage"] = ephemeral_storage
    meta = {
        JobConstants.JOB_ID: _JOB_UUID,
        JobMetaKey.JOB_LAUNCHER_SPEC.value: {site_name: {"k8s": k8s_spec}},
    }
    if study is not None:
        meta[JobMetaKey.STUDY.value] = study
    return meta


def _make_launch_fl_ctx(
    site_name="site-1",
    set_items=None,
    app_custom_folder="",
    workspace_arg="/var/tmp/nvflare/workspace",
    workspace="/fake/workspace",
):
    fl_ctx = FLContext()
    fl_ctx.set_prop(ReservedKey.IDENTITY_NAME, site_name, private=False, sticky=True)
    job_args = {
        JobProcessArgs.EXE_MODULE: ("-m", _WORKER_MODULE),
        JobProcessArgs.WORKSPACE: ("-w", workspace_arg),
        JobProcessArgs.JOB_ID: ("-n", "job-abc"),
    }
    fl_ctx.set_prop(FLContextKey.JOB_PROCESS_ARGS, job_args, private=True, sticky=False)
    args_obj = Mock()
    args_obj.workspace = workspace
    args_obj.set = set_items
    fl_ctx.set_prop(FLContextKey.ARGS, args_obj, private=False, sticky=False)
    workspace_obj = Mock()
    workspace_obj.get_app_custom_dir.return_value = app_custom_folder
    workspace_obj.get_startup_kit_dir.return_value = "/fake/startup"
    workspace_obj.get_site_config_dir.return_value = "/fake/local"
    fl_ctx.set_prop(FLContextKey.WORKSPACE_OBJECT, workspace_obj, private=True, sticky=False)
    engine = Mock()
    engine.cell = Mock()
    fl_ctx.set_prop(ReservedKey.ENGINE, engine, private=True, sticky=False)
    return fl_ctx


class TestK8sJobLauncherLaunchJob:
    """Integration-style tests that exercise the full launch_job() code path.

    The kubernetes API is mocked but the rest of the code — uuid sanitization,
    manifest construction, enter_states polling, and handle construction — runs
    for real.  read_namespaced_pod is primed to return Running immediately so
    enter_states returns True on the first iteration without sleeping.
    """

    def _setup(
        self,
        patches,
        namespace="test-ns",
        study_data_pvc_dict=None,
        default_python_path=None,
        ephemeral_storage=None,
        workspace_mount_path=None,
        image_pull_secrets=None,
        pending_timeout=None,
        config_file_path="/fake/kube/config",
    ):
        from nvflare.app_opt.job_launcher.k8s_launcher import ClientK8sJobLauncher

        mock_open, mock_yaml, mock_core_cls, mock_transfer_cls, *_ = _enter_patches(patches)
        if study_data_pvc_dict is None:
            study_data_pvc_dict = {"study-a": {"training": {"source": "data-pvc", "mode": "ro"}}}
        mock_yaml.safe_load.return_value = study_data_pvc_dict
        self.mock_open = mock_open
        self.mock_yaml = mock_yaml
        mock_api = MagicMock()
        mock_core_cls.return_value = mock_api
        self.mock_transfer_cls = mock_transfer_cls
        mock_transfer = MagicMock()
        mock_transfer_cls.get_or_create.return_value = mock_transfer
        mock_transfer.owner_fqcn = "site-1.parent"
        mock_transfer.add_job.return_value = "transfer-token"
        self.mock_transfer = mock_transfer
        launcher_kwargs = {
            "config_file_path": config_file_path,
            "study_data_pvc_file_path": "/fake/study_data.yaml",
            "namespace": namespace,
            "default_python_path": default_python_path,
        }
        if ephemeral_storage is not None:
            launcher_kwargs["ephemeral_storage"] = ephemeral_storage
        if workspace_mount_path is not None:
            launcher_kwargs["workspace_mount_path"] = workspace_mount_path
        if image_pull_secrets is not None:
            launcher_kwargs["image_pull_secrets"] = image_pull_secrets
        if pending_timeout is not None:
            launcher_kwargs["pending_timeout"] = pending_timeout
        launcher = ClientK8sJobLauncher(**launcher_kwargs)
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

    def test_incluster_config_is_loaded_when_config_file_path_is_none(self):
        from nvflare.app_opt.job_launcher.k8s_launcher import ClientK8sJobLauncher

        captured = {}

        class FakeConfiguration:
            def get_default_copy(self):
                return self

            @staticmethod
            def set_default(configuration):
                captured["configuration"] = configuration

        patches = _make_k8s_launcher_patches()
        patches.extend(
            [
                patch.object(_k8s_config, "load_incluster_config", create=True),
                patch.object(_k8s_client, "Configuration", FakeConfiguration),
            ]
        )
        _mock_open, mock_yaml, mock_core_cls, mock_transfer_cls, mock_load_incluster_config, *_ = _enter_patches(
            patches
        )
        try:
            mock_yaml.safe_load.return_value = {"study-a": {"training": {"source": "data-pvc", "mode": "ro"}}}
            mock_api = MagicMock()
            mock_core_cls.return_value = mock_api
            mock_transfer = MagicMock()
            mock_transfer_cls.get_or_create.return_value = mock_transfer
            mock_transfer.owner_fqcn = "site-1.parent"
            mock_transfer.add_job.return_value = "transfer-token"
            self._prime_running(mock_api)

            launcher = ClientK8sJobLauncher(
                config_file_path=None,
                study_data_pvc_file_path="/fake/study_data.yaml",
                namespace="test-ns",
            )
            launcher.launch_job(_make_launch_job_meta(), _make_launch_fl_ctx())

            mock_load_incluster_config.assert_called_once()
            assert isinstance(captured["configuration"], FakeConfiguration)
        finally:
            _exit_patches(patches)

    def test_handle_preserves_job_id_and_tracks_pod_name(self):
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches)
        self._prime_running(mock_api)
        try:
            handle = launcher.launch_job(_make_launch_job_meta(), _make_launch_fl_ctx())
            assert handle.job_id == _EXPECTED_JOB_ID
            assert handle.pod_name == _EXPECTED_POD_NAME
            mock_api.read_namespaced_pod.assert_called_once_with(name=_EXPECTED_POD_NAME, namespace="test-ns")
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

    def test_handle_uses_launcher_pending_timeout_by_default(self):
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches, pending_timeout=30)
        self._prime_running(mock_api)
        try:
            handle = launcher.launch_job(_make_launch_job_meta(), _make_launch_fl_ctx())
            assert handle.pending_timeout == 30
        finally:
            _exit_patches(patches)

    def test_handle_uses_job_pending_timeout_override(self):
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches, pending_timeout=30)
        self._prime_running(mock_api)
        try:
            meta = _make_launch_job_meta()
            meta[JobMetaKey.JOB_LAUNCHER_SPEC.value]["site-1"]["k8s"]["pending_timeout"] = 0
            handle = launcher.launch_job(meta, _make_launch_fl_ctx())
            assert handle.pending_timeout == 0
        finally:
            _exit_patches(patches)

    def test_handle_uses_job_pending_timeout_null_override(self):
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches, pending_timeout=30)
        self._prime_running(mock_api)
        try:
            meta = _make_launch_job_meta()
            meta[JobMetaKey.JOB_LAUNCHER_SPEC.value]["site-1"]["k8s"]["pending_timeout"] = None
            handle = launcher.launch_job(meta, _make_launch_fl_ctx())
            assert handle.pending_timeout is None
        finally:
            _exit_patches(patches)

    def test_rejects_invalid_job_pending_timeout_override(self):
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches, pending_timeout=30)
        self._prime_running(mock_api)
        try:
            meta = _make_launch_job_meta()
            meta[JobMetaKey.JOB_LAUNCHER_SPEC.value]["site-1"]["k8s"]["pending_timeout"] = -1
            with pytest.raises(RuntimeError, match="pending_timeout"):
                launcher.launch_job(meta, _make_launch_fl_ctx())
            mock_api.create_namespaced_pod.assert_not_called()
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

    @patch("nvflare.app_opt.job_launcher.k8s_launcher.time")
    def test_launch_job_pending_timeout_zero_deletes_pending_pod_immediately(self, mock_time):
        mock_time.time.side_effect = [0.0, 0.0]
        mock_time.sleep = Mock()
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches)
        mock_api.read_namespaced_pod.return_value = _make_resource_pending_pod()
        try:
            meta = _make_launch_job_meta()
            meta[JobMetaKey.JOB_LAUNCHER_SPEC.value]["site-1"]["k8s"]["pending_timeout"] = 0
            handle = launcher.launch_job(meta, _make_launch_fl_ctx())

            assert handle.terminal_state == JobState.TERMINATED
            assert handle.poll() == JobReturnCode.EXCEPTION
            mock_api.read_namespaced_pod.assert_called_once_with(name=_EXPECTED_POD_NAME, namespace="test-ns")
            mock_api.delete_namespaced_pod.assert_called_once_with(
                name=_EXPECTED_POD_NAME,
                namespace="test-ns",
                grace_period_seconds=0,
            )
        finally:
            _exit_patches(patches)

    # -- pod manifest: identity fields ----------------------------------------

    def test_pod_manifest_name_includes_site_to_avoid_namespace_collisions(self):
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches)
        self._prime_running(mock_api)
        try:
            launcher.launch_job(_make_launch_job_meta(), _make_launch_fl_ctx())
            manifest = mock_api.create_namespaced_pod.call_args.kwargs["body"]
            assert manifest["metadata"]["name"] == _EXPECTED_POD_NAME
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

    def test_pod_manifest_image_pull_secrets_from_launcher_config(self):
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches, image_pull_secrets=["job-regcred", "mirror-regcred"])
        self._prime_running(mock_api)
        try:
            launcher.launch_job(_make_launch_job_meta(), _make_launch_fl_ctx())
            manifest = mock_api.create_namespaced_pod.call_args.kwargs["body"]
            assert manifest["spec"]["imagePullSecrets"] == [
                {"name": "job-regcred"},
                {"name": "mirror-regcred"},
            ]
        finally:
            _exit_patches(patches)

    def test_pod_manifest_uses_study_runtime_pod_template(self, tmp_path):
        from nvflare.app_opt.job_launcher.k8s_launcher import ClientK8sJobLauncher

        pod_dir = tmp_path / "local" / "pod_specs"
        pod_dir.mkdir(parents=True)
        pod_file = pod_dir / "study-a-pod.yaml"
        pod_file.write_text(
            """
apiVersion: v1
kind: Pod
metadata:
  name: template-name
  labels:
    from-template: "true"
spec:
  serviceAccountName: custom-sa
  restartPolicy: Always
  nodeSelector:
    accelerator: gpu
  imagePullSecrets:
    - name: template-regcred
  volumes:
    - name: template-volume
      emptyDir: {}
    - name: workspace-job
      emptyDir:
        sizeLimit: old
    - name: startup-kit
      secret:
        secretName: old-startup-secret
  containers:
    - name: nvflare_job
      image: template-image
      imagePullPolicy: IfNotPresent
      env:
        - name: KEEP_ME
          value: "yes"
        - name: NVFL_WORKSPACE_TRANSFER_TOKEN
          value: old-token
      volumeMounts:
        - name: template-volume
          mountPath: /template
        - name: workspace-job
          mountPath: /old-workspace
        - name: startup-kit
          mountPath: /old-startup
          readOnly: false
      resources:
        limits:
          cpu: "999"
    - name: sidecar
      image: sidecar:v1
""",
            encoding="utf-8",
        )
        (tmp_path / "local" / "study_runtime.yaml").write_text(
            "format_version: 2\n"
            "studies:\n"
            "  study-a:\n"
            "    pod_template: pod_specs/study-a-pod.yaml\n"
            "    datasets:\n"
            "      training:\n"
            "        source: study-train-pvc\n"
            "        mode: ro\n",
            encoding="utf-8",
        )

        patches = _make_k8s_launcher_patches(patch_open=False)
        _mock_study_data_yaml, mock_core_cls, mock_transfer_cls, *_ = _enter_patches(patches)
        try:
            mock_api = MagicMock()
            mock_core_cls.return_value = mock_api
            mock_transfer = MagicMock()
            mock_transfer_cls.get_or_create.return_value = mock_transfer
            mock_transfer.owner_fqcn = "site-1.parent"
            mock_transfer.add_job.return_value = "transfer-token"
            self._prime_running(mock_api)

            launcher = ClientK8sJobLauncher(
                config_file_path="/fake/kube/config",
                namespace="test-ns",
                default_python_path="/usr/bin/python3",
                image_pull_secrets=["job-regcred"],
            )
            meta = _make_launch_job_meta(image="repo/nvflare-job:v2", study="study-a", ephemeral_storage="8Gi")
            meta[JobMetaKey.JOB_LAUNCHER_SPEC.value]["site-1"]["k8s"].update({"cpu": "2", "memory": "8Gi"})

            launcher.launch_job(meta, _make_launch_fl_ctx(workspace=str(tmp_path)))

            manifest = mock_api.create_namespaced_pod.call_args.kwargs["body"]
            assert manifest["metadata"]["name"] == _EXPECTED_POD_NAME
            assert manifest["metadata"]["labels"] == {"from-template": "true"}
            assert manifest["spec"]["serviceAccountName"] == "custom-sa"
            assert manifest["spec"]["nodeSelector"] == {"accelerator": "gpu"}
            assert manifest["spec"]["restartPolicy"] == "Never"
            assert manifest["spec"]["imagePullSecrets"] == [
                {"name": "template-regcred"},
                {"name": "job-regcred"},
            ]

            volume_map = {v["name"]: v for v in manifest["spec"]["volumes"]}
            training_volume = study_dataset_volume_name("study-a", "training")
            assert "template-volume" in volume_map
            assert volume_map["workspace-job"]["emptyDir"]["sizeLimit"] == "8Gi"
            assert volume_map["startup-kit"]["secret"]["secretName"] == "nvflare-startup-site-1-f505dccd"
            assert volume_map[training_volume]["persistentVolumeClaim"]["claimName"] == "study-train-pvc"

            job_container = manifest["spec"]["containers"][0]
            assert job_container["name"] == f"container-{_EXPECTED_JOB_ID}"
            assert job_container["image"] == "repo/nvflare-job:v2"
            assert job_container["command"] == ["/usr/bin/python3"]
            assert job_container["imagePullPolicy"] == "IfNotPresent"
            assert _WORKER_MODULE in job_container["args"]
            assert manifest["spec"]["containers"][1] == {"name": "sidecar", "image": "sidecar:v1"}

            mount_map = {m["name"]: m for m in job_container["volumeMounts"]}
            assert mount_map["template-volume"] == {"name": "template-volume", "mountPath": "/template"}
            assert mount_map["workspace-job"]["mountPath"] == WORKSPACE_MOUNT_PATH
            assert mount_map["startup-kit"]["mountPath"] == f"{WORKSPACE_MOUNT_PATH}/startup"
            assert mount_map["startup-kit"]["readOnly"] is True
            assert mount_map[training_volume] == {
                "name": training_volume,
                "mountPath": "/data/study-a/training",
                "readOnly": True,
            }

            env_map = {e["name"]: e.get("value") for e in job_container["env"]}
            assert env_map["KEEP_ME"] == "yes"
            assert env_map[ENV_WORKSPACE_OWNER_FQCN] == "site-1.parent"
            # the template's literal transfer-token env is replaced by the credential-Secret ref
            token_entry = next(e for e in job_container["env"] if e["name"] == ENV_WORKSPACE_TRANSFER_TOKEN)
            assert "value" not in token_entry
            assert token_entry["valueFrom"]["secretKeyRef"]["key"] == ENV_WORKSPACE_TRANSFER_TOKEN

            resources = job_container["resources"]
            assert resources["limits"]["cpu"] == "2"
            assert resources["limits"]["memory"] == "8Gi"
            assert resources["requests"]["ephemeral-storage"] == "8Gi"
            _mock_study_data_yaml.safe_load.assert_not_called()
        finally:
            _exit_patches(patches)

    def _setup_v2(self, patches, tmp_path, study_runtime_text, pod_templates=None):
        from nvflare.app_opt.job_launcher.k8s_launcher import ClientK8sJobLauncher

        local_dir = tmp_path / "local"
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "study_runtime.yaml").write_text(study_runtime_text, encoding="utf-8")
        for rel_path, text in (pod_templates or {}).items():
            template_path = local_dir / rel_path
            template_path.parent.mkdir(parents=True, exist_ok=True)
            template_path.write_text(text, encoding="utf-8")
        self._mock_study_data_yaml, mock_core_cls, mock_transfer_cls, *_ = _enter_patches(patches)
        mock_api = MagicMock()
        mock_core_cls.return_value = mock_api
        self.mock_transfer_cls = mock_transfer_cls
        mock_transfer = MagicMock()
        mock_transfer_cls.get_or_create.return_value = mock_transfer
        mock_transfer.owner_fqcn = "site-1.parent"
        mock_transfer.add_job.return_value = "transfer-token"
        self.mock_transfer = mock_transfer
        launcher = ClientK8sJobLauncher(config_file_path="/fake/kube/config", namespace="test-ns")
        self._prime_running(mock_api)
        return launcher, mock_api

    def test_pod_manifest_uses_study_runtime_template_without_datasets(self, tmp_path):
        patches = _make_k8s_launcher_patches(patch_open=False)
        launcher, mock_api = self._setup_v2(
            patches,
            tmp_path,
            "format_version: 2\nstudies:\n  study-a:\n    pod_template: study-a-pod.yaml\n",
            pod_templates={
                "study-a-pod.yaml": (
                    "apiVersion: v1\n"
                    "kind: Pod\n"
                    "spec:\n"
                    "  serviceAccountName: custom-sa\n"
                    "  containers:\n"
                    "    - name: nvflare_job\n"
                    "      image: template-image\n"
                )
            },
        )
        try:
            launcher.launch_job(_make_launch_job_meta(study="study-a"), _make_launch_fl_ctx(workspace=str(tmp_path)))

            manifest = mock_api.create_namespaced_pod.call_args.kwargs["body"]
            assert manifest["spec"]["serviceAccountName"] == "custom-sa"
            volume_names = {v["name"] for v in manifest["spec"]["volumes"]}
            mount_names = {m["name"] for m in manifest["spec"]["containers"][0]["volumeMounts"]}
            assert volume_names == {"workspace-job", "startup-kit"}
            assert mount_names == {"workspace-job", "startup-kit"}
            self._mock_study_data_yaml.safe_load.assert_not_called()
        finally:
            _exit_patches(patches)

    def test_pod_template_namespace_mismatch_uses_launcher_namespace(self, tmp_path, caplog):
        patches = _make_k8s_launcher_patches(patch_open=False)
        launcher, mock_api = self._setup_v2(
            patches,
            tmp_path,
            "format_version: 2\nstudies:\n  study-a:\n    pod_template: study-a-pod.yaml\n",
            pod_templates={
                "study-a-pod.yaml": (
                    "apiVersion: v1\n"
                    "kind: Pod\n"
                    "metadata:\n"
                    "  namespace: template-ns\n"
                    "spec:\n"
                    "  containers:\n"
                    "    - name: nvflare_job\n"
                    "      image: template-image\n"
                )
            },
        )
        try:
            with caplog.at_level(logging.WARNING):
                launcher.launch_job(
                    _make_launch_job_meta(study="study-a"), _make_launch_fl_ctx(workspace=str(tmp_path))
                )

            manifest = mock_api.create_namespaced_pod.call_args.kwargs["body"]
            assert manifest["metadata"]["namespace"] == "test-ns"
            assert mock_api.create_namespaced_pod.call_args.kwargs["namespace"] == "test-ns"
            assert (
                "job pod is launched in namespace 'test-ns' instead of metadata.namespace 'template-ns'" in caplog.text
            )
        finally:
            _exit_patches(patches)

    def test_pod_template_edits_take_effect_on_next_launch(self, tmp_path):
        patches = _make_k8s_launcher_patches(patch_open=False)
        first_template = (
            "apiVersion: v1\n"
            "kind: Pod\n"
            "spec:\n"
            "  serviceAccountName: first-sa\n"
            "  containers:\n"
            "    - name: nvflare_job\n"
            "      image: template-image\n"
        )
        launcher, mock_api = self._setup_v2(
            patches,
            tmp_path,
            "format_version: 2\nstudies:\n  study-a:\n    pod_template: study-a-pod.yaml\n",
            pod_templates={"study-a-pod.yaml": first_template},
        )
        try:
            launcher.launch_job(_make_launch_job_meta(study="study-a"), _make_launch_fl_ctx(workspace=str(tmp_path)))
            (tmp_path / "local" / "study-a-pod.yaml").write_text(
                first_template.replace("first-sa", "second-sa"), encoding="utf-8"
            )
            launcher.launch_job(_make_launch_job_meta(study="study-a"), _make_launch_fl_ctx(workspace=str(tmp_path)))

            manifest = mock_api.create_namespaced_pod.call_args.kwargs["body"]
            assert manifest["spec"]["serviceAccountName"] == "second-sa"
        finally:
            _exit_patches(patches)

    def test_pod_manifest_uses_builtin_template_when_study_runtime_missing_study(self, tmp_path, caplog):
        patches = _make_k8s_launcher_patches(patch_open=False)
        launcher, mock_api = self._setup_v2(
            patches,
            tmp_path,
            "format_version: 2\nstudies:\n  other-study:\n    pod_template: other-pod.yaml\n",
            pod_templates={
                "other-pod.yaml": (
                    "apiVersion: v1\n"
                    "kind: Pod\n"
                    "spec:\n"
                    "  serviceAccountName: custom-sa\n"
                    "  containers:\n"
                    "    - name: nvflare_job\n"
                    "      image: template-image\n"
                )
            },
        )
        try:
            with caplog.at_level(logging.WARNING):
                launcher.launch_job(
                    _make_launch_job_meta(study="study-a"), _make_launch_fl_ctx(workspace=str(tmp_path))
                )

            manifest = mock_api.create_namespaced_pod.call_args.kwargs["body"]
            assert "serviceAccountName" not in manifest["spec"]
            assert manifest["spec"]["containers"][0]["image"] == "nvflare/nvflare:latest"
            assert "has no entry for study 'study-a'" in caplog.text
        finally:
            _exit_patches(patches)

    def test_pod_manifest_rejects_non_pod_template(self, tmp_path):
        patches = _make_k8s_launcher_patches(patch_open=False)
        launcher, mock_api = self._setup_v2(
            patches,
            tmp_path,
            "format_version: 2\nstudies:\n  study-a:\n    pod_template: job.yaml\n",
            pod_templates={"job.yaml": "apiVersion: batch/v1\nkind: Job\nmetadata:\n  name: not-a-pod\n"},
        )
        try:
            with pytest.raises(ValueError, match="kind: Pod"):
                launcher.launch_job(
                    _make_launch_job_meta(study="study-a"), _make_launch_fl_ctx(workspace=str(tmp_path))
                )
            mock_api.create_namespaced_pod.assert_not_called()
            self.mock_transfer_cls.get_or_create.assert_not_called()
        finally:
            _exit_patches(patches)

    def test_pod_manifest_inline_pod_template(self, tmp_path):
        patches = _make_k8s_launcher_patches(patch_open=False)
        launcher, mock_api = self._setup_v2(
            patches,
            tmp_path,
            "format_version: 2\n"
            "studies:\n"
            "  study-a:\n"
            "    pod_template:\n"
            "      spec:\n"
            "        serviceAccountName: inline-sa\n"
            "        nodeSelector:\n"
            "          accelerator: gpu\n",
        )
        try:
            launcher.launch_job(_make_launch_job_meta(study="study-a"), _make_launch_fl_ctx(workspace=str(tmp_path)))

            manifest = mock_api.create_namespaced_pod.call_args.kwargs["body"]
            assert manifest["spec"]["serviceAccountName"] == "inline-sa"
            assert manifest["spec"]["nodeSelector"] == {"accelerator": "gpu"}
            assert manifest["spec"]["restartPolicy"] == "Never"
        finally:
            _exit_patches(patches)

    def test_study_runtime_env_and_secret_env_emitted(self, tmp_path):
        patches = _make_k8s_launcher_patches(patch_open=False)
        launcher, mock_api = self._setup_v2(
            patches,
            tmp_path,
            "format_version: 2\n"
            "studies:\n"
            "  study-a:\n"
            "    env:\n"
            "      DB_HOST: postgres.svc\n"
            "    secret_env:\n"
            "      DB_PASSWORD: {source: study-db, key: password}\n",
        )
        try:
            launcher.launch_job(_make_launch_job_meta(study="study-a"), _make_launch_fl_ctx(workspace=str(tmp_path)))

            manifest = mock_api.create_namespaced_pod.call_args.kwargs["body"]
            env_items = {e["name"]: e for e in manifest["spec"]["containers"][0]["env"]}
            assert env_items["DB_HOST"]["value"] == "postgres.svc"
            # launcher-owned transfer token rides the credential Secret alongside the study entries
            token_ref = env_items[ENV_WORKSPACE_TRANSFER_TOKEN]["valueFrom"]["secretKeyRef"]
            assert token_ref["key"] == ENV_WORKSPACE_TRANSFER_TOKEN
            assert env_items["DB_PASSWORD"]["valueFrom"] == {"secretKeyRef": {"name": "study-db", "key": "password"}}
            assert "value" not in env_items["DB_PASSWORD"]
        finally:
            _exit_patches(patches)

    def test_study_runtime_secret_mounts_add_secret_volumes(self, tmp_path):
        from nvflare.app_opt.job_launcher.k8s_launcher import study_secret_volume_name

        patches = _make_k8s_launcher_patches(patch_open=False)
        launcher, mock_api = self._setup_v2(
            patches,
            tmp_path,
            "format_version: 2\n"
            "studies:\n"
            "  study-a:\n"
            "    secret_mounts:\n"
            "      db-ca:\n"
            "        source: study-db-ca\n"
            "        mount_path: /var/run/nvflare/secrets/db-ca\n"
            "        mode: ro\n"
            "        items:\n"
            "          ca.crt: ca.crt\n"
            "      api-token:\n"
            "        source: study-api-token\n"
            "        mount_path: /var/run/nvflare/secrets/api\n",
        )
        try:
            launcher.launch_job(_make_launch_job_meta(study="study-a"), _make_launch_fl_ctx(workspace=str(tmp_path)))

            manifest = mock_api.create_namespaced_pod.call_args.kwargs["body"]
            volume_map = {v["name"]: v for v in manifest["spec"]["volumes"]}
            mount_map = {m["name"]: m for m in manifest["spec"]["containers"][0]["volumeMounts"]}
            ca_volume = study_secret_volume_name("study-a", "db-ca")
            token_volume = study_secret_volume_name("study-a", "api-token")
            assert volume_map[ca_volume]["secret"] == {
                "secretName": "study-db-ca",
                "items": [{"key": "ca.crt", "path": "ca.crt"}],
            }
            assert volume_map[token_volume]["secret"] == {"secretName": "study-api-token"}
            assert mount_map[ca_volume] == {
                "name": ca_volume,
                "mountPath": "/var/run/nvflare/secrets/db-ca",
                "readOnly": True,
            }
            assert mount_map[token_volume]["readOnly"] is True
        finally:
            _exit_patches(patches)

    def test_study_container_image_used_when_job_meta_has_none(self, tmp_path):
        patches = _make_k8s_launcher_patches(patch_open=False)
        launcher, mock_api = self._setup_v2(
            patches,
            tmp_path,
            "format_version: 2\nstudies:\n  study-a:\n    container:\n      image: registry.example.com/study:v9\n",
        )
        try:
            launcher.launch_job(
                _make_launch_job_meta(image=None, study="study-a"), _make_launch_fl_ctx(workspace=str(tmp_path))
            )

            manifest = mock_api.create_namespaced_pod.call_args.kwargs["body"]
            assert manifest["spec"]["containers"][0]["image"] == "registry.example.com/study:v9"
        finally:
            _exit_patches(patches)

    def test_job_meta_image_wins_over_study_container_image(self, tmp_path):
        patches = _make_k8s_launcher_patches(patch_open=False)
        launcher, mock_api = self._setup_v2(
            patches,
            tmp_path,
            "format_version: 2\nstudies:\n  study-a:\n    container:\n      image: registry.example.com/study:v9\n",
        )
        try:
            launcher.launch_job(
                _make_launch_job_meta(image="repo/job-image:v1", study="study-a"),
                _make_launch_fl_ctx(workspace=str(tmp_path)),
            )

            manifest = mock_api.create_namespaced_pod.call_args.kwargs["body"]
            assert manifest["spec"]["containers"][0]["image"] == "repo/job-image:v1"
        finally:
            _exit_patches(patches)

    def test_missing_image_error_mentions_study_runtime(self, tmp_path):
        patches = _make_k8s_launcher_patches(patch_open=False)
        launcher, mock_api = self._setup_v2(patches, tmp_path, "format_version: 2\nstudies: {}\n")
        try:
            with pytest.raises(RuntimeError, match="container.image"):
                launcher.launch_job(
                    _make_launch_job_meta(image=None, study="study-a"), _make_launch_fl_ctx(workspace=str(tmp_path))
                )
            mock_api.create_namespaced_pod.assert_not_called()
        finally:
            _exit_patches(patches)

    def test_study_runtime_datasets_without_template_add_pvc_volumes(self, tmp_path):
        patches = _make_k8s_launcher_patches(patch_open=False)
        launcher, mock_api = self._setup_v2(
            patches,
            tmp_path,
            "format_version: 2\n"
            "studies:\n"
            "  study-a:\n"
            "    datasets:\n"
            "      training:\n"
            "        source: study-train-pvc\n"
            "        mode: ro\n",
        )
        try:
            launcher.launch_job(_make_launch_job_meta(study="study-a"), _make_launch_fl_ctx(workspace=str(tmp_path)))

            manifest = mock_api.create_namespaced_pod.call_args.kwargs["body"]
            training_volume = study_dataset_volume_name("study-a", "training")
            volume_map = {v["name"]: v for v in manifest["spec"]["volumes"]}
            mount_map = {m["name"]: m for m in manifest["spec"]["containers"][0]["volumeMounts"]}
            assert volume_map[training_volume]["persistentVolumeClaim"]["claimName"] == "study-train-pvc"
            assert mount_map[training_volume] == {
                "name": training_volume,
                "mountPath": "/data/study-a/training",
                "readOnly": True,
            }
            self._mock_study_data_yaml.safe_load.assert_not_called()
        finally:
            _exit_patches(patches)

    def test_study_runtime_conflicts_with_v1_file(self, tmp_path):
        patches = _make_k8s_launcher_patches(patch_open=False)
        launcher, mock_api = self._setup_v2(patches, tmp_path, "format_version: 2\nstudies: {}\n")
        (tmp_path / "local" / "study_data.yaml").write_text("study-a: {}\n", encoding="utf-8")
        try:
            with pytest.raises(RuntimeError, match="cannot be combined"):
                launcher.launch_job(
                    _make_launch_job_meta(study="study-a"), _make_launch_fl_ctx(workspace=str(tmp_path))
                )
            mock_api.create_namespaced_pod.assert_not_called()
        finally:
            _exit_patches(patches)

    def test_study_runtime_conflicts_with_existing_pvc_file_arg(self, tmp_path):
        legacy_file = tmp_path / "pvc_map.yaml"
        legacy_file.write_text("study-a: {}\n", encoding="utf-8")
        patches = _make_k8s_launcher_patches(patch_open=False)
        launcher, mock_api = self._setup_v2(patches, tmp_path, "format_version: 2\nstudies: {}\n")
        launcher.study_data_pvc_file_path = str(legacy_file)
        try:
            with pytest.raises(RuntimeError, match="cannot be combined"):
                launcher.launch_job(
                    _make_launch_job_meta(study="study-a"), _make_launch_fl_ctx(workspace=str(tmp_path))
                )
            mock_api.create_namespaced_pod.assert_not_called()
        finally:
            _exit_patches(patches)

    def test_study_runtime_allows_pvc_file_arg_pointing_to_missing_file(self, tmp_path):
        patches = _make_k8s_launcher_patches(patch_open=False)
        launcher, mock_api = self._setup_v2(patches, tmp_path, "format_version: 2\nstudies: {}\n")
        launcher.study_data_pvc_file_path = str(tmp_path / "does-not-exist.yaml")
        try:
            launcher.launch_job(_make_launch_job_meta(study="study-a"), _make_launch_fl_ctx(workspace=str(tmp_path)))
            mock_api.create_namespaced_pod.assert_called_once()
        finally:
            _exit_patches(patches)

    def test_multi_container_template_without_main_marker_fails(self, tmp_path):
        patches = _make_k8s_launcher_patches(patch_open=False)
        launcher, mock_api = self._setup_v2(
            patches,
            tmp_path,
            "format_version: 2\n"
            "studies:\n"
            "  study-a:\n"
            "    pod_template: multi.yaml\n"
            "    env:\n"
            "      DB_HOST: postgres.svc\n",
            pod_templates={
                "multi.yaml": (
                    "apiVersion: v1\n"
                    "kind: Pod\n"
                    "spec:\n"
                    "  containers:\n"
                    "    - name: trainer\n"
                    "      image: a\n"
                    "    - name: sidecar\n"
                    "      image: b\n"
                )
            },
        )
        try:
            with pytest.raises(ValueError, match="mark the main container"):
                launcher.launch_job(
                    _make_launch_job_meta(study="study-a"), _make_launch_fl_ctx(workspace=str(tmp_path))
                )
            mock_api.create_namespaced_pod.assert_not_called()
        finally:
            _exit_patches(patches)

    def test_container_image_with_unmarked_multi_container_template_fails(self, tmp_path):
        # the site-default image must not land on a fallback-selected sidecar
        patches = _make_k8s_launcher_patches(patch_open=False)
        launcher, mock_api = self._setup_v2(
            patches,
            tmp_path,
            "format_version: 2\n"
            "studies:\n"
            "  study-a:\n"
            "    container:\n"
            "      image: registry.example.com/study:v9\n"
            "    pod_template: multi.yaml\n",
            pod_templates={
                "multi.yaml": (
                    "apiVersion: v1\n"
                    "kind: Pod\n"
                    "spec:\n"
                    "  containers:\n"
                    "    - name: log-forwarder\n"
                    "      image: a\n"
                    "    - name: trainer\n"
                    "      image: b\n"
                )
            },
        )
        try:
            with pytest.raises(ValueError, match="mark the main container"):
                launcher.launch_job(
                    _make_launch_job_meta(image=None, study="study-a"), _make_launch_fl_ctx(workspace=str(tmp_path))
                )
            mock_api.create_namespaced_pod.assert_not_called()
        finally:
            _exit_patches(patches)

    def test_docker_kwargs_rejected_on_k8s(self, tmp_path):
        patches = _make_k8s_launcher_patches(patch_open=False)
        launcher, mock_api = self._setup_v2(
            patches,
            tmp_path,
            "format_version: 2\nstudies:\n  study-a:\n    docker_kwargs:\n      shm_size: 8g\n",
        )
        try:
            with pytest.raises(ValueError, match="Docker-only"):
                launcher.launch_job(
                    _make_launch_job_meta(study="study-a"), _make_launch_fl_ctx(workspace=str(tmp_path))
                )
            mock_api.create_namespaced_pod.assert_not_called()
        finally:
            _exit_patches(patches)

    def test_multi_container_template_without_main_container_fails(self, tmp_path):
        # Credential secret_env refs are attached to every job, so a multi-container
        # template must name its main container instead of silently repurposing
        # containers[0] (a sidecar) as the job+credential container.
        patches = _make_k8s_launcher_patches(patch_open=False)
        launcher, mock_api = self._setup_v2(
            patches,
            tmp_path,
            "format_version: 2\nstudies:\n  study-a:\n    pod_template: multi.yaml\n",
            pod_templates={
                "multi.yaml": (
                    "apiVersion: v1\n"
                    "kind: Pod\n"
                    "spec:\n"
                    "  containers:\n"
                    "    - name: trainer\n"
                    "      image: a\n"
                    "    - name: sidecar\n"
                    "      image: b\n"
                )
            },
        )
        try:
            with pytest.raises(ValueError, match="mark the main container"):
                launcher.launch_job(
                    _make_launch_job_meta(study="study-a"), _make_launch_fl_ctx(workspace=str(tmp_path))
                )
            mock_api.create_namespaced_pod.assert_not_called()
        finally:
            _exit_patches(patches)

    def test_secret_env_merges_onto_template_main_container(self, tmp_path):
        patches = _make_k8s_launcher_patches(patch_open=False)
        launcher, mock_api = self._setup_v2(
            patches,
            tmp_path,
            "format_version: 2\n"
            "studies:\n"
            "  study-a:\n"
            "    pod_template: multi.yaml\n"
            "    secret_env:\n"
            "      DB_PASSWORD: {source: study-db, key: password}\n",
            pod_templates={
                "multi.yaml": (
                    "apiVersion: v1\n"
                    "kind: Pod\n"
                    "spec:\n"
                    "  containers:\n"
                    "    - name: nvflare_job\n"
                    "      image: a\n"
                    "      env:\n"
                    "        - name: KEEP_ME\n"
                    "          value: 'yes'\n"
                    "        - name: DB_PASSWORD\n"
                    "          value: template-plaintext\n"
                    "    - name: sidecar\n"
                    "      image: b\n"
                )
            },
        )
        try:
            launcher.launch_job(_make_launch_job_meta(study="study-a"), _make_launch_fl_ctx(workspace=str(tmp_path)))

            manifest = mock_api.create_namespaced_pod.call_args.kwargs["body"]
            trainer = manifest["spec"]["containers"][0]
            assert trainer["name"] == f"container-{_EXPECTED_JOB_ID}"
            env_items = {e["name"]: e for e in trainer["env"]}
            assert env_items["KEEP_ME"]["value"] == "yes"
            # the typed secret_env entry replaces the template's plaintext entry by name
            assert env_items["DB_PASSWORD"]["valueFrom"] == {"secretKeyRef": {"name": "study-db", "key": "password"}}
            assert "value" not in env_items["DB_PASSWORD"]
            sidecar = manifest["spec"]["containers"][1]
            assert "env" not in sidecar
        finally:
            _exit_patches(patches)

    def test_secret_env_must_not_override_launcher_owned_env(self, tmp_path):
        patches = _make_k8s_launcher_patches(patch_open=False)
        launcher, mock_api = self._setup_v2(
            patches,
            tmp_path,
            "format_version: 2\n"
            "studies:\n"
            "  study-a:\n"
            "    secret_env:\n"
            "      NVFL_WORKSPACE_TRANSFER_TOKEN: {source: study-db, key: token}\n",
        )
        try:
            with pytest.raises(ValueError, match="launcher-owned"):
                launcher.launch_job(
                    _make_launch_job_meta(study="study-a"), _make_launch_fl_ctx(workspace=str(tmp_path))
                )
            mock_api.create_namespaced_pod.assert_not_called()
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

    def test_pod_manifest_volume_structure(self):
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches)
        self._prime_running(mock_api)
        try:
            launcher.launch_job(_make_launch_job_meta(), _make_launch_fl_ctx())
            manifest = mock_api.create_namespaced_pod.call_args.kwargs["body"]
            volumes = manifest["spec"]["volumes"]
            vol_map = {v["name"]: v for v in volumes}
            # workspace uses ephemeral emptyDir (no shared PVC)
            assert "emptyDir" in vol_map["workspace-job"]
            # startup/ delivered via Secret
            assert "secret" in vol_map["startup-kit"]
            # no study in meta means no study-data mounts
            assert len(volumes) == 2
        finally:
            _exit_patches(patches)

    def test_pod_manifest_uses_custom_workspace_mount_path(self):
        patches = _make_k8s_launcher_patches()
        workspace_mount_path = "/mnt/data/nvflare"
        launcher, mock_api = self._setup(patches, workspace_mount_path=workspace_mount_path)
        self._prime_running(mock_api)
        try:
            app_custom_folder = f"/fake/workspace/{_JOB_UUID}/app_site-1/custom"
            fl_ctx = _make_launch_fl_ctx(app_custom_folder=app_custom_folder, workspace_arg=workspace_mount_path)
            launcher.launch_job(_make_launch_job_meta(), fl_ctx)
            manifest = mock_api.create_namespaced_pod.call_args.kwargs["body"]
            container = manifest["spec"]["containers"][0]
            mount_map = {m["name"]: m for m in container["volumeMounts"]}
            env_map = {e["name"]: e.get("value") for e in container["env"]}

            assert mount_map["workspace-job"]["mountPath"] == workspace_mount_path
            assert mount_map["startup-kit"] == {
                "name": "startup-kit",
                "mountPath": f"{workspace_mount_path}/startup",
                "readOnly": True,
            }
            assert env_map["PYTHONPATH"] == f"{workspace_mount_path}/{_JOB_UUID}/app_site-1/custom"
            assert container["args"][container["args"].index("-w") + 1] == workspace_mount_path
        finally:
            _exit_patches(patches)

    def test_pod_manifest_ephemeral_storage_uses_launcher_default(self):
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches, ephemeral_storage="3Gi")
        self._prime_running(mock_api)
        try:
            launcher.launch_job(_make_launch_job_meta(), _make_launch_fl_ctx())
            manifest = mock_api.create_namespaced_pod.call_args.kwargs["body"]
            vol_map = {v["name"]: v for v in manifest["spec"]["volumes"]}
            resources = manifest["spec"]["containers"][0]["resources"]
            assert vol_map["workspace-job"]["emptyDir"]["sizeLimit"] == "3Gi"
            assert resources["requests"]["ephemeral-storage"] == "3Gi"
            assert resources["limits"]["ephemeral-storage"] == "3Gi"
        finally:
            _exit_patches(patches)

    def test_pod_manifest_ephemeral_storage_from_launcher_spec(self):
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches, ephemeral_storage="3Gi")
        self._prime_running(mock_api)
        try:
            launcher.launch_job(_make_launch_job_meta(ephemeral_storage="8Gi"), _make_launch_fl_ctx())
            manifest = mock_api.create_namespaced_pod.call_args.kwargs["body"]
            vol_map = {v["name"]: v for v in manifest["spec"]["volumes"]}
            resources = manifest["spec"]["containers"][0]["resources"]
            assert vol_map["workspace-job"]["emptyDir"]["sizeLimit"] == "8Gi"
            assert resources["requests"]["ephemeral-storage"] == "8Gi"
            assert resources["limits"]["ephemeral-storage"] == "8Gi"
        finally:
            _exit_patches(patches)

    def test_pod_manifest_ephemeral_storage_null_uses_launcher_default(self):
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches, ephemeral_storage="3Gi")
        self._prime_running(mock_api)
        try:
            meta = _make_launch_job_meta()
            meta[JobMetaKey.JOB_LAUNCHER_SPEC.value]["site-1"]["k8s"]["ephemeral_storage"] = None
            launcher.launch_job(meta, _make_launch_fl_ctx())
            manifest = mock_api.create_namespaced_pod.call_args.kwargs["body"]
            vol_map = {v["name"]: v for v in manifest["spec"]["volumes"]}
            resources = manifest["spec"]["containers"][0]["resources"]
            assert vol_map["workspace-job"]["emptyDir"]["sizeLimit"] == "3Gi"
            assert resources["requests"]["ephemeral-storage"] == "3Gi"
            assert resources["limits"]["ephemeral-storage"] == "3Gi"
        finally:
            _exit_patches(patches)

    def test_pod_manifest_rejects_non_string_ephemeral_storage_from_launcher_spec(self):
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches, ephemeral_storage="3Gi")
        self._prime_running(mock_api)
        try:
            meta = _make_launch_job_meta(ephemeral_storage=1)
            with pytest.raises(RuntimeError, match="ephemeral_storage"):
                launcher.launch_job(meta, _make_launch_fl_ctx())
            mock_api.create_namespaced_pod.assert_not_called()
        finally:
            _exit_patches(patches)

    def test_pod_manifest_ephemeral_storage_from_default_launcher_spec(self):
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches, ephemeral_storage="3Gi")
        self._prime_running(mock_api)
        try:
            meta = {
                JobConstants.JOB_ID: _JOB_UUID,
                JobMetaKey.JOB_LAUNCHER_SPEC.value: {
                    "default": {"k8s": {"image": "nvflare/nvflare:latest", "ephemeral_storage": "4Gi"}},
                    "site-1": {"k8s": {}},
                },
            }
            launcher.launch_job(meta, _make_launch_fl_ctx())
            manifest = mock_api.create_namespaced_pod.call_args.kwargs["body"]
            vol_map = {v["name"]: v for v in manifest["spec"]["volumes"]}
            resources = manifest["spec"]["containers"][0]["resources"]
            assert vol_map["workspace-job"]["emptyDir"]["sizeLimit"] == "4Gi"
            assert resources["requests"]["ephemeral-storage"] == "4Gi"
            assert resources["limits"]["ephemeral-storage"] == "4Gi"
        finally:
            _exit_patches(patches)

    def test_pod_manifest_uses_study_dataset_pvcs(self):
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(
            patches,
            study_data_pvc_dict={
                "study-a": {
                    "training": {"source": "study-train-pvc", "mode": "ro"},
                    "output": {"source": "study-output-pvc", "mode": "rw"},
                }
            },
        )
        self._prime_running(mock_api)
        try:
            launcher.launch_job(_make_launch_job_meta(study="study-a"), _make_launch_fl_ctx())
            manifest = mock_api.create_namespaced_pod.call_args.kwargs["body"]
            vol_map = {v["name"]: v for v in manifest["spec"]["volumes"]}
            mount_map = {m["mountPath"]: m for m in manifest["spec"]["containers"][0]["volumeMounts"]}
            training_volume = study_dataset_volume_name("study-a", "training")
            output_volume = study_dataset_volume_name("study-a", "output")
            assert vol_map[training_volume]["persistentVolumeClaim"]["claimName"] == "study-train-pvc"
            assert vol_map[output_volume]["persistentVolumeClaim"]["claimName"] == "study-output-pvc"
            assert mount_map["/data/study-a/training"] == {
                "name": training_volume,
                "mountPath": "/data/study-a/training",
                "readOnly": True,
            }
            assert mount_map["/data/study-a/output"] == {
                "name": output_volume,
                "mountPath": "/data/study-a/output",
                "readOnly": False,
            }
        finally:
            _exit_patches(patches)

    def test_pod_manifest_omits_data_pvc_when_no_study_in_job_meta(self):
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches)
        self._prime_running(mock_api)
        try:
            launcher.launch_job(_make_launch_job_meta(), _make_launch_fl_ctx())
            manifest = mock_api.create_namespaced_pod.call_args.kwargs["body"]
            vol_names = {v["name"] for v in manifest["spec"]["volumes"]}
            mount_names = {m["name"] for m in manifest["spec"]["containers"][0]["volumeMounts"]}
            assert vol_names == {"workspace-job", "startup-kit"}
            assert mount_names == {"workspace-job", "startup-kit"}
        finally:
            _exit_patches(patches)

    def test_pod_manifest_omits_data_pvc_for_default_study(self):
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches)
        self._prime_running(mock_api)
        try:
            launcher.launch_job(_make_launch_job_meta(study="default"), _make_launch_fl_ctx())
            manifest = mock_api.create_namespaced_pod.call_args.kwargs["body"]
            vol_names = {v["name"] for v in manifest["spec"]["volumes"]}
            mount_names = {m["name"] for m in manifest["spec"]["containers"][0]["volumeMounts"]}
            assert vol_names == {"workspace-job", "startup-kit"}
            assert mount_names == {"workspace-job", "startup-kit"}
            self.mock_open.assert_called_once_with("/fake/study_data.yaml", "rt")
        finally:
            _exit_patches(patches)

    def test_pod_manifest_uses_default_study_mapping_when_present(self):
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(
            patches,
            study_data_pvc_dict={"default": {"training": {"source": "default-train-pvc", "mode": "ro"}}},
        )
        self._prime_running(mock_api)
        try:
            launcher.launch_job(_make_launch_job_meta(study="default"), _make_launch_fl_ctx())
            manifest = mock_api.create_namespaced_pod.call_args.kwargs["body"]
            vol_map = {v["name"]: v for v in manifest["spec"]["volumes"]}
            mount_map = {m["mountPath"]: m for m in manifest["spec"]["containers"][0]["volumeMounts"]}
            training_volume = study_dataset_volume_name("default", "training")
            assert vol_map[training_volume]["persistentVolumeClaim"]["claimName"] == "default-train-pvc"
            assert mount_map["/data/default/training"] == {
                "name": training_volume,
                "mountPath": "/data/default/training",
                "readOnly": True,
            }
        finally:
            _exit_patches(patches)

    def test_pod_manifest_omits_data_pvc_when_study_mapping_is_missing(self, caplog):
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches, study_data_pvc_dict={"other-study": {}})
        self._prime_running(mock_api)
        try:
            with caplog.at_level(logging.WARNING):
                launcher.launch_job(_make_launch_job_meta(study="study-a"), _make_launch_fl_ctx())
            manifest = mock_api.create_namespaced_pod.call_args.kwargs["body"]
            vol_names = {v["name"] for v in manifest["spec"]["volumes"]}
            mount_names = {m["name"] for m in manifest["spec"]["containers"][0]["volumeMounts"]}
            assert vol_names == {"workspace-job", "startup-kit"}
            assert mount_names == {"workspace-job", "startup-kit"}
            assert "has no entry for study 'study-a'" in caplog.text
        finally:
            _exit_patches(patches)

    def test_pod_manifest_omits_data_pvc_when_study_data_file_missing_for_study_job(self):
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches)
        self.mock_open.side_effect = FileNotFoundError("/fake/study_data.yaml")
        self._prime_running(mock_api)
        try:
            launcher.launch_job(_make_launch_job_meta(study="study-a"), _make_launch_fl_ctx())
            manifest = mock_api.create_namespaced_pod.call_args.kwargs["body"]
            vol_names = {v["name"] for v in manifest["spec"]["volumes"]}
            mount_names = {m["name"] for m in manifest["spec"]["containers"][0]["volumeMounts"]}
            assert vol_names == {"workspace-job", "startup-kit"}
            assert mount_names == {"workspace-job", "startup-kit"}
            self.mock_yaml.safe_load.assert_not_called()
        finally:
            _exit_patches(patches)

    def test_pod_manifest_omits_data_pvc_when_study_data_file_missing_without_study(self):
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches)
        self.mock_open.side_effect = FileNotFoundError("/fake/study_data.yaml")
        self._prime_running(mock_api)
        try:
            launcher.launch_job(_make_launch_job_meta(), _make_launch_fl_ctx())
            manifest = mock_api.create_namespaced_pod.call_args.kwargs["body"]
            vol_names = {v["name"] for v in manifest["spec"]["volumes"]}
            mount_names = {m["name"] for m in manifest["spec"]["containers"][0]["volumeMounts"]}
            assert vol_names == {"workspace-job", "startup-kit"}
            assert mount_names == {"workspace-job", "startup-kit"}
            self.mock_open.assert_not_called()
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

    def test_pod_manifest_gpu_limit_from_flat_resource_spec(self):
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches)
        self._prime_running(mock_api)
        try:
            meta = _make_launch_job_meta()
            meta[JobMetaKey.RESOURCE_SPEC.value] = {"site-1": {"num_of_gpus": 2}}
            launcher.launch_job(meta, _make_launch_fl_ctx())
            manifest = mock_api.create_namespaced_pod.call_args.kwargs["body"]
            assert manifest["spec"]["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] == 2
        finally:
            _exit_patches(patches)

    def test_pod_manifest_python_path_from_launcher_spec_overrides_default(self):
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches, default_python_path="/usr/bin/python")
        self._prime_running(mock_api)
        try:
            meta = _make_launch_job_meta()
            meta[JobMetaKey.JOB_LAUNCHER_SPEC.value]["site-1"]["k8s"]["python_path"] = "/opt/conda/bin/python"
            launcher.launch_job(meta, _make_launch_fl_ctx())
            manifest = mock_api.create_namespaced_pod.call_args.kwargs["body"]
            assert manifest["spec"]["containers"][0]["command"] == ["/opt/conda/bin/python"]
        finally:
            _exit_patches(patches)

    def test_pod_manifest_gpu_limit_from_legacy_nested_resource_spec(self):
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches)
        self._prime_running(mock_api)
        try:
            meta = {
                JobConstants.JOB_ID: _JOB_UUID,
                JobMetaKey.RESOURCE_SPEC.value: {
                    "site-1": {"k8s": {"image": "nvflare/nvflare:latest", "num_of_gpus": 2}}
                },
            }
            launcher.launch_job(meta, _make_launch_fl_ctx())
            manifest = mock_api.create_namespaced_pod.call_args.kwargs["body"]
            assert manifest["spec"]["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] == 2
        finally:
            _exit_patches(patches)

    def test_pod_manifest_launcher_spec_gpu_overrides_flat_resource_spec_gpu(self):
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches)
        self._prime_running(mock_api)
        try:
            meta = _make_launch_job_meta(gpu=3)
            meta[JobMetaKey.RESOURCE_SPEC.value] = {"site-1": {"num_of_gpus": 2}}
            launcher.launch_job(meta, _make_launch_fl_ctx())
            manifest = mock_api.create_namespaced_pod.call_args.kwargs["body"]
            assert manifest["spec"]["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] == 3
        finally:
            _exit_patches(patches)

    def test_pod_manifest_flat_gpu_ignored_when_resource_spec_has_mode_keys(self):
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches)
        self._prime_running(mock_api)
        try:
            meta = _make_launch_job_meta()
            meta[JobMetaKey.RESOURCE_SPEC.value] = {"site-1": {"k8s": {"image": "unused"}, "num_of_gpus": 2}}
            launcher.launch_job(meta, _make_launch_fl_ctx())
            manifest = mock_api.create_namespaced_pod.call_args.kwargs["body"]
            assert "nvidia.com/gpu" not in manifest["spec"]["containers"][0]["resources"]["limits"]
        finally:
            _exit_patches(patches)

    def test_pod_manifest_omits_gpu_limit_when_not_specified(self):
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches)
        self._prime_running(mock_api)
        try:
            launcher.launch_job(_make_launch_job_meta(), _make_launch_fl_ctx())
            manifest = mock_api.create_namespaced_pod.call_args.kwargs["body"]
            resources = manifest["spec"]["containers"][0]["resources"]
            assert "nvidia.com/gpu" not in resources.get("limits", {})
        finally:
            _exit_patches(patches)

    def test_pod_manifest_cpu_memory_limits_from_launcher_spec(self):
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches)
        self._prime_running(mock_api)
        try:
            meta = _make_launch_job_meta()
            meta[JobMetaKey.JOB_LAUNCHER_SPEC.value]["site-1"]["k8s"].update({"cpu": "500m", "memory": "2Gi"})
            launcher.launch_job(meta, _make_launch_fl_ctx())
            manifest = mock_api.create_namespaced_pod.call_args.kwargs["body"]
            limits = manifest["spec"]["containers"][0]["resources"]["limits"]
            assert limits["cpu"] == "500m"
            assert limits["memory"] == "2Gi"
            assert "nvidia.com/gpu" not in limits
        finally:
            _exit_patches(patches)

    def test_pod_manifest_gpu_and_cpu_combined(self):
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches)
        self._prime_running(mock_api)
        try:
            meta = _make_launch_job_meta(gpu=2)
            meta[JobMetaKey.JOB_LAUNCHER_SPEC.value]["site-1"]["k8s"].update({"cpu": "1000m", "memory": "4Gi"})
            launcher.launch_job(meta, _make_launch_fl_ctx())
            manifest = mock_api.create_namespaced_pod.call_args.kwargs["body"]
            limits = manifest["spec"]["containers"][0]["resources"]["limits"]
            assert limits["nvidia.com/gpu"] == 2
            assert limits["cpu"] == "1000m"
            assert limits["memory"] == "4Gi"
        finally:
            _exit_patches(patches)

    def test_pod_manifest_cpu_memory_mirrored_to_requests(self):
        # AKS deployment safeguards require explicit cpu/memory requests.
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches)
        self._prime_running(mock_api)
        try:
            meta = _make_launch_job_meta()
            meta[JobMetaKey.JOB_LAUNCHER_SPEC.value]["site-1"]["k8s"].update({"cpu": "500m", "memory": "2Gi"})
            launcher.launch_job(meta, _make_launch_fl_ctx())
            manifest = mock_api.create_namespaced_pod.call_args.kwargs["body"]
            resources = manifest["spec"]["containers"][0]["resources"]
            assert resources["requests"]["cpu"] == "500m"
            assert resources["requests"]["memory"] == "2Gi"
            assert resources["limits"]["cpu"] == "500m"
            assert resources["limits"]["memory"] == "2Gi"
        finally:
            _exit_patches(patches)

    def test_pod_manifest_cpu_request_override(self):
        # cpu_request / memory_request allow request < limit (burstable QoS).
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches)
        self._prime_running(mock_api)
        try:
            meta = _make_launch_job_meta()
            meta[JobMetaKey.JOB_LAUNCHER_SPEC.value]["site-1"]["k8s"].update(
                {"cpu": "2000m", "cpu_request": "500m", "memory": "8Gi", "memory_request": "2Gi"}
            )
            launcher.launch_job(meta, _make_launch_fl_ctx())
            manifest = mock_api.create_namespaced_pod.call_args.kwargs["body"]
            resources = manifest["spec"]["containers"][0]["resources"]
            assert resources["limits"]["cpu"] == "2000m"
            assert resources["requests"]["cpu"] == "500m"
            assert resources["limits"]["memory"] == "8Gi"
            assert resources["requests"]["memory"] == "2Gi"
        finally:
            _exit_patches(patches)

    def test_pod_manifest_gpu_mirrored_to_requests(self):
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches)
        self._prime_running(mock_api)
        try:
            launcher.launch_job(_make_launch_job_meta(gpu=2), _make_launch_fl_ctx())
            manifest = mock_api.create_namespaced_pod.call_args.kwargs["body"]
            resources = manifest["spec"]["containers"][0]["resources"]
            assert resources["requests"]["nvidia.com/gpu"] == 2
            assert resources["limits"]["nvidia.com/gpu"] == 2
        finally:
            _exit_patches(patches)

    def test_pod_manifest_no_cpu_memory_requests_when_not_specified(self):
        # When cpu/memory are absent from launcher_spec, requests has only ephemeral-storage.
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches)
        self._prime_running(mock_api)
        try:
            launcher.launch_job(_make_launch_job_meta(), _make_launch_fl_ctx())
            manifest = mock_api.create_namespaced_pod.call_args.kwargs["body"]
            requests = manifest["spec"]["containers"][0]["resources"]["requests"]
            assert "cpu" not in requests
            assert "memory" not in requests
            assert "ephemeral-storage" in requests
        finally:
            _exit_patches(patches)

    # -- workspace object guard -----------------------------------------------

    def test_raises_when_workspace_object_missing(self):
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches)
        try:
            fl_ctx = _make_launch_fl_ctx()
            fl_ctx.set_prop(FLContextKey.WORKSPACE_OBJECT, None, private=True, sticky=False)
            with pytest.raises(RuntimeError, match="workspace"):
                launcher.launch_job(_make_launch_job_meta(), fl_ctx)
        finally:
            _exit_patches(patches)

    # -- pod manifest: PYTHONPATH env var -------------------------------------

    def test_pod_manifest_pythonpath_env_set_when_custom_folder_present(self):
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches)
        self._prime_running(mock_api)
        try:
            app_custom_folder = f"/fake/workspace/{_JOB_UUID}/app_site-1/custom"
            fl_ctx = _make_launch_fl_ctx(app_custom_folder=app_custom_folder)
            launcher.launch_job(_make_launch_job_meta(), fl_ctx)
            manifest = mock_api.create_namespaced_pod.call_args.kwargs["body"]
            container = manifest["spec"]["containers"][0]
            assert "env" in container
            env_map = {e["name"]: e.get("value") for e in container["env"]}
            assert env_map["PYTHONPATH"] == f"{WORKSPACE_MOUNT_PATH}/{_JOB_UUID}/app_site-1/custom"
        finally:
            _exit_patches(patches)

    def test_pod_manifest_rejects_custom_folder_outside_workspace(self):
        patches = _make_k8s_launcher_patches()
        launcher, _mock_api = self._setup(patches)
        self._prime_running(_mock_api)
        try:
            fl_ctx = _make_launch_fl_ctx(app_custom_folder=f"/tmp/fake/workspace/{_JOB_UUID}/app_site-1/custom")
            with pytest.raises(RuntimeError, match="custom folder .* is not under workspace"):
                launcher.launch_job(_make_launch_job_meta(), fl_ctx)
        finally:
            _exit_patches(patches)

    def test_pod_manifest_no_pythonpath_when_no_custom_folder(self):
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches)
        self._prime_running(mock_api)
        try:
            launcher.launch_job(_make_launch_job_meta(), _make_launch_fl_ctx(app_custom_folder=""))
            manifest = mock_api.create_namespaced_pod.call_args.kwargs["body"]
            container = manifest["spec"]["containers"][0]
            env_map = {e["name"]: e.get("value") for e in container.get("env", [])}
            assert "PYTHONPATH" not in env_map
        finally:
            _exit_patches(patches)

    def test_pod_manifest_sets_workspace_owner_fqcn_env(self):
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches)
        self._prime_running(mock_api)
        try:
            launcher.launch_job(_make_launch_job_meta(), _make_launch_fl_ctx())
            manifest = mock_api.create_namespaced_pod.call_args.kwargs["body"]
            env_entries = {e["name"]: e for e in manifest["spec"]["containers"][0].get("env", [])}
            assert env_entries[ENV_WORKSPACE_OWNER_FQCN]["value"] == "site-1.parent"
            # the transfer token rides the credential Secret, never a literal pod env value
            token_entry = env_entries[ENV_WORKSPACE_TRANSFER_TOKEN]
            assert "value" not in token_entry
            assert token_entry["valueFrom"]["secretKeyRef"]["key"] == ENV_WORKSPACE_TRANSFER_TOKEN
        finally:
            _exit_patches(patches)

    def test_launch_job_debug_log_does_not_include_transfer_token(self):
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches)
        self._prime_running(mock_api)
        try:
            with patch.object(launcher.logger, "debug") as debug_log:
                launcher.launch_job(_make_launch_job_meta(), _make_launch_fl_ctx())
            assert debug_log.called
            logged_text = " ".join(str(part) for part in debug_log.call_args.args)
            assert "transfer-token" not in logged_text
            assert ENV_WORKSPACE_TRANSFER_TOKEN not in logged_text
        finally:
            _exit_patches(patches)

    def test_launch_job_uses_get_or_create_with_cell(self):
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches)
        self._prime_running(mock_api)
        try:
            fl_ctx = _make_launch_fl_ctx()
            owner_cell = fl_ctx.get_engine().cell

            launcher.launch_job(_make_launch_job_meta(), fl_ctx)

            self.mock_transfer_cls.get_or_create.assert_called_once_with(owner_cell)
            self.mock_transfer.add_job.assert_called_once_with(_JOB_UUID, "/fake/workspace")
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
            assert handle.poll() == JobReturnCode.EXCEPTION
            self.mock_transfer.remove_job.assert_called_once_with(_JOB_UUID)
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

    def test_startup_secret_failure_removes_transfer_record(self):
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches)
        try:
            with patch.object(launcher, "_ensure_startup_secret", side_effect=OSError("boom")):
                with pytest.raises(OSError, match="boom"):
                    launcher.launch_job(_make_launch_job_meta(), _make_launch_fl_ctx())
            self.mock_transfer.remove_job.assert_called_once_with(_JOB_UUID)
            mock_api.create_namespaced_pod.assert_not_called()
        finally:
            _exit_patches(patches)

    def test_ensure_startup_secret_replaces_when_create_conflicts(self):
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches)
        try:
            launcher.core_v1 = mock_api
            mock_api.create_namespaced_secret.side_effect = _FakeApiException(status=409, reason="Conflict")

            secret_name = launcher._ensure_startup_secret("site-1", "/fake/startup")

            assert secret_name == f"nvflare-startup-{site_name_to_rfc1123('site-1')}"
            mock_api.create_namespaced_secret.assert_called_once()
            mock_api.replace_namespaced_secret.assert_called_once()
            mock_api.read_namespaced_secret.assert_not_called()
        finally:
            _exit_patches(patches)

    # -- PVC file validation (lazy-loaded in launch_job) ----------------------

    def test_raises_on_non_dict_pvc_file(self):
        from nvflare.app_opt.job_launcher.k8s_launcher import ClientK8sJobLauncher

        patches = _make_k8s_launcher_patches()
        mock_open, mock_yaml, _mock_core_cls, _mock_ws_cls, *_ = _enter_patches(patches)
        try:
            mock_yaml.safe_load.return_value = "not-a-dict"
            launcher = ClientK8sJobLauncher(
                config_file_path="/fake/kube/config",
                study_data_pvc_file_path="/fake/study_data.yaml",
            )
            with pytest.raises(ValueError, match="dictionary"):
                launcher.launch_job(_make_launch_job_meta(study="study-a"), _make_launch_fl_ctx())
        finally:
            _exit_patches(patches)

    def test_raises_on_flat_study_data_map(self):
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches, study_data_pvc_dict={"study-a": "pvc-a"})
        self._prime_running(mock_api)
        try:
            with pytest.raises(ValueError, match="study -> dataset"):
                launcher.launch_job(_make_launch_job_meta(study="study-a"), _make_launch_fl_ctx())
            mock_api.create_namespaced_pod.assert_not_called()
        finally:
            _exit_patches(patches)


# ---------------------------------------------------------------------------
# Volume mount list shape (tested via _DEFAULT_VOLUME_MOUNT_LIST defined above)
# ---------------------------------------------------------------------------
class TestVolumeMountList:
    def test_data_mount_is_read_only(self):
        data_mount = next(m for m in _DEFAULT_VOLUME_MOUNT_LIST if m["name"] == _DEFAULT_DATA_VOLUME_NAME)
        assert data_mount["readOnly"] is True

    def test_workspace_always_writable(self):
        ws_mount = next(m for m in _DEFAULT_VOLUME_MOUNT_LIST if m["name"] == "workspace-job")
        assert "readOnly" not in ws_mount

    def test_startup_kit_is_read_only(self):
        startup_mount = next(m for m in _DEFAULT_VOLUME_MOUNT_LIST if m["name"] == "startup-kit")
        assert startup_mount["readOnly"] is True

    def test_mount_paths(self):
        ws_mount = next(m for m in _DEFAULT_VOLUME_MOUNT_LIST if m["name"] == "workspace-job")
        data_mount = next(m for m in _DEFAULT_VOLUME_MOUNT_LIST if m["name"] == _DEFAULT_DATA_VOLUME_NAME)
        assert ws_mount["mountPath"] == "/var/tmp/nvflare/workspace"
        assert data_mount["mountPath"] == "/data/study-a/training"


# ---------------------------------------------------------------------------
# Pod-level security context
# ---------------------------------------------------------------------------
class TestJobHandleSecurityContext:
    def _make_handle_with_sec(self, security_context=None):
        cfg = _make_job_config()
        if security_context is not None:
            cfg["security_context"] = security_context
        return _make_handle(cfg=cfg, namespace="test")

    def test_no_security_context_by_default(self):
        handle = self._make_handle_with_sec()
        manifest = handle.get_manifest()
        assert "securityContext" not in manifest["spec"]

    def test_security_context_applied(self):
        ctx = {"seLinuxOptions": {"type": "spc_t"}}
        handle = self._make_handle_with_sec(security_context=ctx)
        manifest = handle.get_manifest()
        assert manifest["spec"]["securityContext"] == ctx

    def test_security_context_empty_dict_not_applied(self):
        handle = self._make_handle_with_sec(security_context={})
        manifest = handle.get_manifest()
        assert "securityContext" not in manifest["spec"]


# ---------------------------------------------------------------------------
# Credential transport — per-job Secret + env valueFrom secretKeyRef
# ---------------------------------------------------------------------------

_CREDENTIAL_ENV = {
    "NVFLARE_JOB_AUTH_TOKEN": "secret-token",
    "NVFLARE_JOB_TOKEN_SIGNATURE": "secret-signature",
    "NVFLARE_JOB_SSID": "secret-ssid",
}

_EXPECTED_CRED_SECRET_NAME = f"nvflare-cred-{_EXPECTED_POD_NAME}"


def _make_cred_fl_ctx():
    fl_ctx = _make_launch_fl_ctx()
    fl_ctx.get_prop(FLContextKey.JOB_PROCESS_ARGS).update(
        {
            JobProcessArgs.AUTH_TOKEN: ("-t", "secret-token"),
            JobProcessArgs.TOKEN_SIGNATURE: ("-ts", "secret-signature"),
            JobProcessArgs.SSID: ("-d", "secret-ssid"),
        }
    )
    return fl_ctx


def _cred_secret_bodies(mock_api):
    return [
        c.kwargs["body"]
        for c in mock_api.create_namespaced_secret.call_args_list
        if c.kwargs["body"]["metadata"]["name"].startswith("nvflare-cred-")
    ]


class TestK8sCredentialTransport:
    def _setup(self, patches):
        launcher, mock_api = TestK8sJobLauncherLaunchJob._setup(self, patches)
        TestK8sJobLauncherLaunchJob._prime_running(self, mock_api)
        created_pod = Mock()
        created_pod.metadata.uid = "pod-uid-123"
        mock_api.create_namespaced_pod.return_value = created_pod
        return launcher, mock_api

    def test_secret_created_and_pod_references_it_without_values(self):
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches)
        try:
            launcher.launch_job(_make_launch_job_meta(), _make_cred_fl_ctx())

            (body,) = _cred_secret_bodies(mock_api)
            assert body["metadata"] == {"name": _EXPECTED_CRED_SECRET_NAME, "namespace": "test-ns"}
            assert body["type"] == "Opaque"
            assert body["stringData"] == {**_CREDENTIAL_ENV, ENV_WORKSPACE_TRANSFER_TOKEN: "transfer-token"}

            manifest = mock_api.create_namespaced_pod.call_args.kwargs["body"]
            env_by_name = {item["name"]: item for item in manifest["spec"]["containers"][0]["env"]}
            for env_name in list(_CREDENTIAL_ENV) + [ENV_WORKSPACE_TRANSFER_TOKEN]:
                ref = env_by_name[env_name]["valueFrom"]["secretKeyRef"]
                assert ref == {"name": _EXPECTED_CRED_SECRET_NAME, "key": env_name}
            assert "secret-" not in str(manifest)
            assert "transfer-token" not in str(manifest)
            assert not {"-t", "-ts", "-d"} & set(manifest["spec"]["containers"][0]["args"])
        finally:
            _exit_patches(patches)

    def test_secret_owner_reference_patched_after_pod_create(self):
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches)
        try:
            launcher.launch_job(_make_launch_job_meta(), _make_cred_fl_ctx())

            patch_call = mock_api.patch_namespaced_secret.call_args
            assert patch_call.kwargs["name"] == _EXPECTED_CRED_SECRET_NAME
            assert patch_call.kwargs["body"]["metadata"]["ownerReferences"] == [
                {"apiVersion": "v1", "kind": "Pod", "name": _EXPECTED_POD_NAME, "uid": "pod-uid-123"}
            ]
            call_names = [name for name, _args, _kwargs in mock_api.mock_calls]
            assert call_names.index("create_namespaced_pod") < call_names.index("patch_namespaced_secret")
        finally:
            _exit_patches(patches)

    def test_no_bootstrap_credentials_secret_carries_only_transfer_token(self):
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches)
        try:
            launcher.launch_job(_make_launch_job_meta(), _make_launch_fl_ctx())

            (body,) = _cred_secret_bodies(mock_api)
            assert body["stringData"] == {ENV_WORKSPACE_TRANSFER_TOKEN: "transfer-token"}
        finally:
            _exit_patches(patches)

    def test_handle_deletes_secret_at_terminal_state(self):
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches)
        try:
            handle = launcher.launch_job(_make_launch_job_meta(), _make_cred_fl_ctx())
            assert handle.credential_secret_name == _EXPECTED_CRED_SECRET_NAME
            mock_api.delete_namespaced_secret.assert_not_called()

            succeeded = Mock()
            succeeded.status.phase = PodPhase.SUCCEEDED.value
            mock_api.read_namespaced_pod.return_value = succeeded
            assert handle.poll() == JobReturnCode.SUCCESS

            mock_api.delete_namespaced_secret.assert_called_once_with(
                name=_EXPECTED_CRED_SECRET_NAME, namespace="test-ns"
            )
            assert handle.credential_secret_name is None
            handle.poll()
            mock_api.delete_namespaced_secret.assert_called_once()
        finally:
            _exit_patches(patches)

    def test_terminate_deletes_secret(self):
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches)
        try:
            handle = launcher.launch_job(_make_launch_job_meta(), _make_cred_fl_ctx())
            handle.terminate()
            mock_api.delete_namespaced_secret.assert_called_once_with(
                name=_EXPECTED_CRED_SECRET_NAME, namespace="test-ns"
            )
        finally:
            _exit_patches(patches)

    def test_launch_failure_before_pod_create_deletes_secret(self):
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches)
        try:
            mock_api.create_namespaced_pod.side_effect = _FakeApiException(status=403, reason="quota")
            handle = launcher.launch_job(_make_launch_job_meta(), _make_cred_fl_ctx())
            assert handle.terminal_state is JobState.TERMINATED
            mock_api.delete_namespaced_secret.assert_called_once_with(
                name=_EXPECTED_CRED_SECRET_NAME, namespace="test-ns"
            )
        finally:
            _exit_patches(patches)

    def test_replace_404_falls_back_to_create(self):
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches)
        try:
            launcher.core_v1 = mock_api
            mock_api.create_namespaced_secret.side_effect = [
                _FakeApiException(status=409, reason="Conflict"),
                None,
            ]
            mock_api.replace_namespaced_secret.side_effect = _FakeApiException(status=404, reason="Gone")

            name = launcher._create_or_replace_secret("nvflare-cred-x", {"stringData": {"k": "v"}})

            assert name == "nvflare-cred-x"
            assert mock_api.create_namespaced_secret.call_count == 2
        finally:
            _exit_patches(patches)

    def test_secret_replaced_on_conflict_and_owner_patch_failure_is_non_fatal(self):
        patches = _make_k8s_launcher_patches()
        launcher, mock_api = self._setup(patches)
        try:
            mock_api.create_namespaced_secret.side_effect = _FakeApiException(status=409, reason="Conflict")
            mock_api.patch_namespaced_secret.side_effect = _FakeApiException(status=403, reason="Forbidden")

            handle = launcher.launch_job(_make_launch_job_meta(), _make_cred_fl_ctx())

            replaced = [c.kwargs["name"] for c in mock_api.replace_namespaced_secret.call_args_list]
            assert _EXPECTED_CRED_SECRET_NAME in replaced
            assert isinstance(handle, K8sJobHandle)
            assert handle.terminal_state is None
        finally:
            _exit_patches(patches)
