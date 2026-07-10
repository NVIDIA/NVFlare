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

import json
import os
from unittest.mock import MagicMock

import pytest

from nvflare.apis.app_validation import AppValidationKey
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, SystemConfigs
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_exception import UnsafeComponentError
from nvflare.app_common.default_component_policy import DEFAULT_CLASS_ALLOW_LIST
from nvflare.app_common.widgets.component_path_authorizer import (
    ALLOW_ALL,
    CLASS_ALLOW_LIST,
    CLASS_LIST_ENFORCEMENT_MODE,
    ClassListEnforcementMode,
    ComponentPathAuthorizer,
)
from nvflare.fuel.sec.audit import AuditService
from nvflare.fuel.utils.config_service import ConfigService
from nvflare.fuel.utils.json_scanner import Node
from nvflare.private.event import fire_event


class _FakeWorkspace:
    def __init__(self, resources_file):
        self.resources_file = resources_file

    def get_resources_file_path(self):
        return str(self.resources_file)


@pytest.fixture(autouse=True)
def reset_config_service():
    ConfigService.reset()
    yield
    ConfigService.reset()


def _make_fl_ctx(component_config, node_path="component", job_meta=None, workspace=None, job_id=None):
    fl_ctx = FLContext()
    fl_ctx.set_prop(FLContextKey.COMPONENT_CONFIG, component_config, private=True, sticky=False)
    if job_meta is not None:
        fl_ctx.set_prop(FLContextKey.JOB_META, job_meta, private=True, sticky=False)
    if workspace is not None:
        fl_ctx.set_prop(FLContextKey.WORKSPACE_OBJECT, workspace, private=True, sticky=False)
    if job_id is not None:
        fl_ctx.set_prop(FLContextKey.CURRENT_JOB_ID, job_id, private=True, sticky=False)
    node = Node(component_config)
    node.paths = node_path.split(".")
    fl_ctx.set_prop(FLContextKey.COMPONENT_NODE, node, private=True, sticky=False)
    return fl_ctx


def _set_class_allow_list(allow_list):
    ConfigService.add_section(SystemConfigs.RESOURCES_CONF, {CLASS_ALLOW_LIST: allow_list})


def _set_class_policy(allow_list, enforcement_mode):
    if isinstance(enforcement_mode, ClassListEnforcementMode):
        enforcement_mode = enforcement_mode.value
    ConfigService.add_section(
        SystemConfigs.RESOURCES_CONF,
        {CLASS_ALLOW_LIST: allow_list, CLASS_LIST_ENFORCEMENT_MODE: enforcement_mode},
    )


def test_allows_component_on_exact_path_from_class_allow_list():
    component_path = "nvflare.app_common.widgets.metric_relay.MetricRelay"
    _set_class_allow_list([component_path])
    authorizer = ComponentPathAuthorizer()
    fl_ctx = _make_fl_ctx({"path": component_path})

    authorizer.handle_event(EventType.BEFORE_BUILD_COMPONENT, fl_ctx)


def test_allows_component_from_resources_class_allow_list(tmp_path):
    resources_file = tmp_path / "resources.json"
    resources_file.write_text(json.dumps({CLASS_ALLOW_LIST: ["nvflare.app_common.widgets."]}))
    authorizer = ComponentPathAuthorizer()
    fl_ctx = _make_fl_ctx(
        {"path": "nvflare.app_common.widgets.metric_relay.MetricRelay"},
        workspace=_FakeWorkspace(resources_file),
    )

    authorizer.handle_event(EventType.BEFORE_BUILD_COMPONENT, fl_ctx)


def test_direct_authorize_component_config_uses_workspace_class_allow_list(tmp_path):
    component_path = "nvflare.app_common.widgets.metric_relay.MetricRelay"
    resources_file = tmp_path / "resources.json"
    resources_file.write_text(json.dumps({CLASS_ALLOW_LIST: [component_path]}))
    authorizer = ComponentPathAuthorizer()
    node = Node({"class_path": component_path})
    node.paths = ["component"]

    authorizer.authorize_component_config(
        {"class_path": component_path},
        node=node,
        workspace=_FakeWorkspace(resources_file),
    )


def test_workspace_class_allow_list_is_cached(tmp_path, monkeypatch):
    resources_file = tmp_path / "resources.json"
    resources_file.write_text(json.dumps({CLASS_ALLOW_LIST: ["nvflare.app_common.widgets."]}))
    authorizer = ComponentPathAuthorizer()
    fl_ctx = _make_fl_ctx(
        {"path": "nvflare.app_common.widgets.metric_relay.MetricRelay"},
        workspace=_FakeWorkspace(resources_file),
    )
    load_calls = []
    original_json_load = json.load

    def record_json_load(f):
        load_calls.append(f.name)
        return original_json_load(f)

    monkeypatch.setattr("nvflare.app_common.widgets.component_path_authorizer.json.load", record_json_load)

    authorizer.handle_event(EventType.BEFORE_BUILD_COMPONENT, fl_ctx)
    authorizer.handle_event(EventType.BEFORE_BUILD_COMPONENT, fl_ctx)

    assert load_calls == [str(resources_file)]


def test_workspace_class_allow_list_file_is_read_while_cache_lock_held(tmp_path, monkeypatch):
    resources_file = tmp_path / "resources.json"
    resources_file.write_text(json.dumps({CLASS_ALLOW_LIST: ["nvflare.app_common.widgets."]}))
    authorizer = ComponentPathAuthorizer()
    load_lock_states = []
    original_json_load = json.load

    class RecordingLock:
        def __init__(self):
            self.locked = False

        def __enter__(self):
            self.locked = True

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.locked = False

    lock = RecordingLock()
    authorizer._allow_list_cache_lock = lock

    def record_json_load(f):
        load_lock_states.append(lock.locked)
        return original_json_load(f)

    monkeypatch.setattr("nvflare.app_common.widgets.component_path_authorizer.json.load", record_json_load)

    allow_list, _, _ = authorizer._get_policy_from_file(str(resources_file))
    assert allow_list == ["nvflare.app_common.widgets."]
    assert load_lock_states == [True]


def test_workspace_class_allow_list_cache_invalidates_when_file_changes(tmp_path):
    resources_file = tmp_path / "resources.json"
    resources_file.write_text(json.dumps({CLASS_ALLOW_LIST: ["nvflare."]}))
    authorizer = ComponentPathAuthorizer()
    fl_ctx = _make_fl_ctx(
        {"path": "nvflare.app_common.widgets.metric_relay.MetricRelay"},
        workspace=_FakeWorkspace(resources_file),
    )

    authorizer.handle_event(EventType.BEFORE_BUILD_COMPONENT, fl_ctx)

    stat_result = os.stat(resources_file)
    resources_file.write_text(json.dumps({CLASS_ALLOW_LIST: ["subprocess."]}))
    os.utime(resources_file, ns=(stat_result.st_atime_ns + 1_000_000, stat_result.st_mtime_ns + 1_000_000))

    with pytest.raises(UnsafeComponentError, match="MetricRelay.*allow_list"):
        authorizer.handle_event(EventType.BEFORE_BUILD_COMPONENT, fl_ctx)


def test_allows_component_from_config_service_resources():
    _set_class_allow_list(["nvflare.app_common.widgets."])
    authorizer = ComponentPathAuthorizer()
    fl_ctx = _make_fl_ctx({"path": "nvflare.app_common.widgets.metric_relay.MetricRelay"})

    authorizer.handle_event(EventType.BEFORE_BUILD_COMPONENT, fl_ctx)


def test_uses_default_allow_list_when_resources_are_not_initialized():
    authorizer = ComponentPathAuthorizer()
    fl_ctx = _make_fl_ctx({"path": DEFAULT_CLASS_ALLOW_LIST[0]})

    authorizer.handle_event(EventType.BEFORE_BUILD_COMPONENT, fl_ctx)


def test_implicit_default_allow_list_is_audited_and_warned_once_per_job(monkeypatch, caplog):
    audit = MagicMock(return_value="event-id")
    monkeypatch.setattr(AuditService, "add_event", audit)
    authorizer = ComponentPathAuthorizer()

    with caplog.at_level("WARNING"):
        authorizer.handle_event(
            EventType.BEFORE_BUILD_COMPONENT,
            _make_fl_ctx({"path": DEFAULT_CLASS_ALLOW_LIST[0]}, job_id="job-1"),
        )
        authorizer.handle_event(
            EventType.BEFORE_BUILD_COMPONENT,
            _make_fl_ctx({"path": DEFAULT_CLASS_ALLOW_LIST[1]}, job_id="job-1"),
        )

    audit.assert_called_once()
    assert audit.call_args.kwargs["action"] == "component_authorization.default_class_allow_list_used"
    assert "using the built-in default list" in audit.call_args.kwargs["msg"]
    assert "policy source: resources" in audit.call_args.kwargs["msg"]
    assert caplog.text.count("using the built-in default list") == 1


def test_explicit_allow_list_does_not_emit_default_policy_audit(monkeypatch):
    component_path = "nvflare.app_common.widgets.metric_relay.MetricRelay"
    _set_class_allow_list([component_path])
    audit = MagicMock(return_value="event-id")
    monkeypatch.setattr(AuditService, "add_event", audit)

    ComponentPathAuthorizer().handle_event(
        EventType.BEFORE_BUILD_COMPONENT,
        _make_fl_ctx({"path": component_path}, job_id="job-1"),
    )

    audit.assert_not_called()


def test_uses_default_allow_list_when_workspace_resources_omit_class_allow_list(tmp_path, monkeypatch):
    resources_file = tmp_path / "resources.json"
    resources_file.write_text(json.dumps({"format_version": 2}))
    audit = MagicMock(return_value="event-id")
    monkeypatch.setattr(AuditService, "add_event", audit)
    authorizer = ComponentPathAuthorizer()
    fl_ctx = _make_fl_ctx(
        {"path": DEFAULT_CLASS_ALLOW_LIST[0]},
        workspace=_FakeWorkspace(resources_file),
        job_id="job-1",
    )

    authorizer.handle_event(EventType.BEFORE_BUILD_COMPONENT, fl_ctx)

    audit.assert_called_once()
    assert str(resources_file) in audit.call_args.kwargs["msg"]


def test_default_enforcement_mode_rejects_component_missing_from_default_allow_list():
    authorizer = ComponentPathAuthorizer()

    with pytest.raises(UnsafeComponentError, match="custom.Component.*allow_list"):
        authorizer.handle_event(EventType.BEFORE_BUILD_COMPONENT, _make_fl_ctx({"path": "custom.Component"}))


def test_rejects_component_missing_from_allow_list():
    _set_class_allow_list(["nvflare.app_common.widgets."])
    authorizer = ComponentPathAuthorizer()
    fl_ctx = _make_fl_ctx({"path": "subprocess.Popen"})

    with pytest.raises(UnsafeComponentError, match="subprocess.Popen.*allow_list"):
        authorizer.handle_event(EventType.BEFORE_BUILD_COMPONENT, fl_ctx)


def test_wildcard_retries_failed_audit_without_repeating_warning(monkeypatch, caplog):
    _set_class_allow_list([None, ALLOW_ALL, "not-a-valid-prefix"])
    audit = MagicMock(side_effect=[None, "event-id"])
    monkeypatch.setattr(AuditService, "add_event", audit)
    authorizer = ComponentPathAuthorizer()

    with caplog.at_level("WARNING"):
        authorizer.handle_event(
            EventType.BEFORE_BUILD_COMPONENT, _make_fl_ctx({"path": "subprocess.Popen"}, job_id="job-1")
        )
        authorizer.handle_event(
            EventType.BEFORE_BUILD_COMPONENT, _make_fl_ctx({"path": "custom.Component"}, job_id="job-1")
        )
        authorizer.handle_event(
            EventType.BEFORE_BUILD_COMPONENT, _make_fl_ctx({"path": "another.Component"}, job_id="job-1")
        )

    assert audit.call_count == 2
    assert caplog.text.count("remaining allow-list entries are ignored") == 1
    assert audit.call_args.kwargs["action"] == "component_authorization.class_allow_list_disabled"
    assert "contains '*'" in audit.call_args.kwargs["msg"]
    assert "remaining allow-list entries are ignored" in audit.call_args.kwargs["msg"]
    assert "policy source: resources" in audit.call_args.kwargs["msg"]


def test_workspace_wildcard_uses_cached_policy_and_includes_source_in_audit(tmp_path, monkeypatch):
    resources_file = tmp_path / "resources.json"
    resources_file.write_text(json.dumps({CLASS_ALLOW_LIST: [ALLOW_ALL]}))
    audit = MagicMock(return_value="event-id")
    monkeypatch.setattr(AuditService, "add_event", audit)
    authorizer = ComponentPathAuthorizer()

    authorizer.handle_event(
        EventType.BEFORE_BUILD_COMPONENT,
        _make_fl_ctx({"path": "custom.Component"}, workspace=_FakeWorkspace(resources_file), job_id="job-1"),
    )
    authorizer.handle_event(
        EventType.BEFORE_BUILD_COMPONENT,
        _make_fl_ctx({"path": "another.Component"}, workspace=_FakeWorkspace(resources_file), job_id="job-1"),
    )

    audit.assert_called_once()
    assert str(resources_file) in audit.call_args.kwargs["msg"]


def test_wildcard_resolves_workspace_resources_once(tmp_path, monkeypatch):
    resources_file = tmp_path / "resources.json"
    resources_file.write_text(json.dumps({CLASS_ALLOW_LIST: [ALLOW_ALL]}))
    workspace = MagicMock()
    workspace.get_resources_file_path.return_value = str(resources_file)
    monkeypatch.setattr(AuditService, "add_event", MagicMock(return_value="event-id"))

    ComponentPathAuthorizer().handle_event(
        EventType.BEFORE_BUILD_COMPONENT,
        _make_fl_ctx({"path": "custom.Component"}, workspace=workspace, job_id="job-1"),
    )

    workspace.get_resources_file_path.assert_called_once()


def test_wildcard_audit_io_occurs_outside_state_lock(monkeypatch):
    _set_class_allow_list([ALLOW_ALL])
    authorizer = ComponentPathAuthorizer()

    class RecordingLock:
        def __init__(self):
            self.locked = False

        def __enter__(self):
            self.locked = True

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.locked = False

    lock = RecordingLock()
    authorizer._audit_state_lock = lock

    def audit_event(**kwargs):
        assert not lock.locked
        return "event-id"

    monkeypatch.setattr(AuditService, "add_event", audit_event)

    authorizer.handle_event(
        EventType.BEFORE_BUILD_COMPONENT, _make_fl_ctx({"path": "custom.Component"}, job_id="job-1")
    )


def test_wildcard_audit_does_not_collapse_distinct_current_job_ids(monkeypatch):
    _set_class_allow_list([ALLOW_ALL])
    audit = MagicMock(return_value="event-id")
    monkeypatch.setattr(AuditService, "add_event", audit)
    authorizer = ComponentPathAuthorizer()

    authorizer.handle_event(
        EventType.BEFORE_BUILD_COMPONENT, _make_fl_ctx({"path": "custom.Component"}, job_id="job-1")
    )
    authorizer.handle_event(
        EventType.BEFORE_BUILD_COMPONENT, _make_fl_ctx({"path": "custom.Component"}, job_id="job-2")
    )

    assert audit.call_count == 2
    assert [call.kwargs["ref"] for call in audit.call_args_list] == ["job-1", "job-2"]


def test_warn_mode_logs_audits_and_allows_component_missing_from_allow_list(caplog, monkeypatch):
    _set_class_policy(["nvflare."], ClassListEnforcementMode.WARN)
    audit = MagicMock(return_value="event-id")
    monkeypatch.setattr(AuditService, "add_event", audit)
    authorizer = ComponentPathAuthorizer()
    fl_ctx = _make_fl_ctx({"path": "subprocess.Popen"}, node_path="component.args.child", job_id="job-1")

    with caplog.at_level("WARNING"):
        authorizer.handle_event(EventType.BEFORE_BUILD_COMPONENT, fl_ctx)

    assert "subprocess.Popen" in caplog.text
    assert "component.args.child" in caplog.text
    assert f"{CLASS_LIST_ENFORCEMENT_MODE} is '{ClassListEnforcementMode.WARN.value}'" in caplog.text
    audit.assert_called_once()
    assert audit.call_args.kwargs["action"] == "component_authorization.unlisted_class_allowed"
    assert "subprocess.Popen" in audit.call_args.kwargs["msg"]
    assert "policy source: resources" in audit.call_args.kwargs["msg"]


def test_warn_mode_audit_is_deduplicated_per_job_and_component_but_warning_is_not(caplog, monkeypatch):
    _set_class_policy(["nvflare."], ClassListEnforcementMode.WARN)
    audit = MagicMock(return_value="event-id")
    monkeypatch.setattr(AuditService, "add_event", audit)
    authorizer = ComponentPathAuthorizer()

    with caplog.at_level("WARNING"):
        for component_path in ("custom.Component", "custom.Component", "another.Component"):
            authorizer.handle_event(
                EventType.BEFORE_BUILD_COMPONENT, _make_fl_ctx({"path": component_path}, job_id="job-1")
            )

    assert audit.call_count == 2
    assert caplog.text.count("custom.Component") == 2
    assert caplog.text.count("another.Component") == 1


def test_warn_mode_uses_fl_context_logging(monkeypatch):
    _set_class_policy(["nvflare."], ClassListEnforcementMode.WARN)
    authorizer = ComponentPathAuthorizer()
    log_warning = MagicMock()
    monkeypatch.setattr(authorizer, "log_warning", log_warning)
    fl_ctx = _make_fl_ctx({"path": "custom.Component"}, job_id="job-1")

    authorizer.handle_event(EventType.BEFORE_BUILD_COMPONENT, fl_ctx)

    log_warning.assert_called_once()
    assert log_warning.call_args.kwargs["fl_ctx"] is fl_ctx
    assert log_warning.call_args.kwargs["fire_event"] is False


def test_warn_mode_does_not_log_for_allowed_component(caplog):
    _set_class_policy(["nvflare."], ClassListEnforcementMode.WARN)
    authorizer = ComponentPathAuthorizer()

    with caplog.at_level("WARNING"):
        authorizer.handle_event(
            EventType.BEFORE_BUILD_COMPONENT,
            _make_fl_ctx({"path": "nvflare.app_common.widgets.metric_relay.MetricRelay"}),
        )

    assert not caplog.records


def test_explicit_enforce_mode_rejects_component_missing_from_allow_list():
    _set_class_policy(["nvflare."], ClassListEnforcementMode.ENFORCE)
    authorizer = ComponentPathAuthorizer()

    with pytest.raises(UnsafeComponentError, match="subprocess.Popen.*allow_list"):
        authorizer.handle_event(EventType.BEFORE_BUILD_COMPONENT, _make_fl_ctx({"path": "subprocess.Popen"}))


def test_warn_mode_from_workspace_resources_logs_and_allows(tmp_path, caplog):
    resources_file = tmp_path / "resources.json"
    resources_file.write_text(
        json.dumps(
            {
                CLASS_ALLOW_LIST: ["nvflare."],
                CLASS_LIST_ENFORCEMENT_MODE: ClassListEnforcementMode.WARN.value,
            }
        )
    )
    authorizer = ComponentPathAuthorizer()
    fl_ctx = _make_fl_ctx({"path": "custom.Component"}, workspace=_FakeWorkspace(resources_file))

    with caplog.at_level("WARNING"):
        authorizer.handle_event(EventType.BEFORE_BUILD_COMPONENT, fl_ctx)

    assert "custom.Component" in caplog.text
    assert ClassListEnforcementMode.WARN.value in caplog.text


def test_warn_mode_uses_default_allow_list_when_class_allow_list_is_omitted(tmp_path, caplog):
    resources_file = tmp_path / "resources.json"
    resources_file.write_text(
        json.dumps({"format_version": 2, CLASS_LIST_ENFORCEMENT_MODE: ClassListEnforcementMode.WARN.value})
    )
    authorizer = ComponentPathAuthorizer()
    fl_ctx = _make_fl_ctx({"path": "custom.Component"}, workspace=_FakeWorkspace(resources_file))

    with caplog.at_level("WARNING"):
        authorizer.handle_event(EventType.BEFORE_BUILD_COMPONENT, fl_ctx)

    assert "custom.Component" in caplog.text
    assert ClassListEnforcementMode.WARN.value in caplog.text


def test_skips_allow_list_check_for_byoc_job():
    authorizer = ComponentPathAuthorizer()
    fl_ctx = _make_fl_ctx({"path": "subprocess.Popen"}, job_meta={AppValidationKey.BYOC: True})

    authorizer.handle_event(EventType.BEFORE_BUILD_COMPONENT, fl_ctx)


def test_direct_authorize_component_config_skips_all_checks_for_byoc_job():
    authorizer = ComponentPathAuthorizer()
    fl_ctx = FLContext()
    fl_ctx.set_prop(FLContextKey.JOB_META, {AppValidationKey.BYOC: True}, private=True, sticky=False)

    authorizer.authorize_component_config({"path": "subprocess.Popen"}, fl_ctx=fl_ctx)


@pytest.mark.parametrize(
    "allow_list, component_path",
    [
        (["nvflare."], "nvflareevil.app.Component"),
        (["nvflare.app_common"], "nvflare.app_commonx.widgets.Component"),
        (
            ["nvflare.app_common.widgets.metric_relay.MetricRelay"],
            "nvflare.app_common.widgets.metric_relay.MetricRelay2",
        ),
    ],
)
def test_rejects_component_when_allow_entry_is_not_on_path_boundary(allow_list, component_path):
    _set_class_allow_list(allow_list)
    authorizer = ComponentPathAuthorizer()
    fl_ctx = _make_fl_ctx({"path": component_path})

    with pytest.raises(UnsafeComponentError, match="allow_list"):
        authorizer.handle_event(EventType.BEFORE_BUILD_COMPONENT, fl_ctx)


def test_allows_component_on_dotted_prefix_boundary():
    _set_class_allow_list(["nvflare.app_common.widgets"])
    authorizer = ComponentPathAuthorizer()
    fl_ctx = _make_fl_ctx({"path": "nvflare.app_common.widgets.metric_relay.MetricRelay"})

    authorizer.handle_event(EventType.BEFORE_BUILD_COMPONENT, fl_ctx)


def test_allows_component_with_exact_path_allow_entry():
    component_path = "nvflare.app_common.widgets.metric_relay.MetricRelay"
    _set_class_allow_list([component_path])
    authorizer = ComponentPathAuthorizer()
    fl_ctx = _make_fl_ctx({"path": component_path})

    authorizer.handle_event(EventType.BEFORE_BUILD_COMPONENT, fl_ctx)


def test_allows_component_with_class_path():
    component_path = "nvflare.app_common.widgets.metric_relay.MetricRelay"
    _set_class_allow_list([component_path])
    authorizer = ComponentPathAuthorizer()
    fl_ctx = _make_fl_ctx({"class_path": component_path})

    authorizer.handle_event(EventType.BEFORE_BUILD_COMPONENT, fl_ctx)


def test_rejects_component_with_name():
    authorizer = ComponentPathAuthorizer()
    fl_ctx = _make_fl_ctx({"name": "MetricRelay"})

    with pytest.raises(UnsafeComponentError, match="name is not allowed"):
        authorizer.handle_event(EventType.BEFORE_BUILD_COMPONENT, fl_ctx)


def test_direct_authorize_component_config_rejects_name():
    authorizer = ComponentPathAuthorizer()

    with pytest.raises(UnsafeComponentError, match="name is not allowed"):
        authorizer.authorize_component_config({"name": "MetricRelay"})


def test_path_takes_precedence_over_class_path():
    _set_class_allow_list(["nvflare."])
    authorizer = ComponentPathAuthorizer()
    fl_ctx = _make_fl_ctx(
        {"path": "nvflare.app_common.widgets.metric_relay.MetricRelay", "class_path": "subprocess.Popen"}
    )

    authorizer.handle_event(EventType.BEFORE_BUILD_COMPONENT, fl_ctx)


def test_rejects_component_with_path_and_name():
    _set_class_allow_list(["nvflare."])
    authorizer = ComponentPathAuthorizer()
    fl_ctx = _make_fl_ctx({"path": "nvflare.app_common.widgets.metric_relay.MetricRelay", "name": "Popen"})

    with pytest.raises(UnsafeComponentError, match="name is not allowed"):
        authorizer.handle_event(EventType.BEFORE_BUILD_COMPONENT, fl_ctx)


def test_rejects_component_with_path_missing_from_allow_list_and_name():
    _set_class_allow_list(["nvflare."])
    authorizer = ComponentPathAuthorizer()
    fl_ctx = _make_fl_ctx({"path": "subprocess.Popen", "name": "MetricRelay"})

    with pytest.raises(UnsafeComponentError, match="name is not allowed"):
        authorizer.handle_event(EventType.BEFORE_BUILD_COMPONENT, fl_ctx)


def test_rejects_component_with_node_path_in_error():
    _set_class_allow_list(["nvflare."])
    authorizer = ComponentPathAuthorizer()
    fl_ctx = _make_fl_ctx({"path": "subprocess.Popen"}, node_path="component.args.child")

    with pytest.raises(UnsafeComponentError, match="subprocess.Popen.*component.args.child.*allow_list"):
        authorizer.handle_event(EventType.BEFORE_BUILD_COMPONENT, fl_ctx)


def test_event_dispatch_captures_unsafe_component_error():
    _set_class_allow_list(["nvflare."])
    authorizer = ComponentPathAuthorizer()
    fl_ctx = _make_fl_ctx({"path": "subprocess.Popen"})

    fire_event(EventType.BEFORE_BUILD_COMPONENT, [authorizer], fl_ctx)

    exceptions = fl_ctx.get_prop(FLContextKey.EXCEPTIONS)
    assert isinstance(exceptions[authorizer.name], UnsafeComponentError)


def test_ignores_other_events():
    authorizer = ComponentPathAuthorizer()
    fl_ctx = _make_fl_ctx({"path": "subprocess.Popen"})

    authorizer.handle_event(EventType.START_RUN, fl_ctx)


@pytest.mark.parametrize(
    "allow_list",
    [
        [""],
        [None],
        [1],
        ["nvflare"],
        ["subprocess"],
        [" nvflare."],
        ["nvflare. "],
        ["."],
        ["nvflare..app."],
    ],
)
def test_rejects_invalid_allow_list(allow_list):
    _set_class_allow_list(allow_list)
    authorizer = ComponentPathAuthorizer()
    fl_ctx = _make_fl_ctx({"path": "nvflare.app_common.widgets.metric_relay.MetricRelay"})

    with pytest.raises(UnsafeComponentError):
        authorizer.handle_event(EventType.BEFORE_BUILD_COMPONENT, fl_ctx)


@pytest.mark.parametrize("enforcement_mode", [None, True, "", "WARN", "permissive"])
def test_rejects_invalid_enforcement_mode(enforcement_mode):
    _set_class_policy(["nvflare."], enforcement_mode)
    authorizer = ComponentPathAuthorizer()
    fl_ctx = _make_fl_ctx({"path": "nvflare.app_common.widgets.metric_relay.MetricRelay"})

    with pytest.raises(UnsafeComponentError, match=CLASS_LIST_ENFORCEMENT_MODE):
        authorizer.handle_event(EventType.BEFORE_BUILD_COMPONENT, fl_ctx)


def test_empty_allow_list_rejects_component():
    _set_class_allow_list([])
    authorizer = ComponentPathAuthorizer()
    fl_ctx = _make_fl_ctx({"path": "nvflare.app_common.widgets.metric_relay.MetricRelay"})

    with pytest.raises(UnsafeComponentError, match="allow_list"):
        authorizer.handle_event(EventType.BEFORE_BUILD_COMPONENT, fl_ctx)


@pytest.mark.parametrize(
    "component_config",
    [
        {},
        {"path": ""},
        {"path": 1},
        {"path": "subprocess"},
        {"path": "subprocess.Popen\nforged.log.Entry"},
        {"path": "subprocess.-Popen"},
        {"class_path": ""},
        {"class_path": 1},
        {"name": ""},
        [],
    ],
)
def test_rejects_invalid_component_config(component_config):
    _set_class_allow_list(["nvflare."])
    authorizer = ComponentPathAuthorizer()
    fl_ctx = _make_fl_ctx(component_config)

    with pytest.raises(UnsafeComponentError):
        authorizer.handle_event(EventType.BEFORE_BUILD_COMPONENT, fl_ctx)
