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
import threading

import pytest

from nvflare.apis.app_validation import AppValidationKey
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, SystemConfigs
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_exception import UnsafeComponentError
from nvflare.app_common.executors.multi_process_executor import MultiProcessExecutor, WorkerComponentBuilder
from nvflare.app_common.widgets.component_path_authorizer import CLASS_ALLOW_LIST, ComponentPathAuthorizer
from nvflare.fuel.common.excepts import ComponentNotAuthorized
from nvflare.fuel.common.multi_process_executor_constants import CommunicationMetaData
from nvflare.fuel.utils import class_utils
from nvflare.fuel.utils.config_service import ConfigService
from nvflare.fuel.utils.json_scanner import Node
from nvflare.private.fed.app.client.sub_worker_process import SubWorkerExecutor
from nvflare.private.fed.client.client_run_manager import ClientRunManager
from nvflare.private.fed.server.server_engine import ServerEngine
from nvflare.private.fed.utils import fed_utils
from nvflare.private.fed.utils.fed_utils import authorize_build_component
from nvflare.private.json_configer import ConfigContext, JsonConfigurator


class ParentComponent:
    def __init__(self, child=None):
        self.child = child


class MiddleComponent:
    def __init__(self, child=None):
        self.child = child


class NestedComponent:
    instantiated = False

    def __init__(self):
        NestedComponent.instantiated = True


class ContainerComponent:
    instantiated = False

    def __init__(self, child=None, settings=None, children=None, components=None):
        ContainerComponent.instantiated = True
        self.child = child
        self.settings = settings
        self.children = children
        self.components = components


class ThreadBlockingComponent:
    started = None
    release = None

    def __init__(self):
        if ThreadBlockingComponent.started:
            ThreadBlockingComponent.started.set()
        if ThreadBlockingComponent.release and not ThreadBlockingComponent.release.wait(5.0):
            raise RuntimeError("timed out waiting to release ThreadBlockingComponent")


class _TestMultiProcessExecutor(MultiProcessExecutor):
    instantiated = False

    def __init__(self, *args, **kwargs):
        _TestMultiProcessExecutor.instantiated = True
        super().__init__(*args, **kwargs)

    def get_multi_process_command(self) -> str:
        return "python3"


class FakeWorkspace:
    def __init__(self, meta_path, resources_path=None):
        self.meta_path = meta_path
        self.resources_path = resources_path

    def get_job_meta_path(self, job_id: str):
        return self.meta_path

    def get_resources_file_path(self):
        return self.resources_path


class _NestedComponentConfigurator(JsonConfigurator):
    def __init__(self, config_file_name):
        super().__init__(
            config_file_name=config_file_name,
            base_pkgs=["nvflare"],
            module_names=["apis"],
            exclude_libs=True,
        )
        self.component = None

    def process_config_element(self, config_ctx: ConfigContext, node):
        if node.path() == "component":
            self.component = self.authorize_and_build_component(node.element, config_ctx, node)


def _component_path(component_cls):
    return f"{component_cls.__module__}.{component_cls.__qualname__}"


def _test_component_allow_list():
    return [f"{__name__}."]


def _set_class_allow_list(allow_list):
    ConfigService.add_section(SystemConfigs.RESOURCES_CONF, {CLASS_ALLOW_LIST: allow_list})


def _write_nested_component_config(config_file, parent_path, nested_path, middle_path=None):
    nested_config = {
        "path": nested_path,
        "args": {},
    }
    if middle_path:
        nested_config = {
            "path": middle_path,
            "args": {
                "child": nested_config,
            },
        }

    config_file.write_text(
        json.dumps(
            {
                "component": {
                    "path": parent_path,
                    "args": {
                        "child": nested_config,
                    },
                },
            }
        )
    )


def _write_component_config(config_file, component_config):
    config_file.write_text(json.dumps({"component": component_config}))


def _authorize_with_component_path_authorizer(config_dict, config_ctx, node, authorizer, seen=None):
    if seen is not None:
        seen.append((config_dict.get("path") or config_dict.get("class_path") or config_dict.get("name"), node.path()))
    try:
        authorizer.authorize_component_config(config_dict, node)
    except UnsafeComponentError as ex:
        return str(ex)
    return ""


def _make_authorized_configurator(config_file):
    _set_class_allow_list(_test_component_allow_list())
    configurator = _NestedComponentConfigurator(str(config_file))
    configurator.set_component_build_authorizer(
        _authorize_with_component_path_authorizer,
        authorizer=ComponentPathAuthorizer(),
    )
    return configurator


def _record_load_class_calls(monkeypatch):
    loaded_paths = []

    def record_load_class(class_path):
        loaded_paths.append(class_path)
        raise AssertionError(f"load_class should not be called for blocked component {class_path}")

    monkeypatch.setattr(class_utils, "load_class", record_load_class)
    return loaded_paths


def test_nested_component_uses_config_authorizer(tmp_path):
    parent_path = _component_path(ParentComponent)
    nested_path = _component_path(NestedComponent)
    config_file = tmp_path / "config.json"
    _write_nested_component_config(config_file, parent_path, nested_path)

    seen = []

    def authorize(config_dict, config_ctx, node, seen):
        seen.append((config_dict.get("path"), node.path()))
        if config_dict.get("path") == nested_path:
            return "nested component blocked"
        return ""

    NestedComponent.instantiated = False
    configurator = _NestedComponentConfigurator(str(config_file))
    configurator.set_component_build_authorizer(authorize, seen=seen)

    with pytest.raises(ComponentNotAuthorized, match="nested component blocked"):
        configurator.configure()

    assert seen == [(parent_path, "component"), (nested_path, "component.args.child")]
    assert NestedComponent.instantiated is False


def test_deeply_nested_component_uses_config_authorizer(tmp_path):
    parent_path = _component_path(ParentComponent)
    middle_path = _component_path(MiddleComponent)
    nested_path = _component_path(NestedComponent)
    config_file = tmp_path / "config.json"
    _write_nested_component_config(config_file, parent_path, nested_path, middle_path)

    seen = []

    def authorize(config_dict, config_ctx, node, seen):
        seen.append((config_dict.get("path"), node.path()))
        if config_dict.get("path") == nested_path:
            return "deep nested component blocked"
        return ""

    NestedComponent.instantiated = False
    configurator = _NestedComponentConfigurator(str(config_file))
    configurator.set_component_build_authorizer(authorize, seen=seen)

    with pytest.raises(ComponentNotAuthorized, match="deep nested component blocked"):
        configurator.configure()

    assert seen == [
        (parent_path, "component"),
        (middle_path, "component.args.child"),
        (nested_path, "component.args.child.args.child"),
    ]
    assert NestedComponent.instantiated is False


def test_nested_component_reuses_parent_config_ctx_from_build_stack(tmp_path):
    parent_path = _component_path(ParentComponent)
    nested_path = _component_path(NestedComponent)
    config_file = tmp_path / "config.json"
    _write_nested_component_config(config_file, parent_path, nested_path)

    seen = []

    def authorize(config_dict, config_ctx, node, seen):
        seen.append((config_dict.get("path"), config_ctx, node.path()))
        return ""

    configurator = _NestedComponentConfigurator(str(config_file))
    stale_ctx = ConfigContext()
    parent_ctx = ConfigContext()
    parent_node = Node({"path": parent_path, "args": {}})
    parent_node.processor = configurator
    parent_node.key = "component"
    parent_node.paths = ["component"]

    configurator.config_ctx = stale_ctx
    configurator.set_component_build_authorizer(authorize, seen=seen)
    configurator._build_node_stack.append((parent_ctx, parent_node))
    try:
        configurator.build_nested_component({"path": nested_path, "args": {}}, "child")
    finally:
        configurator._build_node_stack.pop()

    assert seen == [(nested_path, parent_ctx, "component.args.child")]


def test_runtime_build_component_uses_config_ctx_before_configure(tmp_path):
    config_file = tmp_path / "config.json"
    _write_component_config(config_file, {"path": _component_path(ContainerComponent), "args": {}})
    seen = []

    def authorize(config_dict, config_ctx, node, seen):
        seen.append((config_ctx, node.path()))
        return "runtime build blocked"

    configurator = _NestedComponentConfigurator(str(config_file))
    configurator.config_ctx = None
    configurator.set_component_build_authorizer(authorize, seen=seen)

    with pytest.raises(ComponentNotAuthorized, match="runtime build blocked"):
        configurator.build_component({"path": _component_path(ContainerComponent), "args": {}})

    assert len(seen) == 1
    assert isinstance(seen[0][0], ConfigContext)
    assert seen[0][0].config_json == configurator.config_data
    assert seen[0][1] == "runtime_component"


def test_runtime_build_authorization_state_is_thread_local(tmp_path):
    config_file = tmp_path / "config.json"
    _write_component_config(config_file, {"path": _component_path(ContainerComponent), "args": {}})
    seen = []
    seen_lock = threading.Lock()

    def authorize(config_dict, config_ctx, node, seen, seen_lock):
        path = config_dict.get("path")
        with seen_lock:
            seen.append((threading.current_thread().name, path, node.path()))
        if path == "subprocess.Popen":
            return "blocked in concurrent build"
        return ""

    configurator = _NestedComponentConfigurator(str(config_file))
    configurator.set_component_build_authorizer(authorize, seen=seen, seen_lock=seen_lock)
    ThreadBlockingComponent.started = threading.Event()
    ThreadBlockingComponent.release = threading.Event()
    errors = []

    def build_blocking_component():
        try:
            configurator.build_component({"path": _component_path(ThreadBlockingComponent), "args": {}})
        except Exception as ex:
            errors.append(ex)

    thread = threading.Thread(target=build_blocking_component, name="blocking-builder")
    thread.start()
    try:
        assert ThreadBlockingComponent.started.wait(5.0)
        with pytest.raises(ComponentNotAuthorized, match="blocked in concurrent build"):
            configurator.build_component({"path": "subprocess.Popen", "args": {}})
    finally:
        ThreadBlockingComponent.release.set()
        thread.join(5.0)
        ThreadBlockingComponent.started = None
        ThreadBlockingComponent.release = None

    assert not thread.is_alive()
    assert errors == []
    assert ("MainThread", "subprocess.Popen", "runtime_component") in seen


def test_deeply_nested_component_path_authorizer_blocks_before_build(tmp_path):
    parent_path = _component_path(ParentComponent)
    middle_path = _component_path(MiddleComponent)
    nested_path = _component_path(NestedComponent)
    config_file = tmp_path / "config.json"
    _write_nested_component_config(config_file, parent_path, nested_path, middle_path)

    fl_ctx = FLContext()
    _set_class_allow_list([parent_path, middle_path])
    authorizer = ComponentPathAuthorizer()
    seen = []

    def authorize_with_component_path_authorizer(config_dict, config_ctx, node, authorizer, fl_ctx, seen):
        seen.append((config_dict.get("path"), node.path()))
        fl_ctx.set_prop(FLContextKey.COMPONENT_CONFIG, config_dict, sticky=False, private=True)
        fl_ctx.set_prop(FLContextKey.CONFIG_CTX, config_ctx, sticky=False, private=True)
        fl_ctx.set_prop(FLContextKey.COMPONENT_NODE, node, sticky=False, private=True)

        try:
            authorizer.handle_event(EventType.BEFORE_BUILD_COMPONENT, fl_ctx)
        except UnsafeComponentError as ex:
            return str(ex)
        return ""

    NestedComponent.instantiated = False
    configurator = _NestedComponentConfigurator(str(config_file))
    configurator.set_component_build_authorizer(
        authorize_with_component_path_authorizer,
        authorizer=authorizer,
        fl_ctx=fl_ctx,
        seen=seen,
    )

    with pytest.raises(ComponentNotAuthorized, match="component.args.child.args.child.*allow_list"):
        configurator.configure()

    assert seen == [
        (parent_path, "component"),
        (middle_path, "component.args.child"),
        (nested_path, "component.args.child.args.child"),
    ]
    assert NestedComponent.instantiated is False


def test_component_path_authorizer_allows_class_path_component_before_build(tmp_path):
    nested_path = _component_path(NestedComponent)
    config_file = tmp_path / "config.json"
    _write_component_config(config_file, {"class_path": nested_path, "args": {}})

    NestedComponent.instantiated = False
    configurator = _make_authorized_configurator(config_file)

    configurator.configure()

    assert isinstance(configurator.component, NestedComponent)
    assert NestedComponent.instantiated is True


@pytest.mark.parametrize(
    "component_config, expected_path",
    [
        ({"path": "subprocess.Popen", "args": {}}, "subprocess.Popen"),
        (
            {
                "path": _component_path(ContainerComponent),
                "args": {"child": {"path": "subprocess.Popen", "args": {}}},
            },
            "component.args.child",
        ),
        (
            {
                "path": _component_path(ContainerComponent),
                "args": {"settings": {"level1": {"level2": {"path": "subprocess.Popen", "args": {}}}}},
            },
            "component.args.settings.level1.level2",
        ),
        (
            {
                "path": _component_path(ContainerComponent),
                "args": {"children": [{"path": "subprocess.Popen", "args": {}}]},
            },
            "component.args.children.#1",
        ),
        (
            {
                "path": _component_path(ContainerComponent),
                "args": {"settings": {"unsafe": {"path": "socket.socket", "args": {}}}},
            },
            "socket.socket",
        ),
        ({"class_path": "subprocess.Popen", "args": {}}, "subprocess.Popen"),
    ],
)
def test_component_path_authorizer_blocks_dangerous_configs_before_build(tmp_path, component_config, expected_path):
    config_file = tmp_path / "config.json"
    _write_component_config(config_file, component_config)

    ContainerComponent.instantiated = False
    NestedComponent.instantiated = False
    configurator = _NestedComponentConfigurator(str(config_file))
    _set_class_allow_list(_test_component_allow_list())
    configurator.set_component_build_authorizer(
        _authorize_with_component_path_authorizer,
        authorizer=ComponentPathAuthorizer(),
    )

    with pytest.raises(ComponentNotAuthorized, match=expected_path):
        configurator.configure()

    assert ContainerComponent.instantiated is False
    assert NestedComponent.instantiated is False


@pytest.mark.parametrize(
    "component_config, expected_match",
    [
        ({"path": "subprocess.Popen", "args": {}}, "subprocess.Popen"),
        (
            {
                "path": _component_path(ContainerComponent),
                "args": {"child": {"path": "subprocess.Popen", "args": {}}},
            },
            "subprocess.Popen.*component.args.child",
        ),
        (
            {
                "path": _component_path(ContainerComponent),
                "args": {"settings": {"level1": {"level2": {"path": "socket.socket", "args": {}}}}},
            },
            "socket.socket.*component.args.settings.level1.level2",
        ),
        (
            {
                "path": _component_path(ContainerComponent),
                "args": {"children": [{"path": "shutil.rmtree", "args": {}}]},
            },
            "shutil.rmtree.*component.args.children.#1",
        ),
        ({"class_path": "subprocess.Popen", "args": {}}, "subprocess.Popen"),
    ],
)
def test_dangerous_component_configs_block_before_instantiation(
    tmp_path, monkeypatch, component_config, expected_match
):
    config_file = tmp_path / "config.json"
    _write_component_config(config_file, component_config)
    loaded_paths = _record_load_class_calls(monkeypatch)

    ContainerComponent.instantiated = False
    NestedComponent.instantiated = False
    configurator = _make_authorized_configurator(config_file)

    with pytest.raises(ComponentNotAuthorized, match=expected_match):
        configurator.configure()

    assert loaded_paths == []
    assert ContainerComponent.instantiated is False
    assert NestedComponent.instantiated is False


@pytest.mark.parametrize(
    "blocked_path",
    [
        "os.system",
        "socket.socket",
        "builtins.eval",
        "ctypes.CDLL",
        "urllib.request.urlopen",
        "http.client.HTTPConnection",
    ],
)
def test_components_missing_from_allow_list_block_before_instantiation(tmp_path, monkeypatch, blocked_path):
    config_file = tmp_path / "config.json"
    _write_component_config(config_file, {"path": blocked_path, "args": {}})
    loaded_paths = _record_load_class_calls(monkeypatch)

    configurator = _make_authorized_configurator(config_file)

    with pytest.raises(ComponentNotAuthorized, match=blocked_path):
        configurator.configure()

    assert loaded_paths == []


@pytest.mark.parametrize("path_key", ["path", "class_path"])
def test_multi_process_components_with_config_type_dict_are_authorized_before_build(tmp_path, path_key):
    config_file = tmp_path / "config.json"
    _write_component_config(
        config_file,
        {
            "path": _component_path(_TestMultiProcessExecutor),
            "args": {
                "executor_id": "bad",
                "num_of_processes": 1,
                "components": [
                    {
                        "id": "bad",
                        "config_type": "dict",
                        path_key: "subprocess.Popen",
                        "args": {},
                    }
                ],
            },
        },
    )

    _TestMultiProcessExecutor.instantiated = False
    configurator = _make_authorized_configurator(config_file)

    with pytest.raises(ComponentNotAuthorized, match=r"subprocess\.Popen.*component\.args\.components\.#1"):
        configurator.configure()

    assert _TestMultiProcessExecutor.instantiated is False


def test_config_type_dict_with_id_and_path_outside_components_list_remains_data(tmp_path):
    config_file = tmp_path / "config.json"
    settings = {
        "id": "dataset",
        "config_type": "dict",
        "path": "subprocess.Popen",
        "args": {},
    }
    _write_component_config(
        config_file,
        {
            "path": _component_path(ContainerComponent),
            "args": {
                "settings": settings,
            },
        },
    )

    ContainerComponent.instantiated = False
    configurator = _make_authorized_configurator(config_file)
    configurator.configure()

    assert isinstance(configurator.component, ContainerComponent)
    assert configurator.component.settings == settings
    assert ContainerComponent.instantiated is True


def test_runtime_build_component_uses_config_authorizer(tmp_path):
    config_file = tmp_path / "config.json"
    _write_component_config(
        config_file,
        {
            "path": _component_path(ContainerComponent),
            "args": {},
        },
    )

    configurator = _NestedComponentConfigurator(str(config_file))
    _set_class_allow_list(_test_component_allow_list())
    configurator.set_component_build_authorizer(
        _authorize_with_component_path_authorizer,
        authorizer=ComponentPathAuthorizer(),
    )

    with pytest.raises(ComponentNotAuthorized, match="subprocess.Popen.*runtime_component"):
        configurator.build_component({"path": "subprocess.Popen", "args": {}})


def test_client_run_manager_build_component_requires_authorizer(tmp_path):
    config_file = tmp_path / "config.json"
    _write_component_config(config_file, {"path": _component_path(ContainerComponent), "args": {}})

    run_manager = ClientRunManager.__new__(ClientRunManager)
    run_manager.conf = _NestedComponentConfigurator(str(config_file))

    with pytest.raises(RuntimeError, match="No component build authorizer"):
        run_manager.build_component({"path": _component_path(ContainerComponent), "args": {}})


def test_client_run_manager_build_component_uses_config_authorizer(tmp_path):
    config_file = tmp_path / "config.json"
    _write_component_config(config_file, {"path": _component_path(ContainerComponent), "args": {}})

    run_manager = ClientRunManager.__new__(ClientRunManager)
    run_manager.conf = _make_authorized_configurator(config_file)

    with pytest.raises(ComponentNotAuthorized, match="subprocess.Popen.*runtime_component"):
        run_manager.build_component({"path": "subprocess.Popen", "args": {}})


def test_server_engine_build_component_requires_authorizer(tmp_path):
    config_file = tmp_path / "config.json"
    _write_component_config(config_file, {"path": _component_path(ContainerComponent), "args": {}})

    engine = ServerEngine.__new__(ServerEngine)
    engine.conf = _NestedComponentConfigurator(str(config_file))

    with pytest.raises(RuntimeError, match="No component build authorizer"):
        engine.build_component({"path": _component_path(ContainerComponent), "args": {}})


def test_server_engine_build_component_uses_config_authorizer(tmp_path):
    config_file = tmp_path / "config.json"
    _write_component_config(config_file, {"path": _component_path(ContainerComponent), "args": {}})

    engine = ServerEngine.__new__(ServerEngine)
    engine.conf = _make_authorized_configurator(config_file)

    with pytest.raises(ComponentNotAuthorized, match="subprocess.Popen.*runtime_component"):
        engine.build_component({"path": "subprocess.Popen", "args": {}})


def test_authorize_build_component_enforces_resources_allow_list_without_event_handlers(tmp_path):
    meta_path = tmp_path / "meta.json"
    meta_path.write_text("{}")
    resources_path = tmp_path / "resources.json"
    resources_path.write_text(json.dumps({"class_allow_list": _test_component_allow_list()}))

    fl_ctx = FLContext()
    fl_ctx.set_prop(
        FLContextKey.WORKSPACE_OBJECT,
        FakeWorkspace(str(meta_path), str(resources_path)),
        sticky=False,
        private=True,
    )
    fl_ctx.set_prop(FLContextKey.CURRENT_JOB_ID, "job1", sticky=False, private=False)

    err = authorize_build_component(
        {"path": "subprocess.Popen", "args": {}},
        ConfigContext(),
        None,
        fl_ctx=fl_ctx,
        event_handlers=[],
    )

    assert "subprocess.Popen" in err
    assert "allow_list" in err


def test_authorize_build_component_skips_allow_list_for_byoc_without_event_handlers(tmp_path):
    meta_path = tmp_path / "meta.json"
    meta_path.write_text(json.dumps({AppValidationKey.BYOC: True}))
    resources_path = tmp_path / "resources.json"
    resources_path.write_text(json.dumps({"class_allow_list": []}))

    fl_ctx = FLContext()
    fl_ctx.set_prop(
        FLContextKey.WORKSPACE_OBJECT,
        FakeWorkspace(str(meta_path), str(resources_path)),
        sticky=False,
        private=True,
    )
    fl_ctx.set_prop(FLContextKey.CURRENT_JOB_ID, "job1", sticky=False, private=False)

    err = authorize_build_component(
        {"path": "subprocess.Popen", "args": {}},
        ConfigContext(),
        None,
        fl_ctx=fl_ctx,
        event_handlers=[],
    )

    assert err == ""


def test_authorize_build_component_caches_job_meta_for_repeated_authorizations(tmp_path, monkeypatch):
    load_calls = []

    def fake_get_job_meta_from_workspace(workspace, job_id):
        load_calls.append((workspace, job_id))
        return {AppValidationKey.BYOC: True}

    monkeypatch.setattr(fed_utils, "get_job_meta_from_workspace", fake_get_job_meta_from_workspace)

    fl_ctx = FLContext()
    workspace = FakeWorkspace(str(tmp_path / "meta.json"))
    fl_ctx.set_prop(FLContextKey.WORKSPACE_OBJECT, workspace, sticky=False, private=True)
    fl_ctx.set_prop(FLContextKey.CURRENT_JOB_ID, "job1", sticky=False, private=False)

    err = authorize_build_component(
        {"path": "subprocess.Popen", "args": {}},
        ConfigContext(),
        None,
        fl_ctx=fl_ctx,
        event_handlers=[],
    )
    assert err == ""

    fl_ctx.get_prop(FLContextKey.JOB_META)[AppValidationKey.BYOC] = False

    err = authorize_build_component(
        {"path": "socket.socket", "args": {}},
        ConfigContext(),
        None,
        fl_ctx=fl_ctx,
        event_handlers=[],
    )
    assert err == ""
    assert load_calls == [(workspace, "job1")]


def test_worker_component_builder_rejects_component_missing_from_allow_list():
    _set_class_allow_list(_test_component_allow_list())
    builder = WorkerComponentBuilder()
    component_config = {"id": "bad", "path": "socket.socket", "args": {}}
    node = builder.make_component_node(component_config, 1)

    with pytest.raises(ComponentNotAuthorized, match="socket.socket"):
        builder.build_component(component_config, node)


def test_worker_component_builder_blocks_before_instantiation(monkeypatch):
    loaded_paths = _record_load_class_calls(monkeypatch)
    _set_class_allow_list(_test_component_allow_list())
    builder = WorkerComponentBuilder()
    component_config = {"id": "bad", "path": "socket.socket", "args": {}}
    node = builder.make_component_node(component_config, 1)

    with pytest.raises(ComponentNotAuthorized, match="socket.socket.*components.#1"):
        builder.build_component(component_config, node)

    assert loaded_paths == []


def test_worker_component_builder_make_component_node_uses_components_path():
    component_config = {"id": "container", "path": _component_path(ContainerComponent), "args": {}}

    node = WorkerComponentBuilder.make_component_node(component_config, 3)

    assert node.element is component_config
    assert node.key == "#3"
    assert node.path() == "components.#3"


def test_worker_component_builder_rejects_nested_unsafe_config_before_build():
    component_config = {
        "id": "container",
        "path": _component_path(ContainerComponent),
        "args": {"child": {"path": "socket.socket", "args": {}}},
    }
    _set_class_allow_list(_test_component_allow_list())
    builder = WorkerComponentBuilder()
    node = builder.make_component_node(component_config, 1)

    ContainerComponent.instantiated = False
    with pytest.raises(ComponentNotAuthorized, match="socket.socket.*components.#1.args.child"):
        builder.build_component(component_config, node)

    assert ContainerComponent.instantiated is False


def test_multi_process_build_components_rejects_config_type_dict_without_byoc(monkeypatch):
    loaded_paths = _record_load_class_calls(monkeypatch)
    _set_class_allow_list(_test_component_allow_list())
    executor = _TestMultiProcessExecutor.__new__(_TestMultiProcessExecutor)
    executor.components = {}
    executor.handlers = []
    fl_ctx = FLContext()
    fl_ctx.set_prop(FLContextKey.JOB_META, {}, sticky=False, private=True)

    with pytest.raises(ComponentNotAuthorized, match=r"subprocess\.Popen.*components\.#1"):
        executor._build_components(
            [
                {
                    "id": "bad",
                    "config_type": "dict",
                    "path": "subprocess.Popen",
                    "args": {},
                }
            ],
            fl_ctx=fl_ctx,
        )

    assert loaded_paths == []


def test_multi_process_build_components_skips_allow_list_for_byoc_config_type_dict():
    _set_class_allow_list([])
    executor = _TestMultiProcessExecutor.__new__(_TestMultiProcessExecutor)
    executor.components = {}
    executor.handlers = []
    fl_ctx = FLContext()
    fl_ctx.set_prop(FLContextKey.JOB_META, {AppValidationKey.BYOC: True}, sticky=False, private=True)

    NestedComponent.instantiated = False
    executor._build_components(
        [
            {
                "id": "byoc_component",
                "config_type": "dict",
                "path": _component_path(NestedComponent),
                "args": {},
            }
        ],
        fl_ctx=fl_ctx,
    )

    assert isinstance(executor.components["byoc_component"], NestedComponent)
    assert NestedComponent.instantiated is True


def test_sub_worker_initialize_rejects_unsafe_component_payload_before_build(tmp_path):
    sub_worker = SubWorkerExecutor.__new__(SubWorkerExecutor)
    sub_worker.components = {}
    sub_worker.handlers = []
    meta_path = tmp_path / "meta.json"
    meta_path.write_text("{}")
    resources_path = tmp_path / "resources.json"
    resources_path.write_text(json.dumps({"class_allow_list": _test_component_allow_list()}))
    sub_worker.workspace = FakeWorkspace(str(meta_path), str(resources_path))

    fl_ctx = FLContext()
    fl_ctx.set_prop(FLContextKey.JOB_META, {}, sticky=False, private=True)

    data = {
        CommunicationMetaData.FL_CTX: fl_ctx,
        CommunicationMetaData.LOCAL_EXECUTOR: "container",
        CommunicationMetaData.COMPONENTS: [
            {
                "id": "container",
                "path": _component_path(ContainerComponent),
                "args": {"child": {"path": "socket.socket", "args": {}}},
            }
        ],
    }

    ContainerComponent.instantiated = False
    with pytest.raises(ComponentNotAuthorized, match="socket.socket.*components.#1.args.child"):
        sub_worker._initialize(data)

    assert ContainerComponent.instantiated is False
