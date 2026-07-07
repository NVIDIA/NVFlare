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

import pytest

from nvflare.edge.device.config import (
    ComponentResolver,
    ConfigError,
    ConfigKey,
    TrainConfig,
    _determine_value,
    _find_obj,
    _process_components,
    _process_refs,
    _resolve_ref,
    _try_to_resolve,
    process_train_config,
)


class _Component:
    def __init__(self, value=None, dependency=None, nested=None):
        self.value = value
        self.dependency = dependency
        self.nested = nested


class _Resolver(ComponentResolver):
    def resolve(self):
        return _Component(**self.comp_args)


class _FailResolver(ComponentResolver):
    def resolve(self):
        return None


def test_train_config_routes_exact_wildcard_and_trainer():
    trainer = object()
    exact = object()
    wildcard = object()
    config = TrainConfig({ConfigKey.TRAINER: trainer}, [], [], [], {})
    assert config.find_executor("train") is trainer

    config.executors = {"train": exact, "*": wildcard}
    assert config.find_executor("train") is exact
    assert config.find_executor("validate") is wildcard


def test_component_resolver_builds_native_object():
    resolver = ComponentResolver("component", "one", {"value": 3}, _Component)
    assert resolver.resolve().value == 3
    assert ComponentResolver("component", "empty", None, _Component).comp_args == {}


def test_determine_value_resolves_nested_references():
    resolver = ComponentResolver("component", "one", {}, _Component)
    value = {"list": ["literal", "@one", 3], "dict": {"ref": "@one"}}

    resolved = _determine_value(value, {"one": resolver})

    assert resolved["list"] == ["literal", resolver, 3]
    assert resolved["dict"]["ref"] is resolver
    with pytest.raises(ConfigError, match="does not exist"):
        _determine_value("@missing", {})


def test_find_obj_recursively_replaces_ready_resolvers():
    resolver = ComponentResolver("component", "one", {}, _Component)
    native = _Component(value=1)
    value = [resolver, {"nested": resolver}, "literal"]

    assert _find_obj(value, {})[0] is resolver
    replaced = _find_obj(value, {"one": native})
    assert replaced == [native, {"nested": native}, "literal"]


def test_try_to_resolve_waits_for_dependencies_and_rejects_failure():
    dependency = ComponentResolver("component", "dependency", {}, _Component)
    resolver = ComponentResolver("component", "consumer", {"dependency": dependency}, _Component)
    objects = {}
    assert _try_to_resolve(resolver, objects) is None

    native_dependency = _Component(value="ready")
    objects["dependency"] = native_dependency
    consumer = _try_to_resolve(resolver, objects)
    assert consumer.dependency is native_dependency
    assert objects["consumer"] is consumer

    with pytest.raises(ConfigError, match="failed to resolve"):
        _try_to_resolve(_FailResolver("component", "failed", {}), {})


def test_process_components_resolves_dependency_graph():
    config = [
        {ConfigKey.NAME: "base", ConfigKey.TYPE: "native", ConfigKey.ARGS: {"value": 1}},
        {
            ConfigKey.NAME: "consumer",
            ConfigKey.TYPE: "custom",
            ConfigKey.ARGS: {"dependency": "@base", "nested": ["@base", {"item": "@base"}]},
        },
    ]

    objects = _process_components(config, {"native": _Component, "custom": _Resolver})

    assert objects["consumer"].dependency is objects["base"]
    assert objects["consumer"].nested == [objects["base"], {"item": objects["base"]}]


@pytest.mark.parametrize(
    "config, registry, error",
    [
        ([{"name": "one", "type": "missing"}], {}, "no ComponentResolver"),
        (
            [{"name": "one", "type": "native"}, {"name": "one", "type": "native"}],
            {"native": _Component},
            "duplicate component",
        ),
        (
            [
                {"name": "one", "type": "native", "args": {"dependency": "@two"}},
                {"name": "two", "type": "native", "args": {"dependency": "@one"}},
            ],
            {"native": _Component},
            "cannot resolve components",
        ),
    ],
)
def test_process_components_rejects_invalid_definitions(config, registry, error):
    with pytest.raises(ConfigError, match=error):
        _process_components(config, registry)


def test_resolve_and_process_refs_validate_references():
    native = _Component()
    assert _resolve_ref("@one", {"one": native}) is native
    assert _resolve_ref("@missing", {}) is None
    with pytest.raises(ConfigError, match="invalid ref"):
        _resolve_ref("one", {})

    refs = ["@one"]
    _process_refs(refs, {"one": native})
    assert refs == [native]
    with pytest.raises(ConfigError, match="cannot find object"):
        _process_refs(["@missing"], {})


def test_process_train_config_resolves_all_sections():
    config = {
        ConfigKey.COMPONENTS: [
            {ConfigKey.NAME: ConfigKey.TRAINER, ConfigKey.TYPE: "native", ConfigKey.ARGS: {"value": "trainer"}},
            {ConfigKey.NAME: "filter", ConfigKey.TYPE: "native", ConfigKey.ARGS: {"value": "filter"}},
            {ConfigKey.NAME: "handler", ConfigKey.TYPE: "native", ConfigKey.ARGS: {"value": "handler"}},
            {ConfigKey.NAME: "executor", ConfigKey.TYPE: "native", ConfigKey.ARGS: {"value": "executor"}},
        ],
        ConfigKey.IN_FILTERS: ["@filter"],
        ConfigKey.OUT_FILTERS: ["@filter"],
        ConfigKey.HANDLERS: ["@handler"],
        ConfigKey.EXECUTORS: {"train": "@executor"},
    }

    result = process_train_config(config, {"native": _Component})

    assert result.in_filters == [result.objects["filter"]]
    assert result.out_filters == [result.objects["filter"]]
    assert result.event_handlers == [result.objects["handler"]]
    assert result.find_executor("train") is result.objects["executor"]


@pytest.mark.parametrize(
    "config, error",
    [
        ({}, "missing components"),
        ({ConfigKey.COMPONENTS: [{"name": "one", "type": "native"}], ConfigKey.IN_FILTERS: "@one"}, "in_filters"),
        ({ConfigKey.COMPONENTS: [{"name": "one", "type": "native"}], ConfigKey.OUT_FILTERS: "@one"}, "out_filters"),
        ({ConfigKey.COMPONENTS: [{"name": "one", "type": "native"}], ConfigKey.HANDLERS: "@one"}, "handlers"),
        ({ConfigKey.COMPONENTS: [{"name": "one", "type": "native"}], ConfigKey.EXECUTORS: ["@one"]}, "executors"),
    ],
)
def test_process_train_config_validates_sections(config, error):
    with pytest.raises(ConfigError, match=error):
        process_train_config(config, {"native": _Component})
