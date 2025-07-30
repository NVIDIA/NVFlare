# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import inspect
from typing import Any, List


class ConfigError(Exception):
    pass


class ConfigKey:
    NAME = "name"
    TYPE = "type"
    ARGS = "args"
    TRAINER = "trainer"
    EXECUTORS = "executors"
    COMPONENTS = "components"
    IN_FILTERS = "in_filters"
    OUT_FILTERS = "out_filters"
    HANDLERS = "handlers"


class TrainConfig:

    def __init__(self, objects: dict, in_filters, out_filters, event_handlers, executors: dict):
        self.objects = objects
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.event_handlers = event_handlers
        self.executors = executors

    def find_executor(self, task_name: str):
        if not self.executors:
            return self.objects.get(ConfigKey.TRAINER)

        e = self.executors.get(task_name)
        if e:
            return e
        else:
            return self.executors.get("*")


class ComponentResolver:
    """A ComponentResolver resolves component spec into a device-native object."""

    def __init__(self, comp_type, name, args, obj_class=None):
        self.comp_type = comp_type
        self.comp_name = name

        if not args:
            args = {}
        self.comp_args = args
        self.obj_class = obj_class

    def resolve(self) -> Any:
        """Resolve the component spec and create device-native object.

        Returns: a device-native object or None if failed.

        """
        return self.obj_class(**self.comp_args)


def _determine_value(item: Any, resolvers: dict) -> Any:
    """Determine value of the specified item: recursively replace component refs with the ComponentResolver objects
    of the referenced components.

    Args:
        item: the item whose value is to be determined
        resolvers: table of resolvers

    Returns:

    """
    if isinstance(item, list):
        for i, v in enumerate(item):
            item[i] = _determine_value(v, resolvers)
        return item
    elif isinstance(item, dict):
        for k, v in item.items():
            item[k] = _determine_value(v, resolvers)
        return item
    elif not isinstance(item, str):
        return item
    if not item.startswith("@"):
        return item
    else:
        referenced_name = item[1:]
        c = resolvers.get(referenced_name)
        if not c:
            raise ConfigError(f"referenced component '{referenced_name}' does not exist")
        return c


def _find_obj(item, obj_table):
    """Try to find the native object(s) for the item: recursively process all ComponentResolvers and replace them
     with their native objects following the structure of the item (list or dict).

    Args:
        item: the item to be processed
        obj_table: the object table that contains objects already created

    Returns: the item itself (with referenced components replaced with objects);
        or in case that the item is a ComponentResolver, the native object created by it

    """
    if isinstance(item, ComponentResolver):
        # has this component been resolved?
        obj = obj_table.get(item.comp_name)
        if obj is None:
            # not resolved yet
            return item
        else:
            # already resolved
            return obj
    elif isinstance(item, list):
        for i, v in enumerate(item):
            item[i] = _find_obj(v, obj_table)
        return item
    elif isinstance(item, dict):
        for k, v in item.items():
            item[k] = _find_obj(v, obj_table)
        return item
    else:
        return item


def _try_to_resolve(resolver: ComponentResolver, obj_table: dict) -> Any:
    """Try to create device-native object. If created, place the obj in the obj_table.

    Args:
        resolver: the ComponentResolver that will try to resolve its component
        obj_table: object table that keeps objects of resolved components

    Returns: the resolved object, or None if the resolver is not ready

    For the resolver to be ready, all of its args must be resolved already, meaning that if an arg
    references another component, the referenced component must be resolved.

    """
    if isinstance(resolver.comp_args, dict):
        for k, v in resolver.comp_args.items():
            v = _find_obj(v, obj_table)
            resolver.comp_args[k] = v
            if isinstance(v, ComponentResolver):
                # not ready to resolve this component since this referenced component has not been resolved
                return None

    obj = resolver.resolve()
    if obj is None:
        raise ConfigError(f"failed to resolve component {resolver.comp_name}")
    obj_table[resolver.comp_name] = obj
    return obj


def _process_components(component_config: dict, resolver_registry: dict):
    # Step 1: create a ComponentResolver for each component spec in the config
    resolvers = {}  # name => ComponentResolver
    for c in component_config:
        name = c.get(ConfigKey.NAME)
        comp_type = c.get(ConfigKey.TYPE)
        clazz = resolver_registry.get(comp_type)
        if not clazz:
            raise ConfigError(f"no ComponentResolver registered for component type {comp_type}")

        comp_args = c.get(ConfigKey.ARGS)
        if issubclass(clazz, ComponentResolver):
            resolver = clazz(comp_type, name, comp_args)
        else:
            if not inspect.isclass(clazz):
                raise ConfigError(f"resolver for component {comp_type} is not a valid class")

            # the clazz is the native object's class
            resolver = ComponentResolver(comp_type, name, comp_args, clazz)

        if not resolver:
            raise ConfigError(f"cannot make resolver for component {name} of type {comp_type}")

        if name in resolvers:
            raise ConfigError(f"duplicate component definition for '{name}'")
        resolvers[name] = resolver

    # find ComponentResolver for referenced components
    for name, resolver in resolvers.items():
        assert isinstance(resolver, ComponentResolver)
        if not isinstance(resolver.comp_args, dict):
            # the args could be None
            continue

        for k, v in resolver.comp_args.items():
            resolver.comp_args[k] = _determine_value(v, resolvers)

    # repeatedly trying to resolve components until all are done.
    # the "resolve" method of ComponentResolver creates device objects based on the args.
    obj_table = {}
    while resolvers:
        resolved = []
        for name, resolver in resolvers.items():
            obj = _try_to_resolve(resolver, obj_table)
            if obj is not None:
                resolved.append(name)

        if not resolved:
            # nothing is resolved - there are cyclic refs
            raise ConfigError(f"cannot resolve components {resolvers.keys()}")

        for n in resolved:
            resolvers.pop(n)

    return obj_table


def _resolve_ref(ref, obj_table: dict):
    if not ref.startswith("@"):
        raise ConfigError(f"invalid ref {ref}")

    referenced_name = ref[1:]
    return obj_table.get(referenced_name)


def _process_refs(refs: List[str], obj_table: dict):
    for i, r in enumerate(refs):
        obj = _resolve_ref(r, obj_table)
        if not obj:
            raise ConfigError(f"cannot find object for reference {r}")
        refs[i] = obj


def process_train_config(config: dict, resolver_registry: dict) -> TrainConfig:
    components = config.get(ConfigKey.COMPONENTS)
    if not components:
        raise ConfigError(f"missing {ConfigKey.COMPONENTS} in config")

    obj_table = _process_components(components, resolver_registry)

    in_filters = config.get(ConfigKey.IN_FILTERS)
    if in_filters:
        if not isinstance(in_filters, list):
            raise ConfigError(f"{ConfigKey.IN_FILTERS} should be list of str but got {type(in_filters)}")
        _process_refs(in_filters, obj_table)

    out_filters = config.get(ConfigKey.OUT_FILTERS)
    if out_filters:
        if not isinstance(out_filters, list):
            raise ConfigError(f"{ConfigKey.OUT_FILTERS} should be list of str but got {type(out_filters)}")
        _process_refs(out_filters, obj_table)

    handlers = config.get(ConfigKey.HANDLERS)
    if handlers:
        if not isinstance(handlers, list):
            raise ConfigError(f"{ConfigKey.HANDLERS} should be list of str but got {type(handlers)}")
        _process_refs(handlers, obj_table)

    # process executors
    executor_config = config.get(ConfigKey.EXECUTORS)
    executors = {}
    if executor_config:
        if not isinstance(executor_config, dict):
            raise ConfigError(f"{ConfigKey.EXECUTORS} should be dict but got {type(executor_config)}")

        for k, v in executor_config.items():
            executors[k] = _resolve_ref(v, obj_table)

    return TrainConfig(obj_table, in_filters, out_filters, handlers, executors)
