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
from abc import abstractmethod
from typing import Any, List


class ConfigError(Exception):
    pass


class ComponentCreator:
    """A ComponentCreator creates device-native object from component args."""

    def __init__(self, comp_type, name, args):
        self.comp_type = comp_type
        self.name = name

        if not args:
            args = {}
        self.args = args

    @abstractmethod
    def create(self) -> Any:
        """Create device-native object.

        Returns: a device-native object or None if failed.

        """
        pass


def _determine_value(item: Any, creators: dict) -> Any:
    """Determine value of an item: recursively replace component refs with the ComponentCreator objects of the
    referenced components.

    Args:
        item:
        creators:

    Returns:

    """
    if isinstance(item, list):
        for i, v in enumerate(item):
            item[i] = _determine_value(v, creators)
        return item
    elif isinstance(item, dict):
        for k, v in item.items():
            item[k] = _determine_value(v, creators)
        return item
    elif not isinstance(item, str):
        return item
    if not item.startswith("@"):
        return item
    else:
        referenced_name = item[1:]
        c = creators.get(referenced_name)
        if not c:
            raise ConfigError(f"referenced component '{referenced_name}' does not exist")
        return c


def _create_obj(item, obj_table):
    if isinstance(item, ComponentCreator):
        # has this component been created?
        obj = obj_table.get(item.name)
        if obj is None:
            return item
        else:
            return obj
    elif isinstance(item, list):
        for i, v in enumerate(item):
            item[i] = _create_obj(v, obj_table)
        return item
    elif isinstance(item, dict):
        for k, v in item.items():
            item[k] = _create_obj(v, obj_table)
        return item
    else:
        return item


def _try_to_create(creator: ComponentCreator, obj_table: dict) -> Any:
    # are we ready to create device-native object?
    if isinstance(creator.args, dict):
        for k, v in creator.args.items():
            v = _create_obj(v, obj_table)
            creator.args[k] = v
            if isinstance(v, ComponentCreator):
                # not ready to create this component since this referenced component has not been created
                return None

    obj = creator.create()
    if obj is None:
        raise ConfigError(f"failed to create object for component spec {creator.name}")
    return obj


def _process_components(component_config: dict, creator_registry: dict):
    # create a ComponentCreator for each component spec in the config
    creators = {}  # name => ComponentCreator
    for c in component_config:
        name = c.get("name")
        comp_type = c.get("type")
        creator_class = creator_registry.get(comp_type)
        if not creator_class:
            raise ConfigError(f"no ComponentCreator registered for component type {comp_type}")

        if not issubclass(creator_class, ComponentCreator):
            raise ConfigError(f"bad ComponentCreator for component type {comp_type}")

        creator = creator_class(comp_type, name, c.get("args"))
        if not creator:
            raise ConfigError(f"cannot make creator for component {name} of type {comp_type}")

        if name in creators:
            raise ConfigError(f"duplicate component spec for {name}")
        creators[name] = creator

    # find ComponentCreator for referenced components
    for name, creator in creators.items():
        assert isinstance(creator, ComponentCreator)
        if not isinstance(creator.args, dict):
            # the args could be None
            continue

        for k, v in creator.args.items():
            creator.args[k] = _determine_value(v, creators)

    # repeatedly trying to turn component specs to device-native objects until all are done.
    # the "create" method of ComponentCreator creates device objects based on the args.
    obj_table = {}
    while creators:
        created = []
        for name, creator in creators.items():
            obj = _try_to_create(creator, obj_table)
            if obj is not None:
                obj_table[name] = obj
                created.append(name)

        if not created:
            # nothing is created - there are cyclic refs
            raise ConfigError(f"cannot create objects for components {creators.keys()}")

        for n in created:
            creators.pop(n)

    return obj_table


def _process_refs(refs: List[str], obj_table: dict):
    for i, r in enumerate(refs):
        if not r.startswith("@"):
            raise ConfigError(f"invalid ref {r}")

        referenced_name = r[1:]
        obj = obj_table.get(referenced_name)
        if not obj:
            raise ConfigError(f"cannot find object for reference {referenced_name}")
        refs[i] = obj


def process_train_config(config: dict, creator_registry: dict):
    components = config.get("components")
    if not components:
        raise ConfigError("no components in config")

    obj_table = _process_components(config.get("components"), creator_registry)

    filters = config.get("filters")
    if filters:
        if not isinstance(filters, list):
            raise ConfigError(f"filters should be list of str but got {type(filters)}")
        _process_refs(filters, obj_table)

    handlers = config.get("handlers")
    if handlers:
        if not isinstance(handlers, list):
            raise ConfigError(f"handlers should be list of str but got {type(handlers)}")
        _process_refs(handlers, obj_table)

    return obj_table, filters, handlers
