# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import importlib
import inspect
import logging
import os
import sys
from enum import Enum
from os.path import dirname, join
from typing import Any, BinaryIO, Dict, Type, TypeVar, Union

import msgpack

from nvflare.fuel.utils.fobs.datum import DatumManager
from nvflare.fuel.utils.fobs.decomposer import DataClassDecomposer, Decomposer, EnumTypeDecomposer

__all__ = [
    "register",
    "register_data_classes",
    "register_enum_types",
    "auto_register_enum_types",
    "register_folder",
    "num_decomposers",
    "serialize",
    "serialize_stream",
    "deserialize",
    "deserialize_stream",
    "reset",
]

from nvflare.security.logging import secure_format_exception

FOBS_TYPE = "__fobs_type__"
FOBS_DATA = "__fobs_data__"
FOBS_DECOMPOSER = "__fobs_dc__"

MAX_CONTENT_LEN = 128
MSGPACK_TYPES = (type(None), bool, int, float, str, bytes, bytearray, memoryview, list, dict)
T = TypeVar("T")

log = logging.getLogger(__name__)
_decomposers: Dict[str, Decomposer] = {}
_decomposers_registered = False
# If this is enabled, FOBS will try to register generic decomposers automatically
_enum_auto_registration = True
_data_auto_registration = True


def _get_type_name(cls: Type) -> str:
    module = cls.__module__
    if module == "builtins":
        return cls.__qualname__
    return module + "." + cls.__qualname__


def _load_class(type_name: str):
    try:
        parts = type_name.split(".")
        if len(parts) == 1:
            parts = ["builtins", type_name]

        mod = __import__(parts[0])
        for comp in parts[1:]:
            mod = getattr(mod, comp)

        return mod
    except Exception as ex:
        raise TypeError(f"Can't load class {type_name}: {ex}")


def register(decomposer: Union[Decomposer, Type[Decomposer]]) -> None:
    """Register a decomposer. It does nothing if decomposer is already registered for the type

    Args:
        decomposer: The decomposer type or instance
    """

    global _decomposers

    if inspect.isclass(decomposer):
        instance = decomposer()
    else:
        instance = decomposer

    name = _get_type_name(instance.supported_type())
    if name in _decomposers:
        return

    if not isinstance(instance, Decomposer):
        log.error(f"Class {instance.__class__} is not a decomposer")
        return

    _decomposers[name] = instance


class Packer:
    def __init__(self, manager: DatumManager):
        self.manager = manager
        self.enum_decomposer_name = _get_type_name(EnumTypeDecomposer)
        self.data_decomposer_name = _get_type_name(DataClassDecomposer)

    def pack(self, obj: Any) -> dict:

        if type(obj) in MSGPACK_TYPES:
            return obj

        type_name = _get_type_name(obj.__class__)
        if type_name not in _decomposers:
            registered = False
            if isinstance(obj, Enum):
                if _enum_auto_registration:
                    register_enum_types(type(obj))
                    registered = True
            else:
                if callable(obj) or (not hasattr(obj, "__dict__")):
                    raise TypeError(f"{type(obj)} can't be serialized by FOBS without a decomposer")
                if _data_auto_registration:
                    register_data_classes(type(obj))
                    registered = True

            if not registered:
                return obj

        decomposer = _decomposers[type_name]

        decomposed = decomposer.decompose(obj, self.manager)
        if self.manager:
            decomposed = self.manager.externalize(decomposed)

        return {FOBS_TYPE: type_name, FOBS_DATA: decomposed, FOBS_DECOMPOSER: _get_type_name(type(decomposer))}

    def unpack(self, obj: Any) -> Any:

        if type(obj) is not dict or FOBS_TYPE not in obj:
            return obj

        type_name = obj[FOBS_TYPE]
        if type_name not in _decomposers:
            registered = False
            decomposer_name = obj.get(FOBS_DECOMPOSER)
            cls = _load_class(type_name)
            if not decomposer_name:
                # Maintaining backward compatibility with auto enum registration
                if _enum_auto_registration:
                    if issubclass(cls, Enum):
                        register_enum_types(cls)
                        registered = True
            else:
                decomposer_class = _load_class(decomposer_name)
                if decomposer_name == self.enum_decomposer_name or decomposer_name == self.data_decomposer_name:
                    # Generic decomposer's __init__ takes the target class as argument
                    decomposer = decomposer_class(cls)
                else:
                    decomposer = decomposer_class()

                register(decomposer)
                registered = True

            if not registered:
                raise TypeError(f"Type {type_name} has no decomposer registered")

        data = obj[FOBS_DATA]
        if self.manager:
            data = self.manager.internalize(data)

        decomposer = _decomposers[type_name]
        return decomposer.recompose(data, self.manager)


def register_data_classes(*data_classes: Type[T]) -> None:
    """Register generic decomposers for data classes

    Args:
        data_classes: The classes to be registered
    """

    for data_class in data_classes:
        decomposer = DataClassDecomposer(data_class)
        register(decomposer)


def register_enum_types(*enum_types: Type[Enum]) -> None:
    """Register generic decomposers for enum classes

    Args:
        enum_types: The enum classes to be registered
    """

    for enum_type in enum_types:
        if not issubclass(enum_type, Enum):
            raise TypeError(f"Can't register class {enum_type}, which is not a subclass of Enum")
        decomposer = EnumTypeDecomposer(enum_type)
        register(decomposer)


def auto_register_enum_types(enabled=True) -> None:
    """Enable or disable auto registering of enum types

    Args:
        enabled: Auto-registering of enum classes is enabled if True
    """
    global _enum_auto_registration

    _enum_auto_registration = enabled


def auto_register_data_classes(enabled=True) -> None:
    """Enable or disable auto registering of data classes

    Args:
        enabled: Auto-registering of data classes is enabled if True
    """
    global _data_auto_registration

    _enum_data_registration = enabled


def register_folder(folder: str, package: str):
    """Scan the folder and register all decomposers found.

    Args:
        folder: The folder to scan
        package: The package to import the decomposers from
    """
    for module in os.listdir(folder):
        if module != "__init__.py" and module[-3:] == ".py":
            decomposers = package + "." + module[:-3]
            try:
                imported = importlib.import_module(decomposers, __package__)
                for _, cls_obj in inspect.getmembers(imported, inspect.isclass):
                    spec = inspect.getfullargspec(cls_obj.__init__)
                    # classes who are abstract or take extra args in __init__ can't be auto-registered
                    if issubclass(cls_obj, Decomposer) and not inspect.isabstract(cls_obj) and len(spec.args) == 1:
                        register(cls_obj)
            except (ModuleNotFoundError, RuntimeError) as e:
                log.debug(
                    f"Try to import module {decomposers}, but failed: {secure_format_exception(e)}. "
                    f"Can't use name in config to refer to classes in module: {decomposers}."
                )
                pass


def register_custom_folder(folder: str):
    if os.path.isdir(folder) and folder not in sys.path:
        sys.path.append(folder)

    for root, dirs, files in os.walk(folder):
        for filename in files:
            if filename.endswith(".py"):
                module = filename[:-3]
                sub_folder = os.path.relpath(root, folder).strip(".").replace(os.sep, ".")
                if sub_folder:
                    module = sub_folder + "." + module

                try:
                    imported = importlib.import_module(module)
                    for _, cls_obj in inspect.getmembers(imported, inspect.isclass):
                        if issubclass(cls_obj, Decomposer) and not inspect.isabstract(cls_obj):
                            spec = inspect.getfullargspec(cls_obj.__init__)
                            if len(spec.args) == 1:
                                register(cls_obj)
                            else:
                                # Can't handle argument in constructor
                                log.warning(
                                    f"Invalid Decomposer from {module}: can't have argument in Decomposer's constructor"
                                )
                except (ModuleNotFoundError, RuntimeError):
                    pass


def _register_decomposers():
    global _decomposers_registered

    if _decomposers_registered:
        return

    register_folder(join(dirname(__file__), "decomposers"), ".decomposers")
    _decomposers_registered = True


def num_decomposers() -> int:
    """Returns the number of decomposers registered.

    Returns:
        The number of decomposers
    """
    return len(_decomposers)


def serialize(obj: Any, manager: DatumManager = None, **kwargs) -> bytes:
    """Serialize object into bytes.

    Args:
        obj: Object to be serialized
        manager: Datum manager used to externalize datum
        kwargs: Arguments passed to msgpack.packb
    Returns:
        Serialized data
    """
    _register_decomposers()
    packer = Packer(manager)
    try:
        return msgpack.packb(obj, default=packer.pack, strict_types=True, **kwargs)
    except ValueError as ex:
        content = str(obj)
        if len(content) > MAX_CONTENT_LEN:
            content = content[:MAX_CONTENT_LEN] + " ..."
        raise ValueError(f"Object {type(obj)} is not serializable: {secure_format_exception(ex)}: {content}")


def serialize_stream(obj: Any, stream: BinaryIO, manager: DatumManager = None, **kwargs):
    """Serialize object and write the data to a stream.

    Args:
        obj: Object to be serialized
        stream: Stream to write the result to
        manager: Datum manager to externalize datum
        kwargs: Arguments passed to msgpack.packb
    """
    data = serialize(obj, manager, **kwargs)
    stream.write(data)


def deserialize(data: bytes, manager: DatumManager = None, **kwargs) -> Any:
    """Deserialize bytes into an object.

    Args:
        data: Serialized data
        manager: Datum manager to internalize datum
        kwargs: Arguments passed to msgpack.unpackb
    Returns:
        Deserialized object
    """
    _register_decomposers()
    packer = Packer(manager)
    return msgpack.unpackb(data, strict_map_key=False, object_hook=packer.unpack, **kwargs)


def deserialize_stream(stream: BinaryIO, manager: DatumManager = None, **kwargs) -> Any:
    """Deserialize bytes from stream into an object.

    Args:
        stream: Stream to write serialized data to
        manager: Datum manager to internalize datum
        kwargs: Arguments passed to msgpack.unpackb
    Returns:
        Deserialized object
    """
    data = stream.read()
    return deserialize(data, manager, **kwargs)


def reset():
    """Reset FOBS to initial state. Used for unit test"""
    global _decomposers, _decomposers_registered
    _decomposers.clear()
    _decomposers_registered = False
