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
from enum import Enum
from os.path import dirname, join
from typing import Any, BinaryIO, Dict, Type, TypeVar, Union

import msgpack

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
MAX_CONTENT_LEN = 128
MSGPACK_TYPES = (None, bool, int, float, str, bytes, bytearray, memoryview, list, dict)
T = TypeVar("T")

log = logging.getLogger(__name__)
_decomposers: Dict[str, Decomposer] = {}
_decomposers_registered = False
_enum_auto_register = True


def _get_type_name(cls: Type) -> str:
    module = cls.__module__
    if module == "builtins":
        return cls.__qualname__
    return module + "." + cls.__qualname__


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
    """Enable or disable auto registering of enum classes

    Args:
        enabled: Auto-registering of enum classes is enabled if True
    """
    global _enum_auto_register

    _enum_auto_register = enabled


def register_folder(folder: str, package: str):
    """Scan the folder and register all decomposers found.

    Args:
        folder: The folder to scan
        package: The package to import the decomposers from
    """
    for module in os.listdir(folder):
        if module != "__init__.py" and module[-3:] == ".py":
            decomposers = package + "." + module[:-3]
            imported = importlib.import_module(decomposers, __package__)
            for _, cls_obj in inspect.getmembers(imported, inspect.isclass):
                spec = inspect.getfullargspec(cls_obj.__init__)
                # classes who are abstract or take extra args in __init__ can't be auto-registered
                if issubclass(cls_obj, Decomposer) and not inspect.isabstract(cls_obj) and len(spec.args) == 1:
                    register(cls_obj)


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


def _fobs_packer(obj: Any) -> dict:

    if type(obj) in MSGPACK_TYPES:
        return obj

    type_name = _get_type_name(obj.__class__)
    if type_name not in _decomposers:
        if _enum_auto_register and isinstance(obj, Enum):
            register_enum_types(type(obj))
        else:
            return obj

    decomposed = _decomposers[type_name].decompose(obj)
    return {FOBS_TYPE: type_name, FOBS_DATA: decomposed}


def _load_class(type_name: str):
    parts = type_name.split(".")
    if len(parts) == 1:
        parts = ["builtins", type_name]

    mod = __import__(parts[0])
    for comp in parts[1:]:
        mod = getattr(mod, comp)

    return mod


def _fobs_unpacker(obj: Any) -> Any:

    if type(obj) is not dict or FOBS_TYPE not in obj:
        return obj

    type_name = obj[FOBS_TYPE]
    if type_name not in _decomposers:
        error = True
        if _enum_auto_register:
            cls = _load_class(type_name)
            if issubclass(cls, Enum):
                register_enum_types(cls)
                error = False
        if error:
            raise TypeError(f"Unknown type {type_name}, caused by mismatching decomposers")

    decomposer = _decomposers[type_name]
    return decomposer.recompose(obj[FOBS_DATA])


def serialize(obj: Any, **kwargs) -> bytes:
    """Serialize object into bytes.

    Args:
        obj: Object to be serialized
        kwargs: Arguments passed to msgpack.packb
    Returns:
        Serialized data
    """
    _register_decomposers()
    try:
        return msgpack.packb(obj, default=_fobs_packer, strict_types=True, **kwargs)
    except ValueError as ex:
        content = str(obj)
        if len(content) > MAX_CONTENT_LEN:
            content = content[:MAX_CONTENT_LEN] + " ..."
        raise ValueError(f"Object {type(obj)} is not serializable: {secure_format_exception(ex)}: {content}")


def serialize_stream(obj: Any, stream: BinaryIO, **kwargs):
    """Serialize object and write the data to a stream.

    Args:
        obj: Object to be serialized
        stream: Stream to write the result to
        kwargs: Arguments passed to msgpack.packb
    """
    data = serialize(obj, **kwargs)
    stream.write(data)


def deserialize(data: bytes, **kwargs) -> Any:
    """Deserialize bytes into an object.

    Args:
        data: Serialized data
        kwargs: Arguments passed to msgpack.unpackb
    Returns:
        Deserialized object
    """
    _register_decomposers()
    return msgpack.unpackb(data, object_hook=_fobs_unpacker, **kwargs)


def deserialize_stream(stream: BinaryIO, **kwargs) -> Any:
    """Deserialize bytes from stream into an object.

    Args:
        stream: Stream to write serialized data to
        kwargs: Arguments passed to msgpack.unpackb
    Returns:
        Deserialized object
    """
    data = stream.read()
    return deserialize(data, **kwargs)


def reset():
    """Reset FOBS to initial state. Used for unit test"""
    global _decomposers, _decomposers_registered
    _decomposers.clear()
    _decomposers_registered = False
