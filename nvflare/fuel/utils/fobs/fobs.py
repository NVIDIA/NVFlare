# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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
from os.path import dirname, join
from typing import Any, BinaryIO, Dict, Type, Union

import msgpack

from nvflare.fuel.utils.fobs.decomposer import Decomposer

__all__ = [
    "register",
    "register_folder",
    "num_decomposers",
    "serialize",
    "serialize_stream",
    "deserialize",
    "deserialize_stream",
]

FOBS_TYPE = "__fobs_type__"
FOBS_DATA = "__fobs_data__"
MSGPACK_TYPES = (None, bool, int, float, str, bytes, bytearray, memoryview, list, dict)

log = logging.getLogger(__name__)
_decomposers: Dict[str, Decomposer] = {}
_decomposers_registered = False


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

    name = _get_type_name(decomposer.supported_type())
    if name in _decomposers:
        return

    if inspect.isclass(decomposer):
        instance = decomposer()
    else:
        instance = decomposer

    if not isinstance(instance, Decomposer):
        log.error(f"Class {instance.__class__} is not a decomposer")
        return

    _decomposers[name] = instance


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
                if issubclass(cls_obj, Decomposer) and not inspect.isabstract(cls_obj):
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
        return obj

    decomposed = _decomposers[type_name].decompose(obj)
    return {FOBS_TYPE: type_name, FOBS_DATA: decomposed}


def _fobs_unpacker(obj: Any) -> Any:

    if type(obj) is not dict or FOBS_TYPE not in obj:
        return obj

    type_name = obj[FOBS_TYPE]
    if type_name not in _decomposers:
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
    return msgpack.packb(obj, default=_fobs_packer, strict_types=True, **kwargs)


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
