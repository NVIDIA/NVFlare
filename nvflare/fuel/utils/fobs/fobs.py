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
from typing import Any, BinaryIO, Type, TypeVar, Union

import msgpack

from nvflare.fuel.utils.class_loader import get_class_name
from nvflare.fuel.utils.fobs.datum import DatumManager
from nvflare.fuel.utils.fobs.decomposer import (
    DataClassDecomposer,
    Decomposer,
    EnumTypeDecomposer,
    Externalizer,
    Internalizer,
)
from nvflare.fuel.utils.fobs.decomposer_registrar import DecomposerRegistrar
from nvflare.security.logging import secure_format_exception

__all__ = [
    "register",
    "registrar",
    "register_custom_folder",
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
    "get_dot_handler",
]

FOBS_TYPE = "__fobs_type__"
FOBS_DATA = "__fobs_data__"
FOBS_DECOMPOSER = "__fobs_dc__"

MAX_CONTENT_LEN = 128
MSGPACK_TYPES = (type(None), bool, int, float, str, bytes, bytearray, memoryview, list, dict)
T = TypeVar("T")

log = logging.getLogger(__name__)
registrar = DecomposerRegistrar()


def register(decomposer: Union[Decomposer, Type[Decomposer]]) -> None:
    """Register a decomposer. It does nothing if decomposer is already registered for the type

    Args:
        decomposer: The decomposer type or instance
    """

    registrar.register(decomposer)


class Packer:
    def __init__(self, manager: DatumManager):
        self.manager = manager
        self.enum_decomposer_name = get_class_name(EnumTypeDecomposer)
        self.data_decomposer_name = get_class_name(DataClassDecomposer)

    def pack(self, obj: Any) -> dict:

        if type(obj) in MSGPACK_TYPES:
            return obj

        decomposer = registrar.find_for_object(obj)

        decomposed = decomposer.decompose(obj, self.manager)
        if self.manager:
            externalizer = Externalizer(self.manager)
            decomposed = externalizer.externalize(decomposed)

        type_name = get_class_name(obj.__class__)
        return {FOBS_TYPE: type_name, FOBS_DATA: decomposed, FOBS_DECOMPOSER: get_class_name(type(decomposer))}

    def unpack(self, obj: Any) -> Any:

        # FOBS serialized data is always a dict
        if type(obj) is not dict or FOBS_TYPE not in obj:
            return obj

        decomposer_name = obj.get(FOBS_DECOMPOSER)

        # If a decomposer is not registered, reject it due to security reasons
        if not registrar.registered(decomposer_name):
            raise TypeError(f"Decomposer {decomposer_name} is not pre-registered")

        type_name = obj.get(FOBS_TYPE)
        decomposer = registrar.load(decomposer_name, type_name)

        data = obj[FOBS_DATA]
        if self.manager:
            internalizer = Internalizer(self.manager)
            data = internalizer.internalize(data)

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
    """Enable or disable the auto-registration of enum types.

    Args:
        enabled: Auto-registration of enum classes is enabled if True.
    """
    registrar.set_enum_generic_enabled(enabled)


def auto_register_data_classes(enabled=True) -> None:
    """Enable or disable the auto-registration of data classes.

    Args:
        enabled: Auto-registration of data classes is enabled if True.
    """
    registrar.set_data_generic_enabled(enabled)


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
            except (ModuleNotFoundError, RuntimeError, ValueError) as e:
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
                except (ModuleNotFoundError, RuntimeError, ValueError):
                    pass


def num_decomposers() -> int:
    """Returns the number of decomposers registered.

    Returns:
        The number of decomposers
    """
    return registrar.count()


def serialize(obj: Any, manager: DatumManager = None, **kwargs) -> bytes:
    """Serialize object into bytes.

    Args:
        obj: Object to be serialized
        manager: Datum manager used to externalize datum
        kwargs: Arguments passed to msgpack.packb
    Returns:
        Serialized data
    """
    packer = Packer(manager)
    try:
        result = msgpack.packb(obj, default=packer.pack, strict_types=True, **kwargs)
    except ValueError as ex:
        content = str(obj)
        if len(content) > MAX_CONTENT_LEN:
            content = content[:MAX_CONTENT_LEN] + " ..."
        error = f"Object {type(obj)} is not serializable: {secure_format_exception(ex)}: {content}"
        manager.set_error(error)
        result = None
    except Exception as ex:
        error = f"Exception serializing {type(obj)}: {secure_format_exception(ex)}"
        manager.set_error(error)
        result = None

    # must ensure that manager.post_process is always called since some decomposers may need to clean up properly
    manager.post_process()

    error = manager.get_error()
    if error:
        raise RuntimeError(manager.error)

    return result


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
    packer = Packer(manager)
    result = msgpack.unpackb(data, strict_map_key=False, object_hook=packer.unpack, **kwargs)
    manager.post_process()
    return result


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


def get_dot_handler(dot: int) -> Decomposer:
    return registrar.find_for_dot(dot)


def reset():
    """Reset FOBS to initial state. Used for unit test"""
    registrar.reset()
