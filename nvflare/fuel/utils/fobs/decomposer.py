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

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Type, TypeVar

# Generic type supported by the decomposer.
from nvflare.fuel.utils.fobs.datum import Datum, DatumManager, DatumRef, DatumType

T = TypeVar("T")


class Decomposer(ABC):
    """Abstract base class for decomposers.

    Every class to be serialized by FOBS must register a decomposer which is
    a concrete subclass of this class.
    """

    @abstractmethod
    def supported_type(self) -> Type[T]:
        """Returns the type/class supported by this decomposer.

        Returns:
            The class (not instance) of supported type
        """
        pass

    @abstractmethod
    def decompose(self, target: T, manager: DatumManager = None) -> Any:
        """Decompose the target into types supported by msgpack or classes with decomposers registered.

        Msgpack supports primitives, bytes, memoryview, lists, dicts.

        Args:
            target: The instance to be serialized
            manager: Datum manager to store externalized datum

        Returns:
            The decomposed serializable objects
        """
        pass

    @abstractmethod
    def recompose(self, data: Any, manager: DatumManager = None) -> T:
        """Reconstruct the object from decomposed components.

        Args:
            data: The decomposed component
            manager: Datum manager to internalize datum

        Returns:
            The reconstructed object
        """
        pass


def restore_position(manager: DatumManager, datum: Datum, position):
    """
    This function is used for restoring object state at the specified position.

    Args:
        manager: the datum manager
        datum: the datum that contains the value of the original object at the position.
        position: the position to be restored

    Returns: None

    """
    target, key = position
    original_obj = manager.get_original(target)  # also need to restore values in the original object if any
    if datum.datum_type in (DatumType.BLOB, DatumType.TEXT):
        target[key] = datum.value
        if original_obj:
            original_obj[key] = datum.value
    else:
        # file datum - app provided
        target[key] = datum
        if original_obj:
            original_obj[key] = datum


class Externalizer:
    """
    This class is used to help creating 'decompose' method of decomposers of arbitrary classes.

    """

    def __init__(self, manager: DatumManager):
        self.manager = manager

    def _set_position(self, ext_result: Any, target, key):
        if isinstance(ext_result, DatumRef):
            datum = self.manager.get_datum(ext_result.datum_id)
            if datum:
                datum.set_restore_func(restore_position, (target, key))

    def externalize(self, target: Any):
        """Recursively go through object tree (dict or list) and externalize leaf nodes."""
        if not self.manager:
            return target

        if isinstance(target, dict):
            for k, v in target.items():
                d = self.externalize(v)
                target[k] = d
                self._set_position(d, target, k)  # remember the position so it can be restored
        elif isinstance(target, list):  # note: tuple is not supported since it is immutable.
            for i, v in enumerate(target):
                d = self.externalize(v)
                target[i] = d
                self._set_position(d, target, i)
        else:
            # leaf node
            target = self.manager.externalize(target)

        return target


class Internalizer:
    """
    This class is used to help creating 'recompose' method of decomposers of arbitrary classes.

    """

    def __init__(self, manager: DatumManager):
        self.manager = manager

    def internalize(self, target) -> Any:
        """Recursively go through object tree (dict or list) and internalize leaf nodes."""
        if not self.manager:
            return target

        if isinstance(target, dict):
            for k, v in target.items():
                target[k] = self.internalize(v)
        elif isinstance(target, list):
            for i, v in enumerate(target):
                target[i] = self.internalize(v)
        else:
            target = self.manager.internalize(target)

        return target


class DictDecomposer(Decomposer):
    """Generic decomposer for subclasses of dict like Shareable"""

    def __init__(self, dict_type: Type[dict]):
        self.dict_type = dict_type

    def supported_type(self):
        return self.dict_type

    def decompose(self, target: dict, manager: DatumManager = None) -> Any:
        # need to create a new object; otherwise msgpack will try to decompose this object endlessly.
        tc = target.copy()
        manager.register_copy(tc, target)
        externalizer = Externalizer(manager)
        return externalizer.externalize(tc)

    def recompose(self, data: dict, manager: DatumManager = None) -> dict:
        internalizer = Internalizer(manager)
        data = internalizer.internalize(data)
        obj = self.dict_type()
        for k, v in data.items():
            obj[k] = v
        return obj


class DataClassDecomposer(Decomposer):
    """Generic decomposers for data classes, which must meet following requirements:

    1. All class members must be serializable. The type of member must be one of the
       types supported by MessagePack or a decomposer is registered for the type.
    2. The __new__ method only takes one argument which is the class type.
    3. The __init__ method has no side effects. It can only change the states of the
       object. The side effects include creating files, initializing loggers, modifying
       global variables.

    """

    def __init__(self, data_type: Type[T]):
        self.data_type = data_type

    def supported_type(self) -> Type[T]:
        return self.data_type

    def decompose(self, target: T, manager: DatumManager = None) -> Any:
        return vars(target)

    def recompose(self, data: dict, manager: DatumManager = None) -> T:
        instance = self.data_type.__new__(self.data_type)
        instance.__dict__.update(data)
        return instance


class EnumTypeDecomposer(Decomposer):
    """Generic decomposers for enum types."""

    def __init__(self, data_type: Type[Enum]):
        if not issubclass(data_type, Enum):
            raise TypeError(f"{data_type} is not an enum")

        self.data_type = data_type

    def supported_type(self) -> Type[Enum]:
        return self.data_type

    def decompose(self, target: Enum, manager: DatumManager = None) -> Any:
        return target.name

    def recompose(self, data: Any, manager: DatumManager = None) -> Enum:
        return self.data_type[data]
