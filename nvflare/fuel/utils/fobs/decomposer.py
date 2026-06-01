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

import inspect
from abc import ABC, abstractmethod
from enum import Enum
from functools import lru_cache
from typing import Any, List, Optional, Type, TypeVar

# Generic type supported by the decomposer.
from nvflare.fuel.utils.fobs.datum import Datum, DatumManager

T = TypeVar("T")

DICT_CONTENT = "dict"
DATA_CONTENT = "data"
LIST_CONTENT = "list"


# Bounded so the cache cannot grow without limit if dynamically-generated classes are serialized;
# FOBS-handled types are effectively a small, fixed set so this is never a practical constraint.
@lru_cache(maxsize=1024)
def _requires_init_args(cls: type) -> bool:
    """Return True if ``cls`` cannot be instantiated without arguments.

    Builtin containers like ``dict``/``list`` expose no introspectable signature but do accept
    no-arg construction, so they return False. The result is cached per class since it is purely a
    function of the class definition. Note this inspects the call signature only; it deliberately
    does not run ``__init__``, so a constructor that accepts no args but raises internally is still
    reported as no-arg-constructible (its error is allowed to surface at call time, not swallowed).
    This reflects ``__new__`` when ``__init__`` is inherited; a class that requires args in
    ``__new__`` is out of scope (it violates DataClassDecomposer requirement #2 and cannot be rebuilt
    via ``cls.__new__(cls)`` anyway).
    """
    try:
        signature = inspect.signature(cls)
    except (ValueError, TypeError):
        return False
    try:
        signature.bind()
    except TypeError:
        return True
    return False


def new_empty_container(cls: type):
    """Create an empty instance of ``cls`` to be repopulated during (de)serialization.

    The exact (sub)class type is preserved so ``strict_types`` packing routes the value back through
    its decomposer instead of losing the type. When ``cls.__init__`` requires positional arguments
    (e.g. dict-based edge models like ``SelectionRequest(device_info, job_id)``), ``__init__`` is
    bypassed via ``__new__`` since only an empty container is needed. Constructor state such as a
    defaultdict's default_factory is not preserved in that case.
    """
    if _requires_init_args(cls):
        return cls.__new__(cls)
    return cls()


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

    def supported_dots(self) -> Optional[List[int]]:
        """Return the Datum Object Types supported by this decomposer.
        If a DOT is returned, this decomposer's process_datum method will be called for any datum whose DOT
        matches this DOT.

        Returns: None or list of DOTs

        """
        pass

    def process_datum(self, datum: Datum, manager: DatumManager):
        """This method will be called during message deserialization to process the specified datum.

        Args:
            datum: the datum to be processed
            manager: the datum manger

        Returns: None

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


class Externalizer:
    """
    This class is used to help creating 'decompose' method of decomposers of arbitrary classes.

    """

    def __init__(self, manager: DatumManager):
        self.manager = manager

    def externalize(self, target: Any):
        """Recursively externalize leaf nodes without mutating the source containers.

        Dict/list subclasses are reconstructed at every level (see ``new_empty_container``), so nested
        container subclasses (both dict and list) may not preserve constructor state such as a
        defaultdict's default_factory or the capacity/initializer argument of a list subclass.
        """
        if not self.manager:
            return target

        if isinstance(target, dict):
            new_target = new_empty_container(type(target))
            for k, v in target.items():
                new_target[k] = self.externalize(v)
            return new_target
        elif isinstance(target, list):
            # Note: tuple is not supported since it is immutable.
            new_target = new_empty_container(type(target))
            for v in target:
                new_target.append(self.externalize(v))
            return new_target
        else:
            return self.manager.externalize(target)


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
        # Convert the top-level dict subclass to a plain dict. Externalizer preserves dict/list subclass types,
        # including nested subclasses, so starting from a plain copy prevents msgpack from re-entering this decomposer.
        tc = target.copy()
        externalizer = Externalizer(manager)
        return externalizer.externalize(tc)

    def recompose(self, data: dict, manager: DatumManager = None) -> dict:
        internalizer = Internalizer(manager)
        data = internalizer.internalize(data)
        # Mirror the encode side (Externalizer): bypass __init__ for dict subclasses whose
        # constructor requires arguments, so decode does not raise the same TypeError.
        obj = new_empty_container(self.dict_type)
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

    Instance ``__dict__`` attributes are always captured; ``dict`` and ``list`` subclass contents are
    additionally captured so the elements survive the round-trip (the instance is rebuilt via
    ``__new__``, so ``__init__`` is never invoked to repopulate them).
    """

    def __init__(self, data_type: Type[T]):
        self.data_type = data_type

    def supported_type(self) -> Type[T]:
        return self.data_type

    def decompose(self, target: T, manager: DatumManager = None) -> Any:
        data = {}

        if hasattr(target, "__dict__"):
            data[DATA_CONTENT] = vars(target)

        if isinstance(target, dict):
            data[DICT_CONTENT] = dict(target)

        if isinstance(target, list):
            data[LIST_CONTENT] = list(target)

        return data

    def recompose(self, data: dict, manager: DatumManager = None) -> T:
        instance = self.data_type.__new__(self.data_type)

        data_content = data.get(DATA_CONTENT, None)
        if data_content is not None:
            instance.__dict__.update(data_content)

        dict_content = data.get(DICT_CONTENT, None)
        if dict_content is not None:
            instance.update(dict_content)

        list_content = data.get(LIST_CONTENT, None)
        if list_content is not None:
            instance.extend(list_content)

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
