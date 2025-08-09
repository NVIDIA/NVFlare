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
import threading
from enum import Enum
from typing import Any, Dict, Optional, Set, Type, Union

from nvflare.fuel.utils.class_loader import get_class_name, load_class
from nvflare.fuel.utils.fobs import Decomposer
from nvflare.fuel.utils.fobs.builtin_decomposers import BUILTIN_DECOMPOSERS, BUILTIN_HANDLERS
from nvflare.fuel.utils.fobs.decomposer import (
    DataClassDecomposer,
    DictDecomposer,
    EnumTypeDecomposer,
    GenericDecomposer,
)
from nvflare.fuel.utils.log_utils import get_obj_logger


class DecomposerRegistrar:
    """
    DecomposerRegistrar is responsible for managing and registering decomposers for FOBS.

    It supports on-demand registration to avoid type interdependencies with other libraries
    """

    def __init__(self):
        """
        Initializes a new instance of DecomposerRegistrar with mappings for builtin types
        and dot and ensures thread-safe operations using a lock.
        """

        self.logger = get_obj_logger(self)
        self.lock = threading.RLock()

        # All decomposer registered
        self.decomposer_list: Set[str] = set()
        # Object to decomposer name mapping
        self.type_mapping: Dict[str, str] = {}
        # DOT to decomposer name mapping
        self.dot_mapping: Dict[int, str] = {}
        # A cache of decomposers for each type so decomposer is only instantiated once
        self.type_cache: Dict[str, Decomposer] = {}
        # A cache of decomposers for each DOT
        self.dot_cache: Dict[int, Decomposer] = {}

        self.data_generic_enabled = True
        self.enum_generic_enabled = True

        self.register_builtins()

    def reset(self):
        """
        Reset the decomposer registrar to initial state.
        """
        with self.lock:
            self.decomposer_list.clear()
            self.type_mapping.clear()
            self.dot_mapping.clear()
            self.type_cache.clear()
            self.dot_cache.clear()

        self.register_builtins()

    def count(self) -> int:
        """
        Get the number of decomposers registered.
        """
        with self.lock:
            return len(self.decomposer_list)

    def set_data_generic_enabled(self, value: bool):
        """
        Enable or disable the generic data-class decomposer fallback.

        Args:
            value (bool): True to enable the generic data-class decomposer, False to disable it.
        """

        self.data_generic_enabled = value

    def set_enum_generic_enabled(self, value: bool):
        """
        Enable or disable the generic enum decomposer fallback.

        Args:
            value (bool): True to enable the generic enum decomposer, False to disable it.
        """

        self.enum_generic_enabled = value

    def registered(self, decomposer_name: str) -> bool:
        """
        Check whether a decomposer has been registered.

        Args:
            decomposer_name (str): Fully qualified class name of the decomposer.

        Returns:
            bool: True if the decomposer is registered, otherwise False.
        """

        with self.lock:
            return decomposer_name in self.decomposer_list

    def load(self, decomposer_name: str, type_name: str) -> Decomposer:
        """
        Load a decomposer by name. This is used by unpack when the decomposer name
        is already known
        """
        with self.lock:
            if type_name in self.type_cache:
                return self.type_cache[type_name]

            target_type = load_class(type_name)
            instance = self.instantiate_decomposer(decomposer_name, target_type)
            self.register(instance)
            self.type_cache[type_name] = instance
            return instance

    def find_for_object(self, obj: Any) -> Optional[Decomposer]:
        """
        Find and load decomposer for the given object.

        Args:
            obj: The object for which the decomposer is searched.

        Returns:
            Decomposer: The decomposer instance associated with the given object,
            or None if not found.
        """

        target_type = type(obj)
        type_name = get_class_name(target_type)

        with self.lock:

            # Check if the decomposer is in cache
            decomposer = self.type_cache.get(type_name)
            if decomposer:
                return decomposer

            # Check if a decomposer name is registered for the type
            decomposer_name = self.type_mapping.get(type_name)
            if decomposer_name:
                decomposer = self.instantiate_decomposer(decomposer_name, target_type)
                self.type_cache[type_name] = decomposer
                return decomposer

            if self.enum_generic_enabled and issubclass(target_type, Enum):
                decomposer = EnumTypeDecomposer(target_type)
                self.type_cache[type_name] = decomposer
                self.logger.warning(f"Generic enum decomposer is used for {type_name}")
                return decomposer

            if self.data_generic_enabled:
                # Generic decomposer can only handle classes with __dict__
                if not callable(obj) and hasattr(obj, "__dict__"):
                    decomposer = DataClassDecomposer(target_type)
                    self.type_cache[type_name] = decomposer
                    self.logger.warning(f"Generic data class decomposer is used for {type_name}")
                    return decomposer

        raise TypeError(f"Can't find decomposer for type {type_name}")

    def find_for_dot(self, dot: int) -> Optional[Decomposer]:
        """
        Finds and retrieves the decomposer associated with the specified dot identifier.

        Args:
            dot (int): DOT  used to locate the decomposer.

        Returns:
            Decomposer: The decomposer associated with the dot, or None if not found.
        """

        decomposer = self.dot_cache.get(dot)
        if decomposer:
            return decomposer

        decomposer_name = self.dot_mapping.get(dot)
        if decomposer_name:
            decomposer_class = load_class(decomposer_name)
            decomposer = decomposer_class()
            self.dot_cache[dot] = decomposer
            return decomposer

        raise TypeError(f"No decomposer found for DOT {dot}")

    def register(self, decomposer: Union[Decomposer, Type[Decomposer]]):
        """
        Registers a decomposer. This method initiates decomposer right away
        and require it to be available

        Args:
            decomposer (Type|Decomposer|str): The decomposer to be registered. It could be a type,
                                              a decomposer instance

        Raises:
            ValueError: If the decomposer cannot be registered due to invalid input or conflicts.
        """

        if inspect.isclass(decomposer):
            instance = decomposer()
        else:
            instance = decomposer

        if not isinstance(instance, Decomposer):
            raise ValueError(f"Class {instance.__class__} is not a decomposer")

        decomposer_name = get_class_name(type(instance))
        type_name = get_class_name(instance.supported_type())
        self.register_type(type_name, decomposer_name)

        with self.lock:
            self.type_cache[type_name] = instance

        dots = instance.supported_dots()
        if not dots:
            return

        for dot in dots:
            self.register_dot(dot, decomposer_name)

        with self.lock:
            for dot in dots:
                if not isinstance(dot, int):
                    self.logger.error(f"Bad DOT {dot} - it must be a positive int but got {type(dot)}")
                    continue

                if dot <= 0:
                    self.logger.error(f"Bad DOT {dot} - it must be a positive int")
                    continue

                if dot in self.dot_mapping:
                    self.logger.debug(f"Duplicate registration for DOT {dot}: {self.dot_mapping[dot]}")
                    continue

                self.dot_cache[dot] = instance

    def register_decomposers(self, decomposer_names: list[str]):
        """
        Registers a list of decomposers

        Args:
            decomposer_names : List of decomposer names
        """
        for decomposer_name in decomposer_names:
            try:
                decomposer_class = load_class(decomposer_name)
                self.register(decomposer_class)
            except Exception as e:
                self.logger.error(f"Can't load decomposer {decomposer_name}: {e}")

    def register_type(
        self,
        type_name: str,
        decomposer_name: str,
    ):
        """
        Register a type-hint for the decomposer. This call doesn't instantiate the
        decomposer until it's used.

        Args:
            type_name (str): The name of the type being associated with the decomposer.
            decomposer_name (str): The name of the decomposer being registered.

        """
        with self.lock:
            self.type_mapping[type_name] = decomposer_name
            self.decomposer_list.add(decomposer_name)

    def register_types(self, decomposers: dict[str, str]):
        """
        Register multiple type-to-decomposer mappings.

        Args:
            decomposers (dict[str, str]): Mapping of type name to decomposer class name.
        """

        for k, v in decomposers.items():
            self.register_type(k, v)

    def register_dot(self, dot: int, decomposer_name: str):
        """
        Associates a decomposer name with specific dot identifiers. A dot can either be a single integer
        or a list of integers.

        Args:
            dot (int): A single dot or a list of dots for unique identification.
            decomposer_name (str): The name of the decomposer being registered.


        """
        with self.lock:
            self.dot_mapping[dot] = decomposer_name
            self.decomposer_list.add(decomposer_name)

    def register_dots(self, dots: dict[int, str]):
        """
        Register multiple DOT-to-decomposer mappings.

        Args:
            dots (dict[int, str]): Mapping of DOT identifiers to decomposer class names.
        """

        for k, v in dots.items():
            self.register_dot(k, v)

    def register_builtins(self):
        """
        Register built-in type and DOT mappings, and generic decomposer classes.
        """
        self.register_types(BUILTIN_DECOMPOSERS)
        self.register_dots(BUILTIN_HANDLERS)

        # Register all generic classes
        self.decomposer_list.add(get_class_name(EnumTypeDecomposer))
        self.decomposer_list.add(get_class_name(DictDecomposer))
        self.decomposer_list.add(get_class_name(DataClassDecomposer))

    @staticmethod
    def instantiate_decomposer(decomposer_name: str, data_type: Type):
        """
        Instantiate a decomposer by class name, handling generic decomposers that
        require the target data type.

        Args:
            decomposer_name (str): Fully qualified class name of the decomposer to instantiate.
            data_type (Type): The concrete data type the decomposer should support.

        Returns:
            Decomposer: An instance of the requested decomposer.

        Raises:
            ValueError: If the named class is not a decomposer class.
        """

        decomposer_class = load_class(decomposer_name)
        if issubclass(decomposer_class, GenericDecomposer):
            decomposer = decomposer_class(data_type)
        elif issubclass(decomposer_class, Decomposer):
            decomposer = decomposer_class()
        else:
            raise ValueError(f"Class {decomposer_name} is not a decomposer class")

        return decomposer
