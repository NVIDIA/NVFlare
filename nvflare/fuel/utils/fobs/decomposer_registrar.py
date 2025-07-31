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

import threading
from typing import Type, List, Union, Optional

from nvflare.fuel.utils.fobs import Decomposer

class DecomposerRegistrar:
    """
    DecomposerRegistrar is responsible for managing and registering decomposers for FOBS.

    It only stores decomposer's names and instantiates it on demand. This is to avoid type
    interdependencies in FOBS.

    """

    def __init__(self):
        """
        Initializes a new instance of DecomposerRegistrar with empty mappings for type and dot identifiers
        and ensures thread-safe operations using a lock.
        """
        self.lock = threading.Lock()
        self.decomposer_list = set()
        self.type_mapping = {}
        self.dot_mapping = {}
        self.decomposer_cache = {}
        self.generic_enabled = True

    def registered(self, decomposer_name: str) -> bool:
        with self.lock:
            return decomposer_name in self.decomposer_list

    def find_by_type(self, target_type: Union[Type, str]) -> Optional[Decomposer]:
        """
        Finds and retrieves the decomposer associated with the specified data type.

        Args:
            target_type (Type|str): The target type (class or its full name) for which
                                    the decomposer is searched.

        Returns:
            Decomposer: The decomposer associated with the given type, or None if not found.
        """
        pass

    def find_by_dot(self, dot: int) -> Optional[Decomposer]:
        """
        Finds and retrieves the decomposer associated with the specified dot identifier.

        Args:
            dot (int): DOT  used to locate the decomposer.

        Returns:
            Decomposer: The decomposer associated with the dot, or None if not found.
        """
        pass

    def register(self, decomposer: Union[Type, Decomposer, str]):
        """
        Registers a decomposer. This method initiates decomposer right away
        and require it to be available

        Args:
            decomposer (Type|Decomposer|str): The decomposer to be registered. It could be a type,
                                              a decomposer instance, or a name string.

        Raises:
            ValueError: If the decomposer cannot be registered due to invalid input or conflicts.
        """
        pass

    def register_type(self, decomposer_name: str, type_name: str):
        """
        Register a type-hint for the decomposer. This call doesn't instantiate the
        decomposer until it's used.

        Args:
            decomposer_name (str): The name of the decomposer being registered.
            type_name (str): The name of the type being associated with the decomposer.

        Raises:
            KeyError: If the decomposer name or type name already exists in the mapping.
        """
        pass

    def register_dot(self, decomposer_name: str, dots: Union[int, List[int]]):
        """
        Associates a decomposer name with specific dot identifiers. A dot can either be a single integer
        or a list of integers.

        Args:
            decomposer_name (str): The name of the decomposer being registered.
            dots (int|List[int]): A single dot or a list of dots for unique identification.

        Raises:
            KeyError: If the decomposer name or dot identifiers already exist in the mapping.
        """
        pass
