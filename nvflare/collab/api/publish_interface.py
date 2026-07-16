# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from collections.abc import Iterator, Mapping, Sequence
from types import MappingProxyType
from typing import Optional


class PublishInterface(Mapping[str, tuple[str, ...]]):
    """Immutable description of an object's published Collab methods."""

    def __init__(self, methods: Optional[Mapping[str, Sequence[str]]] = None):
        if methods is None:
            methods = {}
        elif not isinstance(methods, Mapping):
            raise TypeError(f"methods must be a mapping but got {type(methods)}")

        normalized = {}
        for method_name, param_names in methods.items():
            if not isinstance(method_name, str):
                raise TypeError(f"method name must be str but got {type(method_name)}")
            if not method_name:
                raise ValueError("method name must not be empty")
            if isinstance(param_names, (str, bytes)) or not isinstance(param_names, Sequence):
                raise TypeError(f"parameters for method '{method_name}' must be a sequence of str")

            params = []
            for param_name in param_names:
                if not isinstance(param_name, str):
                    raise TypeError(f"parameter name for method '{method_name}' must be str but got {type(param_name)}")
                if not param_name:
                    raise ValueError(f"parameter name for method '{method_name}' must not be empty")
                params.append(param_name)

            if len(params) != len(set(params)):
                raise ValueError(f"method '{method_name}' has duplicate parameter names")
            normalized[method_name] = tuple(params)

        self._methods = MappingProxyType(normalized)

    @classmethod
    def from_dict(cls, methods: Optional[Mapping[str, Sequence[str]]]) -> "PublishInterface":
        if isinstance(methods, cls):
            return methods
        return cls(methods)

    def to_dict(self) -> dict[str, list[str]]:
        """Return the plain-dict representation used for synchronization."""
        return {method_name: list(param_names) for method_name, param_names in self._methods.items()}

    def get_method(self, method_name: str) -> Optional[tuple[str, ...]]:
        """Return parameter names, preserving an empty tuple for zero-argument methods."""
        return self._methods.get(method_name)

    def __getitem__(self, method_name: str) -> tuple[str, ...]:
        return self._methods[method_name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._methods)

    def __len__(self) -> int:
        return len(self._methods)

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        memo[id(self)] = self
        return self

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.to_dict()!r})"
