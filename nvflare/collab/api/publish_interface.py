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

_POSITIONAL_ONLY = "POSITIONAL_ONLY"
_POSITIONAL_OR_KEYWORD = "POSITIONAL_OR_KEYWORD"
_KEYWORD_ONLY = "KEYWORD_ONLY"
_SUPPORTED_KINDS = {_POSITIONAL_ONLY, _POSITIONAL_OR_KEYWORD, _KEYWORD_ONLY}


class MethodParameter:
    """Serializable subset of an inspect.Parameter used for remote binding."""

    def __init__(self, name: str, kind: str, required: bool, legacy=False):
        if not isinstance(name, str):
            raise TypeError(f"parameter name must be str but got {type(name)}")
        if not name:
            raise ValueError("parameter name must not be empty")
        if kind not in _SUPPORTED_KINDS:
            raise ValueError(f"unsupported parameter kind '{kind}' for '{name}'")
        if not isinstance(required, bool):
            raise TypeError(f"required for parameter '{name}' must be bool")
        self.name = name
        self.kind = kind
        self.required = required
        self.legacy = legacy

    @classmethod
    def from_wire(cls, value):
        if isinstance(value, str):
            return cls(value, _POSITIONAL_OR_KEYWORD, required=False, legacy=True)
        if not isinstance(value, Mapping):
            raise TypeError(f"parameter specification must be str or mapping but got {type(value)}")
        return cls(
            name=value.get("name"),
            kind=value.get("kind"),
            required=value.get("required"),
        )

    def to_dict(self):
        return {
            "name": self.name,
            "kind": self.kind,
            "required": self.required,
        }


class MethodInterface(Sequence[str]):
    """Immutable method signature used to bind user calls before transport."""

    def __init__(self, parameters):
        self.parameters = tuple(MethodParameter.from_wire(p) for p in parameters)
        self._by_name = MappingProxyType({p.name: p for p in self.parameters})
        if len(self._by_name) != len(self.parameters):
            raise ValueError("method has duplicate parameter names")

    def bind(self, args, kwargs) -> dict:
        positional_parameters = [p for p in self.parameters if p.kind in (_POSITIONAL_ONLY, _POSITIONAL_OR_KEYWORD)]
        if len(args) > len(positional_parameters):
            raise TypeError(f"takes {len(positional_parameters)} positional arguments but {len(args)} were given")

        bound = dict(kwargs)
        for parameter, value in zip(positional_parameters, args):
            if parameter.name in bound:
                raise TypeError(f"got multiple values for argument '{parameter.name}'")
            bound[parameter.name] = value

        for name in kwargs:
            parameter = self._by_name.get(name)
            if parameter is None:
                raise TypeError(f"got an unexpected keyword argument '{name}'")
            if parameter.kind == _POSITIONAL_ONLY:
                raise TypeError(f"got positional-only argument passed as keyword: '{name}'")

        missing = [p.name for p in self.parameters if p.required and p.name not in bound]
        if missing:
            names = ", ".join(repr(name) for name in missing)
            raise TypeError(f"missing required argument(s): {names}")
        return bound

    def validate_normalized(self, args, kwargs):
        if args:
            raise TypeError("positional arguments must be normalized to keyword arguments")
        unexpected = [name for name in kwargs if name not in self._by_name]
        if unexpected:
            raise TypeError(f"got an unexpected keyword argument '{unexpected[0]}'")
        missing = [p.name for p in self.parameters if p.required and p.name not in kwargs]
        if missing:
            names = ", ".join(repr(name) for name in missing)
            raise TypeError(f"missing required argument(s): {names}")

    def prepare_invocation(self, kwargs):
        """Restore positional-only values after transport normalization."""
        call_kwargs = dict(kwargs)
        call_args = []
        for parameter in self.parameters:
            if parameter.kind == _POSITIONAL_ONLY and parameter.name in call_kwargs:
                call_args.append(call_kwargs.pop(parameter.name))
        return call_args, call_kwargs

    def to_wire(self):
        if all(p.legacy for p in self.parameters):
            return [p.name for p in self.parameters]
        return [p.to_dict() for p in self.parameters]

    def __getitem__(self, index):
        return self.parameters[index].name

    def __len__(self):
        return len(self.parameters)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.to_wire()!r})"


class PublishInterface(Mapping[str, MethodInterface]):
    """Immutable description of an object's published Collab methods."""

    def __init__(self, methods: Optional[Mapping[str, Sequence]] = None):
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
                raise TypeError(f"parameters for method '{method_name}' must be a sequence")

            try:
                normalized[method_name] = MethodInterface(param_names)
            except (TypeError, ValueError) as ex:
                raise type(ex)(f"invalid parameters for method '{method_name}': {ex}") from ex

        self._methods = MappingProxyType(normalized)

    @classmethod
    def from_dict(cls, methods: Optional[Mapping[str, Sequence]]) -> "PublishInterface":
        if isinstance(methods, cls):
            return methods
        return cls(methods)

    def to_dict(self) -> dict:
        """Return the plain-dict representation used for synchronization."""
        return {method_name: method.to_wire() for method_name, method in self._methods.items()}

    def get_method(self, method_name: str) -> Optional[MethodInterface]:
        """Return a method interface, preserving an empty interface for zero-argument methods."""
        return self._methods.get(method_name)

    def __getitem__(self, method_name: str) -> MethodInterface:
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
