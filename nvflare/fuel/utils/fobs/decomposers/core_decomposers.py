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
"""Decomposers for Python builtin objects."""
from collections import OrderedDict
from datetime import datetime
from typing import Any

from nvflare.fuel.utils.fobs.decomposer import Decomposer


class TupleDecomposer(Decomposer):
    def supported_type(self):
        return tuple

    def decompose(self, target: tuple) -> Any:
        return list(target)

    def recompose(self, data: Any) -> tuple:
        return tuple(data)


class SetDecomposer(Decomposer):
    def supported_type(self):
        return set

    def decompose(self, target: set) -> Any:
        return list(target)

    def recompose(self, data: Any) -> set:
        return set(data)


class OrderedDictDecomposer(Decomposer):
    def supported_type(self):
        return OrderedDict

    def decompose(self, target: OrderedDict) -> Any:
        return list(target.items())

    def recompose(self, data: Any) -> OrderedDict:
        return OrderedDict(data)


class DatetimeDecomposer(Decomposer):
    def supported_type(self):
        return datetime

    def decompose(self, target: datetime) -> Any:
        return target.isoformat()

    def recompose(self, data: Any) -> datetime:
        return datetime.fromisoformat(data)
