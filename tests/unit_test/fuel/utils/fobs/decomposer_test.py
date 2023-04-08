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

from collections import OrderedDict
from enum import Enum, IntEnum

from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.fobs.decomposer import DictDecomposer


class DataClass:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __eq__(self, other):
        if not isinstance(other, DataClass):
            return False

        return self.a == other.a and self.b == other.b


class EnumClass(str, Enum):
    A = "foo"
    B = "bar"


class IntEnumClass(IntEnum):
    X = 123
    Y = 456


class DictClass(dict):
    pass


class TestDecomposers:
    def test_generic_dict_class(self):
        fobs.register(DictDecomposer(DictClass))
        data = DictClass()
        data["A"] = 123
        data["B"] = "xyz"
        self._check_decomposer(data, False)

    def test_generic_data_class(self):
        fobs.register_data_classes(DataClass)
        data = DataClass("test", 456)
        self._check_decomposer(data, False)

    def test_generic_str_enum_type(self):
        # Decomposers for enum classes are auto-registered by default
        test_enum = EnumClass.A
        self._check_decomposer(test_enum)

    def test_generic_int_enum_type(self):
        # Decomposers for enum classes are auto-registered by default
        test_enum = IntEnumClass.X
        self._check_decomposer(test_enum)

    def test_ordered_dict(self):

        test_list = [(3, "First"), (1, "Middle"), (2, "Last")]
        test_data = OrderedDict(test_list)

        buffer = fobs.dumps(test_data)
        fobs.reset()
        new_data = fobs.loads(buffer)
        new_list = list(new_data.items())

        assert test_list == new_list

    @staticmethod
    def _check_decomposer(data, clear_decomposers=True):

        buffer = fobs.dumps(data)
        if clear_decomposers:
            fobs.reset()
        new_data = fobs.loads(buffer)
        assert type(data) == type(new_data), f"Original type {type(data)} doesn't match new data type {type(new_data)}"
        assert data == new_data, f"Original data {data} doesn't match new data {new_data}"
