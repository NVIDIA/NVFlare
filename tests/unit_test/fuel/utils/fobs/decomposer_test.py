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

import pytest

from nvflare.apis.shareable import Shareable, make_copy
from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.fobs.datum import DatumManager, DatumRef
from nvflare.fuel.utils.fobs.decomposer import DictDecomposer, Externalizer


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
        type_name = "tests.unit_test.fuel.utils.fobs.decomposer_test.EnumClass"
        test_enum = EnumClass.A
        buffer = fobs.dumps(test_enum)
        fobs.reset()
        with pytest.raises(ValueError, match="not allowed"):
            fobs.loads(buffer)

        # Re-add to whitelist after reset so deserialization exercises the whitelist gate.
        fobs.add_type_name_whitelist(type_name)
        new_data = fobs.loads(buffer)
        assert type(test_enum) == type(
            new_data
        ), f"Original type {type(test_enum)} doesn't match new data type {type(new_data)}"
        assert test_enum == new_data, f"Original data {test_enum} doesn't match new data {new_data}"

    def test_generic_int_enum_type(self):
        type_name = "tests.unit_test.fuel.utils.fobs.decomposer_test.IntEnumClass"
        test_enum = IntEnumClass.X
        buffer = fobs.dumps(test_enum)
        fobs.reset()
        with pytest.raises(ValueError, match="not allowed"):
            fobs.loads(buffer)

        # Re-add to whitelist after reset so deserialization exercises the whitelist gate.
        fobs.add_type_name_whitelist(type_name)
        new_data = fobs.loads(buffer)
        assert type(test_enum) == type(
            new_data
        ), f"Original type {type(test_enum)} doesn't match new data type {type(new_data)}"
        assert test_enum == new_data, f"Original data {test_enum} doesn't match new data {new_data}"

    def test_ordered_dict(self):

        test_list = [(3, "First"), (1, "Middle"), (2, "Last")]
        test_data = OrderedDict(test_list)

        buffer = fobs.dumps(test_data)
        fobs.reset()
        new_data = fobs.loads(buffer)
        new_list = list(new_data.items())

        assert test_list == new_list

    def test_externalizer_does_not_mutate_shared_nested_payload(self):
        blob = b"x" * 2048
        list_blob = b"y" * 2048
        nested_list_blob = b"z" * 2048
        source = Shareable({"data": {"blob": blob, "items": [list_blob, {"nested": nested_list_blob}]}})

        first_copy = make_copy(source)
        # Payloads are larger than the 1024-byte threshold, so externalize()
        # must replace them with DatumRefs in the returned tree only.
        Externalizer(DatumManager(1024)).externalize(first_copy)

        assert source["data"]["blob"] == blob
        assert source["data"]["items"][0] == list_blob
        assert source["data"]["items"][1]["nested"] == nested_list_blob
        assert not isinstance(source["data"]["blob"], DatumRef)
        assert not isinstance(source["data"]["items"][0], DatumRef)
        assert not isinstance(source["data"]["items"][1]["nested"], DatumRef)

        second_copy = make_copy(source)
        restored = fobs.loads(fobs.dumps(second_copy, max_value_size=1024))
        assert restored["data"]["blob"] == blob
        assert restored["data"]["items"][0] == list_blob
        assert restored["data"]["items"][1]["nested"] == nested_list_blob

    @staticmethod
    def _check_decomposer(data, clear_decomposers=True):

        buffer = fobs.dumps(data)
        if clear_decomposers:
            fobs.reset()
        new_data = fobs.loads(buffer)
        assert type(data) == type(new_data), f"Original type {type(data)} doesn't match new data type {type(new_data)}"
        assert data == new_data, f"Original data {data} doesn't match new data {new_data}"
