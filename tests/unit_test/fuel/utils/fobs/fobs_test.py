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

import queue
from datetime import datetime
from typing import Any

import pytest

from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.fobs import Decomposer
from nvflare.fuel.utils.fobs.datum import DatumManager


class TestFobs:

    NUMBER = 123456
    FLOAT = 123.456
    NAME = "FOBS Test"
    SET = {4, 5, 6}
    NOW = datetime.now()

    test_data = {
        "str": "Test string",
        "number": NUMBER,
        "float": FLOAT,
        "list": [7, 8, 9],
        "set": SET,
        "tuple": ("abc", "xyz"),
        "time": NOW,
    }

    def test_builtin(self):
        buf = fobs.dumps(TestFobs.test_data)
        data = fobs.loads(buf)
        assert data["number"] == TestFobs.NUMBER
        assert data["set"] == TestFobs.SET

    def test_unsupported_classes(self):
        with pytest.raises(TypeError):
            # Queue contains collections.deque, which has no __dict__, can't be handled as Data Class
            unsupported_class = queue.Queue()
            try:
                fobs.dumps(unsupported_class)
            except Exception as ex:
                print(ex)
                raise ex

    def test_decomposers(self):
        test_class = ExampleClass(TestFobs.NUMBER)
        fobs.register(ExampleClassDecomposer)
        buf = fobs.dumps(test_class)
        new_class = fobs.loads(buf)
        assert new_class.number == TestFobs.NUMBER

    def test_no_registration(self):
        test_class = ExampleClass(TestFobs.NUMBER)
        fobs.register(ExampleClassDecomposer)
        buf = fobs.dumps(test_class)
        # Clear registration before deserializing
        fobs.reset()
        new_class = fobs.loads(buf)
        assert new_class.number == TestFobs.NUMBER

    def test_auto_registration(self):
        fobs.reset()
        test_class = ExampleDataClass(TestFobs.NAME)
        # No decomposer is registered for ExampleDataClass
        buf = fobs.dumps(test_class)
        new_class = fobs.loads(buf)
        assert new_class.name == TestFobs.NAME

    def test_buffer_list(self):
        buf = fobs.dumps(TestFobs.test_data, buffer_list=True)
        data = fobs.loads(buf)
        assert data["number"] == TestFobs.NUMBER


class ExampleClass:
    def __init__(self, number):
        self.number = number


class ExampleDataClass:
    def __init__(self, name):
        self.name = name


class ExampleClassDecomposer(Decomposer):
    def supported_type(self):
        return ExampleClass

    def decompose(self, target: ExampleClass, manager: DatumManager = None) -> Any:
        return target.number

    def recompose(self, data: Any, manager: DatumManager = None) -> ExampleClass:
        return ExampleClass(data)
