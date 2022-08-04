# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Any, Type

import pytest

from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.fobs import Decomposer


class TestFobs:

    NUMBER = 123456
    FLOAT = 123.456
    NOW = datetime.now()

    test_data = {
        "str": "Test string",
        "number": NUMBER,
        "float": FLOAT,
        "list": [7, 8, 9],
        "set": {4, 5, 6},
        "tuple": ("abc", "xyz"),
        "time": NOW,
    }

    def test_builtin(self):
        buf = fobs.dumps(TestFobs.test_data)
        data = fobs.loads(buf)
        assert data["number"] == TestFobs.NUMBER

    def test_aliases(self):
        buf = fobs.dumps(TestFobs.test_data)
        data = fobs.loads(buf)
        assert data["number"] == TestFobs.NUMBER

    def test_unsupported_classes(self):
        with pytest.raises(TypeError):
            # Queue is just a random built-in class not supported by FOBS
            unsupported_class = queue.Queue()
            fobs.dumps(unsupported_class)

    def test_decomposers(self):
        test_class = TestClass(TestFobs.NUMBER)
        fobs.register(TestClassDecomposer)
        buf = fobs.dumps(test_class)
        new_class = fobs.loads(buf)
        assert new_class.number == TestFobs.NUMBER


class TestClass:
    def __init__(self, number):
        self.number = number


class TestClassDecomposer(Decomposer):
    @staticmethod
    def supported_type():
        return TestClass

    def decompose(self, target: TestClass) -> Any:
        return target.number

    def recompose(self, data: Any) -> TestClass:
        return TestClass(data)
