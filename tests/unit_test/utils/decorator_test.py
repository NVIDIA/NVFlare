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

import shutil
import tempfile
from functools import partial
from unittest.mock import MagicMock, patch

import pytest
from requests import Response

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.fuel.utils.network_utils import get_open_ports
from nvflare.tool.package_checker.utils import check_overseer_running, check_response, try_bind_address, try_write_dir
from nvflare.utils.decorators import collect_time, measure_time


def cond(self, duration: float, count: int) -> bool:
    if duration > 1000:
        return True
    if count > 3:
        return True
    return False


class MyClass(FLComponent):
    def __init__(self):
        super().__init__()

    @measure_time
    def method1(self):
        # Some code here
        pass

    @collect_time
    def method2(self) -> dict:
        self.method1()
        pass

    def total(self):
        for i in range(1000):
            self.method2()
        print(" total time (ms) took = ", self.method2.time_taken)
        print(" total count took = ", self.method2.count)

    @collect_time
    def method3(self, x: dict):
        self.method1()
        pass

    @collect_time
    def method3(self, x: dict):
        self.method1()
        pass


class TestDecorators:
    def test_code_timer_on_fn(self):
        @measure_time
        def fn1(x: int, *, a: int, b: int, c: str):
            pass

        a1 = fn1(100, a=1, b=2, c="three")
        print(fn1.time_taken)
        assert fn1.time_taken > 0

    def test_code_timer_on_class_fn(self):
        c = MyClass()

        c.total()

        c.method1()
        assert c.method1.time_taken > 0

        c.method2()
        assert c.method2.time_taken > 0
        c.method2(reset=True)
        assert c.method2.time_taken == 0

        c.method3(c.method2())
        assert c.method3.time_taken > 0

    def test_code_timer_on_class_fn2(self):
        c = MyClass()

        c.total()

        c.method1()
        assert c.method1.time_taken > 0

        c.method2()
        assert c.method2.time_taken > 0
        c.method2(reset=True)
        assert c.method2.time_taken == 0

        c.method3(c.method2())
        assert c.method3.time_taken > 0
