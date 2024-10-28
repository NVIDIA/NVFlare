# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.utils.decorators import collect_time, measure_time


class MyClass:
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
        assert c.method2.count == 0

        for i in range(100):
            c.method3(c.method2())

        assert c.method2.time_taken > 0
        assert c.method3.time_taken > 0
        assert c.method2.count == 100
        assert c.method3.count == 100
