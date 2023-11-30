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

import pytest

from nvflare.fuel.utils.experimental import experimental


class TestExperimental:
    def test_experimental_func_one_arg(self):
        @experimental
        def test_f(a, b):
            print(f"hello {a} and {b}")

        with pytest.warns(Warning, match=r"Call to experimental function test_f."):
            test_f(5, 6)

    def test_experimental_func_with_string(self):
        @experimental("please use new_test_f")
        def test_f(a, b):
            print(f"hello {a} and {b}")

        with pytest.warns(Warning, match=r"Call to experimental function test_f \(please use new_test_f\)."):
            test_f(5, 6)

    def test_experimental_class_one_arg(self):
        @experimental
        class TestClass:
            def __init__(self):
                print("I am a test class")

        with pytest.warns(Warning, match=r"Use of experimental class TestClass."):
            _ = TestClass()

    def test_experimental_class_with_string(self):
        @experimental("Because it's experimental")
        class TestClass:
            def __init__(self):
                print("I am a test class")

        with pytest.warns(Warning, match=r"Use of experimental class TestClass \(Because it's experimental\)."):
            _ = TestClass()

    def test_experimental_class_inheritance_one_arg(self):
        @experimental
        class BaseClass:
            def __init__(self):
                print("I am a base class")

        class TestClass(BaseClass):
            def __init__(self):
                super().__init__()
                print("I am a test class")

        with pytest.warns(Warning, match=r"Use of experimental class BaseClass."):
            _ = TestClass()

    def test_experimental_class_inheritance_with_string(self):
        @experimental("Because it's experimental")
        class BaseClass:
            def __init__(self):
                print("I am a base class")

        class TestClass(BaseClass):
            def __init__(self):
                super().__init__()
                print("I am a test class")

        with pytest.warns(Warning, match=r"Use of experimental class BaseClass \(Because it's experimental\)."):
            _ = TestClass()

    def test_experimental_class_inheritance_one_arg2(self):
        @experimental
        class BaseClass:
            def __init__(self):
                print("I am a base class")

        @experimental
        class TestClass(BaseClass):
            def __init__(self):
                super().__init__()
                print("I am a test class")

        with pytest.warns(Warning, match=r"Use of experimental class BaseClass.") and pytest.warns(
            Warning, match=r"Use of experimental class TestClass."
        ):
            _ = TestClass()

    def test_experimental_class_inheritance_with_string2(self):
        @experimental("Because it's experimental")
        class BaseClass:
            def __init__(self):
                print("I am a base class")

        @experimental("Because it's also experimental")
        class TestClass(BaseClass):
            def __init__(self):
                super().__init__()
                print("I am a test class")

        with pytest.warns(
            Warning, match=r"Use of experimental class BaseClass \(Because it's experimental\)."
        ) and pytest.warns(Warning, match=r"Use of experimental class TestClass \(Because it's also experimental\)."):
            _ = TestClass()

    def test_experimental_func_with_non_string(self):
        with pytest.raises(TypeError):
            a = {"test"}

            @experimental(a)
            def test_f():
                print("test")
