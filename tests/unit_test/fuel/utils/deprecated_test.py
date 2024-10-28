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

from nvflare.fuel.utils.deprecated import deprecated


class TestDeprecated:
    def test_deprecated_func_one_arg(self):
        @deprecated
        def test_f(a, b):
            print(f"hello {a} and {b}")

        with pytest.warns(DeprecationWarning, match=r"Call to deprecated function test_f."):
            test_f(5, 6)

    def test_deprecated_func_with_string(self):
        @deprecated("please use new_test_f")
        def test_f(a, b):
            print(f"hello {a} and {b}")

        with pytest.warns(DeprecationWarning, match=r"Call to deprecated function test_f \(please use new_test_f\)."):
            test_f(5, 6)

    def test_deprecated_class_one_arg(self):
        @deprecated
        class TestClass:
            def __init__(self):
                print("I am a test class")

        with pytest.warns(DeprecationWarning, match=r"Call to deprecated class TestClass."):
            _ = TestClass()

    def test_deprecated_class_with_string(self):
        @deprecated("please use NewTestClass")
        class TestClass:
            def __init__(self):
                print("I am a test class")

        with pytest.warns(DeprecationWarning, match=r"Call to deprecated class TestClass \(please use NewTestClass\)."):
            _ = TestClass()
