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

import warnings

import pytest

from nvflare.fuel.utils.deprecated import _WARNED_DEPRECATION_MESSAGES, deprecated, warn_deprecated

_TEST_WARN_MSG = "custom deprecation message"
_TEST_WARN_ONCE_MSGS = ("unique same deprecation message", "unique different deprecation message")
_TEST_DECORATOR_WARN_MSGS = (
    "Call to deprecated function test_f.",
    "Call to deprecated function test_f (please use new_test_f).",
    "Call to deprecated class TestClass.",
    "Call to deprecated class TestClass (please use NewTestClass).",
)
_TEST_WARN_MSGS = (_TEST_WARN_MSG, *_TEST_WARN_ONCE_MSGS, *_TEST_DECORATOR_WARN_MSGS)


class TestDeprecated:
    def setup_method(self):
        for msg in _TEST_WARN_MSGS:
            _WARNED_DEPRECATION_MESSAGES.discard(msg)

    def test_warn_deprecated(self):
        with warnings.catch_warnings(record=True) as records:
            warn_deprecated(_TEST_WARN_MSG)

        assert len(records) == 1
        assert records[0].category is DeprecationWarning
        assert str(records[0].message) == _TEST_WARN_MSG
        assert records[0].filename == __file__

    def test_warn_deprecated_once_per_message(self):
        with warnings.catch_warnings(record=True) as records:
            warn_deprecated(_TEST_WARN_ONCE_MSGS[0])
            warn_deprecated(_TEST_WARN_ONCE_MSGS[0])
            warn_deprecated(_TEST_WARN_ONCE_MSGS[1])

        assert len(records) == 2
        assert str(records[0].message) == _TEST_WARN_ONCE_MSGS[0]
        assert str(records[1].message) == _TEST_WARN_ONCE_MSGS[1]

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
