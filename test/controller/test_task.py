# Copyright (c) 2021, NVIDIA CORPORATION.
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

from .controller_test import create_task


def _get_create_task_cases():
    test_cases = [
        [
            {"timeout": -1},
            ValueError,
            "timeout must >= 0.",
        ],
        [
            {"timeout": 1.1},
            TypeError,
            "timeout must be an instance of int.",
        ],
        [
            {"before_task_sent_cb": list()},
            TypeError,
            "before_task_sent must be a callable function.",
        ],
        [
            {"result_received_cb": list()},
            TypeError,
            "result_received must be a callable function.",
        ],
        [
            {"task_done_cb": list()},
            TypeError,
            "task_done must be a callable function.",
        ],
    ]
    return test_cases


class TestTask:
    @pytest.mark.parametrize("kwargs,error,msg", _get_create_task_cases())
    def test_create_task_with_invalid_input(self, kwargs, error, msg):
        with pytest.raises(error, match=msg):
            _ = create_task(name="__test_task", **kwargs)

    def test_set_task_prop(self):
        task = create_task(name="__test_task")
        task.set_prop("hello", "world")
        assert task.props["hello"] == "world"

    def test_get_task_prop(self):
        task = create_task(name="__test_task")
        task.props["hello"] = "world"
        assert task.get_prop("hello") == "world"

    def test_set_task_prop_invalid_key(self):
        task = create_task(name="__test_task")
        with pytest.raises(ValueError, match="Keys start with __ is reserved. Please use other key."):
            task.set_prop("__test", "world")

    def test_get_task_prop_invalid_key(self):
        task = create_task(name="__test_task")
        with pytest.raises(ValueError, match="Keys start with __ is reserved. Please use other key."):
            task.get_prop("__test")
