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

from nvflare.apis.controller_spec import SendOrder
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable

from .controller_test import TestController, create_client, create_task


def _get_common_test_cases():
    test_cases = [
        [{"task": list(), "fl_ctx": FLContext()}, TypeError, "task must be an instance of Task."],
        [
            {"task": create_task("__test"), "fl_ctx": list()},
            TypeError,
            "fl_ctx must be an instance of FLContext.",
        ],
        [
            {"task": create_task("__test"), "fl_ctx": FLContext(), "targets": dict()},
            TypeError,
            "targets must be a list of Client or string.",
        ],
        [
            {"task": create_task("__test"), "fl_ctx": FLContext(), "targets": [1, 2, 3]},
            TypeError,
            "targets must be a list of Client or string.",
        ],
    ]
    return test_cases


def _get_broadcast_test_cases():
    test_cases = [
        [
            {"task": create_task("__test"), "fl_ctx": FLContext(), "min_responses": -1},
            ValueError,
            "min_responses must >= 0.",
        ],
        [
            {"task": create_task("__test"), "fl_ctx": FLContext(), "min_responses": 1.1},
            TypeError,
            "min_responses must be an instance of int.",
        ],
    ]
    return test_cases


def _get_send_test_cases():
    test_cases = [
        [
            {"task": create_task("__test"), "fl_ctx": FLContext(), "task_assignment_timeout": -1},
            ValueError,
            "task_assignment_timeout must >= 0.",
        ],
        [
            {"task": create_task("__test"), "fl_ctx": FLContext(), "task_assignment_timeout": 1.1},
            TypeError,
            "task_assignment_timeout must be an instance of int.",
        ],
        [
            {
                "task": create_task("__test"),
                "fl_ctx": FLContext(),
                "send_order": SendOrder.SEQUENTIAL,
                "targets": [],
            },
            ValueError,
            "Targets must be provided for send",
        ],
        [
            {"task": create_task("__test"), "fl_ctx": FLContext(), "send_order": "hello"},
            TypeError,
            "send_order must be in Enum SendOrder.",
        ],
        [
            {
                "task": create_task("__test", timeout=2),
                "fl_ctx": FLContext(),
                "task_assignment_timeout": 3,
            },
            ValueError,
            "task_assignment_timeout need to be less than or equal to task.timeout.",
        ],
    ]
    return test_cases


def _get_relay_test_cases():
    test_cases = [
        [
            {"task": create_task("__test"), "fl_ctx": FLContext(), "task_assignment_timeout": -1},
            ValueError,
            "task_assignment_timeout must >= 0.",
        ],
        [
            {"task": create_task("__test"), "fl_ctx": FLContext(), "task_assignment_timeout": 1.1},
            TypeError,
            "task_assignment_timeout must be an instance of int.",
        ],
        [
            {"task": create_task("__test"), "fl_ctx": FLContext(), "task_result_timeout": -1},
            ValueError,
            "task_result_timeout must >= 0.",
        ],
        [
            {"task": create_task("__test"), "fl_ctx": FLContext(), "task_result_timeout": 1.1},
            TypeError,
            "task_result_timeout must be an instance of int.",
        ],
        [
            {"task": create_task("__test"), "fl_ctx": FLContext(), "send_order": "hello"},
            TypeError,
            "send_order must be in Enum SendOrder.",
        ],
        [
            {
                "task": create_task("__test", timeout=2),
                "fl_ctx": FLContext(),
                "task_assignment_timeout": 3,
            },
            ValueError,
            "task_assignment_timeout need to be less than or equal to task.timeout.",
        ],
        [
            {
                "task": create_task("__test", timeout=2),
                "fl_ctx": FLContext(),
                "task_result_timeout": 3,
            },
            ValueError,
            "task_result_timeout need to be less than or equal to task.timeout.",
        ],
        [
            {"task": create_task("__test"), "fl_ctx": FLContext(), "dynamic_targets": False},
            ValueError,
            "Need to provide targets when dynamic_targets is set to False.",
        ],
    ]
    return test_cases


def _get_process_submission_test_cases():
    return [
        [
            {
                "client": None,
                "task_name": "__test_task",
                "fl_ctx": FLContext(),
                "task_id": "abc",
                "result": Shareable(),
            },
            TypeError,
            "client must be an instance of Client.",
        ],
        [
            {
                "client": create_client("__test"),
                "task_name": "__test_task",
                "fl_ctx": None,
                "task_id": "abc",
                "result": Shareable(),
            },
            TypeError,
            "fl_ctx must be an instance of FLContext.",
        ],
        [
            {
                "client": create_client("__test"),
                "task_name": "__test_task",
                "fl_ctx": FLContext(),
                "task_id": "abc",
                "result": "abc",
            },
            TypeError,
            "result must be an instance of Shareable.",
        ],
    ]


class TestInvalidInput(TestController):
    @pytest.mark.parametrize("method", TestController.ALL_APIS)
    @pytest.mark.parametrize("kwargs,error,msg", _get_common_test_cases())
    def test_invalid_input(self, method, kwargs, error, msg):
        controller, fl_ctx = self.start_controller()
        with pytest.raises(error, match=msg):
            if method == "broadcast":
                controller.broadcast(**kwargs)
            elif method == "broadcast_and_wait":
                controller.broadcast_and_wait(**kwargs)
            elif method == "send":
                controller.send(**kwargs)
            elif method == "send_and_wait":
                controller.send_and_wait(**kwargs)
            elif method == "relay":
                controller.relay(**kwargs)
            elif method == "relay_and_wait":
                controller.relay_and_wait(**kwargs)
        self.stop_controller(controller, fl_ctx)

    @pytest.mark.parametrize("method", ["broadcast", "broadcast_and_wait"])
    @pytest.mark.parametrize("kwargs,error,msg", _get_broadcast_test_cases())
    def test_broadcast_invalid_input(self, method, kwargs, error, msg):
        controller, fl_ctx = self.start_controller()
        with pytest.raises(error, match=msg):
            if method == "broadcast":
                controller.broadcast(**kwargs)
            else:
                controller.broadcast_and_wait(**kwargs)
        self.stop_controller(controller, fl_ctx)

    @pytest.mark.parametrize("method", ["send", "send_and_wait"])
    @pytest.mark.parametrize("kwargs,error,msg", _get_send_test_cases())
    def test_send_invalid_input(self, method, kwargs, error, msg):
        controller, fl_ctx = self.start_controller()
        with pytest.raises(error, match=msg):
            if method == "send":
                controller.send(**kwargs)
            else:
                controller.send_and_wait(**kwargs)
        self.stop_controller(controller, fl_ctx)

    @pytest.mark.parametrize("method", ["relay", "relay_and_wait"])
    @pytest.mark.parametrize("kwargs,error,msg", _get_relay_test_cases())
    def test_relay_invalid_input(self, method, kwargs, error, msg):
        controller, fl_ctx = self.start_controller()
        with pytest.raises(error, match=msg):
            if method == "relay":
                controller.relay(**kwargs)
            else:
                controller.relay_and_wait(**kwargs)
        self.stop_controller(controller, fl_ctx)

    @pytest.mark.parametrize("method", TestController.ALL_APIS)
    @pytest.mark.parametrize("kwargs,error,msg", _get_process_submission_test_cases())
    def test_process_submission_invalid_input(self, method, kwargs, error, msg):
        controller, fl_ctx = self.start_controller()

        with pytest.raises(error, match=msg):
            controller.process_submission(**kwargs)
        self.stop_controller(controller, fl_ctx)
