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
import time
from collections import deque

import pytest

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext, FLContextManager
from nvflare.app_common.resource_managers.list_resource_manager import ListResourceManager


class MockEngine:
    def __init__(self, run_name="exp1"):
        self.fl_ctx_mgr = FLContextManager(
            engine=self,
            identity_name="__mock_engine",
            job_id=run_name,
            public_stickers={},
            private_stickers={},
        )

    def new_context(self):
        return self.fl_ctx_mgr.new_context()

    def fire_event(self, event_type: str, fl_ctx: FLContext):
        pass


CHECK_TEST_CASES = [
    ({"gpu": [1, 2, 3, 4]}, {"gpu": 1}, True, {"gpu": [1]}),
    ({"gpu": [1, 2, 3, 4]}, {"gpu": 4}, True, {"gpu": [1, 2, 3, 4]}),
    ({"gpu": [1]}, {"gpu": 1}, True, {"gpu": [1]}),
    ({"gpu": [1], "cpu": [1, 2, 3, 4, 5]}, {"gpu": 1, "cpu": 3}, True, {"gpu": [1], "cpu": [1, 2, 3]}),
    ({"gpu": [1]}, {"gpu": 2}, False, {}),
    ({"gpu": [1, 2]}, {"gpu": 5}, False, {}),
    ({"gpu": [1, 2]}, {"cpu": 1}, False, {}),
]


TEST_CASES = [
    ({"gpu": [1, 2, 3, 4]}, {"gpu": 1}, {"gpu": [1]}),
    ({"gpu": [1, 2, 3, 4]}, {"gpu": 4}, {"gpu": [1, 2, 3, 4]}),
    ({"gpu": [1]}, {"gpu": 1}, {"gpu": [1]}),
    ({"gpu": [1], "cpu": [1, 2, 3, 4, 5]}, {"gpu": 1, "cpu": 3}, {"gpu": [1], "cpu": [1, 2, 3]}),
]


class TestListResourceManager:
    @pytest.mark.parametrize(
        "resources, resource_requirement, expected_check_result, expected_reserved_resources", CHECK_TEST_CASES
    )
    def test_check_resource(self, resources, resource_requirement, expected_check_result, expected_reserved_resources):
        engine = MockEngine()
        list_resource_manager = ListResourceManager(resources=resources)
        with engine.new_context() as fl_ctx:
            check_result, token = list_resource_manager.check_resources(
                resource_requirement=resource_requirement, fl_ctx=fl_ctx
            )
        assert expected_check_result == check_result
        if expected_check_result:
            assert expected_reserved_resources == list_resource_manager.reserved_resources[token][0]

    @pytest.mark.parametrize("resources, resource_requirement, expected_reserved_resources", TEST_CASES)
    def test_cancel_resource(self, resources, resource_requirement, expected_reserved_resources):
        engine = MockEngine()
        list_resource_manager = ListResourceManager(resources=resources)
        with engine.new_context() as fl_ctx:
            _, token = list_resource_manager.check_resources(resource_requirement=resource_requirement, fl_ctx=fl_ctx)
        assert expected_reserved_resources == list_resource_manager.reserved_resources[token][0]
        with engine.new_context() as fl_ctx:
            list_resource_manager.cancel_resources(
                resource_requirement=resource_requirement, token=token, fl_ctx=fl_ctx
            )
        assert list_resource_manager.reserved_resources == {}

    @pytest.mark.parametrize("resources, resource_requirement, expected_reserved_resources", TEST_CASES)
    def test_allocate_resource(self, resources, resource_requirement, expected_reserved_resources):
        engine = MockEngine()
        list_resource_manager = ListResourceManager(resources=resources)
        with engine.new_context() as fl_ctx:
            _, token = list_resource_manager.check_resources(resource_requirement=resource_requirement, fl_ctx=fl_ctx)
        assert expected_reserved_resources == list_resource_manager.reserved_resources[token][0]
        with engine.new_context() as fl_ctx:
            result = list_resource_manager.allocate_resources(
                resource_requirement=resource_requirement, token=token, fl_ctx=fl_ctx
            )
        assert result == expected_reserved_resources

    @pytest.mark.parametrize("resources, resource_requirement, expected_reserved_resources", TEST_CASES)
    def test_free_resource(self, resources, resource_requirement, expected_reserved_resources):
        engine = MockEngine()
        list_resource_manager = ListResourceManager(resources=resources)
        with engine.new_context() as fl_ctx:
            check_result, token = list_resource_manager.check_resources(
                resource_requirement=resource_requirement, fl_ctx=fl_ctx
            )
        assert expected_reserved_resources == list_resource_manager.reserved_resources[token][0]
        with engine.new_context() as fl_ctx:
            result = list_resource_manager.allocate_resources(
                resource_requirement=resource_requirement, token=token, fl_ctx=fl_ctx
            )
        with engine.new_context() as fl_ctx:
            list_resource_manager.free_resources(resources=result, token=token, fl_ctx=fl_ctx)
        assert list_resource_manager.reserved_resources == {}

    def test_check_one_check_two_then_allocate_two_allocate_one(self):
        engine = MockEngine()
        list_resource_manager = ListResourceManager(resources={"gpu": [f"gpu_{i}" for i in range(4)]})
        resource_requirement = {"gpu": 1}

        with engine.new_context() as fl_ctx:
            check1, token1 = list_resource_manager.check_resources(
                resource_requirement=resource_requirement, fl_ctx=fl_ctx
            )
            check2, token2 = list_resource_manager.check_resources(
                resource_requirement=resource_requirement, fl_ctx=fl_ctx
            )

        with engine.new_context() as fl_ctx:
            result = list_resource_manager.allocate_resources(
                resource_requirement=resource_requirement, token=token2, fl_ctx=fl_ctx
            )
        assert result == {"gpu": ["gpu_1"]}

        with engine.new_context() as fl_ctx:
            result = list_resource_manager.allocate_resources(
                resource_requirement=resource_requirement, token=token1, fl_ctx=fl_ctx
            )
        assert result == {"gpu": ["gpu_0"]}

    def test_check_one_cancel_one_check_four_then_allocate_four(self):
        engine = MockEngine()
        list_resource_manager = ListResourceManager(resources={"gpu": [f"gpu_{i}" for i in range(4)]})
        resource_requirement1 = {"gpu": 1}
        resource_requirement2 = {"gpu": 4}

        with engine.new_context() as fl_ctx:
            check1, token1 = list_resource_manager.check_resources(
                resource_requirement=resource_requirement1, fl_ctx=fl_ctx
            )

        with engine.new_context() as fl_ctx:
            list_resource_manager.cancel_resources(
                resource_requirement=resource_requirement1, token=token1, fl_ctx=fl_ctx
            )

        with engine.new_context() as fl_ctx:
            check2, token2 = list_resource_manager.check_resources(
                resource_requirement=resource_requirement2, fl_ctx=fl_ctx
            )
            result = list_resource_manager.allocate_resources(
                resource_requirement=resource_requirement2, token=token2, fl_ctx=fl_ctx
            )
        assert result == {"gpu": ["gpu_0", "gpu_1", "gpu_2", "gpu_3"]}

    def test_check_and_timeout(self):
        timeout = 5
        engine = MockEngine()
        list_resource_manager = ListResourceManager(
            resources={"gpu": [f"gpu_{i}" for i in range(4)]}, expiration_period=timeout
        )
        resource_requirement = {"gpu": 1}

        with engine.new_context() as fl_ctx:
            list_resource_manager.handle_event(event_type=EventType.SYSTEM_START, fl_ctx=fl_ctx)
            check_result, token = list_resource_manager.check_resources(
                resource_requirement=resource_requirement, fl_ctx=fl_ctx
            )
            assert {"gpu": ["gpu_0"]} == list_resource_manager.reserved_resources[token][0]
        time.sleep(timeout + 1)
        with engine.new_context() as fl_ctx:
            list_resource_manager.handle_event(event_type=EventType.SYSTEM_END, fl_ctx=fl_ctx)
        assert list_resource_manager.reserved_resources == {}
        assert list_resource_manager.resources == {"gpu": deque(["gpu_1", "gpu_2", "gpu_3", "gpu_0"])}
