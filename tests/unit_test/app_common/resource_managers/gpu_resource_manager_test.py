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

import time
from unittest.mock import patch

import pytest

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext, FLContextManager
from nvflare.app_common.resource_managers.gpu_resource_manager import GPUResourceManager

NUM_GPU_KEY = "num_of_gpus"
GPU_MEM_KEY = "mem_per_gpu_in_GiB"


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


def _gen_requirement(gpus, gpu_mem):
    return {NUM_GPU_KEY: gpus, GPU_MEM_KEY: gpu_mem}


CHECK_TEST_CASES = [
    (4, 16, _gen_requirement(1, 8), True, {0: 8}),
    (4, 16, _gen_requirement(2, 8), True, {0: 8, 1: 8}),
]


TEST_CASES = [
    (4, 16, _gen_requirement(1, 8), {0: 8}),
    (4, 16, _gen_requirement(2, 8), {0: 8, 1: 8}),
]


@pytest.fixture(scope="class", autouse=True)
def mock_get_host_gpu_ids():
    with patch("nvflare.app_common.resource_managers.gpu_resource_manager.get_host_gpu_ids") as _fixture:
        _fixture.return_value = [0, 1, 2, 3]
        yield _fixture


class TestGPUResourceManager:
    @pytest.mark.parametrize(
        "gpus, gpu_mem, resource_requirement, expected_check_result, expected_reserved_resources", CHECK_TEST_CASES
    )
    def test_check_resource(
        self,
        mock_get_host_gpu_ids,
        gpus,
        gpu_mem,
        resource_requirement,
        expected_check_result,
        expected_reserved_resources,
    ):
        engine = MockEngine()
        gpu_resource_manager = GPUResourceManager(num_of_gpus=gpus, mem_per_gpu_in_GiB=gpu_mem)
        with engine.new_context() as fl_ctx:
            check_result, token = gpu_resource_manager.check_resources(
                resource_requirement=resource_requirement, fl_ctx=fl_ctx
            )
        assert expected_check_result == check_result
        if expected_check_result:
            assert expected_reserved_resources == gpu_resource_manager.reserved_resources[token][0]

    @pytest.mark.parametrize("gpus, gpu_mem, resource_requirement, expected_reserved_resources", TEST_CASES)
    def test_cancel_resource(
        self, mock_get_host_gpu_ids, gpus, gpu_mem, resource_requirement, expected_reserved_resources
    ):
        engine = MockEngine()
        gpu_resource_manager = GPUResourceManager(num_of_gpus=gpus, mem_per_gpu_in_GiB=gpu_mem)
        with engine.new_context() as fl_ctx:
            _, token = gpu_resource_manager.check_resources(resource_requirement=resource_requirement, fl_ctx=fl_ctx)
        assert expected_reserved_resources == gpu_resource_manager.reserved_resources[token][0]
        with engine.new_context() as fl_ctx:
            gpu_resource_manager.cancel_resources(resource_requirement=resource_requirement, token=token, fl_ctx=fl_ctx)
        assert gpu_resource_manager.reserved_resources == {}

    @pytest.mark.parametrize("gpus, gpu_mem, resource_requirement, expected_reserved_resources", TEST_CASES)
    def test_allocate_resource(
        self, mock_get_host_gpu_ids, gpus, gpu_mem, resource_requirement, expected_reserved_resources
    ):
        engine = MockEngine()
        gpu_resource_manager = GPUResourceManager(num_of_gpus=gpus, mem_per_gpu_in_GiB=gpu_mem)
        with engine.new_context() as fl_ctx:
            _, token = gpu_resource_manager.check_resources(resource_requirement=resource_requirement, fl_ctx=fl_ctx)
        assert expected_reserved_resources == gpu_resource_manager.reserved_resources[token][0]
        with engine.new_context() as fl_ctx:
            result = gpu_resource_manager.allocate_resources(
                resource_requirement=resource_requirement, token=token, fl_ctx=fl_ctx
            )
        assert result == expected_reserved_resources

    @pytest.mark.parametrize("gpus, gpu_mem, resource_requirement, expected_reserved_resources", TEST_CASES)
    def test_free_resource(
        self, mock_get_host_gpu_ids, gpus, gpu_mem, resource_requirement, expected_reserved_resources
    ):
        engine = MockEngine()
        gpu_resource_manager = GPUResourceManager(num_of_gpus=gpus, mem_per_gpu_in_GiB=gpu_mem)
        with engine.new_context() as fl_ctx:
            check_result, token = gpu_resource_manager.check_resources(
                resource_requirement=resource_requirement, fl_ctx=fl_ctx
            )
        assert expected_reserved_resources == gpu_resource_manager.reserved_resources[token][0]
        with engine.new_context() as fl_ctx:
            result = gpu_resource_manager.allocate_resources(
                resource_requirement=resource_requirement, token=token, fl_ctx=fl_ctx
            )
        with engine.new_context() as fl_ctx:
            gpu_resource_manager.free_resources(resources=result, token=token, fl_ctx=fl_ctx)
        assert gpu_resource_manager.reserved_resources == {}

    def test_check_four_allocate_four(self, mock_get_host_gpu_ids):
        engine = MockEngine()
        gpu_resource_manager = GPUResourceManager(num_of_gpus=4, mem_per_gpu_in_GiB=16)

        with engine.new_context() as fl_ctx:
            check1, token1 = gpu_resource_manager.check_resources(
                resource_requirement=_gen_requirement(1, 8), fl_ctx=fl_ctx
            )
            check2, token2 = gpu_resource_manager.check_resources(
                resource_requirement=_gen_requirement(1, 8), fl_ctx=fl_ctx
            )
            check3, token3 = gpu_resource_manager.check_resources(
                resource_requirement=_gen_requirement(1, 12), fl_ctx=fl_ctx
            )
            check4, token4 = gpu_resource_manager.check_resources(
                resource_requirement=_gen_requirement(1, 12), fl_ctx=fl_ctx
            )

        with engine.new_context() as fl_ctx:
            result = gpu_resource_manager.allocate_resources(
                resource_requirement=_gen_requirement(1, 8), token=token2, fl_ctx=fl_ctx
            )
        assert result == {0: 8}

        with engine.new_context() as fl_ctx:
            result = gpu_resource_manager.allocate_resources(
                resource_requirement=_gen_requirement(1, 8), token=token1, fl_ctx=fl_ctx
            )
        assert result == {0: 8}

        with engine.new_context() as fl_ctx:
            result = gpu_resource_manager.allocate_resources(
                resource_requirement=_gen_requirement(1, 12), token=token3, fl_ctx=fl_ctx
            )
        assert result == {1: 12}

        with engine.new_context() as fl_ctx:
            result = gpu_resource_manager.allocate_resources(
                resource_requirement=_gen_requirement(1, 12), token=token4, fl_ctx=fl_ctx
            )
        assert result == {2: 12}

    def test_check_one_cancel_one_check_four_then_allocate_four(self):
        engine = MockEngine()
        gpu_resource_manager = GPUResourceManager(num_of_gpus=4, mem_per_gpu_in_GiB=16)
        resource_requirement1 = _gen_requirement(1, 8)
        resource_requirement2 = _gen_requirement(4, 8)

        with engine.new_context() as fl_ctx:
            check1, token1 = gpu_resource_manager.check_resources(
                resource_requirement=resource_requirement1, fl_ctx=fl_ctx
            )

        with engine.new_context() as fl_ctx:
            gpu_resource_manager.cancel_resources(
                resource_requirement=resource_requirement1, token=token1, fl_ctx=fl_ctx
            )

        with engine.new_context() as fl_ctx:
            check2, token2 = gpu_resource_manager.check_resources(
                resource_requirement=resource_requirement2, fl_ctx=fl_ctx
            )
            result = gpu_resource_manager.allocate_resources(
                resource_requirement=resource_requirement2, token=token2, fl_ctx=fl_ctx
            )
        assert result == {0: 8, 1: 8, 2: 8, 3: 8}

    def test_check_and_timeout(self):
        timeout = 5
        engine = MockEngine()
        gpu_resource_manager = GPUResourceManager(num_of_gpus=4, mem_per_gpu_in_GiB=16, expiration_period=timeout)
        resource_requirement = _gen_requirement(1, 8)

        with engine.new_context() as fl_ctx:
            gpu_resource_manager.handle_event(event_type=EventType.SYSTEM_START, fl_ctx=fl_ctx)
            check_result, token = gpu_resource_manager.check_resources(
                resource_requirement=resource_requirement, fl_ctx=fl_ctx
            )
            assert {0: 8} == gpu_resource_manager.reserved_resources[token][0]
        time.sleep(timeout + 1)
        with engine.new_context() as fl_ctx:
            gpu_resource_manager.handle_event(event_type=EventType.SYSTEM_END, fl_ctx=fl_ctx)
        assert gpu_resource_manager.reserved_resources == {}
        for r, v in gpu_resource_manager.resources.items():
            assert v.memory == 16
