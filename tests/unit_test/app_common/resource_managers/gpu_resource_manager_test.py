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


@pytest.fixture(autouse=True)
def clear_cuda_visible_devices(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)


@pytest.fixture(scope="class", autouse=True)
def mock_get_host_gpu_ids():
    with (
        patch("nvflare.app_common.resource_managers.gpu_resource_manager.get_host_gpu_ids") as ids_fixture,
        patch("nvflare.app_common.resource_managers.gpu_resource_manager.get_host_gpu_memory_total") as mem_fixture,
    ):
        ids_fixture.return_value = [0, 1, 2, 3]
        mem_fixture.return_value = [32768, 32768, 32768, 32768]
        yield ids_fixture


class TestGPUResourceManager:
    def test_init_respects_cuda_visible_devices_for_memory_check(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
        with (
            patch("nvflare.app_common.resource_managers.gpu_resource_manager.get_host_gpu_ids") as mock_ids,
            patch("nvflare.app_common.resource_managers.gpu_resource_manager.get_host_gpu_memory_total") as mock_mem,
        ):
            mock_ids.return_value = [0, 1, 2, 3, 4]
            mock_mem.return_value = [81920, 81920, 81920, 4096, 81920]

            gpu_resource_manager = GPUResourceManager(num_of_gpus=1, mem_per_gpu_in_GiB=16)

        assert list(gpu_resource_manager.resources) == [0]

    def test_init_uses_visible_gpu_ids_as_managed_resources(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
        with (
            patch("nvflare.app_common.resource_managers.gpu_resource_manager.get_host_gpu_ids") as mock_ids,
            patch("nvflare.app_common.resource_managers.gpu_resource_manager.get_host_gpu_memory_total") as mock_mem,
        ):
            mock_ids.return_value = [0, 1, 2, 3, 4]
            mock_mem.return_value = [81920, 81920, 81920, 4096, 81920]

            gpu_resource_manager = GPUResourceManager(num_of_gpus=1, mem_per_gpu_in_GiB=16)

        assert list(gpu_resource_manager.resources) == [4]

    def test_init_checks_selected_visible_gpu_memory(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
        with (
            patch("nvflare.app_common.resource_managers.gpu_resource_manager.get_host_gpu_ids") as mock_ids,
            patch("nvflare.app_common.resource_managers.gpu_resource_manager.get_host_gpu_memory_total") as mock_mem,
        ):
            mock_ids.return_value = [0, 1, 2, 3, 4]
            mock_mem.return_value = [81920, 81920, 81920, 4096, 81920]

            with pytest.raises(ValueError, match="4096"):
                GPUResourceManager(num_of_gpus=1, mem_per_gpu_in_GiB=16)

    def test_init_reports_selected_gpu_id_for_insufficient_memory(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
        with (
            patch("nvflare.app_common.resource_managers.gpu_resource_manager.get_host_gpu_ids") as mock_ids,
            patch("nvflare.app_common.resource_managers.gpu_resource_manager.get_host_gpu_memory_total") as mock_mem,
        ):
            mock_ids.return_value = [0, 1, 2, 3, 4]
            mock_mem.return_value = [81920, 81920, 81920, 4096, 81920]

            with pytest.raises(ValueError, match="GPU ID 3"):
                GPUResourceManager(num_of_gpus=1, mem_per_gpu_in_GiB=16)

    def test_init_raises_when_selected_gpu_memory_is_missing(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
        with (
            patch("nvflare.app_common.resource_managers.gpu_resource_manager.get_host_gpu_ids") as mock_ids,
            patch("nvflare.app_common.resource_managers.gpu_resource_manager.get_host_gpu_memory_total") as mock_mem,
        ):
            mock_ids.return_value = [0, 1, 2, 3]
            mock_mem.return_value = [32768, 32768, 32768]

            with pytest.raises(RuntimeError, match="GPU ID 3"):
                GPUResourceManager(num_of_gpus=1, mem_per_gpu_in_GiB=16)

    def test_init_checks_visible_gpu_count(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
        with (
            patch("nvflare.app_common.resource_managers.gpu_resource_manager.get_host_gpu_ids") as mock_ids,
            patch("nvflare.app_common.resource_managers.gpu_resource_manager.get_host_gpu_memory_total") as mock_mem,
        ):
            mock_ids.return_value = [0, 1, 2, 3]
            mock_mem.return_value = [32768, 32768, 32768, 32768]

            with pytest.raises(ValueError, match="exceeds available GPUs: 1"):
                GPUResourceManager(num_of_gpus=2, mem_per_gpu_in_GiB=16)

    def test_init_stops_at_invalid_cuda_visible_device_index(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,2,-1,1")
        with (
            patch("nvflare.app_common.resource_managers.gpu_resource_manager.get_host_gpu_ids") as mock_ids,
            patch("nvflare.app_common.resource_managers.gpu_resource_manager.get_host_gpu_memory_total") as mock_mem,
        ):
            mock_ids.return_value = [0, 1, 2, 3]
            mock_mem.return_value = [32768, 32768, 32768, 32768]

            gpu_resource_manager = GPUResourceManager(num_of_gpus=2, mem_per_gpu_in_GiB=16)

        assert list(gpu_resource_manager.resources) == [0, 2]

    def test_init_empty_cuda_visible_devices_has_no_available_gpus(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
        with (
            patch("nvflare.app_common.resource_managers.gpu_resource_manager.get_host_gpu_ids") as mock_ids,
            patch("nvflare.app_common.resource_managers.gpu_resource_manager.get_host_gpu_memory_total") as mock_mem,
        ):
            mock_ids.return_value = [0, 1, 2, 3]
            mock_mem.return_value = [32768, 32768, 32768, 32768]

            with pytest.raises(ValueError, match="exceeds available GPUs: 0"):
                GPUResourceManager(num_of_gpus=1, mem_per_gpu_in_GiB=16)

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
        engine = MockEngine()
        gpu_resource_manager = GPUResourceManager(num_of_gpus=4, mem_per_gpu_in_GiB=16, expiration_period=1)
        resource_requirement = _gen_requirement(1, 8)

        with engine.new_context() as fl_ctx:
            check_result, token = gpu_resource_manager.check_resources(
                resource_requirement=resource_requirement, fl_ctx=fl_ctx
            )
            assert {0: 8} == gpu_resource_manager.reserved_resources[token][0]

        with patch(
            "nvflare.app_common.resource_managers.auto_clean_resource_manager.time.sleep",
            side_effect=lambda _: gpu_resource_manager._stop_event.set(),
        ):
            gpu_resource_manager._check_expired()

        assert gpu_resource_manager.reserved_resources == {}
        for r, v in gpu_resource_manager.resources.items():
            assert v.memory == 16
