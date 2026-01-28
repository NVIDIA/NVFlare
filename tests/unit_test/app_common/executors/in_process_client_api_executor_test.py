# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Tests for InProcessClientAPIExecutor memory management parameters."""

import pytest

from nvflare.app_common.executors.in_process_client_api_executor import InProcessClientAPIExecutor
from nvflare.client.config import ExchangeFormat, TransferType


class TestInProcessClientAPIExecutorMemory:
    """Test memory management parameters in InProcessClientAPIExecutor."""

    @pytest.fixture
    def base_executor_params(self):
        """Base parameters for creating executor instances."""
        return {
            "task_script_path": "train.py",
            "task_script_args": "--epochs 10",
            "params_exchange_format": ExchangeFormat.NUMPY,
            "params_transfer_type": TransferType.FULL,
        }

    def test_default_memory_parameters(self, base_executor_params):
        """Test that memory management parameters default to disabled."""
        executor = InProcessClientAPIExecutor(**base_executor_params)

        assert executor._memory_gc_rounds == 0
        assert executor._torch_cuda_empty_cache is False

    def test_memory_parameters_enabled(self, base_executor_params):
        """Test memory parameters can be enabled."""
        executor = InProcessClientAPIExecutor(
            memory_gc_rounds=5,
            torch_cuda_empty_cache=True,
            **base_executor_params,
        )

        assert executor._memory_gc_rounds == 5
        assert executor._torch_cuda_empty_cache is True

    @pytest.mark.parametrize(
        "gc_rounds,cuda_empty",
        [
            (0, False),  # Disabled
            (1, True),  # Every round with CUDA
            (1, False),  # Every round without CUDA
            (5, True),  # Every 5 rounds with CUDA
            (10, False),  # Every 10 rounds without CUDA
        ],
    )
    def test_memory_parameter_combinations(self, base_executor_params, gc_rounds, cuda_empty):
        """Test various memory parameter combinations."""
        executor = InProcessClientAPIExecutor(
            memory_gc_rounds=gc_rounds,
            torch_cuda_empty_cache=cuda_empty,
            **base_executor_params,
        )

        assert executor._memory_gc_rounds == gc_rounds
        assert executor._torch_cuda_empty_cache == cuda_empty

    def test_memory_parameters_with_other_options(self, base_executor_params):
        """Test that memory parameters work with other executor options."""
        executor = InProcessClientAPIExecutor(
            task_wait_time=30.0,
            result_pull_interval=1.0,
            train_with_evaluation=True,
            memory_gc_rounds=2,
            torch_cuda_empty_cache=True,
            **base_executor_params,
        )

        assert executor._memory_gc_rounds == 2
        assert executor._torch_cuda_empty_cache is True
        assert executor._task_wait_time == 30.0
        assert executor._result_pull_interval == 1.0
        assert executor._train_with_evaluation is True
