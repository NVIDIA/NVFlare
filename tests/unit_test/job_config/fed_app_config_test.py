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

"""Generic executor registration in ClientAppConfig.add_executor.

Reusing one executor object accumulates task routes in one exported definition. Executor-
specific runtime constraints, including ClientAPIExecutor multiplicity, are validated by
the runtime configurator."""

import pytest

from nvflare.apis.executor import Executor
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.executors.client_api_executor import ClientAPIExecutor
from nvflare.job_config.fed_app_config import ClientAppConfig


class _PlainExecutor(Executor):
    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        return Shareable()


def _in_process_executor():
    return ClientAPIExecutor(execution_mode="in_process", task_script_path="train.py")


def _external_executor():
    return ClientAPIExecutor(execution_mode="external_process", command="python custom/train.py")


class TestExecutorRegistration:
    def test_distinct_client_api_executors_are_registered_generically(self):
        config = ClientAppConfig()
        config.add_executor(["train"], _in_process_executor())
        config.add_executor(["evaluate"], _external_executor())

        assert len(config.executors) == 2

    def test_one_executor_can_route_multiple_tasks(self):
        config = ClientAppConfig()
        config.add_executor(["train", "evaluate", "submit_model"], _external_executor())
        assert len(config.executors) == 1
        assert config.executors[0].tasks == ["train", "evaluate", "submit_model"]

    def test_same_executor_instance_merges_additional_tasks(self):
        config = ClientAppConfig()
        executor = _external_executor()

        config.add_executor(["train"], executor)
        config.add_executor(["evaluate", "submit_model"], executor)

        assert len(config.executors) == 1
        assert config.executors[0].executor is executor
        assert config.executors[0].tasks == ["train", "evaluate", "submit_model"]

    def test_plain_executors_are_not_restricted(self):
        config = ClientAppConfig()
        config.add_executor(["train"], _PlainExecutor())
        config.add_executor(["evaluate"], _PlainExecutor())
        config.add_executor(["submit_model"], _in_process_executor())
        assert len(config.executors) == 3

    def test_duplicate_task_check_still_first(self):
        config = ClientAppConfig()
        config.add_executor(["train"], _in_process_executor())
        with pytest.raises(RuntimeError, match="already exist"):
            config.add_executor(["train"], _PlainExecutor())
