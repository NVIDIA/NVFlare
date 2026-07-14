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

"""Build-time validation in ClientAppConfig.add_executor.

V1 supports one ClientAPIExecutor per execution mode per client job (the runtime
SingleBackendGuard enforces the same at START_RUN); the build-time check here fails the
misconfiguration while the config is still in the user's hands."""

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


class TestOneClientAPIExecutorPerMode:
    def test_second_same_mode_executor_is_rejected(self):
        config = ClientAppConfig()
        config.add_executor(["train"], _in_process_executor())
        with pytest.raises(RuntimeError, match="one ClientAPIExecutor per execution mode"):
            config.add_executor(["evaluate"], _in_process_executor())

    def test_different_modes_may_coexist(self):
        config = ClientAppConfig()
        config.add_executor(["train"], _in_process_executor())
        config.add_executor(["evaluate"], _external_executor())
        assert len(config.executors) == 2

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
