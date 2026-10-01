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

from unittest.mock import Mock

import pytest

from nvflare.apis.shareable import Shareable
from nvflare.app_common.executors.client_api_executor import ClientAPIExecutor
from nvflare.private.fed.task_worker.client_api import TaskClientAPI, execute_client_api_task
from nvflare.private.fed.task_worker.protocol import TaskAttemptIdentity


def test_task_client_api_requires_exactly_one_durable_send():
    store = Mock()
    identity = TaskAttemptIdentity("job", "site", "task", "train", "attempt")
    api = TaskClientAPI({}, store, identity)
    with pytest.raises(RuntimeError, match="without sending"):
        api.get_result()
    api._publish_result(Shareable())
    assert api.get_result().reference is store.stage_result.return_value
    assert not api.is_running()
    assert api.receive() is None
    with pytest.raises(RuntimeError, match="exactly one"):
        api._publish_result(Shareable())
    store.stage_result.assert_called_once()
    api.close()


def test_task_adapter_rejects_custom_client_api_subclass():
    class CustomExecutor(ClientAPIExecutor):
        pass

    executor = CustomExecutor(execution_mode="in_process", task_script_path="train.py")
    with pytest.raises(RuntimeError, match="subclasses"):
        execute_client_api_task(executor, None, None, None, None)


def test_task_adapter_rejects_external_process_mode():
    executor = ClientAPIExecutor(execution_mode="external_process", command=["python", "train.py"])
    with pytest.raises(RuntimeError, match="in_process scripts only"):
        execute_client_api_task(executor, None, None, None, None)
