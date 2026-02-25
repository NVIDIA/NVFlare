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

from types import SimpleNamespace

from nvflare.apis.shareable import ReservedHeaderKey, Shareable
from nvflare.private.fed.client import client_runner as client_runner_module
from nvflare.private.fed.client.client_runner import ClientRunner


def _make_runner():
    runner = ClientRunner.__new__(ClientRunner)
    runner.run_abort_signal = SimpleNamespace(triggered=False)
    runner.task_check_interval = 0.0
    runner.log_debug = lambda *_args, **_kwargs: None
    runner.log_info = lambda *_args, **_kwargs: None
    runner.log_error = lambda *_args, **_kwargs: None
    return runner


def test_send_task_result_reuses_msg_root_id_and_deletes_once(monkeypatch):
    runner = _make_runner()
    observed_ids = []
    results = [
        client_runner_module._TASK_CHECK_RESULT_TRY_AGAIN,
        client_runner_module._TASK_CHECK_RESULT_OK,
    ]

    def _fake_try_send(result, task_id, fl_ctx):
        _ = task_id
        _ = fl_ctx
        observed_ids.append(result.get_header(ReservedHeaderKey.MSG_ROOT_ID))
        return results.pop(0)

    runner._wait_task_ready_and_send_once = _fake_try_send

    deleted_ids = []
    monkeypatch.setattr(client_runner_module, "delete_msg_root", lambda msg_root_id: deleted_ids.append(msg_root_id))

    result = Shareable()
    ok = runner._send_task_result(result=result, task_id="task-1", fl_ctx=None)

    assert ok is True
    assert len(observed_ids) == 2
    assert observed_ids[0] == observed_ids[1]
    assert deleted_ids == [observed_ids[0]]


def test_send_task_result_cleans_msg_root_when_task_gone(monkeypatch):
    runner = _make_runner()
    observed_ids = []
    results = [
        client_runner_module._TASK_CHECK_RESULT_TRY_AGAIN,
        client_runner_module._TASK_CHECK_RESULT_TASK_GONE,
    ]

    def _fake_try_send(result, task_id, fl_ctx):
        _ = task_id
        _ = fl_ctx
        observed_ids.append(result.get_header(ReservedHeaderKey.MSG_ROOT_ID))
        return results.pop(0)

    runner._wait_task_ready_and_send_once = _fake_try_send

    deleted_ids = []
    monkeypatch.setattr(client_runner_module, "delete_msg_root", lambda msg_root_id: deleted_ids.append(msg_root_id))

    result = Shareable()
    ok = runner._send_task_result(result=result, task_id="task-2", fl_ctx=None)

    assert ok is False
    assert len(observed_ids) == 2
    assert observed_ids[0] == observed_ids[1]
    assert deleted_ids == [observed_ids[0]]
