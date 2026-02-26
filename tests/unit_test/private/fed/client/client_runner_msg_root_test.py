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
    runner.submit_task_result_timeout = None
    runner.parent_target = "server"
    runner.log_debug = lambda *_args, **_kwargs: None
    runner.log_info = lambda *_args, **_kwargs: None
    runner.log_error = lambda *_args, **_kwargs: None
    return runner


def test_send_task_result_reuses_msg_root_id_and_does_not_delete_on_retry_success(monkeypatch):
    runner = _make_runner()
    observed_ids = []
    results = [
        client_runner_module._TASK_CHECK_RESULT_TRY_AGAIN,
        client_runner_module._TASK_CHECK_RESULT_OK,
    ]

    def _fake_try_send(result, task_id, fl_ctx, submit_timeout):
        _ = task_id
        _ = fl_ctx
        _ = submit_timeout
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
    assert deleted_ids == []


def test_send_task_result_success_does_not_delete_msg_root(monkeypatch):
    runner = _make_runner()
    observed_ids = []

    def _fake_try_send(result, task_id, fl_ctx, submit_timeout):
        _ = task_id
        _ = fl_ctx
        _ = submit_timeout
        observed_ids.append(result.get_header(ReservedHeaderKey.MSG_ROOT_ID))
        return client_runner_module._TASK_CHECK_RESULT_OK

    runner._wait_task_ready_and_send_once = _fake_try_send

    deleted_ids = []
    monkeypatch.setattr(client_runner_module, "delete_msg_root", lambda msg_root_id: deleted_ids.append(msg_root_id))

    result = Shareable()
    ok = runner._send_task_result(result=result, task_id="task-0", fl_ctx=None)

    assert ok is True
    assert len(observed_ids) == 1
    assert deleted_ids == []


def test_send_task_result_retry_then_task_gone_keeps_msg_root_alive(monkeypatch):
    runner = _make_runner()
    observed_ids = []
    results = [
        client_runner_module._TASK_CHECK_RESULT_TRY_AGAIN,
        client_runner_module._TASK_CHECK_RESULT_TASK_GONE,
    ]

    def _fake_try_send(result, task_id, fl_ctx, submit_timeout):
        _ = task_id
        _ = fl_ctx
        _ = submit_timeout
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
    assert deleted_ids == []


def test_send_task_result_task_gone_without_retry_cleans_msg_root(monkeypatch):
    runner = _make_runner()
    observed_ids = []

    def _fake_try_send(result, task_id, fl_ctx, submit_timeout):
        _ = task_id
        _ = fl_ctx
        _ = submit_timeout
        observed_ids.append(result.get_header(ReservedHeaderKey.MSG_ROOT_ID))
        return client_runner_module._TASK_CHECK_RESULT_TASK_GONE

    runner._wait_task_ready_and_send_once = _fake_try_send

    deleted_ids = []
    monkeypatch.setattr(client_runner_module, "delete_msg_root", lambda msg_root_id: deleted_ids.append(msg_root_id))

    result = Shareable()
    ok = runner._send_task_result(result=result, task_id="task-3", fl_ctx=None)

    assert ok is False
    assert len(observed_ids) == 1
    assert deleted_ids == [observed_ids[0]]


def test_send_task_result_uses_3_tries_with_doubling_timeout(monkeypatch):
    runner = _make_runner()
    submit_timeouts = []

    def _fake_try_send(result, task_id, fl_ctx, submit_timeout):
        _ = result
        _ = task_id
        _ = fl_ctx
        submit_timeouts.append(submit_timeout)
        return client_runner_module._TASK_CHECK_RESULT_TRY_AGAIN

    runner._wait_task_ready_and_send_once = _fake_try_send

    deleted_ids = []
    monkeypatch.setattr(client_runner_module, "delete_msg_root", lambda msg_root_id: deleted_ids.append(msg_root_id))

    result = Shareable()
    ok = runner._send_task_result(result=result, task_id="task-4", fl_ctx=None)

    assert ok is False
    assert submit_timeouts == [300.0, 600.0, 1200.0]
    assert deleted_ids == []
