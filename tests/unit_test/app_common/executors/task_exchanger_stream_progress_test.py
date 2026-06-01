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

import threading
from unittest.mock import MagicMock

from nvflare.apis.fl_constant import FLContextKey, FLMetaKey
from nvflare.apis.shareable import ReturnCode, Shareable
from nvflare.app_common.executors import task_exchanger as task_exchanger_module
from nvflare.app_common.executors.task_exchanger import TaskExchanger
from nvflare.fuel.f3.cellnet import cell as cell_module
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.cellnet.defs import ReturnCode as CellReturnCode
from nvflare.fuel.f3.cellnet.utils import make_reply
from nvflare.fuel.f3.message import Message as CellMessage
from nvflare.fuel.f3.streaming import download_service as download_service_module
from nvflare.fuel.f3.streaming import transfer_progress as transfer_progress_module
from nvflare.fuel.f3.streaming.download_service import Consumer, ProduceRC, _PropKey, download_object
from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.constants import Mode
from nvflare.fuel.utils.fobs.decomposers.via_downloader import ViaDownloaderDecomposer
from nvflare.fuel.utils.pipe.cell_pipe import CellPipe
from nvflare.fuel.utils.pipe.pipe import Message, Pipe, Topic
from nvflare.fuel.utils.waiter_utils import WaiterRC

DIRECTION_TASK_PAYLOAD_DOWNLOAD = task_exchanger_module.DIRECTION_TASK_PAYLOAD_DOWNLOAD


class _AbortSignal:
    triggered = False

    def trigger(self, reason=None):
        self.triggered = True


class _FakePipeHandler:
    def __init__(self, send_cb):
        self.send_cb = send_cb
        self.asked_to_stop = False
        self.replies = []
        self.send_calls = 0
        self.aborted = []

    def send_to_peer(self, msg, timeout=None, abort_signal=None):
        self.send_calls += 1
        return self.send_cb(self, msg, timeout, abort_signal)

    def get_next(self):
        if self.replies:
            return self.replies.pop(0)
        return None

    def notify_abort(self, task_id):
        self.aborted.append(task_id)

    def stop(self, close_pipe=True):
        self.asked_to_stop = True


class _DummyPipe(Pipe):
    def __init__(self):
        super().__init__(Mode.ACTIVE)

    def open(self, name: str):
        pass

    def clear(self):
        pass

    def send(self, msg: Message, timeout=None) -> bool:
        return True

    def receive(self, timeout=None):
        return None

    def close(self):
        pass

    def can_resend(self) -> bool:
        return False


class _ReadyStreamFuture:
    def __init__(self, headers=None, payload=None):
        self.headers = headers or {}
        self._payload = payload
        self.error = False
        self.waiter = threading.Event()
        self.waiter.set()

    def get_progress(self):
        return 1

    def result(self):
        return self._payload


class _CellStackFake:
    """Fake only the transport boundary while reusing Cell._send_request/_send_one_request."""

    def __init__(self):
        self.requests_dict = {}
        self.decode_pass_through_channels = set()
        self.sent_blobs = []
        self.logger = MagicMock()

    def get_fobs_context(self, props=None):
        ctx = {}
        if props:
            ctx.update(props)
        return ctx

    def send_blob(self, channel, topic, target, message, secure=False, optional=False):
        self.sent_blobs.append((channel, topic, target, message))
        return _ReadyStreamFuture()

    def send_request(self, *args, **kwargs):
        return cell_module.Cell._send_request(self, *args, **kwargs)

    def _encode_message(self, *args, **kwargs):
        return cell_module.Cell._encode_message(self, *args, **kwargs)

    def _send_one_request(self, *args, **kwargs):
        return cell_module.Cell._send_one_request(self, *args, **kwargs)

    def _get_result(self, *args, **kwargs):
        return cell_module.Cell._get_result(self, *args, **kwargs)

    def _future_wait(self, future, timeout, abort_signal):
        return True


def _make_cell_stack_pipe(cell):
    pipe = CellPipe.__new__(CellPipe)
    Pipe.__init__(pipe, Mode.ACTIVE)
    pipe.cell = cell
    pipe.channel = "cell_pipe.task"
    pipe.peer_fqcn = "site-1.site-1_job_active"
    pipe.hb_seq = 1
    pipe.pass_through_on_send = False
    pipe.logger = MagicMock()
    pipe.pipe_lock = threading.Lock()
    pipe.closed = False
    return pipe


def _make_fl_ctx(job_id="job-1"):
    fl_ctx = MagicMock()
    fl_ctx.get_job_id.return_value = job_id
    fl_ctx.get_identity_name.return_value = "site-1"
    return fl_ctx


def _make_task(task_id="task-1"):
    shareable = Shareable()
    shareable.set_header(FLContextKey.TASK_ID, task_id)
    return shareable


def _patch_logs(monkeypatch):
    logs = []

    monkeypatch.setattr(TaskExchanger, "log_info", lambda self, fl_ctx, msg: logs.append(("info", msg)))
    monkeypatch.setattr(TaskExchanger, "log_debug", lambda self, fl_ctx, msg: logs.append(("debug", msg)))
    monkeypatch.setattr(TaskExchanger, "log_warning", lambda self, fl_ctx, msg: logs.append(("warning", msg)))
    monkeypatch.setattr(TaskExchanger, "log_error", lambda self, fl_ctx, msg: logs.append(("error", msg)))
    return logs


def _reply_for(req):
    result = Shareable()
    result.set_return_code(ReturnCode.OK)
    return Message.new_reply(topic=req.topic, data=result, req_msg_id=req.msg_id)


def _progress(
    task_id,
    transfer_id="transfer-1",
    sequence=1,
    bytes_done=1024,
    state="active",
    job_id="job-1",
    items_done=None,
    transfer_id_kind=None,
    receiver_id=None,
    direction=DIRECTION_TASK_PAYLOAD_DOWNLOAD,
):
    data = {
        "job_id": job_id,
        "task_id": task_id,
        "transfer_id": transfer_id,
        "direction": direction,
        "sequence": sequence,
        "bytes_done": bytes_done,
        "state": state,
    }
    if items_done is not None:
        data["items_done"] = items_done
    if transfer_id_kind is not None:
        data["transfer_id_kind"] = transfer_id_kind
    if receiver_id is not None:
        data["receiver_id"] = receiver_id
    return Message.new_request(
        Topic.STREAM_PROGRESS,
        data,
    )


def test_task_send_no_progress_waits_beyond_peer_read_timeout_and_fails_at_streaming_idle_timeout(monkeypatch):
    logs = _patch_logs(monkeypatch)
    now = [1000.0]
    monkeypatch.setattr(task_exchanger_module.time, "time", lambda: now[0])
    executor = TaskExchanger(
        pipe_id="pipe",
        peer_read_timeout=1.0,
        streaming_idle_timeout=3.0,
        result_poll_interval=0.01,
    )

    def send_cb(handler, msg, timeout, abort_signal):
        now[0] += timeout
        assert msg._progress_wait_cb() is True
        now[0] += timeout
        assert msg._progress_wait_cb() is True
        now[0] += timeout
        assert msg._progress_wait_cb() is False
        return False

    handler = _FakePipeHandler(send_cb)
    executor.pipe_handler = handler

    result = executor._do_execute("train", _make_task(), _make_fl_ctx(), _AbortSignal())

    assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION
    assert handler.send_calls == 1
    assert any("no stream progress record exists yet" in msg for _, msg in logs)


def test_task_send_no_progress_startup_budget_uses_streaming_idle_timeout(monkeypatch):
    _patch_logs(monkeypatch)
    now = [1000.0]
    monkeypatch.setattr(task_exchanger_module.time, "time", lambda: now[0])
    executor = TaskExchanger(
        pipe_id="pipe",
        peer_read_timeout=100.0,
        streaming_idle_timeout=3.0,
        result_poll_interval=0.01,
    )

    def send_cb(handler, msg, timeout, abort_signal):
        assert timeout == 3.0
        now[0] += 2.9
        assert msg._progress_wait_cb() is True
        now[0] += 0.2
        assert msg._progress_wait_cb() is False
        return False

    handler = _FakePipeHandler(send_cb)
    executor.pipe_handler = handler

    result = executor._do_execute("train", _make_task(), _make_fl_ctx(), _AbortSignal())

    assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION
    assert handler.send_calls == 1


def test_task_send_no_progress_startup_budget_uses_peer_read_timeout_floor(monkeypatch):
    logs = _patch_logs(monkeypatch)
    now = [1000.0]
    monkeypatch.setattr(task_exchanger_module.time, "time", lambda: now[0])
    executor = TaskExchanger(
        pipe_id="pipe",
        peer_read_timeout=60.0,
        streaming_idle_timeout=600.0,
        result_poll_interval=0.01,
    )

    def send_cb(handler, msg, timeout, abort_signal):
        assert timeout == task_exchanger_module.STREAM_PROGRESS_COMPLETION_ACK_GRACE
        now[0] += timeout
        assert msg._progress_wait_cb() is True
        now[0] += 29.9
        assert msg._progress_wait_cb() is True
        now[0] += 0.2
        assert msg._progress_wait_cb() is False
        return False

    handler = _FakePipeHandler(send_cb)
    executor.pipe_handler = handler

    result = executor._do_execute("train", _make_task(), _make_fl_ctx(), _AbortSignal())

    assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION
    assert handler.send_calls == 1
    assert any("task_send_wait_budget=60.0s" in msg for _, msg in logs)


def test_task_send_no_progress_budget_uses_idle_timeout_when_peer_read_timeout_disabled():
    executor = TaskExchanger(pipe_id="pipe", peer_read_timeout=None, streaming_idle_timeout=600.0)

    assert TaskExchanger._get_task_send_no_progress_budget(600.0, None) == 600.0
    assert executor._get_task_send_peer_read_timeout() == task_exchanger_module.STREAM_PROGRESS_COMPLETION_ACK_GRACE


def test_task_send_peer_read_timeout_disabled_still_aborts_failed_effective_send(monkeypatch):
    _patch_logs(monkeypatch)
    executor = TaskExchanger(
        pipe_id="pipe",
        peer_read_timeout=None,
        streaming_idle_timeout=600.0,
        result_poll_interval=0.01,
    )

    def send_cb(handler, msg, timeout, abort_signal):
        assert timeout == task_exchanger_module.STREAM_PROGRESS_COMPLETION_ACK_GRACE
        return False

    handler = _FakePipeHandler(send_cb)
    executor.pipe_handler = handler

    result = executor._do_execute("train", _make_task(), _make_fl_ctx(), _AbortSignal())

    assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION
    assert handler.send_calls == 1


def test_explicit_high_peer_read_timeout_logs_startup_budget_clamp(monkeypatch):
    logs = _patch_logs(monkeypatch)
    now = [1000.0]
    monkeypatch.setattr(task_exchanger_module.time, "time", lambda: now[0])
    executor = TaskExchanger(
        pipe_id="pipe",
        peer_read_timeout=100.0,
        peer_read_timeout_explicit=True,
        streaming_idle_timeout=3.0,
        result_poll_interval=0.01,
    )

    def send_cb(handler, msg, timeout, abort_signal):
        assert timeout == 3.0
        return False

    handler = _FakePipeHandler(send_cb)
    executor.pipe_handler = handler

    result = executor._do_execute("train", _make_task(), _make_fl_ctx(), _AbortSignal())

    assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION
    assert any("using streaming_idle_timeout as the no-progress task-send startup budget" in msg for _, msg in logs)


def test_task_send_continues_after_peer_read_timeout_when_stream_progress_is_recent(monkeypatch):
    logs = _patch_logs(monkeypatch)
    executor = TaskExchanger(pipe_id="pipe", peer_read_timeout=0.01, streaming_idle_timeout=10.0)

    def send_cb(handler, msg, timeout, abort_signal):
        executor._handle_stream_progress_message(_progress(task_id=msg.msg_id))
        assert msg._progress_wait_cb() is True
        handler.replies.append(_reply_for(msg))
        return True

    handler = _FakePipeHandler(send_cb)
    executor.pipe_handler = handler

    result = executor._do_execute("train", _make_task(), _make_fl_ctx(), _AbortSignal())

    assert result.get_return_code() == ReturnCode.OK
    assert handler.send_calls == 1
    assert any("continuing to wait" in msg for _, msg in logs)


def test_task_send_progress_after_peer_read_timeout_intervals_suppresses_resend_and_succeeds(monkeypatch):
    logs = _patch_logs(monkeypatch)
    now = [1000.0]
    monkeypatch.setattr(task_exchanger_module.time, "time", lambda: now[0])
    executor = TaskExchanger(
        pipe_id="pipe",
        peer_read_timeout=1.0,
        streaming_idle_timeout=5.0,
        result_poll_interval=0.01,
    )

    def send_cb(handler, msg, timeout, abort_signal):
        now[0] += timeout
        assert msg._progress_wait_cb() is True
        now[0] += timeout
        assert msg._progress_wait_cb() is True
        executor._handle_stream_progress_message(_progress(task_id=msg.msg_id, sequence=1, bytes_done=1024))
        now[0] += timeout
        assert msg._progress_wait_cb() is True
        handler.replies.append(_reply_for(msg))
        return True

    handler = _FakePipeHandler(send_cb)
    executor.pipe_handler = handler

    result = executor._do_execute("train", _make_task(), _make_fl_ctx(), _AbortSignal())

    assert result.get_return_code() == ReturnCode.OK
    assert handler.send_calls == 1
    assert any("stream transfer" in msg for _, msg in logs)
    assert any("after 3.00 secs" in msg for _, msg in logs)
    assert not any("after 1.0 secs, but stream transfer" in msg for _, msg in logs)


def test_task_send_completed_progress_holds_wait_briefly_for_ack(monkeypatch):
    logs = _patch_logs(monkeypatch)
    now = [1000.0]
    monkeypatch.setattr(task_exchanger_module.time, "time", lambda: now[0])
    executor = TaskExchanger(
        pipe_id="pipe",
        peer_read_timeout=1.0,
        streaming_idle_timeout=10.0,
        result_poll_interval=0.01,
    )

    def send_cb(handler, msg, timeout, abort_signal):
        executor._handle_stream_progress_message(
            _progress(task_id=msg.msg_id, sequence=1, bytes_done=1024, state="completed")
        )
        assert msg._progress_wait_cb() is True
        now[0] += task_exchanger_module.STREAM_PROGRESS_COMPLETION_ACK_GRACE + 0.1
        assert msg._progress_wait_cb() is False
        return False

    handler = _FakePipeHandler(send_cb)
    executor.pipe_handler = handler

    result = executor._do_execute("train", _make_task(), _make_fl_ctx(), _AbortSignal())

    assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION
    assert any("completed recently" in msg for _, msg in logs)


def test_task_send_completion_grace_uses_bounded_poll_interval(monkeypatch):
    _patch_logs(monkeypatch)
    now = [1000.0]
    monkeypatch.setattr(task_exchanger_module.time, "time", lambda: now[0])
    executor = TaskExchanger(
        pipe_id="pipe",
        peer_read_timeout=600.0,
        streaming_idle_timeout=600.0,
        result_poll_interval=0.01,
    )
    executor.pipe = object.__new__(CellPipe)

    def send_cb(handler, msg, timeout, abort_signal):
        assert timeout == task_exchanger_module.STREAM_PROGRESS_COMPLETION_ACK_GRACE
        executor._handle_stream_progress_message(
            _progress(task_id=msg.msg_id, sequence=1, bytes_done=1024, state="completed")
        )
        assert msg._progress_wait_cb() is True
        now[0] += timeout + 0.1
        assert msg._progress_wait_cb() is False
        return False

    handler = _FakePipeHandler(send_cb)
    executor.pipe_handler = handler

    result = executor._do_execute("train", _make_task(), _make_fl_ctx(), _AbortSignal())

    assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION


def test_non_cell_pipe_keeps_peer_read_timeout_for_task_send():
    executor = TaskExchanger(pipe_id="pipe", peer_read_timeout=600.0, streaming_idle_timeout=600.0)
    executor.pipe = _DummyPipe()

    assert executor._get_task_send_peer_read_timeout() == 600.0


def test_non_cell_pipe_keeps_disabled_peer_read_timeout_for_task_send():
    executor = TaskExchanger(pipe_id="pipe", peer_read_timeout=None, streaming_idle_timeout=600.0)
    executor.pipe = _DummyPipe()

    assert executor._get_task_send_peer_read_timeout() is None


def test_task_send_many_active_transfers_progress_past_fixed_timeout_without_resend(monkeypatch):
    """Synthetic congestion: many streamed refs advance across repeated old timeout windows."""
    _patch_logs(monkeypatch)
    now = [1000.0]
    monkeypatch.setattr(task_exchanger_module.time, "time", lambda: now[0])
    executor = TaskExchanger(
        pipe_id="pipe",
        peer_read_timeout=1.0,
        streaming_idle_timeout=5.0,
        result_poll_interval=0.01,
    )
    transfer_ids = [f"transfer-{i}" for i in range(16)]

    def send_cb(handler, msg, timeout, abort_signal):
        now[0] += timeout
        assert msg._progress_wait_cb() is True

        for sequence in range(1, 7):
            for index, transfer_id in enumerate(transfer_ids):
                executor._handle_stream_progress_message(
                    _progress(
                        task_id=msg.msg_id,
                        transfer_id=transfer_id,
                        sequence=sequence,
                        bytes_done=(sequence * 1024) + index,
                    )
                )
            now[0] += timeout
            assert msg._progress_wait_cb() is True

        handler.replies.append(_reply_for(msg))
        return True

    handler = _FakePipeHandler(send_cb)
    executor.pipe_handler = handler

    result = executor._do_execute("train", _make_task(), _make_fl_ctx(), _AbortSignal())

    assert result.get_return_code() == ReturnCode.OK
    assert handler.send_calls == 1
    records, active_records = executor._get_active_task_payload_records("task-1", "job-1")
    assert len(records) == 16
    assert active_records == []


def test_task_send_does_not_let_recent_sibling_mask_stalled_transfer(monkeypatch):
    _patch_logs(monkeypatch)
    now = [1000.0]
    monkeypatch.setattr(task_exchanger_module.time, "time", lambda: now[0])
    executor = TaskExchanger(pipe_id="pipe", peer_read_timeout=0.01, streaming_idle_timeout=10.0)
    executor._handle_stream_progress_message(
        _progress(task_id="task-1", transfer_id="stalled-transfer", sequence=1, bytes_done=1024)
    )
    now[0] += 11.0
    executor._handle_stream_progress_message(
        _progress(task_id="task-1", transfer_id="recent-transfer", sequence=1, bytes_done=2048)
    )

    def send_cb(handler, msg, timeout, abort_signal):
        assert msg._progress_wait_cb() is False
        return False

    handler = _FakePipeHandler(send_cb)
    executor.pipe_handler = handler

    result = executor._do_execute("train", _make_task(task_id="task-1"), _make_fl_ctx(), _AbortSignal())

    assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION
    assert handler.send_calls == 1


def test_task_send_does_not_use_progress_from_another_job(monkeypatch):
    _patch_logs(monkeypatch)
    now = [1000.0]
    monkeypatch.setattr(task_exchanger_module.time, "time", lambda: now[0])
    executor = TaskExchanger(pipe_id="pipe", peer_read_timeout=1.0, streaming_idle_timeout=3.0)

    def send_cb(handler, msg, timeout, abort_signal):
        executor._handle_stream_progress_message(_progress(task_id=msg.msg_id, job_id="other-job"))
        now[0] += timeout
        assert msg._progress_wait_cb() is True
        now[0] += timeout
        assert msg._progress_wait_cb() is True
        now[0] += timeout
        assert msg._progress_wait_cb() is False
        return False

    handler = _FakePipeHandler(send_cb)
    executor.pipe_handler = handler

    result = executor._do_execute("train", _make_task(task_id="task-1"), _make_fl_ctx(), _AbortSignal())

    assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION
    assert handler.send_calls == 1


def test_unknown_job_id_does_not_match_other_job_progress_after_idle(monkeypatch):
    _patch_logs(monkeypatch)
    now = [1000.0]
    monkeypatch.setattr(task_exchanger_module.time, "time", lambda: now[0])
    executor = TaskExchanger(pipe_id="pipe", peer_read_timeout=1.0, streaming_idle_timeout=10.0)
    executor._handle_stream_progress_message(_progress(task_id="task-1", job_id="job-2"))

    assert (
        executor._should_continue_task_send_waiting(
            task_name="train",
            task_id="task-1",
            job_id=None,
            send_start_time=0.0,
            fl_ctx=_make_fl_ctx(),
        )
        is False
    )


def test_integer_zero_job_id_matches_stream_progress(monkeypatch):
    _patch_logs(monkeypatch)
    now = [1000.0]
    monkeypatch.setattr(task_exchanger_module.time, "time", lambda: now[0])
    executor = TaskExchanger(pipe_id="pipe", peer_read_timeout=1.0, streaming_idle_timeout=10.0)
    executor._handle_stream_progress_message(_progress(task_id="task-1", job_id=0))

    assert (
        executor._should_continue_task_send_waiting(
            task_name="train",
            task_id="task-1",
            job_id=0,
            send_start_time=0.0,
            fl_ctx=_make_fl_ctx(),
        )
        is True
    )


def test_execute_stamps_integer_zero_job_id(monkeypatch):
    _patch_logs(monkeypatch)
    executor = TaskExchanger(pipe_id="pipe", peer_read_timeout=1.0, streaming_idle_timeout=10.0)

    def send_cb(handler, msg, timeout, abort_signal):
        assert msg.data.get_header(FLMetaKey.JOB_ID) == "0"
        handler.replies.append(_reply_for(msg))
        return True

    handler = _FakePipeHandler(send_cb)
    executor.pipe_handler = handler

    result = executor._do_execute("train", _make_task(task_id="task-1"), _make_fl_ctx(job_id=0), _AbortSignal())

    assert result.get_return_code() == ReturnCode.OK


def test_execute_empty_job_id_uses_progress_to_extend_task_send_wait(monkeypatch):
    _patch_logs(monkeypatch)
    now = [1000.0]
    monkeypatch.setattr(task_exchanger_module.time, "time", lambda: now[0])
    monkeypatch.setattr(transfer_progress_module.time, "time", lambda: now[0])
    executor = TaskExchanger(pipe_id="pipe", peer_read_timeout=1.0, streaming_idle_timeout=10.0)

    def send_cb(handler, msg, timeout, abort_signal):
        assert msg.data.get_header(FLMetaKey.JOB_ID) == ""
        now[0] += 20.0
        executor._handle_stream_progress_message(_progress(task_id=msg.msg_id, job_id="", sequence=1, bytes_done=1024))
        assert msg._progress_wait_cb() is True
        handler.replies.append(_reply_for(msg))
        return True

    handler = _FakePipeHandler(send_cb)
    executor.pipe_handler = handler

    result = executor._do_execute("train", _make_task(task_id="task-1"), _make_fl_ctx(job_id=None), _AbortSignal())

    assert result.get_return_code() == ReturnCode.OK
    assert handler.send_calls == 1


def test_task_send_times_out_when_activity_does_not_advance_counters(monkeypatch):
    _patch_logs(monkeypatch)
    now = [1000.0]
    monkeypatch.setattr(task_exchanger_module.time, "time", lambda: now[0])
    executor = TaskExchanger(pipe_id="pipe", peer_read_timeout=0.01, streaming_idle_timeout=10.0)
    executor._handle_stream_progress_message(_progress(task_id="task-1", sequence=1, bytes_done=1024))
    now[0] += 11.0
    executor._handle_stream_progress_message(_progress(task_id="task-1", sequence=2, bytes_done=1024))

    handler = _FakePipeHandler(lambda handler, msg, timeout, abort_signal: False)
    executor.pipe_handler = handler

    result = executor._do_execute("train", _make_task(task_id="task-1"), _make_fl_ctx(), _AbortSignal())

    assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION
    assert handler.send_calls == 1


def test_task_send_times_out_when_emitter_stops_after_progress(monkeypatch):
    _patch_logs(monkeypatch)
    now = [1000.0]
    monkeypatch.setattr(task_exchanger_module.time, "time", lambda: now[0])
    executor = TaskExchanger(pipe_id="pipe", peer_read_timeout=0.01, streaming_idle_timeout=10.0)

    def send_cb(handler, msg, timeout, abort_signal):
        executor._handle_stream_progress_message(_progress(task_id=msg.msg_id, sequence=1, bytes_done=1024))
        assert msg._progress_wait_cb() is True
        now[0] += 11.0
        assert msg._progress_wait_cb() is False
        return False

    handler = _FakePipeHandler(send_cb)
    executor.pipe_handler = handler

    result = executor._do_execute("train", _make_task(task_id="task-1"), _make_fl_ctx(), _AbortSignal())

    assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION
    assert handler.send_calls == 1


def test_task_send_does_not_use_result_upload_progress_for_forward_wait(monkeypatch):
    _patch_logs(monkeypatch)
    now = [1000.0]
    monkeypatch.setattr(task_exchanger_module.time, "time", lambda: now[0])
    executor = TaskExchanger(pipe_id="pipe", peer_read_timeout=0.01, streaming_idle_timeout=10.0)
    executor._handle_stream_progress_message(
        _progress(
            task_id="task-1",
            transfer_id="result-ref",
            sequence=1,
            bytes_done=1024,
            direction="result_upload",
        )
    )

    def send_cb(handler, msg, timeout, abort_signal):
        now[0] += 11.0
        assert msg._progress_wait_cb() is False
        return False

    handler = _FakePipeHandler(send_cb)
    executor.pipe_handler = handler

    result = executor._do_execute("train", _make_task(task_id="task-1"), _make_fl_ctx(), _AbortSignal())

    assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION
    assert handler.send_calls == 1


def test_explicit_low_peer_read_timeout_honors_fast_fail(monkeypatch):
    logs = _patch_logs(monkeypatch)
    executor = TaskExchanger(
        pipe_id="pipe",
        peer_read_timeout=1.0,
        peer_read_timeout_explicit=True,
        streaming_idle_timeout=10.0,
    )
    executor._handle_stream_progress_message(_progress(task_id="task-1", sequence=1, bytes_done=1024))

    def send_cb(handler, msg, timeout, abort_signal):
        assert msg._progress_wait_cb() is False
        return False

    handler = _FakePipeHandler(send_cb)
    executor.pipe_handler = handler

    result = executor._do_execute("train", _make_task(task_id="task-1"), _make_fl_ctx(), _AbortSignal())

    assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION
    assert any("honoring fast-fail behavior" in msg for _, msg in logs)


def test_task_abort_marks_active_stream_progress_terminal(monkeypatch):
    _patch_logs(monkeypatch)
    executor = TaskExchanger(pipe_id="pipe", peer_read_timeout=0.01, streaming_idle_timeout=10.0)
    executor._handle_stream_progress_message(_progress(task_id="task-1", sequence=1, bytes_done=1024))
    abort_signal = _AbortSignal()

    def send_cb(handler, msg, timeout, abort_signal):
        abort_signal.trigger("test")
        return True

    handler = _FakePipeHandler(send_cb)
    executor.pipe_handler = handler

    result = executor._do_execute("train", _make_task(task_id="task-1"), _make_fl_ctx(), abort_signal)

    record = executor._stream_progress_tracker.get_record(
        job_id="job-1",
        task_id="task-1",
        transfer_id="transfer-1",
        direction=DIRECTION_TASK_PAYLOAD_DOWNLOAD,
    )
    assert result.get_return_code() == ReturnCode.TASK_ABORTED
    assert record.terminal is True


def test_pipe_status_callback_marks_active_stream_progress_terminal(monkeypatch):
    _patch_logs(monkeypatch)
    executor = TaskExchanger(pipe_id="pipe", peer_read_timeout=0.01, streaming_idle_timeout=10.0)
    executor.pipe = _DummyPipe()
    handler = executor._create_pipe_handler()
    executor._handle_stream_progress_message(_progress(task_id="task-1", sequence=1, bytes_done=1024))

    handler.status_cb(Message.new_request(Topic.PEER_GONE, "peer stopped"))

    record = executor._stream_progress_tracker.get_record(
        job_id="job-1",
        task_id="task-1",
        transfer_id="transfer-1",
        direction=DIRECTION_TASK_PAYLOAD_DOWNLOAD,
    )
    assert handler.asked_to_stop is True
    assert record.terminal is True
    assert record.state == task_exchanger_module.TransferProgressState.ABORTED


def test_old_terminal_stream_progress_records_are_pruned(monkeypatch):
    _patch_logs(monkeypatch)
    now = [1000.0]
    monkeypatch.setattr(task_exchanger_module.time, "time", lambda: now[0])
    executor = TaskExchanger(pipe_id="pipe", peer_read_timeout=0.01, streaming_idle_timeout=10.0)
    executor._handle_stream_progress_message(
        _progress(task_id="old-task", transfer_id="old-ref", sequence=1, bytes_done=1024, state="completed")
    )

    now[0] += task_exchanger_module.STREAM_PROGRESS_TERMINAL_RECORD_TTL + 1.0
    executor._handle_stream_progress_message(
        _progress(task_id="new-task", transfer_id="new-ref", sequence=1, bytes_done=1024, state="completed")
    )

    assert (
        executor._stream_progress_tracker.get_record(
            job_id="job-1",
            task_id="old-task",
            transfer_id="old-ref",
            direction=DIRECTION_TASK_PAYLOAD_DOWNLOAD,
        )
        is None
    )
    assert executor._stream_progress_tracker.get_record(
        job_id="job-1",
        task_id="new-task",
        transfer_id="new-ref",
        direction=DIRECTION_TASK_PAYLOAD_DOWNLOAD,
    )


def test_stream_progress_wait_uses_record_snapshots(monkeypatch):
    _patch_logs(monkeypatch)
    executor = TaskExchanger(pipe_id="pipe")
    executor._handle_stream_progress_message(_progress(task_id="task-1", transfer_id="ref-1"))

    records, active_records = executor._get_active_task_payload_records(task_id="task-1", job_id="job-1")
    live_record = executor._stream_progress_tracker.get_record(
        job_id="job-1",
        task_id="task-1",
        transfer_id="ref-1",
        direction=DIRECTION_TASK_PAYLOAD_DOWNLOAD,
    )
    live_record.state = task_exchanger_module.TransferProgressState.COMPLETED

    assert records[0].state == task_exchanger_module.TransferProgressState.ACTIVE
    assert active_records[0].state == task_exchanger_module.TransferProgressState.ACTIVE


def test_stream_progress_events_are_tracked_by_transfer_id_and_status(monkeypatch, caplog):
    _patch_logs(monkeypatch)
    executor = TaskExchanger(pipe_id="pipe")
    caplog.set_level("INFO", logger=executor.logger.name)

    events = [
        ("transfer-start", 1, "start", 0),
        ("transfer-active", 1, "active", 10),
        ("transfer-completed", 1, "completed", 20),
        ("transfer-failed", 1, "failed", 20),
    ]
    for transfer_id, sequence, state, bytes_done in events:
        executor._handle_stream_progress_message(
            _progress(task_id="task-1", transfer_id=transfer_id, sequence=sequence, bytes_done=bytes_done, state=state)
        )

    completed = executor._stream_progress_tracker.get_record(
        job_id="job-1",
        task_id="task-1",
        transfer_id="transfer-completed",
        direction=DIRECTION_TASK_PAYLOAD_DOWNLOAD,
    )
    failed = executor._stream_progress_tracker.get_record(
        job_id="job-1",
        task_id="task-1",
        transfer_id="transfer-failed",
        direction=DIRECTION_TASK_PAYLOAD_DOWNLOAD,
    )
    assert completed.bytes_done == 20
    assert completed.terminal is True
    assert failed.bytes_done == 20
    assert failed.terminal is True
    assert "accepted stream progress start" in caplog.text
    assert "accepted stream progress active" in caplog.text
    assert "accepted stream progress completion" in caplog.text
    assert "accepted stream progress failure" in caplog.text


def test_stream_progress_records_transfer_id_kind(monkeypatch):
    _patch_logs(monkeypatch)
    executor = TaskExchanger(pipe_id="pipe")

    executor._handle_stream_progress_message(
        _progress(
            task_id="task-1",
            transfer_id="ref-1",
            sequence=1,
            bytes_done=1024,
            transfer_id_kind="download_ref",
        )
    )

    record = executor._stream_progress_tracker.get_record(
        job_id="job-1",
        task_id="task-1",
        transfer_id="ref-1",
        direction=DIRECTION_TASK_PAYLOAD_DOWNLOAD,
    )
    assert record.transfer_id_kind == "download_ref"


def test_stream_progress_tolerates_receiver_id_without_forward_keying(monkeypatch):
    _patch_logs(monkeypatch)
    executor = TaskExchanger(pipe_id="pipe")

    executor._handle_stream_progress_message(
        _progress(
            task_id="task-1",
            transfer_id="ref-1",
            sequence=1,
            bytes_done=1024,
            receiver_id="receiver-a",
        )
    )
    executor._handle_stream_progress_message(
        _progress(
            task_id="task-1",
            transfer_id="ref-1",
            sequence=2,
            bytes_done=2048,
            receiver_id="receiver-b",
        )
    )

    records, active_records = executor._get_active_task_payload_records(task_id="task-1", job_id="job-1")

    assert len(records) == 1
    assert len(active_records) == 1
    assert records[0].transfer_id == "ref-1"
    assert records[0].bytes_done == 2048


def test_stream_progress_rejects_malformed_items_done(monkeypatch):
    _patch_logs(monkeypatch)
    executor = TaskExchanger(pipe_id="pipe")

    msg = _progress(task_id="task-1", transfer_id="ref-1", sequence=1, bytes_done=1024)
    msg.data["items_done"] = "bad"
    executor._handle_stream_progress_message(msg)

    assert (
        executor._stream_progress_tracker.get_record(
            job_id="job-1",
            task_id="task-1",
            transfer_id="ref-1",
            direction=DIRECTION_TASK_PAYLOAD_DOWNLOAD,
        )
        is None
    )


def test_task_payload_stream_progress_requires_scoped_event(monkeypatch):
    _patch_logs(monkeypatch)
    executor = TaskExchanger(pipe_id="pipe")

    missing_job_id = _progress(task_id="task-1", transfer_id="ref-missing-job")
    missing_job_id.data.pop("job_id")
    executor._handle_stream_progress_message(missing_job_id)

    missing_task_id = _progress(task_id="task-1", transfer_id="ref-missing-task")
    missing_task_id.data.pop("task_id")
    executor._handle_stream_progress_message(missing_task_id)

    missing_direction = _progress(task_id="task-1", transfer_id="ref-missing-direction")
    missing_direction.data.pop("direction")
    executor._handle_stream_progress_message(missing_direction)

    assert executor._stream_progress_tracker.records(direction=DIRECTION_TASK_PAYLOAD_DOWNLOAD) == []


def test_task_payload_stream_progress_capacity_bounds_new_records(monkeypatch, caplog):
    _patch_logs(monkeypatch)
    monkeypatch.setattr(task_exchanger_module, "STREAM_PROGRESS_MAX_TRACKED_RECORDS", 2)
    executor = TaskExchanger(pipe_id="pipe", streaming_idle_timeout=10.0)

    executor._handle_stream_progress_message(_progress(task_id="task-1", transfer_id="ref-1"))
    executor._handle_stream_progress_message(_progress(task_id="task-1", transfer_id="ref-2"))
    executor._handle_stream_progress_message(_progress(task_id="task-1", transfer_id="ref-3"))

    records = executor._stream_progress_tracker.records(direction=DIRECTION_TASK_PAYLOAD_DOWNLOAD)
    assert len(records) == 2
    assert {record.transfer_id for record in records} == {"ref-1", "ref-2"}
    assert "records=2 max=2" in caplog.text


def test_stream_progress_ignores_generic_offset_and_current_fields(monkeypatch):
    _patch_logs(monkeypatch)
    executor = TaskExchanger(pipe_id="pipe")

    msg = _progress(task_id="task-1", transfer_id="ref-1", sequence=1, bytes_done=1024)
    msg.data.pop("bytes_done")
    msg.data["bytes"] = 2048
    msg.data["offset"] = "not-a-byte-count"
    msg.data["current"] = "not-a-byte-count"

    executor._handle_stream_progress_message(msg)

    record = executor._stream_progress_tracker.get_record(
        job_id="job-1",
        task_id="task-1",
        transfer_id="ref-1",
        direction=DIRECTION_TASK_PAYLOAD_DOWNLOAD,
    )
    assert record is not None
    assert record.bytes_done == 2048


class _DownloadConsumer(Consumer):
    def __init__(self):
        super().__init__()
        self.completed = False
        self.failed = None

    def consume(self, ref_id: str, state: dict, data):
        return {"next": state.get("index", 0) + 1 if state else 1}

    def download_completed(self, ref_id: str):
        self.completed = True

    def download_failed(self, ref_id: str, reason: str):
        self.failed = reason


class _ChunkCell:
    def __init__(self):
        self.calls = 0

    def send_request(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return make_reply(
                "ok",
                body={
                    _PropKey.STATUS: ProduceRC.OK,
                    _PropKey.STATE: {"index": 0},
                    _PropKey.DATA: [b"abc"],
                },
            )
        return make_reply("ok", body={_PropKey.STATUS: ProduceRC.EOF})


class _LargeModelChunkCell:
    def __init__(self, chunk_count, chunk_size):
        self.chunk_count = chunk_count
        self.chunk = b"x" * chunk_size
        self.calls = 0

    def send_request(self, **kwargs):
        if self.calls < self.chunk_count:
            self.calls += 1
            return make_reply(
                "ok",
                body={
                    _PropKey.STATUS: ProduceRC.OK,
                    _PropKey.STATE: {"index": self.calls},
                    _PropKey.DATA: [self.chunk],
                },
            )
        return make_reply("ok", body={_PropKey.STATUS: ProduceRC.EOF})


class _ProgressPipe:
    def __init__(self, executor):
        self.executor = executor
        self.sent_payloads = []

    def send(self, msg, timeout=None):
        self.sent_payloads.append(msg.data)
        self.executor._handle_stream_progress_message(msg)
        return True


def test_download_object_progress_flows_through_subprocess_pipe_to_task_exchanger(monkeypatch):
    _patch_logs(monkeypatch)
    executor = TaskExchanger(pipe_id="pipe", peer_read_timeout=0.01, streaming_idle_timeout=10.0)
    progress_pipe = _ProgressPipe(executor)
    req = Message.new_request("train", _make_task(), msg_id="task-1")
    observed_wait = []

    def _send_stream_progress(**kwargs):
        progress_pipe.send(Message.new_request(Topic.STREAM_PROGRESS, kwargs))
        if kwargs.get("state") == "active" and kwargs.get("bytes_done", 0) > 0:
            observed_wait.append(req._progress_wait_cb())

    cell_msg = CellMessage(
        headers={
            MessageHeaderKey.MSG_ROOT_ID: "task-1",
            MessageHeaderKey.REQ_ID: "task-1",
            FLMetaKey.JOB_ID: "job-1",
        },
        payload=None,
    )
    fobs_ctx = {
        fobs.FOBSContextKey.MESSAGE: cell_msg,
        fobs.FOBSContextKey.STREAM_PROGRESS_CB: _send_stream_progress,
    }
    progress_cb = ViaDownloaderDecomposer._make_stream_progress_cb(fobs_ctx, "ref-1")
    assert progress_cb is not None

    req._progress_wait_cb = lambda: executor._should_continue_task_send_waiting(
        task_name=req.topic,
        task_id=req.msg_id,
        job_id="job-1",
        send_start_time=0.0,
        fl_ctx=_make_fl_ctx(),
    )
    consumer = _DownloadConsumer()
    download_object(
        from_fqcn="server",
        ref_id="ref-1",
        per_request_timeout=1.0,
        cell=_ChunkCell(),
        consumer=consumer,
        progress_cb=progress_cb,
        progress_interval=0.0,
    )

    assert consumer.completed is True
    assert consumer.failed is None
    assert observed_wait == [True]
    assert any(payload.get("job_id") == "job-1" for payload in progress_pipe.sent_payloads)
    assert any(payload.get("task_id") == "task-1" for payload in progress_pipe.sent_payloads)


def test_swarm_task_payload_progress_from_peer_parent_suppresses_resend(monkeypatch):
    # Focused unit coverage for the task-payload wait policy; test_swarm_progress_e2e.py
    # covers the same behavior through SwarmClientController -> TaskExchanger -> CellPipe.
    _patch_logs(monkeypatch)
    executor = TaskExchanger(pipe_id="pipe", peer_read_timeout=0.01, streaming_idle_timeout=10.0)
    progress_pipe = _ProgressPipe(executor)
    req = Message.new_request("train", _make_task(), msg_id="task-1")
    observed_wait = []

    def _send_stream_progress(**kwargs):
        progress_pipe.send(Message.new_request(Topic.STREAM_PROGRESS, kwargs))
        if kwargs.get("state") == "active" and kwargs.get("bytes_done", 0) > 0:
            observed_wait.append(req._progress_wait_cb())

    cell_msg = CellMessage(
        headers={
            MessageHeaderKey.MSG_ROOT_ID: "task-1",
            MessageHeaderKey.REQ_ID: "task-1",
            FLMetaKey.JOB_ID: "job-1",
        },
        payload=None,
    )
    fobs_ctx = {
        fobs.FOBSContextKey.MESSAGE: cell_msg,
        fobs.FOBSContextKey.STREAM_PROGRESS_CB: _send_stream_progress,
    }
    progress_cb = ViaDownloaderDecomposer._make_stream_progress_cb(fobs_ctx, "ref-1")
    assert progress_cb is not None

    req._progress_wait_cb = lambda: executor._should_continue_task_send_waiting(
        task_name=req.topic,
        task_id=req.msg_id,
        job_id="job-1",
        send_start_time=0.0,
        fl_ctx=_make_fl_ctx(),
    )
    consumer = _DownloadConsumer()
    download_object(
        from_fqcn="site-2.site-2_job-1_active",
        ref_id="ref-1",
        per_request_timeout=1.0,
        cell=_ChunkCell(),
        consumer=consumer,
        progress_cb=progress_cb,
        progress_interval=0.0,
    )

    assert consumer.completed is True
    assert consumer.failed is None
    assert observed_wait == [True]
    assert progress_pipe.sent_payloads[0]["direction"] == DIRECTION_TASK_PAYLOAD_DOWNLOAD
    assert any(payload.get("transfer_id") == "ref-1" for payload in progress_pipe.sent_payloads)


def test_swarm_task_payload_progress_during_send_to_peer_suppresses_resend(monkeypatch):
    logs = _patch_logs(monkeypatch)
    executor = TaskExchanger(pipe_id="pipe", peer_read_timeout=0.01, streaming_idle_timeout=10.0)
    progress_pipe = _ProgressPipe(executor)

    def _send_stream_progress(**kwargs):
        progress_pipe.send(Message.new_request(Topic.STREAM_PROGRESS, kwargs))

    def send_cb(handler, msg, timeout, abort_signal):
        cell_msg = CellMessage(
            headers={
                MessageHeaderKey.MSG_ROOT_ID: msg.msg_id,
                MessageHeaderKey.REQ_ID: msg.msg_id,
                FLMetaKey.JOB_ID: "job-1",
            },
            payload=None,
        )
        progress_cb = ViaDownloaderDecomposer._make_stream_progress_cb(
            {
                fobs.FOBSContextKey.MESSAGE: cell_msg,
                fobs.FOBSContextKey.STREAM_PROGRESS_CB: _send_stream_progress,
            },
            "peer-parent-ref",
        )
        progress_cb(
            sequence=1,
            bytes_done=1024,
            state="active",
            receiver_id="site-2.job-1",
        )
        assert msg._progress_wait_cb() is True
        handler.replies.append(_reply_for(msg))
        return True

    handler = _FakePipeHandler(send_cb)
    executor.pipe_handler = handler

    result = executor._do_execute("train", _make_task(task_id="task-1"), _make_fl_ctx(), _AbortSignal())

    assert result.get_return_code() == ReturnCode.OK
    assert handler.send_calls == 1
    assert progress_pipe.sent_payloads[0]["transfer_id"] == "peer-parent-ref"
    assert progress_pipe.sent_payloads[0]["receiver_id"] == "site-2.job-1"
    assert any("continuing to wait" in msg for _, msg in logs)


def test_simulated_large_model_broadcast_many_clients_progress_suppresses_resend(monkeypatch):
    _patch_logs(monkeypatch)
    now = [1000.0]
    monkeypatch.setattr(task_exchanger_module.time, "time", lambda: now[0])
    monkeypatch.setattr(download_service_module.time, "time", lambda: now[0])

    client_count = 16
    chunks_per_client = 6
    chunk_size = 4096
    wait_results = []
    completed = 0

    for client_idx in range(client_count):
        task_id = f"task-{client_idx}"
        job_id = "large-model-job"
        transfer_id = f"large-model-ref-{client_idx}"
        executor = TaskExchanger(
            pipe_id=f"pipe-{client_idx}",
            peer_read_timeout=1.0,
            streaming_idle_timeout=10.0,
            result_poll_interval=0.01,
        )
        progress_pipe = _ProgressPipe(executor)
        req = Message.new_request("train", _make_task(task_id=task_id), msg_id=task_id)
        send_start_time = now[0]

        req._progress_wait_cb = lambda executor=executor, task_id=task_id: executor._should_continue_task_send_waiting(
            task_name="train",
            task_id=task_id,
            job_id=job_id,
            send_start_time=send_start_time,
            fl_ctx=_make_fl_ctx(job_id=job_id),
        )

        def _send_stream_progress(**kwargs):
            progress_pipe.send(Message.new_request(Topic.STREAM_PROGRESS, kwargs))
            if kwargs.get("state") == "active" and kwargs.get("bytes_done", 0) > 0:
                now[0] += 9.0
                wait_results.append(req._progress_wait_cb())

        cell_msg = CellMessage(
            headers={
                MessageHeaderKey.MSG_ROOT_ID: task_id,
                MessageHeaderKey.REQ_ID: task_id,
                FLMetaKey.JOB_ID: job_id,
            },
            payload=None,
        )
        progress_cb = ViaDownloaderDecomposer._make_stream_progress_cb(
            {
                fobs.FOBSContextKey.MESSAGE: cell_msg,
                fobs.FOBSContextKey.STREAM_PROGRESS_CB: _send_stream_progress,
            },
            transfer_id,
        )
        assert progress_cb is not None

        consumer = _DownloadConsumer()
        download_object(
            from_fqcn="server",
            ref_id=transfer_id,
            per_request_timeout=1.0,
            cell=_LargeModelChunkCell(chunk_count=chunks_per_client, chunk_size=chunk_size),
            consumer=consumer,
            progress_cb=progress_cb,
            progress_interval=0.0,
        )

        assert consumer.completed is True
        assert consumer.failed is None
        completed += 1

    assert completed == client_count
    assert len(wait_results) == client_count * chunks_per_client
    assert all(wait_results)


def test_simulated_large_model_broadcast_many_clients_delayed_ack_uses_progress_not_retry(monkeypatch):
    _patch_logs(monkeypatch)
    now = [1000.0]
    monkeypatch.setattr(task_exchanger_module.time, "time", lambda: now[0])

    client_count = 16
    timeout_windows = 6
    total_bytes = 5 * 1024 * 1024 * 1024
    bytes_per_window = total_bytes // timeout_windows
    send_calls = []

    for client_idx in range(client_count):
        task_id = f"task-{client_idx}"
        job_id = "large-model-job"
        transfer_id = f"large-model-ref-{client_idx}"
        executor = TaskExchanger(
            pipe_id=f"pipe-{client_idx}",
            peer_read_timeout=1.0,
            streaming_idle_timeout=10.0,
            result_poll_interval=0.01,
        )
        progress_pipe = _ProgressPipe(executor)

        def send_cb(handler, msg, timeout, abort_signal):
            send_calls.append((client_idx, msg.msg_id))

            def _send_stream_progress(**kwargs):
                progress_pipe.send(Message.new_request(Topic.STREAM_PROGRESS, kwargs))

            cell_msg = CellMessage(
                headers={
                    MessageHeaderKey.MSG_ROOT_ID: msg.msg_id,
                    MessageHeaderKey.REQ_ID: msg.msg_id,
                    FLMetaKey.JOB_ID: job_id,
                },
                payload=None,
            )
            progress_cb = ViaDownloaderDecomposer._make_stream_progress_cb(
                {
                    fobs.FOBSContextKey.MESSAGE: cell_msg,
                    fobs.FOBSContextKey.STREAM_PROGRESS_CB: _send_stream_progress,
                },
                transfer_id,
            )

            for sequence in range(1, timeout_windows + 1):
                progress_cb(
                    sequence=sequence,
                    bytes_done=sequence * bytes_per_window,
                    state="active",
                )
                now[0] += timeout + 0.1
                assert msg._progress_wait_cb() is True

            progress_cb(sequence=timeout_windows + 1, bytes_done=total_bytes, state="completed")
            now[0] += timeout + 0.1
            assert msg._progress_wait_cb() is True
            handler.replies.append(_reply_for(msg))
            return True

        handler = _FakePipeHandler(send_cb)
        executor.pipe_handler = handler

        result = executor._do_execute("train", _make_task(task_id=task_id), _make_fl_ctx(job_id=job_id), _AbortSignal())

        assert result.get_return_code() == ReturnCode.OK
        assert handler.send_calls == 1

    assert len(send_calls) == client_count


def test_simulated_large_model_broadcast_many_clients_stalled_receiver_fails_wait(monkeypatch):
    _patch_logs(monkeypatch)
    now = [1000.0]
    monkeypatch.setattr(task_exchanger_module.time, "time", lambda: now[0])
    monkeypatch.setattr(download_service_module.time, "time", lambda: now[0])

    client_count = 16
    chunks_per_client = 3
    chunk_size = 4096
    stalled_client = 7
    completed = 0
    stalled_wait_result = None

    for client_idx in range(client_count):
        task_id = f"task-{client_idx}"
        job_id = "large-model-job"
        transfer_id = f"large-model-ref-{client_idx}"
        executor = TaskExchanger(
            pipe_id=f"pipe-{client_idx}",
            peer_read_timeout=1.0,
            streaming_idle_timeout=10.0,
            result_poll_interval=0.01,
        )
        progress_pipe = _ProgressPipe(executor)
        req = Message.new_request("train", _make_task(task_id=task_id), msg_id=task_id)
        send_start_time = now[0]

        req._progress_wait_cb = lambda executor=executor, task_id=task_id: executor._should_continue_task_send_waiting(
            task_name="train",
            task_id=task_id,
            job_id=job_id,
            send_start_time=send_start_time,
            fl_ctx=_make_fl_ctx(job_id=job_id),
        )

        def _send_stream_progress(**kwargs):
            progress_pipe.send(Message.new_request(Topic.STREAM_PROGRESS, kwargs))

        cell_msg = CellMessage(
            headers={
                MessageHeaderKey.MSG_ROOT_ID: task_id,
                MessageHeaderKey.REQ_ID: task_id,
                FLMetaKey.JOB_ID: job_id,
            },
            payload=None,
        )
        progress_cb = ViaDownloaderDecomposer._make_stream_progress_cb(
            {
                fobs.FOBSContextKey.MESSAGE: cell_msg,
                fobs.FOBSContextKey.STREAM_PROGRESS_CB: _send_stream_progress,
            },
            transfer_id,
        )
        assert progress_cb is not None

        if client_idx == stalled_client:
            progress_cb(sequence=1, bytes_done=0, state="active")
            assert req._progress_wait_cb() is True
            now[0] += 11.0
            stalled_wait_result = req._progress_wait_cb()
            continue

        consumer = _DownloadConsumer()
        download_object(
            from_fqcn="server",
            ref_id=transfer_id,
            per_request_timeout=1.0,
            cell=_LargeModelChunkCell(chunk_count=chunks_per_client, chunk_size=chunk_size),
            consumer=consumer,
            progress_cb=progress_cb,
            progress_interval=0.0,
        )
        assert consumer.completed is True
        assert consumer.failed is None
        completed += 1

    assert completed == client_count - 1
    assert stalled_wait_result is False


def test_simulated_large_model_broadcast_many_clients_delayed_ack_without_progress_times_out(monkeypatch):
    _patch_logs(monkeypatch)
    now = [1000.0]
    monkeypatch.setattr(task_exchanger_module.time, "time", lambda: now[0])

    client_count = 16
    failures = 0

    for client_idx in range(client_count):
        task_id = f"task-{client_idx}"
        job_id = "large-model-job"
        executor = TaskExchanger(
            pipe_id=f"pipe-{client_idx}",
            peer_read_timeout=1.0,
            streaming_idle_timeout=10.0,
            result_poll_interval=0.01,
        )

        def send_cb(handler, msg, timeout, abort_signal):
            now[0] += 11.0
            assert msg._progress_wait_cb() is False
            return False

        handler = _FakePipeHandler(send_cb)
        executor.pipe_handler = handler

        result = executor._do_execute("train", _make_task(task_id=task_id), _make_fl_ctx(job_id=job_id), _AbortSignal())

        assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION
        assert handler.send_calls == 1
        failures += 1

    assert failures == client_count


def test_cellpipe_stack_delayed_ack_continues_on_progress_without_resend(monkeypatch):
    logs = _patch_logs(monkeypatch)
    now = [1000.0]
    monkeypatch.setattr(task_exchanger_module.time, "time", lambda: now[0])
    monkeypatch.setattr(transfer_progress_module.time, "time", lambda: now[0])

    executor = TaskExchanger(
        pipe_id="pipe",
        peer_read_timeout=1.0,
        streaming_idle_timeout=10.0,
        resend_interval=0.01,
        max_resends=0,
    )
    cell = _CellStackFake()
    executor.pipe = _make_cell_stack_pipe(cell)
    handler = executor._create_pipe_handler()
    decisions = []
    original_should_continue = executor._should_continue_task_send_waiting

    def _record_should_continue(*args, **kwargs):
        decision = original_should_continue(*args, **kwargs)
        decisions.append(decision)
        return decision

    monkeypatch.setattr(executor, "_should_continue_task_send_waiting", _record_should_continue)
    task_id = "task-stack"
    job_id = "large-model-job"
    transfer_id = "large-model-ref"
    wait_calls = []

    def _conditional_wait(event, timeout, abort_signal, **kwargs):
        wait_calls.append(timeout)
        if len(wait_calls) <= 2:
            now[0] += timeout + 0.1
            handler.msg_cb(
                _progress(
                    task_id=task_id,
                    job_id=job_id,
                    transfer_id=transfer_id,
                    sequence=len(wait_calls),
                    bytes_done=len(wait_calls) * 1024 * 1024,
                )
            )
            return WaiterRC.TIMEOUT

        waiter = next(iter(cell.requests_dict.values()))
        waiter.receiving_future = _ReadyStreamFuture(
            headers={MessageHeaderKey.RETURN_CODE: CellReturnCode.OK},
            payload=None,
        )
        waiter.in_receiving.set()
        return WaiterRC.IS_SET

    monkeypatch.setattr(cell_module, "conditional_wait", _conditional_wait)

    shareable = _make_task(task_id=task_id)
    shareable.set_header(FLMetaKey.JOB_ID, job_id)
    req = Message.new_request("train", shareable, msg_id=task_id)
    sent = executor._send_task_to_peer(req, _make_fl_ctx(job_id), _AbortSignal())

    assert sent is True
    assert len(cell.sent_blobs) == 1
    assert decisions == [True, True]
    assert any("continuing to wait" in msg for _, msg in logs)
    assert not hasattr(req, "_cached_cell_msg")


def test_stream_progress_callback_error_does_not_escape_pipe_reader(monkeypatch):
    executor = TaskExchanger(pipe_id="pipe", peer_read_timeout=1.0, streaming_idle_timeout=10.0)
    executor.pipe = _DummyPipe()
    executor.logger = MagicMock()
    handler = executor._create_pipe_handler()

    def _raise(_msg):
        raise RuntimeError("bad progress")

    monkeypatch.setattr(executor, "_handle_stream_progress_message", _raise)

    handler.msg_cb(Message.new_request(Topic.STREAM_PROGRESS, {"task_id": "task-1"}))

    assert list(handler.messages) == []
    executor.logger.warning.assert_called_once()
    assert "ignored stream progress after handler error" in executor.logger.warning.call_args.args[0]


def test_cellpipe_stack_delayed_ack_without_progress_times_out_without_resend(monkeypatch):
    _patch_logs(monkeypatch)
    now = [1000.0]
    monkeypatch.setattr(task_exchanger_module.time, "time", lambda: now[0])
    monkeypatch.setattr(transfer_progress_module.time, "time", lambda: now[0])

    executor = TaskExchanger(
        pipe_id="pipe",
        peer_read_timeout=1.0,
        streaming_idle_timeout=3.0,
        resend_interval=0.01,
        max_resends=0,
    )
    cell = _CellStackFake()
    executor.pipe = _make_cell_stack_pipe(cell)
    executor._create_pipe_handler()
    decisions = []
    original_should_continue = executor._should_continue_task_send_waiting

    def _record_should_continue(*args, **kwargs):
        decision = original_should_continue(*args, **kwargs)
        decisions.append(decision)
        return decision

    monkeypatch.setattr(executor, "_should_continue_task_send_waiting", _record_should_continue)
    wait_calls = []

    def _conditional_wait(event, timeout, abort_signal, **kwargs):
        wait_calls.append(timeout)
        now[0] += timeout + 0.1
        return WaiterRC.TIMEOUT

    monkeypatch.setattr(cell_module, "conditional_wait", _conditional_wait)

    task_id = "task-stack"
    job_id = "large-model-job"
    shareable = _make_task(task_id=task_id)
    shareable.set_header(FLMetaKey.JOB_ID, job_id)
    req = Message.new_request("train", shareable, msg_id=task_id)
    sent = executor._send_task_to_peer(req, _make_fl_ctx(job_id), _AbortSignal())

    assert sent is False
    assert len(cell.sent_blobs) == 1
    assert decisions == [True, True, False]
    assert not hasattr(req, "_cached_cell_msg")
