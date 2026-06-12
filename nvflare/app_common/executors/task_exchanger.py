# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import time
from dataclasses import dataclass
from typing import Optional

from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey, FLMetaKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.fuel.f3.streaming.download_service import DownloadService
from nvflare.fuel.f3.streaming.transfer_progress import (
    DEFAULT_STREAMING_IDLE_TIMEOUT,
    DIRECTION_TASK_PAYLOAD_DOWNLOAD,
    STREAM_PROGRESS_COMPLETION_ACK_GRACE,
    TransferProgressState,
    TransferProgressTracker,
)
from nvflare.fuel.utils.constants import PipeChannelName
from nvflare.fuel.utils.pipe.cell_pipe import CellPipe
from nvflare.fuel.utils.pipe.pipe import Message, Pipe, Topic
from nvflare.fuel.utils.pipe.pipe_handler import PipeHandler
from nvflare.fuel.utils.validation_utils import (
    check_non_negative_int,
    check_non_negative_number,
    check_positive_number,
    check_str,
)
from nvflare.security.logging import secure_format_exception

STREAM_PROGRESS_TASK_ID_KEYS = ("task_id", "req_id", "request_id", "msg_id")
STREAM_PROGRESS_JOB_ID_KEYS = ("job_id",)
STREAM_PROGRESS_TRANSFER_ID_KEYS = ("transfer_id", "ref_id", "stream_id")
STREAM_PROGRESS_TRANSFER_ID_KIND_KEYS = ("transfer_id_kind", "stream_id_kind")
STREAM_PROGRESS_DIRECTION_KEYS = ("direction",)
STREAM_PROGRESS_RECEIVER_ID_KEYS = ("receiver_id", "requester_id", "requester_fqcn")
STREAM_PROGRESS_SEQUENCE_KEYS = ("sequence", "seq")
STREAM_PROGRESS_BYTES_KEYS = ("bytes_done", "progress", "bytes", "bytes_read", "bytes_received")
STREAM_PROGRESS_ITEM_KEYS = ("items_done", "items", "item_count")
STREAM_PROGRESS_STATE_KEYS = ("state", "status", "event", "event_type")
STREAM_PROGRESS_START_STATUSES = ("start", "started")
_DEFAULT_STREAMING_IDLE_TIMEOUT_SECS = DEFAULT_STREAMING_IDLE_TIMEOUT
STREAM_PROGRESS_MAX_TRACKED_RECORDS = 4096
# Match the DownloadService finished-ref tombstone window so late EOF/completion
# replies after clean transfer completion can still find the progress record.
STREAM_PROGRESS_TERMINAL_RECORD_TTL = DownloadService.FINISHED_REFS_TTL

STREAM_PROGRESS_STATE_ALIASES = {
    "active": "active",
    "progress": "active",
    "in_progress": "active",
    "running": "active",
    "start": "active",
    "started": "active",
    "completed": "completed",
    "complete": "completed",
    "done": "completed",
    "success": "completed",
    "failed": "failed",
    "fail": "failed",
    "failure": "failed",
    "error": "failed",
    "exception": "failed",
    "aborted": "aborted",
    "abort": "aborted",
    "cancelled": "aborted",
}


@dataclass(frozen=True)
class _StreamProgressRecordSnapshot:
    job_id: str
    task_id: str
    transfer_id: str
    direction: str
    sequence: int
    bytes_done: int
    items_done: Optional[int]
    started_time: float
    last_progress_time: float
    state: str
    transfer_id_kind: Optional[str]

    @property
    def terminal(self) -> bool:
        return self.state in TransferProgressState.TERMINAL_STATES


@dataclass(frozen=True)
class _StreamingTimeoutSnapshot:
    peer_read_timeout_explicit: bool
    peer_read_timeout: Optional[float]
    streaming_idle_timeout: Optional[float]


class TaskExchanger(Executor):
    def __init__(
        self,
        pipe_id: str,
        read_interval: float = 0.5,
        heartbeat_interval: float = 5.0,
        heartbeat_timeout: Optional[float] = 60.0,
        resend_interval: float = 2.0,
        max_resends: Optional[int] = None,
        peer_read_timeout: Optional[float] = 60.0,
        peer_read_timeout_explicit: bool = False,
        streaming_idle_timeout: Optional[float] = _DEFAULT_STREAMING_IDLE_TIMEOUT_SECS,
        task_wait_time: Optional[float] = None,
        result_poll_interval: float = 0.5,
        pipe_channel_name=PipeChannelName.TASK,
    ):
        """Constructor of TaskExchanger.

        Args:
            pipe_id (str): component id of pipe.
            read_interval (float): how often to read from pipe.
            heartbeat_interval (float): how often to send heartbeat to peer.
            heartbeat_timeout (float, optional): how long to wait for a
                heartbeat from the peer before treating the peer as dead,
                0 means DO NOT check for heartbeat.
            resend_interval (float): how often to resend a message if failing to send.
                None means no resend. Note that if the pipe does not support resending,
                then no resend.
            max_resends (int, optional): max number of resend. None means no limit.
                Defaults to None.
            peer_read_timeout (float, optional): time to wait for peer to accept sent message.
            peer_read_timeout_explicit (bool): whether peer_read_timeout came from an explicit user override. When
                lower than streaming_idle_timeout, this preserves fast-fail behavior instead of progress-extending.
            streaming_idle_timeout (float, optional): when task-send peer-read times out, continue waiting only while
                the exact transfer has made monotonic stream progress within this many seconds.
            task_wait_time (float, optional): how long to wait for a task to complete.
                None means waiting forever. Defaults to None.
            result_poll_interval (float): how often to poll task result.
                Defaults to 0.5.
            pipe_channel_name: the channel name for sending task requests.
                Defaults to "task".
        """
        Executor.__init__(self)
        check_str("pipe_id", pipe_id)
        check_positive_number("read_interval", read_interval)
        check_positive_number("heartbeat_interval", heartbeat_interval)
        if heartbeat_timeout is not None:
            check_non_negative_number("heartbeat_timeout", heartbeat_timeout)
        check_positive_number("resend_interval", resend_interval)
        if max_resends is not None:
            check_non_negative_int("max_resends", max_resends)
        if peer_read_timeout is not None:
            check_positive_number("peer_read_timeout", peer_read_timeout)
        if streaming_idle_timeout is not None:
            check_positive_number("streaming_idle_timeout", streaming_idle_timeout)
        if task_wait_time is not None:
            check_positive_number("task_wait_time", task_wait_time)
        check_positive_number("result_poll_interval", result_poll_interval)
        check_str("pipe_channel_name", pipe_channel_name)

        self.pipe_id = pipe_id
        self.read_interval = read_interval
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_timeout = heartbeat_timeout
        self.resend_interval = resend_interval
        self.max_resends = max_resends
        # Timeout values are immutable after construction. Send-loop readers use snapshots to keep related
        # values consistent and to make future runtime reconfiguration explicit.
        self.peer_read_timeout = peer_read_timeout
        self.peer_read_timeout_explicit = peer_read_timeout_explicit
        self.streaming_idle_timeout = streaming_idle_timeout
        self.task_wait_time = task_wait_time
        self.result_poll_interval = result_poll_interval
        self.pipe_channel_name = pipe_channel_name
        self.pipe = None
        self.pipe_handler = None
        self._executing = threading.Event()
        self._executing_lock = threading.Lock()
        self._stream_progress_lock = threading.Lock()
        self._stream_progress_tracker = self._make_stream_progress_tracker()
        self._explicit_peer_read_timeout_warned = False
        self._task_send_startup_budget_info_logged = False
        self._peer_read_timeout_once_lock = threading.Lock()

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            engine = fl_ctx.get_engine()
            self.pipe = engine.get_component(self.pipe_id)
            if not isinstance(self.pipe, Pipe):
                self.system_panic(f"component of {self.pipe_id} must be Pipe but got {type(self.pipe)}", fl_ctx)
                return
            self.pipe.open(self.pipe_channel_name)
        elif event_type == EventType.BEFORE_TASK_EXECUTION:
            with self._executing_lock:
                if self._executing.is_set():
                    skip = True
                else:
                    skip = False
            if skip:
                self.log_debug(fl_ctx, "skipping pipe handler reset: execute() is in progress")
                return
            # Ensure pipe is initialized and is a Pipe before operating on it.
            if not isinstance(self.pipe, Pipe):
                self.log_debug(fl_ctx, "pipe not initialized or not a Pipe; skipping pipe handler reset")
                return
            if self.pipe_handler:
                self.pipe_handler.stop(close_pipe=False)
            self.pipe.clear()
            self._create_pipe_handler()
            self.pipe_handler.start()
        elif event_type == EventType.ABOUT_TO_END_RUN:
            self.log_debug(fl_ctx, "Stopping pipe handler")
            self._mark_all_stream_progress_terminal(TransferProgressState.ABORTED)
            if self.pipe_handler:
                self.pipe_handler.notify_end("end_of_job")
                self.pipe_handler.stop(close_pipe=False)
            if self.pipe:
                self.pipe.close()

    def _create_pipe_handler(self):
        """Create a new PipeHandler for self.pipe with a handler-bound status callback.

        Each handler gets its own closure that checks identity before stopping,
        so a late PEER_GONE from a previous handler cannot kill the current one.
        The callback uses close_pipe=False because CellPipe.close() is irreversible.
        """
        if self.heartbeat_timeout is None:
            raise ValueError(
                "heartbeat_timeout is None. Set heartbeat_timeout to 0 to disable heartbeat checking, "
                "or to a non-negative timeout value."
            )
        handler = PipeHandler(
            pipe=self.pipe,
            read_interval=self.read_interval,
            heartbeat_interval=self.heartbeat_interval,
            heartbeat_timeout=self.heartbeat_timeout,
            resend_interval=self.resend_interval,
            max_resends=self.max_resends,
        )

        def _bound_status_cb(msg, _h=handler):
            if self.pipe_handler is not _h:
                self.logger.debug(f"Ignoring late {msg.topic} from a previous pipe handler")
                return
            self.logger.info(f"pipe status changed to {msg.topic}: {msg.data}")
            self._mark_all_stream_progress_terminal(TransferProgressState.ABORTED)
            _h.stop(close_pipe=False)

        handler.set_status_cb(_bound_status_cb)

        def _bound_msg_cb(msg, _h=handler):
            if self.pipe_handler is not _h:
                self.logger.debug(f"Ignoring late {msg.topic} from a previous pipe handler")
                return
            if msg.topic == Topic.STREAM_PROGRESS:
                try:
                    self._handle_stream_progress_message(msg)
                except Exception as ex:
                    self.logger.warning(f"ignored stream progress after handler error: {secure_format_exception(ex)}")
            else:
                with _h.lock:
                    _h.messages.append(msg)

        handler.set_message_cb(_bound_msg_cb)
        self.pipe_handler = handler
        return handler

    def _make_stream_progress_tracker(self):
        idle_timeout = self.streaming_idle_timeout or _DEFAULT_STREAMING_IDLE_TIMEOUT_SECS
        return TransferProgressTracker(idle_timeout=idle_timeout)

    @staticmethod
    def _get_progress_event_value(data, keys):
        for key in keys:
            if isinstance(data, dict):
                if key in data:
                    return data.get(key)
            elif hasattr(data, key):
                return getattr(data, key)
        return None

    @staticmethod
    def _normalize_progress_status(status) -> str:
        if status is None:
            return ""
        return str(status).lower()

    @staticmethod
    def _normalize_tracker_state(status: str) -> str:
        return STREAM_PROGRESS_STATE_ALIASES.get(status, TransferProgressState.ACTIVE)

    def _handle_stream_progress_message(self, msg: Message):
        data = msg.data
        if not isinstance(data, dict):
            self.logger.warning(f"ignored stream progress with invalid payload: {data}")
            return

        task_id = self._get_progress_event_value(data, STREAM_PROGRESS_TASK_ID_KEYS)
        transfer_id = self._get_progress_event_value(data, STREAM_PROGRESS_TRANSFER_ID_KEYS)
        job_id = self._get_progress_event_value(data, STREAM_PROGRESS_JOB_ID_KEYS)
        direction = self._get_progress_event_value(data, STREAM_PROGRESS_DIRECTION_KEYS)
        if direction is None:
            self.logger.warning(f"ignored stream progress without direction: {data}")
            return
        direction = str(direction)
        if direction != DIRECTION_TASK_PAYLOAD_DOWNLOAD:
            self.logger.debug(f"ignored stream progress for unsupported direction {direction}: {data}")
            return
        if job_id is None or task_id is None or transfer_id is None:
            self.logger.warning(f"ignored unscoped task_payload_download stream progress: {data}")
            return

        receiver_id = self._get_progress_event_value(data, STREAM_PROGRESS_RECEIVER_ID_KEYS)
        transfer_id_kind = self._get_progress_event_value(data, STREAM_PROGRESS_TRANSFER_ID_KIND_KEYS)
        status = self._normalize_progress_status(self._get_progress_event_value(data, STREAM_PROGRESS_STATE_KEYS))
        bytes_done = self._get_progress_event_value(data, STREAM_PROGRESS_BYTES_KEYS)
        items_done = self._get_progress_event_value(data, STREAM_PROGRESS_ITEM_KEYS)
        sequence = self._get_progress_event_value(data, STREAM_PROGRESS_SEQUENCE_KEYS)

        try:
            bytes_done_value = int(bytes_done) if bytes_done is not None else 0
        except (TypeError, ValueError):
            self.logger.warning(f"ignored stream progress with invalid bytes_done value: {data}")
            return

        try:
            items_done_value = int(items_done) if items_done is not None else None
        except (TypeError, ValueError):
            self.logger.warning(f"ignored stream progress with invalid items_done value: {data}")
            return

        job_id = str(job_id)
        task_id = str(task_id)
        transfer_id = str(transfer_id)
        # execute() normalizes a missing FLContext job id to an empty string and stamps the same
        # value into the task header. Treat that empty string as a valid progress scope here.
        if not task_id or not transfer_id:
            self.logger.warning(f"ignored unscoped task_payload_download stream progress: {data}")
            return
        transfer_id_kind = None if transfer_id_kind is None else str(transfer_id_kind)
        receiver_id = None if receiver_id is None else str(receiver_id)
        state = self._normalize_tracker_state(status)

        with self._stream_progress_lock:
            # Forward task payload aggregation is task/transfer scoped. receiver_id is tolerated for schema
            # compatibility but intentionally not part of the forward-path tracker key.
            record = self._stream_progress_tracker.get_record(
                job_id=job_id,
                task_id=task_id,
                transfer_id=transfer_id,
                direction=direction,
            )
            if record is None:
                has_capacity, record_count = self._stream_progress_tracker_capacity_locked(direction)
                if not has_capacity:
                    self.logger.warning(
                        f"ignored stream progress for task={task_id} transfer={transfer_id} direction={direction}: "
                        f"progress tracker is at capacity records={record_count} max={STREAM_PROGRESS_MAX_TRACKED_RECORDS}"
                    )
                    return
            try:
                sequence_value = int(sequence) if sequence is not None else (record.sequence + 1 if record else 0)
            except (TypeError, ValueError):
                sequence_value = record.sequence + 1 if record else 0
            update = self._stream_progress_tracker.update(
                job_id=job_id,
                task_id=task_id,
                transfer_id=transfer_id,
                direction=direction,
                sequence=sequence_value,
                bytes_done=bytes_done_value,
                items_done=items_done_value,
                state=state,
                transfer_id_kind=transfer_id_kind,
            )
            if update.accepted and update.record and update.record.terminal:
                self._prune_terminal_stream_progress_records_locked()

        if update.accepted:
            if status in STREAM_PROGRESS_START_STATUSES:
                event_kind = "start"
            elif state == TransferProgressState.COMPLETED:
                event_kind = "completion"
            elif state in (TransferProgressState.FAILED, TransferProgressState.ABORTED):
                event_kind = "failure"
            else:
                event_kind = "active"
            self.logger.info(
                f"accepted stream progress {event_kind} for task={task_id} transfer={transfer_id} direction={direction} "
                f"receiver_id={receiver_id} state={state} sequence={sequence_value} bytes_done={bytes_done_value} "
                f"items_done={items_done_value} progressed={update.progressed}"
            )
        else:
            self.logger.debug(
                f"ignored stream progress for task={task_id} transfer={transfer_id} direction={direction}: "
                f"{update.reason}"
            )

    def _get_active_task_payload_records(self, task_id: str, job_id: Optional[str] = None):
        normalized_job_id = "" if job_id is None else str(job_id)
        with self._stream_progress_lock:
            records = [
                _StreamProgressRecordSnapshot(
                    job_id=record.job_id,
                    task_id=record.task_id,
                    transfer_id=record.transfer_id,
                    direction=record.direction,
                    sequence=record.sequence,
                    bytes_done=record.bytes_done,
                    items_done=record.items_done,
                    started_time=record.started_time,
                    last_progress_time=record.last_progress_time,
                    state=record.state,
                    transfer_id_kind=record.transfer_id_kind,
                )
                for record in self._stream_progress_tracker.records(
                    job_id=normalized_job_id,
                    task_id=str(task_id),
                    direction=DIRECTION_TASK_PAYLOAD_DOWNLOAD,
                )
            ]
        return records, [record for record in records if not record.terminal]

    def _prune_terminal_stream_progress_records_locked(self):
        self._stream_progress_tracker.prune(before_time=time.time() - STREAM_PROGRESS_TERMINAL_RECORD_TTL)

    def _stream_progress_tracker_capacity_locked(self, direction: str) -> tuple[bool, int]:
        max_records = STREAM_PROGRESS_MAX_TRACKED_RECORDS
        if max_records <= 0:
            return True, 0

        record_count = len(self._stream_progress_tracker.records(direction=direction))
        if record_count < max_records:
            return True, record_count

        removed_count = self._stream_progress_tracker.prune(
            before_time=time.time() - STREAM_PROGRESS_TERMINAL_RECORD_TTL,
            direction=direction,
        )
        record_count -= removed_count
        if record_count < max_records:
            return True, record_count

        idle_timeout = self.streaming_idle_timeout or _DEFAULT_STREAMING_IDLE_TIMEOUT_SECS
        removed_count = self._stream_progress_tracker.prune(
            before_time=time.time() - idle_timeout,
            include_active=True,
            direction=direction,
        )
        record_count -= removed_count
        return record_count < max_records, record_count

    def _recent_completed_records_hold_wait(
        self,
        records,
        now: float,
        fl_ctx: FLContext,
        task_name: str,
        completed_ack_budget: Optional[float],
    ) -> bool:
        if not records:
            return False
        completed_records = [record for record in records if record.state == TransferProgressState.COMPLETED]
        if len(completed_records) != len(records):
            return False
        latest_record = max(completed_records, key=lambda record: record.last_progress_time)
        elapsed = now - latest_record.last_progress_time
        if completed_ack_budget is not None and elapsed >= completed_ack_budget:
            return False
        completed_ack_budget_text = "unbounded" if completed_ack_budget is None else f"{completed_ack_budget}s"
        self.log_info(
            fl_ctx,
            f"peer has not ACKed task '{task_name}' yet, but stream transfer "
            f"'{latest_record.transfer_id}' completed {elapsed:.2f} secs ago; continuing to wait "
            f"until task_send_completed_ack_budget={completed_ack_budget_text}",
        )
        return True

    @staticmethod
    def _get_task_send_startup_budget(
        streaming_idle_timeout: float,
        peer_read_timeout: Optional[float] = None,
    ) -> Optional[float]:
        if peer_read_timeout is None:
            return None
        peer_read_budget = peer_read_timeout
        return min(streaming_idle_timeout, max(peer_read_budget, STREAM_PROGRESS_COMPLETION_ACK_GRACE))

    @staticmethod
    def _get_task_send_completed_ack_budget(
        streaming_idle_timeout: float,
        peer_read_timeout: Optional[float] = None,
    ) -> Optional[float]:
        if peer_read_timeout is None:
            return None
        peer_read_budget = peer_read_timeout
        return min(streaming_idle_timeout, max(peer_read_budget, STREAM_PROGRESS_COMPLETION_ACK_GRACE))

    @staticmethod
    def _is_explicit_peer_read_timeout_fast_fail(
        peer_read_timeout_explicit: bool,
        peer_read_timeout: Optional[float],
        streaming_idle_timeout: Optional[float],
    ) -> bool:
        return (
            peer_read_timeout_explicit
            and peer_read_timeout is not None
            and streaming_idle_timeout is not None
            and peer_read_timeout < streaming_idle_timeout
        )

    def _get_streaming_timeout_snapshot(self):
        with self._stream_progress_lock:
            return _StreamingTimeoutSnapshot(
                peer_read_timeout_explicit=self.peer_read_timeout_explicit,
                peer_read_timeout=self.peer_read_timeout,
                streaming_idle_timeout=self.streaming_idle_timeout,
            )

    def _should_continue_task_send_waiting(
        self,
        task_name: str,
        task_id: str,
        job_id: Optional[str],
        send_start_time: float,
        fl_ctx: FLContext,
    ) -> bool:
        timeout_snapshot = self._get_streaming_timeout_snapshot()
        peer_read_timeout = timeout_snapshot.peer_read_timeout
        streaming_idle_timeout = timeout_snapshot.streaming_idle_timeout

        if not streaming_idle_timeout:
            return False

        now = time.time()
        records, active_records = self._get_active_task_payload_records(task_id, job_id)
        if not records:
            elapsed = now - send_start_time
            wait_budget = self._get_task_send_startup_budget(streaming_idle_timeout, peer_read_timeout)
            if wait_budget is not None and elapsed >= wait_budget:
                return False
            wait_budget_text = "unbounded" if wait_budget is None else f"{wait_budget}s"
            self.log_info(
                fl_ctx,
                f"peer has not read task '{task_name}' after {elapsed} secs and no stream progress record "
                f"exists yet; continuing to wait until task_send_wait_budget={wait_budget_text}",
            )
            return True

        if not active_records:
            completed_ack_budget = self._get_task_send_completed_ack_budget(streaming_idle_timeout, peer_read_timeout)
            if self._recent_completed_records_hold_wait(records, now, fl_ctx, task_name, completed_ack_budget):
                return True
            return False

        terminal_failure_records = [
            record
            for record in records
            if record.state in (TransferProgressState.FAILED, TransferProgressState.ABORTED)
        ]
        if terminal_failure_records:
            return False

        elapsed = now - send_start_time
        recent_records = [record for record in records if now - record.last_progress_time < streaming_idle_timeout]
        if not recent_records:
            return False

        record = max(recent_records, key=lambda item: item.last_progress_time)
        self.log_info(
            fl_ctx,
            f"peer has not read task '{task_name}' after {elapsed:.2f} secs, "
            f"but stream transfer '{record.transfer_id}' has recent activity "
            f"(state={record.state}, bytes_done={record.bytes_done}, items_done={record.items_done}); "
            "continuing to wait",
        )
        return True

    def _mark_task_stream_progress_terminal(self, task_id: str, state: str, job_id: Optional[str] = None):
        if not task_id:
            return

        with self._stream_progress_lock:
            records = list(
                self._stream_progress_tracker.records(
                    job_id="" if job_id is None else str(job_id),
                    task_id=str(task_id),
                    direction=DIRECTION_TASK_PAYLOAD_DOWNLOAD,
                )
            )
            for record in records:
                if not record.terminal:
                    self._stream_progress_tracker.mark_terminal(
                        job_id=record.job_id,
                        task_id=record.task_id,
                        transfer_id=record.transfer_id,
                        direction=record.direction,
                        state=state,
                    )
            self._prune_terminal_stream_progress_records_locked()

    def _mark_all_stream_progress_terminal(self, state: str):
        with self._stream_progress_lock:
            records = list(self._stream_progress_tracker.records(direction=DIRECTION_TASK_PAYLOAD_DOWNLOAD))
            for record in records:
                if not record.terminal:
                    self._stream_progress_tracker.mark_terminal(
                        job_id=record.job_id,
                        task_id=record.task_id,
                        transfer_id=record.transfer_id,
                        direction=record.direction,
                        state=state,
                    )
            self._prune_terminal_stream_progress_records_locked()

    def _should_honor_explicit_peer_read_timeout(self):
        timeout_snapshot = self._get_streaming_timeout_snapshot()
        return self._is_explicit_peer_read_timeout_fast_fail(
            timeout_snapshot.peer_read_timeout_explicit,
            timeout_snapshot.peer_read_timeout,
            timeout_snapshot.streaming_idle_timeout,
        )

    def _log_explicit_peer_read_timeout_warning_once(self, fl_ctx: FLContext):
        timeout_snapshot = self._get_streaming_timeout_snapshot()
        with self._peer_read_timeout_once_lock:
            if self._explicit_peer_read_timeout_warned:
                return
            self._explicit_peer_read_timeout_warned = True
        self.log_warning(
            fl_ctx,
            f"explicit peer_read_timeout ({timeout_snapshot.peer_read_timeout}s) is lower than "
            f"streaming_idle_timeout ({timeout_snapshot.streaming_idle_timeout}s); honoring fast-fail behavior "
            "instead of extending the wait on stream progress",
        )

    def _should_log_clamped_task_send_startup_budget(self):
        timeout_snapshot = self._get_streaming_timeout_snapshot()
        return (
            timeout_snapshot.peer_read_timeout_explicit
            and timeout_snapshot.peer_read_timeout is not None
            and timeout_snapshot.streaming_idle_timeout is not None
            and timeout_snapshot.peer_read_timeout > timeout_snapshot.streaming_idle_timeout
            and (self.pipe is None or isinstance(self.pipe, CellPipe))
        )

    def _log_clamped_task_send_startup_budget_once(self, fl_ctx: FLContext):
        if not self._should_log_clamped_task_send_startup_budget():
            return
        timeout_snapshot = self._get_streaming_timeout_snapshot()
        with self._peer_read_timeout_once_lock:
            if self._task_send_startup_budget_info_logged:
                return
            self._task_send_startup_budget_info_logged = True
        self.log_info(
            fl_ctx,
            f"explicit peer_read_timeout ({timeout_snapshot.peer_read_timeout}s) is higher than streaming_idle_timeout "
            f"({timeout_snapshot.streaming_idle_timeout}s); using streaming_idle_timeout as the no-progress task-send "
            "startup budget",
        )

    def _get_task_send_peer_read_timeout(self):
        """Return the per-send wait timeout used by PipeHandler.

        CellPipe can extend bounded waits through progress callbacks. Other pipe types do not honor
        those callbacks, so an explicitly disabled peer timeout stays `None` and defers to
        PipeHandler.default_request_timeout.
        """
        timeout_snapshot = self._get_streaming_timeout_snapshot()
        peer_read_timeout = timeout_snapshot.peer_read_timeout
        streaming_idle_timeout = timeout_snapshot.streaming_idle_timeout
        if peer_read_timeout is None:
            if streaming_idle_timeout and (self.pipe is None or isinstance(self.pipe, CellPipe)):
                return min(streaming_idle_timeout, STREAM_PROGRESS_COMPLETION_ACK_GRACE)
            return None
        if not streaming_idle_timeout or self._is_explicit_peer_read_timeout_fast_fail(
            timeout_snapshot.peer_read_timeout_explicit, peer_read_timeout, streaming_idle_timeout
        ):
            return peer_read_timeout
        if self.pipe is not None and not isinstance(self.pipe, CellPipe):
            return peer_read_timeout
        return min(peer_read_timeout, streaming_idle_timeout, STREAM_PROGRESS_COMPLETION_ACK_GRACE)

    def _unread_task_send_is_failure(self):
        timeout_snapshot = self._get_streaming_timeout_snapshot()
        peer_read_timeout = timeout_snapshot.peer_read_timeout
        streaming_idle_timeout = timeout_snapshot.streaming_idle_timeout
        if peer_read_timeout is not None:
            return True
        if self.pipe is not None and not isinstance(self.pipe, CellPipe):
            return True
        return bool(streaming_idle_timeout and (self.pipe is None or isinstance(self.pipe, CellPipe)))

    def _send_task_to_peer(self, req: Message, fl_ctx: FLContext, abort_signal: Signal) -> bool:
        job_id = None
        get_header = getattr(req.data, "get_header", None)
        if callable(get_header):
            job_id = get_header(FLMetaKey.JOB_ID)
        job_id = "" if job_id is None else str(job_id)
        send_start_time = time.time()

        def _progress_wait_cb():
            if self._should_honor_explicit_peer_read_timeout():
                self._log_explicit_peer_read_timeout_warning_once(fl_ctx)
                return False

            return self._should_continue_task_send_waiting(
                task_name=req.topic,
                task_id=req.msg_id,
                job_id=job_id,
                send_start_time=send_start_time,
                fl_ctx=fl_ctx,
            )

        req._progress_wait_cb = _progress_wait_cb
        self._log_clamped_task_send_startup_budget_once(fl_ctx)
        return self.pipe_handler.send_to_peer(
            req, timeout=self._get_task_send_peer_read_timeout(), abort_signal=abort_signal
        )

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        """
        The TaskExchanger always sends the Shareable to the peer, and expects to receive a Shareable object from the
        peer. The peer can convert the Shareable object to whatever format that is best for its applications (e.g.
        DXO or FLModel object). Similarly, when submitting result, the peer must convert its result object to a
        Shareable object before sending it back to the TaskExchanger.

        This "late-binding" (binding of the Shareable object to an application-friendly object) strategy makes the
        TaskExchanger generic and can be reused for any applications (e.g. Shareable based, DXO based, or any custom
        data based).
        """
        with self._executing_lock:
            acquired = not self._executing.is_set()
            if acquired:
                self._executing.set()
        try:
            return self._do_execute(task_name, shareable, fl_ctx, abort_signal)
        finally:
            if acquired:
                self._executing.clear()

    def _do_execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        if not self.check_input_shareable(task_name, shareable, fl_ctx):
            self.log_error(fl_ctx, "bad input task shareable")
            return make_reply(ReturnCode.BAD_TASK_DATA)

        job_id = fl_ctx.get_job_id()
        job_id = "" if job_id is None else str(job_id)
        shareable.set_header(FLMetaKey.JOB_ID, job_id)
        shareable.set_header(FLMetaKey.SITE_NAME, fl_ctx.get_identity_name())
        task_id = shareable.get_header(key=FLContextKey.TASK_ID)

        # send to peer
        self.log_info(fl_ctx, f"sending task to peer {self.peer_read_timeout=}")
        req = Message.new_request(topic=task_name, data=shareable, msg_id=task_id)
        start_time = time.time()
        has_been_read = self._send_task_to_peer(req, fl_ctx, abort_signal)
        if not has_been_read:
            if self._unread_task_send_is_failure():
                self._mark_task_stream_progress_terminal(task_id, TransferProgressState.ABORTED, job_id=job_id)
                self.log_error(
                    fl_ctx,
                    f"peer does not accept task '{task_name}' in {time.time() - start_time} secs - aborting task!",
                )
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)
            self.log_info(
                fl_ctx,
                f"peer did not confirm reading task '{task_name}' in {time.time() - start_time} secs; "
                "continuing because peer_read_timeout is disabled",
            )
        else:
            self.log_info(fl_ctx, f"task {task_name} sent to peer in {time.time() - start_time} secs")

        # wait for result
        self.log_debug(fl_ctx, "Waiting for result from peer")
        start = time.time()
        while True:
            if abort_signal.triggered:
                # notify peer that the task is aborted
                self.log_debug(fl_ctx, f"task '{task_name}' is aborted.")
                self._mark_task_stream_progress_terminal(task_id, TransferProgressState.ABORTED, job_id=job_id)
                self.pipe_handler.notify_abort(task_id)
                self.pipe_handler.stop()
                return make_reply(ReturnCode.TASK_ABORTED)

            if self.pipe_handler.asked_to_stop:
                self.log_info(fl_ctx, "task pipe stopped! aborting task.")
                self._mark_task_stream_progress_terminal(task_id, TransferProgressState.ABORTED, job_id=job_id)
                self.pipe_handler.notify_abort(task_id)
                abort_signal.trigger("task pipe stopped!")
                return make_reply(ReturnCode.TASK_ABORTED)

            reply: Optional[Message] = self.pipe_handler.get_next()
            if reply is None:
                if self.task_wait_time and time.time() - start > self.task_wait_time:
                    # timed out
                    self.log_error(fl_ctx, f"task '{task_name}' timeout after {self.task_wait_time} secs")
                    # also tell peer to abort the task
                    self._mark_task_stream_progress_terminal(task_id, TransferProgressState.ABORTED, job_id=job_id)
                    self.pipe_handler.notify_abort(task_id)
                    abort_signal.trigger(f"task '{task_name}' timeout after {self.task_wait_time} secs")
                    return make_reply(ReturnCode.EXECUTION_EXCEPTION)
            elif reply.msg_type != Message.REPLY:
                self.log_warning(
                    fl_ctx, f"ignored reply: '{reply}' (wrong message type) while waiting for the result of {task_name}"
                )
            elif req.topic != reply.topic:
                # ignore wrong topic
                self.log_warning(
                    fl_ctx,
                    f"ignored reply: '{reply}' (reply topic does not match req: '{req}') while waiting for the result of {task_name}",
                )
            elif req.msg_id != reply.req_id:
                self.log_warning(
                    fl_ctx,
                    f"ignored reply: '{reply}' (reply req_id does not match req msg_id: '{req}') while waiting for the result of {task_name}",
                )
            else:
                self.log_info(fl_ctx, f"got result '{reply}' for task '{task_name}'")

                try:
                    result = reply.data
                    if not isinstance(result, Shareable):
                        self._mark_task_stream_progress_terminal(task_id, TransferProgressState.FAILED, job_id=job_id)
                        self.log_error(fl_ctx, f"bad task result from peer: expect Shareable but got {type(result)}")
                        return make_reply(ReturnCode.EXECUTION_EXCEPTION)

                    current_round = shareable.get_header(AppConstants.CURRENT_ROUND)
                    if current_round is not None:
                        result.set_header(AppConstants.CURRENT_ROUND, current_round)

                    if not self.check_output_shareable(task_name, result, fl_ctx):
                        self._mark_task_stream_progress_terminal(task_id, TransferProgressState.FAILED, job_id=job_id)
                        self.log_error(fl_ctx, "bad task result from peer")
                        return make_reply(ReturnCode.EXECUTION_EXCEPTION)

                    self.log_info(fl_ctx, f"received result of {task_name} from peer in {time.time() - start} secs")
                    self._mark_task_stream_progress_terminal(task_id, TransferProgressState.COMPLETED, job_id=job_id)
                    return result
                except Exception as ex:
                    self._mark_task_stream_progress_terminal(task_id, TransferProgressState.FAILED, job_id=job_id)
                    self.log_error(fl_ctx, f"Failed to convert result: {secure_format_exception(ex)}")
                    return make_reply(ReturnCode.EXECUTION_EXCEPTION)
            time.sleep(self.result_poll_interval)

    def check_input_shareable(self, task_name: str, shareable: Shareable, fl_ctx: FLContext) -> bool:
        """Checks input shareable before execute.

        Returns:
            True, if input shareable looks good; False, otherwise.
        """
        return True

    def check_output_shareable(self, task_name: str, shareable: Shareable, fl_ctx: FLContext) -> bool:
        """Checks output shareable after execute.

        Returns:
            True, if output shareable looks good; False, otherwise.
        """
        return True

    def ask_peer_to_end(self, fl_ctx: FLContext) -> bool:
        req = Message.new_request(topic=Topic.END, data="END")
        has_been_read = self.pipe_handler.send_to_peer(req, timeout=self.peer_read_timeout)
        if self.peer_read_timeout and not has_been_read:
            self.log_warning(
                fl_ctx,
                f"3rd party does not read END msg in {self.peer_read_timeout} secs!",
            )
            return False
        return True

    def peer_is_up_or_dead(self) -> bool:
        return self.pipe_handler.peer_is_up_or_dead.is_set()

    def reset_peer_is_up_or_dead(self):
        self.pipe_handler.peer_is_up_or_dead.clear()

    def pause_pipe_handler(self):
        """Stops pipe_handler heartbeat."""
        if self.pipe_handler:
            self.pipe_handler.pause()

    def resume_pipe_handler(self):
        """Resumes pipe_handler heartbeat."""
        if self.pipe_handler:
            self.pipe_handler.resume()

    def get_pipe(self):
        """Gets pipe."""
        return self.pipe

    def get_pipe_channel_name(self):
        """Gets pipe_channel_name."""
        return self.pipe_channel_name
