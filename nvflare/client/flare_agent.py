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

import atexit
import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any, Optional

from nvflare.apis.dxo import DXO, MetaKey, from_shareable
from nvflare.apis.fl_constant import FLContextKey, FLMetaKey
from nvflare.apis.fl_constant import ReturnCode as RC
from nvflare.apis.shareable import Shareable
from nvflare.apis.utils.decomposers import flare_decomposers
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.decomposers import common_decomposers
from nvflare.fuel.f3.streaming.download_service import DownloadService, TransactionDoneStatus
from nvflare.fuel.f3.streaming.transfer_progress import (
    DEFAULT_STREAMING_IDLE_TIMEOUT,
    DIRECTION_RESULT_UPLOAD,
    TransferProgressState,
    TransferProgressTracker,
)
from nvflare.fuel.utils.constants import PipeChannelName
from nvflare.fuel.utils.fobs import FOBSContextKey
from nvflare.fuel.utils.fobs.decomposers.via_downloader import (
    RESULT_UPLOAD_PROGRESS_CTX_KEY,
    RESULT_UPLOAD_TX_CREATED_CB_CTX_KEY,
    ResultUploadProgressContextKey,
    clear_download_initiated,
    get_download_transactions,
    was_download_initiated,
)
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.fuel.utils.pipe.cell_pipe import CellPipe
from nvflare.fuel.utils.pipe.pipe import Message, Mode, Pipe
from nvflare.fuel.utils.pipe.pipe_handler import PipeHandler
from nvflare.private.fed.utils.fed_utils import register_ext_decomposers

STREAM_PROGRESS_COMPLETION_ACK_GRACE = 30.0
_REVERSE_RESULT_UPLOAD_POLL_INTERVAL = 0.5


@dataclass
class _ReverseResultUploadDecision:
    done: bool
    success: bool
    reason: str = ""


class _ReverseResultUploadProgressTracker:
    def __init__(self, *, idle_timeout: float, clock=None):
        if idle_timeout <= 0:
            raise ValueError(f"idle_timeout must be > 0, but got {idle_timeout}")
        self.idle_timeout = float(idle_timeout)
        self.clock = clock or time.time
        self.progress_tracker = TransferProgressTracker(idle_timeout=self.idle_timeout, clock=self.clock)
        self.expected = {}
        self.record_keys = {}
        self.all_success_since = None
        self.lock = threading.Lock()

    def register_transaction(self, tx_id: str, expected_pairs, created_time: Optional[float] = None):
        if created_time is None:
            created_time = self.clock()
        with self.lock:
            for ref_id, receiver_id in expected_pairs:
                normalized_receiver = None if receiver_id is None else str(receiver_id)
                self.expected[(str(tx_id), str(ref_id), normalized_receiver)] = created_time

    def update(
        self,
        *,
        tx_id: Optional[str],
        transfer_id: str,
        receiver_id: Optional[str],
        sequence: int,
        bytes_done: int,
        items_done: Optional[int],
        state: str,
        timestamp: Optional[float] = None,
        job_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ):
        state = self._normalize_state(state)
        transfer_id = str(transfer_id)
        receiver_id = None if receiver_id is None else str(receiver_id)
        with self.lock:
            tx_id = self._normalize_tx_id(tx_id, transfer_id, receiver_id)
            if tx_id is None:
                return False, "unexpected_pair"

            job_id = str(job_id) if job_id is not None else tx_id
            task_id = "" if task_id is None else str(task_id)
            key = (tx_id, transfer_id, receiver_id)
            result = self.progress_tracker.update(
                job_id=job_id,
                task_id=task_id,
                transfer_id=transfer_id,
                direction=DIRECTION_RESULT_UPLOAD,
                receiver_id=receiver_id,
                sequence=sequence,
                bytes_done=bytes_done,
                items_done=items_done,
                state=state,
                transfer_id_kind="download_ref",
                timestamp=timestamp,
            )
            if result.accepted:
                self.record_keys[key] = (job_id, task_id)
                self.all_success_since = None
            return result.accepted, result.reason

    def decide(self, callback_fired: bool = False, callback_status: Optional[str] = None):
        now = self.clock()
        with self.lock:
            if callback_fired:
                if _transaction_status_is_success(callback_status):
                    return _ReverseResultUploadDecision(done=True, success=True, reason="download_complete_cb")
                return _ReverseResultUploadDecision(
                    done=True,
                    success=False,
                    reason=f"download transaction ended with status={callback_status}",
                )

            for key in self.expected:
                record = self._get_record(key)
                if record and self._is_terminal_failure(record):
                    return _ReverseResultUploadDecision(
                        done=True,
                        success=False,
                        reason=f"result_upload transfer {record.transfer_id} ended with state={record.state}",
                    )

            if self.expected and all(
                self._get_record(key) is not None and self._is_terminal_success(self._get_record(key))
                for key in self.expected
            ):
                if self.all_success_since is None:
                    self.all_success_since = now
                    return _ReverseResultUploadDecision(done=False, success=False, reason="completion_grace")
                if now - self.all_success_since >= STREAM_PROGRESS_COMPLETION_ACK_GRACE:
                    return _ReverseResultUploadDecision(done=True, success=True, reason="all_completed")
                return _ReverseResultUploadDecision(done=False, success=False, reason="completion_grace")

            for key, created_time in self.expected.items():
                record = self._get_record(key)
                if record:
                    if record.terminal:
                        continue
                    if self._is_started(record):
                        if (
                            record.last_progress_time is not None
                            and now - record.last_progress_time >= self.idle_timeout
                        ):
                            return _ReverseResultUploadDecision(
                                done=True,
                                success=False,
                                reason=f"result_upload transfer {record.transfer_id} stalled",
                            )
                        continue
                if now - created_time >= self.idle_timeout:
                    _, ref_id, receiver_id = key
                    return _ReverseResultUploadDecision(
                        done=True,
                        success=False,
                        reason=f"result_upload transfer {ref_id} receiver={receiver_id} did not start",
                    )

        return _ReverseResultUploadDecision(done=False, success=False)

    def mark_abandoned(self, timestamp: Optional[float] = None):
        if timestamp is None:
            timestamp = self.clock()
        with self.lock:
            for key in self.expected:
                tx_id, ref_id, receiver_id = key
                record_key = self.record_keys.get(key)
                if record_key is None:
                    job_id, task_id = tx_id, ""
                    self.record_keys[key] = (job_id, task_id)
                else:
                    job_id, task_id = record_key

                record = self._get_record(key)
                if record and record.terminal:
                    continue
                sequence = 0 if record is None else record.sequence + 1
                self.progress_tracker.mark_terminal(
                    job_id=job_id,
                    task_id=task_id,
                    transfer_id=ref_id,
                    direction=DIRECTION_RESULT_UPLOAD,
                    receiver_id=receiver_id,
                    state=TransferProgressState.ABORTED,
                    sequence=sequence,
                    timestamp=timestamp,
                )

    def _normalize_tx_id(self, tx_id: Optional[str], transfer_id: str, receiver_id: Optional[str]):
        if tx_id is not None and (str(tx_id), transfer_id, receiver_id) in self.expected:
            return str(tx_id)
        matches = [
            expected_tx_id
            for expected_tx_id, expected_ref_id, expected_receiver_id in self.expected
            if expected_ref_id == transfer_id and expected_receiver_id == receiver_id
        ]
        if len(matches) == 1:
            return matches[0]
        return None

    def resolve_tx_id(self, tx_id: Optional[str], transfer_id: str, receiver_id: Optional[str]):
        transfer_id = str(transfer_id)
        receiver_id = None if receiver_id is None else str(receiver_id)
        with self.lock:
            return self._normalize_tx_id(tx_id, transfer_id, receiver_id)

    def completion_grace_remaining(self) -> Optional[float]:
        with self.lock:
            if self.all_success_since is None:
                return None
            return max(0.0, STREAM_PROGRESS_COMPLETION_ACK_GRACE - (self.clock() - self.all_success_since))

    @staticmethod
    def _normalize_state(state: str):
        normalized = str(state).lower()
        if normalized in ("completed", "complete", "done", "success", "finished"):
            return TransferProgressState.COMPLETED
        if normalized in ("failed", "fail", "failure", "error", "exception", "timeout", "deleted"):
            return TransferProgressState.FAILED
        if normalized in ("aborted", "abort", "cancelled", "canceled"):
            return TransferProgressState.ABORTED
        return TransferProgressState.ACTIVE

    @staticmethod
    def _is_started(record) -> bool:
        # A zero-byte ACTIVE record is still a real downstream pull signal:
        # DownloadService emits it when the receiver's first request arrives,
        # before the first data chunk is served.  Treat any accepted record as
        # started so the no-start budget is measured from the first pull, not
        # from transaction encoding time.
        return record.started_time is not None

    @staticmethod
    def _is_terminal_success(record) -> bool:
        return record.state == TransferProgressState.COMPLETED

    @staticmethod
    def _is_terminal_failure(record) -> bool:
        return record.state in (TransferProgressState.FAILED, TransferProgressState.ABORTED)

    def _get_record(self, key):
        record_key = self.record_keys.get(key)
        if record_key is None:
            return None
        job_id, task_id = record_key
        _, ref_id, receiver_id = key
        return self.progress_tracker.get_record(
            job_id=job_id,
            task_id=task_id,
            transfer_id=ref_id,
            direction=DIRECTION_RESULT_UPLOAD,
            receiver_id=receiver_id,
        )


def _transaction_status_is_success(status) -> bool:
    return str(status).lower() == TransactionDoneStatus.FINISHED


class FlareAgentException(Exception):
    pass


class AgentClosed(FlareAgentException):
    pass


class CallStateError(FlareAgentException):
    pass


class Task:
    def __init__(self, task_name: str, task_id: str, data):
        self.task_name = task_name
        self.task_id = task_id
        self.data = data

    def __str__(self):
        return f"'{self.task_name} {self.task_id}'"


class _TaskContext:
    def __init__(self, task_id, task_name: str, msg_id, result_receiver_ids=None):
        self.task_id = task_id
        self.task_name = task_name
        self.msg_id = msg_id
        self.result_receiver_ids = result_receiver_ids


class FlareAgent:
    def __init__(
        self,
        pipe: Optional[Pipe] = None,
        read_interval=0.1,
        heartbeat_interval=5.0,
        heartbeat_timeout=60.0,
        resend_interval=2.0,
        max_resends=None,
        submit_result_timeout=60.0,
        metric_pipe: Optional[Pipe] = None,
        task_channel_name: str = PipeChannelName.TASK,
        metric_channel_name: str = PipeChannelName.METRIC,
        close_pipe: bool = True,
        close_metric_pipe: bool = True,
        decomposer_module: str = None,
        download_complete_timeout: float = DownloadService.FINISHED_REFS_TTL,
        streaming_idle_timeout: float = DEFAULT_STREAMING_IDLE_TIMEOUT,
        launch_once: bool = False,
    ):
        """Constructor of Flare Agent.

        The agent is responsible for communicating with the Flare Client Job cell (CJ)
        to get task and to submit task result.

        Args:
            pipe (Pipe): pipe for task communication.
            read_interval (float): how often to read from the pipe. Defaults to 0.1.
            heartbeat_interval (float): how often to send a heartbeat to the peer. Defaults to 5.0.
            heartbeat_timeout (float): how long to wait for a heartbeat from the peer before treating the peer as dead,
                0 means DO NOT check for heartbeat. Defaults to 30.0.
            resend_interval (float): how often to resend a message if failing to send. None means no resend.
                Note that if the pipe does not support resending, then no resend. Defaults to 2.0.
            max_resends (int, optional): max number of resend. None means no limit. Defaults to None.
            submit_result_timeout (float): when submitting task result,
                how long to wait for response from the CJ. Defaults to 30.0.
            metric_pipe (Pipe, optional): pipe for metric communication. Defaults to None.
            task_channel_name (str): channel name for task. Defaults to ``task``.
            metric_channel_name (str): channel name for metric. Defaults to ``metric``.
            close_pipe (bool): whether to close the task pipe when stopped. Defaults to True.
                Usually for ``FilePipe`` we set to False, for ``CellPipe`` we set to True.
            close_metric_pipe (bool): whether to close the metric pipe when stopped. Defaults to True.
                Usually for ``FilePipe`` we set to False, for ``CellPipe`` we set to True.
            decomposer_module (str): the module name which contains the external decomposers.
            download_complete_timeout (float): how long to wait after send_to_peer() ACKs for the
                server to finish downloading tensors from this subprocess's DownloadService.
                Only active when pipe is a CellPipe with pass_through_on_send=True.
                Defaults to DownloadService.FINISHED_REFS_TTL.
            streaming_idle_timeout (float): idle timeout for progress-aware reverse result_upload waiting.
                Defaults to 600.0.
        """
        if pipe is None and metric_pipe is None:
            raise RuntimeError(
                "Please configure at least one pipe. Both the task pipe and the metric pipe are set to None."
            )
        flare_decomposers.register()
        common_decomposers.register()
        if decomposer_module:
            register_ext_decomposers(decomposer_module)

        self.logger = get_obj_logger(self)
        self.pipe = pipe
        self.pipe_handler = None
        if self.pipe:
            self.pipe_handler = PipeHandler(
                pipe=self.pipe,
                read_interval=read_interval,
                heartbeat_interval=heartbeat_interval,
                heartbeat_timeout=heartbeat_timeout,
                resend_interval=resend_interval,
                max_resends=max_resends,
            )
        self.submit_result_timeout = submit_result_timeout
        self.task_channel_name = task_channel_name
        self.metric_channel_name = metric_channel_name

        self.metric_pipe = metric_pipe
        self.metric_pipe_handler = None
        if self.metric_pipe:
            self.metric_pipe_handler = PipeHandler(
                pipe=self.metric_pipe,
                read_interval=read_interval,
                heartbeat_interval=heartbeat_interval,
                heartbeat_timeout=heartbeat_timeout,
                resend_interval=resend_interval,
                max_resends=max_resends,
            )

        self.current_task = None
        self.task_lock = threading.Lock()
        self.asked_to_stop = False
        self._close_pipe = close_pipe
        self._close_metric_pipe = close_metric_pipe
        self._download_complete_timeout = download_complete_timeout
        self._streaming_idle_timeout = streaming_idle_timeout
        self._result_upload_poll_interval = _REVERSE_RESULT_UPLOAD_POLL_INTERVAL
        self._launch_once = launch_once

    def start(self):
        """Start the agent.

        This method must be called to enable CJ/Agent communication.

        Returns: None

        """
        if self.pipe:
            self.pipe.open(self.task_channel_name)
            self.pipe_handler.set_status_cb(
                self._status_cb, pipe_handler=self.pipe_handler, channel=self.task_channel_name
            )
            self.pipe_handler.start()

        if self.metric_pipe:
            self.metric_pipe.open(self.metric_channel_name)
            self.metric_pipe_handler.set_status_cb(
                self._metrics_status_cb, pipe_handler=self.metric_pipe_handler, channel=self.metric_channel_name
            )
            self.metric_pipe_handler.start()

    def _status_cb(self, msg: Message, pipe_handler: PipeHandler, channel):
        self.logger.info(f"{channel} pipe status changed to {msg.topic}: {msg.data}")
        self.asked_to_stop = True
        pipe_handler.stop(self._close_pipe)

    def _metrics_status_cb(self, msg: Message, pipe_handler: PipeHandler, channel):
        self.logger.info(f"{channel} pipe status changed to {msg.topic}: {msg.data}")
        self.asked_to_stop = True
        pipe_handler.stop(self._close_metric_pipe)

    def stop(self):
        """Stop the agent.

        After this is called, there will be no more communications between CJ and agent.

        Returns: None

        """
        self.asked_to_stop = True
        if self.pipe_handler:
            self.pipe_handler.stop(self._close_pipe)
        if self.metric_pipe_handler:
            self.metric_pipe_handler.stop(self._close_metric_pipe)

    def shareable_to_task_data(self, shareable: Shareable) -> Any:
        """Convert the Shareable object received from the TaskExchanger to an app-friendly format.

        Subclass can override this method to convert to its own app-friendly task data.
        By default, we convert to DXO object.

        Args:
            shareable: the Shareable object received from the TaskExchanger.

        Returns:
            task data.
        """
        try:
            dxo = from_shareable(shareable)

            # add training-related headers carried in the Shareable header to the DXO meta.
            total_rounds = shareable.get_header(AppConstants.NUM_ROUNDS)
            if total_rounds is not None:
                dxo.set_meta_prop(MetaKey.TOTAL_ROUNDS, total_rounds)
            current_round = shareable.get_header(AppConstants.CURRENT_ROUND)
            if current_round is not None:
                dxo.set_meta_prop(MetaKey.CURRENT_ROUND, current_round)
            return dxo
        except Exception as ex:
            self.logger.error(f"failed to extract DXO from shareable object: {ex}")
            raise ex

    @staticmethod
    def _normalize_result_receiver_ids(receiver_ids):
        if receiver_ids is None:
            return None
        if isinstance(receiver_ids, str):
            values = [receiver_ids]
        else:
            try:
                values = list(receiver_ids)
            except TypeError:
                return None

        normalized = tuple(str(receiver_id) for receiver_id in values if receiver_id is not None and str(receiver_id))
        return normalized or None

    def get_task(self, timeout: Optional[float] = None) -> Optional[Task]:
        """Get a task from FLARE. This is a blocking call.

        Args:
            timeout (float, optional): If specified, this call is blocked only for the specified amount of time.
                If not specified, this call is blocked forever until a task has been received or agent has been closed.

        Returns:
            None if no task is available before timeout; or a Task object if task is available.

        Raises:
            AgentClosed exception if the agent has been closed before timeout.
            CallStateError exception if the call has not been made properly.
            AgentAbortException: If the other endpoint of the pipe requests to abort.
            AgentEndException: If the other endpoint has ended.
            AgentPeerGoneException: If the other endpoint is gone.

        Note: the application must make the call only when it is just started or after a previous task's result
        has been submitted.

        """
        if not self.pipe_handler:
            raise RuntimeError("task pipe is not available")
        start_time = time.time()
        while True:
            if self.asked_to_stop:
                raise AgentClosed("agent closed")

            if self.current_task:
                raise CallStateError("application called get_task while the current task is not processed")

            if timeout is not None and time.time() - start_time >= timeout:
                self.logger.debug("get request timeout")
                return None

            req: Optional[Message] = self.pipe_handler.get_next()
            if req is not None:
                if not isinstance(req.data, Shareable):
                    self.logger.error(f"bad task: expect request data to be Shareable but got {type(req.data)}")
                    raise RuntimeError("bad request data")

                shareable = req.data
                task_data = self.shareable_to_task_data(shareable)
                task_id = shareable.get_header(FLContextKey.TASK_ID)
                task_name = shareable.get_header(FLContextKey.TASK_NAME)
                result_receiver_ids = self._normalize_result_receiver_ids(
                    shareable.get_header(FOBSContextKey.RECEIVER_IDS)
                )

                tc = _TaskContext(
                    task_id=task_id,
                    task_name=task_name,
                    msg_id=req.msg_id,
                    result_receiver_ids=result_receiver_ids,
                )
                self.current_task = tc
                return Task(task_name=tc.task_name, task_id=tc.task_id, data=task_data)
            time.sleep(0.5)

    def submit_result(self, result, rc=RC.OK) -> bool:
        """Submit the result of the current task.

        This is a blocking call. The agent will try to send the result to flare site until it is successfully sent or
        the task is aborted or the agent is closed.

        Args:
            result: result to be submitted
            rc: return code

        Returns:
            whether the result is submitted successfully

        Raises:
            the CallStateError exception if the submit_result call is not made properly.

        Notes: the application must only make this call after the received task is processed. The call can only be
        made a single time regardless whether the submission is successful.

        """
        if not self.pipe_handler:
            raise RuntimeError("task pipe is not available")
        with self.task_lock:
            current_task = self.current_task
            if not current_task:
                self.logger.error("submit_result is called but there is no current task!")
                return False

        try:
            result = self._do_submit_result(current_task, result, rc)
        except Exception as ex:
            self.logger.error(f"exception submitting result to {current_task.sender}: {ex}")
            traceback.print_exc()
            result = False

        with self.task_lock:
            self.current_task = None

        return result

    def task_result_to_shareable(self, result: Any, rc) -> Shareable:
        """Convert the result object to Shareable object before sending back to the TaskExchanger.

        Subclass can override this method to convert its app-friendly result type to Shareable.
        By default, we expect the result to be DXO object.

        Args:
            result: the result object to be converted to Shareable.
                If None, an empty Shareable object will be created with the rc only.
            rc: the return code.

        Returns:
            A Shareable object
        """
        if result is not None:
            if not isinstance(result, DXO):
                self.logger.error(f"expect result to be DXO but got {type(result)}")
                raise RuntimeError("bad result data")
            result = result.to_shareable()
        else:
            result = Shareable()
        result.set_return_code(rc)
        return result

    @staticmethod
    def _get_fobs_context_value(cell, key):
        get_fobs_context = getattr(cell, "get_fobs_context", None)
        if not callable(get_fobs_context):
            return None
        ctx = get_fobs_context()
        if not isinstance(ctx, dict):
            return None
        return ctx.get(key)

    def _make_reverse_result_upload_tracker(self):
        return _ReverseResultUploadProgressTracker(
            idle_timeout=getattr(self, "_streaming_idle_timeout", DEFAULT_STREAMING_IDLE_TIMEOUT)
        )

    def _register_download_transactions(self, tracker: _ReverseResultUploadProgressTracker, transactions=None):
        transactions = get_download_transactions() if transactions is None else transactions
        for transaction in transactions:
            tracker.register_transaction(
                tx_id=transaction.tx_id,
                expected_pairs=transaction.expected_pairs,
                created_time=transaction.created_time,
            )
        return transactions

    def _wait_for_download_complete_fixed(self, download_done, download_status, wait_start):
        # Legacy compatibility path: timeout warns but still returns success so
        # progress-untrackable result uploads keep the pre-progress behavior.
        if download_done.wait(timeout=self._download_complete_timeout):
            download_elapsed = time.time() - wait_start
            ds = download_status[0]
            if _transaction_status_is_success(ds):
                self.logger.info(f"[subprocess] server download complete: elapsed={download_elapsed:.2f}s")
            else:
                self.logger.warning(
                    f"[subprocess] download transaction ended with status={ds} after {download_elapsed:.2f}s"
                )
        else:
            self.logger.warning(
                f"[subprocess] download not signalled within {self._download_complete_timeout}s; "
                "proceeding (server may still be downloading from this process)"
            )
        return True

    def _fail_reverse_download_transactions(self, tracker, transactions):
        tracker.mark_abandoned()
        for transaction in transactions or ():
            try:
                DownloadService.delete_transaction(transaction.tx_id)
            except Exception as ex:
                self.logger.warning(
                    f"[subprocess] failed to delete abandoned result_upload transaction " f"{transaction.tx_id}: {ex}"
                )

    def _wait_for_reverse_result_upload(
        self, tracker, progress_event, download_done, download_status, wait_start, transactions=()
    ):
        while True:
            abandon_reason = self._get_reverse_result_upload_abandon_reason()
            if abandon_reason:
                self.logger.warning(f"[subprocess] abandoning result_upload wait: {abandon_reason}")
                self._fail_reverse_download_transactions(tracker, transactions)
                return False

            decision = tracker.decide(callback_fired=download_done.is_set(), callback_status=download_status[0])
            if decision.done:
                elapsed = tracker.clock() - wait_start
                if decision.success:
                    self.logger.info(
                        f"[subprocess] server download complete: elapsed={elapsed:.2f}s reason={decision.reason}"
                    )
                else:
                    self.logger.warning(
                        f"[subprocess] result_upload progress-aware wait failed after {elapsed:.2f}s: "
                        f"{decision.reason}"
                    )
                    self._fail_reverse_download_transactions(tracker, transactions)
                return decision.success

            wait_timeout = getattr(self, "_result_upload_poll_interval", _REVERSE_RESULT_UPLOAD_POLL_INTERVAL)
            if decision.reason == "completion_grace":
                remaining_grace = tracker.completion_grace_remaining()
                if remaining_grace is not None:
                    wait_timeout = remaining_grace
            progress_event.wait(timeout=wait_timeout)
            progress_event.clear()

    def _get_reverse_result_upload_abandon_reason(self):
        if self.asked_to_stop:
            return "agent is stopping"
        if self.pipe_handler and self.pipe_handler.asked_to_stop:
            return "task pipe handler is stopping"
        if getattr(self.pipe, "closed", False):
            return "task pipe is closed"
        return None

    def _update_reverse_result_upload_progress(self, tracker, progress_event, **kwargs):
        if kwargs.get("direction") != DIRECTION_RESULT_UPLOAD:
            return
        tx_id = kwargs.get("tx_id")
        receiver_id = None if kwargs.get("receiver_id") is None else str(kwargs.get("receiver_id"))
        tx_log_id = tx_id or "<unknown>"

        try:
            transfer_id = str(kwargs.get("transfer_id") or kwargs["ref_id"])
            sequence = int(kwargs.get("sequence", 0))
            bytes_done = int(kwargs.get("bytes_done", 0))
            items_done = kwargs.get("items_done")
            if items_done is not None:
                items_done = int(items_done)
        except (KeyError, TypeError, ValueError):
            self.logger.warning(f"[subprocess] ignored invalid result_upload progress event: {kwargs}")
            return

        accepted, reason = tracker.update(
            tx_id=tx_id,
            transfer_id=transfer_id,
            receiver_id=receiver_id,
            sequence=sequence,
            bytes_done=bytes_done,
            items_done=items_done,
            state=kwargs.get("state", TransferProgressState.ACTIVE),
            timestamp=kwargs.get("timestamp"),
            job_id=kwargs.get("job_id"),
            task_id=kwargs.get("task_id"),
        )
        if accepted:
            if not tx_id:
                tx_log_id = tracker.resolve_tx_id(tx_id, transfer_id, receiver_id) or tx_log_id
            self.logger.info(
                f"[subprocess] result_upload progress tx={tx_log_id} task={kwargs.get('task_id')} "
                f"transfer={transfer_id} receiver={receiver_id} state={kwargs.get('state')} "
                f"sequence={sequence} bytes_done={bytes_done} items_done={items_done}"
            )
        else:
            msg = (
                f"[subprocess] ignored result_upload progress tx={tx_log_id} "
                f"transfer={transfer_id} receiver={receiver_id}: {reason}"
            )
            if reason == "unexpected_pair":
                self.logger.warning(msg)
            else:
                self.logger.debug(msg)
        progress_event.set()

    def _do_submit_result(self, current_task: _TaskContext, result, rc):
        result_shareable = self.task_result_to_shareable(result, rc)
        reply = Message.new_reply(topic=current_task.task_name, req_msg_id=current_task.msg_id, data=result_shareable)

        # Gate subprocess exit on reverse result_upload progress for the PASS_THROUGH path
        # (subprocess → CJ → server).  CJ ACKs send_to_peer() immediately after creating
        # LazyDownloadRef objects; the server then downloads tensors asynchronously from
        # this subprocess's DownloadService.  Registering DOWNLOAD_COMPLETE_CB before
        # serialisation ensures _create_downloader() wires it as the transaction_done_cb,
        # while the local progress callback keeps this subprocess waiting only while
        # expected DownloadService refs continue making monotonic progress.
        #
        # For validate results (metrics only, no tensors), _finalize_download_tx() creates
        # no download transaction and never fires DOWNLOAD_COMPLETE_CB.  We detect this via
        # was_download_initiated() (thread-local set by _finalize_download_tx()) and return
        # immediately without waiting — fixing the 1800s hang on CSE round 2+ (RC12 Bug 3).
        if isinstance(self.pipe, CellPipe) and self.pipe.pass_through_on_send:
            download_done = threading.Event()
            progress_event = threading.Event()
            download_status = [None]
            result_upload_tracker = self._make_reverse_result_upload_tracker()
            result_upload_transactions = []

            def _on_download_done(tid, status, objs):
                download_status[0] = status
                download_done.set()
                progress_event.set()

            def _on_result_upload_progress(**kwargs):
                self._update_reverse_result_upload_progress(result_upload_tracker, progress_event, **kwargs)

            def _on_download_transaction_created(transaction):
                result_upload_tracker.register_transaction(
                    tx_id=transaction.tx_id,
                    expected_pairs=transaction.expected_pairs,
                    created_time=transaction.created_time,
                )
                result_upload_transactions.append(transaction)

            previous_stream_progress_cb = self._get_fobs_context_value(
                self.pipe.cell, FOBSContextKey.STREAM_PROGRESS_CB
            )
            previous_receiver_ids = self._get_fobs_context_value(self.pipe.cell, FOBSContextKey.RECEIVER_IDS)
            previous_tx_created_cb = self._get_fobs_context_value(self.pipe.cell, RESULT_UPLOAD_TX_CREATED_CB_CTX_KEY)
            streaming_idle_timeout = getattr(self, "_streaming_idle_timeout", DEFAULT_STREAMING_IDLE_TIMEOUT)
            result_upload_progress_context = {
                ResultUploadProgressContextKey.JOB_ID: result_shareable.get_header(FLMetaKey.JOB_ID) or "",
                ResultUploadProgressContextKey.TASK_ID: current_task.task_id or current_task.msg_id,
                ResultUploadProgressContextKey.STREAMING_IDLE_TIMEOUT: streaming_idle_timeout,
            }
            result_receiver_ids = getattr(current_task, "result_receiver_ids", None)
            self.pipe.cell.update_fobs_context(
                {
                    FOBSContextKey.DOWNLOAD_COMPLETE_CB: _on_download_done,
                    FOBSContextKey.STREAM_PROGRESS_CB: _on_result_upload_progress,
                    FOBSContextKey.RECEIVER_IDS: result_receiver_ids,
                    RESULT_UPLOAD_PROGRESS_CTX_KEY: result_upload_progress_context,
                    RESULT_UPLOAD_TX_CREATED_CB_CTX_KEY: _on_download_transaction_created,
                }
            )
            # Preserve the legacy message-root TTL for fallback paths where reverse
            # progress tracking cannot be installed.  The ViaDownloader decomposer
            # switches progress-trackable transactions to streaming_idle_timeout.
            reply._dl_ttl = self._download_complete_timeout
            reply._receiver_ids = result_receiver_ids
            # Reset thread-local so a stale True from a previous training round does not
            # carry over to the current validate round (no tensors → False expected).
            clear_download_initiated()
            try:
                send_start = time.time()
                sent = self.pipe_handler.send_to_peer(reply, self.submit_result_timeout)
                if not sent:
                    self.logger.warning(
                        f"[subprocess] send_to_peer failed: task_ph.asked_to_stop={self.pipe_handler.asked_to_stop}"
                    )
                    return False
                send_elapsed = time.time() - send_start

                # _finalize_download_tx() runs synchronously inside send_to_peer().
                # was_download_initiated() is True iff it created a download transaction
                # (i.e. the result contained large tensors requiring via-downloader transfer).
                # False means validate result (metrics only) — skip the download wait and
                # fall through to the launch_once shutdown block below.
                transactions = tuple(result_upload_transactions)
                download_initiated = bool(transactions) or was_download_initiated()
                if download_initiated:
                    if not transactions:
                        transactions = self._register_download_transactions(result_upload_tracker)
                    self.logger.info(
                        f"[subprocess] result ACK'd by CJ in {send_elapsed:.2f}s; " "waiting for server tensor download"
                    )
                    if transactions:
                        wait_start = result_upload_tracker.clock()
                        result_ok = self._wait_for_reverse_result_upload(
                            result_upload_tracker,
                            progress_event,
                            download_done,
                            download_status,
                            wait_start,
                            transactions=transactions,
                        )
                    else:
                        self.logger.info(
                            "[subprocess] result_upload progress tracking unavailable; "
                            f"falling back to download_complete_timeout={self._download_complete_timeout}s"
                        )
                        wait_start = time.time()
                        result_ok = self._wait_for_download_complete_fixed(download_done, download_status, wait_start)
                    if not result_ok:
                        return False
                else:
                    self.logger.info(
                        f"[subprocess] result ACK'd by CJ in {send_elapsed:.2f}s; "
                        "no tensors in result — proceeding immediately"
                    )
            finally:
                # Always clear the callback so stale refs do not accumulate across rounds.
                self.pipe.cell.update_fobs_context(
                    {
                        FOBSContextKey.DOWNLOAD_COMPLETE_CB: None,
                        FOBSContextKey.STREAM_PROGRESS_CB: previous_stream_progress_cb,
                        FOBSContextKey.RECEIVER_IDS: previous_receiver_ids,
                        RESULT_UPLOAD_PROGRESS_CTX_KEY: None,
                        RESULT_UPLOAD_TX_CREATED_CB_CTX_KEY: previous_tx_created_cb,
                    }
                )
            if self._launch_once:
                # launch_once=True: subprocess handles multiple rounds; do NOT exit here.
                # Register atexit once so os._exit(0) is called when main() finally returns,
                # bypassing Python's thread-join wait on non-daemon CoreCell threads.
                if not getattr(self, "_atexit_registered", False):
                    atexit.register(os._exit, 0)
                    self._atexit_registered = True
                return True
            else:
                # launch_once=False: subprocess handles exactly one round; exit now so the
                # deferred-stop poller on the CJ side unblocks immediately.
                self.logger.info("[subprocess] exiting after server download")
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(0)
                return True

        return self.pipe_handler.send_to_peer(reply, self.submit_result_timeout)

    def log(self, record: DXO) -> bool:
        """Logs a metric record.

        Args:
            record (DXO): A metric record.

        Returns:
            whether the metric record is submitted successfully
        """
        if not self.metric_pipe_handler:
            raise RuntimeError("metric pipe is not available")

        msg = Message.new_request(topic="metric", data=record)
        return self.metric_pipe_handler.send_to_peer(msg, self.submit_result_timeout)


class FlareAgentWithCellPipe(FlareAgent):
    def __init__(
        self,
        agent_id: str,
        site_name: str,
        root_url: str,
        secure_mode: bool,
        workspace_dir: str,
        read_interval=0.1,
        heartbeat_interval=5.0,
        heartbeat_timeout=60.0,  # increased from 30.0 — 30s too tight under large-model GC/relay
        resend_interval=2.0,
        max_resends=None,
        submit_result_timeout=60.0,  # increased from 30.0 — gives CJ enough time to ACK under load
        has_metrics=False,
        download_complete_timeout=DownloadService.FINISHED_REFS_TTL,
        streaming_idle_timeout=DEFAULT_STREAMING_IDLE_TIMEOUT,
        launch_once: bool = False,
    ):
        """Constructor of Flare Agent with Cell Pipe. This is a convenient class.

        Args:
            agent_id (str): unique id to guarantee the uniqueness of cell's FQCN.
            site_name (str): name of the FLARE site
            root_url (str): the root url of the cellnet that the pipe's cell will join
            secure_mode (bool): whether connection to the root is secure (TLS)
            workspace_dir (str): the directory that contains startup for joining the cellnet. Required only in secure mode
            read_interval (float): how often to read from the pipe.
            heartbeat_interval (float): how often to send a heartbeat to the peer.
            heartbeat_timeout (float): how long to wait for a heartbeat from the peer before treating the peer as gone,
                0 means DO NOT check for heartbeat. Defaults to 60.0.
            resend_interval (float): how often to resend a message if failing to send. None means no resend.
                Note that if the pipe does not support resending, then no resend.
            max_resends (int, optional): max number of resend. None means no limit.
            submit_result_timeout (float): when submitting task result, how long to wait for response from the CJ.
                Defaults to 60.0.
            has_metrics (bool): has metric pipe or not.
            download_complete_timeout (float): how long to wait after send_to_peer() ACKs for the server to finish
                downloading tensors from this subprocess's DownloadService.
                Defaults to DownloadService.FINISHED_REFS_TTL.
            streaming_idle_timeout (float): idle timeout for progress-aware reverse result_upload waiting.
        """
        pipe = CellPipe(
            mode=Mode.ACTIVE,
            token=agent_id,
            site_name=site_name,
            root_url=root_url,
            secure_mode=secure_mode,
            workspace_dir=workspace_dir,
        )

        metric_pipe = None
        if has_metrics:
            metric_pipe = CellPipe(
                mode=Mode.ACTIVE,
                token=agent_id,
                site_name=site_name,
                root_url=root_url,
                secure_mode=secure_mode,
                workspace_dir=workspace_dir,
            )

        super().__init__(
            pipe,
            read_interval=read_interval,
            heartbeat_interval=heartbeat_interval,
            heartbeat_timeout=heartbeat_timeout,
            resend_interval=resend_interval,
            max_resends=max_resends,
            submit_result_timeout=submit_result_timeout,
            metric_pipe=metric_pipe,
            download_complete_timeout=download_complete_timeout,
            streaming_idle_timeout=streaming_idle_timeout,
            launch_once=launch_once,
        )
