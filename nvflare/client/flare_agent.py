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

import os
import threading
import time
import traceback
from typing import Any, Optional

from nvflare.apis.dxo import DXO, MetaKey, from_shareable
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_constant import ReturnCode as RC
from nvflare.apis.shareable import Shareable
from nvflare.apis.utils.decomposers import flare_decomposers
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.decomposers import common_decomposers
from nvflare.fuel.f3.streaming.download_service import TransactionDoneStatus
from nvflare.fuel.utils.constants import PipeChannelName
from nvflare.fuel.utils.fobs import FOBSContextKey
from nvflare.fuel.utils.fobs.decomposers.via_downloader import clear_download_initiated, was_download_initiated
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.fuel.utils.pipe.cell_pipe import CellPipe
from nvflare.fuel.utils.pipe.pipe import Message, Mode, Pipe
from nvflare.fuel.utils.pipe.pipe_handler import PipeHandler
from nvflare.private.fed.utils.fed_utils import register_ext_decomposers


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
    def __init__(self, task_id, task_name: str, msg_id):
        self.task_id = task_id
        self.task_name = task_name
        self.msg_id = msg_id


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
        download_complete_timeout: float = 1800.0,
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
                Defaults to 1800.0.
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

                tc = _TaskContext(
                    task_id=task_id,
                    task_name=task_name,
                    msg_id=req.msg_id,
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
            self.logger.info(
                f"[subprocess] submit_result: task_ph.asked_to_stop={self.pipe_handler.asked_to_stop if self.pipe_handler else 'N/A'}"
                f" agent.asked_to_stop={self.asked_to_stop}"
            )
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

    def _do_submit_result(self, current_task: _TaskContext, result, rc):
        result_shareable = self.task_result_to_shareable(result, rc)
        reply = Message.new_reply(topic=current_task.task_name, req_msg_id=current_task.msg_id, data=result_shareable)

        # Gate subprocess exit on download completion for the reverse PASS_THROUGH path
        # (subprocess → CJ → server).  CJ ACKs send_to_peer() immediately after creating
        # LazyDownloadRef objects; the server then downloads tensors asynchronously from
        # this subprocess's DownloadService.  Registering DOWNLOAD_COMPLETE_CB before
        # serialisation ensures _create_downloader() wires it as the transaction_done_cb,
        # so the event is set exactly when the last receiver finishes downloading.
        #
        # For validate results (metrics only, no tensors), _finalize_download_tx() creates
        # no download transaction and never fires DOWNLOAD_COMPLETE_CB.  We detect this via
        # was_download_initiated() (thread-local set by _finalize_download_tx()) and return
        # immediately without waiting — fixing the 1800s hang on CSE round 2+ (RC12 Bug 3).
        if isinstance(self.pipe, CellPipe) and self.pipe.pass_through_on_send:
            download_done = threading.Event()
            download_status = [None]

            def _on_download_done(tid, status, objs):
                download_status[0] = status
                download_done.set()

            self.pipe.cell.update_fobs_context({FOBSContextKey.DOWNLOAD_COMPLETE_CB: _on_download_done})
            # Tell cell_pipe.py to use download_complete_timeout as MSG_ROOT_TTL so the
            # subprocess's DownloadService transaction stays alive long enough for the server
            # to finish pulling tensors.  submit_result_timeout is the CJ-ACK timeout and is
            # unrelated to transfer duration — using it here would kill the transaction too early.
            reply._dl_ttl = self._download_complete_timeout
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
                # False means validate result (metrics only) — proceed immediately.
                if not was_download_initiated():
                    self.logger.info(
                        f"[subprocess] result ACK'd by CJ in {send_elapsed:.2f}s; "
                        "no tensors in result — proceeding immediately"
                    )
                    return True

                self.logger.info(
                    f"[subprocess] result ACK'd by CJ in {send_elapsed:.2f}s; "
                    f"waiting up to {self._download_complete_timeout}s for server tensor download"
                )
                wait_start = time.time()
                if download_done.wait(timeout=self._download_complete_timeout):
                    download_elapsed = time.time() - wait_start
                    ds = download_status[0]
                    if ds == TransactionDoneStatus.FINISHED:
                        self.logger.info(f"[subprocess] server download complete: elapsed={download_elapsed:.2f}s")
                    else:
                        self.logger.warning(
                            f"[subprocess] download transaction ended with status={ds} "
                            f"after {download_elapsed:.2f}s"
                        )
                else:
                    self.logger.warning(
                        f"[subprocess] download not signalled within {self._download_complete_timeout}s; "
                        "proceeding (server may still be downloading from this process)"
                    )
            finally:
                # Always clear the callback so stale refs do not accumulate across rounds.
                self.pipe.cell.update_fobs_context({FOBSContextKey.DOWNLOAD_COMPLETE_CB: None})
            # Server download is complete (or timed out). The subprocess has no further work
            # to do; exit immediately so the deferred-stop poller on the CJ side unblocks.
            # os._exit() bypasses Python's thread-join wait, which would otherwise block
            # forever on the non-daemon CoreCell network threads.
            self.logger.info("[subprocess] exiting after server download")
            import sys
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)

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
        download_complete_timeout=1800.0,  # new — gate subprocess exit until server finishes tensor download
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
                downloading tensors from this subprocess's DownloadService. Defaults to 1800.0.
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
        )
