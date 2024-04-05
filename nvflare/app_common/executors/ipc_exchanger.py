# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Union

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.client.ipc import defs
from nvflare.fuel.f3.cellnet.cell import Cell, Message, MessageHeaderKey
from nvflare.fuel.f3.cellnet.cell import ReturnCode as CellReturnCode
from nvflare.fuel.f3.cellnet.utils import make_reply as make_cell_reply
from nvflare.security.logging import secure_format_traceback

_SHORT_SLEEP_TIME = 0.2


class _TaskContext:
    def __init__(self, task_name: str, task_id: str, fl_ctx: FLContext):
        self.task_id = task_id
        self.task_name = task_name
        self.fl_ctx = fl_ctx
        self.send_rc = None
        self.result_rc = None
        self.result_error = None
        self.result = None
        self.result_received_time = None
        self.result_waiter = threading.Event()

    def __str__(self):
        return f"'{self.task_name} {self.task_id}'"


class IPCExchanger(Executor):
    def __init__(
        self,
        send_task_timeout=5.0,
        resend_task_interval=2.0,
        agent_connection_timeout=60.0,
        agent_heartbeat_timeout=None,
        agent_heartbeat_interval=5.0,
        agent_ack_timeout=5.0,
        agent_id=None,
    ):
        """Constructor of IPCExchanger

        Args:
            send_task_timeout: when sending task to Agent, how long to wait for response
            resend_task_interval: when failed to send task to agent, how often to resend
            agent_heartbeat_timeout: time allowed to miss heartbeat ack from agent before stopping
            agent_connection_timeout: time allowed to miss heartbeat ack from agent for considering it disconnected
            agent_heartbeat_interval: how often to send heartbeats to the agent
            agent_ack_timeout: how long to wait for agent ack (for heartbeat and bye messages)
            agent_id: the unique ID of the agent. If not specified, will get it from job's meta
        """
        Executor.__init__(self)
        self.flare_agent_fqcn = None
        self.agent_ack_timeout = agent_ack_timeout
        self.agent_heartbeat_interval = agent_heartbeat_interval
        self.agent_heartbeat_timeout = agent_heartbeat_timeout
        self.agent_connection_timeout = agent_connection_timeout
        self.send_task_timeout = send_task_timeout
        self.resend_task_interval = resend_task_interval
        self.agent_id = agent_id
        self.last_agent_ack_time = time.time()
        self.engine = None
        self.cell = None
        self.is_done = False
        self.is_connected = False
        self.task_ctx = None

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.engine = fl_ctx.get_engine()
            self.cell = self.engine.get_cell()

            self.cell.register_request_cb(
                channel=defs.CHANNEL,
                topic=defs.TOPIC_SUBMIT_RESULT,
                cb=self._receive_result,
            )

            # get meta
            if not self.agent_id:
                agent_id = None
                meta = fl_ctx.get_prop(FLContextKey.JOB_META)
                if isinstance(meta, dict):
                    agent_id = meta.get(defs.JOB_META_KEY_AGENT_ID)

                if not agent_id:
                    self.system_panic(reason=f"missing {defs.JOB_META_KEY_AGENT_ID} from job meta", fl_ctx=fl_ctx)
                    return

                if not isinstance(agent_id, str):
                    self.system_panic(
                        reason=f"invalid {defs.JOB_META_KEY_AGENT_ID} from job meta: {agent_id}. "
                        f"must be str but got {type(agent_id)}",
                        fl_ctx=fl_ctx,
                    )
                    return

                self.agent_id = agent_id

            client_name = fl_ctx.get_identity_name()
            self.flare_agent_fqcn = defs.agent_site_fqcn(client_name, self.agent_id)
            self.log_info(fl_ctx, f"Flare Agent FQCN: {self.flare_agent_fqcn}")
            t = threading.Thread(target=self._monitor, daemon=True)
            t.start()
        elif event_type == EventType.END_RUN:
            self.is_done = True
            self._say_goodbye()

    def _say_goodbye(self):
        # say goodbye to agent
        self.logger.info(f"job done - say goodbye to {self.flare_agent_fqcn}")
        reply = self.cell.send_request(
            channel=defs.CHANNEL,
            topic=defs.TOPIC_BYE,
            target=self.flare_agent_fqcn,
            request=Message(),
            optional=True,
            timeout=self.agent_ack_timeout,
        )
        if reply:
            rc = reply.get_header(MessageHeaderKey.RETURN_CODE)
            if rc != CellReturnCode.OK:
                self.logger.warning(f"return code from agent {self.flare_agent_fqcn} for bye: {rc}")

    def _monitor(self):
        # try to connect the flare agent
        self.logger.info(f"waiting for flare agent {self.flare_agent_fqcn} ...")
        assert isinstance(self.cell, Cell)

        last_hb_time = 0
        while True:
            if self.is_done:
                return

            if time.time() - last_hb_time > self.agent_heartbeat_interval:
                reply = self.cell.send_request(
                    channel=defs.CHANNEL,
                    topic=defs.TOPIC_HEARTBEAT,
                    target=self.flare_agent_fqcn,
                    request=Message(),
                    timeout=self.agent_ack_timeout,
                    optional=True,
                )
                last_hb_time = time.time()
                rc = reply.get_header(MessageHeaderKey.RETURN_CODE)
                if rc == CellReturnCode.OK:
                    self.last_agent_ack_time = last_hb_time
                    if not self.is_connected:
                        self.logger.info(f"agent {self.flare_agent_fqcn} connected")
                        self.is_connected = True
                else:
                    since_last_ack = last_hb_time - self.last_agent_ack_time
                    if since_last_ack > self.agent_connection_timeout:
                        if self.is_connected:
                            self.logger.warning(
                                f"agent {self.flare_agent_fqcn} disconnected: "
                                f"no heartbeat for {self.agent_connection_timeout} secs"
                            )
                            self.is_connected = False

                    if self.agent_heartbeat_timeout and since_last_ack > self.agent_heartbeat_timeout:
                        self.is_done = True
                        with self.engine.new_context() as fl_ctx:
                            self.system_panic(
                                f"agent {self.flare_agent_fqcn} is dead: "
                                f"no heartbeat for {self.agent_heartbeat_timeout} secs",
                                fl_ctx=fl_ctx,
                            )
                        return

            # sleep only small amount of time, so we can check other conditions frequently
            time.sleep(_SHORT_SLEEP_TIME)

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        task_id = shareable.get_header(key=FLContextKey.TASK_ID)
        current_task = self.task_ctx
        if current_task:
            # still working on previous task!
            self.log_error(fl_ctx, f"got new task {task_name=} {task_id=} while still working on {current_task}")
            return make_reply(ReturnCode.BAD_REQUEST_DATA)

        # wait for flare agent
        while True:
            if self.is_done or abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)

            if self.is_connected:
                break
            else:
                time.sleep(_SHORT_SLEEP_TIME)

        self.task_ctx = _TaskContext(task_name, task_id, fl_ctx)
        result = self._do_execute(task_name, shareable, fl_ctx, abort_signal)
        self.task_ctx = None
        return result

    def _send_task(self, task_ctx: _TaskContext, msg, abort_signal):
        # keep sending until done
        fl_ctx = task_ctx.fl_ctx
        task_name = task_ctx.task_name
        task_id = task_ctx.task_id
        task_ctx.send_rc = ReturnCode.OK
        last_send_time = 0

        while True:
            if self.is_done or abort_signal.triggered:
                self.log_info(fl_ctx, "task aborted - ask agent to abort the task")

                # it's possible that the agent may have already received the task
                # we ask it to abort the task.
                self._ask_agent_to_abort_task(task_name, task_id)
                task_ctx.send_rc = ReturnCode.TASK_ABORTED
                return

            if task_ctx.result_received_time:
                # the result has been received
                # this could happen only when we thought the previous send didn't succeed, but it actually did!
                return

            if self.is_connected and time.time() - last_send_time > self.resend_task_interval:
                self.log_info(fl_ctx, f"try to send task to {self.flare_agent_fqcn}")
                start = time.time()
                reply = self.cell.send_request(
                    channel=defs.CHANNEL,
                    topic=defs.TOPIC_GET_TASK,
                    request=msg,
                    target=self.flare_agent_fqcn,
                    timeout=self.send_task_timeout,
                )
                last_send_time = time.time()
                rc = reply.get_header(MessageHeaderKey.RETURN_CODE)
                if rc == CellReturnCode.OK:
                    self.log_info(fl_ctx, f"Sent task to {self.flare_agent_fqcn} in {time.time() - start} secs")
                    return
                elif rc == CellReturnCode.INVALID_REQUEST:
                    self.log_error(fl_ctx, f"Task rejected by {self.flare_agent_fqcn}: {rc}")
                    task_ctx.send_rc = ReturnCode.BAD_REQUEST_DATA
                    return
                else:
                    self.log_error(
                        fl_ctx,
                        f"Failed to send task to {self.flare_agent_fqcn}: {rc}. "
                        "Will retry in {self.resend_task_interval} secs",
                    )
            time.sleep(_SHORT_SLEEP_TIME)

    def _do_execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        try:
            dxo = from_shareable(shareable)
        except:
            self.log_error(fl_ctx, f"Unable to extract dxo from shareable: {secure_format_traceback()}")
            return make_reply(ReturnCode.BAD_TASK_DATA)

        # send to flare agent
        is_app_defined = False
        task_ctx = self.task_ctx
        task_id = task_ctx.task_id
        data = dxo.data
        if dxo.data_kind == DataKind.APP_DEFINED:
            is_app_defined = True
        if not data:
            data = {}
        meta = dxo.meta
        if not meta:
            meta = {}

        current_round = shareable.get_header(AppConstants.CURRENT_ROUND, None)
        total_rounds = shareable.get_header(AppConstants.NUM_ROUNDS, None)

        meta[defs.MetaKey.DATA_KIND] = dxo.data_kind
        if current_round is not None:
            meta[defs.MetaKey.CURRENT_ROUND] = current_round
        if total_rounds is not None:
            meta[defs.MetaKey.TOTAL_ROUND] = total_rounds

        msg = Message(
            headers={
                defs.MsgHeader.TASK_ID: task_id,
                defs.MsgHeader.TASK_NAME: task_name,
            },
            payload={defs.PayloadKey.DATA: data, defs.PayloadKey.META: meta},
        )

        # keep sending until done
        self._send_task(task_ctx, msg, abort_signal)
        if task_ctx.send_rc != ReturnCode.OK:
            # send_task failed
            return make_reply(task_ctx.send_rc)

        # wait for result
        self.log_info(fl_ctx, f"Waiting for result from {self.flare_agent_fqcn}")
        while True:
            if task_ctx.result_waiter.wait(timeout=_SHORT_SLEEP_TIME):
                # result available
                break
            else:
                # timed out - check other conditions
                if self.is_done or abort_signal.triggered:
                    self.log_info(fl_ctx, "task is aborted")

                    # notify the agent
                    self._ask_agent_to_abort_task(task_name, task_id)
                    self.task_ctx = None
                    return make_reply(ReturnCode.TASK_ABORTED)

        # convert the result
        if task_ctx.result_rc not in [defs.RC.OK, defs.RC.EARLY_TERMINATION]:
            return make_reply(task_ctx.result_rc)

        result = task_ctx.result
        meta = result.get(defs.PayloadKey.META)

        data = result.get(defs.PayloadKey.DATA)
        if is_app_defined:
            data_kind = DataKind.APP_DEFINED
        else:
            data_kind = meta.get(defs.MetaKey.DATA_KIND, DataKind.WEIGHTS)

        dxo = DXO(
            data_kind=data_kind,
            data=data,
            meta=meta,
        )
        s = dxo.to_shareable()
        s.set_return_code(task_ctx.result_rc)
        return s

    def _ask_agent_to_abort_task(self, task_name, task_id):
        msg = Message(
            headers={
                defs.MsgHeader.TASK_ID: task_id,
                defs.MsgHeader.TASK_NAME: task_name,
            }
        )

        self.cell.fire_and_forget(
            channel=defs.CHANNEL,
            topic=defs.TOPIC_ABORT,
            message=msg,
            targets=[self.flare_agent_fqcn],
            optional=True,
        )

    @staticmethod
    def _finish_result(task_ctx: _TaskContext, result_rc="", result=None, result_is_valid=True):
        task_ctx.result_rc = result_rc
        task_ctx.result = result
        task_ctx.result_received_time = time.time()
        task_ctx.result_waiter.set()
        if result_is_valid:
            return make_cell_reply(CellReturnCode.OK)
        else:
            return make_cell_reply(CellReturnCode.INVALID_REQUEST)

    def _receive_result(self, request: Message) -> Union[None, Message]:
        sender = request.get_header(MessageHeaderKey.ORIGIN)
        task_id = request.get_header(defs.MsgHeader.TASK_ID)

        # When the agent is restarted, it sends a result to us, in case we are waiting for the result
        # of the current task. In this case, the task_id is empty.
        task_ctx = self.task_ctx
        if not task_ctx:
            # we are not waiting for any result
            if not task_id:
                # this was sent by the agent when it's started or restarted - just ignore
                return make_cell_reply(CellReturnCode.OK)

            self.logger.error(f"received result from {sender} for task {task_id} while not waiting for result!")
            return make_cell_reply(CellReturnCode.INVALID_REQUEST)

        # the agent could send us valid result after restarted
        fl_ctx = task_ctx.fl_ctx
        if task_id and task_id != task_ctx.task_id:
            self.log_error(fl_ctx, f"received task id {task_id} != expected {task_ctx.task_id}")
            return make_cell_reply(CellReturnCode.INVALID_REQUEST)

        if task_ctx.result_received_time:
            # already received - this is a dup
            self.log_info(fl_ctx, f"received duplicate result from {sender}")
            return make_cell_reply(CellReturnCode.OK)

        payload = request.payload
        if not isinstance(payload, dict):
            self.log_error(fl_ctx, f"bad result from {sender}: expect dict but got {type(payload)}")
            return self._finish_result(task_ctx, result_is_valid=False, result_rc=ReturnCode.EXECUTION_EXCEPTION)

        data = payload.get(defs.PayloadKey.DATA)
        if data is None:
            self.log_error(fl_ctx, f"bad result from {sender}: missing {defs.PayloadKey.DATA}")
            return self._finish_result(task_ctx, result_is_valid=False, result_rc=ReturnCode.EXECUTION_EXCEPTION)

        meta = payload.get(defs.PayloadKey.META)
        if meta is None:
            self.log_error(fl_ctx, f"bad result from {sender}: missing {defs.PayloadKey.META}")
            return self._finish_result(task_ctx, result_is_valid=False, result_rc=ReturnCode.EXECUTION_EXCEPTION)

        self.log_info(fl_ctx, f"received result from {sender}")
        return self._finish_result(
            task_ctx,
            result_is_valid=True,
            result_rc=request.get_header(defs.MsgHeader.RC, defs.RC.OK),
            result=payload,
        )
