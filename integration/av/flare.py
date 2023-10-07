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

import logging
import threading
import time
import traceback
from typing import Union

from nvflare.app_common.decomposers import common_decomposers
from nvflare.fuel.f3.cellnet.cell import Cell, Message
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.net_agent import NetAgent
from nvflare.fuel.f3.cellnet.utils import make_reply, new_message
from nvflare.fuel.f3.drivers.driver_params import DriverParams
from nvflare.fuel.utils.config_service import ConfigService

from . import defs
from .defs import RC, MsgHeader, PayloadKey

SSL_PRIVATE_KEY = "client.key"
SSL_CERT = "client.crt"
SSL_ROOT_CERT = "rootCA.pem"


class Task:
    def __init__(self, task_name: str, task_id: str, meta: dict, data):
        self.task_name = task_name
        self.task_id = task_id
        self.meta = meta
        self.data = data

    def __str__(self):
        return f"'{self.task_name} {self.task_id}'"


class TaskResult:
    def __init__(self, meta: dict, data, return_code=RC.OK):
        if not meta:
            meta = {}

        if not isinstance(meta, dict):
            raise TypeError(f"meta must be dict but got {type(meta)}")

        if not data:
            data = {}

        if not isinstance(return_code, str):
            raise TypeError(f"return_code must be str but got {type(return_code)}")

        self.return_code = return_code
        self.meta = meta
        self.data = data


class AgentClosed(Exception):
    pass


class CallStateError(Exception):
    pass


class _TaskContext:

    NEW = 0
    FETCHED = 1
    PROCESSED = 2

    def __init__(self, sender: str, task_name: str, task_id: str, meta: dict, data):
        self.sender = sender
        self.task_name = task_name
        self.task_id = task_id
        self.meta = meta
        self.data = data
        self.status = _TaskContext.NEW
        self.last_send_result_time = None
        self.aborted = False
        self.already_received = False

    def __str__(self):
        return f"'{self.task_name} {self.task_id}'"


class FlareAgent:
    def __init__(
        self,
        root_url: str,
        flare_site_name: str,
        agent_id: str,
        workspace_dir: str,
        secure_mode=False,
        submit_result_timeout=30.0,
        flare_site_ready_timeout=60.0,
    ):
        ConfigService.initialize(section_files={}, config_path=[workspace_dir])

        self.logger = logging.getLogger("FlareAgent")
        self.cell_name = defs.agent_site_fqcn(flare_site_name, agent_id)
        self.workspace_dir = workspace_dir
        self.secure_mode = secure_mode
        self.root_url = root_url
        self.submit_result_timeout = submit_result_timeout
        self.flare_site_ready_timeout = flare_site_ready_timeout
        self.current_task = None
        self.pending_task = None
        self.task_lock = threading.Lock()
        self.last_hb_time = time.time()
        self.is_done = False
        self.credentials = {}

        if secure_mode:
            root_cert_path = ConfigService.find_file(SSL_ROOT_CERT)
            if not root_cert_path:
                raise ValueError(f"cannot find {SSL_ROOT_CERT} from config path {workspace_dir}")

            cert_path = ConfigService.find_file(SSL_CERT)
            if not cert_path:
                raise ValueError(f"cannot find {SSL_CERT} from config path {workspace_dir}")

            private_key_path = ConfigService.find_file(SSL_PRIVATE_KEY)
            if not private_key_path:
                raise ValueError(f"cannot find {SSL_PRIVATE_KEY} from config path {workspace_dir}")

            self.credentials = {
                DriverParams.CA_CERT.value: root_cert_path,
                DriverParams.CLIENT_CERT.value: cert_path,
                DriverParams.CLIENT_KEY.value: private_key_path,
            }

        self.cell = Cell(
            fqcn=self.cell_name,
            root_url=self.root_url,
            secure=self.secure_mode,
            credentials=self.credentials,
            create_internal_listener=False,
        )
        self.agent = NetAgent(self.cell)

        self.cell.register_request_cb(channel=defs.CHANNEL, topic=defs.TOPIC_GET_TASK, cb=self._receive_task)
        self.logger.info(f"registered task CB for {defs.CHANNEL} {defs.TOPIC_GET_TASK}")
        self.cell.register_request_cb(channel=defs.CHANNEL, topic=defs.TOPIC_HELLO, cb=self._handle_hello)
        self.cell.register_request_cb(channel=defs.CHANNEL, topic=defs.TOPIC_HEARTBEAT, cb=self._handle_heartbeat)
        self.cell.register_request_cb(channel=defs.CHANNEL, topic=defs.TOPIC_BYE, cb=self._handle_bye)
        self.cell.register_request_cb(channel=defs.CHANNEL, topic=defs.TOPIC_ABORT, cb=self._handle_abort_task)
        common_decomposers.register()

    def start(self):
        self.logger.info(f"starting agent {self.cell_name} ...")
        self.cell.start()
        t = threading.Thread(target=self._maintain, daemon=True)
        t.start()

    def stop(self):
        self.cell.stop()
        self.agent.close()

    def _maintain(self):
        self.logger.info("started to maintain ...")
        while True:
            if time.time() - self.last_hb_time > self.flare_site_ready_timeout:
                self.logger.info(
                    f"closing agent {self.cell_name}: flare site not ready in {self.flare_site_ready_timeout} seconds"
                )
                self.is_done = True
                return
            time.sleep(1.0)

    def _handle_hello(self, request: Message) -> Union[None, Message]:
        sender = request.get_header(MessageHeaderKey.ORIGIN)
        self.logger.info(f"got hello from {sender}")
        self.last_hb_time = time.time()
        return make_reply(ReturnCode.OK)

    def _handle_bye(self, request: Message) -> Union[None, Message]:
        sender = request.get_header(MessageHeaderKey.ORIGIN)
        self.logger.info(f"got goodbye from {sender}")
        self.is_done = True
        return make_reply(ReturnCode.OK)

    def _handle_heartbeat(self, request: Message) -> Union[None, Message]:
        self.last_hb_time = time.time()
        sender = request.get_header(MessageHeaderKey.ORIGIN)
        self.logger.info(f"got heartbeat from {sender}")
        return make_reply(ReturnCode.OK)

    def _handle_abort_task(self, request: Message) -> Union[None, Message]:
        sender = request.get_header(MessageHeaderKey.ORIGIN)
        task_id = request.get_header(MsgHeader.TASK_ID)
        task_name = request.get_header(MsgHeader.TASK_NAME)
        self.logger.warning(f"received from {sender} to abort {task_name=} {task_id=}")
        with self.task_lock:
            if self.current_task and task_id == self.current_task.task_id:
                self.current_task.aborted = True
            elif self.pending_task and task_id == self.pending_task.task_id:
                self.pending_task = None
        return make_reply(ReturnCode.OK)

    def _receive_task(self, request: Message) -> Union[None, Message]:
        self.logger.info("receiving task ...")
        with self.task_lock:
            return self._do_receive_task(request)

    def _create_task(self, request: Message):
        sender = request.get_header(MessageHeaderKey.ORIGIN)
        task_id = request.get_header(MsgHeader.TASK_ID)
        task_name = request.get_header(MsgHeader.TASK_NAME)

        task_data = request.payload
        if not isinstance(task_data, dict):
            self.logger.error(f"bad task data from {sender}: expect dict but got {type(task_data)}")
            return None

        data = task_data.get(PayloadKey.DATA)
        if not data:
            self.logger.error(f"bad task data from {sender}: missing {PayloadKey.DATA}")
            return None

        meta = task_data.get(PayloadKey.META)
        if not meta:
            self.logger.error(f"bad task data from {sender}: missing {PayloadKey.META}")
            return None

        return _TaskContext(sender, task_name, task_id, meta, data)

    def _do_receive_task(self, request: Message) -> Union[None, Message]:
        sender = request.get_header(MessageHeaderKey.ORIGIN)
        task_id = request.get_header(MsgHeader.TASK_ID)
        task_name = request.get_header(MsgHeader.TASK_NAME)
        self.logger.info(f"_do_receive_task from {sender}: {task_name=} {task_id=}")

        if self.pending_task:
            if task_id == self.pending_task.task_id:
                return make_reply(ReturnCode.OK)
            else:
                self.logger.error(f"got a new task while already have a pending task!")
                return make_reply(ReturnCode.INVALID_REQUEST)

        current_task = self.current_task
        if current_task:
            assert isinstance(current_task, _TaskContext)
            if task_id == current_task.task_id:
                self.logger.info(f"received duplicate task {task_id} from {sender}")
                return make_reply(ReturnCode.OK)

            if current_task.last_send_result_time:
                # we already tried to send result back
                # assume that the flare site has received
                # we set the flag so the sending process will end quickly
                # in the meanwhile we ask flare site to retry later
                current_task.already_received = True
                self.pending_task = self._create_task(request)
                if self.pending_task:
                    return make_reply(ReturnCode.OK)
                else:
                    return make_reply(ReturnCode.INVALID_REQUEST)
            else:
                # error - one task at a time
                self.logger.error(
                    f"got task {task_name} {task_id} from {sender} "
                    f"while still working on {current_task.task_name} {current_task.task_id}"
                )
                return make_reply(ReturnCode.INVALID_REQUEST)

        self.current_task = self._create_task(request)
        if self.current_task:
            return make_reply(ReturnCode.OK)
        else:
            return make_reply(ReturnCode.INVALID_REQUEST)

    def get_task(self, timeout=None):
        """Get a task from FLARE. This is a blocking call.
        If timeout is specified, this call is blocked only for the specified amount of time.
        If timeout is not specified, this call is blocked forever until a task is received or agent is closed.

        Returns: None if no task is available during before timeout; or a Task object if task is available.
        Raises:
            AgentClosed exception if the agent is closed before timeout.
            CallStateError exception if the call is not made properly.

        Note: the application must make the call only when it is just started or after a previous task's result
        has been submitted.

        """
        if timeout is not None:
            if not isinstance(timeout, (int, float)):
                raise TypeError(f"timeout must be (int, float) but got {type(timeout)}")
            if timeout <= 0:
                raise ValueError(f"timeout must > 0, but got {timeout}")

        start = time.time()
        while True:
            if self.is_done:
                self.logger.info("no task available - agent closed")
                raise AgentClosed("flare agent is closed")

            with self.task_lock:
                current_task = self.current_task
                if current_task:
                    assert isinstance(current_task, _TaskContext)
                    if current_task.aborted:
                        pass
                    elif current_task.status == _TaskContext.NEW:
                        current_task.status = _TaskContext.FETCHED
                        return Task(current_task.task_name, current_task.task_id, current_task.meta, current_task.data)
                    else:
                        raise CallStateError(
                            f"application called get_task while the current task is in status {current_task.status}"
                        )
            if timeout and time.time() - start > timeout:
                # no task available before timeout
                self.logger.info(f"get_task timeout after {timeout} seconds")
                return None
            time.sleep(0.5)

    def submit_result(self, result: TaskResult) -> bool:
        """Submit the result of the current task.
        This is a blocking call. The agent will try to send the result to flare site until it is successfully sent or
        the task is aborted or the agent is closed.

        Args:
            result: result to be submitted

        Returns: whether the result is submitted successfully
        Raises: the CallStateError exception if the submit_result call is not made properly.

        Notes: the application must only make this call after the received task is processed. The call can only be
        made a single time regardless whether the submission is successful.

        """
        if not isinstance(result, TaskResult):
            raise TypeError(f"result must be TaskResult but got {type(result)}")

        with self.task_lock:
            current_task = self.current_task
            if not current_task:
                self.logger.error("submit_result is called but there is no current task!")
                return False

            assert isinstance(current_task, _TaskContext)
            if current_task.aborted:
                return False
            if current_task.status != _TaskContext.FETCHED:
                raise CallStateError(f"submit_result is called while current task is in status {current_task.status}")
            current_task.status = _TaskContext.PROCESSED
        try:
            result = self._do_submit_result(current_task, result)
        except:
            self.logger.error(f"exception submitting result to {current_task.sender}")
            traceback.print_exc()
            result = False

        with self.task_lock:
            self.current_task = None
            if self.pending_task:
                # a new task is waiting for the current task to finish
                self.current_task = self.pending_task
                self.pending_task = None
        return result

    def _do_submit_result(self, current_task: _TaskContext, result: TaskResult):
        meta = result.meta
        rc = result.return_code
        data = result.data

        msg = new_message(
            headers={
                MsgHeader.TASK_NAME: current_task.task_name,
                MsgHeader.TASK_ID: current_task.task_id,
                MsgHeader.RC: rc,
            },
            payload={
                PayloadKey.META: meta,
                PayloadKey.DATA: data,
            },
        )
        while True:
            if current_task.already_received:
                if not current_task.last_send_result_time:
                    self.logger.warning(f"task {current_task} was marked already_received but has been sent!")
                return True

            if self.is_done:
                self.logger.error(f"quit submitting result for task {current_task} since agent is closed")
                return False

            if current_task.aborted:
                self.logger.error(f"quit submitting result for task {current_task} since it is aborted")
                return False

            current_task.last_send_result_time = time.time()
            self.logger.info(f"sending result to {current_task.sender} for task {current_task}")
            reply = self.cell.send_request(
                channel=defs.CHANNEL,
                topic=defs.TOPIC_SUBMIT_RESULT,
                target=current_task.sender,
                request=msg,
                timeout=self.submit_result_timeout,
            )
            if reply:
                rc = reply.get_header(MessageHeaderKey.RETURN_CODE)
                sender = reply.get_header(MessageHeaderKey.ORIGIN)
                if rc == ReturnCode.OK:
                    return True
                elif rc == ReturnCode.INVALID_REQUEST:
                    self.logger.error(f"received return code from {sender}: {rc}")
                    return False
                else:
                    self.logger.info(f"failed to send to {current_task.sender}: {rc} - will retry")
            time.sleep(2.0)
