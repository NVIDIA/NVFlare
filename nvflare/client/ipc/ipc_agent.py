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

import logging
import threading
import time
import traceback
from typing import Union

from nvflare.app_common.decomposers import numpy_decomposers
from nvflare.client.ipc import defs
from nvflare.fuel.f3.cellnet.cell import Cell, Message
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.net_agent import NetAgent
from nvflare.fuel.f3.cellnet.utils import make_reply
from nvflare.fuel.f3.drivers.driver_params import DriverParams
from nvflare.fuel.utils.config_service import ConfigService
from nvflare.private.fed.utils.fed_utils import register_ext_decomposers

_SSL_ROOT_CERT = "rootCA.pem"
_SHORT_SLEEP_TIME = 0.2


class IPCAgent:
    def __init__(
        self,
        flare_site_url: str,
        flare_site_name: str,
        agent_id: str,
        workspace_dir: str,
        secure_mode=False,
        submit_result_timeout=30.0,
        flare_site_connection_timeout=60.0,
        flare_site_heartbeat_timeout=None,
        resend_result_interval=2.0,
        decomposer_module=None,
    ):
        """Constructor of Flare Agent. The agent is responsible for communicating with the Flare Client Job cell (CJ)
        to get task and to submit task result.

        Args:
            flare_site_url: the URL to the client parent cell (CP)
            flare_site_name: the CJ's site name (client name)
            agent_id: the unique ID of the agent
            workspace_dir: directory that contains startup folder and comm_config.json
            secure_mode: whether the connection is in secure mode or not
            submit_result_timeout: when submitting task result, how long to wait for response from the CJ
            flare_site_heartbeat_timeout: time for missing heartbeats from CJ before considering it dead
            flare_site_connection_timeout: time for missing heartbeats from CJ before considering it disconnected
        """
        ConfigService.initialize(section_files={}, config_path=[workspace_dir])

        self.logger = logging.getLogger(self.__class__.__name__)
        self.cell_name = defs.agent_site_fqcn(flare_site_name, agent_id)
        self.workspace_dir = workspace_dir
        self.secure_mode = secure_mode
        self.flare_site_url = flare_site_url
        self.submit_result_timeout = submit_result_timeout
        self.flare_site_heartbeat_timeout = flare_site_heartbeat_timeout
        self.flare_site_connection_timeout = flare_site_connection_timeout
        self.resend_result_interval = resend_result_interval
        self.num_results_submitted = 0
        self.current_task = None
        self.pending_task = None
        self.task_lock = threading.Lock()
        self.last_msg_time = time.time()  # last time to get msg from flare site
        self.peer_fqcn = None
        self.is_done = False
        self.is_started = False  # has the agent been started?
        self.is_stopped = False  # has the agent been stopped?
        self.is_connected = False  # is the agent connected to the flare site?
        self.credentials = {}  # security credentials for secure connection

        if secure_mode:
            root_cert_path = ConfigService.find_file(_SSL_ROOT_CERT)
            if not root_cert_path:
                raise ValueError(f"cannot find {_SSL_ROOT_CERT} from config path {workspace_dir}")

            self.credentials = {
                DriverParams.CA_CERT.value: root_cert_path,
            }

        self.cell = Cell(
            fqcn=self.cell_name,
            root_url="",
            parent_url=self.flare_site_url,
            secure=self.secure_mode,
            credentials=self.credentials,
            create_internal_listener=False,
        )
        self.net_agent = NetAgent(self.cell)

        self.cell.register_request_cb(channel=defs.CHANNEL, topic=defs.TOPIC_GET_TASK, cb=self._receive_task)
        self.logger.info(f"registered task CB for {defs.CHANNEL} {defs.TOPIC_GET_TASK}")
        self.cell.register_request_cb(channel=defs.CHANNEL, topic=defs.TOPIC_HEARTBEAT, cb=self._handle_heartbeat)
        self.cell.register_request_cb(channel=defs.CHANNEL, topic=defs.TOPIC_BYE, cb=self._handle_bye)
        self.cell.register_request_cb(channel=defs.CHANNEL, topic=defs.TOPIC_ABORT, cb=self._handle_abort_task)
        self.cell.core_cell.add_incoming_request_filter(
            channel="*",
            topic="*",
            cb=self._msg_received,
        )
        self.cell.core_cell.add_incoming_reply_filter(
            channel="*",
            topic="*",
            cb=self._msg_received,
        )
        numpy_decomposers.register()
        if decomposer_module:
            register_ext_decomposers(decomposer_module)

    def start(self):
        """Start the agent. This method must be called to enable CJ/Agent communication.

        Returns: None

        """
        if self.is_started:
            self.logger.warning("the agent is already started")
            return

        if self.is_stopped:
            raise defs.CallStateError("cannot start the agent since it is already stopped")

        self.is_started = True
        self.logger.info(f"starting agent {self.cell_name} ...")
        self.cell.start()
        t = threading.Thread(target=self._monitor, daemon=True)
        t.start()

    def stop(self):
        """Stop the agent. After this is called, there will be no more communications between CJ and agent.

        Returns: None

        """
        if not self.is_started:
            self.logger.warning("cannot stop the agent since it is not started")
            return

        if self.is_stopped:
            self.logger.warning("agent is already stopped")
            return

        self.is_stopped = True
        self.cell.stop()
        self.net_agent.close()

    def _monitor(self):
        while True:
            since_last_msg = time.time() - self.last_msg_time
            if since_last_msg > self.flare_site_connection_timeout:
                if self.is_connected:
                    self.logger.error(
                        "flare site disconnected since no message received "
                        f"for {self.flare_site_connection_timeout} seconds"
                    )
                self.is_connected = False

            if self.flare_site_heartbeat_timeout and since_last_msg > self.flare_site_heartbeat_timeout:
                self.logger.error(
                    f"flare site is dead since no message received for {self.flare_site_heartbeat_timeout} seconds"
                )
                self.is_done = True
                return

            time.sleep(_SHORT_SLEEP_TIME)

    def _handle_bye(self, request: Message) -> Union[None, Message]:
        peer = request.get_header(MessageHeaderKey.ORIGIN)
        self.logger.info(f"got goodbye from {peer}")
        self.is_done = True
        return make_reply(ReturnCode.OK)

    def _msg_received(self, request: Message):
        peer = request.get_header(MessageHeaderKey.ORIGIN)
        if self.peer_fqcn and self.peer_fqcn != peer:
            # this could happen when a new job is started for the same training
            self.logger.warning(f"got peer FQCN '{peer}' while expecting '{self.peer_fqcn}'")

        self.peer_fqcn = peer
        self.last_msg_time = time.time()
        if not self.is_connected:
            self.is_connected = True
            self.logger.info(f"connected to flare site {peer}")

    def _handle_heartbeat(self, request: Message) -> Union[None, Message]:
        peer = request.get_header(MessageHeaderKey.ORIGIN)
        self.logger.debug(f"got heartbeat from {peer}")
        return make_reply(ReturnCode.OK)

    def _handle_abort_task(self, request: Message) -> Union[None, Message]:
        peer = request.get_header(MessageHeaderKey.ORIGIN)
        task_id = request.get_header(defs.MsgHeader.TASK_ID)
        task_name = request.get_header(defs.MsgHeader.TASK_NAME)
        self.logger.warning(f"received from {peer} to abort {task_name=} {task_id=}")
        with self.task_lock:
            if self.current_task and task_id == self.current_task.task_id:
                self.current_task.aborted = True
            elif self.pending_task and task_id == self.pending_task.task_id:
                self.pending_task = None
        return make_reply(ReturnCode.OK)

    def _receive_task(self, request: Message) -> Union[None, Message]:
        with self.task_lock:
            return self._do_receive_task(request)

    def _create_task(self, request: Message):
        peer = request.get_header(MessageHeaderKey.ORIGIN)
        task_id = request.get_header(defs.MsgHeader.TASK_ID)
        task_name = request.get_header(defs.MsgHeader.TASK_NAME)
        self.logger.info(f"received task from {peer}: {task_name=} {task_id=}")

        task_data = request.payload
        if not isinstance(task_data, dict):
            self.logger.error(f"bad task data from {peer}: expect dict but got {type(task_data)}")
            return None

        data = task_data.get(defs.PayloadKey.DATA)
        if not data:
            self.logger.error(f"bad task data from {peer}: missing {defs.PayloadKey.DATA}")
            return None

        meta = task_data.get(defs.PayloadKey.META)
        if not meta:
            self.logger.error(f"bad task data from {peer}: missing {defs.PayloadKey.META}")
            return None

        return defs.Task(task_name, task_id, meta, data)

    def _do_receive_task(self, request: Message) -> Union[None, Message]:
        peer = request.get_header(MessageHeaderKey.ORIGIN)
        task_id = request.get_header(defs.MsgHeader.TASK_ID)
        task_name = request.get_header(defs.MsgHeader.TASK_NAME)

        # create a new task
        new_task = self._create_task(request)
        if not new_task:
            return make_reply(ReturnCode.INVALID_REQUEST)

        if self.pending_task:
            assert isinstance(self.pending_task, defs.Task)
            if task_id == self.pending_task.task_id:
                return make_reply(ReturnCode.OK)
            else:
                # this could happen when the CJ is restarted
                self.logger.warning(f"got new task from {peer} while already having a pending task!")

                # replace the pending task
                self.pending_task = new_task
                return make_reply(ReturnCode.OK)

        current_task = self.current_task
        if current_task:
            assert isinstance(current_task, defs.Task)
            if task_id == current_task.task_id:
                self.logger.info(f"received duplicate task {task_id} from {peer}")
                return make_reply(ReturnCode.OK)

            if current_task.last_send_result_time:
                # we already tried to send result back
                # assume that the flare site has received
                # we set the flag so the sending process will end quickly
                # in the meanwhile we ask flare site to retry later
                current_task.already_received = True
            else:
                # error - one task at a time
                self.logger.warning(
                    f"got task {task_name} {task_id} from {peer} "
                    f"while still working on {current_task.task_name} {current_task.task_id}"
                )

                # this could happen when CJ is restarted while we are processing current task
                # we set the current_task to be aborted. App should check this flag frequently to abort processing
                current_task.aborted = True

            # treat the new task as pending task - it will become current after the current_task is submitted
            self.pending_task = new_task
            return make_reply(ReturnCode.OK)
        else:
            # no current task
            self.current_task = new_task
            return make_reply(ReturnCode.OK)

    def get_task(self, timeout=None):
        """Get a task from FLARE. This is a blocking call.

        If timeout is specified, this call is blocked only for the specified amount of time.
        If timeout is not specified, this call is blocked forever until a task is received or agent is closed.

        Args:
            timeout: amount of time to block

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
            if self.is_done or self.is_stopped:
                self.logger.info("no more tasks - agent closed")
                raise defs.AgentClosed("flare agent is closed")

            with self.task_lock:
                current_task = self.current_task
                if current_task:
                    assert isinstance(current_task, defs.Task)
                    if current_task.aborted:
                        pass
                    elif current_task.status == defs.Task.NEW:
                        current_task.status = defs.Task.FETCHED
                        return current_task
                    else:
                        raise defs.CallStateError(
                            f"application called get_task while the current task is in status {current_task.status}"
                        )
            if timeout and time.time() - start > timeout:
                # no task available before timeout
                self.logger.info(f"get_task timeout after {timeout} seconds")
                return None
            time.sleep(_SHORT_SLEEP_TIME)

    def submit_result(self, result: defs.TaskResult) -> bool:
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
        try:
            result_submitted = self._do_submit_result(result)
        except Exception as ex:
            self.logger.error(f"exception encountered: {ex}")
            result_submitted = False

        with self.task_lock:
            self.current_task = None
            if self.pending_task:
                # a new task is waiting for the current task to finish
                self.current_task = self.pending_task
                self.pending_task = None
        return result_submitted

    def _do_submit_result(self, result: defs.TaskResult) -> bool:
        if not isinstance(result, defs.TaskResult):
            raise TypeError(f"result must be TaskResult but got {type(result)}")

        with self.task_lock:
            current_task = self.current_task
            if current_task:
                if current_task.aborted:
                    return False
                if current_task.status != defs.Task.FETCHED:
                    raise defs.CallStateError(
                        f"submit_result is called while current task is in status {current_task.status}"
                    )
                current_task.status = defs.Task.PROCESSED
            elif self.num_results_submitted > 0:
                self.logger.error("submit_result is called but there is no current task!")
                return False
            else:
                # if the agent is restarted, it may pick up from previous checkpoint and continue training.
                # then it can send the result after finish training.
                pass
            self.num_results_submitted += 1
        try:
            return self._send_result(current_task, result)
        except:
            self.logger.error(f"exception submitting result to {current_task.sender}")
            traceback.print_exc()
            return False

    def _send_result(self, current_task: defs.Task, result: defs.TaskResult):
        meta = result.meta
        rc = result.return_code
        data = result.data

        msg = Message(
            headers={
                defs.MsgHeader.TASK_NAME: current_task.task_name if current_task else "",
                defs.MsgHeader.TASK_ID: current_task.task_id if current_task else "",
                defs.MsgHeader.RC: rc,
            },
            payload={
                defs.PayloadKey.META: meta,
                defs.PayloadKey.DATA: data,
            },
        )

        last_send_time = 0
        while True:
            if self.is_done or self.is_stopped:
                self.logger.error(f"quit submitting result for task {current_task} since agent is closed")
                raise defs.AgentClosed("agent is stopped")

            if current_task and current_task.already_received:
                if not current_task.last_send_result_time:
                    self.logger.warning(f"task {current_task} was marked already_received but has been sent!")
                return True

            if current_task and current_task.aborted:
                self.logger.error(f"quit submitting result for task {current_task} since it is aborted")
                return False

            if self.is_connected and time.time() - last_send_time > self.resend_result_interval:
                self.logger.info(f"sending result to {self.peer_fqcn} for task {current_task}")
                if current_task:
                    current_task.last_send_result_time = time.time()
                reply = self.cell.send_request(
                    channel=defs.CHANNEL,
                    topic=defs.TOPIC_SUBMIT_RESULT,
                    target=self.peer_fqcn,
                    request=msg,
                    timeout=self.submit_result_timeout,
                )
                last_send_time = time.time()
                if reply:
                    rc = reply.get_header(MessageHeaderKey.RETURN_CODE)
                    peer = reply.get_header(MessageHeaderKey.ORIGIN)
                    if rc == ReturnCode.OK:
                        return True
                    elif rc == ReturnCode.INVALID_REQUEST:
                        self.logger.error(f"received return code from {peer}: {rc}")
                        return False
                    else:
                        self.logger.info(f"failed to send to {self.peer_fqcn}: {rc} - will retry")
            time.sleep(_SHORT_SLEEP_TIME)
