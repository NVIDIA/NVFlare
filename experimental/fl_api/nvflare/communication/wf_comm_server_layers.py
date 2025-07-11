from typing import Any, List, Optional, Tuple, Dict

import time

from experimental.fl_api.nvflare.communication.wf_comm_client_layers import MessageType
from nvflare.apis.dxo import from_shareable, DXO
from experimental.fl_api.common.interfaces import CommunicationLayer, siteOrSiteList
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.wf_comm_spec import WFCommSpec
from experimental.fl_api.common.interfaces import FLMessage, MessageEnvelope
from nvflare.apis.shareable import Shareable
from nvflare.apis.controller_spec import Task, ClientTask
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.ccwf.common import StrategyConstants


class ServerCommLayer(CommunicationLayer, FLComponent):
    """
    Communication layer for FL Server using Controller (delegates to controller.communicator).
    Implements the CommunicationLayer interface.
    """

    def __init__(self, communicator: WFCommSpec, fl_ctx: FLContext):
        super().__init__()

        if communicator is None:
            raise ValueError("communicator must not be None.")
        if fl_ctx is None:
            raise ValueError("fl_ctx must not be None.")

        self.communicator: WFCommSpec = communicator
        self.fl_ctx: FLContext = fl_ctx
        self.response = {}
        self.errors = {}

    @property
    def comm(self) -> WFCommSpec:
        return self.communicator

    def broadcast_and_wait(self, sites: List[str], message: MessageType) -> Dict[str, MessageType]:
        """
        Send a message to the specified site(s).
        If message is FLMessage, create Task with FLMessage, get task name from context.
        If message is MessageEnvelope, create Task with MessageEnvelope, get task name from meta.
        Set up a callback to handle the response.
        """
        if isinstance(message, FLMessage):
            task_name = message.context.get("task_name")
            min_responses = message.context.get("min_responses", len(sites))
            if not task_name:
                raise ValueError("FLMessage.context must contain 'task_name'.")
            shareable = Shareable({"ins": message.__dict__})
        elif isinstance(message, MessageEnvelope):
            task_name = message.meta.get("task_name")
            if not task_name:
                raise ValueError("MessageEnvelope.meta must contain 'task_name'.")

            if message.sender is None:
                message.sender = self.fl_ctx.get_identity_name()
            if message.receiver and message.receiver != sites:
                self.log_info(
                    self.fl_ctx,
                    f"[WARNING] MessageEnvelope.receiver ({message.receiver}) is different "
                    f"from broadcast_to_queue targets ({sites}), overwriting receiver with sites.",
                )

            if message.type is None:
                message.type = "task"

            message.receiver = sites
            min_responses = message.meta.get("min_responses", len(sites))
            shareable = Shareable({StrategyConstants.INPUT: message.__dict__})
        else:
            raise TypeError("Message must be FLMessage or MessageEnvelope.")

        task = Task(
            name=task_name,
            data=shareable,
            result_received_cb=self.result_received_cb,
        )

        self.communicator.broadcast_and_wait(task=task, fl_ctx=self.fl_ctx, targets=sites, min_responses=min_responses)

        if self.errors:
            raise RuntimeError(f"Received errors from clients: {self.errors}")
        else:
            if len(self.response) == len(sites):
                return self.response
            else:
                self._wait_for_result(min_responses)
                return self.response

    def push_to_peers(
        self,
        sender_id: str,
        recipients: siteOrSiteList,
        message_type: str,
        payload: Any,
        timeout: Optional[float] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[str], List[MessageType]]:
        # return self.comm.push_to_peers(sender_id, recipients, message_type, payload, timeout, meta)
        raise NotImplementedError

    def result_received_cb(self, client_task: ClientTask, fl_ctx: FLContext):
        client_name = client_task.client.name
        task_name = client_task.task.name
        self.log_info(fl_ctx, f"Processing {task_name} result from client {client_name}")

        result = client_task.result
        rc = result.get_return_code()
        if rc == ReturnCode.OK:
            self.log_info(fl_ctx, f"Received result entries from client:{client_name}, " f"for task {task_name}")
            dxo: DXO = from_shareable(result)
            dx = dxo.data
            message = dx.get(StrategyConstants.OUTPUT, None)
            # message will be either MessageEnvelope (FLMessage or MessageEnvelope)
            self.response.update({client_name: message})
        else:
            self.errors.update({client_name: rc})

        # Cleanup task result
        client_task.result = None

    def _wait_for_result(self, min_responses, wait_timeout: int = 5):
        response_count = len(self.response)
        t = 0
        while t < wait_timeout and min_responses > response_count:
            time.sleep(1)
            t += 1
            response_count = len(self.response)
