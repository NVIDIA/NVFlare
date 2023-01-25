# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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
import time
from typing import List, Union

from .cmi import CellMessageInterface
from nvflare.apis.client import Client
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, ReservedHeaderKey, make_reply
from nvflare.fuel.f3.cellnet.cell import (
    Cell, FQCN, new_message, MessageHeaderKey, ReturnCode as CellReturnCode,
    Message, FqcnInfo, CellAgent
)
from nvflare.private.defs import CellChannel, AdminTopic, ParentReplyKey


class JobCellMessageInterface(CellMessageInterface):

    def __init__(
            self,
            engine,
            job_id: str
    ):
        CellMessageInterface.__init__(self, engine)
        self.job_id = job_id
        self.clients = []
        self.name_to_clients = {}
        self.parent_fqcn = FQCN.get_parent(self.cell.get_fqcn())

        self.cell.set_cell_connected_cb(self._cell_connected)

        self.cell.add_incoming_request_filter(
            channel="*",
            topic="*",
            cb=self._filter_incoming
        )
        self.cell.add_incoming_reply_filter(
            channel="*",
            topic="*",
            cb=self._filter_incoming
        )
        self.cell.add_outgoing_request_filter(
            channel="*",
            topic="*",
            cb=self._filter_outgoing
        )
        self.cell.add_outgoing_reply_filter(
            channel="*",
            topic="*",
            cb=self._filter_outgoing
        )

    def _filter_incoming(
            self,
            message: Message
    ):
        origin = message.get_header(MessageHeaderKey.ORIGIN)
        origin_info = FqcnInfo(origin)
        if not origin_info.is_on_server:
            client_name = FQCN.get_root(origin)
            client = self.name_to_clients.get(client_name)
            if not client:
                self.logger.error(f"no Client found for origin {origin}")
            else:
                message.set_prop(self.PROP_KEY_CLIENT, client)

        job_id = message.get_header(self.HEADER_JOB_ID)
        if job_id and job_id != self.job_id:
            self.logger.error(f"received job id {job_id} != my job id {self.job_id}")

    def _filter_outgoing(
            self,
            message: Message
    ):
        message.set_header(self.HEADER_JOB_ID, self.job_id)

    def get_clients(self) -> List[Client]:
        return self.clients

    def get_client_names(self) -> List[str]:
        return list(self.name_to_clients.keys())

    def get_client(self, name: str):
        return self.name_to_clients.get(name)

    def validate_clients(self, names: List[str]) -> (List[str], List[str]):
        valid_names = []
        invalid_names = []
        for name in names:
            if name in self.name_to_clients:
                valid_names.append(name)
            else:
                invalid_names.append(name)
        return valid_names, invalid_names

    def _do_child_report(self, parent_name: str):
        for i in range(5):
            reply = self.cell.send_request(
                channel=CellChannel.ADMIN,
                topic=AdminTopic.CHILD_REPORT,
                target=parent_name,
                request=new_message(payload=self.job_id)
            )
            rc = reply.get_header(MessageHeaderKey.RETURN_CODE)
            if rc == ReturnCode.OK:
                result = reply.payload
                assert isinstance(result, dict)
                for k, v in result.items():
                    if k == ParentReplyKey.JOB_ID:
                        if v != self.job_id:
                            raise RuntimeError(f"parent job id {v} does not match mine {self.job_id}")
                    elif k == ParentReplyKey.JOB_PARTICIPANTS:
                        if not isinstance(v, dict):
                            raise RuntimeError(f"job participants from parent must be dict but got {type(v)}")
                        for name, token in v.items():
                            client = Client(name=name, token=token)
                            self.clients.append(client)
                            self.name_to_clients[name] = client
                        self.logger.info(f"{self.cell.get_fqcn()}: job participants: {result.keys()}")
                return
            else:
                self.logger.error(f"{self.cell.get_fqcn()}:failed report to parent: {rc} - retry in 2 secs")
                time.sleep(2.0)
        raise RuntimeError(f"{self.cell.get_fqcn()}: cannot report to parent {parent_name}")

    def _cell_connected(self, connected_cell: CellAgent):
        if self.cell.my_info.gen == 2:
            # I'm a job cell
            if FQCN.is_parent(connected_cell.get_fqcn(), self.cell.get_fqcn()):
                # peer is my parent - ask it for job participants
                self._do_child_report(connected_cell.get_fqcn())
                self.ready = True

    def send_to_job_cell(
            self,
            targets: [],
            channel: str,
            topic: str,
            request: Shareable,
            timeout: float,
            fl_ctx: FLContext,
            bulk_send=False
    ) -> dict:
        """Send request to the job cells of other target sites.

        Args:
            targets (list): list of client names that the request will be sent to
            channel (str): channel of the request
            topic (str): topic of the request
            request (Shareable): request
            timeout (float): how long to wait for result. 0 means fire-and-forget
            fl_ctx (FLContext): the FL context
            bulk_send: whether to bulk send this request

        Returns:
            A dict of Shareables
        """
        if not isinstance(request, Shareable):
            raise ValueError(f"invalid request type: expect Shareable but got {type(request)}")

        if not targets:
            raise ValueError("targets must be specified")

        if targets is not None and not isinstance(targets, list):
            raise TypeError(f"targets must be a list of str, but got {type(targets)}")

        if not isinstance(topic, str):
            raise TypeError(f"invalid topic '{topic}': expects str but got {type(topic)}")

        if not topic:
            raise ValueError("invalid topic: must not be empty")

        if not isinstance(timeout, float):
            raise TypeError(f"invalid timeout: expects float but got {type(timeout)}")

        if timeout < 0:
            raise ValueError(f"invalid timeout value {timeout}: must >= 0.0")

        if not isinstance(fl_ctx, FLContext):
            raise TypeError(f"invalid fl_ctx: expects FLContext but got {type(fl_ctx)}")

        request.set_header(ReservedHeaderKey.TOPIC, topic)
        job_id = fl_ctx.get_job_id()
        cell = self.engine.get_cell()
        assert isinstance(cell, Cell)

        target_names = []
        for t in targets:
            if not isinstance(t, str):
                raise ValueError(f"invalid target name {t}: expect str but got {type(t)}")
            if t not in target_names:
                target_names.append(t)

        target_fqcns = []
        for name in target_names:
            target_fqcns.append(FQCN.join([name, job_id]))

        cell_msg = self.new_cmi_message(fl_ctx, payload=request)
        if timeout > 0:
            cell_replies = cell.broadcast_request(
                channel=channel,
                topic=topic,
                request=cell_msg,
                targets=target_fqcns,
                timeout=timeout
            )

            replies = {}
            if cell_replies:
                for k, v in cell_replies.items():
                    assert isinstance(v, Message)
                    rc = v.get_header(MessageHeaderKey.RETURN_CODE, ReturnCode.OK)
                    client_name = FQCN.get_root(k)
                    if rc == CellReturnCode.OK:
                        result = v.payload
                        if not isinstance(result, Shareable):
                            self.logger.error(f"reply of {channel}:{topic} must be dict but got {type(result)}")
                            result = make_reply(ReturnCode.ERROR)
                        replies[client_name] = result
                    else:
                        src = self._convert_return_code(rc)
                        replies[client_name] = make_reply(src)
            return replies
        else:
            if bulk_send:
                cell.queue_message(
                    channel=channel,
                    topic=topic,
                    message=cell_msg,
                    targets=target_fqcns
                )
            else:
                cell.fire_and_forget(
                    channel=channel,
                    topic=topic,
                    message=cell_msg,
                    targets=target_fqcns,
                )
            return {}

    def send_to_parent_cell(
            self,
            channel: str,
            topic: str,
            request: Shareable,
            timeout: float,
            fl_ctx: Union[None, FLContext]
    ) -> Union[None, Shareable]:
        if not isinstance(request, Shareable):
            raise ValueError(f"invalid request type: expect Shareable but got {type(request)}")

        if not isinstance(topic, str):
            raise TypeError(f"invalid topic '{topic}': expects str but got {type(topic)}")

        if not topic:
            raise ValueError("invalid topic: must not be empty")

        if not isinstance(timeout, float):
            raise TypeError(f"invalid timeout: expects float but got {type(timeout)}")

        if timeout < 0:
            raise ValueError(f"invalid timeout value {timeout}: must >= 0.0")

        if fl_ctx and not isinstance(fl_ctx, FLContext):
            raise TypeError(f"invalid fl_ctx: expects FLContext but got {type(fl_ctx)}")

        cell = self.engine.get_cell()
        assert isinstance(cell, Cell)
        cell_msg = self.new_cmi_message(fl_ctx, payload=request)
        if timeout > 0:
            reply = cell.send_request(
                channel=channel,
                topic=topic,
                request=cell_msg,
                target=self.parent_fqcn,
                timeout=timeout
            )

            assert isinstance(reply, Message)
            rc = reply.get_header(MessageHeaderKey.RETURN_CODE, ReturnCode.OK)
            if rc == CellReturnCode.OK:
                result = reply.payload
                if result and not isinstance(result, Shareable):
                    self.logger.error(f"reply data must be Shareable but got {type(result)}")
                return result
            else:
                src = self._convert_return_code(rc)
                return make_reply(src)
        else:
            cell.fire_and_forget(
                channel=channel,
                topic=topic,
                message=cell_msg,
                targets=self.parent_fqcn,
            )
