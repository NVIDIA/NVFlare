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

from typing import Union, Dict

from .cmi import CellMessageInterface

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.fuel.f3.cellnet.cell import MessageHeaderKey, FQCN, ReturnCode, TargetMessage, Message


class RootCellMessageInterface(CellMessageInterface):

    def __init__(self, engine):
        CellMessageInterface.__init__(self, engine)

    def _make_job_cell_fqcn(self, job_id: str):
        return FQCN.join([self.cell.get_fqcn(), job_id])

    def is_job_cell_reachable(self, job_id):
        return self.cell.is_cell_reachable(self._make_job_cell_fqcn(job_id))

    def send_to_job_cell(
            self,
            job_id: str,
            channel: str,
            topic: str,
            fl_ctx: Union[None, FLContext],
            headers: Union[None, dict],
            payload: Union[None, Shareable],
            timeout: float
    ) -> Union[None, Shareable]:
        if not payload:
            payload = Shareable()
        cell_msg = self.new_cmi_message(fl_ctx, payload=payload, headers=headers)
        job_cell_fqcn = self._make_job_cell_fqcn(job_id)
        if timeout <= 0:
            # fire and forget
            self.cell.fire_and_forget(
                channel=channel,
                topic=topic,
                message=cell_msg,
                targets=[job_cell_fqcn],
            )
        else:
            reply = self.cell.send_request(
                channel=channel,
                topic=topic,
                request=cell_msg,
                target=job_cell_fqcn,
                timeout=timeout
            )
            rc = reply.get_header(MessageHeaderKey.RETURN_CODE, ReturnCode.OK)
            if rc == ReturnCode.OK:
                result = reply.payload
                if result and not isinstance(result, Shareable):
                    self.logger.error(f"reply data must be Shareable but got {type(result)}")
                return result
            else:
                src = self._convert_return_code(rc)
                return make_reply(src)

    def send_to_server(
            self,
            channel: str,
            topic: str,
            fl_ctx: Union[None, FLContext],
            headers: dict,
            payload: Union[None, Shareable],
            timeout: float) -> Union[None, Shareable]:
        if not payload:
            payload = Shareable()
        cell_msg = self.new_cmi_message(fl_ctx, payload=payload, headers=headers)
        if timeout <= 0:
            # fire and forget
            self.cell.fire_and_forget(
                channel=channel,
                topic=topic,
                message=cell_msg,
                targets=[FQCN.ROOT_SERVER],
            )
        else:
            reply = self.cell.send_request(
                channel=channel,
                topic=topic,
                request=cell_msg,
                target=FQCN.ROOT_SERVER,
                timeout=timeout
            )
            rc = reply.get_header(MessageHeaderKey.RETURN_CODE, ReturnCode.OK)
            if rc == ReturnCode.OK:
                result = reply.payload
                if result and not isinstance(result, Shareable):
                    self.logger.error(f"reply data must be Shareable but got {type(result)}")
                return result
            else:
                src = self._convert_return_code(rc)
                return make_reply(src)

    def send_client_messages(
            self,
            channel: str,
            topic: str,
            client_messages: Dict[str, Shareable],
            timeout: float,
            fl_ctx: Union[None, FLContext]
    ) -> Dict[str, Shareable]:
        target_msgs = {}
        for client_name, req in client_messages.items():
            target_msgs[client_name] = TargetMessage(
                target=client_name,
                channel=channel,
                topic=topic,
                message=self.new_cmi_message(fl_ctx=fl_ctx, payload=req)
            )

        if not target_msgs:
            return {}

        if timeout <= 0.0:
            # this is fire-and-forget!
            self.cell.fire_multi_requests_and_forget(target_msgs)
            return {}
        else:
            result = {}
            replies = self.cell.broadcast_multi_requests(target_msgs, timeout)
            for name, reply in replies:
                assert isinstance(reply, Message)
                rc = reply.get_header(MessageHeaderKey.RETURN_CODE, ReturnCode.OK)
                if rc == ReturnCode.OK:
                    result[name] = reply.payload
                else:
                    src = self._convert_return_code(rc)
                    result[name] = make_reply(src)
            return result
