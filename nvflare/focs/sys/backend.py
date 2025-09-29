# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from nvflare.focs.api.backend import Backend
from nvflare.focs.api.constants import CollabMethodArgName, CollabMethodOptionName
from nvflare.focs.api.resp import Resp
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.utils import new_cell_message
from nvflare.fuel.f3.message import Message
from nvflare.fuel.utils.log_utils import get_obj_logger

from .constants import MSG_CHANNEL, MSG_TOPIC, CallReplyKey, ObjectCallKey


class SysBackend(Backend):

    def __init__(self, caller, cell, target_fqcn, abort_signal, thread_executor):
        Backend.__init__(self, abort_signal)
        self.logger = get_obj_logger(self)
        self.caller = caller
        self.cell = cell
        self.target_fqcn = target_fqcn
        self.thread_executor = thread_executor

    def call_target(self, target_name: str, func_name: str, *args, **kwargs):
        blocking = kwargs.pop(CollabMethodOptionName.BLOCKING, True)
        timeout = kwargs.pop(CollabMethodOptionName.TIMEOUT, 10.0)
        kwargs.pop(CollabMethodArgName.CONTEXT, None)

        payload = {
            ObjectCallKey.CALLER: self.caller,
            ObjectCallKey.TARGET_NAME: target_name,
            ObjectCallKey.METHOD_NAME: func_name,
            ObjectCallKey.ARGS: args,
            ObjectCallKey.KWARGS: kwargs,
        }
        request = new_cell_message({}, payload)

        if blocking:
            self.logger.info(f"send_request from {self.cell.get_fqcn()} to {self.target_fqcn}: {payload=} {timeout=}")

            reply = self.cell.send_request(
                channel=MSG_CHANNEL,
                target=self.target_fqcn,
                topic=MSG_TOPIC,
                request=request,
                timeout=timeout,
                secure=False,
                optional=False,
                abort_signal=self.abort_signal,
            )
            assert isinstance(reply, Message)
            rc = reply.get_header(MessageHeaderKey.RETURN_CODE, ReturnCode.OK)
            if rc == ReturnCode.TIMEOUT:
                raise TimeoutError(f"function {func_name} timed out after {timeout} seconds")
            elif rc != ReturnCode.OK:
                raise RuntimeError(f"function {func_name} failed: {rc}")

            if not isinstance(reply.payload, dict):
                raise RuntimeError(f"function {func_name} failed: reply must be dict but got {type(reply.payload)}")

            error = reply.payload.get(CallReplyKey.ERROR)
            if error:
                raise RuntimeError(f"function {func_name} failed: {error}")

            result = reply.payload.get(CallReplyKey.RESULT)
            self.logger.info(f"got result from {self.target_fqcn}: {result}")
            return result
        else:
            # fire and forget
            self.logger.info(f"fire_and_forget from {self.cell.get_fqcn()} to {self.target_fqcn}")
            self.cell.fire_and_forget(
                channel=MSG_CHANNEL,
                topic=MSG_TOPIC,
                targets=self.target_fqcn,
                message=request,
                secure=False,
                optional=False,
            )

    def call_target_with_resp(self, resp: Resp, target_name: str, func_name: str, *args, **kwargs):
        self.thread_executor.submit(self._run_func, resp, target_name, func_name, args, kwargs)

    def _run_func(self, resp: Resp, target_name: str, func_name: str, args, kwargs):
        try:
            result = self.call_target(target_name, func_name, *args, **kwargs)
            resp.set_result(result)
        except Exception as ex:
            resp.set_exception(ex)
