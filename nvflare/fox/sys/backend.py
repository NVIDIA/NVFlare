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
from nvflare.fox.api.backend import Backend
from nvflare.fox.api.call_opt import CallOption
from nvflare.fox.api.ctx import set_call_context
from nvflare.fox.api.gcc import GroupCallContext
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.utils import new_cell_message
from nvflare.fuel.f3.message import Message
from nvflare.security.logging import secure_log_traceback

from .constants import MSG_CHANNEL, MSG_TOPIC, CallReplyKey, ObjectCallKey


class FlareBackend(Backend):

    def __init__(self, manager, engine, caller, cell, target_fqcn, abort_signal, thread_executor):
        Backend.__init__(self, abort_signal)
        self.manager = manager
        self.engine = engine
        self.caller = caller
        self.cell = cell
        self.target_fqcn = target_fqcn
        self.thread_executor = thread_executor

    def call_target(self, context, target_name: str, call_opt: CallOption, func_name: str, *args, **kwargs):
        return self._call_target(
            context=context,
            target_name=target_name,
            call_opt=call_opt,
            send_complete_cb=None,
            cb_kwargs={},
            func_name=func_name,
            *args,
            **kwargs,
        )

    def _call_target(
        self,
        context,
        target_name: str,
        call_opt: CallOption,
        send_complete_cb,
        cb_kwargs,
        func_name: str,
        *args,
        **kwargs,
    ):
        set_call_context(context)

        payload = {
            ObjectCallKey.CALLER: self.caller,
            ObjectCallKey.TARGET_NAME: target_name,
            ObjectCallKey.METHOD_NAME: func_name,
            ObjectCallKey.ARGS: args,
            ObjectCallKey.KWARGS: kwargs,
        }
        request = new_cell_message({}, payload)

        timeout = call_opt.timeout
        if call_opt.expect_result:
            self.logger.debug(
                f"send_request from {self.cell.get_fqcn()} to {self.target_fqcn}: {func_name=} {call_opt}"
            )

            reply = self.cell.send_request(
                channel=MSG_CHANNEL,
                target=self.target_fqcn,
                topic=MSG_TOPIC,
                request=request,
                timeout=timeout,
                secure=call_opt.secure,
                optional=call_opt.optional,
                abort_signal=self.abort_signal,
                send_complete_cb=send_complete_cb,
                **cb_kwargs,
            )
            if not isinstance(reply, Message):
                self.logger.error(f"cell message reply must be Message but got {type(reply)}")
                raise RuntimeError(f"function {func_name} failed with internal error")

            rc = reply.get_header(MessageHeaderKey.RETURN_CODE, ReturnCode.OK)
            if rc == ReturnCode.TIMEOUT:
                raise TimeoutError(f"function {func_name} timed out after {timeout} seconds")
            elif rc != ReturnCode.OK:
                error = None
                if isinstance(reply.payload, dict):
                    error = reply.payload.get(CallReplyKey.ERROR)
                raise RuntimeError(f"function {func_name} failed: {rc=} {error=}")

            if not isinstance(reply.payload, dict):
                raise RuntimeError(f"function {func_name} failed: reply must be dict but got {type(reply.payload)}")

            error = reply.payload.get(CallReplyKey.ERROR)
            if error:
                raise RuntimeError(f"function {func_name} failed: {error}")

            result = reply.payload.get(CallReplyKey.RESULT)
            self.logger.debug(f"got result from {self.target_fqcn} {func_name=}")
            return result
        else:
            # fire and forget
            self.logger.info(f"fire_and_forget from {self.cell.get_fqcn()} to {self.target_fqcn}")
            self.cell.fire_and_forget(
                channel=MSG_CHANNEL,
                topic=MSG_TOPIC,
                targets=self.target_fqcn,
                message=request,
                secure=call_opt.secure,
                optional=call_opt.optional,
            )
            return None

    def call_target_in_group(self, gcc: GroupCallContext, func_name: str, *args, **kwargs):
        self.thread_executor.submit(self._run_func, gcc, func_name, args, kwargs)

    def _run_func(self, gcc: GroupCallContext, func_name: str, args, kwargs):
        try:
            result = self._call_target(
                context=gcc.context,
                target_name=gcc.target_name,
                call_opt=gcc.call_opt,
                func_name=func_name,
                send_complete_cb=self._msg_sent,
                cb_kwargs={"gcc": gcc},
                *args,
                **kwargs,
            )
            gcc.set_result(result)
        except Exception as ex:
            gcc.set_exception(ex)

    def _msg_sent(self, gcc: GroupCallContext):
        gcc.send_completed()

    def handle_exception(self, exception: Exception):
        fl_ctx = self.engine.new_context()
        secure_log_traceback(self.logger)
        self.manager.system_panic(f"exception occurred: {exception}", fl_ctx)
