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
from concurrent.futures import CancelledError

from nvflare.collab.api._invocation import _InvocationDispatcher
from nvflare.collab.api.call_opt import CallOption
from nvflare.collab.api.context import get_call_context, set_call_context
from nvflare.collab.api.exceptions import CollabCallError
from nvflare.collab.api.group_call_context import GroupCallContext
from nvflare.collab.runtime.defs import MSG_CHANNEL, MSG_TOPIC, CallReplyKey, ObjectCallKey
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.utils import new_cell_message
from nvflare.fuel.f3.message import Message
from nvflare.security.logging import secure_log_traceback


class CellDispatcher(_InvocationDispatcher):

    def __init__(self, manager, engine, caller, cell, target_fqcn, abort_signal, thread_executor):
        _InvocationDispatcher.__init__(self, abort_signal)
        self.manager = manager
        self.engine = engine
        self.caller = caller
        self.cell = cell
        self.target_fqcn = target_fqcn
        self.thread_executor = thread_executor

    def _call_target(
        self,
        context,
        target_name: str,
        call_opt: CallOption,
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
            self.logger.info(f"send_request from {self.cell.get_fqcn()} to {self.target_fqcn}: {func_name=} {call_opt}")

            reply = self.cell.send_request(
                channel=MSG_CHANNEL,
                target=self.target_fqcn,
                topic=MSG_TOPIC,
                request=request,
                timeout=timeout,
                secure=call_opt.secure,
                optional=call_opt.optional,
                abort_signal=self.abort_signal,
            )
            if not isinstance(reply, Message):
                self.logger.error(f"cell message reply must be Message but got {type(reply)}")
                raise RuntimeError(f"function {func_name} failed with internal error")

            rc = reply.get_header(MessageHeaderKey.RETURN_CODE, ReturnCode.OK)
            if rc == ReturnCode.TIMEOUT:
                cause = TimeoutError(f"function {func_name} timed out after {timeout} seconds")
                raise CollabCallError(target_name, func_name, cause)
            elif rc != ReturnCode.OK:
                error = None
                error_type = None
                error_traceback = None
                if isinstance(reply.payload, dict):
                    error = reply.payload.get(CallReplyKey.ERROR)
                    error_type = reply.payload.get(CallReplyKey.ERROR_TYPE)
                    error_traceback = reply.payload.get(CallReplyKey.ERROR_TRACEBACK)
                cause = error or f"remote call returned {rc=}"
                raise CollabCallError(
                    target_name,
                    func_name,
                    cause,
                    cause_type=error_type,
                    remote_traceback=error_traceback,
                )

            if not isinstance(reply.payload, dict):
                raise RuntimeError(f"function {func_name} failed: reply must be dict but got {type(reply.payload)}")

            error = reply.payload.get(CallReplyKey.ERROR)
            if error:
                raise CollabCallError(
                    target_name,
                    func_name,
                    error,
                    cause_type=reply.payload.get(CallReplyKey.ERROR_TYPE),
                    remote_traceback=reply.payload.get(CallReplyKey.ERROR_TRACEBACK),
                )

            result = reply.payload.get(CallReplyKey.RESULT)
            self.logger.info(f"got result from {self.target_fqcn} {func_name=}")
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
        future = self.thread_executor.submit(self._run_func, gcc, func_name, args, kwargs)
        future.add_done_callback(lambda done: self._group_call_done(done, gcc, func_name))

    @staticmethod
    def _group_call_done(future, gcc: GroupCallContext, func_name: str):
        if future.cancelled():
            gcc.set_exception(CancelledError(f"function {func_name} was cancelled before execution"))
            gcc.call_completed()

    def _run_func(self, gcc: GroupCallContext, func_name: str, args, kwargs):
        previous_ctx = get_call_context()
        try:
            result = self._call_target(
                context=gcc.context,
                target_name=gcc.target_name,
                call_opt=gcc.call_opt,
                func_name=func_name,
                *args,
                **kwargs,
            )
            gcc.set_result(result)
        except Exception as ex:
            gcc.set_exception(ex)
        finally:
            set_call_context(previous_ctx)
            gcc.call_completed()

    def handle_exception(self, exception: Exception):
        fl_ctx = self.engine.new_context()
        secure_log_traceback(self.logger)
        self.manager.system_panic(f"exception occurred: {exception}", fl_ctx)
