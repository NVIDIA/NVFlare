# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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
from nvflare.collab.api.app import App
from nvflare.collab.api.call_utils import check_call_args
from nvflare.collab.api.constants import CollabMethodArgName
from nvflare.collab.api.context import get_call_context, set_call_context
from nvflare.collab.api.decorators import adjust_kwargs
from nvflare.fuel.f3.cellnet.cell import Adapter
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.fuel.f3.cellnet.utils import new_cell_message
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.streaming.stream_types import StreamError
from nvflare.security.logging import secure_format_exception, secure_format_traceback

from .defs import CALL_PROTOCOL_VERSION, MSG_CHANNEL, MSG_TOPIC, CallHeaderKey, CallReplyKey, ObjectCallKey


class CollabCallAuthorizer:
    """Authorizes a Collab stream before its serialized body is received.

    The participant lookup relies on CellNet's authenticated ORIGIN. Secure
    deployments bind that identity to connection credentials; insecure CellNet
    deployments do not provide the same identity guarantee.
    """

    def __init__(self, app: App, local_fqcn: str, participants: dict[str, str], logger):
        self.app = app
        self.local_fqcn = local_fqcn
        self.participants = dict(participants)
        self.logger = logger

    def _reject(self, origin: str, reason: str) -> StreamError:
        self.logger.warning(f"rejected Collab call from {origin or '<missing>'}: {reason}")
        return StreamError("Collab call rejected")

    def authorize(self, headers: dict) -> StreamError | None:
        origin = headers.get(MessageHeaderKey.ORIGIN)
        caller = self.participants.get(origin)
        if not caller:
            return self._reject(origin, "origin is not a participant in this job")

        destination = headers.get(MessageHeaderKey.DESTINATION)
        if destination != self.local_fqcn:
            return self._reject(origin, "destination does not match this job cell")

        envelope_headers = (
            CallHeaderKey.PROTOCOL_VERSION,
            CallHeaderKey.TARGET_NAME,
            CallHeaderKey.METHOD_NAME,
        )
        if all(headers.get(key) is None for key in envelope_headers):
            return self._reject(origin, "missing Collab call envelope; peer may be running an older NVFlare version")

        version = headers.get(CallHeaderKey.PROTOCOL_VERSION)
        if version != CALL_PROTOCOL_VERSION:
            return self._reject(origin, f"unsupported Collab protocol version {version!r}")

        target_name = headers.get(CallHeaderKey.TARGET_NAME)
        if not isinstance(target_name, str) or not target_name:
            return self._reject(origin, "missing or invalid target name")

        target_parts = target_name.split(".")
        if len(target_parts) not in (1, 2) or target_parts[0] != self.app.name or any(not p for p in target_parts):
            return self._reject(origin, "target does not match the receiving application")

        method_name = headers.get(CallHeaderKey.METHOD_NAME)
        if not isinstance(method_name, str) or not method_name:
            return self._reject(origin, "missing or invalid method name")

        # Always overwrite a value supplied by the sender. Only this local
        # preflight result is trusted by the application dispatch path.
        headers[CallHeaderKey.AUTHENTICATED_CALLER] = caller
        return None


def make_participant_map(
    server_fqcn: str,
    job_id: str,
    clients,
) -> dict[str, str]:
    """Build an exact job-cell FQCN to logical Collab name mapping."""
    if not isinstance(server_fqcn, str) or not server_fqcn:
        raise ValueError("server FQCN must be a non-empty string")
    if not isinstance(job_id, str) or not job_id:
        raise ValueError("job ID must be a non-empty string")
    participants = {server_fqcn: "server"}
    for client in clients:
        client_job_fqcn = FQCN.join([client.get_fqcn(), job_id])
        if not isinstance(client_job_fqcn, str) or not client_job_fqcn:
            raise ValueError(f"missing job-cell FQCN for client {client.name}")
        existing = participants.get(client_job_fqcn)
        if existing and existing != client.name:
            raise ValueError(f"job-cell FQCN {client_job_fqcn} maps to both {existing} and {client.name}")
        participants[client_job_fqcn] = client.name
    return participants


def prepare_for_remote_call(cell, app, logger, executor, participants: dict[str, str]):
    logger.info(f"register cb for cell {cell.get_fqcn()}: {type(cell)}")
    authorizer = CollabCallAuthorizer(app, cell.get_fqcn(), participants, logger)
    adapter = Adapter(_submit_app_method, cell.core_cell.my_info, cell)
    cell.register_blob_cb(
        channel=MSG_CHANNEL,
        topic=MSG_TOPIC,
        blob_cb=adapter.call,
        preflight_cb=authorizer.authorize,
        app=app,
        logger=logger,
        executor=executor,
    )
    logger.info(f"registered request CB for {MSG_CHANNEL}/{MSG_TOPIC}")


def _submit_app_method(request: Message, app: App, logger, executor):
    """Move user code off the shared Cell/stream callback executor."""
    try:
        return executor.submit(_call_app_method, request, app, logger)
    except RuntimeError:
        return _error_reply(
            "cannot process remote call because the Collab runtime is shutting down",
            logger,
            error_type=RuntimeError.__name__,
        )


def _error_reply(error: str, logger, error_type: str = None, traceback_text: str = None) -> Message:
    logger.error(error)
    payload = {CallReplyKey.ERROR: error}
    if error_type:
        payload[CallReplyKey.ERROR_TYPE] = error_type
    if traceback_text:
        payload[CallReplyKey.ERROR_TRACEBACK] = traceback_text
    return new_cell_message(
        headers={MessageHeaderKey.RETURN_CODE: ReturnCode.PROCESS_EXCEPTION},
        payload=payload,
    )


def _preprocess(app: App, caller, target_obj_name, func_name, func, args, kwargs):
    callee = f"{app.name}.{target_obj_name}" if target_obj_name else app.name
    ctx = app.new_context(caller=caller, callee=callee)

    # make sure the final kwargs conforms to func interface
    obj_itf = app.get_target_object_publish_interface(target_obj_name)
    if not obj_itf:
        raise RuntimeError(f"cannot find collab interface for object {target_obj_name}")

    func_itf = obj_itf.get_method(func_name)
    if func_itf is None:
        raise RuntimeError(f"cannot find interface for func '{func_name}' of object {target_obj_name}")

    check_call_args(func_name, func_itf, args, kwargs)

    kwargs[CollabMethodArgName.CONTEXT] = ctx
    adjust_kwargs(func, kwargs)
    call_args, kwargs = func_itf.prepare_invocation(kwargs)
    return ctx, call_args, kwargs


def _call_app_method(request: Message, app: App, logger) -> Message:
    logger.debug("got a remote call")
    payload = request.payload
    if not isinstance(payload, dict):
        raise RuntimeError(f"request payload must be dict but got {type(payload)}")

    caller = request.get_header(CallHeaderKey.AUTHENTICATED_CALLER)
    if not isinstance(caller, str) or not caller:
        return _error_reply("missing authenticated caller from call", logger)

    payload_caller = payload.get(ObjectCallKey.CALLER)
    if payload_caller is not None and payload_caller != caller:
        return _error_reply("payload caller does not match authenticated origin", logger)

    method_name = request.get_header(CallHeaderKey.METHOD_NAME)
    if not isinstance(method_name, str) or not method_name:
        return _error_reply(f"missing or invalid '{CallHeaderKey.METHOD_NAME}' from call", logger)
    payload_method_name = payload.get(ObjectCallKey.METHOD_NAME)
    if payload_method_name is not None and payload_method_name != method_name:
        return _error_reply("payload method name does not match authenticated call envelope", logger)

    target_name = request.get_header(CallHeaderKey.TARGET_NAME)
    if not isinstance(target_name, str):
        return _error_reply(
            f"bad '{CallHeaderKey.TARGET_NAME}' from call: expect str but got {type(target_name)}",
            logger,
        )
    payload_target_name = payload.get(ObjectCallKey.TARGET_NAME)
    if payload_target_name is not None and payload_target_name != target_name:
        return _error_reply("payload target name does not match authenticated call envelope", logger)

    method_args = payload.get(ObjectCallKey.ARGS)
    if not method_args:
        method_args = []
    elif not isinstance(method_args, (list, tuple)):
        return _error_reply(f"bad method args: should be list/tuple but got {type(method_args)}", logger)
    elif method_args:
        # Proxy.adjust_func_args normalizes all positional arguments to named
        # arguments before transport so filters and validation can identify
        # them reliably. Reject malformed callers that bypass that invariant.
        return _error_reply("bad method args: positional arguments must be normalized to kwargs", logger)

    method_kwargs = payload.get(ObjectCallKey.KWARGS)
    if not method_kwargs:
        method_kwargs = {}
    elif not isinstance(method_kwargs, dict):
        return _error_reply(f"bad method kwargs: should be dict but got {type(method_kwargs)}", logger)

    parts = target_name.split(".")
    if len(parts) not in (1, 2) or parts[0] != app.name or any(not p for p in parts):
        return _error_reply(f"target '{target_name}' does not match receiving app '{app.name}'", logger)
    obj_name = ""
    if len(parts) == 2:
        obj_name = parts[1]
    if obj_name:
        target_objs = app.get_collab_objects()
        target_obj = target_objs.get(obj_name)
        logger.debug(f"calling target obj: {app.name}.{obj_name}")
    else:
        target_obj = app
        logger.debug(f"calling target app: {app.name}")

    if not target_obj:
        return _error_reply(f"no object named '{target_name}'", logger)

    m = app.find_collab_method(target_obj, method_name)
    if not m:
        return _error_reply(f"no method named '{method_name}' or it is not collab", logger)
    else:
        logger.debug(f"found method for {method_name}")

    # invoke this method
    previous_ctx = get_call_context()
    try:
        _, method_args, method_kwargs = _preprocess(app, caller, obj_name, method_name, m, method_args, method_kwargs)
        result = m(*method_args, **method_kwargs)

        return new_cell_message(
            headers={MessageHeaderKey.RETURN_CODE: ReturnCode.OK}, payload={CallReplyKey.RESULT: result}
        )
    except Exception as ex:
        traceback_text = secure_format_traceback()
        logger.error(traceback_text)
        return _error_reply(
            secure_format_exception(ex),
            logger,
            error_type=type(ex).__name__,
            traceback_text=traceback_text,
        )
    finally:
        set_call_context(previous_ctx)
