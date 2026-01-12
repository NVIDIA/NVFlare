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
from nvflare.fox.api.app import App
from nvflare.fox.api.constants import CollabMethodArgName
from nvflare.fox.api.dec import adjust_kwargs
from nvflare.fox.api.utils import check_call_args
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.utils import new_cell_message
from nvflare.fuel.f3.message import Message

from ...security.logging import secure_log_traceback
from .constants import MSG_CHANNEL, MSG_TOPIC, CallReplyKey, ObjectCallKey


def prepare_for_remote_call(cell, app, logger):
    """Register callback for in-process method execution."""
    logger.debug(f"register cb for cell {cell.get_fqcn()}: {type(cell)}")
    cell.register_request_cb(channel=MSG_CHANNEL, topic=MSG_TOPIC, cb=_call_app_method, app=app, logger=logger)
    logger.debug(f"registered request CB for {MSG_CHANNEL}/{MSG_TOPIC}")


def prepare_for_subprocess_call(cell, app, subprocess_launcher, logger):
    """Register callback for subprocess method execution.

    Instead of executing methods locally, this forwards calls to the
    subprocess worker via SubprocessLauncher.
    """
    logger.debug(f"register subprocess cb for cell {cell.get_fqcn()}")
    cell.register_request_cb(
        channel=MSG_CHANNEL,
        topic=MSG_TOPIC,
        cb=_call_subprocess_method,
        app=app,
        subprocess_launcher=subprocess_launcher,
        logger=logger,
    )
    logger.debug(f"registered subprocess request CB for {MSG_CHANNEL}/{MSG_TOPIC}")


def _error_reply(error: str, logger) -> Message:
    logger.error(error)
    return new_cell_message(
        headers={MessageHeaderKey.RETURN_CODE: ReturnCode.PROCESS_EXCEPTION}, payload={CallReplyKey.ERROR: error}
    )


def _preprocess(app: App, caller, target_obj_name, target_name, func_name, func, args, kwargs):
    ctx = app.new_context(caller=caller, callee=app.name)
    kwargs = app.apply_incoming_call_filters(target_name, func_name, kwargs, ctx)

    # make sure the final kwargs conforms to func interface
    obj_itf = app.get_target_object_collab_interface(target_obj_name)
    if not obj_itf:
        raise RuntimeError(f"cannot find collab interface for object {target_obj_name}")

    func_itf = obj_itf.get(func_name)
    if not func_itf:
        raise RuntimeError(f"cannot find interface for func '{func_name}' of object {target_obj_name}")

    check_call_args(func_name, func_itf, args, kwargs)

    kwargs[CollabMethodArgName.CONTEXT] = ctx
    adjust_kwargs(func, kwargs)
    return ctx, kwargs


def _call_app_method(request: Message, app: App, logger) -> Message:
    logger.debug("got a remote call")
    payload = request.payload
    if not isinstance(payload, dict):
        raise RuntimeError(f"request payload must be dict but got {type(payload)}")

    caller = payload.get(ObjectCallKey.CALLER)
    if not caller:
        return _error_reply(f"missing '{ObjectCallKey.CALLER}' from call", logger)

    method_name = payload.get(ObjectCallKey.METHOD_NAME)
    if not method_name:
        return _error_reply(f"missing '{ObjectCallKey.METHOD_NAME}' from call", logger)

    target_name = payload.get(ObjectCallKey.TARGET_NAME)
    if not isinstance(target_name, str):
        return _error_reply(
            f"bad '{ObjectCallKey.TARGET_NAME}' from call: expect str but got {type(target_name)}",
            logger,
        )

    method_args = payload.get(ObjectCallKey.ARGS)
    if not method_args:
        method_args = []
    elif not isinstance(method_args, (list, tuple)):
        return _error_reply(f"bad method args: should be list/tuple but got {type(method_args)}", logger)

    method_kwargs = payload.get(ObjectCallKey.KWARGS)
    if not method_kwargs:
        method_kwargs = {}
    elif not isinstance(method_kwargs, dict):
        return _error_reply(f"bad method kwargs: should be dict but got {type(method_kwargs)}", logger)

    parts = target_name.split(".")
    obj_name = ""
    if len(parts) >= 2:
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
    try:
        ctx, method_kwargs = _preprocess(app, caller, obj_name, target_name, method_name, m, method_args, method_kwargs)
        result = m(*method_args, **method_kwargs)

        # apply result filters
        result = app.apply_outgoing_result_filters(target_name, method_name, result, ctx)

        return new_cell_message(
            headers={MessageHeaderKey.RETURN_CODE: ReturnCode.OK}, payload={CallReplyKey.RESULT: result}
        )
    except Exception as ex:
        secure_log_traceback(logger)
        return _error_reply(f"exception {type(ex)}", logger)


def _call_subprocess_method(request: Message, app: App, subprocess_launcher, logger) -> Message:
    """Handle remote call by forwarding to subprocess worker.

    This is used in subprocess mode where the actual training runs in a
    separate process (e.g., launched via torchrun for multi-GPU DDP).
    """
    logger.debug("got a remote call (forwarding to subprocess)")
    payload = request.payload
    if not isinstance(payload, dict):
        raise RuntimeError(f"request payload must be dict but got {type(payload)}")

    caller = payload.get(ObjectCallKey.CALLER)
    if not caller:
        return _error_reply(f"missing '{ObjectCallKey.CALLER}' from call", logger)

    method_name = payload.get(ObjectCallKey.METHOD_NAME)
    if not method_name:
        return _error_reply(f"missing '{ObjectCallKey.METHOD_NAME}' from call", logger)

    target_name = payload.get(ObjectCallKey.TARGET_NAME)
    if not isinstance(target_name, str):
        return _error_reply(
            f"bad '{ObjectCallKey.TARGET_NAME}' from call: expect str but got {type(target_name)}",
            logger,
        )

    method_args = payload.get(ObjectCallKey.ARGS)
    if not method_args:
        method_args = []
    elif not isinstance(method_args, (list, tuple)):
        return _error_reply(f"bad method args: should be list/tuple but got {type(method_args)}", logger)

    method_kwargs = payload.get(ObjectCallKey.KWARGS)
    if not method_kwargs:
        method_kwargs = {}
    elif not isinstance(method_kwargs, dict):
        return _error_reply(f"bad method kwargs: should be dict but got {type(method_kwargs)}", logger)

    # Forward call to subprocess worker
    try:
        logger.debug(f"forwarding {method_name} to subprocess worker")
        result = subprocess_launcher.call(method_name, args=tuple(method_args), kwargs=method_kwargs)

        # Apply result filters (still done in parent process)
        ctx = app.new_context(caller=caller, callee=app.name)
        result = app.apply_outgoing_result_filters(target_name, method_name, result, ctx)

        return new_cell_message(
            headers={MessageHeaderKey.RETURN_CODE: ReturnCode.OK}, payload={CallReplyKey.RESULT: result}
        )
    except Exception as ex:
        secure_log_traceback(logger)
        return _error_reply(f"subprocess exception: {type(ex).__name__}: {ex}", logger)
