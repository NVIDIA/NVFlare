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
from nvflare.focs.api.app import App
from nvflare.focs.api.constants import CollabMethodArgName
from nvflare.focs.api.utils import check_context_support
from nvflare.fuel.f3.cellnet.utils import new_cell_message
from nvflare.fuel.f3.message import Message

from .constants import MSG_CHANNEL, CallReplyKey, ObjectCallKey


def prepare_for_remote_call(cell, app, logger):
    cell.register_request_cb(channel=MSG_CHANNEL, topic="*", cb=_call_app_method, app=app, logger=logger)


def _error_reply(error: str, logger) -> Message:
    logger.error(error)
    return new_cell_message(headers={}, payload={CallReplyKey.ERROR: error})


def _call_app_method(request: Message, app: App, logger) -> Message:
    logger.info(f"got call")
    payload = request.payload
    assert isinstance(payload, dict)

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
    elif not isinstance(method_args, list):
        return _error_reply(f"bad method args: should be list but got {type(method_args)}", logger)

    method_kwargs = payload.get(ObjectCallKey.KWARGS)
    if not method_kwargs:
        method_kwargs = {}
    elif not isinstance(method_kwargs, dict):
        return _error_reply(f"bad method kwargs: should be dict but got {type(method_kwargs)}", logger)

    parts = target_name.split(".")
    obj_name = None
    if len(parts) >= 2:
        obj_name = parts[1]
    if obj_name:
        target_objs = app.get_target_objects()
        target_obj = target_objs.get(obj_name)
    else:
        target_obj = app
    if not target_obj:
        return _error_reply(f"no object named '{target_name}'", logger)

    m = app.find_collab_method(target_obj, method_name)
    if not m:
        return _error_reply(f"no method named '{method_name}'", logger)

    # invoke this method
    try:
        ctx = app.new_context(caller=caller, callee=app.name)
        method_kwargs[CollabMethodArgName.CONTEXT] = ctx
        check_context_support(m, method_kwargs)
        result = m(*method_args, **method_kwargs)
        return new_cell_message(headers={}, payload={CallReplyKey.RESULT: result})
    except Exception as ex:
        return _error_reply(f"exception {type(ex)}", logger)
