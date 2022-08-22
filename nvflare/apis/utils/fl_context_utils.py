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

import logging

from nvflare.apis.fl_constant import FLContextKey, NonSerializableKeys
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.fuel.utils import fobs


def get_serializable_data(fl_ctx: FLContext):
    logger = logging.getLogger("fl_context_utils")
    new_fl_ctx = FLContext()
    for k, v in fl_ctx.props.items():
        if k not in NonSerializableKeys.KEYS:
            try:
                fobs.dumps(v)
                new_fl_ctx.props[k] = v
            except BaseException as e:
                logger.warning(
                    generate_log_message(fl_ctx, f"Object is not serializable (discarded): {k} - {v} Error: {str(e)}")
                )
    return new_fl_ctx


def generate_log_message(fl_ctx: FLContext, msg: str):
    _identity_ = "identity"
    _my_run = "run"
    _peer_run = "peer_run"
    _peer_name = "peer"
    _task_name = "task_name"
    _task_id = "task_id"
    _rc = "peer_rc"
    _wf = "wf"

    all_kvs = {}
    all_kvs[_identity_] = fl_ctx.get_identity_name()
    my_run = fl_ctx.get_job_id()
    if not my_run:
        my_run = "?"
    all_kvs[_my_run] = my_run

    task_name = fl_ctx.get_prop(FLContextKey.TASK_NAME, None)
    task_id = fl_ctx.get_prop(FLContextKey.TASK_ID, None)

    if task_name:
        all_kvs[_task_name] = task_name

    if task_id:
        all_kvs[_task_id] = task_id

    wf_id = fl_ctx.get_prop(FLContextKey.WORKFLOW, None)
    if wf_id is not None:
        all_kvs[_wf] = wf_id

    peer_ctx = fl_ctx.get_peer_context()
    if peer_ctx:
        if not isinstance(peer_ctx, FLContext):
            raise TypeError("peer_ctx must be an instance of FLContext, but got {}".format(type(peer_ctx)))
        peer_run = peer_ctx.get_job_id()
        if not peer_run:
            peer_run = "?"
        all_kvs[_peer_run] = peer_run

        peer_name = peer_ctx.get_identity_name()
        if not peer_name:
            peer_name = "?"
        all_kvs[_peer_name] = peer_name

    reply = fl_ctx.get_prop(FLContextKey.REPLY, None)
    if isinstance(reply, Shareable):
        rc = reply.get_return_code("OK")
        all_kvs[_rc] = rc

    item_order = [_identity_, _my_run, _wf, _peer_name, _peer_run, _rc, _task_name, _task_id]
    ctx_items = []
    for item in item_order:
        if item in all_kvs:
            ctx_items.append(item + "=" + str(all_kvs[item]))

    return "[" + ", ".join(ctx_items) + "]: " + msg
