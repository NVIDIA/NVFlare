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
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable, make_reply
from nvflare.edge.constants import EdgeTaskHeaderKey
from nvflare.security.logging import secure_format_exception

TOPIC_PREFIX = "SAGE"


def message_topic_for_task_report(task_name: str) -> str:
    return f"{TOPIC_PREFIX}__{task_name}_report"


def message_topic_for_task_end(task_name: str) -> str:
    return f"{TOPIC_PREFIX}__{task_name}_end"


def make_aggr_reply(rc: str, seq: int) -> Shareable:
    return make_reply(rc, {EdgeTaskHeaderKey.TASK_SEQ: seq})


def process_aggr_result_from_child(
    processor: FLComponent,
    request: Shareable,
    current_task_seq: int,
    fl_ctx: FLContext,
    accept_f,
    **kwargs,
) -> (bool, Shareable):
    """Process aggregation report sent from a child client.

    Args:
        processor: the component that received the report from the child.
        request: the report request
        current_task_seq: sequence number of the current task
        fl_ctx: FLContext object
        accept_f: the function to be called to accept the aggregation report
        **kwargs: args to be passed to accept_f

    Returns: a tuple of (whether the report is accepted, reply to be sent back to the reporter).

    """
    peer_ctx = fl_ctx.get_peer_context()
    assert isinstance(peer_ctx, FLContext)
    child_name = peer_ctx.get_identity_name()

    task_seq = request.get_header(EdgeTaskHeaderKey.TASK_SEQ)
    if not task_seq:
        processor.log_error(fl_ctx, f"missing {EdgeTaskHeaderKey.TASK_SEQ} from aggr result header")
        return False, make_aggr_reply(ReturnCode.BAD_REQUEST_DATA, current_task_seq)

    if task_seq != current_task_seq:
        rc = ReturnCode.TASK_ABORTED
    else:
        rc = ReturnCode.OK

    if task_seq != current_task_seq:
        if current_task_seq == 0:
            # this means no current task
            processor.log_warning(
                fl_ctx, f"dropped aggr result from {child_name}: got task seq {task_seq} but no current task"
            )
        else:
            processor.log_warning(
                fl_ctx, f"dropped aggr result from {child_name}: expect task seq {current_task_seq} but got {task_seq}"
            )
        return False, make_reply(rc, current_task_seq)

    has_aggr_data = request.get_header(EdgeTaskHeaderKey.HAS_AGGR_DATA)

    if has_aggr_data is None:
        processor.log_info(fl_ctx, f"request does not have header {EdgeTaskHeaderKey.HAS_AGGR_DATA}")

    processor.log_info(fl_ctx, f"result has aggr data: {has_aggr_data=}")
    if not has_aggr_data:
        return False, make_reply(rc, current_task_seq)

    try:
        accepted = accept_f(request, fl_ctx, **kwargs)
    except Exception as ex:
        processor.log_error(fl_ctx, f"exception accepting aggr result: {secure_format_exception(ex)}")
        accepted = False

    return accepted, make_aggr_reply(rc, current_task_seq)
