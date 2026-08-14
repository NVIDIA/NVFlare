# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from unittest.mock import MagicMock, patch

import pytest

from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReservedHeaderKey, Shareable
from nvflare.fuel.f3.cellnet.core_cell import CoreCell
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.cellnet.defs import ReturnCode as CellReturnCode
from nvflare.fuel.f3.message import Message
from nvflare.private.defs import CellMessageHeaderKeys
from nvflare.private.fed.cmi import CellMessageInterface, JobCellMessenger
from tests.unit_test.fl_context_helper import make_fl_context


def _messenger():
    cell = CoreCell.__new__(CoreCell)
    cell.add_incoming_request_filter = MagicMock()
    cell.add_outgoing_reply_filter = MagicMock()
    cell.add_outgoing_request_filter = MagicMock()
    cell.add_incoming_reply_filter = MagicMock()
    cell.broadcast_request = MagicMock()
    cell.queue_message = MagicMock()
    cell.fire_and_forget = MagicMock()
    engine = MagicMock()
    engine.get_cell.return_value = cell
    engine.new_context.return_value = FLContext()
    return JobCellMessenger(engine, "job-1"), engine, cell


def _context():
    return make_fl_context(run_num="job-1")


def test_init_registers_base_and_job_filters():
    messenger, _, cell = _messenger()

    assert cell.add_incoming_request_filter.call_count == 2
    assert cell.add_incoming_reply_filter.call_count == 2
    assert cell.add_outgoing_request_filter.call_count == 2
    assert cell.add_outgoing_reply_filter.call_count == 2
    assert messenger.job_id == "job-1"


def test_new_message_and_filters_propagate_context_properties():
    messenger, _, _ = _messenger()
    fl_ctx = _context()
    fl_ctx.set_prop(CellMessageHeaderKeys.SSID, "ssid", private=False, sticky=False)
    fl_ctx.set_prop(CellMessageHeaderKeys.PROJECT_NAME, "project", private=False, sticky=False)
    fl_ctx.set_prop(FLContextKey.CLIENT_NAME, "site-1", private=False, sticky=False)
    fl_ctx.set_prop(CellMessageHeaderKeys.TOKEN, "token", private=False, sticky=False)
    fl_ctx.set_prop("public", "value", private=False, sticky=False)
    message = messenger.new_cmi_message(fl_ctx, payload=Shareable())

    messenger._filter_outgoing_message(message)

    assert message.get_prop(messenger.PROP_KEY_FL_CTX) is fl_ctx
    assert message.get_header(messenger.HEADER_SSID) == "ssid"
    assert message.get_header(messenger.HEADER_PROJECT_NAME) == "project"
    assert message.get_header(messenger.HEADER_CLIENT_NAME) == "site-1"
    assert message.get_header(messenger.HEADER_CLIENT_TOKEN) == "token"
    assert message.get_header(messenger.HEADER_KEY_PEER_PROPS)["public"] == "value"

    incoming = Message(
        headers={messenger.HEADER_KEY_PEER_PROPS: {"peer": "prop"}},
        payload=Shareable(),
    )
    messenger._filter_incoming_message(incoming)
    peer_ctx = incoming.get_prop(messenger.PROP_KEY_PEER_CTX)
    assert peer_ctx.get_prop("peer") == "prop"
    assert incoming.payload.get_peer_props() == {"peer": "prop"}


def test_incoming_request_creates_context_and_attaches_peer():
    messenger, engine, _ = _messenger()
    message = Message(headers={messenger.HEADER_KEY_PEER_PROPS: {"peer": "value"}}, payload=Shareable())

    messenger._filter_incoming_request(message)

    attached = message.get_prop(messenger.PROP_KEY_FL_CTX)
    assert attached is engine.new_context.return_value
    assert attached.get_peer_context().get_prop("peer") == "value"


@pytest.mark.parametrize(
    "cell_rc, api_rc",
    [
        (CellReturnCode.TIMEOUT, ReturnCode.COMMUNICATION_ERROR),
        (CellReturnCode.COMM_ERROR, ReturnCode.COMMUNICATION_ERROR),
        (CellReturnCode.PROCESS_EXCEPTION, ReturnCode.EXECUTION_EXCEPTION),
        # known leak in RC_TABLE: maps to a CellReturnCode instead of an API ReturnCode
        (CellReturnCode.AUTHENTICATION_ERROR, CellReturnCode.UNAUTHENTICATED),
        ("unknown", ReturnCode.ERROR),
    ],
)
def test_convert_return_code(cell_rc, api_rc):
    assert CellMessageInterface._convert_return_code(cell_rc) == api_rc


def test_job_filters_stamp_and_validate_job_id():
    messenger, _, _ = _messenger()
    outgoing = Message()
    messenger._filter_outgoing(outgoing)
    assert outgoing.get_header(messenger.HEADER_JOB_ID) == "job-1"

    incoming = Message(headers={messenger.HEADER_JOB_ID: "other-job"})
    messenger._filter_incoming(incoming)
    with patch.object(messenger.logger, "error") as log_error:
        messenger._filter_incoming(incoming)
    log_error.assert_called_once()


@pytest.mark.parametrize(
    "kwargs, exception, message",
    [
        ({"request": object()}, ValueError, "invalid request type"),
        ({"targets": []}, ValueError, "targets must be specified"),
        ({"targets": "site-1"}, TypeError, "targets must be a list"),
        ({"topic": 1}, TypeError, "invalid topic"),
        ({"topic": ""}, ValueError, "must not be empty"),
        ({"timeout": 1}, TypeError, "invalid timeout"),
        ({"timeout": -1.0}, ValueError, "must >= 0.0"),
        ({"fl_ctx": object()}, TypeError, "invalid fl_ctx"),
        ({"targets": [1]}, ValueError, "invalid target name"),
    ],
)
def test_send_to_cell_validates_inputs(kwargs, exception, message):
    messenger, _, _ = _messenger()
    params = {
        "targets": ["site-1"],
        "channel": "channel",
        "topic": "topic",
        "request": Shareable(),
        "timeout": 1.0,
        "fl_ctx": _context(),
    }
    params.update(kwargs)
    with pytest.raises(exception, match=message):
        messenger.send_to_cell(**params)


def test_request_reply_maps_cells_and_return_codes():
    messenger, _, cell = _messenger()
    ok_reply = Shareable()
    cell.broadcast_request.return_value = {
        "site-1.job-1": Message(headers={MessageHeaderKey.RETURN_CODE: CellReturnCode.OK}, payload=ok_reply),
        "site-2.job-1": Message(headers={MessageHeaderKey.RETURN_CODE: CellReturnCode.TIMEOUT}),
        "site-3.job-1": Message(headers={MessageHeaderKey.RETURN_CODE: CellReturnCode.OK}, payload=object()),
    }
    request = Shareable()

    replies = messenger.send_to_cell(
        targets=["site-1", "site-1", "site-2", "site-3"],
        channel="channel",
        topic="topic",
        request=request,
        timeout=2.0,
        fl_ctx=_context(),
        optional=True,
    )

    assert replies["site-1"] is ok_reply
    assert replies["site-2"].get_return_code() == ReturnCode.COMMUNICATION_ERROR
    assert replies["site-3"].get_return_code() == ReturnCode.ERROR
    assert request.get_header(ReservedHeaderKey.TOPIC) == "topic"
    assert cell.broadcast_request.call_args.kwargs["targets"] == ["site-1.job-1", "site-2.job-1", "site-3.job-1"]


def test_fire_and_forget_and_bulk_queue_messages():
    messenger, _, cell = _messenger()
    params = {
        "targets": ["site-1"],
        "channel": "channel",
        "topic": "topic",
        "request": Shareable(),
        "timeout": 0.0,
        "fl_ctx": _context(),
    }

    assert messenger.send_to_cell(**params) == {}
    cell.fire_and_forget.assert_called_once()

    params["bulk_send"] = True
    assert messenger.send_to_cell(**params) == {}
    cell.queue_message.assert_called_once()
