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

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from nvflare.collab.api.app import ClientApp
from nvflare.collab.api.context import get_call_context, set_call_context
from nvflare.collab.api.decorators import publish
from nvflare.collab.runtime.defs import (
    CALL_PROTOCOL_VERSION,
    MSG_CHANNEL,
    MSG_TOPIC,
    CallHeaderKey,
    CallReplyKey,
    ObjectCallKey,
)
from nvflare.collab.runtime.dispatch import (
    CollabCallAuthorizer,
    _call_app_method,
    _CollabStreamFilter,
    _submit_app_method,
    make_participant_map,
    prepare_for_remote_call,
)
from nvflare.fuel.f3.cellnet.cell import Adapter
from nvflare.fuel.f3.cellnet.defs import CellChannel, MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.utils import new_cell_message
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.streaming.stream_const import STREAM_CHANNEL, STREAM_DATA_TOPIC, StreamHeaderKey
from nvflare.fuel.f3.streaming.stream_types import StreamError


class _FailingClient:
    @publish
    def fail(self):
        raise ValueError("invalid input")


class _SuccessfulClient:
    @publish
    def succeed(self):
        return "result"


class _SignatureClient:
    @publish
    def scale(self, value, /, *, factor):
        return value * factor


class _ThreadCapturingClient:
    def __init__(self):
        self.thread_name = None

    @publish
    def run(self):
        self.thread_name = threading.current_thread().name
        return "result"


class _ContextCapturingClient:
    @publish
    def identify(self, context=None):
        return context.caller, context.callee


def _trusted_headers(target_name: str, method_name: str, caller: str = "server"):
    return {
        CallHeaderKey.AUTHENTICATED_CALLER: caller,
        CallHeaderKey.TARGET_NAME: target_name,
        CallHeaderKey.METHOD_NAME: method_name,
    }


def test_prepare_for_remote_call_registers_blob_callback():
    cell = MagicMock()
    app = MagicMock()
    logger = MagicMock()
    executor = MagicMock()
    callback = MagicMock()
    adapter = MagicMock(call=callback)
    cell.get_fqcn.return_value = "site-1.job"

    with patch("nvflare.collab.runtime.dispatch.Adapter", return_value=adapter):
        prepare_for_remote_call(cell, app, logger, executor, {"server.job": "server"})

    cell.register_blob_cb.assert_called_once()
    registration = cell.register_blob_cb.call_args.kwargs
    assert registration["channel"] == MSG_CHANNEL
    assert registration["topic"] == MSG_TOPIC
    assert registration["blob_cb"] is callback
    assert registration["app"] is app
    assert registration["logger"] is logger
    assert registration["executor"] is executor
    assert "preflight_cb" not in registration
    cell.core_cell.add_incoming_filter.assert_called_once()
    filter_registration = cell.core_cell.add_incoming_filter.call_args.args
    assert filter_registration[:2] == (STREAM_CHANNEL, STREAM_DATA_TOPIC)
    assert callable(filter_registration[2])


def test_collab_stream_filter_rejects_before_receive_state_is_created():
    authorizer = MagicMock()
    rejection = StreamError("Collab call rejected")
    authorizer.authorize.return_value = rejection
    byte_receiver = MagicMock()
    stream_filter = _CollabStreamFilter(authorizer, byte_receiver)
    message = Message(
        {
            MessageHeaderKey.ORIGIN: "server.other-job",
            StreamHeaderKey.STREAM_ID: 123,
            StreamHeaderKey.CHANNEL: MSG_CHANNEL,
            StreamHeaderKey.TOPIC: MSG_TOPIC,
        }
    )

    response = stream_filter.filter(message)

    assert isinstance(response, Message)
    authorizer.authorize.assert_called_once_with(message.headers)
    byte_receiver.reject.assert_called_once_with(message, rejection)


def test_collab_stream_filter_ignores_other_streams():
    authorizer = MagicMock()
    byte_receiver = MagicMock()
    stream_filter = _CollabStreamFilter(authorizer, byte_receiver)
    message = Message(
        {
            StreamHeaderKey.CHANNEL: "other-channel",
            StreamHeaderKey.TOPIC: "other-topic",
        }
    )

    assert stream_filter.filter(message) is None
    authorizer.authorize.assert_not_called()
    byte_receiver.reject.assert_not_called()


def test_remote_call_returns_secure_exception_detail():
    app = ClientApp(_FailingClient())
    app.name = "site-1"
    request = new_cell_message(
        _trusted_headers("site-1.client", "fail"),
        {
            ObjectCallKey.CALLER: "server",
            ObjectCallKey.TARGET_NAME: "site-1.client",
            ObjectCallKey.METHOD_NAME: "fail",
        },
    )
    logger = MagicMock()

    previous_ctx = MagicMock()
    set_call_context(previous_ctx)
    try:
        with (
            patch(
                "nvflare.collab.runtime.dispatch.secure_format_exception",
                return_value="ValueError: invalid input",
            ) as format_exception,
            patch(
                "nvflare.collab.runtime.dispatch.secure_format_traceback",
                return_value="remote traceback",
            ),
        ):
            reply = _call_app_method(request, app, logger)
        assert get_call_context() is previous_ctx
    finally:
        set_call_context(None)

    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.PROCESS_EXCEPTION
    assert reply.payload[CallReplyKey.ERROR] == "ValueError: invalid input"
    assert reply.payload[CallReplyKey.ERROR_TYPE] == "ValueError"
    assert reply.payload[CallReplyKey.ERROR_TRACEBACK] == "remote traceback"
    format_exception.assert_called_once()


def test_remote_call_restores_previous_context_after_success():
    app = ClientApp(_SuccessfulClient())
    app.name = "site-1"
    request = new_cell_message(
        _trusted_headers("site-1.client", "succeed"),
        {
            ObjectCallKey.CALLER: "server",
            ObjectCallKey.TARGET_NAME: "site-1.client",
            ObjectCallKey.METHOD_NAME: "succeed",
        },
    )
    previous_ctx = MagicMock()

    set_call_context(previous_ctx)
    try:
        reply = _call_app_method(request, app, MagicMock())
        assert get_call_context() is previous_ctx
    finally:
        set_call_context(None)

    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK
    assert reply.payload[CallReplyKey.RESULT] == "result"


def test_named_object_call_context_uses_fully_qualified_callee():
    app = ClientApp(_SuccessfulClient())
    app.name = "site-1"
    app.add_collab_object("trainer", _ContextCapturingClient())
    request = new_cell_message(
        _trusted_headers("site-1.trainer", "identify"),
        {
            ObjectCallKey.CALLER: "server",
            ObjectCallKey.TARGET_NAME: "site-1.trainer",
            ObjectCallKey.METHOD_NAME: "identify",
        },
    )

    reply = _call_app_method(request, app, MagicMock())

    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK
    assert reply.payload[CallReplyKey.RESULT] == ("server", "site-1.trainer")


def test_remote_call_restores_positional_only_arguments_for_invocation():
    app = ClientApp(_SignatureClient())
    app.name = "site-1"
    request = new_cell_message(
        _trusted_headers("site-1.client", "scale"),
        {
            ObjectCallKey.CALLER: "server",
            ObjectCallKey.TARGET_NAME: "site-1.client",
            ObjectCallKey.METHOD_NAME: "scale",
            ObjectCallKey.KWARGS: {"value": 3, "factor": 4},
        },
    )

    reply = _call_app_method(request, app, MagicMock())

    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK
    assert reply.payload[CallReplyKey.RESULT] == 12


def test_remote_call_rejects_unnormalized_positional_args():
    app = ClientApp(_FailingClient())
    app.name = "site-1"
    request = new_cell_message(
        _trusted_headers("site-1", "fail"),
        {
            ObjectCallKey.CALLER: "server",
            ObjectCallKey.TARGET_NAME: "site-1",
            ObjectCallKey.METHOD_NAME: "fail",
            ObjectCallKey.ARGS: ["unexpected"],
        },
    )

    reply = _call_app_method(request, app, MagicMock())

    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.PROCESS_EXCEPTION
    assert reply.payload[CallReplyKey.ERROR] == "bad method args: positional arguments must be normalized to kwargs"


def test_remote_user_function_is_submitted_to_collab_executor():
    client = _ThreadCapturingClient()
    app = ClientApp(client)
    app.name = "site-1"
    request = new_cell_message(
        _trusted_headers("site-1.client", "run"),
        {
            ObjectCallKey.CALLER: "server",
            ObjectCallKey.TARGET_NAME: "site-1.client",
            ObjectCallKey.METHOD_NAME: "run",
        },
    )

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="collab_call") as executor:
        result = _submit_app_method(request, app, MagicMock(), executor)
        assert isinstance(result, Future)
        reply = result.result(timeout=1.0)

    assert reply.payload[CallReplyKey.RESULT] == "result"
    assert client.thread_name.startswith("collab_call")


def test_remote_call_after_executor_shutdown_returns_error_reply():
    executor = ThreadPoolExecutor(max_workers=1)
    executor.shutdown()

    reply = _submit_app_method(MagicMock(), MagicMock(), MagicMock(), executor)

    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.PROCESS_EXCEPTION
    assert reply.payload[CallReplyKey.ERROR] == (
        "cannot process remote call because the Collab runtime is shutting down"
    )
    assert reply.payload[CallReplyKey.ERROR_TYPE] == "RuntimeError"


def test_stream_adapter_sends_future_response_asynchronously():
    response_future = Future()
    cell = MagicMock()
    cell.get_fobs_context.return_value = {}
    adapter = Adapter(lambda _request: response_future, MagicMock(), cell)
    incoming = MagicMock()
    incoming.headers = {
        StreamHeaderKey.STREAM_REQ_ID: "stream-1",
        StreamHeaderKey.CHANNEL: "collab",
        StreamHeaderKey.TOPIC: "call",
        MessageHeaderKey.ORIGIN: "server",
        MessageHeaderKey.REQ_ID: "request-1",
    }
    incoming.result.return_value = {}

    with (
        patch("nvflare.fuel.f3.cellnet.cell.decode_payload"),
        patch("nvflare.fuel.f3.cellnet.cell.encode_payload"),
    ):
        adapter.call(incoming)
        cell.send_blob.assert_not_called()

        response_future.set_result(new_cell_message({}, {"result": "done"}))

    cell.send_blob.assert_called_once()
    assert cell.send_blob.call_args.args[0] == CellChannel.RETURN_ONLY


@pytest.mark.parametrize("secure", [False, True])
def test_authorizer_derives_caller_and_overwrites_sender_value(secure):
    app = ClientApp(_SuccessfulClient())
    app.name = "site-1"
    authorizer = CollabCallAuthorizer(
        app=app,
        local_fqcn="site-1.job-1",
        participants={"server.job-1": "server"},
        logger=MagicMock(),
    )
    headers = {
        MessageHeaderKey.ORIGIN: "server.job-1",
        MessageHeaderKey.DESTINATION: "site-1.job-1",
        CallHeaderKey.PROTOCOL_VERSION: CALL_PROTOCOL_VERSION,
        CallHeaderKey.TARGET_NAME: "site-1.client",
        CallHeaderKey.METHOD_NAME: "succeed",
        CallHeaderKey.AUTHENTICATED_CALLER: "spoofed-site",
        MessageHeaderKey.SECURE: secure,
    }

    assert authorizer.authorize(headers) is None
    assert headers[CallHeaderKey.AUTHENTICATED_CALLER] == "server"


def test_authorizer_rejects_origin_from_another_job():
    app = ClientApp(_SuccessfulClient())
    app.name = "site-1"
    authorizer = CollabCallAuthorizer(
        app=app,
        local_fqcn="site-1.job-1",
        participants={"server.job-1": "server"},
        logger=MagicMock(),
    )
    headers = {
        MessageHeaderKey.ORIGIN: "server.job-2",
        MessageHeaderKey.DESTINATION: "site-1.job-1",
        CallHeaderKey.PROTOCOL_VERSION: CALL_PROTOCOL_VERSION,
        CallHeaderKey.TARGET_NAME: "site-1.client",
        CallHeaderKey.METHOD_NAME: "succeed",
    }

    response = authorizer.authorize(headers)

    assert isinstance(response, StreamError)
    assert str(response) == "Collab call rejected"
    assert CallHeaderKey.AUTHENTICATED_CALLER not in headers


def test_authorizer_rejects_wrong_target_before_dispatch():
    app = ClientApp(_SuccessfulClient())
    app.name = "site-1"
    authorizer = CollabCallAuthorizer(
        app=app,
        local_fqcn="site-1.job-1",
        participants={"server.job-1": "server"},
        logger=MagicMock(),
    )
    headers = {
        MessageHeaderKey.ORIGIN: "server.job-1",
        MessageHeaderKey.DESTINATION: "site-1.job-1",
        CallHeaderKey.PROTOCOL_VERSION: CALL_PROTOCOL_VERSION,
        CallHeaderKey.TARGET_NAME: "site-2.client.extra",
        CallHeaderKey.METHOD_NAME: "succeed",
    }

    response = authorizer.authorize(headers)

    assert isinstance(response, StreamError)


@pytest.mark.parametrize(
    "header,value",
    [
        (MessageHeaderKey.DESTINATION, "site-1.job-2"),
        (CallHeaderKey.PROTOCOL_VERSION, CALL_PROTOCOL_VERSION + 1),
        (CallHeaderKey.TARGET_NAME, "site-1.client.extra"),
        (CallHeaderKey.METHOD_NAME, ""),
    ],
)
def test_authorizer_rejects_malformed_envelope(header, value):
    app = ClientApp(_SuccessfulClient())
    app.name = "site-1"
    authorizer = CollabCallAuthorizer(
        app=app,
        local_fqcn="site-1.job-1",
        participants={"server.job-1": "server"},
        logger=MagicMock(),
    )
    headers = {
        MessageHeaderKey.ORIGIN: "server.job-1",
        MessageHeaderKey.DESTINATION: "site-1.job-1",
        CallHeaderKey.PROTOCOL_VERSION: CALL_PROTOCOL_VERSION,
        CallHeaderKey.TARGET_NAME: "site-1.client",
        CallHeaderKey.METHOD_NAME: "succeed",
    }
    headers[header] = value

    response = authorizer.authorize(headers)

    assert isinstance(response, StreamError)


def test_authorizer_diagnoses_missing_legacy_envelope():
    app = ClientApp(_SuccessfulClient())
    app.name = "site-1"
    logger = MagicMock()
    authorizer = CollabCallAuthorizer(
        app=app,
        local_fqcn="site-1.job-1",
        participants={"server.job-1": "server"},
        logger=logger,
    )
    headers = {
        MessageHeaderKey.ORIGIN: "server.job-1",
        MessageHeaderKey.DESTINATION: "site-1.job-1",
    }

    response = authorizer.authorize(headers)

    assert isinstance(response, StreamError)
    assert str(response) == "Collab call rejected"
    logger.warning.assert_called_once()
    assert "older NVFlare" in logger.warning.call_args.args[0]


def test_dispatch_rejects_payload_caller_spoofing():
    app = ClientApp(_SuccessfulClient())
    app.name = "site-1"
    request = new_cell_message(
        _trusted_headers("site-1.client", "succeed", caller="site-2"),
        {
            ObjectCallKey.CALLER: "server",
            ObjectCallKey.TARGET_NAME: "site-1.client",
            ObjectCallKey.METHOD_NAME: "succeed",
        },
    )

    reply = _call_app_method(request, app, MagicMock())

    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.PROCESS_EXCEPTION
    assert reply.payload[CallReplyKey.ERROR] == "payload caller does not match authenticated origin"


def test_make_participant_map_uses_standard_job_cell_fqcns():
    clients = [
        SimpleNamespace(name="site-1", get_fqcn=lambda: "relay.site-1"),
        SimpleNamespace(name="site-2", get_fqcn=lambda: "relay.site-2"),
    ]

    participants = make_participant_map(
        server_fqcn="server.job-1",
        job_id="job-1",
        clients=clients,
    )

    assert participants == {
        "server.job-1": "server",
        "relay.site-1.job-1": "site-1",
        "relay.site-2.job-1": "site-2",
    }
