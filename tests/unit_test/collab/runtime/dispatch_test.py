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

from nvflare.collab.api.app import ClientApp
from nvflare.collab.api.context import get_call_context, set_call_context
from nvflare.collab.api.decorators import publish
from nvflare.collab.runtime.defs import CallReplyKey, ObjectCallKey
from nvflare.collab.runtime.dispatch import _call_app_method
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.utils import new_cell_message


class _FailingClient:
    @publish
    def fail(self):
        raise ValueError("invalid input")


class _SuccessfulClient:
    @publish
    def succeed(self):
        return "result"


def test_remote_call_returns_secure_exception_detail():
    app = ClientApp(_FailingClient())
    app.name = "site-1"
    request = new_cell_message(
        {},
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
        {},
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


def test_remote_call_rejects_unnormalized_positional_args():
    app = ClientApp(_FailingClient())
    app.name = "site-1"
    request = new_cell_message(
        {},
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
