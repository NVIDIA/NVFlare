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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from nvflare.private.admin_defs import Message
from nvflare.private.fed.resource_stats.f3_counter import F3Counter, F3TrafficClass
from nvflare.private.fed.server.message_send import send_requests


def _clients():
    return {
        "token-1": SimpleNamespace(token="token-1", name="site-1", fqcn="site-1"),
        "token-2": SimpleNamespace(token="token-2", name="site-2", fqcn="site-2"),
    }


def _requests():
    # Deployment currently reuses the same inner admin message for fan-out.
    # The outer Cell messages must still receive independent contexts.
    request = Message(topic="deploy", body=b"application")
    return {"token-1": request, "token-2": request}


def test_trusted_admin_fanout_creates_one_context_per_destination():
    cell = MagicMock()
    cell.broadcast_multi_requests.return_value = {}
    counter = F3Counter()

    assert (
        send_requests(
            cell=cell,
            command="admin",
            requests=_requests(),
            clients=_clients(),
            logical_send_accounting=counter,
            logical_send_traffic_class=F3TrafficClass.JOB_APPLICATION,
        )
        == []
    )

    target_messages = cell.broadcast_multi_requests.call_args.args[0]
    assert set(target_messages) == {"site-1", "site-2"}
    contexts = [target_messages[target].message.get_logical_send_context() for target in sorted(target_messages)]
    assert all(context is not None for context in contexts)
    assert contexts[0] is not contexts[1]
    assert all(context._accounting is counter for context in contexts)
    assert all(context._traffic_class is F3TrafficClass.JOB_APPLICATION for context in contexts)
    assert [context._recipient_name for context in contexts] == ["site-1", "site-2"]


def test_generic_admin_route_does_not_infer_a_traffic_class():
    cell = MagicMock()
    cell.broadcast_multi_requests.return_value = {}

    send_requests(cell=cell, command="admin", requests=_requests(), clients=_clients())

    target_messages = cell.broadcast_multi_requests.call_args.args[0]
    assert all(message.message.get_logical_send_context() is None for message in target_messages.values())


def test_attachment_failure_does_not_prevent_real_admin_send():
    cell = MagicMock()
    cell.broadcast_multi_requests.return_value = {}

    with patch("nvflare.private.fed.server.message_send.attach_f3_context", return_value=False):
        replies = send_requests(
            cell=cell,
            command="admin",
            requests=_requests(),
            clients=_clients(),
            logical_send_accounting=F3Counter(),
            logical_send_traffic_class=F3TrafficClass.JOB_APPLICATION,
        )

    assert replies == []
    cell.broadcast_multi_requests.assert_called_once()
