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
from unittest.mock import MagicMock, patch

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_exception import FLCommunicationError
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.private.defs import CLIENT_REGISTRATION_REQUIRED, ClientRegMsgKey, new_cell_message
from nvflare.private.fed.client import communicator as communicator_module
from nvflare.private.fed.client.communicator import Communicator
from tests.timing_utils import isolate_time


def test_get_site_config_for_registration_from_loaded_client_config():
    site_config = {"labels": {"region": "us-east"}}
    communicator = Communicator(client_config={"client_name": "site-1", ClientRegMsgKey.SITE_CONFIG: site_config})

    assert communicator._get_site_config_for_registration(FLContext()) == site_config


def test_get_site_config_for_registration_ignores_non_dict_config():
    communicator = Communicator(client_config={"client_name": "site-1", ClientRegMsgKey.SITE_CONFIG: ["bad"]})

    assert communicator._get_site_config_for_registration(FLContext()) is None


def test_unknown_heartbeat_registration_recovers_after_server_restart():
    communicator = Communicator(client_config={"client_name": "site-1"})
    client = MagicMock(client_name="site-1", token="old-token", token_signature="old-signature", ssid="old-ssid")
    client._shutdown_lock = threading.Lock()
    engine = MagicMock(client=client)
    engine.new_context.side_effect = [FLContext(), FLContext()]
    engine.get_all_job_ids.return_value = []

    rejected = new_cell_message(
        {
            MessageHeaderKey.RETURN_CODE: ReturnCode.UNAUTHENTICATED,
            MessageHeaderKey.ERROR: CLIENT_REGISTRATION_REQUIRED,
        },
        None,
    )
    accepted = new_cell_message({MessageHeaderKey.RETURN_CODE: ReturnCode.OK}, None)
    communicator.cell = MagicMock()
    with patch.object(communicator, "client_registration", return_value=("new-token", "new-signature", "new-ssid")):
        communicator.cell.send_request.side_effect = lambda **kwargs: _heartbeat_reply(communicator, rejected, accepted)
        communicator.send_heartbeat({}, "project", "old-token", "old-ssid", "site-1", engine, interval=0)
        communicator.client_registration.assert_called_once()

    assert client.token == "new-token"
    assert client.token_signature == "new-signature"
    assert client.ssid == "new-ssid"
    assert communicator.cell.send_request.call_count == 2
    assert [args.args[0] for args in engine.fire_event.call_args_list] == [
        EventType.BEFORE_CLIENT_HEARTBEAT,
        EventType.BEFORE_CLIENT_REGISTER,
        EventType.AFTER_CLIENT_REGISTER,
        EventType.BEFORE_CLIENT_HEARTBEAT,
        EventType.AFTER_CLIENT_HEARTBEAT,
    ]
    assert engine.fire_event.call_args_list[2].args[1].get_prop(FLContextKey.CLIENT_TOKEN) == "new-token"


def test_lost_registration_reply_retries_without_another_heartbeat(monkeypatch):
    communicator = Communicator(client_config={"client_name": "site-1"})
    client = MagicMock(client_name="site-1", token="old-token")
    client._shutdown_lock = threading.Lock()
    engine = MagicMock(client=client)
    engine.new_context.side_effect = [FLContext(), FLContext(), FLContext()]
    engine.get_all_job_ids.return_value = []
    rejected = new_cell_message(
        {
            MessageHeaderKey.RETURN_CODE: ReturnCode.UNAUTHENTICATED,
            MessageHeaderKey.ERROR: CLIENT_REGISTRATION_REQUIRED,
        },
        None,
    )
    accepted = new_cell_message({MessageHeaderKey.RETURN_CODE: ReturnCode.OK}, None)
    communicator.cell = MagicMock()
    communicator.cell.send_request.side_effect = lambda **kwargs: _heartbeat_reply(communicator, rejected, accepted)
    clock = isolate_time(monkeypatch, communicator_module)
    monkeypatch.setattr(clock, "sleep", lambda _seconds: None)

    with patch.object(
        communicator,
        "client_registration",
        side_effect=[FLCommunicationError("registration reply lost"), ("new-token", "signature", "ssid")],
    ) as register:
        communicator.send_heartbeat({}, "project", "old-token", "old-ssid", "site-1", engine, interval=0)

    assert register.call_count == 2
    assert communicator.cell.send_request.call_count == 2
    assert client.token == "new-token"


def _heartbeat_reply(communicator, rejected, accepted):
    if communicator.cell.send_request.call_count == 1:
        return rejected
    communicator.heartbeat_done = True
    return accepted
