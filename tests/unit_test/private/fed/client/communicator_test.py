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

from unittest.mock import MagicMock

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext, FLContextManager
from nvflare.apis.signal import Signal
from nvflare.fuel.f3.cellnet.defs import ReturnCode
from nvflare.fuel.f3.cellnet.utils import make_reply
from nvflare.private.defs import ClientRegMsgKey
from nvflare.private.fed.client.communicator import Communicator


def test_get_site_config_for_registration_from_loaded_client_config():
    site_config = {"labels": {"region": "us-east"}}
    communicator = Communicator(client_config={"client_name": "site-1", ClientRegMsgKey.SITE_CONFIG: site_config})

    assert communicator._get_site_config_for_registration(FLContext()) == site_config


def test_get_site_config_for_registration_ignores_non_dict_config():
    communicator = Communicator(client_config={"client_name": "site-1", ClientRegMsgKey.SITE_CONFIG: ["bad"]})

    assert communicator._get_site_config_for_registration(FLContext()) is None


def test_pull_task_process_exception_aborts_run(monkeypatch):
    engine = MagicMock()
    abort_signal = Signal()
    fl_ctx = FLContextManager(engine=engine, identity_name="site-1", job_id="job-1").new_context()
    fl_ctx.set_prop(FLContextKey.RUN_ABORT_SIGNAL, abort_signal, private=True, sticky=False)

    error = "Declared blob size 2097152 exceeds configured limit 1048576"
    cell = MagicMock()
    cell.send_request.return_value = make_reply(ReturnCode.PROCESS_EXCEPTION, error=error)
    communicator = Communicator(client_config={"client_name": "site-1"}, cell=cell)
    communicator.engine = engine
    monkeypatch.setattr("nvflare.private.fed.client.communicator.determine_parent_fqcn", lambda *_args: "server")

    task = communicator.pull_task("project", "token", "ssid", fl_ctx)

    assert task is None
    assert abort_signal.triggered
    assert error in fl_ctx.get_prop(FLContextKey.EVENT_DATA)
    engine.fire_event.assert_called_once_with(EventType.FATAL_SYSTEM_ERROR, fl_ctx)
    assert cell.send_request.call_args.kwargs["target"] == "server.job-1"
