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
import warnings

import pytest

from nvflare.apis.fl_context import FLContext
from nvflare.app_opt.flower.connectors.http_client_connector import HttpClientConnector
from nvflare.app_opt.flower.connectors.http_server_connector import HttpServerConnector
from nvflare.app_opt.flower.controller import FlowerController
from nvflare.app_opt.flower.executor import FlowerExecutor


def test_controller_uses_http_server_connector():
    connector = FlowerController(num_rounds=2, monitor_interval=0.7).get_connector(FLContext())

    assert isinstance(connector, HttpServerConnector)
    assert connector.monitor_interval == 0.7


def test_controller_warns_that_grpc_options_are_ignored():
    with pytest.warns(DeprecationWarning, match="int_client_grpc_options is deprecated"):
        controller = FlowerController(int_client_grpc_options=[("grpc.max_send_message_length", 1)])

    assert isinstance(controller.get_connector(FLContext()), HttpServerConnector)


def test_controller_without_grpc_options_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        FlowerController()


def test_executor_uses_http_client_connector_with_configured_timeouts():
    executor = FlowerExecutor(per_msg_timeout=11.0, tx_timeout=222.0, client_shutdown_timeout=3.5)

    connector = executor.get_connector(FLContext())

    assert isinstance(connector, HttpClientConnector)
    assert (connector.per_msg_timeout, connector.tx_timeout, connector.client_shutdown_timeout) == (11.0, 222.0, 3.5)
