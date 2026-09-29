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

"""Regression coverage for transport lifecycle fixes, independent of certificate renewal."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nvflare.fuel.f3.comm_error import CommError
from nvflare.fuel.f3.drivers import aio_grpc_driver, grpc_driver
from nvflare.fuel.f3.drivers.aio_http_driver import AioHttpDriver
from nvflare.fuel.f3.drivers.connector_info import Mode
from nvflare.fuel.f3.drivers.tcp_driver import TcpDriver
from nvflare.fuel.f3.endpoint import Endpoint
from nvflare.fuel.f3.sfm.conn_manager import ConnManager


def test_close_all_does_not_hold_driver_lock():
    driver = TcpDriver()

    def close():
        assert driver.conn_lock.acquire(blocking=False)
        driver.conn_lock.release()
        driver.connections.clear()

    driver.connections["test"] = MagicMock(close=close)
    driver.close_all()
    assert not driver.connections


@pytest.mark.parametrize("operation", ["stop", "remove"])
def test_connector_shutdown_does_not_hold_manager_lock(operation):
    manager = ConnManager(Endpoint("test"))
    driver = TcpDriver()
    handle = manager.add_connector(driver, {"scheme": "tcp", "url": "tcp://localhost:1234"}, Mode.PASSIVE)

    def shutdown():
        assert manager.lock.acquire(blocking=False)
        manager.lock.release()

    try:
        with patch.object(driver, "shutdown", side_effect=shutdown):
            if operation == "remove":
                manager.remove_connector(handle)
            manager.stop()
    finally:
        manager.conn_mgr_executor.shutdown()
        manager.frame_mgr_executor.shutdown()


def test_grpc_bind_failure_stops_server_and_propagates():
    connector = MagicMock(params={"scheme": "grpc", "host": "localhost", "port": 1234})
    with patch.object(grpc_driver.grpc, "server") as create:
        create.return_value.add_insecure_port.side_effect = RuntimeError("address unavailable")
        with pytest.raises(CommError, match="address unavailable"):
            grpc_driver.Server(MagicMock(), connector, 1, [])
        create.return_value.stop.assert_called_once_with(grace=0)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_grpc_server_shutdown_is_repeatable(asynchronous):
    server = SimpleNamespace(grpc_server=MagicMock(), grpc_server_stop_grace=0, logger=MagicMock())
    native_server = server.grpc_server
    if asynchronous:
        native_server.stop = AsyncMock()

        async def shutdown():
            await aio_grpc_driver.Server.shutdown(server)
            await aio_grpc_driver.Server.shutdown(server)

        asyncio.run(shutdown())
        assert native_server.stop.await_count == 2
    else:
        grpc_driver.Server.shutdown(server)
        grpc_driver.Server.shutdown(server)
        assert native_server.stop.call_count == 2


def test_http_server_shutdown_is_repeatable():
    async def shutdown():
        context = MagicMock()
        context.get_event_loop.return_value = asyncio.get_running_loop()
        with patch("nvflare.fuel.f3.drivers.aio_http_driver.AioContext.get_global_context", return_value=context):
            driver = AioHttpDriver()
        await driver._async_shutdown()
        await driver._async_shutdown()
        assert driver.stop_event.done()

    asyncio.run(shutdown())
