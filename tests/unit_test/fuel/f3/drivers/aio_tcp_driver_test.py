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

from nvflare.fuel.f3.drivers.aio_context import AioContext
from nvflare.fuel.f3.drivers.aio_tcp_driver import AioTcpDriver


def test_shutdown_runs_on_aio_loop_and_is_idempotent(monkeypatch):
    aio_ctx = AioContext("aio_tcp_driver_test")
    loop_thread = threading.Thread(target=aio_ctx.run_aio_loop)
    loop_thread.start()
    assert aio_ctx.ready.wait(timeout=5)

    monkeypatch.setattr(AioContext, "get_global_context", lambda: aio_ctx)
    driver = AioTcpDriver()
    calls = []

    class _Server:
        def close(self):
            calls.append(("server.close", threading.get_ident()))

    def close_all():
        calls.append(("close_all", threading.get_ident()))

    monkeypatch.setattr(driver, "close_all", close_all)
    driver.server = _Server()

    try:
        driver.shutdown()
        driver.shutdown()
    finally:
        # Queued callbacks must run even when the global AIO context stops immediately.
        aio_ctx.stop_aio_loop()
        loop_thread.join(timeout=5)

    assert not loop_thread.is_alive()
    assert calls == [
        ("close_all", loop_thread.ident),
        ("server.close", loop_thread.ident),
        ("close_all", loop_thread.ident),
    ]
    assert driver.server is None
