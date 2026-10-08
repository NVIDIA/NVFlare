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

import asyncio

from nvflare.fuel.f3.drivers.base_driver import BaseDriver


class AioBaseDriver(BaseDriver):
    """Own asynchronous connection attempts through their finalizers.

    Task registration and cancellation both run on the driver's event loop.
    Shutdown awaits actual tasks, rather than cancelling the thread-safe future
    and releasing connector threads before native resource cleanup completes.
    """

    def __init__(self):
        super().__init__()
        self._connect_tasks = set()
        self._shutdown_lock = asyncio.Lock()

    async def _run_connect(self, connector, operation):
        if self.is_stopping() or connector.stopped.is_set():
            return
        task = asyncio.current_task()
        self._connect_tasks.add(task)
        try:
            return await operation()
        finally:
            self._connect_tasks.discard(task)

    async def _cancel_connect_tasks(self):
        tasks = list(self._connect_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
