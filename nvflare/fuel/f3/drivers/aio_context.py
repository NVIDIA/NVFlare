# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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
import logging
import threading
import time
import traceback

from nvflare.fuel.f3.mpm import MainProcessMonitor


class AioContext:

    counter_lock = threading.Lock()
    thread_count = 0

    def __init__(self, name):
        self.closed = False
        self.name = name
        with AioContext.counter_lock:
            AioContext.thread_count += 1
        thread_name = f"aio_ctx_{AioContext.thread_count}"
        self.thread = threading.Thread(target=self._run_aio_loop, name=thread_name)
        self.thread.daemon = True
        self.loop = None
        self.ready = False
        self.logger = logging.getLogger(self.__class__.__name__)
        self.thread.start()
        MainProcessMonitor.add_cleanup_cb(self._aio_cleanup)

    def _aio_cleanup(self):
        MainProcessMonitor.add_cleanup_cb(self._aio_shutdown)

    def _run_aio_loop(self):
        self.logger.debug(f"{self.name}: started AioContext in thread {threading.current_thread().name}")
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.logger.debug(f"{self.name}: got loop: {id(self.loop)}")
        self.ready = True
        self.loop.run_forever()

    def run_coro(self, coro):
        while not self.ready:
            self.logger.debug(f"{self.name}: waiting for loop to be ready")
            time.sleep(0.5)
        self.logger.debug(f"{self.name}: coro loop: {id(self.loop)}")
        asyncio.run_coroutine_threadsafe(coro, self.loop)

    def _aio_shutdown(self):
        try:
            self.loop.stop()
            pending_tasks = asyncio.all_tasks(self.loop)
            for task in pending_tasks:
                self.logger.info(f"{self.name}: cancelled a task")
                task.cancel()
            # asyncio.sleep(0)
            # self.loop.close()
        except:
            self.logger.error(f"{self.name}: error in _aio_shutdown")
            self.logger.error(traceback.format_exc())
