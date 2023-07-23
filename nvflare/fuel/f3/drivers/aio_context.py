# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import os
import threading
import time

from nvflare.fuel.utils.obj_utils import get_logger
from nvflare.security.logging import secure_format_exception


class AioContext:
    """Asyncio context. Used to share the asyncio event loop among multiple classes"""

    _ctx_lock = threading.Lock()
    _global_ctx = None

    def __init__(self, name):
        self.closed = False
        self.name = name
        self.loop = None
        self.ready = threading.Event()
        self.logger = get_logger(self)
        self.logger.debug(f"{os.getpid()}: ******** Created AioContext {name}")

    def get_event_loop(self):
        t = threading.current_thread()
        if not self.ready.is_set():
            self.logger.debug(f"{os.getpid()} {t.name}: {self.name}: waiting for loop to be ready")
            self.ready.wait()

        return self.loop

    def run_aio_loop(self):
        self.logger.debug(f"{self.name}: started AioContext in thread {threading.current_thread().name}")
        # self.loop = asyncio.get_event_loop()
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.logger.debug(f"{self.name}: got loop: {id(self.loop)}")
        self.ready.set()
        try:
            self.loop.run_forever()
            pending_tasks = asyncio.all_tasks(self.loop)
            for t in [t for t in pending_tasks if not (t.done() or t.cancelled())]:
                # give canceled tasks the last chance to run
                self.loop.run_until_complete(t)
            # self.loop.run_until_complete(self.loop.shutdown_asyncgens())
        except Exception as ex:
            self.logger.error(f"error running aio loop: {secure_format_exception(ex)}")
            raise ex
        finally:
            self.logger.debug(f"{self.name}: AIO Loop run done!")
            self.loop.close()
        self.logger.debug(f"{self.name}: AIO Loop Completed!")

    def run_coro(self, coro):
        event_loop = self.get_event_loop()
        return asyncio.run_coroutine_threadsafe(coro, event_loop)

    def stop_aio_loop(self, grace=1.0):
        self.logger.debug("Cancelling pending tasks")
        pending_tasks = asyncio.all_tasks(self.loop)
        for task in pending_tasks:
            self.logger.debug(f"{self.name}: cancelled a task")
            try:
                # task.cancel()
                self.loop.call_soon_threadsafe(task.cancel)
            except Exception as ex:
                self.logger.debug(f"{self.name}: error cancelling task {type(ex)}")

        # wait until all pending tasks are done
        start = time.time()
        while asyncio.all_tasks(self.loop):
            if time.time() - start > grace:
                self.logger.debug(f"pending tasks are not cancelled in {grace} seconds")
                break
            time.sleep(0.1)

        self.logger.debug("Stopping AIO loop")
        try:
            self.loop.call_soon_threadsafe(self.loop.stop)
        except Exception as ex:
            self.logger.debug(f"Loop stopping error: {secure_format_exception(ex)}")

        start = time.time()
        while self.loop.is_running():
            self.logger.debug("looping still running ...")
            time.sleep(0.1)
            if time.time() - start > grace:
                break

        if self.loop.is_running():
            self.logger.error("could not stop AIO loop")
        else:
            self.logger.debug("stopped loop!")

    @classmethod
    def get_global_context(cls):
        with cls._ctx_lock:
            if not cls._global_ctx:
                cls._global_ctx = AioContext(f"Ctx_{os.getpid()}")
                t = threading.Thread(target=cls._global_ctx.run_aio_loop, name="aio_ctx")
                t.daemon = True
                t.start()
        return cls._global_ctx

    @classmethod
    def close_global_context(cls):
        with cls._ctx_lock:
            if cls._global_ctx:
                cls._global_ctx.stop_aio_loop()
                cls._global_ctx = None
