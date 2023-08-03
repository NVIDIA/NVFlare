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
import logging
import os
import signal
import threading
import time

from nvflare.fuel.common.excepts import ComponentNotAuthorized, ConfigError
from nvflare.fuel.common.exit_codes import ProcessExitCode
from nvflare.fuel.f3.drivers.aio_context import AioContext
from nvflare.security.logging import secure_format_exception, secure_format_traceback


class MainProcessMonitor:
    """MPM (Main Process Monitor). It's used to run main thread and to handle graceful shutdown"""

    name = "MPM"
    _cleanup_cbs = []
    _stopping = False
    _logger = None
    _aio_ctx = None

    @classmethod
    def set_name(cls, name: str):
        if not name:
            raise ValueError("name must be specified")
        if not isinstance(name, str):
            raise ValueError(f"name must be str but got {type(name)}")
        cls.name = name

    @classmethod
    def is_stopping(cls):
        return cls._stopping

    @classmethod
    def get_aio_context(cls):
        if not cls._aio_ctx:
            cls._aio_ctx = AioContext.get_global_context()
        return cls._aio_ctx

    @classmethod
    def logger(cls):
        if not cls._logger:
            cls._logger = logging.getLogger(cls.name)
        return cls._logger

    @classmethod
    def add_cleanup_cb(cls, cb, *args, **kwargs):
        if not callable(cb):
            raise ValueError(f"specified cleanup_cb {type(cb)} is not callable")
        for _cb in cls._cleanup_cbs:
            if cb == _cb[0]:
                raise RuntimeError(f"cleanup CB {cb.__name__} is already registered")
        cls._cleanup_cbs.append((cb, args, kwargs))

    @classmethod
    def _call_cb(cls, t: tuple):
        cb, args, kwargs = t[0], t[1], t[2]
        try:
            return cb(*args, **kwargs)
        except Exception as ex:
            cls.logger().error(f"exception from CB {cb.__name__}: {type(secure_format_exception(ex))}")

    @classmethod
    def _start_shutdown(cls, shutdown_grace_time, cleanup_grace_time):
        logger = cls.logger()
        if not cls._cleanup_cbs:
            logger.debug(f"=========== {cls.name}: Nothing to cleanup ...")
            return

        logger.debug(f"=========== {cls.name}: Shutting down. Starting cleanup ...")
        time.sleep(shutdown_grace_time)  # let pending activities finish

        cleanup_waiter = threading.Event()
        t = threading.Thread(target=cls._do_cleanup, args=(cleanup_waiter,))
        t.daemon = True
        t.start()

        if not cleanup_waiter.wait(timeout=cleanup_grace_time):
            logger.warning(f"======== {cls.name}: Cleanup did not complete within {cleanup_grace_time} secs")

    @classmethod
    def _cleanup_one_round(cls, cbs):
        logger = cls.logger()
        for _cb in cbs:
            cb_name = ""
            try:
                cb_name = _cb[0].__name__
                logger.debug(f"{cls.name}: calling cleanup CB {cb_name}")
                cls._call_cb(_cb)
                logger.debug(f"{cls.name}: finished cleanup CB {cb_name}")
            except Exception as ex:
                logger.warning(f"{cls.name}: exception {secure_format_exception(ex)} from cleanup CB {cb_name}")

    @classmethod
    def _do_cleanup(cls, waiter: threading.Event):
        max_cleanup_rounds = 10
        logger = cls.logger()

        # during cleanup, a cleanup CB can add another cleanup CB
        # we will call cleanup multiple rounds until no more CBs are added or tried max number of rounds
        for i in range(max_cleanup_rounds):
            cbs = cls._cleanup_cbs
            cls._cleanup_cbs = []
            if cbs:
                logger.debug(f"{cls.name}: cleanup round {i + 1}")
                cls._cleanup_one_round(cbs)
                logger.debug(f"{cls.name}: finished cleanup round {i + 1}")
            else:
                break

        if cls._cleanup_cbs:
            logger.warning(f"{cls.name}: there are still cleanup CBs after {max_cleanup_rounds} rounds")

        logger.debug(f"{cls.name}: Cleanup Finished!")
        waiter.set()

    @classmethod
    def run(cls, main_func, shutdown_grace_time=1.5, cleanup_grace_time=1.5):
        if not callable(main_func):
            raise ValueError("main_func must be runnable")

        # this method must be called from main method
        t = threading.current_thread()
        if t.name != "MainThread":
            raise RuntimeError(
                f"{cls.name}: the mpm.run() method is called from {t.name}: it must be called from the MainThread"
            )

        # call and wait for the main_func to complete
        logger = cls.logger()
        logger.debug(f"=========== {cls.name}: started to run forever")
        try:
            rc = main_func()
        except ConfigError as ex:
            # already handled
            rc = ProcessExitCode.CONFIG_ERROR
            logger.error(secure_format_traceback())
        except ComponentNotAuthorized as ex:
            rc = ProcessExitCode.UNSAFE_COMPONENT
            logger.error(secure_format_traceback())
        except Exception as ex:
            rc = ProcessExitCode.EXCEPTION
            logger.error(f"Execute exception: {secure_format_exception(ex)}")
            logger.error(secure_format_traceback())

        # start shutdown process
        cls._stopping = True
        cls._start_shutdown(shutdown_grace_time, cleanup_grace_time)

        # We can now stop the AIO loop!
        AioContext.close_global_context()

        logger.debug(f"=========== {cls.name}: checking running threads")
        num_active_threads = 0
        for thread in threading.enumerate():
            if thread.name != "MainThread" and not thread.daemon:
                logger.warning(f"#### {cls.name}: still running thread {thread.name}")
                num_active_threads += 1

        logger.info(f"{cls.name}: Good Bye!")
        if num_active_threads > 0:
            try:
                os.kill(os.getpid(), signal.SIGKILL)
            except Exception as ex:
                logger.debug(f"Failed to kill process {os.getpid()}: {secure_format_exception(ex)}")

        return rc
