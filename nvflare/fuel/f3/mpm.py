#  Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import logging
import os
import signal
import threading
import time


class MainProcessMonitor:

    cleanup_cbs = []
    run_monitors = []
    asked_to_stop = False
    stopping = False
    name = "MPM"
    _logger = None

    @classmethod
    def logger(cls):
        if not cls._logger:
            cls._logger = logging.getLogger("MPM")
        return cls._logger

    @classmethod
    def add_cleanup_cb(cls, cb, *args, **kwargs):
        if not callable(cb):
            raise ValueError(f"specified cleanup_cb {type(cb)} is not callable")
        cls.cleanup_cbs.append((cb, args, kwargs))

    @classmethod
    def add_run_monitor(cls, cb, *args, **kwargs):
        if not callable(cb):
            raise ValueError(f"specified monitor {type(cb)} is not callable")
        cls.run_monitors.append((cb, args, kwargs))

    @classmethod
    def _call_cb(cls, t: tuple):
        logger = cls.logger()
        cb, args, kwargs = t[0], t[1], t[2]
        try:
            return cb(*args, **kwargs)
        except BaseException as ex:
            logger.error(f"exception from CB {cb.__name__}: {type(ex)}")

    @classmethod
    def stop(cls):
        if cls.stopping:
            return
        else:
            cls.asked_to_stop = True

    @classmethod
    def run(cls, name: str, shutdown_grace_time=2.0, cleanup_grace_time=3.0):
        logger = cls.logger()

        # this method must be called from main method
        t = threading.current_thread()
        if t.name != "MainThread":
            raise RuntimeError(
                f"{name}: the mpm.run() method is called from {t.name}: it must be called from the MainThread")

        if cls.stopping:
            return

        logger.info(f"=========== {name}: started to run forever")
        while not cls.asked_to_stop:
            for m in cls.run_monitors:
                should_stop = cls._call_cb(m)
                if should_stop:
                    logger.info(f"{name}: CB {m[0].__name__} asked to stop!")
                    break
            time.sleep(0.5)

        logger.info(f"=========== {name}: Shutting down. Starting cleanup ...")
        time.sleep(shutdown_grace_time)  # let pending activities to finish

        cleanup_waiter = threading.Event()
        t = threading.Thread(target=cls._do_cleanup, args=(name, cleanup_waiter,))
        t.start()

        if not cleanup_waiter.wait(timeout=cleanup_grace_time):
            logger.warning(f"======== {name}: Cleanup did not complete within {cleanup_grace_time} secs")

        num_active_threads = 0
        for thread in threading.enumerate():
            if thread.name != "MainThread" and not thread.daemon:
                logger.warning(f"#### {name}: still running thread {thread.name}")
                num_active_threads += 1

        logger.info(f"{name}: Good Bye!")
        if num_active_threads > 0:
            try:
                os.kill(os.getpid(), signal.SIGKILL)
            except:
                pass

    @classmethod
    def _cleanup_one_round(cls, name, cbs):
        logger = cls.logger()
        for _cb in cbs:
            cb_name = ""
            try:
                cb_name = _cb[0].__name__
                logger.info(f"{name}: calling cleanup CB {cb_name}")
                cls._call_cb(_cb)
                logger.debug(f"{name}: finished cleanup CB {cb_name}")
            except BaseException as ex:
                logger.warning(f"{name}: exception {ex} from cleanup CB {cb_name}")

    @classmethod
    def _do_cleanup(cls, name: str, waiter: threading.Event):
        max_cleanup_rounds = 10
        logger = cls.logger()
        logger.debug(f"{name}: Start system cleanup ...")
        if cls.cleanup_cbs:
            # during cleanup, a cleanup CB can add another cleanup CB
            # we will call cleanup multiple rounds until no more CBs are added or tried max number of rounds
            for i in range(max_cleanup_rounds):
                cbs = cls.cleanup_cbs
                cls.cleanup_cbs = []
                if cbs:
                    logger.debug(f"{name}: cleanup round {i + 1}")
                    cls._cleanup_one_round(name, cbs)
                    logger.debug(f"{name}: finished cleanup round {i + 1}")
                else:
                    break

            if cls.cleanup_cbs:
                logger.warning(f"{name}: there are still cleanup CBs after {max_cleanup_rounds} rounds")
        else:
            logger.debug(f"{name}: nothing to cleanup!")

        logger.debug(f"{name}: Cleanup Finished!")
        waiter.set()
