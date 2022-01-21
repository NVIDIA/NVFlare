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

import logging
import threading


class LocalLogger:
    handlers = None
    loggers = {}
    lock = threading.Lock()

    @staticmethod
    def initialize():
        """Initializes the LocalLogger."""
        if not LocalLogger.handlers:
            LocalLogger.handlers = []
            for handler in logging.root.handlers:
                LocalLogger.handlers.append(handler)

    @staticmethod
    def get_logger(name=None) -> logging.Logger:
        """Gets a logger only do the local logging.

        Args:
            name: logger name

        Returns:
            A local logger.
        """
        with LocalLogger.lock:
            if not LocalLogger.handlers:
                LocalLogger.initialize()

            logger = LocalLogger.loggers.get(name)
            if logger:
                return logger

            logger = logging.getLogger(name)
            LocalLogger.loggers[name] = logger
            for handler in LocalLogger.handlers:
                logger.addHandler(handler)
            logger.propagate = False

            return logger
