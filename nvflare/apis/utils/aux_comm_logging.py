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

from nvflare.app_common.widgets.streaming import LogSender


def aux_comm_logging(logger, message):
    """
    LogSender sends the logging data to the server. It must remove itself when calling the logger function
    during the process.
    Args:
        logger: logger object
        message: message to be logged
    Returns:

    """
    log_senders = []
    for handler in logging.root.handlers:
        if isinstance(handler, LogSender):
            log_senders.append(handler)
    for handler in log_senders:
        logger.root.removeHandler(handler)
    logger.info(message)
    for handler in log_senders:
        logger.root.addHandler(handler)
