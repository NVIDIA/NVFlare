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

import sys
import os
import traceback
import logging


def is_secure() -> bool:
    """Is logging set in secure mode?
    This is controlled by the system environment variable NVFLARE_SECURE_LOGGING.
    To set secure mode, set this var to 'true' or '1'.

    Returns: whether logging is set in secure mode.
    """
    secure_logging = os.environ.get("NVFLARE_SECURE_LOGGING", False)
    if isinstance(secure_logging, str):
        secure_logging = secure_logging.lower()
        return secure_logging == '1' or secure_logging == 'true'
    else:
        return False


class _Frame(object):

    def __init__(self, line_text):
        self.line_text = line_text
        self.count = 1


def _format_exc_securely():
    """
    Mimic traceback.format_exc() but exclude detailed call info and exception detail since
    they might contain sensitive info.

    Returns: a formatted string of current exception and call stack.

    """
    exc_type, exc_obj, tb = sys.exc_info()
    result = ["Traceback (most recent call last):"]
    frames = []
    last_frame = None

    # traceback (tb) stack is a linked list of frames
    while tb:
        file_name = tb.tb_frame.f_code.co_filename
        func_name = tb.tb_frame.f_code.co_name
        line = tb.tb_lineno
        line_text = 'File "{}", line {}, in {}'.format(file_name, line, func_name)

        if not last_frame or last_frame.line_text != line_text:
            last_frame = _Frame(line_text)
            frames.append(last_frame)
        else:
            # same text as last frame
            last_frame.count += 1
        tb = tb.tb_next

    for f in frames:
        result.append(f.line_text)
        if f.count > 1:
            result.append("[Previous line repeated {} more times]".format(f.count-1))

    text = "\r\n  ".join(result)
    return "{}\r\n{}".format(text, 'Exception Type: {}'.format(exc_type))


def log_exception(logger: logging.Logger=None):
    if is_secure():
        exc_detail = _format_exc_securely()
    else:
        exc_detail = traceback.format_exc()

    if logger:
        logger.error(exc_detail)
    else:
        print(exc_detail)
