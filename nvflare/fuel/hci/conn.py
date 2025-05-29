# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from typing import List

from nvflare.fuel.common.ctx import BaseContext

from .proto import ALL_END, LINE_END, MAX_BLOCK_SIZE, Buffer, validate_proto
from .table import Table

# ASCII Message Format:
#
# Only ASCII chars can be used in message;
# A message consists of multiple lines, each ended with the LINE_END char;
# The message is ended with the ALL_END char.
# Returns:
# seg1 - the text before the end
# seg2 - the text after the end
# if end is not found, seg2 is None
# if end is found, seg2 is a string


def _split_data(data: str):
    # first determine whether the data contains ALL_END
    # anything after ALL_END is dropped
    all_done = False
    idx = data.find(ALL_END)
    if idx >= 0:
        data = data[:idx]
        all_done = True

    # find lines separated by LINE_END
    parts = data.split(LINE_END)
    return parts, all_done


def _process_one_line(line: str, process_json_func):
    """Validate and process one line, which should be a str containing a JSON document."""
    json_data = validate_proto(line)
    process_json_func(json_data)


def receive_bytes_and_process(sock, receiver):
    receiver.receive(sock)
    return True


def receive_and_process(sock, process_json_func):
    """Receives and sends lines to process with process_json_func."""
    leftover = ""
    while True:
        data = str(sock.recv(MAX_BLOCK_SIZE), "utf-8")
        if len(data) <= 0:
            return False

        segs, all_done = _split_data(data)
        if all_done:
            for seg in segs:
                line = leftover + seg
                if len(line) > 0:
                    _process_one_line(line, process_json_func)
                leftover = ""
            return True

        for i in range(len(segs) - 1):
            line = leftover + segs[i]
            if len(line) > 0:
                _process_one_line(line, process_json_func)
            leftover = ""

        leftover += segs[len(segs) - 1]


class Connection(BaseContext):
    def __init__(self):
        """Object containing connection information and buffer to build and send a line with socket passed in at init."""
        BaseContext.__init__(self)
        self.app_ctx = None
        self.ended = False
        self.request = None
        self.command = None
        self.args = None
        self.buffer = Buffer()
        self.binary_mode = False
        self.bytes_sender = None
        self.content_type = None
        self.extra = None

    def append_table(self, headers: List[str], name=None) -> Table:
        return self.buffer.append_table(headers, name=name)

    def append_string(self, data: str, meta: dict = None):
        self.buffer.append_string(data, meta=meta)

    def append_success(self, data: str, meta: dict = None):
        self.buffer.append_success(data, meta=meta)

    def append_dict(self, data: dict, meta: dict = None):
        self.buffer.append_dict(data, meta=meta)

    def append_error(self, data: str, meta: dict = None):
        self.buffer.append_error(data, meta=meta)

    def append_command(self, cmd: str):
        self.buffer.append_command(cmd)

    def append_token(self, token: str):
        self.buffer.append_token(token)

    def append_shutdown(self, msg: str):
        self.buffer.append_shutdown(msg)

    def append_any(self, data, meta: dict = None):
        if data is None:
            return

        if isinstance(data, str):
            self.append_string(data, meta=meta)
        elif isinstance(data, dict):
            self.append_dict(data, meta)
        else:
            self.append_error("unsupported data type {}".format(type(data)))

    def update_meta(self, meta: dict):
        self.buffer.update_meta(meta)

    def close(self):
        line = self.buffer.encode()
        self.buffer.reset()
        return line
