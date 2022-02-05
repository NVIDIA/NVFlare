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

import json
from datetime import datetime
from typing import List

from .table import Table


class Buffer(object):
    def __init__(self):
        """Buffer to append to for :class:`nvflare.fuel.hci.conn.Connection`."""
        self.output = {"time": "{}".format(datetime.now()), "data": []}

    def append_table(self, headers: List[str]) -> Table:
        t = Table(headers)
        self.output["data"].append({"type": "table", "rows": t.rows})
        return t

    def append_string(self, data: str):
        self.output["data"].append({"type": "string", "data": data})

    def append_dict(self, data: dict):
        self.output["data"].append({"type": "dict", "data": data})

    def append_success(self, data: str):
        self.output["data"].append({"type": "success", "data": data})

    def append_error(self, data: str):
        self.output["data"].append({"type": "error", "data": data})

    def append_command(self, cmd: str):
        self.output["data"].append({"type": "command", "data": cmd})

    def append_token(self, token: str):
        self.output["data"].append({"type": "token", "data": token})

    def append_shutdown(self, msg: str):
        self.output["data"].append({"type": "shutdown", "data": msg})

    def encode(self):
        if len(self.output["data"]) <= 0:
            return None

        return json.dumps(self.output)

    def reset(self):
        self.output = {"time": "{}".format(datetime.now()), "data": []}


def make_error(data: str):
    buf = Buffer()
    buf.append_error(data)
    return buf.output


def validate_proto(line: str):
    """Validate that the line being received is of the expected format.

    Args:
        line: str containing a JSON document

    Returns: deserialized JSON document
    """
    all_types = ["string", "success", "error", "table", "command", "token", "shutdown", "dict"]
    types_with_data = ["string", "success", "error", "command", "token", "shutdown"]
    try:
        json_data = json.loads(line)
        assert isinstance(json_data, dict)
        assert "data" in json_data
        data = json_data["data"]
        assert isinstance(data, list)
        for item in data:
            assert isinstance(item, dict)
            assert "type" in item
            it = item["type"]
            assert it in all_types

            if it in types_with_data:
                assert "data" in item
                assert isinstance(item["data"], str)
            elif it == "table":
                assert "rows" in item
                rows = item["rows"]
                assert isinstance(rows, list)
                for row in rows:
                    assert isinstance(row, list)

        return json_data
    except BaseException:
        return None
