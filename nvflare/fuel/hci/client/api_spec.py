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

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from nvflare.fuel.hci.table import Table


class ReplyProcessor:
    """A base class for parsing server's response."""

    def reply_start(self, api: AdminAPISpec, reply_json):
        pass

    def process_string(self, api: AdminAPISpec, item: str):
        pass

    def process_success(self, api: AdminAPISpec, item: str):
        pass

    def process_error(self, api: AdminAPISpec, err: str):
        pass

    def process_table(self, api: AdminAPISpec, table: Table):
        pass

    def process_dict(self, api: AdminAPISpec, data: dict):
        pass

    def process_shutdown(self, api: AdminAPISpec, msg: str):
        pass

    def process_token(self, api: AdminAPISpec, token: str):
        pass

    def protocol_error(self, api: AdminAPISpec, err: str):
        pass

    def reply_done(self, api: AdminAPISpec):
        pass


class AdminAPISpec(ABC):
    def __init__(self):
        self.reply_processor = None
        self.command_result = None

    @abstractmethod
    def server_execute(self, command: str, reply_processor: Optional[ReplyProcessor] = None):
        """Executes a command on server side.

        Args:
            command: The command to be executed.
            reply_processor: Reply callback to use.
        """
        pass

    def set_command_result(self, result):
        """Sets the result returning from executing the command."""
        self.command_result = result

    def get_command_result(self):
        """Gets the result returning from executing the command."""
        return self.command_result
