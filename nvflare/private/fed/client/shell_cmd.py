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

import subprocess

from nvflare.private.admin_defs import Message
from nvflare.private.defs import SysCommandTopic

from .admin import RequestProcessor


class ShellCommandProcessor(RequestProcessor):
    def get_topics(self) -> [str]:
        return [SysCommandTopic.SHELL]

    def process(self, req: Message, app_ctx) -> Message:
        shell_cmd = req.body
        output = subprocess.getoutput(shell_cmd)
        return Message(topic="reply_" + req.topic, body=output)
