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
import shlex
from typing import List

from nvflare.fuel.hci.shell_cmd_val import (
    CatValidator,
    GrepValidator,
    HeadValidator,
    LsValidator,
    PwdValidator,
    TailValidator,
)
from nvflare.private.admin_defs import Message
from nvflare.private.defs import SysCommandTopic
from nvflare.private.fed.client.admin import RequestProcessor
from nvflare.private.fed.utils.fed_utils import execute_command_directly

SHELL_CMD_VALIDATORS = {
    "pwd": PwdValidator(),
    "tail": TailValidator(),
    "head": HeadValidator(),
    "grep": GrepValidator(),
    "cat": CatValidator(),
    "ls": LsValidator(),
}


class ShellCommandProcessor(RequestProcessor):
    def get_topics(self) -> List[str]:
        return [SysCommandTopic.SHELL]

    def process(self, req: Message, app_ctx) -> Message:
        shell_cmd_args = shlex.split(req.body)
        validator = SHELL_CMD_VALIDATORS.get(shell_cmd_args[0], None)
        if not validator:
            output = f"Error: {req.body} is not a supported shell command"
        else:
            err, _ = validator.validate(shell_cmd_args[1:])
            if len(err) > 0:
                output = f"Error: {err}"
            else:
                output = execute_command_directly(shell_cmd_args)
        return Message(topic="reply_" + req.topic, body=output)
