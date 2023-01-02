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

from nvflare.private.fed.cmd_agent import CommandAgent

from nvflare.private.fed.server.job.server_commands import (
    AbortCommand, GetRunInfoCommand, HandleDeadJobCommand, ShowStatsCommand, GetErrorsCommand
)


class ServerCommandAgent(CommandAgent):
    """AdminCommands contains all the commands for processing the commands from the parent process."""

    COMMANDS = [
        AbortCommand(),
        GetRunInfoCommand(),
        HandleDeadJobCommand(),
        ShowStatsCommand(),
        GetErrorsCommand(),
    ]

    def __init__(self, engine):
        CommandAgent.__init__(self, engine, commands=self.COMMANDS)
