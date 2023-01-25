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
from nvflare.private.fed.client.job.admin_commands import (
    CheckStatusCommand, AbortCommand, AbortTaskCommand, ShowStatsCommand,
    ShowErrorsCommand, ResetErrorsCommand
)


class ClientCommandAgent(CommandAgent):

    COMMANDS = [
        CheckStatusCommand(),
        AbortCommand(),
        AbortTaskCommand(),
        ShowStatsCommand(),
        ShowErrorsCommand(),
        ResetErrorsCommand()
    ]

    def __init__(self, engine):
        """To init the CommandAgent.
        """
        CommandAgent.__init__(self, engine, self.COMMANDS)
