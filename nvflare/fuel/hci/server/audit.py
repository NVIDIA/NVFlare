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
from typing import List

from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.sec.audit import Auditor

from .constants import ConnProps
from .reg import CommandFilter


class CommandAudit(CommandFilter):
    def __init__(self, auditor: Auditor):
        """Command filter for auditing by adding events.

        This filter needs to be registered after the login filter because it needs the username established
        by the login filter.

        Args:
            auditor: instance of Auditor
        """
        CommandFilter.__init__(self)
        assert isinstance(auditor, Auditor), "auditor must be Auditor but got {}".format(type(auditor))
        self.auditor = auditor

    def pre_command(self, conn: Connection, args: List[str]):
        user_name = conn.get_prop(ConnProps.USER_NAME, "?")

        event_id = self.auditor.add_event(
            user=user_name,
            action=conn.command[:100],  # at most 100 chars
        )

        conn.set_prop(ConnProps.EVENT_ID, event_id)
        return True
