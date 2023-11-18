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
import enum
import logging
from typing import List

from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.hci.proto import MetaStatusValue, make_meta
from nvflare.fuel.hci.reg import CommandEntry
from nvflare.fuel.sec.authz import AuthorizationService, AuthzContext, Person

from .constants import ConnProps
from .reg import CommandFilter

log = logging.getLogger(__name__)


class PreAuthzReturnCode(enum.Enum):

    OK = 0  # command preprocessed successfully, and no authz needed
    ERROR = 1  # error occurred in command processing
    REQUIRE_AUTHZ = 2  # command preprocessed successfully, further authz required


def command_handler_func_signature(conn: Connection, args: List[str]):
    pass


def command_authz_func_signature(conn: Connection, args: List[str]) -> PreAuthzReturnCode:
    pass


class AuthzFilter(CommandFilter):
    def __init__(self):
        """Filter for authorization of admin commands."""
        CommandFilter.__init__(self)

    def pre_command(self, conn: Connection, args: List[str]):
        cmd_entry = conn.get_prop(ConnProps.CMD_ENTRY, None)
        if not cmd_entry:
            return True

        assert isinstance(cmd_entry, CommandEntry)
        authz_func = cmd_entry.authz_func
        if not authz_func:
            return True

        return_code = authz_func(conn, args)

        if return_code == PreAuthzReturnCode.OK:
            return True

        if return_code == PreAuthzReturnCode.ERROR:
            return False

        # authz required - the command name is the name of the right to be checked!
        user = Person(
            name=conn.get_prop(ConnProps.USER_NAME, ""),
            org=conn.get_prop(ConnProps.USER_ORG, ""),
            role=conn.get_prop(ConnProps.USER_ROLE, ""),
        )

        submitter = Person(
            name=conn.get_prop(ConnProps.SUBMITTER_NAME, ""),
            org=conn.get_prop(ConnProps.SUBMITTER_ORG, ""),
            role=conn.get_prop(ConnProps.SUBMITTER_ORG, ""),
        )

        ctx = AuthzContext(user=user, submitter=submitter, right=cmd_entry.name)

        log.debug("User: {} Submitter: {}  Right: {}".format(user, submitter, cmd_entry.name))
        authorized, err = AuthorizationService.authorize(ctx)
        if err:
            conn.append_error(f"Authorization Error: {err}", meta=make_meta(MetaStatusValue.NOT_AUTHORIZED, err))
            return False

        if not authorized:
            conn.append_error(
                "This action is not authorized", meta=make_meta(MetaStatusValue.NOT_AUTHORIZED, "not authorized")
            )
            return False
        return True
