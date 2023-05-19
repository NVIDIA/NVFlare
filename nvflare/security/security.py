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

from typing import Tuple

from nvflare.apis.fl_constant import AdminCommandNames as AC
from nvflare.fuel.sec.authz import Authorizer, AuthzContext


class CommandCategory(object):

    MANAGE_JOB = "manage_job"
    OPERATE = "operate"
    VIEW = "view"
    SHELL_COMMANDS = "shell_commands"


COMMAND_CATEGORIES = {
    AC.ABORT: CommandCategory.MANAGE_JOB,
    AC.ABORT_TASK: CommandCategory.MANAGE_JOB,
    AC.ABORT_JOB: CommandCategory.MANAGE_JOB,
    AC.START_APP: CommandCategory.MANAGE_JOB,
    AC.DELETE_JOB: CommandCategory.MANAGE_JOB,
    AC.DELETE_WORKSPACE: CommandCategory.MANAGE_JOB,
    AC.CHECK_STATUS: CommandCategory.VIEW,
    AC.SHOW_SCOPES: CommandCategory.VIEW,
    AC.SHOW_STATS: CommandCategory.VIEW,
    AC.RESET_ERRORS: CommandCategory.VIEW,
    AC.SHOW_ERRORS: CommandCategory.VIEW,
    AC.LIST_JOBS: CommandCategory.VIEW,
    AC.GET_JOB_META: CommandCategory.VIEW,
    AC.SYS_INFO: CommandCategory.OPERATE,
    AC.REPORT_RESOURCES: CommandCategory.OPERATE,
    AC.RESTART: CommandCategory.OPERATE,
    AC.SHUTDOWN: CommandCategory.OPERATE,
    AC.REMOVE_CLIENT: CommandCategory.OPERATE,
    AC.SET_TIMEOUT: CommandCategory.OPERATE,
    AC.CALL: CommandCategory.OPERATE,
    AC.SHELL_CAT: CommandCategory.SHELL_COMMANDS,
    AC.SHELL_GREP: CommandCategory.SHELL_COMMANDS,
    AC.SHELL_HEAD: CommandCategory.SHELL_COMMANDS,
    AC.SHELL_LS: CommandCategory.SHELL_COMMANDS,
    AC.SHELL_PWD: CommandCategory.SHELL_COMMANDS,
    AC.SHELL_TAIL: CommandCategory.SHELL_COMMANDS,
}


class FLAuthorizer(Authorizer):
    def __init__(self, for_org: str, policy_config: dict):
        """System-wide authorization class.

        Examine if a user has certain rights on a specific site
        based on authorization.json file.

        """
        assert isinstance(policy_config, dict), "policy_config must be a dict but got {}".format(type(policy_config))
        Authorizer.__init__(self, for_org, COMMAND_CATEGORIES)
        err = self.load_policy(policy_config)
        if err:
            raise SyntaxError("invalid policy config: {}".format(err))


class EmptyAuthorizer(Authorizer):
    def __init__(self):
        Authorizer.__init__(self, "dummy")

    def authorize(self, ctx: AuthzContext) -> Tuple[bool, str]:
        return True, ""
