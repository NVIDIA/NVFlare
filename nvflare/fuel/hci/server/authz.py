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
import os
from typing import List

from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.hci.reg import CommandEntry, CommandModule, CommandModuleSpec, CommandSpec
from nvflare.fuel.sec.authz import Authorizer, AuthzContext
from nvflare.fuel.sec.security_content_service import LoadResult, SecurityContentService
from nvflare.fuel.utils.time_utils import time_to_string

from .constants import ConnProps
from .reg import CommandFilter

AUTHORIZATION_POLICY_FILE = "authorization.json"


class AuthorizationService(object):

    the_authorizer = None

    @staticmethod
    def initialize(authorizer: Authorizer, policy_file: str = AUTHORIZATION_POLICY_FILE) -> (Authorizer, str):
        assert isinstance(authorizer, Authorizer), "authorizer must be Authorizer but got {}".format(type(authorizer))

        if not AuthorizationService.the_authorizer:
            # get secure content of the policy file
            policy_conf, result = SecurityContentService.load_json(policy_file)
            if result == LoadResult.NOT_MANAGED or result == LoadResult.NO_SUCH_CONTENT:
                # no authorization needed
                AuthorizationService.the_authorizer = authorizer
            elif result == LoadResult.OK:
                err = authorizer.load_policy(policy_conf)
                if err:
                    return None, err
                AuthorizationService.the_authorizer = authorizer
            else:
                return None, "invalid policy file {}: {}".format(policy_file, result)

        return AuthorizationService.the_authorizer, ""

    @staticmethod
    def initialize_with_policy(authorizer: Authorizer, policy_file_path: str) -> (Authorizer, str):
        assert isinstance(authorizer, Authorizer), "authorizer must be Authorizer but got {}".format(type(authorizer))

        if AuthorizationService.the_authorizer:
            return AuthorizationService.the_authorizer

        if not os.path.exists(policy_file_path):
            return None, 'policy file "{}" does not exist'.format(policy_file_path)

        with open(policy_file_path) as file:
            try:
                policy_conf = json.load(file)
            except json.JSONDecodeError:
                return None, 'policy file "{}" is invalid'.format(policy_file_path)

        err = authorizer.load_policy(policy_conf)
        if err:
            return None, err

        AuthorizationService.the_authorizer = authorizer
        return AuthorizationService.the_authorizer, ""

    @staticmethod
    def get_authorizer():
        return AuthorizationService.the_authorizer

    @staticmethod
    def authorize(ctx: AuthzContext):
        if not AuthorizationService.the_authorizer:
            return None, "no authorizer defined"
        return AuthorizationService.the_authorizer.authorize(ctx)


class AuthzFilter(CommandFilter):
    def __init__(self, authorizer: Authorizer):
        """Filter for authorization of admin commands.

        Args:
            authorizer: instance of Authorizer
        """
        CommandFilter.__init__(self)
        assert isinstance(authorizer, Authorizer), "authorizer must be Authorizer but got {}".format(type(authorizer))
        self.authorizer = authorizer

    def pre_command(self, conn: Connection, args: List[str]):
        cmd_entry = conn.get_prop(ConnProps.CMD_ENTRY, None)
        if not cmd_entry:
            return True

        assert isinstance(cmd_entry, CommandEntry)
        authz_func = cmd_entry.authz_func
        if not authz_func:
            return True

        valid, authz_ctx = authz_func(conn, args)

        if not valid:
            return False

        if not authz_ctx or (isinstance(authz_ctx, tuple) and not any(authz_ctx)):
            # no authz needed
            return True

        if isinstance(authz_ctx, tuple):
            for authz in authz_ctx:
                result = self.check_authz(authz, conn)
                if not result:
                    return False
            return True
        else:
            return self.check_authz(authz_ctx, conn)

    def check_authz(self, authz_ctx, conn: Connection):
        assert isinstance(authz_ctx, AuthzContext), "authz_ctx must be AuthzContext but got {}".format(type(authz_ctx))

        authz_ctx.user_name = conn.get_prop(ConnProps.USER_NAME, "")
        conn.set_prop(ConnProps.AUTHZ_CTX, authz_ctx)
        authorized, err = self.authorizer.authorize(ctx=authz_ctx)
        if err:
            conn.append_error("Authorization Error: {}".format(err))
            return False

        if not authorized:
            conn.append_error("This action is not authorized")
            return False
        return True


class AuthzCommandModule(CommandModule):
    def __init__(self, authorizer: Authorizer):
        """Authorization command module.

        Args:
            authorizer: instance of Authorizer
        """
        assert isinstance(authorizer, Authorizer), "authorizer must be Authorizer but got {}".format(type(authorizer))
        self.authorizer = authorizer

    def get_spec(self):
        return CommandModuleSpec(
            name="authz",
            cmd_specs=[
                CommandSpec(
                    name="show_info",
                    description="show general info of authorization policy",
                    usage="show_info",
                    handler_func=self.show_info,
                    authz_func=self._authorize_cmd,
                    visible=True,
                ),
                CommandSpec(
                    name="show_users",
                    description="show users configured for authorization",
                    usage="show_users",
                    handler_func=self.show_users,
                    authz_func=self._authorize_cmd,
                    visible=True,
                ),
                CommandSpec(
                    name="show_sites",
                    description="show sites configured for authorization",
                    usage="show_sites",
                    handler_func=self.show_sites,
                    authz_func=self._authorize_cmd,
                    visible=True,
                ),
                CommandSpec(
                    name="show_rules",
                    description="show rules configured for authorization",
                    usage="show_rules",
                    handler_func=self.show_rules,
                    authz_func=self._authorize_cmd,
                    visible=True,
                ),
                CommandSpec(
                    name="show_rights",
                    description="show rights configured for authorization",
                    usage="show_rights",
                    handler_func=self.show_rights,
                    authz_func=self._authorize_cmd,
                    visible=True,
                ),
                CommandSpec(
                    name="show_config",
                    description="show authorization config",
                    usage="show_config",
                    handler_func=self.show_config,
                    authz_func=self._authorize_cmd,
                    visible=True,
                ),
                CommandSpec(
                    name="eval_right",
                    description="check if a user has a right on a site",
                    usage="eval_right user_name right_name [site_name...]",
                    handler_func=self.eval_right,
                    authz_func=self._authorize_cmd,
                    visible=True,
                ),
                CommandSpec(
                    name="eval_rule",
                    description="evaluate a site against a rule",
                    usage="eval_rule site_name rule_name",
                    handler_func=self.eval_rule,
                    authz_func=self._authorize_cmd,
                    visible=True,
                ),
            ],
        )

    def _authorize_cmd(self, conn: Connection, args: List[str]):
        # Use this method to pre-process the command,
        # mainly to get authorizer and policy and store them in conn
        # so the command handler can get them from the conn.
        # This way, command handlers don't need to know how to get authorizer or policy.
        # In case we use different way to get authorizer, this is the only place to change!
        policy = self.authorizer.get_policy()
        if not policy:
            conn.append_error("no authorization policy defined")
            return False, None

        conn.set_prop("authorizer", self.authorizer)
        conn.set_prop("policy", policy)

        # return a None authz_ctx because the command does not need to be authorized!
        return True, None

    def show_info(self, conn: Connection, args: List[str]):
        authorizer = conn.get_prop("authorizer")
        conn.append_string("Last Loaded: {}".format(time_to_string(authorizer.last_load_time)))

    def show_users(self, conn: Connection, args: List[str]):
        policy = conn.get_prop("policy")
        users = policy.get_users()
        table = conn.append_table(["user", "org", "roles"])
        for user_name, user_def in users.items():
            table.add_row([user_name, user_def["org"], ",".join(user_def["roles"])])

    def show_sites(self, conn: Connection, args: List[str]):
        policy = conn.get_prop("policy")
        if not policy:
            conn.append_error("no authorization policy")
            return

        sites = policy.get_sites()
        table = conn.append_table(["site", "org"])
        for site_name, org in sites.items():
            table.add_row([site_name, org])

    def show_rights(self, conn: Connection, args: List[str]):
        policy = conn.get_prop("policy")
        if not policy:
            conn.append_error("no authorization policy")
            return

        rights = policy.get_rights()
        table = conn.append_table(["name", "description", "default"])
        for name, right_def in rights.items():
            desc = right_def.get("desc", "")
            default = right_def.get("default", "")
            table.add_row([name, desc, "{}".format(default)])

    def show_rules(self, conn: Connection, args: List[str]):
        policy = conn.get_prop("policy")
        if not policy:
            conn.append_error("no authorization policy")
            return

        rules = policy.get_rules()
        table = conn.append_table(["name", "description", "default"])
        for name, rule_def in rules.items():
            desc = rule_def.get("desc", "")
            default = rule_def.get("default", "")
            table.add_row([name, desc, "{}".format(default)])

    def show_config(self, conn: Connection, args: List[str]):
        policy = conn.get_prop("policy")
        config = policy.get_config()
        conn.append_string(json.dumps(config, indent=1))

    def eval_right(self, conn: Connection, args: List[str]):
        if len(args) < 3:
            conn.append_error("Usage: {} user_name right_name [site_name...]".format(args[0]))
            return

        authorizer = conn.get_prop("authorizer")
        user_name = args[1]
        right_name = args[2]
        site_names = args[3:]
        if not site_names:
            # all sites
            policy = conn.get_prop("policy")
            if not policy:
                conn.append_error("no authorization policy")
                return

            sites = policy.get_sites()
            for s, _ in sites.items():
                site_names.append(s)

        table = conn.append_table(["Site", "Result"])
        for s in site_names:
            result, err = authorizer.evaluate_user_right_on_site(
                user_name=user_name, site_name=s, right_name=right_name
            )
            if err:
                result = err
            else:
                result = str(result)
            table.add_row([s, result])

    def eval_rule(self, conn: Connection, args: List[str]):
        if len(args) != 3:
            conn.append_error("Usage: {} site_name rule_name".format(args[0]))
            return

        authorizer = conn.get_prop("authorizer")
        site_name = args[1]
        rule_name = args[2]
        result, err = authorizer.evaluate_rule_on_site(site_name=site_name, rule_name=rule_name)
        if err:
            conn.append_error(err)
        else:
            conn.append_string("{}".format(result))
