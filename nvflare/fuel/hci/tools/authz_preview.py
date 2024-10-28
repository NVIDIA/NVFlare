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

import argparse
import cmd
import json
import sys

from nvflare.fuel.hci.cmd_arg_utils import split_to_args
from nvflare.fuel.hci.table import Table
from nvflare.fuel.sec.authz import AuthzContext, Person, Policy, parse_policy_config
from nvflare.security.security import COMMAND_CATEGORIES


class Commander(cmd.Cmd):
    def __init__(self, policy: Policy):
        """Command line prompt helper tool for getting information for authorization configurations.

        Args:
            policy: authorization policy object
        """
        cmd.Cmd.__init__(self)
        self.policy = policy
        self.intro = "Type help or ? to list commands.\n"
        self.prompt = "> "

    def do_bye(self, arg):
        """Exits from the client."""
        return True

    def emptyline(self):
        return

    def _split_to_args(self, arg):
        if len(arg) <= 0:
            return []
        else:
            return split_to_args(arg)

    def do_show_rights(self, arg):
        rights = self.policy.rights
        table = Table(["right"])
        for r in rights:
            table.add_row([r])
        self.write_table(table)

    def do_show_roles(self, arg):
        roles = self.policy.roles
        table = Table(["role"])
        for r in roles:
            table.add_row([r])
        self.write_table(table)

    def do_show_config(self, arg):
        config = self.policy.config
        self.write_string(json.dumps(config, indent=1))

    def do_show_role_rights(self, arg):
        role_rights = self.policy.role_rights
        table = Table(["role", "right", "conditions"])
        for role_name in sorted(role_rights):
            right_conds = role_rights[role_name]
            for right_name in sorted(right_conds):
                conds = right_conds[right_name]
                table.add_row([role_name, right_name, str(conds)])
        self.write_table(table)

    def _parse_person(self, spec: str):
        parts = spec.split(":")
        if len(parts) != 3:
            return "must be like name:org:role"
        return Person(parts[0], parts[1], parts[2])

    def do_eval_right(self, arg):
        args = ["eval_right"] + self._split_to_args(arg)
        if len(args) < 4:
            self.write_string(
                "Usage: {} site_org right_name user_name:org:role [submitter_name:org:role]".format(args[0])
            )
            return

        site_org = args[1]
        right_name = args[2]
        user_spec = args[3]

        submitter_spec = None
        if len(args) > 4:
            submitter_spec = args[4]

        parsed = self._parse_person(user_spec)
        if isinstance(parsed, str):
            # error
            return self.write_error("bad user spec: " + parsed)
        user = parsed

        submitter = None
        if submitter_spec:
            parsed = self._parse_person(submitter_spec)
            if isinstance(parsed, str):
                # error
                return self.write_error("bad submitter spec: " + parsed)
            submitter = parsed

        result, err = self.policy.evaluate(
            site_org=site_org, ctx=AuthzContext(right=right_name, user=user, submitter=submitter)
        )
        if err:
            self.write_error(err)
        elif result is None:
            self.write_string("undetermined")
        else:
            self.write_string(str(result))

    def write_string(self, data: str):
        content = data + "\n"
        self.stdout.write(content)

    def write_table(self, table: Table):
        table.write(self.stdout)

    def write_error(self, err: str):
        content = "Error: " + err + "\n"
        self.stdout.write(content)


def define_authz_preview_parser(parser):
    parser.add_argument("--policy", "-p", type=str, help="authz policy file", required=True)


def load_policy(policy_file_path):
    with open(policy_file_path) as file:
        config = json.load(file)
        policy, err = parse_policy_config(config, COMMAND_CATEGORIES)
        if err:
            print("Policy config error: {}".format(err))
            sys.exit(1)
    return policy


def run_command(args):
    policy = load_policy(args.policy)
    commander = Commander(policy)
    commander.cmdloop(intro="Type help or ? to list commands.")


def main():
    """Tool to help preview and see the details of an authorization policy with command line commands."""
    parser = argparse.ArgumentParser()
    define_authz_preview_parser(parser)
    args = parser.parse_args()
    run_command(args)


if __name__ == "__main__":
    main()
