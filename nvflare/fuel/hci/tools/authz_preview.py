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

import argparse
import cmd
import json

from nvflare.fuel.hci.cmd_arg_utils import split_to_args
from nvflare.fuel.hci.table import Table
from nvflare.fuel.sec.authz import Policy, validate_policy_config


class Commander(cmd.Cmd):
    def __init__(self, policy: Policy):
        """Command line prompt helper tool for getting information for authorization configurations.

        Args:
            policy: authorization policy object
        """
        cmd.Cmd.__init__(self)
        self.policy = policy
        self.intro = "Type help or ? to list commands.\n"
        self.prompt = ">"

    def do_bye(self, arg):
        """Exits from the client."""
        return True

    def emptyline(self):
        return

    def do_show_users(self, arg):
        users = self.policy.get_users()
        table = Table(["user", "org", "roles"])
        for user_name, user_def in users.items():
            table.add_row([user_name, user_def["org"], ",".join(user_def["roles"])])
        self.write_table(table)

    def _split_to_args(self, arg):
        if len(arg) <= 0:
            return []
        else:
            return split_to_args(arg)

    def do_show_sites(self, arg):
        sites = self.policy.get_sites()
        table = Table(["site", "org"])
        for site_name, org in sites.items():
            table.add_row([site_name, org])
        self.write_table(table)

    def do_show_rights(self, arg):
        rights = self.policy.get_rights()
        table = Table(["name", "description", "default"])
        for name, right_def in rights.items():
            desc = right_def.get("desc", "")
            default = right_def.get("default", "")
            table.add_row([name, desc, "{}".format(default)])
        self.write_table(table)

    def do_show_rules(self, arg):
        rules = self.policy.get_rules()
        table = Table(["name", "description", "default"])
        for name, rule_def in rules.items():
            desc = rule_def.get("desc", "")
            default = rule_def.get("default", "")
            table.add_row([name, desc, "{}".format(default)])
        self.write_table(table)

    def do_show_config(self, arg):
        config = self.policy.get_config()
        self.write_string(json.dumps(config, indent=1))

    def do_show_site_rules(self, arg):
        args = ["show_site_rules"] + self._split_to_args(arg)
        if len(args) != 2:
            self.write_string("Usage: {} site_name".format(args[0]))
            return

        site_name = args[1]
        rules = self.policy.get_rules()
        table = Table(["Rule", "Result"])
        for rule_name, _ in rules.items():
            result, err = self._eval_rule(site_name, rule_name)
            if err:
                self.write_error(err)
                return
            else:
                table.add_row([rule_name, result])
        self.write_table(table)

    def _eval_right(self, user_name, right_name, site_name):
        result, err = self.policy.evaluate_user_right_on_site(
            user_name=user_name, site_name=site_name, right_name=right_name
        )
        if err:
            return result, err
        if result is None:
            rights = self.policy.get_rights()
            right_def = rights[right_name]
            return "({})".format(right_def.get("default", "?")), ""
        else:
            return "{}".format(result), ""

    def do_eval_right(self, arg):
        args = ["eval_right"] + self._split_to_args(arg)
        if len(args) != 4:
            self.write_string("Usage: {} user_name right_name site_name".format(args[0]))
            return

        user_name = args[1]
        right_name = args[2]
        site_name = args[3]
        result, err = self._eval_right(user_name=user_name, site_name=site_name, right_name=right_name)
        if err:
            self.write_error(err)
        else:
            self.write_string(result)

    def do_eval_user(self, arg):
        args = ["eval_user"] + self._split_to_args(arg)
        if len(args) != 3:
            self.write_string("Usage: {} user_name site_name".format(args[0]))
            return

        user_name = args[1]
        site_name = args[2]
        table = Table(["Right", "Result"])
        rights = self.policy.get_rights()
        for right_name, right_def in rights.items():
            result, err = self._eval_right(user_name=user_name, site_name=site_name, right_name=right_name)
            if err:
                self.write_error(err)
                return
            else:
                table.add_row([right_name, result])
        self.write_table(table)

    def _eval_rule(self, site_name, rule_name):
        result, err = self.policy.evaluate_rule_on_site(site_name=site_name, rule_name=rule_name)
        if err:
            return result, err

        if result is None:
            rules = self.policy.get_rules()
            rule_def = rules[rule_name]
            return "({})".format(rule_def.get("default", "?")), ""
        else:
            return "{}".format(result), ""

    def do_eval_rule(self, arg):
        args = ["eval_rule"] + self._split_to_args(arg)
        if len(args) != 3:
            self.write_string("Usage: {} site_name rule_name".format(args[0]))
            return

        site_name = args[1]
        rule_name = args[2]
        result, err = self._eval_rule(site_name=site_name, rule_name=rule_name)
        if err:
            self.write_error(err)
        else:
            self.write_string(result)

    def write_string(self, data: str):
        content = data + "\n"
        self.stdout.write(content)

    def write_table(self, table: Table):
        table.write(self.stdout)

    def write_error(self, err: str):
        content = "Error: " + err + "\n"
        self.stdout.write(content)


def main():
    """Tool to help preview and see the details of an authorization policy with command line commands."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", "-p", type=str, help="authz policy file", required=False, default="")
    parser.add_argument("--defs", "-d", type=str, help="authz definition file", required=False, default="")
    parser.add_argument("--config", "-c", type=str, help="authz config file", required=False, default="")

    args = parser.parse_args()

    if args.policy:
        with open(args.policy) as file:
            config = json.load(file)
            err = validate_policy_config(config)
            if err:
                print("Policy config error: {}".format(err))
                return
    else:
        assert args.defs, "missing authz definition file"
        assert args.config, "missing authz config file"
        with open(args.defs) as file:
            defs = json.load(file)

        with open(args.config) as file:
            config = json.load(file)

        config.update(defs)

    commander = Commander(Policy(config))
    commander.cmdloop(intro="Type help or ? to list commands.")


if __name__ == "__main__":
    main()
