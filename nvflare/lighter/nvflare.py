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
import os
import sys

from nvflare.lighter.cli_exception import CLIException
from nvflare.lighter.poc_commands import def_poc_parser, handle_poc_cmd, is_poc
from nvflare.lighter.provision import define_provision_parser, handle_provision
from nvflare.tool.preflight_check import check_packages, define_preflight_check_parser

CMD_POC = "poc"
CMD_PROVISION = "provision"
CMD_PREFLIGHT_CHECK = "preflight_check"


def is_provision(cmd_args) -> bool:
    return (
        hasattr(cmd_args, "add_user")
        or hasattr(cmd_args, "add_client")
        or hasattr(cmd_args, "project_file")
        or hasattr(cmd_args, "ui_tool")
    )


def is_preflight_checker(cmd_args) -> bool:
    print(cmd_args)
    return hasattr(cmd_args, "package_root") or hasattr(cmd_args, "packages]")


def def_provision_parser(sub_cmd):
    cmd = CMD_PROVISION
    provision_parser = sub_cmd.add_parser(cmd)
    define_provision_parser(provision_parser)
    return {cmd: [provision_parser]}


def def_preflight_check_parser(sub_cmd):
    cmd = CMD_PREFLIGHT_CHECK
    checker_parser = sub_cmd.add_parser(cmd)
    define_preflight_check_parser(checker_parser)
    return {cmd: checker_parser}


def parse_args(prog_name: str):
    _parser = argparse.ArgumentParser(description=prog_name)
    sub_cmd = _parser.add_subparsers(description="sub command parser")
    sub_cmd_parsers = {}
    sub_cmd_parsers.update(def_poc_parser(sub_cmd))
    sub_cmd_parsers.update(def_preflight_check_parser(sub_cmd))
    sub_cmd_parsers.update(def_provision_parser(sub_cmd))

    return _parser, _parser.parse_args(), sub_cmd_parsers


def get_sub_cmd(prog_args):
    if is_poc(prog_args):
        return CMD_POC
    elif is_provision(prog_args):
        return CMD_PROVISION
    elif is_preflight_checker(prog_args):
        return CMD_PREFLIGHT_CHECK
    else:
        return None


handlers = {
    CMD_POC: handle_poc_cmd,
    CMD_PROVISION: handle_provision,
    CMD_PREFLIGHT_CHECK: check_packages,
}


def run(prog_name):
    cwd = os.getcwd()
    sys.path.append(cwd)
    prog_parser, prog_args, sub_cmd_parsers = parse_args(prog_name)
    sub_cmd = None
    try:
        sub_cmd = get_sub_cmd(prog_args)
        if sub_cmd:
            handlers[sub_cmd](prog_args)
        else:
            prog_parser.print_help()

    except CLIException as e:
        print(e)
        sys.exit(1)
    except Exception as e:
        print("unable to handle command, please check syntax")
        print_help(prog_parser, sub_cmd, sub_cmd_parsers)


def print_help(prog_parser, sub_cmd, sub_cmd_parsers):
    if sub_cmd:
        sub_parser = sub_cmd_parsers[sub_cmd]
        if sub_parser:
            sub_parser.print_help()
        else:
            prog_parser.print_help()
    else:
        prog_parser.print_help()


def main():
    run("nvflare")


if __name__ == "__main__":
    main()
