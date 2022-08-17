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

from nvflare.lighter.poc_commands import def_poc_parser, handle_poc_cmd, is_poc
from nvflare.lighter.provision import define_provision_parser, handle_provision
from nvflare.tool.preflight_check import define_preflight_check_parser, check_packages


def is_provision(cmd_args) -> bool:
    return (
            hasattr(cmd_args, "add_user")
            or hasattr(cmd_args, "add_client")
            or hasattr(cmd_args, "project_file")
            or hasattr(cmd_args, "ui_tool")
    )


def is_preflight_checker(cmd_args) -> bool:
    print(cmd_args)
    return (
            hasattr(cmd_args, "package_root")
            or hasattr(cmd_args, "packages]")
    )


def def_provision_parser(sub_cmd):
    provision_parser = sub_cmd.add_parser("provision")
    define_provision_parser(provision_parser)


def def_preflight_check_parser(sub_cmd):
    checker_parser = sub_cmd.add_parser("preflight_check")
    define_preflight_check_parser(checker_parser)


def parse_args(prog_name: str):
    _parser = argparse.ArgumentParser(description=prog_name)
    sub_cmd = _parser.add_subparsers(description="sub command parser")
    def_poc_parser(sub_cmd)
    def_provision_parser(sub_cmd)
    def_preflight_check_parser(sub_cmd)
    return _parser, _parser.parse_args()


def run(prog_name):
    cwd = os.getcwd()
    sys.path.append(cwd)
    prog_parser, prog_args = parse_args(prog_name)
    try:
        if is_poc(prog_args):
            handle_poc_cmd(prog_args)
        elif is_provision(prog_args):
            handle_provision(prog_args)
        elif is_preflight_checker(prog_args):
            check_packages(prog_args)
        else:
            prog_parser.print_help()
    except Exception as e:
        print("unable to handle command, please check syntax")
        prog_parser.print_help()


def main():
    run("nvflare")


if __name__ == "__main__":
    main()
