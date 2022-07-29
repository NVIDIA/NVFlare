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


def handle_provision_cmd(cmd_args):
    print("handle provision command")
    pass


#
#
# def is_provision(cmd_args) -> bool:
#     #  todo add provision handling
#     return False
#
#
# def def_provision_parser(sub_cmd, prog_name: str):
#     provision_parser = sub_cmd.add_parser("provision")
#     provision_parser.add_argument(
#         "-n", "--n_clients", type=int, nargs="?", default=2, help="number of sites or clients"
#     )


def parse_args(prog_name: str):
    _parser = argparse.ArgumentParser(description="nvflare parser")
    sub_cmd = _parser.add_subparsers(description="sub command parser")
    def_poc_parser(sub_cmd, prog_name)
    # def_provision_parser(sub_cmd, prog_name)
    return _parser, _parser.parse_args()


def run(prog_name):
    cwd = os.getcwd()
    sys.path.append(cwd)
    prog_parser, prog_args = parse_args(prog_name)

    if is_poc(prog_args):
        handle_poc_cmd(prog_args)
    # elif is_provision(prog_args):
    #     handle_provision_cmd(prog_args)
    else:
        prog_parser.print_help()


def main():
    run("nvflare")


if __name__ == "__main__":
    main()
