#!/usr/bin/env python3

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.tool.code_pre_installer.install import define_pre_install_parser
from nvflare.tool.code_pre_installer.install import install as install_run
from nvflare.tool.code_pre_installer.prepare import define_prepare_parser
from nvflare.tool.code_pre_installer.prepare import prepare as prepare_run


def def_pre_install_parser(cmd, sub_cmd):
    parser = sub_cmd.add_parser(cmd)

    # Add subcommands
    pre_install_parser = parser.add_subparsers(title=cmd, dest="pre_install_sub_cmd", help="pre-install subcommand")

    # Add prepare subcommand
    define_prepare_parser("prepare", pre_install_parser)

    # Add install subcommand
    define_pre_install_parser("install", pre_install_parser)

    return {cmd: parser}


def handle_pre_install_cmd(args):
    """Handle pre-install commands."""
    if args.pre_install_sub_cmd == "prepare":
        prepare_run(args)
    elif args.pre_install_sub_cmd == "install":
        install_run(args)
    else:
        raise RuntimeError("Unknown pre-install subcommand")
