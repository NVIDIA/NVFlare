# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""nvflare deploy subcommands."""

import argparse
import sys
from typing import Optional

from nvflare.tool.cli_output import output_usage_error
from nvflare.tool.cli_schema import handle_schema_flag

_deploy_root_parser: Optional[argparse.ArgumentParser] = None
_deploy_prepare_parser: Optional[argparse.ArgumentParser] = None

_DEPLOY_PREPARE_EXAMPLES = [
    "nvflare deploy prepare ./site-1",
    "nvflare deploy prepare ./site-1 --output ./site-1-docker --config docker.yaml",
    "nvflare deploy prepare ./site-1 --output ./site-1-k8s --config k8s.yaml",
]


def def_deploy_cli_parser(sub_cmd) -> dict:
    """Register 'nvflare deploy' with the top-level sub_cmd parser."""
    global _deploy_root_parser, _deploy_prepare_parser

    cmd = "deploy"
    parser = sub_cmd.add_parser(cmd, help="prepare startup kits for deployment runtimes")
    _deploy_root_parser = parser
    deploy_subparser = parser.add_subparsers(title="deploy subcommands", metavar="", dest="deploy_sub_cmd")

    prepare_parser = deploy_subparser.add_parser(
        "prepare",
        description="Prepare an existing server/client startup kit for Docker or Kubernetes.",
        help="prepare a startup kit for Docker or Kubernetes",
    )
    prepare_parser.add_argument("kit", nargs="?", help="Existing input startup kit directory.")
    prepare_parser.add_argument("--kit", dest="kit_flag", help="Existing input startup kit directory.")
    prepare_parser.add_argument(
        "--output",
        help="Directory to write the prepared startup kit copy. Defaults to <kit>/prepared/<runtime>.",
    )
    prepare_parser.add_argument(
        "--config",
        help="YAML runtime config file with top-level runtime: docker or runtime: k8s. Defaults to <kit>/config.yaml.",
    )
    prepare_parser.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    _deploy_prepare_parser = prepare_parser

    return {cmd: parser}


def handle_deploy_cmd(args):
    sub_cmd = getattr(args, "deploy_sub_cmd", None)
    if sub_cmd == "prepare":
        handle_schema_flag(_deploy_prepare_parser, "nvflare deploy prepare", _DEPLOY_PREPARE_EXAMPLES, sys.argv[1:])

        from nvflare.tool.deploy.deploy_commands import prepare_deployment

        prepare_deployment(args)
        return

    output_usage_error(_deploy_root_parser, "deploy subcommand required", exit_code=4)
    raise SystemExit(4)
