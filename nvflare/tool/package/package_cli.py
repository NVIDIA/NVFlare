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

"""nvflare package subcommand: parser registration and dispatch."""

import argparse
from argparse import ArgumentTypeError
from typing import Optional

_package_parser: Optional[argparse.ArgumentParser] = None

_ADMIN_ROLES = {"org_admin", "lead", "member"}


def _parse_bool_arg(x: str) -> bool:
    xl = x.lower()
    if xl in ("true", "1", "yes"):
        return True
    if xl in ("false", "0", "no"):
        return False
    raise ArgumentTypeError(f"Invalid boolean value {x!r}. Use true/false, 1/0, or yes/no.")


def def_package_cli_parser(sub_cmd) -> dict:
    """Register 'nvflare package' with the top-level sub_cmd parser."""
    global _package_parser
    p = sub_cmd.add_parser(
        "package",
        description=(
            "Assemble a startup kit for manual (distributed) provisioning. "
            "No signature.json is generated — mTLS is the trust anchor."
        ),
        help="Assemble a startup kit from a locally generated key and Project Admin cert.",
    )
    p.add_argument(
        "-t",
        "--type",
        required=True,
        dest="kit_type",
        choices=["client", "server", "org_admin", "lead", "member"],
        help="Kit type: client, server, org_admin, lead, or member.",
    )
    p.add_argument(
        "-e",
        "--endpoint",
        required=False,
        default=None,
        help="Server endpoint URI (grpc://host:port or tcp://host:port). Required for all kit types.",
    )
    p.add_argument(
        "-n",
        "--name",
        required=False,
        default=None,
        help="Participant name. Auto-detected from *.key filename when --dir is used.",
    )
    p.add_argument(
        "--dir",
        required=False,
        default=None,
        help="Working directory with key + cert + rootCA by convention. Mutually exclusive with --cert/--key/--rootca.",
    )
    p.add_argument(
        "--cert",
        required=False,
        default=None,
        help="Signed certificate from Project Admin.",
    )
    p.add_argument(
        "--key",
        required=False,
        default=None,
        help="Private key generated locally by 'nvflare cert csr'.",
    )
    p.add_argument(
        "--rootca",
        required=False,
        default=None,
        help="rootCA.pem from Project Admin.",
    )
    p.add_argument(
        "-o",
        "--output-dir",
        required=False,
        default=None,
        dest="output_dir",
        help="Output directory. Default: ./<name>",
    )
    p.add_argument(
        "--server-name",
        required=False,
        default=None,
        dest="server_name",
        help="Server identity name for mTLS validation. Required for client and admin-role types.",
    )
    p.add_argument(
        "--project-name",
        required=False,
        default=None,
        dest="project_name",
        help="Project name used in fed_server.json and fed_admin.json for challenge-response auth. Defaults to server name.",
    )
    p.add_argument(
        "--admin-port",
        required=False,
        type=int,
        default=None,
        dest="admin_port",
        help="Server admin port. Default: service port + 1.",
    )
    p.add_argument(
        "--require-signed-jobs",
        required=False,
        default=None,
        type=_parse_bool_arg,
        dest="require_signed_jobs",
        help="Server only. Reject unsigned job submissions. Default: true.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Overwrite existing output directory without prompting.",
    )
    p.add_argument(
        "--output",
        choices=["json", "quiet"],
        default=None,
        dest="output_fmt",
        help="Output format. Default: human-readable text.",
    )
    p.add_argument(
        "--schema",
        action="store_true",
        default=False,
        help="Print JSON schema for this command's arguments and exit.",
    )
    _package_parser = p
    return {"package": p}


def handle_package_cmd(args):
    """Dispatch to package handler."""
    from nvflare.tool.package.package_commands import handle_package

    handle_package(args)
