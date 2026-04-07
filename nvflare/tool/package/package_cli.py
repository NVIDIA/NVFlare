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
from typing import Optional

_package_parser: Optional[argparse.ArgumentParser] = None


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
        required=False,
        default=None,
        dest="kit_type",
        choices=["client", "server", "org_admin", "lead", "member"],
        help=(
            "In yaml mode (--project-file), optional filter to build only participants of this type. "
            "Not used in single-participant mode: the kit type is always derived from the signed "
            "certificate's embedded type (set by 'nvflare cert sign')."
        ),
    )
    p.add_argument(
        "-e",
        "--endpoint",
        required=False,
        default=None,
        help="Server endpoint URI (grpc://host:port, tcp://host:port, or http://host:port). Required for all kit types.",
    )
    p.add_argument(
        "-n",
        "--name",
        required=False,
        default=None,
        help=(
            "Participant name. Auto-detected from *.key filename when --dir is used. "
            "For admin kit types (org_admin, lead, member), must be an email address "
            "(e.g. alice@myorg.com)."
        ),
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
        "-w",
        "--workspace",
        required=False,
        default="workspace",
        dest="workspace",
        help="Workspace root directory. Output goes to <workspace>/<project-name>/prod_NN/<name>/. Default: workspace",
    )
    p.add_argument(
        "--project-name",
        required=False,
        default=None,
        dest="project_name",
        help=(
            "Project name. Used in the output path (<workspace>/<project-name>/prod_NN/<name>/) "
            "and in fed_server.json/fed_admin.json for challenge-response auth. Default: project"
        ),
    )
    p.add_argument(
        "-p",
        "--project-file",
        required=False,
        default=None,
        dest="project_file",
        help=(
            "Site-scoped project YAML defining participants and optional custom builders "
            "(schema-compatible with 'nvflare provision' project.yaml). "
            "When given, -t becomes an optional type filter. "
            "WorkspaceBuilder and StaticFileBuilder are always managed by nvflare package "
            "(scheme is derived from --endpoint); any YAML entries for these builders, "
            "including custom args such as config_folder, are ignored. "
            "Mutually exclusive with -n and --cert/--key/--rootca. Use -e, --dir, and -p together."
        ),
    )
    p.add_argument(
        "--admin-port",
        required=False,
        type=int,
        default=None,
        dest="admin_port",
        help="Server admin port. Default: same as service port (single-port mode).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Allow re-packaging when this participant name already appears in the most recent prod_NN directory (a new prod_NN is created alongside).",
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
