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

_PACKAGE_EXAMPLES = [
    "nvflare package hospital-1.signed.zip --fingerprint <expected_fingerprint>",
    "nvflare package hospital-1.signed.zip --request-dir ./hospital-1 --fingerprint <expected_fingerprint>",
]

_PACKAGE_HELP_EXAMPLES = """Examples:
  Build one kit from an approved signed zip:
    nvflare package hospital-1.signed.zip --fingerprint <expected_fingerprint>

  Build with an explicit local request directory and non-interactive root CA verification:
    nvflare package hospital-1.signed.zip --request-dir ./hospital-1 \\
      --fingerprint <expected_fingerprint>

  Custom builders are honored when they are present in the local participant definition
  saved by nvflare cert request.
"""


def _add_compat_output_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output",
        dest="compat_output_format",
        choices=["json", "quiet"],
        default=None,
        help=argparse.SUPPRESS,
    )


def def_package_cli_parser(sub_cmd) -> dict:
    """Register 'nvflare package' with the top-level sub_cmd parser."""
    global _package_parser
    p = sub_cmd.add_parser(
        "package",
        description=(
            "Assemble a startup kit from a distributed provisioning signed zip. "
            "No signature.json is generated; certificate-based connection security is the trust anchor."
        ),
        help="Assemble a startup kit from a signed approval zip.",
        epilog=_PACKAGE_HELP_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "input",
        help="Approved signed zip returned by 'nvflare cert approve' (for example hospital-1.signed.zip).",
    )
    p.add_argument(
        "-w",
        "--workspace",
        required=False,
        default="workspace",
        dest="workspace",
        help=(
            "Workspace root directory. Signed zip output goes to "
            "<workspace>/<project-name>/prod_<provision_version>/<name>/. Default: workspace"
        ),
    )
    p.add_argument(
        "--request-dir",
        required=False,
        default=None,
        dest="request_dir",
        help="Local request directory containing the private key for signed zip mode.",
    )
    p.add_argument(
        "--fingerprint",
        "--expected-fingerprint",
        required=False,
        default=None,
        dest="expected_fingerprint",
        help=(
            "Expected SHA256 fingerprint for rootCA.pem in signed zip. "
            "Use this for non-interactive out-of-band root CA verification."
        ),
    )
    p.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Allow replacing an existing participant output when packaging into the signed provision version.",
    )
    p.add_argument(
        "--schema",
        action="store_true",
        default=False,
        help="Print JSON schema for this command's arguments and exit.",
    )
    _add_compat_output_arg(p)
    _package_parser = p
    return {"package": p}


def handle_package_cmd(args):
    """Dispatch to package handler."""
    from nvflare.tool.cli_output import set_output_format
    from nvflare.tool.package.package_commands import handle_package

    compat_output_format = getattr(args, "compat_output_format", None)
    if compat_output_format:
        set_output_format("json" if compat_output_format == "json" else "txt")

    return handle_package(args)
