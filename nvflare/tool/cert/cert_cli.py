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

"""nvflare cert subcommand: parser registration and dispatch."""

import argparse
from typing import Optional

from nvflare.tool.cert.cert_constants import is_valid_provision_version

# Module-level parser references used by --schema in handlers and for help fallback.
_cert_init_parser: argparse.ArgumentParser | None = None
_cert_request_parser: argparse.ArgumentParser | None = None
_cert_approve_parser: argparse.ArgumentParser | None = None
_cert_parser: argparse.ArgumentParser | None = None


def _name_type(value: str) -> str:
    """Argparse type function: validate name length."""
    if len(value) > 64:
        raise argparse.ArgumentTypeError(f"name must be 64 characters or fewer (got {len(value)})")
    return value


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return parsed


def _provision_version_type(value: str) -> str:
    if not is_valid_provision_version(value):
        raise argparse.ArgumentTypeError("version must be exactly two digits, for example 00")
    return value


def _add_compat_output_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output",
        dest="compat_output_format",
        choices=["json", "txt", "human"],
        default=None,
        help=argparse.SUPPRESS,
    )


def _def_cert_init_parser(cert_sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    global _cert_init_parser
    p = cert_sub.add_parser(
        "init",
        description="Initialize a root CA for distributed provisioning. Project Admin only. One-time per federation.",
        help="Initialize root CA for a distributed provisioning federation (Project Admin only).",
    )
    profile_arg = p.add_argument(
        "--profile",
        required=False,
        default=None,
        dest="profile",
        metavar="PROJECT_PROFILE",
        help="Project profile yaml file. The profile name is used as the CN of the root CA certificate.",
    )
    profile_arg.schema_required = True
    output_dir_arg = p.add_argument(
        "-o",
        "--output-dir",
        required=False,
        default=None,
        dest="output_dir",
        metavar="OUTPUT_DIR",
        help="Directory where CA files are written. Created if it does not exist.",
    )
    output_dir_arg.schema_required = True
    p.add_argument(
        "--org",
        required=False,
        default=None,
        dest="org",
        metavar="ORG",
        help="Organization name for the CA certificate (O field). Default: omitted.",
    )
    p.add_argument(
        "--valid-days",
        required=False,
        type=_positive_int,
        default=3650,
        dest="valid_days",
        metavar="DAYS",
        help="Validity period for the root CA certificate in days. Default: 3650.",
    )
    p.add_argument(
        "--deploy-version",
        required=False,
        type=_provision_version_type,
        default="00",
        dest="deploy_version",
        metavar="NN",
        help=(
            "Deployment generation used for package output directory prod_<NN>. "
            "Default: 00. Normally ignore this; use 01, 02, etc. only when creating a new deployment CA."
        ),
    )
    p.add_argument(
        "--force",
        action="store_true",
        default=False,
        help=(
            "Replace existing CA files only when --deploy-version differs from the existing CA deploy version. "
            "Fails with an error if --deploy-version matches the existing CA deploy version. "
            "If ca.json is absent, --deploy-version is not checked. Backs up existing files first."
        ),
    )
    p.add_argument(
        "--schema",
        action="store_true",
        default=False,
        help="Print JSON schema for this command and exit.",
    )
    _add_compat_output_arg(p)
    _cert_init_parser = p
    return p


def _def_cert_request_parser(cert_sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    global _cert_request_parser
    p = cert_sub.add_parser(
        "request",
        description="Create a local key, CSR, metadata, and request zip.",
        help="Create a distributed provisioning request zip.",
    )
    p.add_argument(
        "-p",
        "--participant",
        required=True,
        dest="participant",
        help="Participant definition YAML with top-level name and participants[0].",
    )
    p.add_argument(
        "--out",
        required=False,
        default=None,
        dest="output_dir",
        help="Request folder. Default: ./<name>.",
    )
    p.add_argument("--force", action="store_true", default=False, help="Overwrite existing request files.")
    p.add_argument("--schema", action="store_true", default=False, help="Print JSON schema for this command and exit.")
    _add_compat_output_arg(p)
    _cert_request_parser = p
    return p


def _def_cert_approve_parser(cert_sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    global _cert_approve_parser
    p = cert_sub.add_parser(
        "approve",
        description="Approve a request zip and create a signed zip. Project Admin only.",
        help="Approve a distributed provisioning request zip.",
    )
    p.add_argument("request_zip", help="Request zip produced by 'nvflare cert request'.")
    p.add_argument(
        "-c",
        "--ca-dir",
        required=True,
        dest="ca_dir",
        help="Directory containing rootCA.pem, rootCA.key, and ca.json.",
    )
    p.add_argument(
        "--out",
        required=False,
        default=None,
        dest="signed_zip",
        help="Signed zip output path. Default: <name>.signed.zip.",
    )
    p.add_argument(
        "--profile",
        required=True,
        dest="profile",
        help="Project profile YAML containing name, scheme, and connection_security.",
    )
    p.add_argument(
        "--valid-days",
        required=False,
        type=_positive_int,
        default=1095,
        dest="valid_days",
        help="Certificate validity in days. Default: 1095 (3 years).",
    )
    p.add_argument("--force", action="store_true", default=False, help="Overwrite existing signed zip.")
    p.add_argument("--schema", action="store_true", default=False, help="Print JSON schema for this command and exit.")
    _add_compat_output_arg(p)
    _cert_approve_parser = p
    return p


def _ensure_parsers_initialized() -> None:
    """Ensure module-level parser references are populated.

    The parsers are normally registered when the CLI entry point calls
    ``def_cert_cli_parser``.  In contexts where that has not happened (unit
    tests, ``--schema`` invoked standalone) this function performs a one-time
    initialization using a throwaway top-level parser so that the module-level
    ``_cert_*_parser`` references are populated.
    """
    global _cert_init_parser, _cert_request_parser, _cert_approve_parser
    if _cert_init_parser is None or _cert_request_parser is None or _cert_approve_parser is None:
        import argparse as _argparse

        _dummy = _argparse.ArgumentParser()
        _sub = _dummy.add_subparsers()
        _cert_sub = _sub.add_parser("cert").add_subparsers(metavar="{init,request,approve}")
        _def_cert_init_parser(_cert_sub)
        _def_cert_request_parser(_cert_sub)
        _def_cert_approve_parser(_cert_sub)


def def_cert_cli_parser(sub_cmd) -> dict:
    """Register 'nvflare cert' and its subcommands with the top-level sub_cmd parser."""
    global _cert_parser
    _cert_parser = sub_cmd.add_parser(
        "cert",
        help="Certificate management for distributed provisioning.",
        description="Manage certificates for FLARE distributed provisioning.",
    )
    cert_sub = _cert_parser.add_subparsers(dest="cert_sub_command", metavar="{init,request,approve}")
    _def_cert_init_parser(cert_sub)
    _def_cert_request_parser(cert_sub)
    _def_cert_approve_parser(cert_sub)
    return {"cert": _cert_parser}


def handle_cert_cmd(args):
    """Dispatch to the appropriate cert subcommand handler."""
    from nvflare.tool.cert.cert_commands import handle_cert_approve, handle_cert_init, handle_cert_request
    from nvflare.tool.cli_output import output_usage_error, set_output_format

    compat_output_format = getattr(args, "compat_output_format", None)
    if compat_output_format:
        set_output_format(compat_output_format)

    dispatch = {
        "init": handle_cert_init,
        "request": handle_cert_request,
        "approve": handle_cert_approve,
    }
    handler = dispatch.get(getattr(args, "cert_sub_command", None))
    if not handler:
        detail = (
            "cert subcommand required" if getattr(args, "cert_sub_command", None) is None else "invalid cert subcommand"
        )
        output_usage_error(_cert_parser, detail, exit_code=4)
        return 1
    return handler(args)
