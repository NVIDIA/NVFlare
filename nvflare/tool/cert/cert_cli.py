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

_VALID_CERT_TYPES = ["client", "server", "org_admin", "lead", "member"]

# Module-level parser references — used by --schema in handlers and for help fallback
_cert_init_parser: Optional[argparse.ArgumentParser] = None
_cert_csr_parser: Optional[argparse.ArgumentParser] = None
_cert_sign_parser: Optional[argparse.ArgumentParser] = None
_cert_parser: Optional[argparse.ArgumentParser] = None


def _name_type(value: str) -> str:
    """Argparse type function: validate name length."""
    if len(value) > 64:
        raise argparse.ArgumentTypeError(f"name must be 64 characters or fewer (got {len(value)})")
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
    p.add_argument(
        "--project",
        required=False,
        default=None,
        type=_name_type,
        dest="project",
        metavar="PROJECT_NAME",
        help="Project name. Used as the CN of the root CA certificate. Max 64 chars.",
    )
    p.add_argument(
        "-o",
        "--output-dir",
        required=False,
        default=None,
        dest="output_dir",
        metavar="OUTPUT_DIR",
        help="Directory where CA files are written. Created if it does not exist.",
    )
    p.add_argument(
        "--org",
        required=False,
        default=None,
        dest="org",
        metavar="ORG",
        help="Organization name for the CA certificate (O field). Default: omitted.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Overwrite existing CA files without prompting. Backs up existing files first.",
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


def _def_cert_csr_parser(cert_sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    global _cert_csr_parser
    p = cert_sub.add_parser(
        "csr",
        description="Generate a private key and CSR for a participant. Site Admin or job submitter.",
        help="Generate a private key and CSR for manual provisioning.",
    )
    p.add_argument(
        "-n",
        "--name",
        required=False,
        default=None,
        type=_name_type,
        dest="name",
        metavar="NAME",
        help="Participant name (e.g. hospital-1, fl-server). Used as the cert Common Name. Max 64 chars.",
    )
    p.add_argument(
        "-o",
        "--output-dir",
        required=False,
        default=None,
        dest="output_dir",
        metavar="OUTPUT_DIR",
        help="Output directory for the .key and .csr files.",
    )
    p.add_argument(
        "--org",
        required=False,
        default=None,
        dest="org",
        metavar="ORG",
        help="Organization name for the certificate.",
    )
    p.add_argument(
        "--project-file",
        required=False,
        default=None,
        dest="project_file",
        metavar="SITE_YAML",
        help="Single-site YAML with name/org/type. Use this instead of --name/--org/--type.",
    )
    p.add_argument(
        "-t",
        "--type",
        required=False,
        default=None,
        dest="cert_type",
        choices=_VALID_CERT_TYPES,
        help=(
            "Proposed certificate type. Embedded in the CSR as a hint for the Project Admin. "
            "The Project Admin may override this when running 'nvflare cert sign'. "
            "Typically set by the org admin on behalf of the participant."
        ),
    )
    p.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Overwrite existing key/CSR without prompting.",
    )
    p.add_argument(
        "--schema",
        action="store_true",
        default=False,
        help="Print JSON schema for this command and exit.",
    )
    _add_compat_output_arg(p)
    _cert_csr_parser = p
    return p


def _def_cert_sign_parser(cert_sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    global _cert_sign_parser
    p = cert_sub.add_parser(
        "sign",
        description="Sign a CSR with the project root CA. Project Admin only.",
        help="Sign a CSR with the project root CA (Project Admin only).",
    )
    p.add_argument(
        "-r",
        "--csr",
        required=False,
        default=None,
        dest="csr_path",
        metavar="CSR_FILE",
        help="Path to the .csr file received from the site admin.",
    )
    p.add_argument(
        "-c",
        "--ca-dir",
        required=False,
        default=None,
        dest="ca_dir",
        metavar="CA_DIR",
        help="Directory containing rootCA.pem, rootCA.key, and ca.json.",
    )
    p.add_argument(
        "-o",
        "--output-dir",
        required=False,
        default=None,
        dest="output_dir",
        metavar="OUTPUT_DIR",
        help="Output directory for the signed certificate and rootCA.pem copy.",
    )
    p.add_argument(
        "-t",
        "--type",
        required=False,
        default=None,
        dest="cert_type",
        choices=_VALID_CERT_TYPES,
        help="Cert type to issue. Authoritative — embedded in signed cert UNSTRUCTURED_NAME.",
    )
    p.add_argument(
        "--valid-days",
        required=False,
        type=int,
        default=1095,
        dest="valid_days",
        help="Certificate validity in days. Default: 1095 (3 years).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Overwrite existing signed cert without prompting.",
    )
    p.add_argument(
        "--schema",
        action="store_true",
        default=False,
        help="Print JSON schema for this command and exit.",
    )
    _add_compat_output_arg(p)
    _cert_sign_parser = p
    return p


def _ensure_parsers_initialized() -> None:
    """Ensure module-level parser references are populated.

    The parsers are normally registered when the CLI entry point calls
    ``def_cert_cli_parser``.  In contexts where that has not happened (unit
    tests, ``--schema`` invoked standalone) this function performs a one-time
    initialization using a throwaway top-level parser so that the module-level
    ``_cert_*_parser`` references are populated.
    """
    global _cert_init_parser, _cert_csr_parser, _cert_sign_parser
    if _cert_init_parser is None or _cert_csr_parser is None or _cert_sign_parser is None:
        import argparse as _argparse

        _dummy = _argparse.ArgumentParser()
        _sub = _dummy.add_subparsers()
        _cert_sub = _sub.add_parser("cert").add_subparsers()
        _def_cert_init_parser(_cert_sub)
        _def_cert_csr_parser(_cert_sub)
        _def_cert_sign_parser(_cert_sub)


def def_cert_cli_parser(sub_cmd) -> dict:
    """Register 'nvflare cert' and its subcommands with the top-level sub_cmd parser."""
    global _cert_parser
    _cert_parser = sub_cmd.add_parser(
        "cert",
        help="Certificate management for distributed provisioning.",
        description="Manage certificates for FLARE distributed (manual) provisioning.",
    )
    cert_sub = _cert_parser.add_subparsers(dest="cert_sub_command")
    _def_cert_init_parser(cert_sub)
    _def_cert_csr_parser(cert_sub)
    _def_cert_sign_parser(cert_sub)
    return {"cert": _cert_parser}


def handle_cert_cmd(args):
    """Dispatch to the appropriate cert subcommand handler."""
    from nvflare.tool.cert.cert_commands import handle_cert_csr, handle_cert_init, handle_cert_sign
    from nvflare.tool.cli_output import output_usage_error, set_output_format

    compat_output_format = getattr(args, "compat_output_format", None)
    if compat_output_format:
        set_output_format(compat_output_format)

    dispatch = {
        "init": handle_cert_init,
        "csr": handle_cert_csr,
        "sign": handle_cert_sign,
    }
    handler = dispatch.get(getattr(args, "cert_sub_command", None))
    if not handler:
        detail = (
            "cert subcommand required" if getattr(args, "cert_sub_command", None) is None else "invalid cert subcommand"
        )
        output_usage_error(_cert_parser, detail, exit_code=4)
    handler(args)
