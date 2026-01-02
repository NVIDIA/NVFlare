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

"""CLI commands for FLARE Enrollment.

This module provides the `nvflare enrollment` subcommand for managing
enrollment tokens and related operations.

Environment Variables:
    NVFLARE_CA_PATH: Path to directory containing rootCA.pem and rootCA.key
    NVFLARE_ENROLLMENT_POLICY: Path to policy YAML file (optional, uses default if not set)

Usage:
    # Minimal (with env vars set)
    nvflare enrollment token generate -s site-1

    # With explicit options
    nvflare enrollment token generate -s site-1 -c /path/to/ca -p policy.yaml

    # Batch generate
    nvflare enrollment token batch -n 10 --prefix hospital -o tokens.csv

    # Inspect token
    nvflare enrollment token info -t <jwt_token>
"""

import os
import sys
import tempfile

CMD_ENROLLMENT = "enrollment"
SUBCMD_TOKEN = "token"

# Token subcommands
TOKEN_GENERATE = "generate"
TOKEN_BATCH = "batch"
TOKEN_INFO = "info"

# Environment variable names
ENV_CA_PATH = "NVFLARE_CA_PATH"
ENV_ENROLLMENT_POLICY = "NVFLARE_ENROLLMENT_POLICY"

# Built-in default policy (simple auto-approve for quick start)
DEFAULT_POLICY = """
metadata:
  project: "nvflare-default"
  description: "Built-in default policy for quick start"
  version: "1.0"

token:
  validity: 7d
  max_uses: 1

site:
  name_pattern: "*"

user:
  allowed_roles:
    - researcher
    - org_admin
    - project_admin
  default_role: researcher

approval:
  method: policy
  rules:
    - name: "auto-approve-all"
      description: "Auto-approve all enrollment requests"
      match: {}
      action: approve

notifications:
  enabled: false
"""


def _check_jwt_dependency():
    """Check if PyJWT is installed (optional dependency)."""
    try:
        import jwt  # noqa: F401

        return True
    except ImportError:
        print("\nError: PyJWT is required for enrollment token operations.")
        print("Install it with: pip install PyJWT")
        print("Or install nvflare with enrollment support: pip install nvflare[enrollment]")
        sys.exit(1)


def _get_ca_path(args):
    """Resolve CA path from args or environment variable.

    Priority: CLI arg > Environment variable
    """
    ca_path = getattr(args, "ca_path", None)
    if ca_path:
        return ca_path

    ca_path = os.environ.get(ENV_CA_PATH)
    if ca_path:
        return ca_path

    print("\nError: CA path is required.")
    print(f"Provide via -c/--ca_path or set {ENV_CA_PATH} environment variable.")
    sys.exit(1)


def _get_policy_path(args):
    """Resolve policy path from args, environment variable, or use built-in default.

    Priority: CLI arg > Environment variable > Built-in default

    Returns:
        tuple: (policy_path, is_temp_file) - is_temp_file indicates if caller should clean up
    """
    policy_path = getattr(args, "policy", None)
    if policy_path:
        return policy_path, False

    policy_path = os.environ.get(ENV_ENROLLMENT_POLICY)
    if policy_path:
        return policy_path, False

    # Use built-in default policy - write to temp file
    fd, temp_path = tempfile.mkstemp(suffix=".yaml", prefix="nvflare_default_policy_")
    try:
        os.write(fd, DEFAULT_POLICY.encode("utf-8"))
    finally:
        os.close(fd)

    return temp_path, True


# =============================================================================
# Token Subcommand Parsers
# =============================================================================


def define_token_generate_parser(sub_parser):
    """Define parser for 'nvflare enrollment token generate' command."""
    parser = sub_parser.add_parser(
        TOKEN_GENERATE,
        help="Generate a single enrollment token",
        description=(
            "Generate a policy-based enrollment token (JWT) for a single subject.\n\n"
            f"Environment variables:\n"
            f"  {ENV_CA_PATH}: Path to CA directory (alternative to -c)\n"
            f"  {ENV_ENROLLMENT_POLICY}: Path to policy file (alternative to -p)"
        ),
        formatter_class=lambda prog: __import__("argparse").RawDescriptionHelpFormatter(prog, max_help_position=40),
    )

    # Required arguments
    parser.add_argument(
        "-s",
        "--subject",
        type=str,
        required=True,
        help="Subject identifier (site name, user email, or pattern like 'hospital-*')",
    )

    # Optional arguments (can be set via env vars)
    parser.add_argument(
        "-c",
        "--ca_path",
        type=str,
        default=None,
        help=f"Path to CA directory (or set {ENV_CA_PATH})",
    )
    parser.add_argument(
        "-p",
        "--policy",
        type=str,
        default=None,
        help=f"Path to policy YAML file (or set {ENV_ENROLLMENT_POLICY}, uses default if not set)",
    )

    # Optional arguments with defaults
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        default="client",
        choices=["client", "admin", "relay", "pattern"],
        help="Subject type (default: client)",
    )
    parser.add_argument(
        "-v",
        "--validity",
        type=str,
        default=None,
        help="Token validity duration (e.g., '7d', '24h', '30m'). Defaults to policy setting.",
    )
    parser.add_argument(
        "-r",
        "--roles",
        type=str,
        nargs="+",
        default=None,
        help="Roles for admin tokens (e.g., 'org_admin researcher')",
    )
    parser.add_argument(
        "--source_ips",
        type=str,
        nargs="+",
        default=None,
        help="Source IP restrictions in CIDR format (e.g., '10.0.0.0/8')",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output file to save token (prints to stdout if not specified)",
    )


def define_token_batch_parser(sub_parser):
    """Define parser for 'nvflare enrollment token batch' command."""
    parser = sub_parser.add_parser(
        TOKEN_BATCH,
        help="Generate multiple enrollment tokens",
        description=(
            "Generate multiple policy-based enrollment tokens in batch.\n\n"
            f"Environment variables:\n"
            f"  {ENV_CA_PATH}: Path to CA directory (alternative to -c)\n"
            f"  {ENV_ENROLLMENT_POLICY}: Path to policy file (alternative to -p)"
        ),
        formatter_class=lambda prog: __import__("argparse").RawDescriptionHelpFormatter(prog, max_help_position=40),
    )

    # Batch options (one of these required)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-n",
        "--count",
        type=int,
        help="Number of tokens to generate (used with --prefix)",
    )
    group.add_argument(
        "--names",
        type=str,
        nargs="+",
        help="Explicit list of subject names",
    )

    # Required output
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output file to save tokens (.csv or .txt)",
    )

    # Optional arguments (can be set via env vars)
    parser.add_argument(
        "-c",
        "--ca_path",
        type=str,
        default=None,
        help=f"Path to CA directory (or set {ENV_CA_PATH})",
    )
    parser.add_argument(
        "-p",
        "--policy",
        type=str,
        default=None,
        help=f"Path to policy YAML file (or set {ENV_ENROLLMENT_POLICY}, uses default if not set)",
    )

    # Optional arguments with defaults
    parser.add_argument(
        "--prefix",
        type=str,
        default="client",
        help="Prefix for auto-generated names when using --count (default: 'client')",
    )
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        default="client",
        choices=["client", "admin", "relay"],
        help="Subject type for all tokens (default: client)",
    )
    parser.add_argument(
        "-v",
        "--validity",
        type=str,
        default=None,
        help="Token validity duration (e.g., '7d', '24h')",
    )


def define_token_info_parser(sub_parser):
    """Define parser for 'nvflare enrollment token info' command."""
    parser = sub_parser.add_parser(
        TOKEN_INFO,
        help="Display token information",
        description="Decode and display enrollment token information (without verification).",
    )

    parser.add_argument(
        "-t",
        "--token",
        type=str,
        required=True,
        help="JWT token string or path to file containing token",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format",
    )


def define_token_parser(sub_parser):
    """Define parser for 'nvflare enrollment token' subcommand."""
    parser = sub_parser.add_parser(
        SUBCMD_TOKEN,
        help="Manage enrollment tokens",
        description="Commands for generating and inspecting enrollment tokens (JWT).",
    )

    token_sub = parser.add_subparsers(
        title="token commands",
        dest="token_sub_cmd",
        help="Token subcommand",
    )

    define_token_generate_parser(token_sub)
    define_token_batch_parser(token_sub)
    define_token_info_parser(token_sub)

    return parser


# =============================================================================
# Main Enrollment Parser
# =============================================================================


def def_enrollment_parser(sub_cmd):
    """Define the enrollment subcommand parser.

    Args:
        sub_cmd: Parent subparser to add enrollment command to

    Returns:
        Dict mapping command name to parser
    """
    cmd = CMD_ENROLLMENT
    parser = sub_cmd.add_parser(
        cmd,
        help="Enrollment token and certificate management",
        description=(
            "Commands for managing FLARE enrollment tokens and certificates.\n\n"
            f"Environment variables:\n"
            f"  {ENV_CA_PATH}: Default CA directory path\n"
            f"  {ENV_ENROLLMENT_POLICY}: Default policy file path"
        ),
        formatter_class=lambda prog: __import__("argparse").RawDescriptionHelpFormatter(prog, max_help_position=40),
    )

    enrollment_sub = parser.add_subparsers(
        title="enrollment commands",
        dest="enrollment_sub_cmd",
        help="Enrollment subcommand",
    )

    # Add token subcommand
    define_token_parser(enrollment_sub)

    # Future: Add more enrollment subcommands here
    # define_status_parser(enrollment_sub)
    # define_revoke_parser(enrollment_sub)

    return {cmd: parser}


# =============================================================================
# Command Handlers
# =============================================================================


def handle_token_generate_cmd(args):
    """Handle 'nvflare enrollment token generate' command."""
    _check_jwt_dependency()

    from nvflare.lighter.constants import ParticipantType
    from nvflare.tool.enrollment.token_service import TokenService

    # Map type argument to ParticipantType
    type_map = {
        "client": ParticipantType.CLIENT,
        "admin": ParticipantType.ADMIN,
        "relay": ParticipantType.RELAY,
        "pattern": TokenService.SUBJECT_TYPE_PATTERN,
    }

    # Resolve CA path and policy
    ca_path = _get_ca_path(args)
    policy_path, is_temp_policy = _get_policy_path(args)

    try:
        service = TokenService(ca_path)

        # Build claims
        claims = {}
        if args.roles:
            claims["roles"] = args.roles
        if args.source_ips:
            claims["source_ips"] = args.source_ips

        token = service.generate_token_from_file(
            policy_file=policy_path,
            subject=args.subject,
            subject_type=type_map[args.type],
            validity=args.validity,
            **claims,
        )

        if args.output:
            with open(args.output, "w") as f:
                f.write(token)
            print(f"Token saved to: {args.output}")
        else:
            print("\nGenerated Enrollment Token:")
            print("-" * 60)
            print(token)
            print("-" * 60)

            # Also show token info
            info = service.get_token_info(token)
            print("\nToken Info:")
            print(f"  Subject: {info['subject']}")
            print(f"  Type: {info['subject_type']}")
            print(f"  Expires: {info['expires_at']}")
            if is_temp_policy:
                print("  Policy: (using built-in default)")

    except FileNotFoundError as e:
        print(f"\nError: File not found - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nError generating token: {e}")
        sys.exit(1)
    finally:
        # Clean up temp policy file if we created one
        if is_temp_policy and os.path.exists(policy_path):
            os.remove(policy_path)


def handle_token_batch_cmd(args):
    """Handle 'nvflare enrollment token batch' command."""
    _check_jwt_dependency()

    from nvflare.lighter.constants import ParticipantType
    from nvflare.tool.enrollment.token_service import TokenService

    type_map = {
        "client": ParticipantType.CLIENT,
        "admin": ParticipantType.ADMIN,
        "relay": ParticipantType.RELAY,
    }

    # Resolve CA path and policy
    ca_path = _get_ca_path(args)
    policy_path, is_temp_policy = _get_policy_path(args)

    try:
        service = TokenService(ca_path)

        results = service.batch_generate_tokens(
            policy_file=policy_path,
            count=args.count or 0,
            name_prefix=args.prefix,
            names=args.names,
            subject_type=type_map[args.type],
            validity=args.validity,
            output_file=args.output,
        )

        print(f"\nGenerated {len(results)} tokens")
        print(f"Saved to: {args.output}")
        if is_temp_policy:
            print("Policy: (using built-in default)")

        # Show first few as preview
        print("\nPreview (first 3):")
        for item in results[:3]:
            print(f"  {item['name']}: {item['token'][:50]}...")

    except Exception as e:
        print(f"\nError generating tokens: {e}")
        sys.exit(1)
    finally:
        # Clean up temp policy file if we created one
        if is_temp_policy and os.path.exists(policy_path):
            os.remove(policy_path)


def handle_token_info_cmd(args):
    """Handle 'nvflare enrollment token info' command."""
    _check_jwt_dependency()

    import json

    # Get token from argument or file
    token = args.token
    if os.path.isfile(token):
        with open(token, "r") as f:
            token = f.read().strip()

    try:
        import jwt

        payload = jwt.decode(token, options={"verify_signature": False})

        info = {
            "token_id": payload.get("jti"),
            "subject": payload.get("sub"),
            "subject_type": payload.get("subject_type"),
            "issuer": payload.get("iss"),
            "issued_at": payload.get("iat"),
            "expires_at": payload.get("exp"),
            "max_uses": payload.get("max_uses", 1),
            "roles": payload.get("roles"),
            "source_ips": payload.get("source_ips"),
            "policy_project": payload.get("policy", {}).get("metadata", {}).get("project"),
            "policy_version": payload.get("policy", {}).get("metadata", {}).get("version"),
        }

        # Convert timestamps
        from datetime import datetime, timezone

        if info["issued_at"]:
            info["issued_at"] = datetime.fromtimestamp(info["issued_at"], tz=timezone.utc).isoformat()
        if info["expires_at"]:
            info["expires_at"] = datetime.fromtimestamp(info["expires_at"], tz=timezone.utc).isoformat()

        if args.json:
            print(json.dumps(info, indent=2))
        else:
            print("\nToken Information:")
            print("-" * 40)
            for key, value in info.items():
                if value is not None:
                    print(f"  {key}: {value}")
            print("-" * 40)

    except Exception as e:
        print(f"\nError decoding token: {e}")
        sys.exit(1)


def handle_token_cmd(args):
    """Handle 'nvflare enrollment token' subcommand."""
    token_sub = getattr(args, "token_sub_cmd", None)

    if token_sub == TOKEN_GENERATE:
        handle_token_generate_cmd(args)
    elif token_sub == TOKEN_BATCH:
        handle_token_batch_cmd(args)
    elif token_sub == TOKEN_INFO:
        handle_token_info_cmd(args)
    else:
        from nvflare.cli_unknown_cmd_exception import CLIUnknownCmdException

        raise CLIUnknownCmdException(
            "\nPlease specify a token subcommand: generate, batch, or info\n"
            "Use 'nvflare enrollment token -h' for help."
        )


def handle_enrollment_cmd(args):
    """Handle 'nvflare enrollment' command and its subcommands."""
    enrollment_sub = getattr(args, "enrollment_sub_cmd", None)

    if enrollment_sub == SUBCMD_TOKEN:
        handle_token_cmd(args)
    else:
        from nvflare.cli_unknown_cmd_exception import CLIUnknownCmdException

        raise CLIUnknownCmdException(
            "\nPlease specify an enrollment subcommand: token\n" "Use 'nvflare enrollment -h' for help."
        )
