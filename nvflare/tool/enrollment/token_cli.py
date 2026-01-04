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

"""CLI commands for FLARE enrollment token management.

This module provides the `nvflare token` command for generating and
inspecting enrollment tokens.

Environment Variables:
    NVFLARE_CA_PATH: Path to directory containing rootCA.pem and rootCA.key
    NVFLARE_ENROLLMENT_POLICY: Path to policy YAML file (optional, uses default if not set)

Usage:
    # Generate single token
    nvflare token generate -s site-1

    # Batch generate
    nvflare token batch -n 10 --prefix hospital -o tokens.csv

    # Inspect token
    nvflare token info -t <jwt_token>
"""

import os
import sys
import tempfile
from typing import Optional

CMD_TOKEN = "token"

# Subcommands
SUBCMD_GENERATE = "generate"
SUBCMD_BATCH = "batch"
SUBCMD_INFO = "info"

# Environment variable names
ENV_CA_PATH = "NVFLARE_CA_PATH"
ENV_ENROLLMENT_POLICY = "NVFLARE_ENROLLMENT_POLICY"
ENV_CERT_SERVICE_URL = "NVFLARE_CERT_SERVICE_URL"
ENV_API_KEY = "NVFLARE_API_KEY"

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
    """Check if PyJWT is installed."""
    try:
        import jwt  # noqa: F401

        return True
    except ImportError:
        print("\nError: PyJWT is required for token operations.")
        print("PyJWT should be installed as part of nvflare dependencies.")
        sys.exit(1)


def _get_ca_path(args, required: bool = True):
    """Resolve CA path from args or environment variable."""
    ca_path = getattr(args, "ca_path", None)
    if ca_path:
        return ca_path

    ca_path = os.environ.get(ENV_CA_PATH)
    if ca_path:
        return ca_path

    if required:
        print("\nError: CA path is required for local token generation.")
        print("The CA path should point to the provisioned workspace directory")
        print("(e.g., /path/to/workspace/my_project) created by 'nvflare provision'.")
        print(f"\nProvide via -c/--ca_path or set {ENV_CA_PATH} environment variable.")
        print("\nAlternatively, use --cert-service for remote token generation via Certificate Service.")
        sys.exit(1)

    return None


def _get_cert_service_url(args):
    """Resolve Certificate Service URL from args or environment variable."""
    url = getattr(args, "cert_service", None)
    if url:
        return url.rstrip("/")

    url = os.environ.get(ENV_CERT_SERVICE_URL)
    if url:
        return url.rstrip("/")

    return None


def _get_api_key(args):
    """Resolve API key from args or environment variable."""
    token = getattr(args, "api_key", None)
    if token:
        return token

    token = os.environ.get(ENV_API_KEY)
    if token:
        return token

    return None


def _generate_token_remote(cert_service_url: str, api_key: Optional[str], request_data: dict) -> str:
    """Generate token via Certificate Service API."""
    try:
        import requests
    except ImportError:
        print("\nError: requests library required for remote token generation.")
        print("Install with: pip install requests")
        sys.exit(1)

    if not api_key:
        print("\nError: API key required for remote token generation.")
        print(f"Provide via --api-key or set {ENV_API_KEY} environment variable.")
        sys.exit(1)

    url = f"{cert_service_url}/api/v1/token"
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        response = requests.post(url, json=request_data, headers=headers, timeout=30)

        if response.status_code == 401:
            print("\nError: Authentication failed. Check your API key.")
            sys.exit(1)
        elif response.status_code == 403:
            print("\nError: Access denied. Admin privileges required.")
            sys.exit(1)
        elif response.status_code != 200:
            error = response.json().get("error", response.text)
            print(f"\nError from Certificate Service: {error}")
            sys.exit(1)

        result = response.json()
        return result.get("token")

    except requests.RequestException as e:
        print(f"\nError connecting to Certificate Service: {e}")
        sys.exit(1)


def _parse_validity_to_days(validity_str: Optional[str]) -> Optional[int]:
    """Parse validity string (e.g., '7d', '24h') to days.

    Args:
        validity_str: Duration string (e.g., '7d', '24h', '1w')

    Returns:
        Number of days (rounded up for sub-day durations)
    """
    if not validity_str:
        return None

    validity_str = validity_str.strip().lower()
    if validity_str.endswith("d"):
        return int(validity_str[:-1])
    elif validity_str.endswith("w"):
        return int(validity_str[:-1]) * 7
    elif validity_str.endswith("h"):
        hours = int(validity_str[:-1])
        return max(1, (hours + 23) // 24)  # Round up to at least 1 day
    elif validity_str.endswith("m"):
        minutes = int(validity_str[:-1])
        return max(1, (minutes + 1439) // 1440)  # Round up to at least 1 day
    else:
        # Assume days if no unit
        return int(validity_str)


def _get_policy_path(args):
    """Resolve policy path from args, environment variable, or use built-in default."""
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
# Subcommand Parsers
# =============================================================================


def _define_generate_parser(sub_parser):
    """Define parser for 'nvflare token generate' command."""
    parser = sub_parser.add_parser(
        SUBCMD_GENERATE,
        help="Generate a single enrollment token",
        description=(
            "Generate a policy-based enrollment token (JWT) for a single subject.\n\n"
            "Environment variables:\n"
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

    # Remote token generation (Certificate Service API)
    parser.add_argument(
        "--cert-service",
        type=str,
        default=None,
        metavar="URL",
        help=f"Certificate Service URL for remote token generation (or set {ENV_CERT_SERVICE_URL})",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        metavar="KEY",
        help=f"Certificate Service API key (or set {ENV_API_KEY})",
    )


def _define_batch_parser(sub_parser):
    """Define parser for 'nvflare token batch' command."""
    parser = sub_parser.add_parser(
        SUBCMD_BATCH,
        help="Generate multiple enrollment tokens",
        description=(
            "Generate multiple policy-based enrollment tokens in batch.\n\n"
            "Environment variables:\n"
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

    # Remote token generation (Certificate Service API)
    parser.add_argument(
        "--cert-service",
        type=str,
        default=None,
        metavar="URL",
        help=f"Certificate Service URL for remote token generation (or set {ENV_CERT_SERVICE_URL})",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        metavar="KEY",
        help=f"Certificate Service API key (or set {ENV_API_KEY})",
    )


def _define_info_parser(sub_parser):
    """Define parser for 'nvflare token info' command."""
    parser = sub_parser.add_parser(
        SUBCMD_INFO,
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


# =============================================================================
# Main Parser
# =============================================================================


def def_token_parser(sub_cmd):
    """Define the token command parser.

    Args:
        sub_cmd: Parent subparser to add token command to

    Returns:
        Dict mapping command name to parser
    """
    cmd = CMD_TOKEN
    parser = sub_cmd.add_parser(
        cmd,
        help="Generate and manage enrollment tokens",
        description=(
            "Commands for generating and inspecting enrollment tokens (JWT).\n\n"
            "Environment variables:\n"
            f"  {ENV_CA_PATH}: CA directory path (local generation)\n"
            f"  {ENV_ENROLLMENT_POLICY}: Policy file path (local generation)\n"
            f"  {ENV_CERT_SERVICE_URL}: Certificate Service URL (remote generation)\n"
            f"  {ENV_API_KEY}: API key (remote generation)"
        ),
        formatter_class=lambda prog: __import__("argparse").RawDescriptionHelpFormatter(prog, max_help_position=40),
    )

    token_sub = parser.add_subparsers(
        title="token commands",
        dest="token_sub_cmd",
        help="Token subcommand",
    )

    _define_generate_parser(token_sub)
    _define_batch_parser(token_sub)
    _define_info_parser(token_sub)

    return {cmd: parser}


# =============================================================================
# Command Handlers
# =============================================================================


def _handle_generate_cmd(args):
    """Handle 'nvflare token generate' command."""
    _check_jwt_dependency()

    from nvflare.lighter.constants import ParticipantType
    from nvflare.tool.enrollment.token_service import TokenService

    type_map = {
        "client": ParticipantType.CLIENT,
        "admin": ParticipantType.ADMIN,
        "relay": ParticipantType.RELAY,
        "pattern": TokenService.SUBJECT_TYPE_PATTERN,
    }

    # Check for remote generation first
    cert_service_url = _get_cert_service_url(args)
    if cert_service_url:
        # Remote token generation via Certificate Service API
        api_key = _get_api_key(args)

        request_data = {
            "name": args.subject,
            "entity_type": type_map[args.type],
        }
        # Parse validity string to days for API (e.g., "7d" -> 7)
        if args.validity:
            validity_days = _parse_validity_to_days(args.validity)
            if validity_days:
                request_data["valid_days"] = validity_days

        if args.roles:
            request_data["role"] = args.roles[0] if len(args.roles) == 1 else args.roles
        if args.source_ips:
            request_data["source_ips"] = args.source_ips

        token = _generate_token_remote(cert_service_url, api_key, request_data)

        if args.output:
            with open(args.output, "w") as f:
                f.write(token)
            print(f"Token saved to: {args.output}")
        else:
            print("\nGenerated Enrollment Token (via Certificate Service):")
            print("-" * 60)
            print(token)
            print("-" * 60)

            # Decode token info locally
            import jwt

            payload = jwt.decode(token, options={"verify_signature": False})
            print("\nToken Info:")
            print(f"  Subject: {payload.get('sub')}")
            print(f"  Type: {payload.get('subject_type')}")
            print(f"  Expires: {payload.get('exp')}")

        return

    # Local token generation (requires CA path)
    ca_path = _get_ca_path(args, required=True)
    policy_path, is_temp_policy = _get_policy_path(args)

    try:
        service = TokenService(ca_path)

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
        if is_temp_policy and os.path.exists(policy_path):
            os.remove(policy_path)


def _handle_batch_cmd(args):
    """Handle 'nvflare token batch' command."""
    _check_jwt_dependency()

    from nvflare.lighter.constants import ParticipantType
    from nvflare.tool.enrollment.token_service import TokenService

    type_map = {
        "client": ParticipantType.CLIENT,
        "admin": ParticipantType.ADMIN,
        "relay": ParticipantType.RELAY,
    }

    # Generate name list
    if args.names:
        names = args.names
    else:
        names = [f"{args.prefix}-{i:03d}" for i in range(1, args.count + 1)]

    # Check for remote generation first
    cert_service_url = _get_cert_service_url(args)
    if cert_service_url:
        # Remote batch generation via Certificate Service API
        api_key = _get_api_key(args)

        # Parse validity string to days for API
        valid_days = _parse_validity_to_days(args.validity) if args.validity else 7

        # Use batch endpoint if available
        request_data = {
            "names": names,
            "entity_type": type_map[args.type],
            "valid_days": valid_days,
        }

        try:
            import requests
        except ImportError:
            print("\nError: requests library required for remote token generation.")
            print("Install with: pip install requests")
            sys.exit(1)

        try:
            url = f"{cert_service_url}/api/v1/token"
            headers = {"Authorization": f"Bearer {api_key}"}

            response = requests.post(url, json=request_data, headers=headers, timeout=60)

            if response.status_code == 401:
                print("\nError: Authentication failed. Check your API key.")
                sys.exit(1)
            elif response.status_code == 403:
                print("\nError: Access denied. Admin privileges required.")
                sys.exit(1)
            elif response.status_code != 200:
                error = response.json().get("error", response.text)
                print(f"\nError from Certificate Service: {error}")
                sys.exit(1)

            results = response.json().get("tokens", [])

        except requests.RequestException as e:
            print(f"\nError connecting to Certificate Service: {e}")
            sys.exit(1)

        # Save to output file
        with open(args.output, "w") as f:
            for item in results:
                f.write(f"{item['name']},{item['token']}\n")

        print(f"\nGenerated {len(results)} tokens (via Certificate Service)")
        print(f"Saved to: {args.output}")

        print("\nPreview (first 3):")
        for item in results[:3]:
            print(f"  {item['name']}: {item['token'][:50]}...")

        return

    # Local batch generation (requires CA path)
    ca_path = _get_ca_path(args, required=True)
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

        print("\nPreview (first 3):")
        for item in results[:3]:
            print(f"  {item['name']}: {item['token'][:50]}...")

    except Exception as e:
        print(f"\nError generating tokens: {e}")
        sys.exit(1)
    finally:
        if is_temp_policy and os.path.exists(policy_path):
            os.remove(policy_path)


def _handle_info_cmd(args):
    """Handle 'nvflare token info' command."""
    _check_jwt_dependency()

    import json

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
            "roles": payload.get("roles"),
            "source_ips": payload.get("source_ips"),
            "policy_project": payload.get("policy", {}).get("metadata", {}).get("project"),
            "policy_version": payload.get("policy", {}).get("metadata", {}).get("version"),
        }

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
    """Handle 'nvflare token' command and its subcommands."""
    token_sub = getattr(args, "token_sub_cmd", None)

    if token_sub == SUBCMD_GENERATE:
        _handle_generate_cmd(args)
    elif token_sub == SUBCMD_BATCH:
        _handle_batch_cmd(args)
    elif token_sub == SUBCMD_INFO:
        _handle_info_cmd(args)
    else:
        from nvflare.cli_unknown_cmd_exception import CLIUnknownCmdException

        raise CLIUnknownCmdException(
            "\nPlease specify a subcommand: generate, batch, or info\n" "Use 'nvflare token -h' for help."
        )
