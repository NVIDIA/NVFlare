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

"""NVFLARE Enrollment CLI - Manage pending enrollment requests.

This module provides CLI commands to interact with the Certificate Service
for managing pending enrollment requests in the Auto-Scale workflow.

Commands:
    nvflare enrollment list       - List pending enrollment requests
    nvflare enrollment info       - View details of a pending request
    nvflare enrollment approve    - Approve pending requests
    nvflare enrollment reject     - Reject pending requests
    nvflare enrollment enrolled   - List enrolled entities

Environment Variables:
    NVFLARE_CERT_SERVICE_URL  - Certificate Service URL (required)
    NVFLARE_API_KEY           - Certificate Service API key (required)
"""

import argparse
import os
import sys
from typing import Optional

# Command and subcommand names
CMD_ENROLLMENT = "enrollment"
SUBCMD_LIST = "list"
SUBCMD_INFO = "info"
SUBCMD_APPROVE = "approve"
SUBCMD_REJECT = "reject"
SUBCMD_ENROLLED = "enrolled"

# Environment variables
ENV_CERT_SERVICE_URL = "NVFLARE_CERT_SERVICE_URL"
ENV_API_KEY = "NVFLARE_API_KEY"


def _get_cert_service_url(args: argparse.Namespace) -> str:
    """Get Certificate Service URL from args or environment."""
    url = getattr(args, "cert_service", None) or os.environ.get(ENV_CERT_SERVICE_URL)
    if not url:
        print(f"Error: Certificate Service URL required. Use --cert-service or set {ENV_CERT_SERVICE_URL}")
        sys.exit(1)
    return url.rstrip("/")


def _get_api_key(args: argparse.Namespace) -> str:
    """Get API key from args or environment."""
    token = getattr(args, "api_key", None) or os.environ.get(ENV_API_KEY)
    if not token:
        print(f"Error: API key required. Use --api-key or set {ENV_API_KEY}")
        sys.exit(1)
    return token


def _make_request(url: str, token: str, method: str = "GET", json_data: dict = None) -> dict:
    """Make HTTP request to Certificate Service."""
    try:
        import requests
    except ImportError:
        print("Error: requests library required. Install with: pip install requests")
        sys.exit(1)

    headers = {"Authorization": f"Bearer {token}"}

    try:
        if method == "GET":
            response = requests.get(url, headers=headers, timeout=30)
        elif method == "POST":
            response = requests.post(url, headers=headers, json=json_data, timeout=30)
        else:
            raise ValueError(f"Unsupported method: {method}")

        if response.status_code == 401:
            print("Error: Authentication failed. Check your API key.")
            sys.exit(1)
        elif response.status_code == 403:
            print("Error: Access denied. Admin privileges required.")
            sys.exit(1)
        elif response.status_code == 404:
            return {"error": "Not found"}
        elif response.status_code != 200:
            error = response.json().get("error", response.text)
            print(f"Error: {error}")
            sys.exit(1)

        return response.json()

    except requests.RequestException as e:
        print(f"Error: Failed to connect to Certificate Service: {e}")
        sys.exit(1)


def _handle_list(args: argparse.Namespace) -> None:
    """Handle 'enrollment list' command."""
    url = _get_cert_service_url(args)
    token = _get_api_key(args)

    endpoint = f"{url}/api/v1/pending"
    if args.entity_type:
        endpoint += f"?type={args.entity_type}"

    result = _make_request(endpoint, token)
    pending = result.get("pending", [])

    if not pending:
        print("No pending enrollment requests.")
        return

    # Print header
    print(f"{'Name':<20} {'Type':<10} {'Org':<15} {'Submitted':<20} {'Status':<10}")
    print("-" * 80)

    for item in pending:
        print(
            f"{item.get('name', 'N/A'):<20} "
            f"{item.get('type', 'N/A'):<10} "
            f"{item.get('org', 'N/A'):<15} "
            f"{item.get('submitted', 'N/A'):<20} "
            f"{item.get('status', 'pending'):<10}"
        )


def _handle_info(args: argparse.Namespace) -> None:
    """Handle 'enrollment info' command."""
    url = _get_cert_service_url(args)
    token = _get_api_key(args)

    if not args.name:
        print("Error: Name is required for info command.")
        sys.exit(1)

    endpoint = f"{url}/api/v1/pending/{args.name}"
    if args.entity_type:
        endpoint += f"?type={args.entity_type}"

    result = _make_request(endpoint, token)

    if result.get("error"):
        print(f"Not found: {args.name}")
        return

    # Print detailed info
    print(f"Name:            {result.get('name', 'N/A')}")
    print(f"Type:            {result.get('type', 'N/A')}")
    print(f"Organization:    {result.get('org', 'N/A')}")
    print(f"Status:          {result.get('status', 'pending')}")
    print(f"Submitted:       {result.get('submitted', 'N/A')}")
    print(f"Token Subject:   {result.get('token_subject', 'N/A')}")
    if result.get("source_ip"):
        print(f"Source IP:       {result.get('source_ip')}")
    if result.get("csr_subject"):
        print(f"CSR Subject:     {result.get('csr_subject')}")
    if result.get("role"):
        print(f"Role:            {result.get('role')}")


def _handle_approve(args: argparse.Namespace) -> None:
    """Handle 'enrollment approve' command."""
    url = _get_cert_service_url(args)
    token = _get_api_key(args)

    if args.pattern:
        # Bulk approve by pattern
        endpoint = f"{url}/api/v1/pending/approve_batch"
        data = {"pattern": args.pattern, "type": args.entity_type}
    elif args.name:
        # Single approve
        endpoint = f"{url}/api/v1/pending/{args.name}/approve"
        data = {"type": args.entity_type}
    else:
        print("Error: Either --name or --pattern is required.")
        sys.exit(1)

    result = _make_request(endpoint, token, method="POST", json_data=data)

    approved_count = result.get("approved", 0)
    if isinstance(approved_count, list):
        approved_count = len(approved_count)

    print(f"Approved {approved_count} enrollment request(s).")


def _handle_reject(args: argparse.Namespace) -> None:
    """Handle 'enrollment reject' command."""
    url = _get_cert_service_url(args)
    token = _get_api_key(args)

    if not args.name:
        print("Error: Name is required for reject command.")
        sys.exit(1)

    endpoint = f"{url}/api/v1/pending/{args.name}/reject"
    data = {
        "type": args.entity_type,
        "reason": args.reason or "Rejected by admin",
    }

    _make_request(endpoint, token, method="POST", json_data=data)

    print(f"Rejected enrollment request for: {args.name}")


def _handle_enrolled(args: argparse.Namespace) -> None:
    """Handle 'enrollment enrolled' command."""
    url = _get_cert_service_url(args)
    token = _get_api_key(args)

    endpoint = f"{url}/api/v1/enrolled"
    if args.entity_type:
        endpoint += f"?type={args.entity_type}"

    result = _make_request(endpoint, token)
    enrolled = result.get("enrolled", [])

    if not enrolled:
        print("No enrolled entities.")
        return

    # Print header
    print(f"{'Name':<25} {'Type':<10} {'Org':<15} {'Enrolled At':<20}")
    print("-" * 75)

    for item in enrolled:
        print(
            f"{item.get('name', 'N/A'):<25} "
            f"{item.get('type', 'N/A'):<10} "
            f"{item.get('org', 'N/A'):<15} "
            f"{item.get('enrolled_at', 'N/A'):<20}"
        )


def define_enrollment_parser(sub_cmd) -> dict:
    """Define enrollment subcommand parser."""
    enrollment_parser = sub_cmd.add_parser(
        CMD_ENROLLMENT,
        help="Manage pending enrollment requests",
        description="Manage pending enrollment requests on the Certificate Service.",
    )

    enrollment_sub = enrollment_parser.add_subparsers(dest="enrollment_cmd", help="Enrollment commands")

    # Common arguments
    def add_common_args(parser):
        parser.add_argument(
            "--cert-service",
            metavar="URL",
            help=f"Certificate Service URL (or set {ENV_CERT_SERVICE_URL})",
        )
        parser.add_argument(
            "--api-key",
            metavar="TOKEN",
            help=f"Certificate Service API key (or set {ENV_API_KEY})",
        )
        parser.add_argument(
            "--type",
            dest="entity_type",
            choices=["client", "server", "relay", "user"],
            help="Filter by entity type",
        )

    # list subcommand
    list_parser = enrollment_sub.add_parser(
        SUBCMD_LIST,
        help="List pending enrollment requests",
    )
    add_common_args(list_parser)

    # info subcommand
    info_parser = enrollment_sub.add_parser(
        SUBCMD_INFO,
        help="View details of a pending request",
    )
    info_parser.add_argument("name", help="Name of the entity")
    add_common_args(info_parser)

    # approve subcommand
    approve_parser = enrollment_sub.add_parser(
        SUBCMD_APPROVE,
        help="Approve pending enrollment requests",
    )
    approve_parser.add_argument("name", nargs="?", help="Name of the entity to approve")
    approve_parser.add_argument("--pattern", help="Approve all matching pattern (e.g., 'hospital-*')")
    add_common_args(approve_parser)

    # reject subcommand
    reject_parser = enrollment_sub.add_parser(
        SUBCMD_REJECT,
        help="Reject pending enrollment request",
    )
    reject_parser.add_argument("name", help="Name of the entity to reject")
    reject_parser.add_argument("--reason", help="Rejection reason")
    add_common_args(reject_parser)

    # enrolled subcommand
    enrolled_parser = enrollment_sub.add_parser(
        SUBCMD_ENROLLED,
        help="List enrolled entities",
    )
    add_common_args(enrolled_parser)

    return {CMD_ENROLLMENT: enrollment_parser}


def handle_enrollment_cmd(args: argparse.Namespace) -> Optional[str]:
    """Handle enrollment commands."""
    cmd = getattr(args, "enrollment_cmd", None)

    if cmd == SUBCMD_LIST:
        _handle_list(args)
    elif cmd == SUBCMD_INFO:
        _handle_info(args)
    elif cmd == SUBCMD_APPROVE:
        _handle_approve(args)
    elif cmd == SUBCMD_REJECT:
        _handle_reject(args)
    elif cmd == SUBCMD_ENROLLED:
        _handle_enrolled(args)
    else:
        print("Error: No subcommand specified. Use --help for usage.")
        sys.exit(1)

    return None
