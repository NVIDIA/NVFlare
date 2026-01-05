# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""Generate generic startup kit packages for token-based enrollment.

This module creates startup kits without certificates by reusing the existing
provisioner but filtering out CertBuilder and SignatureBuilder.

Two modes of operation:
1. With -p project_file: Package all participants from project.yml (without certs)
2. Without -p: Create a single participant package using CLI args
"""

import os
import pathlib
import re
import shutil

from nvflare.lighter.constants import ParticipantType, PropKey
from nvflare.lighter.prov_utils import prepare_builders
from nvflare.lighter.provision import prepare_project
from nvflare.lighter.provisioner import Provisioner
from nvflare.lighter.utils import load_yaml

# Builders to exclude for generic packages (no certificates)
EXCLUDED_BUILDERS = [
    "nvflare.lighter.impl.cert.CertBuilder",
    "nvflare.lighter.impl.signature.SignatureBuilder",
]


def define_package_parser(parser):
    """Define CLI arguments for the package command."""
    # Project file mode - packages all participants
    parser.add_argument(
        "-p",
        "--project_file",
        type=str,
        default=None,
        help="Project YAML file. If provided, packages ALL participants (no other args needed)",
    )
    parser.add_argument(
        "-w", "--workspace", type=str, default="workspace", help="Output workspace directory (default: workspace)"
    )

    # Single participant mode - when no project file
    parser.add_argument("-n", "--name", type=str, default=None, help="Participant name (required if no -p)")
    parser.add_argument(
        "-e", "--endpoint", type=str, default=None, help="Connection URI, e.g., grpc://server:8002 (required if no -p)"
    )
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        choices=["server", "client", "relay", "admin"],
        default="client",
        help="Package type (default: client)",
    )
    parser.add_argument("--org", type=str, default="org", help="Organization name (default: org)")
    parser.add_argument(
        "--role",
        type=str,
        choices=["lead", "member", "org_admin"],
        default="lead",
        help="Role for admin type (default: lead)",
    )
    parser.add_argument(
        "--listening_host", type=str, default="localhost", help="Listening host for relay type (default: localhost)"
    )
    parser.add_argument(
        "--listening_port", type=int, default=8002, help="Listening port for relay type (default: 8002)"
    )

    # Auto-Scale workflow options
    parser.add_argument(
        "--cert-service",
        type=str,
        default=None,
        help="Certificate Service URL for auto-enrollment (e.g., https://cert-service:8443)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Enrollment token (will be saved to startup/enrollment_token)",
    )


def parse_endpoint_uri(uri: str) -> dict:
    """Parse endpoint URI into components.

    Args:
        uri: Connection URI (e.g., "grpc://server:8002" or "grpc://server:8002:8003")

    Returns:
        Dict with keys: scheme, host, fl_port, admin_port
    """
    pattern = r"^(\w+)://([^:]+):(\d+)(?::(\d+))?$"
    match = re.match(pattern, uri)

    if not match:
        raise ValueError(
            f"Invalid endpoint URI: {uri}\n"
            "Expected format: scheme://host:port or scheme://host:fl_port:admin_port\n"
            "Examples: grpc://server:8002, http://server:8443:8444"
        )

    scheme = match.group(1).lower()
    host = match.group(2)
    fl_port = int(match.group(3))
    admin_port = int(match.group(4)) if match.group(4) else fl_port

    return {"scheme": scheme, "host": host, "fl_port": fl_port, "admin_port": admin_port}


def filter_builders(builders_config: list) -> list:
    """Filter out CertBuilder and SignatureBuilder from builders config.

    Args:
        builders_config: List of builder configurations

    Returns:
        Filtered list without cert/signature builders
    """
    return [b for b in builders_config if b.get("path") not in EXCLUDED_BUILDERS]


def load_default_project() -> dict:
    """Load the default project.yml from the lighter package.

    Returns:
        Project dictionary from dummy_project.yml
    """
    lighter_dir = pathlib.Path(__file__).parent.absolute()
    default_project = os.path.join(lighter_dir, "dummy_project.yml")
    return load_yaml(default_project)


def handle_package(args):
    """Handle the package command."""
    project_file = args.project_file
    workspace = args.workspace

    if project_file:
        # Mode 1: Package all participants from project.yml
        return handle_project_file_mode(project_file, workspace)
    else:
        # Mode 2: Create single participant package
        return handle_single_participant_mode(args, workspace)


def handle_project_file_mode(project_file: str, workspace: str) -> int:
    """Package all participants from a project.yml file.

    Args:
        project_file: Path to project.yml
        workspace: Output workspace directory

    Returns:
        Exit code (0 for success)
    """
    if not os.path.exists(project_file):
        print(f"Error: Project file not found: {project_file}")
        return 1

    try:
        result_dir = package_from_project(project_file, workspace)
        print(f"Packages generated successfully: {result_dir}")
        print("\nGenerated packages for all participants (without certificates).")
        print("\nNext steps:")
        print("1. Copy rootCA.pem from your provisioned server workspace to each startup/ folder")
        print("2. Distribute packages with enrollment tokens to each site")
        return 0
    except Exception as e:
        print(f"Error generating packages: {e}")
        return 1


def handle_single_participant_mode(args, workspace: str) -> int:
    """Create a single participant package from CLI args.

    Args:
        args: CLI arguments
        workspace: Output workspace directory

    Returns:
        Exit code (0 for success)
    """
    # Validate required args
    if not args.name:
        print("Error: -n/--name is required when not using -p/--project_file")
        return 1
    if not args.endpoint:
        print("Error: -e/--endpoint is required when not using -p/--project_file")
        return 1

    try:
        endpoint_info = parse_endpoint_uri(args.endpoint)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    # Get optional enrollment options
    cert_service_url = getattr(args, "cert_service", None)
    enrollment_token = getattr(args, "token", None)

    try:
        result_dir = generate_single_package(
            name=args.name,
            participant_type=args.type,
            endpoint_info=endpoint_info,
            workspace=workspace,
            org=args.org,
            role=args.role,
            listening_host=args.listening_host,
            listening_port=args.listening_port,
            cert_service_url=cert_service_url,
            enrollment_token=enrollment_token,
        )
        print(f"Package generated successfully: {result_dir}")

        # Show what was included
        if cert_service_url:
            print(f"\nCertificate Service URL embedded: {cert_service_url}")
        if enrollment_token:
            print("Enrollment token embedded: startup/enrollment_token")

        print("\nNext steps:")
        if args.type == "server":
            if cert_service_url and enrollment_token:
                # Auto-Scale workflow for server
                print("1. Start the server: cd {result_dir} && ./startup/start.sh")
                print("   (Server will auto-enroll with Certificate Service)")
            else:
                # Manual workflow for server
                print("1. Copy server certificates from project admin to startup/ folder:")
                print("   - rootCA.pem (root CA certificate)")
                print("   - server.crt (server certificate)")
                print("   - server.key (server private key)")
                print(f"2. Start the server: cd {result_dir} && ./startup/start.sh")
        else:
            if cert_service_url and enrollment_token:
                # Auto-Scale workflow - everything is embedded
                print(f"1. Start: cd {result_dir} && ./startup/start.sh")
                print("   (Client will auto-enroll with Certificate Service)")
            elif cert_service_url:
                # URL embedded, token needs to be provided
                print("1. Set enrollment token: export NVFLARE_ENROLLMENT_TOKEN=<your_token>")
                print(f"2. Start: cd {result_dir} && ./startup/start.sh")
            else:
                # No cert-service URL embedded - show both workflow options
                print("\nChoose one of the following workflows:\n")
                print("Option A: Manual Workflow (small scale, no Certificate Service)")
                print("  1. Obtain certificates from Project Admin:")
                print("     - rootCA.pem, client.crt, client.key")
                print("  2. Copy certificates to the startup/ folder")
                print(f"  3. Start: cd {result_dir} && ./startup/start.sh")
                print("")
                print("Option B: Auto-Scale Workflow (with Certificate Service)")
                print("  1. Set Certificate Service URL:")
                print("     export NVFLARE_CERT_SERVICE_URL=https://<cert-service>:8443")
                print("  2. Set enrollment token from Project Admin:")
                print("     export NVFLARE_ENROLLMENT_TOKEN=<your_token>")
                print(f"  3. Start: cd {result_dir} && ./startup/start.sh")
                print("     (Client will auto-enroll and obtain certificates)")
        return 0
    except Exception as e:
        print(f"Error generating package: {e}")
        return 1


def package_from_project(project_file: str, workspace: str) -> str:
    """Package all participants from a project.yml without certificates.

    This is like 'nvflare provision' but without CertBuilder and SignatureBuilder.

    Args:
        project_file: Path to project.yml file
        workspace: Output workspace directory

    Returns:
        Path to the generated workspace
    """
    # Load project
    project_dict = load_yaml(project_file)

    # Filter out CertBuilder and SignatureBuilder
    builders_config = project_dict.get("builders", [])
    project_dict["builders"] = filter_builders(builders_config)

    # Use existing provisioner logic (same as provision.py)
    project = prepare_project(project_dict)
    builders = prepare_builders(project_dict)

    workspace_path = os.path.abspath(workspace)
    provisioner = Provisioner(workspace_path, builders)
    ctx = provisioner.provision(project)

    return ctx.get_result_location()


def generate_single_package(
    name: str,
    participant_type: str,
    endpoint_info: dict,
    workspace: str,
    org: str = "org",
    role: str = "lead",
    listening_host: str = "localhost",
    listening_port: int = 8002,
    cert_service_url: str = None,
    enrollment_token: str = None,
) -> str:
    """Generate a single participant package without certificates.

    Args:
        name: Participant name
        participant_type: Type of participant (server, client, relay, admin)
        endpoint_info: Dict with scheme, host, fl_port, admin_port
        workspace: Output workspace directory
        org: Organization name
        role: Role for admin participants
        listening_host: Listening host for relay
        listening_port: Listening port for relay
        cert_service_url: Certificate Service URL for auto-enrollment (optional)
        enrollment_token: Enrollment token for auto-enrollment (optional)

    Returns:
        Path to the generated package directory
    """
    scheme = endpoint_info["scheme"]
    host = endpoint_info["host"]
    fl_port = endpoint_info["fl_port"]
    admin_port = endpoint_info["admin_port"]

    participants = []

    if participant_type == ParticipantType.SERVER:
        # For server type, the participant is the server itself
        server_def = {
            PropKey.NAME: name,
            PropKey.TYPE: ParticipantType.SERVER,
            PropKey.ORG: org,
            PropKey.FED_LEARN_PORT: fl_port,
            PropKey.ADMIN_PORT: admin_port,
            PropKey.DEFAULT_HOST: host,
        }
        participants.append(server_def)
    else:
        # For non-server types, we need a server reference + the participant
        server_def = {
            PropKey.NAME: host,
            PropKey.TYPE: ParticipantType.SERVER,
            PropKey.ORG: org,
            PropKey.FED_LEARN_PORT: fl_port,
            PropKey.ADMIN_PORT: admin_port,
        }
        participants.append(server_def)

        participant_def = {
            PropKey.NAME: name,
            PropKey.TYPE: participant_type,
            PropKey.ORG: org,
        }

        if participant_type == ParticipantType.ADMIN:
            participant_def[PropKey.ROLE] = role
        elif participant_type == ParticipantType.RELAY:
            participant_def[PropKey.LISTENING_HOST] = {
                "default_host": listening_host,
                "port": listening_port,
                "scheme": scheme,
            }

        participants.append(participant_def)

    # Build project with filtered builders
    builders_config = [
        {"path": "nvflare.lighter.impl.workspace.WorkspaceBuilder"},
        {"path": "nvflare.lighter.impl.static_file.StaticFileBuilder", "args": {"scheme": scheme}},
    ]

    project_dict = {
        PropKey.API_VERSION: 3,
        PropKey.NAME: "generic_package",
        PropKey.DESCRIPTION: "Generic package for token-based enrollment",
        "participants": participants,
        "builders": builders_config,
    }

    # Use existing provisioner logic
    project = prepare_project(project_dict)
    builders = prepare_builders(project_dict)

    workspace_path = os.path.abspath(workspace)
    temp_workspace = os.path.join(workspace_path, f".nvflare_package_{name}")

    try:
        provisioner = Provisioner(temp_workspace, builders)
        ctx = provisioner.provision(project)

        result_dir = ctx.get_result_location()
        if not result_dir:
            raise RuntimeError("Provisioning did not produce output")

        participant_dir = os.path.join(result_dir, name)
        if not os.path.exists(participant_dir):
            raise RuntimeError(f"Expected participant directory not found: {participant_dir}")

        # Move to final location
        output_dir = os.path.join(workspace_path, name)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        shutil.move(participant_dir, output_dir)

        # Create enrollment files if cert_service_url or token provided
        _create_enrollment_files(output_dir, cert_service_url, enrollment_token)

        return output_dir
    finally:
        if os.path.exists(temp_workspace):
            shutil.rmtree(temp_workspace)


def _create_enrollment_files(output_dir: str, cert_service_url: str = None, enrollment_token: str = None):
    """Create enrollment configuration files in the startup directory.

    Args:
        output_dir: Package output directory
        cert_service_url: Certificate Service URL (optional)
        enrollment_token: Enrollment token (optional)
    """
    import json

    startup_dir = os.path.join(output_dir, "startup")
    if not os.path.exists(startup_dir):
        os.makedirs(startup_dir)

    # Create enrollment.json with cert service URL
    if cert_service_url:
        enrollment_config = {"cert_service_url": cert_service_url}
        enrollment_json_path = os.path.join(startup_dir, "enrollment.json")
        with open(enrollment_json_path, "w") as f:
            json.dump(enrollment_config, f, indent=2)

    # Create enrollment_token file
    if enrollment_token:
        token_path = os.path.join(startup_dir, "enrollment_token")
        with open(token_path, "w") as f:
            f.write(enrollment_token)
        # Set restrictive permissions on token file
        os.chmod(token_path, 0o600)


# Keep for backward compatibility with existing tests
def generate_package(
    name: str,
    participant_type: str,
    endpoint_info: dict,
    output_dir: str,
    project_file: str = None,
    org: str = "org",
    role: str = "lead",
    listening_host: str = "localhost",
    listening_port: int = 8002,
) -> str:
    """Generate a generic startup kit package without certificates.

    Args:
        name: Participant name
        participant_type: Type of participant (client, relay, admin)
        endpoint_info: Dict with scheme, host, fl_port, admin_port
        output_dir: Output directory path
        project_file: Optional custom project.yml file path (ignored, for compatibility)
        org: Organization name
        role: Role for admin participants
        listening_host: Listening host for relay
        listening_port: Listening port for relay

    Returns:
        Path to the generated package directory
    """
    workspace = os.path.dirname(output_dir) or os.getcwd()
    result = generate_single_package(
        name=name,
        participant_type=participant_type,
        endpoint_info=endpoint_info,
        workspace=workspace,
        org=org,
        role=role,
        listening_host=listening_host,
        listening_port=listening_port,
    )
    # Move to expected output_dir if different
    if result != output_dir:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        shutil.move(result, output_dir)
        return output_dir
    return result
