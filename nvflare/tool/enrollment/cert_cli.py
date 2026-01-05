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

"""CLI for certificate generation.

This module provides the `nvflare cert` command for generating:
- Root CA certificates (init)
- Site certificates for any participant type (site)
- API keys for Certificate Service (api-key)

These are used in conjunction with `nvflare package` and `nvflare token`
for token-based enrollment workflows.

Manual Workflow:
  1. nvflare cert init -o ./ca                    # Create root CA
  2. nvflare cert site -n server1 -t server -c ./ca -o ./certs
  3. nvflare cert site -n client1 -t client -c ./ca -o ./certs
  4. nvflare cert site -n admin@org.com -t admin -c ./ca -o ./certs
  5. nvflare package -n client1 -e grpc://server1:8002 -t client
"""

import json
import os
import secrets

from nvflare.lighter.constants import AdminRole, ParticipantType
from nvflare.lighter.utils import Identity, generate_cert, generate_keys, serialize_cert, serialize_pri_key


def define_cert_parser(parser):
    """Define CLI arguments for the cert command."""
    subparsers = parser.add_subparsers(dest="cert_cmd", help="Certificate commands")

    # nvflare cert init
    init_parser = subparsers.add_parser("init", help="Initialize root CA")
    init_parser.add_argument("-n", "--name", type=str, default="NVFlare", help="Project/CA name (default: NVFlare)")
    init_parser.add_argument(
        "-o", "--output", type=str, default=".", help="Output directory (default: current directory)"
    )
    init_parser.add_argument(
        "--valid_days", type=int, default=3650, help="Certificate validity in days (default: 3650 = 10 years)"
    )

    # nvflare cert site - generic certificate generation for any participant type
    site_parser = subparsers.add_parser("site", help="Generate certificate for any site (server/client/relay/admin)")
    site_parser.add_argument("-n", "--name", type=str, required=True, help="Site name (used as CN)")
    site_parser.add_argument(
        "-t",
        "--type",
        type=str,
        choices=[ParticipantType.SERVER, ParticipantType.CLIENT, ParticipantType.RELAY, ParticipantType.ADMIN],
        default=ParticipantType.CLIENT,
        help="Participant type (default: client)",
    )
    site_parser.add_argument(
        "-c",
        "--ca_path",
        type=str,
        required=True,
        help="Path to CA directory (containing rootCA.pem, rootCA.key, or state/cert.json)",
    )
    site_parser.add_argument(
        "-o", "--output", type=str, default=".", help="Output directory (default: current directory)"
    )
    site_parser.add_argument("--org", type=str, default="org", help="Organization name (default: org)")
    site_parser.add_argument("--valid_days", type=int, default=365, help="Certificate validity in days (default: 365)")
    # Server-specific options
    site_parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Primary host for server SAN extension. Can be DNS name (server.example.com) or IP address (192.168.1.10). Default: same as --name",
    )
    site_parser.add_argument(
        "--additional_hosts",
        type=str,
        nargs="*",
        default=None,
        help="Additional hosts for server SAN. Can be DNS names or IP addresses (e.g., --additional_hosts 10.0.0.1 server.local)",
    )
    # Admin-specific options
    site_parser.add_argument(
        "--role",
        type=str,
        choices=[AdminRole.PROJECT_ADMIN, AdminRole.ORG_ADMIN, AdminRole.LEAD, AdminRole.MEMBER],
        default=AdminRole.LEAD,
        help="Role for admin type (default: lead, has job submission permissions)",
    )

    # nvflare cert api-key
    apikey_parser = subparsers.add_parser("api-key", help="Generate API key for Certificate Service")
    apikey_parser.add_argument(
        "-l", "--length", type=int, default=32, help="Key length in bytes (default: 32 = 256 bits)"
    )
    apikey_parser.add_argument(
        "-o", "--output", type=str, default=None, help="Output file path (default: print to stdout)"
    )
    apikey_parser.add_argument(
        "--format",
        type=str,
        choices=["hex", "base64", "urlsafe"],
        default="hex",
        help="Output format: hex (default), base64, or urlsafe (base64 URL-safe)",
    )


def handle_cert(args):
    """Handle the cert command."""
    if not hasattr(args, "cert_cmd") or args.cert_cmd is None:
        print("Error: Please specify a subcommand: init, site, or api-key")
        print("Usage: nvflare cert {init|site|api-key} [options]")
        return 1

    if args.cert_cmd == "init":
        return _handle_init(args)
    elif args.cert_cmd == "site":
        return _handle_site(args)
    elif args.cert_cmd == "server":
        # Backward compatibility - redirect to site with type=server
        args.type = ParticipantType.SERVER
        return _handle_site(args)
    elif args.cert_cmd == "api-key":
        return _handle_api_key(args)
    else:
        print(f"Error: Unknown subcommand: {args.cert_cmd}")
        return 1


def _handle_init(args):
    """Handle the 'init' subcommand - generate root CA."""
    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    # Create state directory for cert.json
    state_dir = os.path.join(output_dir, "state")
    os.makedirs(state_dir, exist_ok=True)

    try:
        # Generate root CA
        pri_key, pub_key = generate_keys()
        root_cert = generate_cert(
            subject=Identity(args.name),
            issuer=Identity(args.name),
            signing_pri_key=pri_key,
            subject_pub_key=pub_key,
            valid_days=args.valid_days,
            ca=True,
        )

        # Write rootCA.pem
        root_cert_path = os.path.join(output_dir, "rootCA.pem")
        with open(root_cert_path, "wb") as f:
            f.write(serialize_cert(root_cert))

        # Write rootCA.key
        root_key_path = os.path.join(output_dir, "rootCA.key")
        with open(root_key_path, "wb") as f:
            f.write(serialize_pri_key(pri_key))

        # Write state/cert.json (for compatibility with TokenService)
        cert_json_path = os.path.join(state_dir, "cert.json")
        cert_state = {
            "root_cert": serialize_cert(root_cert).decode("utf-8"),
            "root_pri_key": serialize_pri_key(pri_key).decode("utf-8"),
            "issuer": args.name,
        }
        with open(cert_json_path, "w") as f:
            json.dump(cert_state, f, indent=2)

        # Set restrictive permissions on private key
        os.chmod(root_key_path, 0o600)
        os.chmod(cert_json_path, 0o600)

        print(f"Root CA initialized successfully in: {output_dir}")
        print("  - rootCA.pem: Public certificate (distribute to all sites)")
        print("  - rootCA.key: Private key (keep secure!)")
        print("  - state/cert.json: State file for nvflare token command")
        print(f"\nValidity: {args.valid_days} days")
        print("\nNext steps:")
        print(f"  1. Generate server certificate: nvflare cert site -n <server> -t server -c {output_dir}")
        print(f"  2. Generate client certificate: nvflare cert site -n <client> -t client -c {output_dir}")
        print(f"  3. Generate tokens (if using auto-scale): nvflare token generate -s <site> -c {output_dir}")
        return 0

    except Exception as e:
        print(f"Error initializing root CA: {e}")
        return 1


def _handle_site(args):
    """Handle the 'site' subcommand - generate certificate for any participant type."""
    ca_path = os.path.abspath(args.ca_path)
    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    site_name = args.name
    site_type = args.type

    # Truncate name if > 63 chars (CN limit)
    if len(site_name) > 63:
        site_name = site_name[:63]
        print(f"Warning: Site name truncated to 63 chars: {site_name}")

    try:
        # Load root CA
        root_cert_pem, root_key_pem, issuer = _load_root_ca(ca_path)

        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives.serialization import load_pem_private_key

        root_pri_key = load_pem_private_key(root_key_pem, password=None, backend=default_backend())

        # Generate site key pair
        site_pri_key, site_pub_key = generate_keys()

        # Determine certificate filename prefix based on type
        if site_type == ParticipantType.SERVER:
            cert_prefix = "server"
        else:
            cert_prefix = "client"  # client, relay, admin all use client.crt/client.key

        # Build identity - include role for admin types
        if site_type == ParticipantType.ADMIN and hasattr(args, "role"):
            identity = Identity(site_name, args.org, args.role)
        else:
            identity = Identity(site_name, args.org)

        # Generate certificate - server gets SAN extensions
        if site_type == ParticipantType.SERVER:
            default_host = args.host or args.name
            site_cert = generate_cert(
                subject=identity,
                issuer=Identity(issuer),
                signing_pri_key=root_pri_key,
                subject_pub_key=site_pub_key,
                valid_days=args.valid_days,
                ca=False,
                server_default_host=default_host,
                server_additional_hosts=args.additional_hosts,
            )
        else:
            site_cert = generate_cert(
                subject=identity,
                issuer=Identity(issuer),
                signing_pri_key=root_pri_key,
                subject_pub_key=site_pub_key,
                valid_days=args.valid_days,
                ca=False,
            )

        # Write certificate
        cert_path = os.path.join(output_dir, f"{cert_prefix}.crt")
        with open(cert_path, "wb") as f:
            f.write(serialize_cert(site_cert))

        # Write private key
        key_path = os.path.join(output_dir, f"{cert_prefix}.key")
        with open(key_path, "wb") as f:
            f.write(serialize_pri_key(site_pri_key))

        # Copy rootCA.pem to output
        root_ca_output = os.path.join(output_dir, "rootCA.pem")
        with open(root_ca_output, "wb") as f:
            f.write(root_cert_pem)

        # Set restrictive permissions on private key
        os.chmod(key_path, 0o600)

        print(f"Certificate generated successfully in: {output_dir}")
        print(f"  - {cert_prefix}.crt: Site certificate")
        print(f"  - {cert_prefix}.key: Site private key (keep secure!)")
        print("  - rootCA.pem: Root CA certificate (for TLS verification)")
        print(f"\nSite name: {site_name}")
        print(f"Site type: {site_type}")
        print(f"Organization: {args.org}")
        if site_type == ParticipantType.SERVER and args.host:
            print(f"Default host: {args.host}")
            if args.additional_hosts:
                print(f"Additional hosts: {', '.join(args.additional_hosts)}")
        if site_type == ParticipantType.ADMIN:
            print(f"Role: {args.role}")
        print(f"Validity: {args.valid_days} days")
        print("\nNext steps:")
        print(f"  1. Generate startup kit: nvflare package -n {args.name} -e grpc://<server>:8002 -t {site_type}")
        print("  2. Copy these certificate files to the startup kit's startup/ directory")
        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Error generating certificate: {e}")
        return 1


def _load_root_ca(ca_path: str) -> tuple:
    """Load root CA certificate and private key.

    Args:
        ca_path: Path to CA directory

    Returns:
        Tuple of (root_cert_pem, root_key_pem, issuer_name)

    Raises:
        FileNotFoundError: If CA files not found
    """
    # Try state/cert.json first
    cert_json_path = os.path.join(ca_path, "state", "cert.json")
    if os.path.exists(cert_json_path):
        with open(cert_json_path, "r") as f:
            cert_state = json.load(f)
        root_cert_pem = cert_state["root_cert"].encode("utf-8")
        root_key_pem = cert_state["root_pri_key"].encode("utf-8")
        issuer = cert_state.get("issuer", "NVFlare")
        return root_cert_pem, root_key_pem, issuer

    # Fall back to rootCA.pem and rootCA.key
    root_cert_path = os.path.join(ca_path, "rootCA.pem")
    root_key_path = os.path.join(ca_path, "rootCA.key")

    if not os.path.exists(root_cert_path):
        raise FileNotFoundError(
            f"Root CA not found at: {ca_path}\n"
            "Expected: state/cert.json or rootCA.pem + rootCA.key\n"
            "Run 'nvflare cert init' first to create a root CA."
        )

    if not os.path.exists(root_key_path):
        raise FileNotFoundError(
            f"Root CA private key not found: {root_key_path}\n" "The private key is required to sign certificates."
        )

    with open(root_cert_path, "rb") as f:
        root_cert_pem = f.read()

    with open(root_key_path, "rb") as f:
        root_key_pem = f.read()

    # Extract issuer from cert (simplified - just use "NVFlare")
    issuer = "NVFlare"

    return root_cert_pem, root_key_pem, issuer


def _handle_api_key(args):
    """Handle the 'api-key' subcommand - generate API key for Certificate Service."""
    import base64

    length = args.length
    if length < 16:
        print("Warning: Key length less than 16 bytes (128 bits) is not recommended.")
    if length > 64:
        print("Warning: Key length greater than 64 bytes may be excessive.")

    # Generate cryptographically secure random bytes
    key_bytes = secrets.token_bytes(length)

    # Format the key
    if args.format == "hex":
        api_key = key_bytes.hex()
    elif args.format == "base64":
        api_key = base64.b64encode(key_bytes).decode("ascii")
    elif args.format == "urlsafe":
        api_key = base64.urlsafe_b64encode(key_bytes).decode("ascii").rstrip("=")
    else:
        api_key = key_bytes.hex()

    # Output
    if args.output:
        output_path = os.path.abspath(args.output)
        # Note: API key is intentionally stored in clear text, similar to SSH keys
        # or cloud credential files. Security is provided by file permissions.
        with open(output_path, "w") as f:
            f.write(api_key)  # nosec B105 - intentional clear-text storage with restricted permissions
        # Set restrictive permissions (owner read/write only)
        os.chmod(output_path, 0o600)
        print(f"API key saved to: {output_path}")
        print(f"Key length: {length} bytes ({length * 8} bits)")
        print(f"Format: {args.format}")
    else:
        # Intentional: command's purpose is to output the generated key
        print(api_key)  # noqa: T201  # nosec B105

    # Show usage instructions (intentionally includes key for copy-paste convenience)
    print("\n--- Usage Instructions ---")  # noqa: T201
    print("\n1. Set as environment variable:")  # noqa: T201
    print(f"   export NVFLARE_API_KEY='{api_key}'")  # noqa: T201  # nosec B105
    print("\n2. Or add to Certificate Service config (cert_service_config.yaml):")  # noqa: T201
    print(f'   api_key: "{api_key}"')  # noqa: T201  # nosec B105
    print("\n3. Use with CLI commands:")  # noqa: T201
    print(
        f"   nvflare token generate -n site-1 --cert-service https://... --api-key '{api_key}'"
    )  # noqa: T201  # nosec B105
    print(f"   nvflare enrollment list --cert-service https://... --api-key '{api_key}'")

    return 0
