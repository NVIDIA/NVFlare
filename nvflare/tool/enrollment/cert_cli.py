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
- Server certificates (server)
- API keys for Certificate Service (api-key)

These are used in conjunction with `nvflare package` and `nvflare token`
for token-based enrollment workflows.
"""

import json
import os
import secrets

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

    # nvflare cert server
    server_parser = subparsers.add_parser("server", help="Generate server certificate")
    server_parser.add_argument("-n", "--name", type=str, required=True, help="Server name (used as CN and identity)")
    server_parser.add_argument(
        "-c",
        "--ca_path",
        type=str,
        required=True,
        help="Path to CA directory (containing rootCA.pem, rootCA.key, or state/cert.json)",
    )
    server_parser.add_argument(
        "-o", "--output", type=str, default=".", help="Output directory (default: current directory)"
    )
    server_parser.add_argument("--org", type=str, default="org", help="Organization name (default: org)")
    server_parser.add_argument(
        "--host", type=str, default=None, help="Default host name (default: same as server name)"
    )
    server_parser.add_argument(
        "--additional_hosts", type=str, nargs="*", default=None, help="Additional host names for SAN extension"
    )
    server_parser.add_argument(
        "--valid_days", type=int, default=365, help="Certificate validity in days (default: 365)"
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
        print("Error: Please specify a subcommand: init, server, or api-key")
        print("Usage: nvflare cert {init|server|api-key} [options]")
        return 1

    if args.cert_cmd == "init":
        return _handle_init(args)
    elif args.cert_cmd == "server":
        return _handle_server(args)
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
        print(f"  1. Generate server certificate: nvflare cert server -n <server_name> -c {output_dir}")
        print(f"  2. Generate tokens: nvflare token generate -s <site_name> -c {output_dir}")
        return 0

    except Exception as e:
        print(f"Error initializing root CA: {e}")
        return 1


def _handle_server(args):
    """Handle the 'server' subcommand - generate server certificate."""
    ca_path = os.path.abspath(args.ca_path)
    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    # Truncate server name if > 63 chars (CN limit)
    server_name = args.name
    if len(server_name) > 63:
        server_name = server_name[:63]
        print(f"Warning: Server name truncated to 63 chars: {server_name}")

    # Default host is server name if not specified
    default_host = args.host or args.name

    try:
        # Load root CA
        root_cert_pem, root_key_pem, issuer = _load_root_ca(ca_path)

        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives.serialization import load_pem_private_key

        root_pri_key = load_pem_private_key(root_key_pem, password=None, backend=default_backend())

        # Generate server key pair
        server_pri_key, server_pub_key = generate_keys()

        # Generate server certificate with SAN
        server_cert = generate_cert(
            subject=Identity(server_name, args.org),
            issuer=Identity(issuer),
            signing_pri_key=root_pri_key,
            subject_pub_key=server_pub_key,
            valid_days=args.valid_days,
            ca=False,
            server_default_host=default_host,
            server_additional_hosts=args.additional_hosts,
        )

        # Write server.crt
        server_cert_path = os.path.join(output_dir, "server.crt")
        with open(server_cert_path, "wb") as f:
            f.write(serialize_cert(server_cert))

        # Write server.key
        server_key_path = os.path.join(output_dir, "server.key")
        with open(server_key_path, "wb") as f:
            f.write(serialize_pri_key(server_pri_key))

        # Copy rootCA.pem to output
        root_ca_output = os.path.join(output_dir, "rootCA.pem")
        with open(root_ca_output, "wb") as f:
            f.write(root_cert_pem)

        # Set restrictive permissions on private key
        os.chmod(server_key_path, 0o600)

        print(f"Server certificate generated successfully in: {output_dir}")
        print("  - server.crt: Server certificate")
        print("  - server.key: Server private key (keep secure!)")
        print("  - rootCA.pem: Root CA certificate (for TLS verification)")
        print(f"\nServer name: {server_name}")
        print(f"Organization: {args.org}")
        print(f"Default host: {default_host}")
        if args.additional_hosts:
            print(f"Additional hosts: {', '.join(args.additional_hosts)}")
        print(f"Validity: {args.valid_days} days")
        print("\nNext steps:")
        print("  1. Copy these 3 files to your server's startup/ directory")
        print(
            f"  2. Generate server startup kit: nvflare package -n {args.name} -e grpc://{default_host}:8002 -t server"
        )
        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Error generating server certificate: {e}")
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
            f"Root CA private key not found: {root_key_path}\n"
            "The private key is required to sign server certificates."
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
        with open(output_path, "w") as f:
            f.write(api_key)
        # Set restrictive permissions
        os.chmod(output_path, 0o600)
        print(f"API key saved to: {output_path}")
        print(f"Key length: {length} bytes ({length * 8} bits)")
        print(f"Format: {args.format}")
    else:
        print(api_key)

    # Show usage instructions
    print("\n--- Usage Instructions ---")
    print("\n1. Set as environment variable:")
    print(f"   export NVFLARE_API_KEY='{api_key}'")
    print("\n2. Or add to Certificate Service config (cert_service_config.yaml):")
    print(f'   api_key: "{api_key}"')
    print("\n3. Use with CLI commands:")
    print(f"   nvflare token generate -n site-1 --cert-service https://... --api-key '{api_key}'")
    print(f"   nvflare enrollment list --cert-service https://... --api-key '{api_key}'")

    return 0
