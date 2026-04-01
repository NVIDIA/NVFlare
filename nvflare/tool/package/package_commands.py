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

"""nvflare package command handler."""

import datetime
import json
import os
import re
import shutil

import yaml

from nvflare.lighter.utils import load_crt, sh_replace, verify_cert
from nvflare.tool.cli_errors import get_error
from nvflare.tool.cli_output import output, output_error

_ENDPOINT_PATTERN = re.compile(r"^(grpc|tcp)://([^:/]+):(\d+)$")

_PACKAGE_EXAMPLES = [
    "nvflare package -t lead -e grpc://fl-server:8002 --dir ./alice",
    "nvflare package -t client -e grpc://fl-server:8002 --dir ./hospital-1",
    "nvflare package -n hospital-1 -t client -e grpc://fl-server:8002 --cert ./signed/client.crt --key ./csr/hospital-1.key --rootca ./signed/rootCA.pem",
]

_TEMPLATE_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "lighter", "templates", "master_template.yml")
)


def _load_templates() -> dict:
    """Load master_template.yml and return the parsed YAML dict."""
    with open(_TEMPLATE_PATH, "r") as f:
        return yaml.safe_load(f)


def _parse_endpoint(endpoint: str) -> tuple:
    """Return (scheme, host, port: int). Raises ValueError on invalid format."""
    m = _ENDPOINT_PATTERN.match(endpoint)
    if not m:
        raise ValueError(f"Invalid endpoint URI: {endpoint!r}")
    return m.group(1), m.group(2), int(m.group(3))


def _discover_name_from_dir(work_dir: str, fmt) -> str:
    """Find the single *.key file in work_dir and return its stem as the participant name."""
    keys = [f for f in os.listdir(work_dir) if f.endswith(".key")]
    if len(keys) == 0:
        output_error(
            "KEY_NOT_FOUND",
            f"No *.key file found in {work_dir}",
            "Run 'nvflare cert csr' to generate a key, or use --key explicitly.",
            fmt,
            exit_code=1,
        )
    if len(keys) > 1:
        output_error(
            "AMBIGUOUS_KEY",
            f"Multiple *.key files found in {work_dir}: {keys}",
            "Specify the participant name with -n, or use --key explicitly.",
            fmt,
            exit_code=1,
        )
    return keys[0][:-4]  # strip ".key"


def _write_file(path: str, content: str, mode_bits: int = None):
    """Write text content to path. Optionally chmod to mode_bits."""
    with open(path, "w") as f:
        f.write(content)
    if mode_bits is not None:
        os.chmod(path, mode_bits)


def _copy_file(src: str, dst: str, mode_bits: int = None):
    """Copy src to dst. Optionally chmod dst to mode_bits."""
    shutil.copy2(src, dst)
    if mode_bits is not None:
        os.chmod(dst, mode_bits)


def _make_fed_client_json(name, scheme, host, port, admin_port, server_name, project_name=None):
    # servers[0].name must be project_name to match the server's fed_server.json servers[0].name.
    proj = project_name or server_name
    sp_end_point = f"{host}:{port}:{admin_port}"
    return {
        "format_version": 2,
        "servers": [
            {
                "name": proj,
                "service": {
                    "scheme": scheme,
                },
                "identity": server_name,
            }
        ],
        "client": {
            "ssl_private_key": "client.key",
            "ssl_cert": "client.crt",
            "ssl_root_cert": "rootCA.pem",
            "fqsn": name,
            "is_leaf": True,
            "connection_security": "mtls",
        },
        "overseer_agent": {
            "path": "nvflare.ha.dummy_overseer_agent.DummyOverseerAgent",
            "args": {
                "sp_end_point": sp_end_point,
            },
        },
    }


def _make_fed_server_json(name, scheme, host, port, admin_port, require_signed_jobs, project_name=None):
    # servers[0].name must match project_name in fed_admin.json for challenge-response auth.
    proj = project_name or name
    sp_end_point = f"{name}:{port}:{admin_port}"
    return {
        "format_version": 2,
        "require_signed_jobs": require_signed_jobs,
        "servers": [
            {
                "name": proj,
                "service": {
                    "target": f"{host}:{port}",
                    "scheme": scheme,
                },
                "admin_server": name,
                "admin_port": admin_port,
                "ssl_private_key": "server.key",
                "ssl_cert": "server.crt",
                "ssl_root_cert": "rootCA.pem",
                "connection_security": "mtls",
            }
        ],
        "overseer_agent": {
            "path": "nvflare.ha.dummy_overseer_agent.DummyOverseerAgent",
            "args": {
                "sp_end_point": sp_end_point,
            },
        },
    }


def _make_fed_admin_json(name, server_name, host, admin_port, project_name=None):
    # project_name must match servers[0].name in fed_server.json for challenge-response auth.
    # Admin always connects over HTTP (not gRPC).
    proj = project_name or server_name
    return {
        "format_version": 1,
        "admin": {
            "project_name": proj,
            "username": name,
            "server_identity": server_name,
            "scheme": "http",
            "host": host,
            "port": admin_port,
            "connection_security": "mtls",
            "uid_source": "user_input",
            "with_file_transfer": True,
            "upload_dir": "transfer",
            "download_dir": "transfer",
            "client_key": "client.key",
            "client_cert": "client.crt",
            "ca_cert": "rootCA.pem",
        },
    }


def _build_client_kit(args, startup_dir, local_dir, templates, scheme, host, port):
    admin_port = args.admin_port if args.admin_port is not None else port + 1

    # 1. Copy certs
    _copy_file(args.cert, os.path.join(startup_dir, "client.crt"))
    _copy_file(args.key, os.path.join(startup_dir, "client.key"), mode_bits=0o600)
    _copy_file(args.rootca, os.path.join(startup_dir, "rootCA.pem"))

    # 2. fed_client.json — server identity is always the endpoint hostname
    fed_client = _make_fed_client_json(args.name, scheme, host, port, admin_port, host, args.project_name or host)
    _write_file(os.path.join(startup_dir, "fed_client.json"), json.dumps(fed_client, indent=2))

    # 3. Shell scripts
    _write_file(os.path.join(startup_dir, "start.sh"), templates["start_cln_sh"], mode_bits=0o755)
    sub_content = sh_replace(
        templates["sub_start_sh"],
        {
            "type": "client",
            "app_name": "client_train",
            "cln_uid": f"uid={args.name}",
            "org_name": args.name,
            "config_folder": "config",
        },
    )
    _write_file(os.path.join(startup_dir, "sub_start.sh"), sub_content, mode_bits=0o755)
    _write_file(os.path.join(startup_dir, "stop_fl.sh"), templates["stop_fl_sh"], mode_bits=0o755)

    # 4. local/ files
    client_res = sh_replace(templates["local_client_resources"], {"num_gpus": "0", "gpu_mem": "0"})
    _write_file(os.path.join(local_dir, "resources.json.default"), client_res)
    _write_file(os.path.join(local_dir, "log_config.json.default"), templates["log_config"])
    _write_file(os.path.join(local_dir, "privacy.json.sample"), templates["sample_privacy"])
    _write_file(os.path.join(local_dir, "authorization.json.default"), templates["default_authz"])


def _build_server_kit(args, startup_dir, local_dir, templates, scheme, host, port):
    admin_port = args.admin_port if args.admin_port is not None else port + 1
    require_signed = True  # signed jobs are always required

    # 1. Copy certs
    _copy_file(args.cert, os.path.join(startup_dir, "server.crt"))
    _copy_file(args.key, os.path.join(startup_dir, "server.key"), mode_bits=0o600)
    _copy_file(args.rootca, os.path.join(startup_dir, "rootCA.pem"))

    # 2. fed_server.json
    fed_server = _make_fed_server_json(
        args.name, scheme, host, port, admin_port, require_signed, args.project_name or args.name
    )
    _write_file(os.path.join(startup_dir, "fed_server.json"), json.dumps(fed_server, indent=2))

    # 3. local/ authorization template — provisioner puts this in local/ as .default, not startup/
    _write_file(os.path.join(local_dir, "authorization.json.default"), templates["default_authz"])

    # 4. Shell scripts
    start_content = sh_replace(templates["start_svr_sh"], {"ha_mode": "false"})
    _write_file(os.path.join(startup_dir, "start.sh"), start_content, mode_bits=0o755)
    sub_content = sh_replace(
        templates["sub_start_sh"],
        {
            "type": "server",
            "app_name": "server_train",
            "cln_uid": "",
            "org_name": args.name,
            "config_folder": "config",
        },
    )
    _write_file(os.path.join(startup_dir, "sub_start.sh"), sub_content, mode_bits=0o755)
    _write_file(os.path.join(startup_dir, "stop_fl.sh"), templates["stop_fl_sh"], mode_bits=0o755)

    # 5. local/ files
    _write_file(os.path.join(local_dir, "resources.json.default"), templates["local_server_resources"])
    _write_file(os.path.join(local_dir, "log_config.json.default"), templates["log_config"])
    _write_file(os.path.join(local_dir, "privacy.json.sample"), templates["sample_privacy"])


def _build_user_kit(args, startup_dir, local_dir, templates, scheme, host, port):
    admin_port = args.admin_port if args.admin_port is not None else (port + 1 if port else 8003)

    # 1. Copy certs (admin uses client.crt/client.key naming convention)
    _copy_file(args.cert, os.path.join(startup_dir, "client.crt"))
    _copy_file(args.key, os.path.join(startup_dir, "client.key"), mode_bits=0o600)
    _copy_file(args.rootca, os.path.join(startup_dir, "rootCA.pem"))

    # 2. fed_admin.json — server identity is always the endpoint hostname
    fed_admin = _make_fed_admin_json(args.name, host, host, admin_port, args.project_name or host)
    _write_file(os.path.join(startup_dir, "fed_admin.json"), json.dumps(fed_admin, indent=2))

    # 3. fl_admin.sh
    _write_file(os.path.join(startup_dir, "fl_admin.sh"), templates["fl_admin_sh"], mode_bits=0o755)

    # 4. local/ files
    _write_file(os.path.join(local_dir, "resources.json.default"), templates["default_admin_resources"])


def _build_kit(args, output_dir: str, scheme, host, port):
    templates = _load_templates()
    startup_dir = os.path.join(output_dir, "startup")
    local_dir = os.path.join(output_dir, "local")
    os.makedirs(startup_dir, exist_ok=True)
    os.makedirs(local_dir, exist_ok=True)

    if args.kit_type == "client":
        _build_client_kit(args, startup_dir, local_dir, templates, scheme, host, port)
    elif args.kit_type == "server":
        _build_server_kit(args, startup_dir, local_dir, templates, scheme, host, port)
    elif args.kit_type in ("org_admin", "lead", "member"):
        _build_user_kit(args, startup_dir, local_dir, templates, scheme, host, port)


def _build_result(args, output_dir: str, scheme, host, port):
    """Build the result dict for output."""
    if args.kit_type in ("org_admin", "lead", "member"):
        cert_filename = "client.crt"
        key_filename = "client.key"
        next_step = f"cd {args.name} && ./startup/fl_admin.sh"
    elif args.kit_type == "server":
        cert_filename = "server.crt"
        key_filename = "server.key"
        next_step = f"cd {args.name} && ./startup/start.sh"
    else:
        cert_filename = "client.crt"
        key_filename = "client.key"
        next_step = f"cd {args.name} && ./startup/start.sh"

    return {
        "output_dir": output_dir,
        "name": args.name,
        "type": args.kit_type,
        "endpoint": args.endpoint or "(not set)",
        "cert": os.path.join(output_dir, "startup", cert_filename),
        "key": os.path.join(output_dir, "startup", key_filename) + "  (permissions: 0600)",
        "rootca": os.path.join(output_dir, "startup", "rootCA.pem"),
        "next_step": next_step,
    }


def handle_package(args):
    """Assemble a startup kit from locally generated key + Project Admin cert + rootCA.pem."""

    # Step 1: --schema check (before any other work)
    if getattr(args, "schema", False):
        from nvflare.tool.cli_schema import parser_to_schema
        from nvflare.tool.package.package_cli import _package_parser

        schema = parser_to_schema(_package_parser, "nvflare package", examples=_PACKAGE_EXAMPLES)
        print(json.dumps(schema, indent=2))
        return 0

    fmt = getattr(args, "output_fmt", None)

    # Step 2: Validate required args and endpoint up front (before any file I/O)
    if not getattr(args, "kit_type", None):
        msg, hint = get_error("INVALID_ARGS", detail="-t/--type is required")
        output_error("INVALID_ARGS", msg, hint, fmt, exit_code=2)

    if not getattr(args, "endpoint", None):
        output_error(
            "INVALID_ARGS",
            f"--endpoint is required for -t {args.kit_type}.",
            "Provide the server endpoint URI, e.g. grpc://server.example.com:8002",
            fmt,
            exit_code=4,
        )
    try:
        _parse_endpoint(args.endpoint)
    except ValueError:
        msg, hint = get_error("INVALID_ENDPOINT", endpoint=args.endpoint)
        output_error("INVALID_ENDPOINT", msg, hint, fmt, exit_code=4)

    # Step 3: --output json implies --force
    if fmt == "json":
        args.force = True

    # Step 4: Resolve --dir vs explicit --cert/--key/--rootca
    has_dir = bool(getattr(args, "dir", None))
    has_explicit = any([getattr(args, "cert", None), getattr(args, "key", None), getattr(args, "rootca", None)])

    if has_dir and has_explicit:
        output_error(
            "INVALID_ARGS",
            "--dir and --cert/--key/--rootca are mutually exclusive.",
            "Use --dir for convention-based discovery, or --cert/--key/--rootca for explicit paths.",
            fmt,
            exit_code=4,
        )
    if not has_dir and not has_explicit:
        output_error(
            "INVALID_ARGS",
            "Provide either --dir or all of --cert, --key, --rootca.",
            "Use --dir <work-dir> if all files are in one directory.",
            fmt,
            exit_code=4,
        )

    if has_dir:
        # Auto-detect name from *.key if -n not given
        if not args.name:
            args.name = _discover_name_from_dir(args.dir, fmt)
        # Resolve paths by convention
        args.cert = os.path.join(args.dir, f"{args.kit_type}.crt")
        args.key = os.path.join(args.dir, f"{args.name}.key")
        args.rootca = os.path.join(args.dir, "rootCA.pem")
    else:
        # Explicit mode: -n/--name is required (no key file to auto-detect from)
        if not args.name:
            output_error(
                "INVALID_ARGS",
                "-n/--name is required when using --cert/--key/--rootca.",
                "Provide the participant name, e.g. -n hospital-1",
                fmt,
                exit_code=4,
            )

    # Step 6: Validate resolved cert/key/rootca exist
    if not os.path.isfile(args.cert):
        hint = (
            f"Place the signed {os.path.basename(args.cert)} from the Project Admin into {args.dir}."
            if has_dir
            else "Provide the signed certificate received from the Project Admin."
        )
        output_error("CERT_NOT_FOUND", f"Certificate file not found: {args.cert}.", hint, fmt, exit_code=1)

    if not os.path.isfile(args.key):
        hint = (
            f"Run 'nvflare cert csr -n {args.name} -o {args.dir}' to generate a key."
            if has_dir
            else "Provide the private key generated by 'nvflare cert csr'."
        )
        output_error("KEY_NOT_FOUND", f"Private key file not found: {args.key}.", hint, fmt, exit_code=1)

    if not os.path.isfile(args.rootca):
        hint = (
            f"Place the rootCA.pem from the Project Admin into {args.dir}."
            if has_dir
            else "Provide the rootCA.pem received from the Project Admin."
        )
        output_error("ROOTCA_NOT_FOUND", f"Root CA file not found: {args.rootca}.", hint, fmt, exit_code=1)

    # Step 7: Load and validate cert chain
    try:
        cert = load_crt(args.cert)
        ca_cert = load_crt(args.rootca)
        verify_cert(cert, ca_cert.public_key())
    except Exception:
        msg, hint = get_error("CERT_CHAIN_INVALID", cert=args.cert, rootca=args.rootca)
        output_error("CERT_CHAIN_INVALID", msg, hint, fmt, exit_code=1)

    # Step 8: Validate cert expiry
    try:
        expiry = cert.not_valid_after_utc
        now = datetime.datetime.now(datetime.timezone.utc)
    except AttributeError:
        # cryptography < 42.0
        expiry = cert.not_valid_after
        now = datetime.datetime.utcnow()
    if expiry < now:
        msg, hint = get_error("CERT_EXPIRED", cert=args.cert, expiry=expiry.isoformat())
        output_error("CERT_EXPIRED", msg, hint, fmt, exit_code=1)

    # Step 9: Parse endpoint (already validated in Step 2)
    scheme, host, port = _parse_endpoint(args.endpoint)

    # Step 10: Resolve output directory
    output_dir = args.output_dir if args.output_dir else os.path.join(os.getcwd(), args.name)
    output_dir = os.path.abspath(output_dir)

    # Step 11: Check output dir existence
    if os.path.exists(output_dir) and not args.force:
        msg, hint = get_error("OUTPUT_DIR_EXISTS", path=output_dir)
        output_error("OUTPUT_DIR_EXISTS", msg, hint, fmt, exit_code=1)

    # Step 12: Build the kit
    _build_kit(args, output_dir, scheme, host, port)

    # Step 13: Output result
    result = _build_result(args, output_dir, scheme, host, port)
    output(result, fmt)
