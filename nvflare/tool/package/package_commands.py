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

from nvflare.lighter.constants import AdminRole, CtxKey, ParticipantType, PropKey
from nvflare.lighter.entity import Project
from nvflare.lighter.impl.static_file import StaticFileBuilder
from nvflare.lighter.impl.workspace import WorkspaceBuilder
from nvflare.lighter.prov_utils import prepare_builders
from nvflare.lighter.provision import prepare_project
from nvflare.lighter.provisioner import Provisioner
from nvflare.lighter.spec import Builder
from nvflare.lighter.utils import load_crt, load_yaml, verify_cert
from nvflare.tool.cli_errors import get_error
from nvflare.tool.cli_output import output, output_error

_ENDPOINT_PATTERN = re.compile(r"^(grpc|tcp|http)://([^:/]+):(\d+)$")
_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
_ADMIN_ROLES = {AdminRole.ORG_ADMIN, AdminRole.LEAD, AdminRole.MEMBER}
_KIT_TYPE_TO_ROLE = {
    "org_admin": AdminRole.ORG_ADMIN,
    "lead": AdminRole.LEAD,
    "member": AdminRole.MEMBER,
}
_DUMMY_SERVER_NAME = "dummy-server"
_DUMMY_ORG = "myorg"

_PACKAGE_EXAMPLES = [
    "nvflare package -t lead -e grpc://fl-server:8002 --dir ./alice --project-name myproject",
    "nvflare package -t client -e grpc://fl-server:8002 --dir ./hospital-1 -w ./workspace --project-name myproject",
    "nvflare package -n hospital-1 -t client -e grpc://fl-server:8002 --cert ./signed/hospital-1/hospital-1.crt --key ./csr/hospital-1.key --rootca ./signed/hospital-1/rootCA.pem",
]


class PrebuiltCertBuilder(Builder):
    """Builder that installs pre-made cert/key/rootCA into each participant's kit directory.

    Replaces CertBuilder for the ``nvflare package`` workflow where certs are already signed
    externally.  The caller supplies the cert/key/rootCA paths; this builder simply copies them
    into the correct locations that StaticFileBuilder and the runtime expect.

    Two modes:
    - Single-participant: pass cert_path/key_path/rootca_path + optional target_name.
    - Multi-participant: pass cert_map={name: (cert_path, key_path)} + rootca_path.
      Only participants present in cert_map receive certs; others (e.g. dummy server) are skipped.
    """

    def __init__(
        self,
        cert_path: str = None,
        key_path: str = None,
        rootca_path: str = None,
        target_name: str = None,
        cert_map: dict = None,
    ):
        self.cert_path = cert_path
        self.key_path = key_path
        self.rootca_path = rootca_path
        self.target_name = target_name
        self.cert_map = cert_map  # {participant_name: (cert_path, key_path)}

    def build(self, project: Project, ctx):
        for p in project.get_all_participants():
            if self.cert_map is not None:
                if p.name not in self.cert_map:
                    continue
                cert_path, key_path = self.cert_map[p.name]
            else:
                # Skip placeholder participants (e.g. dummy server added for non-server kits).
                if self.target_name is not None and p.name != self.target_name:
                    continue
                cert_path = self.cert_path
                key_path = self.key_path
            kit_dir = ctx.get_kit_dir(p)
            os.makedirs(kit_dir, exist_ok=True)

            if p.type == ParticipantType.SERVER:
                cert_dst = os.path.join(kit_dir, "server.crt")
                key_dst = os.path.join(kit_dir, "server.key")
            else:
                cert_dst = os.path.join(kit_dir, "client.crt")
                key_dst = os.path.join(kit_dir, "client.key")

            shutil.copy2(cert_path, cert_dst)
            shutil.copy2(key_path, key_dst)
            os.chmod(key_dst, 0o600)

            rootca_dst = os.path.join(kit_dir, "rootCA.pem")
            shutil.copy2(self.rootca_path, rootca_dst)
            os.chmod(rootca_dst, 0o644)


def _parse_endpoint(endpoint: str) -> tuple:
    """Return (scheme, host, port: int). Raises ValueError on invalid format."""
    m = _ENDPOINT_PATTERN.match(endpoint)
    if not m:
        raise ValueError(f"Invalid endpoint URI: {endpoint!r}")
    port = int(m.group(3))
    if not (1 <= port <= 65535):
        raise ValueError(f"Invalid endpoint URI: {endpoint!r} — port must be 1–65535")
    return m.group(1), m.group(2), port


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


def _latest_prod_dir(workspace: str, project_name: str):
    """Return the path to the highest-numbered existing prod_NN under workspace/project_name/.

    Returns None if no prod_NN directories exist.
    """
    project_dir = os.path.join(workspace, project_name)
    if not os.path.exists(project_dir):
        return None
    dirs = [
        d
        for d in os.listdir(project_dir)
        if os.path.isdir(os.path.join(project_dir, d)) and re.match(r"^prod_\d{2}$", d)
    ]
    if not dirs:
        return None
    return os.path.join(project_dir, max(dirs))


def _load_project_from_file(path: str, fmt) -> tuple:
    """Load and validate a site-scoped project yaml. Returns (project, custom_builders).

    Validates:
    - File exists and is valid yaml
    - api_version == 3
    - No relay participants (hierarchical FL not supported)
    """
    if not os.path.isfile(path):
        output_error(
            "PROJECT_FILE_NOT_FOUND",
            f"Project file not found: {path}.",
            "Provide the path to a site-scoped project yaml file.",
            fmt,
            exit_code=1,
        )
    try:
        project_dict = load_yaml(path)
    except Exception as ex:
        output_error(
            "INVALID_PROJECT_FILE",
            f"Failed to parse project file: {ex}",
            "Ensure the file is valid YAML.",
            fmt,
            exit_code=1,
        )

    try:
        project = prepare_project(project_dict)
    except Exception as ex:
        output_error(
            "INVALID_PROJECT_FILE",
            f"Invalid project file: {ex}",
            "Ensure the file is schema-compatible with 'nvflare provision' project.yaml (api_version: 3).",
            fmt,
            exit_code=1,
        )

    if project.get_relays():
        output_error(
            "UNSUPPORTED_TOPOLOGY",
            "Relay participants found in project file — hierarchical FL is not supported by 'nvflare package'.",
            "Use 'nvflare provision' for relay topologies.",
            fmt,
            exit_code=4,
        )

    custom_builders = prepare_builders(project_dict)
    return project, custom_builders


def _handle_package_yaml_mode(args, fmt, scheme, host, port):
    """Build kits for all participants defined in a project yaml file.

    Called by handle_package when --project-file is given.  The cert/key for each participant
    is resolved from --dir/<name>.crt and --dir/<name>.key; rootCA is --dir/rootCA.pem.
    """
    project_from_yaml, custom_builders = _load_project_from_file(args.project_file, fmt)

    # --dir is required in yaml mode (no single key auto-discovery)
    if not getattr(args, "dir", None):
        output_error(
            "INVALID_ARGS",
            "--dir is required when using --project-file.",
            "Provide the directory containing cert files named by participant CN.",
            fmt,
            exit_code=4,
        )

    rootca_path = os.path.join(args.dir, "rootCA.pem")
    if not os.path.isfile(rootca_path):
        output_error(
            "ROOTCA_NOT_FOUND",
            f"Root CA file not found: {rootca_path}.",
            f"Place the rootCA.pem from the Project Admin into {args.dir}.",
            fmt,
            exit_code=1,
        )

    workspace = os.path.abspath(getattr(args, "workspace", None) or "workspace")
    project_name = getattr(args, "project_name", None) or project_from_yaml.name or "project"
    admin_port = args.admin_port if args.admin_port is not None else port

    # Collect participants to build: clients, admins (server handled separately if -t server).
    # Each entry is (kit_type_str, participant).
    all_participants = []
    server = project_from_yaml.get_server()
    if server and getattr(args, "kit_type", None) == "server":
        all_participants.append(("server", server))
    for p in project_from_yaml.get_clients():
        all_participants.append(("client", p))
    for p in project_from_yaml.get_admins():
        role = p.get_prop(PropKey.ROLE, AdminRole.LEAD)
        # Map canonical role names to the kit_type strings used by the provisioner
        kit_type = role if role in _ADMIN_ROLES else AdminRole.LEAD
        all_participants.append((kit_type, p))

    # Apply -t filter when given
    kit_type_filter = getattr(args, "kit_type", None)
    if kit_type_filter:
        if kit_type_filter == "server":
            all_participants = [(t, p) for t, p in all_participants if t == "server"]
        elif kit_type_filter == "client":
            all_participants = [(t, p) for t, p in all_participants if t == "client"]
        else:
            # admin role filter
            all_participants = [(t, p) for t, p in all_participants if t == kit_type_filter]

    if not all_participants:
        output_error(
            "NO_PARTICIPANTS",
            "No participants to build after applying type filter.",
            "Check the project file and -t filter.",
            fmt,
            exit_code=1,
        )

    server_props = {
        PropKey.FED_LEARN_PORT: port,
        PropKey.ADMIN_PORT: admin_port,
    }

    # Validate all cert/key files up front before building anything.
    cert_map = {}
    try:
        ca_cert = load_crt(rootca_path)
    except Exception as e:
        output_error(
            "ROOTCA_INVALID",
            f"Failed to load root CA certificate '{rootca_path}': {e}.",
            "Ensure the file is a valid PEM-encoded certificate.",
            fmt,
            exit_code=1,
        )
    for kit_type, participant in all_participants:
        p_name = participant.name
        cert_path = os.path.join(args.dir, f"{p_name}.crt")
        key_path = os.path.join(args.dir, f"{p_name}.key")

        if not os.path.isfile(cert_path):
            output_error(
                "CERT_NOT_FOUND",
                f"Certificate not found: {cert_path}.",
                f"Drop {p_name}.crt from the Project Admin into {args.dir}.",
                fmt,
                exit_code=1,
            )
        if not os.path.isfile(key_path):
            output_error(
                "KEY_NOT_FOUND",
                f"Private key not found: {key_path}.",
                f"Run 'nvflare cert csr -n {p_name}' to generate a key.",
                fmt,
                exit_code=1,
            )

        try:
            cert = load_crt(cert_path)
            verify_cert(cert, ca_cert.public_key())
        except Exception:
            msg, hint = get_error("CERT_CHAIN_INVALID", cert=cert_path, rootca=rootca_path)
            output_error("CERT_CHAIN_INVALID", msg, hint, fmt, exit_code=1)

        try:
            expiry = cert.not_valid_after_utc
            now = datetime.datetime.now(datetime.timezone.utc)
        except AttributeError:
            expiry = cert.not_valid_after
            now = datetime.datetime.utcnow()
        if expiry < now:
            msg, hint = get_error("CERT_EXPIRED", cert=cert_path, expiry=expiry.isoformat())
            output_error("CERT_EXPIRED", msg, hint, fmt, exit_code=1)

        cert_map[p_name] = (cert_path, key_path)

    # Pre-check: if any participant already exists in the latest prod dir, reject unless --force.
    latest_prod = _latest_prod_dir(workspace, project_name)
    if latest_prod:
        for _, participant in all_participants:
            existing_path = os.path.join(latest_prod, participant.name)
            if os.path.exists(existing_path) and not args.force:
                msg, hint = get_error("OUTPUT_DIR_EXISTS", path=existing_path)
                output_error("OUTPUT_DIR_EXISTS", msg, hint, fmt, exit_code=1)

    # Build all participants in a single provisioner call so they land in the same prod_NN.
    has_server = any(kt == "server" for kt, _ in all_participants)
    project = Project(name=project_name, description="")
    if has_server:
        server_participant = next(p for kt, p in all_participants if kt == "server")
        project.set_server(server_participant.name, _DUMMY_ORG, server_props)
    else:
        project.set_server(host, _DUMMY_ORG, server_props)

    for kit_type, participant in all_participants:
        if kit_type == "server":
            continue
        if kit_type == "client":
            project.add_client(participant.name, _DUMMY_ORG, {})
        else:
            project.add_admin(participant.name, _DUMMY_ORG, {PropKey.ROLE: _KIT_TYPE_TO_ROLE[kit_type]})

    cert_builder = PrebuiltCertBuilder(rootca_path=rootca_path, cert_map=cert_map)
    static_builder = StaticFileBuilder(scheme=scheme)
    workspace_builder = WorkspaceBuilder()
    all_builders = [workspace_builder, cert_builder, static_builder] + custom_builders

    provisioner = Provisioner(root_dir=workspace, builders=all_builders)
    ctx = provisioner.provision(project)

    if ctx.get(CtxKey.BUILD_ERROR):
        output_error("BUILD_FAILED", "Kit assembly failed.", "See error output above for details.", fmt, exit_code=1)

    prod_dir = ctx[CtxKey.CURRENT_PROD_DIR]

    # Remove placeholder server directory if we added a dummy server.
    if not has_server:
        shutil.rmtree(os.path.join(prod_dir, host), ignore_errors=True)

    results = []
    for kit_type, participant in all_participants:
        p_name = participant.name
        output_dir = os.path.join(prod_dir, p_name)
        is_server = kit_type == "server"
        is_admin = kit_type in _ADMIN_ROLES
        cert_filename = "server.crt" if is_server else "client.crt"
        key_filename = "server.key" if is_server else "client.key"
        startup_script = "fl_admin.sh" if is_admin else "start.sh"
        results.append(
            {
                "output_dir": output_dir,
                "name": p_name,
                "type": kit_type,
                "endpoint": args.endpoint,
                "cert": os.path.join(output_dir, "startup", cert_filename),
                "key": os.path.join(output_dir, "startup", key_filename) + "  (permissions: 0600)",
                "rootca": os.path.join(output_dir, "startup", "rootCA.pem"),
                "transfer_dir": os.path.join(output_dir, "transfer"),
                "next_step": f"cd {output_dir} && ./startup/{startup_script}",
            }
        )

    if len(results) == 1:
        output(results[0], fmt)
    else:
        output(results, fmt)
    return 0


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
    has_project_file = bool(getattr(args, "project_file", None))

    if not has_project_file and not getattr(args, "kit_type", None):
        msg, hint = get_error("INVALID_ARGS", detail="-t/--type is required")
        output_error("INVALID_ARGS", msg, hint, fmt, exit_code=2)

    if has_project_file and getattr(args, "name", None):
        output_error(
            "INVALID_ARGS",
            "--project-file and -n/--name are mutually exclusive.",
            "Use --project-file to define participants via yaml, or -n to name a single participant.",
            fmt,
            exit_code=4,
        )

    if has_project_file and any(getattr(args, attr, None) for attr in ("cert", "key", "rootca")):
        output_error(
            "INVALID_ARGS",
            "--project-file and --cert/--key/--rootca are mutually exclusive.",
            "In yaml mode, cert files are discovered from --dir by participant name. Do not pass --cert/--key/--rootca.",
            fmt,
            exit_code=4,
        )

    if not getattr(args, "endpoint", None):
        detail = f"for -t {args.kit_type}" if getattr(args, "kit_type", None) else "for this command"
        output_error(
            "INVALID_ARGS",
            f"--endpoint is required {detail}.",
            "Provide the server endpoint URI, e.g. grpc://server.example.com:8002",
            fmt,
            exit_code=4,
        )
    try:
        scheme, host, port = _parse_endpoint(args.endpoint)
    except ValueError:
        msg, hint = get_error("INVALID_ENDPOINT", endpoint=args.endpoint)
        output_error("INVALID_ENDPOINT", msg, hint, fmt, exit_code=4)

    # Step 3: --output json implies --force
    if fmt == "json":
        args.force = True

    # -----------------------------------------------------------------------
    # YAML mode: --project-file given — build kits for all participants defined
    # in the yaml file.  --dir is required (per-participant certs named by CN).
    # -----------------------------------------------------------------------
    if has_project_file:
        return _handle_package_yaml_mode(args, fmt, scheme, host, port)

    # Step 4: Resolve --dir vs explicit --cert/--key/--rootca
    has_dir = bool(getattr(args, "dir", None))
    has_explicit = any(getattr(args, attr, None) for attr in ("cert", "key", "rootca"))

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
        # Resolve paths by convention — all files are named after the participant
        args.cert = os.path.join(args.dir, f"{args.name}.crt")
        args.key = os.path.join(args.dir, f"{args.name}.key")
        args.rootca = os.path.join(args.dir, "rootCA.pem")
    else:
        # Explicit mode: all three paths and -n/--name are required.
        missing = [f"--{f}" for f in ("cert", "key", "rootca") if not getattr(args, f, None)]
        if missing:
            output_error(
                "INVALID_ARGS",
                f"Missing required argument(s): {', '.join(missing)}.",
                "When using explicit mode, --cert, --key, and --rootca must all be provided.",
                fmt,
                exit_code=4,
            )
        if not args.name:
            output_error(
                "INVALID_ARGS",
                "-n/--name is required when using --cert/--key/--rootca.",
                "Provide the participant name, e.g. -n hospital-1",
                fmt,
                exit_code=4,
            )

    # Step 5: Pre-flight: validate admin name is email-format; guard sentinel name collision.
    if args.kit_type in _ADMIN_ROLES and not _EMAIL_RE.match(args.name):
        output_error(
            "INVALID_ARGS",
            f"Admin name must be an email address (got {args.name!r}).",
            "Use an email-format name, e.g. alice@myorg.com",
            fmt,
            exit_code=4,
        )

    if args.name == _DUMMY_SERVER_NAME:
        output_error(
            "INVALID_ARGS",
            f"Participant name {_DUMMY_SERVER_NAME!r} is reserved and cannot be used.",
            "Choose a different name for this participant.",
            fmt,
            exit_code=4,
        )

    # For non-server kits the endpoint hostname is used as the server placeholder name.
    # The participant name must not collide with it, otherwise the provisioner would have
    # two participants with the same name.
    if args.kit_type != "server" and args.name == host:
        output_error(
            "INVALID_ARGS",
            f"Participant name {args.name!r} collides with the server endpoint hostname.",
            "Use a different -n/--name that is distinct from the server hostname in --endpoint.",
            fmt,
            exit_code=4,
        )

    # Step 6: Validate resolved cert/key/rootca exist
    if not os.path.isfile(args.cert):
        if has_dir:
            hint = (
                f"{os.path.basename(args.cert)} is the signed certificate from the Project Admin "
                f"(different from {args.name}.csr, which is the signing request you generated). "
                f"Ask your Project Admin to run 'nvflare cert sign' on your {args.name}.csr, "
                f"then place the resulting {os.path.basename(args.cert)} and rootCA.pem into {args.dir}."
            )
        else:
            hint = "Provide the signed certificate received from the Project Admin."
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

    admin_port = args.admin_port if args.admin_port is not None else port

    # Step 9: Resolve workspace and project name
    workspace = os.path.abspath(getattr(args, "workspace", None) or "workspace")
    project_name = getattr(args, "project_name", None) or "project"

    # Step 10: Pre-check — if participant already exists in the latest prod dir and --force is not set, error out.
    latest_prod = _latest_prod_dir(workspace, project_name)
    if latest_prod:
        existing_path = os.path.join(latest_prod, args.name)
        if os.path.exists(existing_path) and not args.force:
            msg, hint = get_error("OUTPUT_DIR_EXISTS", path=existing_path)
            output_error("OUTPUT_DIR_EXISTS", msg, hint, fmt, exit_code=1)

    # Step 11: Construct project and provision via the provisioner infrastructure.
    is_server = args.kit_type == "server"

    server_props = {
        PropKey.FED_LEARN_PORT: port,
        PropKey.ADMIN_PORT: admin_port,
    }

    project = Project(name=project_name, description="")
    if is_server:
        # server.name = args.name (identity / cert CN) — used for target, admin_server, sp_end_point.
        project.set_server(args.name, _DUMMY_ORG, server_props)
    else:
        # For non-server kits, use the endpoint hostname as the server name so that
        # server_identity in fed_client.json / fed_admin.json points to the real server.
        # The pre-flight checks above ensure args.name != host and args.name != _DUMMY_SERVER_NAME.
        project.set_server(host, _DUMMY_ORG, server_props)

    if not is_server:
        if args.kit_type == "client":
            project.add_client(args.name, _DUMMY_ORG, {})
        else:
            project.add_admin(args.name, _DUMMY_ORG, {PropKey.ROLE: _KIT_TYPE_TO_ROLE[args.kit_type]})

    cert_builder = PrebuiltCertBuilder(
        cert_path=args.cert,
        key_path=args.key,
        rootca_path=args.rootca,
        target_name=args.name,
    )
    static_builder = StaticFileBuilder(scheme=scheme)
    workspace_builder = WorkspaceBuilder()

    provisioner = Provisioner(
        root_dir=workspace,
        builders=[workspace_builder, cert_builder, static_builder],
    )
    ctx = provisioner.provision(project)

    if ctx.get(CtxKey.BUILD_ERROR):
        output_error(
            "BUILD_FAILED",
            "Kit assembly failed during provisioning.",
            "See error output above for details.",
            fmt,
            exit_code=1,
        )

    prod_dir = ctx[CtxKey.CURRENT_PROD_DIR]

    # Step 12: For non-server kits, remove the server placeholder directory from prod output.
    if not is_server:
        shutil.rmtree(os.path.join(prod_dir, host), ignore_errors=True)

    # Step 13: Determine result paths.
    output_dir = os.path.join(prod_dir, args.name)

    is_admin = args.kit_type in _ADMIN_ROLES
    cert_filename = "server.crt" if is_server else "client.crt"
    key_filename = "server.key" if is_server else "client.key"
    startup_script = "fl_admin.sh" if is_admin else "start.sh"
    next_step = f"cd {output_dir} && ./startup/{startup_script}"

    result = {
        "output_dir": output_dir,
        "name": args.name,
        "type": args.kit_type,
        "endpoint": args.endpoint or "(not set)",
        "cert": os.path.join(output_dir, "startup", cert_filename),
        "key": os.path.join(output_dir, "startup", key_filename) + "  (permissions: 0600)",
        "rootca": os.path.join(output_dir, "startup", "rootCA.pem"),
        "transfer_dir": os.path.join(output_dir, "transfer"),
        "next_step": next_step,
    }
    output(result, fmt)
    return 0
