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
import os
import re
import shutil
import sys
from urllib.parse import urlparse

from nvflare.apis.utils.format_check import name_check
from nvflare.lighter.constants import AdminRole, CtxKey, ParticipantType, PropKey
from nvflare.lighter.entity import Project
from nvflare.lighter.impl.static_file import StaticFileBuilder
from nvflare.lighter.impl.workspace import WorkspaceBuilder
from nvflare.lighter.prov_utils import prepare_builders
from nvflare.lighter.provision import prepare_project
from nvflare.lighter.provisioner import Provisioner
from nvflare.lighter.spec import Builder
from nvflare.lighter.utils import load_crt, load_yaml, verify_cert
from nvflare.tool.cli_output import is_json_mode, output_error, output_error_message, output_ok, output_usage_error
from nvflare.tool.cli_schema import handle_schema_flag

_VALID_SCHEMES = {"grpc", "tcp", "http"}
_ADMIN_ROLES = {AdminRole.ORG_ADMIN, AdminRole.LEAD, AdminRole.MEMBER}
_VALID_CERT_TYPES = {"client", "server", "org_admin", "lead", "member"}
_KIT_TYPE_TO_ROLE = {
    "org_admin": AdminRole.ORG_ADMIN,
    "lead": AdminRole.LEAD,
    "member": AdminRole.MEMBER,
}
_DUMMY_SERVER_NAME = "dummy-server"
_DUMMY_ORG = "myorg"


def _read_cert_type_from_cert(cert) -> str:
    """Return the kit type embedded in cert's UNSTRUCTURED_NAME, or '' if absent."""
    from cryptography.x509.oid import NameOID

    attrs = cert.subject.get_attributes_for_oid(NameOID.UNSTRUCTURED_NAME)
    return attrs[0].value if attrs else ""


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
            fd = os.open(key_dst, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "wb") as _kf, open(key_path, "rb") as _src:
                _kf.write(_src.read())

            rootca_dst = os.path.join(kit_dir, "rootCA.pem")
            shutil.copy2(self.rootca_path, rootca_dst)
            os.chmod(rootca_dst, 0o644)


def _parse_endpoint(endpoint: str) -> tuple:
    """Return (scheme, host, port: int). Raises ValueError on invalid format."""
    parsed = urlparse(endpoint)
    if parsed.scheme not in _VALID_SCHEMES or not parsed.hostname or not parsed.port:
        raise ValueError(f"Invalid endpoint URI: {endpoint!r}")
    if not (1 <= parsed.port <= 65535):
        raise ValueError(f"Invalid endpoint URI: {endpoint!r} — port must be 1–65535")
    return parsed.scheme, parsed.hostname, parsed.port


def _discover_name_from_dir(work_dir: str, _args=None) -> str:
    """Find the single *.key file in work_dir and return its stem as the participant name."""
    keys = [f for f in os.listdir(work_dir) if f.endswith(".key")]
    if len(keys) == 0:
        output_error_message(
            "KEY_NOT_FOUND",
            f"No *.key file found in {work_dir}",
            "Run 'nvflare cert csr' to generate a key, or use --key explicitly.",
            None,
            exit_code=1,
        )
    if len(keys) > 1:
        output_error_message(
            "AMBIGUOUS_KEY",
            f"Multiple *.key files found in {work_dir}: {keys}",
            "Specify the participant name with -n, or use --key explicitly.",
            None,
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


def _load_project_from_file(path: str) -> tuple:
    """Load and validate a project yaml or single-site yaml. Returns (project, custom_builders).

    Accepts:
    - Full project YAML (api_version 3/4) with participants
    - Single-site YAML with name/org/type (converted to a minimal project)
    """
    if not os.path.isfile(path):
        output_error_message(
            "PROJECT_FILE_NOT_FOUND",
            f"Project file not found: {path}.",
            "Provide the path to a site-scoped project yaml file.",
            None,
            exit_code=1,
        )
    try:
        project_dict = load_yaml(path)
    except Exception as ex:
        output_error_message(
            "INVALID_PROJECT_FILE",
            f"Failed to parse project file: {ex}",
            "Ensure the file is valid YAML.",
            None,
            exit_code=1,
        )

    # Single-site yaml support: name/org/type only
    if (
        isinstance(project_dict, dict)
        and "name" in project_dict
        and "org" in project_dict
        and "type" in project_dict
        and "participants" not in project_dict
        and PropKey.API_VERSION not in project_dict
    ):
        p_type = project_dict.get("type")
        if p_type not in _VALID_CERT_TYPES:
            output_error_message(
                "INVALID_PROJECT_FILE",
                f"Invalid site type: {p_type}",
                "Use one of: client, server, org_admin, lead, member.",
                None,
                exit_code=1,
            )
        participant = {"name": project_dict.get("name"), "org": project_dict.get("org")}
        if p_type in _ADMIN_ROLES:
            participant.update({"type": "admin", "role": _KIT_TYPE_TO_ROLE[p_type]})
        else:
            participant.update({"type": p_type})
        project_dict = {
            PropKey.API_VERSION: 3,
            PropKey.NAME: project_dict.get("project_name", "project"),
            "participants": [participant],
        }

    try:
        project = prepare_project(project_dict)
    except Exception as ex:
        output_error_message(
            "INVALID_PROJECT_FILE",
            f"Invalid project file: {ex}",
            "Ensure the file is schema-compatible with 'nvflare provision' project.yaml (api_version: 3 or 4), "
            "or provide a single-site yaml with name/org/type.",
            None,
            exit_code=1,
        )

    if project.get_relays():
        output_error_message(
            "UNSUPPORTED_TOPOLOGY",
            "Relay participants found in project file — hierarchical FL is not supported by 'nvflare package'.",
            "Use 'nvflare provision' for relay topologies.",
            None,
            exit_code=4,
        )

    custom_builders = prepare_builders(project_dict)
    return project, custom_builders


def _handle_package_yaml_mode(args, scheme, host, port):
    """Build kits for all participants defined in a project yaml file.

    Called by handle_package when --project-file is given.  The cert/key for each participant
    is resolved from --dir/<name>.crt and --dir/<name>.key; rootCA is --dir/rootCA.pem.
    """
    project_from_yaml, custom_builders = _load_project_from_file(args.project_file)

    # --dir is required in yaml mode (no single key auto-discovery)
    if not getattr(args, "dir", None):
        output_error_message(
            "INVALID_ARGS",
            "--dir is required when using --project-file.",
            "Provide the directory containing cert files named by participant CN.",
            None,
            exit_code=4,
        )

    rootca_path = os.path.join(args.dir, "rootCA.pem")
    if not os.path.isfile(rootca_path):
        output_error_message(
            "ROOTCA_NOT_FOUND",
            f"Root CA file not found: {rootca_path}.",
            f"Place the rootCA.pem from the Project Admin into {args.dir}.",
            None,
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

    # Include server if this is a server-only project and no -t filter was given.
    if not all_participants and server and not getattr(args, "kit_type", None):
        all_participants.append(("server", server))

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
        output_error_message(
            "NO_PARTICIPANTS",
            "No participants to build after applying type filter.",
            "Check the project file and -t filter.",
            None,
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
        output_error_message(
            "ROOTCA_INVALID",
            f"Failed to load root CA certificate '{rootca_path}': {e}.",
            "Ensure the file is a valid PEM-encoded certificate.",
            None,
            exit_code=1,
        )
    for kit_type, participant in all_participants:
        p_name = participant.name
        cert_path = os.path.join(args.dir, f"{p_name}.crt")
        key_path = os.path.join(args.dir, f"{p_name}.key")

        if not os.path.isfile(cert_path):
            output_error_message(
                "CERT_NOT_FOUND",
                f"Certificate not found: {cert_path}.",
                f"Drop {p_name}.crt from the Project Admin into {args.dir}.",
                None,
                exit_code=1,
            )
        if not os.path.isfile(key_path):
            output_error_message(
                "KEY_NOT_FOUND",
                f"Private key not found: {key_path}.",
                f"Run 'nvflare cert csr -n {p_name}' to generate a key.",
                None,
                exit_code=1,
            )

        try:
            cert = load_crt(cert_path)
            verify_cert(cert, ca_cert.public_key())
        except Exception:
            output_error("CERT_CHAIN_INVALID", exit_code=1, cert=cert_path, rootca=rootca_path)

        try:
            expiry = cert.not_valid_after_utc
            now = datetime.datetime.now(datetime.timezone.utc)
        except AttributeError:
            expiry = cert.not_valid_after
            now = datetime.datetime.utcnow()
        if expiry < now:
            output_error("CERT_EXPIRED", exit_code=1, cert=cert_path, expiry=expiry.isoformat())

        cert_map[p_name] = (cert_path, key_path)

    # Pre-check: if any participant already exists in the latest prod dir, reject unless --force.
    latest_prod = _latest_prod_dir(workspace, project_name)
    if latest_prod:
        for _, participant in all_participants:
            existing_path = os.path.join(latest_prod, participant.name)
            if os.path.exists(existing_path) and not args.force:
                output_error("OUTPUT_DIR_EXISTS", exit_code=1, path=existing_path)

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
    # WorkspaceBuilder and StaticFileBuilder are always managed by nvflare package:
    # StaticFileBuilder is constructed with the scheme derived from --endpoint, and
    # WorkspaceBuilder is always default. Any YAML builder entries for these types are
    # stripped to prevent double finalize() calls (BUILD_FAILED). Custom args such as
    # config_folder or app_validator on YAML StaticFileBuilder entries are intentionally
    # ignored — nvflare package enforces a fixed startup-kit layout.
    _MANAGED_BUILDER_TYPES = (WorkspaceBuilder, StaticFileBuilder)
    for b in custom_builders:
        if isinstance(b, _MANAGED_BUILDER_TYPES):
            import warnings

            warnings.warn(
                f"{type(b).__name__} in project YAML builders is ignored by 'nvflare package'. "
                "WorkspaceBuilder and StaticFileBuilder are always provided by nvflare package "
                "with settings derived from --endpoint. Custom args (e.g. config_folder) have no effect.",
                UserWarning,
                stacklevel=2,
            )
    filtered_custom = [b for b in custom_builders if not isinstance(b, _MANAGED_BUILDER_TYPES)]
    all_builders = [workspace_builder, cert_builder, static_builder] + filtered_custom

    provisioner = Provisioner(root_dir=workspace, builders=all_builders)
    ctx = provisioner.provision(project)

    if ctx.get(CtxKey.BUILD_ERROR):
        output_error_message(
            "BUILD_FAILED",
            "Kit assembly failed.",
            "See error output above for details.",
            None,
            exit_code=1,
        )

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
        output_ok(results[0])
    else:
        output_ok(results)
    return 0


def handle_package(args):
    """Assemble a startup kit from locally generated key + Project Admin cert + rootCA.pem."""

    # Step 1: --schema check (before any other work)
    from nvflare.tool.package.package_cli import _PACKAGE_EXAMPLES, _package_parser

    handle_schema_flag(_package_parser, "nvflare package", _PACKAGE_EXAMPLES, sys.argv[1:])

    # Step 2: Validate required args and endpoint up front (before any file I/O)
    has_project_file = bool(getattr(args, "project_file", None))

    if has_project_file and getattr(args, "name", None):
        output_error_message(
            "INVALID_ARGS",
            "--project-file and -n/--name are mutually exclusive.",
            "Use --project-file to define participants via yaml, or -n to name a single participant.",
            None,
            exit_code=4,
        )

    if has_project_file and any(getattr(args, attr, None) for attr in ("cert", "key", "rootca")):
        output_error_message(
            "INVALID_ARGS",
            "--project-file and --cert/--key/--rootca are mutually exclusive.",
            "In yaml mode, cert files are discovered from --dir by participant name. Do not pass --cert/--key/--rootca.",
            None,
            exit_code=4,
        )

    if not getattr(args, "endpoint", None):
        detail = f"for -t {args.kit_type}" if getattr(args, "kit_type", None) else "for this command"
        output_usage_error(
            _package_parser if not is_json_mode() else None,
            f"--endpoint is required {detail}.",
            exit_code=4,
            hint="Provide the server endpoint URI, e.g. grpc://server.example.com:8002",
        )
    try:
        scheme, host, port = _parse_endpoint(args.endpoint)
    except ValueError:
        output_error("INVALID_ENDPOINT", exit_code=4, endpoint=args.endpoint)

    # -----------------------------------------------------------------------
    # YAML mode: --project-file given — build kits for all participants defined
    # in the yaml file.  --dir is required (per-participant certs named by CN).
    # -----------------------------------------------------------------------
    if has_project_file:
        return _handle_package_yaml_mode(args, scheme, host, port)

    # Step 4: Resolve --dir vs explicit --cert/--key/--rootca
    has_dir = bool(getattr(args, "dir", None))
    has_explicit = any(getattr(args, attr, None) for attr in ("cert", "key", "rootca"))

    if has_dir and has_explicit:
        output_error_message(
            "INVALID_ARGS",
            "--dir and --cert/--key/--rootca are mutually exclusive.",
            "Use --dir for convention-based discovery, or --cert/--key/--rootca for explicit paths.",
            None,
            exit_code=4,
        )
    if not has_dir and not has_explicit:
        output_error_message(
            "INVALID_ARGS",
            "Provide either --dir or all of --cert, --key, --rootca.",
            "Use --dir <work-dir> if all files are in one directory.",
            None,
            exit_code=4,
        )

    if has_dir:
        # Auto-detect name from *.key if -n not given
        if not args.name:
            args.name = _discover_name_from_dir(args.dir)
        # Resolve paths by convention — all files are named after the participant
        args.cert = os.path.join(args.dir, f"{args.name}.crt")
        args.key = os.path.join(args.dir, f"{args.name}.key")
        args.rootca = os.path.join(args.dir, "rootCA.pem")
    else:
        # Explicit mode: all three paths and -n/--name are required.
        missing = [f"--{f}" for f in ("cert", "key", "rootca") if not getattr(args, f, None)]
        if missing:
            output_error_message(
                "INVALID_ARGS",
                f"Missing required argument(s): {', '.join(missing)}.",
                "When using explicit mode, --cert, --key, and --rootca must all be provided.",
                None,
                exit_code=4,
            )
        if not args.name:
            output_error_message(
                "INVALID_ARGS",
                "-n/--name is required when using --cert/--key/--rootca.",
                "Provide the participant name, e.g. -n hospital-1",
                None,
                exit_code=4,
            )

    # Step 5: Guard sentinel name collision (host collision check happens after kit_type is known).
    if args.name == _DUMMY_SERVER_NAME:
        output_error_message(
            "INVALID_ARGS",
            f"Participant name {_DUMMY_SERVER_NAME!r} is reserved and cannot be used.",
            "Choose a different name for this participant.",
            None,
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
        output_error_message("CERT_NOT_FOUND", f"Certificate file not found: {args.cert}.", hint, None, exit_code=1)

    if not os.path.isfile(args.key):
        hint = (
            f"Run 'nvflare cert csr -n {args.name} -o {args.dir}' to generate a key."
            if has_dir
            else "Provide the private key generated by 'nvflare cert csr'."
        )
        output_error_message("KEY_NOT_FOUND", f"Private key file not found: {args.key}.", hint, None, exit_code=1)

    if not os.path.isfile(args.rootca):
        hint = (
            f"Place the rootCA.pem from the Project Admin into {args.dir}."
            if has_dir
            else "Provide the rootCA.pem received from the Project Admin."
        )
        output_error_message("ROOTCA_NOT_FOUND", f"Root CA file not found: {args.rootca}.", hint, None, exit_code=1)

    # Step 7: Load and validate cert chain
    try:
        cert = load_crt(args.cert)
        ca_cert = load_crt(args.rootca)
        verify_cert(cert, ca_cert.public_key())
    except Exception:
        output_error("CERT_CHAIN_INVALID", exit_code=1, cert=args.cert, rootca=args.rootca)

    # Step 8: Validate cert expiry
    try:
        expiry = cert.not_valid_after_utc
        now = datetime.datetime.now(datetime.timezone.utc)
    except AttributeError:
        # cryptography < 42.0
        expiry = cert.not_valid_after
        now = datetime.datetime.utcnow()
    if expiry < now:
        output_error("CERT_EXPIRED", exit_code=1, cert=args.cert, expiry=expiry.isoformat())

    # Step 8b: Derive kit_type from cert's UNSTRUCTURED_NAME.
    # The signed cert's embedded type is the sole authoritative source.
    kit_type = _read_cert_type_from_cert(cert)
    if not kit_type or kit_type not in _VALID_CERT_TYPES:
        output_error("CERT_TYPE_UNKNOWN", exit_code=1, cert=args.cert)
    args.kit_type = kit_type

    # Step 8c: Type-dependent pre-flight checks (require kit_type to be resolved first).
    if args.kit_type in _ADMIN_ROLES and name_check(args.name, "admin")[0]:
        output_error_message(
            "INVALID_ARGS",
            f"Admin name must be an email address (got {args.name!r}).",
            "Use an email-format name, e.g. alice@myorg.com",
            None,
            exit_code=4,
        )

    # For non-server kits the endpoint hostname is used as the server placeholder name.
    # The participant name must not collide with it.
    if args.kit_type != "server" and args.name == host:
        output_error_message(
            "INVALID_ARGS",
            f"Participant name {args.name!r} collides with the server endpoint hostname.",
            "Use a different -n/--name that is distinct from the server hostname in --endpoint.",
            None,
            exit_code=4,
        )

    admin_port = args.admin_port if args.admin_port is not None else port

    # Step 9: Resolve workspace and project name
    workspace = os.path.abspath(getattr(args, "workspace", None) or "workspace")
    project_name = getattr(args, "project_name", None) or "project"

    # Step 10: Pre-check — if participant already exists in the latest prod dir and --force is not set, error out.
    latest_prod = _latest_prod_dir(workspace, project_name)
    if latest_prod:
        existing_path = os.path.join(latest_prod, args.name)
        if os.path.exists(existing_path) and not args.force:
            output_error("OUTPUT_DIR_EXISTS", exit_code=1, path=existing_path)

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
        output_error_message(
            "BUILD_FAILED",
            "Kit assembly failed during provisioning.",
            "See error output above for details.",
            None,
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
    output_ok(result)
    return 0
