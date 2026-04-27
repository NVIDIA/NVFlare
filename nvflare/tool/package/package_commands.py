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
import hashlib
import json
import os
import posixpath
import re
import shutil
import stat
import sys
import tempfile
import zipfile
from typing import Optional
from urllib.parse import urlparse

import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.x509.oid import NameOID

from nvflare.apis.utils.format_check import name_check
from nvflare.lighter.constants import AdminRole, CtxKey, ParticipantType, PropKey
from nvflare.lighter.entity import Project
from nvflare.lighter.impl.cert import CertBuilder
from nvflare.lighter.impl.signature import SignatureBuilder
from nvflare.lighter.impl.static_file import StaticFileBuilder
from nvflare.lighter.impl.workspace import WorkspaceBuilder
from nvflare.lighter.prov_utils import prepare_builders
from nvflare.lighter.provision import prepare_project
from nvflare.lighter.provisioner import Provisioner
from nvflare.lighter.spec import Builder
from nvflare.lighter.utils import load_crt_bytes, load_yaml, verify_cert
from nvflare.tool.cert.cert_constants import ADMIN_CERT_TYPES, KIT_TYPE_TO_ROLE, VALID_CERT_TYPES
from nvflare.tool.cert.fingerprint import cert_fingerprint_sha256, normalize_sha256_fingerprint
from nvflare.tool.cli_output import (
    is_json_mode,
    output_error,
    output_error_message,
    output_ok,
    output_usage_error,
    print_human,
    prompt_yn,
)
from nvflare.tool.cli_schema import handle_schema_flag

_VALID_SCHEMES = {"grpc", "tcp", "http"}
_ADMIN_ROLES = set(ADMIN_CERT_TYPES)
_VALID_CERT_TYPES = set(VALID_CERT_TYPES)
_KIT_TYPE_TO_ROLE = KIT_TYPE_TO_ROLE
_DUMMY_SERVER_NAME = "__nvflare_dummy_server__"
_DUMMY_ORG = "myorg"
_MAX_ZIP_MEMBER_SIZE = 10 * 1024 * 1024
_SAFE_PROJECT_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_REQUEST_ID_PATTERN = re.compile(
    r"(?:[0-9a-fA-F]{32}|[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"
)
_SIGNED_KIND_TO_CERT_TYPE = {
    "site": "client",
    "server": "server",
}
_SIGNED_USER_CERT_TYPES = _VALID_CERT_TYPES - {"client", "server"}


def _reject_invalid_project_name(project_name: str, *, code: str, hint: str) -> None:
    output_error_message(
        code,
        f"Invalid project name: {project_name!r}.",
        hint,
        None,
        exit_code=4,
    )


def _validate_safe_project_name(project_name: str, *, code: str = "INVALID_PROJECT_NAME") -> bool:
    hint = "Project names must match [A-Za-z0-9][A-Za-z0-9._-]* and must not contain path separators."
    if not project_name or not isinstance(project_name, str) or not project_name.strip():
        _reject_invalid_project_name(project_name, code=code, hint=hint)
        return False
    if len(project_name) > 64:
        _reject_invalid_project_name(project_name, code=code, hint="Project names must be 64 characters or fewer.")
        return False
    if os.sep in project_name or (os.altsep and os.altsep in project_name) or project_name.startswith("."):
        _reject_invalid_project_name(project_name, code=code, hint=hint)
        return False
    if not _SAFE_PROJECT_NAME_PATTERN.fullmatch(project_name):
        _reject_invalid_project_name(project_name, code=code, hint=hint)
        return False
    return True


def _validate_request_id(request_id: str, *, code: str = "INVALID_SIGNED_ZIP") -> bool:
    if not isinstance(request_id, str) or not _REQUEST_ID_PATTERN.fullmatch(request_id):
        output_error_message(
            code,
            f"Invalid request_id: {request_id!r}.",
            "Signed zip request_id must be a UUID hex string.",
            None,
            exit_code=4,
        )
        return False
    return True


def _validate_org_name(org: str, *, code: str = "INVALID_SIGNED_ZIP") -> bool:
    if not isinstance(org, str):
        invalid, reason = True, "org must be a string"
    else:
        invalid, reason = name_check(org, "org")
    if invalid:
        output_error_message(
            code,
            f"Invalid org name: {org!r}.",
            reason,
            None,
            exit_code=4,
        )
        return False
    return True


def _validate_signed_kind_cert_type(kind: str, cert_type: str) -> bool:
    if kind in _SIGNED_KIND_TO_CERT_TYPE:
        expected = _SIGNED_KIND_TO_CERT_TYPE[kind]
        if cert_type != expected:
            output_error_message(
                "INVALID_SIGNED_ZIP",
                f"Signed zip kind {kind!r} requires cert_type {expected!r}.",
                "Ask the Project Admin to regenerate the signed zip.",
                None,
                exit_code=4,
            )
            return False
        return True
    if kind == "user" and cert_type in _SIGNED_USER_CERT_TYPES:
        return True
    output_error_message(
        "INVALID_SIGNED_ZIP",
        f"Invalid signed zip kind/cert_type combination: kind={kind!r}, cert_type={cert_type!r}.",
        "Ask the Project Admin to regenerate the signed zip.",
        None,
        exit_code=4,
    )
    return False


def _validate_participant_name(name: str, kit_type: str, *, code: str = "INVALID_SIGNED_ZIP") -> bool:
    if kit_type == "client":
        entity_type = "client"
    elif kit_type == "server":
        entity_type = "server"
    elif kit_type in _ADMIN_ROLES:
        entity_type = "admin"
    else:
        output_error_message(
            code,
            f"Invalid participant type: {kit_type!r}.",
            "Ask the Project Admin to regenerate the signed zip.",
            None,
            exit_code=4,
        )
        return False

    if not isinstance(name, str):
        invalid, reason = True, f"name must be a string for entity_type={entity_type}"
    else:
        invalid, reason = name_check(name, entity_type)
    if invalid:
        output_error_message(
            code,
            f"Invalid participant name: {name!r}.",
            reason,
            None,
            exit_code=4,
        )
        return False
    return True


def _project_dir_under_workspace(workspace: str, project_name: str) -> Optional[str]:
    if not _validate_safe_project_name(project_name):
        return None
    workspace_abs = os.path.abspath(workspace)
    project_dir = os.path.abspath(os.path.join(workspace_abs, project_name))
    try:
        is_inside = os.path.commonpath([workspace_abs, project_dir]) == workspace_abs
    except ValueError:
        is_inside = False
    if not is_inside:
        output_error_message(
            "INVALID_PROJECT_NAME",
            f"Project path escapes workspace: {project_name!r}.",
            "Use a path-safe project name.",
            None,
            exit_code=4,
        )
        return None
    return project_dir


def _read_cert_type_from_cert(cert) -> str:
    """Return the kit type embedded in cert's UNSTRUCTURED_NAME, or '' if absent."""
    attrs = cert.subject.get_attributes_for_oid(NameOID.UNSTRUCTURED_NAME)
    return attrs[0].value if attrs else ""


def _read_cert_common_name(cert) -> str:
    attrs = cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)
    return attrs[0].value if attrs else ""


def _read_cert_org(cert) -> str:
    attrs = cert.subject.get_attributes_for_oid(NameOID.ORGANIZATION_NAME)
    return attrs[0].value if attrs else ""


def _write_file_nofollow(path: str, content: bytes, mode: int = 0o644) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, mode)
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(fd, mode)
    except Exception:
        os.close(fd)
        try:
            os.unlink(path)
        except OSError:
            pass
        raise
    with os.fdopen(fd, "wb") as f:
        try:
            f.write(content)
        except Exception:
            try:
                os.unlink(path)
            except OSError:
                pass
            raise


def _read_file_nofollow(path: str, max_size: int = _MAX_ZIP_MEMBER_SIZE) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    with os.fdopen(fd, "rb") as f:
        content = f.read(max_size + 1)
    if len(content) > max_size:
        raise ValueError(f"file exceeds maximum size: {path}")
    return content


def _read_json_file_nofollow(path: str) -> dict:
    return json.loads(_read_file_nofollow(path).decode("utf-8"))


def _load_crt_nofollow(path: str):
    return load_crt_bytes(_read_file_nofollow(path))


def _load_private_key_nofollow(path: str):
    return serialization.load_pem_private_key(_read_file_nofollow(path), password=None)


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
        built_count = 0
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

            _write_file_nofollow(cert_dst, _read_file_nofollow(cert_path), mode=0o644)
            _write_file_nofollow(key_dst, _read_file_nofollow(key_path), mode=0o600)

            rootca_dst = os.path.join(kit_dir, "rootCA.pem")
            _write_file_nofollow(rootca_dst, _read_file_nofollow(self.rootca_path), mode=0o644)
            built_count += 1

        if built_count == 0:
            raise ValueError(
                f"no participant kit was built; target_name={self.target_name!r} did not match any project participant"
            )


def _parse_endpoint(endpoint: str) -> tuple:
    """Return (scheme, host, port: int). Raises ValueError on invalid format."""
    parsed = urlparse(endpoint)
    if parsed.scheme not in _VALID_SCHEMES or not parsed.hostname or not parsed.port:
        raise ValueError(f"Invalid endpoint URI: {endpoint!r}")
    if not (1 <= parsed.port <= 65535):
        raise ValueError(f"Invalid endpoint URI: {endpoint!r}: port must be 1-65535")
    return parsed.scheme, parsed.hostname, parsed.port


def _discover_name_from_dir(work_dir: str) -> str:
    """Find the single *.key file in work_dir and return its stem as the participant name."""
    keys = [f for f in os.listdir(work_dir) if f.endswith(".key") and not f.startswith(".")]
    if len(keys) == 0:
        output_error(
            "KEY_NOT_FOUND",
            exit_code=1,
            path=work_dir,
            detail="No *.key file found. Use the private key generated by 'nvflare cert request'.",
        )
    if len(keys) > 1:
        output_error_message(
            "AMBIGUOUS_KEY",
            f"Multiple *.key files found in {work_dir}: {keys}",
            "Provide exactly one non-hidden participant key in the material directory.",
            None,
            exit_code=1,
        )
    return keys[0][:-4]  # strip ".key"


def _latest_prod_dir(workspace: str, project_name: str):
    """Return the path to the highest-numbered existing prod_NN under workspace/project_name/.

    Returns None if no prod_NN directories exist.
    """
    project_dir = _project_dir_under_workspace(workspace, project_name)
    if not project_dir:
        return None
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
        return None, None
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
        return None, None

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
            return None, None
        participant = {k: v for k, v in project_dict.items() if k not in {"name", "org", "type", "project_name"}}
        participant.update({"name": project_dict.get("name"), "org": project_dict.get("org")})
        if p_type in _ADMIN_ROLES:
            participant.update({"type": "admin", "role": _KIT_TYPE_TO_ROLE[p_type]})
        else:
            participant.update({"type": p_type})
        project_dict = {
            PropKey.API_VERSION: 3,
            PropKey.NAME: project_dict.get("project") or project_dict.get("project_name", "project"),
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
        return None, None
    if not _validate_safe_project_name(project.name, code="INVALID_PROJECT_FILE"):
        return None, None

    if project.get_relays():
        output_error_message(
            "UNSUPPORTED_TOPOLOGY",
            "Relay participants found in project file; hierarchical FL is not supported by 'nvflare package'.",
            "Use 'nvflare provision' for relay topologies.",
            None,
            exit_code=4,
        )
        return None, None

    custom_builders = prepare_builders(project_dict)
    return project, custom_builders


def _participant_kit_type(participant) -> str:
    if participant.type == ParticipantType.SERVER:
        return "server"
    if participant.type == ParticipantType.CLIENT:
        return "client"
    if participant.type == ParticipantType.ADMIN:
        role = participant.get_prop(PropKey.ROLE, AdminRole.LEAD)
        return role if role in _ADMIN_ROLES else AdminRole.LEAD
    return participant.type


def _validate_cert_material(cert_path: str, key_path: str, rootca_path: str, *, validate_key_match: bool = False):
    try:
        cert = _load_crt_nofollow(cert_path)
        ca_cert = _load_crt_nofollow(rootca_path)
        verify_cert(cert, ca_cert.public_key())
    except Exception as e:
        output_error("CERT_CHAIN_INVALID", exit_code=1, cert=cert_path, rootca=rootca_path, detail=str(e))
        return None

    try:
        expiry = cert.not_valid_after_utc
        now = datetime.datetime.now(datetime.timezone.utc)
    except AttributeError:
        expiry = cert.not_valid_after
        now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    if expiry < now:
        output_error("CERT_EXPIRED", exit_code=1, cert=cert_path, expiry=expiry.isoformat())
        return None

    if validate_key_match:
        cert_public = None
        key_public = None
        try:
            private_key = _load_private_key_nofollow(key_path)
            cert_public = cert.public_key().public_bytes(
                serialization.Encoding.PEM,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            )
            key_public = private_key.public_key().public_bytes(
                serialization.Encoding.PEM,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        except Exception as e:
            output_error_message(
                "KEY_INVALID",
                f"Failed to load private key '{key_path}': {e}.",
                "Provide the private key generated with the original request.",
                None,
                exit_code=1,
            )
            return None
        if cert_public != key_public:
            output_error_message(
                "KEY_CERT_MISMATCH",
                "The local private key does not match the signed certificate.",
                "Use the request directory that contains the key generated for this signed zip.",
                None,
                exit_code=1,
            )
            return None
    return cert


def _build_package_builders(custom_builders, cert_builder, scheme):
    has_cert = False
    all_builders = []
    for b in custom_builders or []:
        if isinstance(b, SignatureBuilder):
            continue
        if isinstance(b, CertBuilder):
            all_builders.append(cert_builder)
            has_cert = True
        elif isinstance(b, StaticFileBuilder):
            # The signed-zip package endpoint is supplied at package time. Keep
            # any custom StaticFileBuilder instance from the project file, but
            # make its connection scheme match -e/--endpoint.
            b.scheme = scheme
            all_builders.append(b)
        else:
            all_builders.append(b)

    if not has_cert:
        ws_pos = next((i for i, b in enumerate(all_builders) if isinstance(b, WorkspaceBuilder)), -1)
        all_builders.insert(ws_pos + 1, cert_builder)

    if not any(isinstance(b, WorkspaceBuilder) for b in all_builders):
        all_builders.insert(0, WorkspaceBuilder())

    if not any(isinstance(b, StaticFileBuilder) for b in all_builders):
        cert_pos = next((i for i, b in enumerate(all_builders) if isinstance(b, PrebuiltCertBuilder)), None)
        if cert_pos is None:
            raise RuntimeError("PrebuiltCertBuilder missing from builder list; this is a bug")
        all_builders.insert(cert_pos + 1, StaticFileBuilder(scheme=scheme))

    return all_builders


def _make_package_result(prod_dir: str, name: str, kit_type: str, endpoint: str):
    is_server = kit_type == "server"
    is_admin = kit_type in _ADMIN_ROLES
    cert_filename = "server.crt" if is_server else "client.crt"
    key_filename = "server.key" if is_server else "client.key"
    startup_script = "fl_admin.sh" if is_admin else "start.sh"
    output_dir = os.path.join(prod_dir, name)
    return {
        "output_dir": output_dir,
        "name": name,
        "type": kit_type,
        "endpoint": endpoint or "(not set)",
        "cert": os.path.join(output_dir, "startup", cert_filename),
        "key": os.path.join(output_dir, "startup", key_filename) + "  (permissions: 0600)",
        "rootca": os.path.join(output_dir, "startup", "rootCA.pem"),
        "transfer_dir": os.path.join(output_dir, "transfer"),
        "next_step": f"cd {output_dir} && ./startup/{startup_script}",
    }


def _build_selected_participant_package(
    *,
    args,
    scheme: str,
    host: str,
    port: int,
    name: str,
    org: str,
    kit_type: str,
    cert_path: str,
    key_path: str,
    rootca_path: str,
    project_name: str,
    participant_props: dict = None,
    custom_builders=None,
):
    """Build exactly one participant through the existing provisioner/builders."""
    if name == _DUMMY_SERVER_NAME:
        output_error_message(
            "INVALID_ARGS",
            f"Participant name {_DUMMY_SERVER_NAME!r} is reserved and cannot be used.",
            "Choose a different name for this participant.",
            None,
            exit_code=4,
        )
        return 1

    if kit_type in _ADMIN_ROLES and name_check(name, "admin")[0]:
        output_error_message(
            "INVALID_ARGS",
            f"Admin name must be an email address (got {name!r}).",
            "Use an email-format name, e.g. alice@myorg.com",
            None,
            exit_code=4,
        )
        return 1

    if kit_type != "server" and name == host:
        output_error_message(
            "INVALID_ARGS",
            f"Participant name {name!r} collides with the server endpoint hostname.",
            "Use a participant name that is distinct from the server hostname in --endpoint.",
            None,
            exit_code=4,
        )
        return 1

    workspace = os.path.abspath(getattr(args, "workspace", None) or "workspace")
    admin_port = args.admin_port if args.admin_port is not None else port
    project_name = project_name or getattr(args, "project_name", None) or "project"
    if _project_dir_under_workspace(workspace, project_name) is None:
        return 1

    latest_prod = _latest_prod_dir(workspace, project_name)
    if latest_prod:
        existing_path = os.path.join(latest_prod, name)
        if os.path.exists(existing_path) and not args.force:
            output_error("OUTPUT_DIR_EXISTS", exit_code=1, path=existing_path)
            return 1

    is_server = kit_type == "server"
    server_props = {
        PropKey.FED_LEARN_PORT: port,
        PropKey.ADMIN_PORT: admin_port,
    }

    project = Project(name=project_name, description="")
    participant_props = dict(participant_props or {})
    if is_server:
        server_props_with_yaml = dict(participant_props)
        server_props_with_yaml.update(server_props)
        project.set_server(name, org or _DUMMY_ORG, server_props_with_yaml)
    else:
        project.set_server(host, _DUMMY_ORG, server_props)
        if kit_type == "client":
            project.add_client(name, org or _DUMMY_ORG, participant_props)
        else:
            participant_props[PropKey.ROLE] = _KIT_TYPE_TO_ROLE[kit_type]
            project.add_admin(name, org or _DUMMY_ORG, participant_props)

    cert_builder = PrebuiltCertBuilder(
        cert_path=cert_path,
        key_path=key_path,
        rootca_path=rootca_path,
        target_name=name,
    )
    provisioner = Provisioner(
        root_dir=workspace,
        builders=_build_package_builders(custom_builders, cert_builder, scheme),
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
        return 1

    prod_dir = ctx[CtxKey.CURRENT_PROD_DIR]
    if not is_server:
        shutil.rmtree(os.path.join(prod_dir, host), ignore_errors=True)

    return _make_package_result(prod_dir, name, kit_type, args.endpoint)


def _handle_package_yaml_mode(args, scheme, host, port):
    """Build kits for all participants defined in a project yaml file.

    Internal helper for project-file mode. The cert/key for each participant is
    resolved from a material directory by participant name; rootCA is read from
    the same directory.
    """
    project_from_yaml, custom_builders = _load_project_from_file(args.project_file)
    if project_from_yaml is None:
        return 1

    # A material directory is required in internal yaml mode (no single key auto-discovery).
    if not getattr(args, "dir", None):
        output_error_message(
            "INVALID_ARGS",
            "A material directory is required when using internal project-file packaging.",
            "Provide the directory containing cert files named by participant CN.",
            None,
            exit_code=4,
        )
        return 1

    rootca_path = os.path.join(args.dir, "rootCA.pem")
    if not os.path.isfile(rootca_path):
        output_error(
            "ROOTCA_NOT_FOUND",
            exit_code=1,
            path=rootca_path,
            detail=f"Place the rootCA.pem from the Project Admin into {args.dir}.",
        )
        return 1

    workspace = os.path.abspath(getattr(args, "workspace", None) or "workspace")
    project_name = getattr(args, "project_name", None) or project_from_yaml.name or "project"
    admin_port = args.admin_port if args.admin_port is not None else port

    # Collect participants to build: clients, admins (server handled separately when requested).
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

    # Include server if this is a server-only project and no type filter was given.
    if not all_participants and server and not getattr(args, "kit_type", None):
        all_participants.append(("server", server))

    # Apply internal type filter when given.
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
            "Check the project file and participant type filter.",
            None,
            exit_code=1,
        )
        return 1

    server_props = {
        PropKey.FED_LEARN_PORT: port,
        PropKey.ADMIN_PORT: admin_port,
    }

    # Validate all cert/key files up front before building anything.
    cert_map = {}
    for kit_type, participant in all_participants:
        p_name = participant.name
        cert_path = os.path.join(args.dir, f"{p_name}.crt")
        key_path = os.path.join(args.dir, f"{p_name}.key")

        if not os.path.isfile(cert_path):
            output_error(
                "CERT_NOT_FOUND",
                exit_code=1,
                path=cert_path,
                detail=f"Drop {p_name}.crt from the Project Admin into {args.dir}.",
            )
            return 1
        if not os.path.isfile(key_path):
            output_error(
                "KEY_NOT_FOUND",
                exit_code=1,
                path=key_path,
                detail=f"Use the private key generated by 'nvflare cert request' for {p_name}.",
            )
            return 1

        cert = _validate_cert_material(cert_path, key_path, rootca_path, validate_key_match=True)
        if cert is None:
            return 1

        cert_map[p_name] = (cert_path, key_path)

    # Pre-check: if any participant already exists in the latest prod dir, reject unless --force.
    latest_prod = _latest_prod_dir(workspace, project_name)
    if latest_prod:
        for _, participant in all_participants:
            existing_path = os.path.join(latest_prod, participant.name)
            if os.path.exists(existing_path) and not args.force:
                output_error("OUTPUT_DIR_EXISTS", exit_code=1, path=existing_path)
                return 1

    # Build all participants in a single provisioner call so they land in the same prod_NN.
    has_server = any(kt == "server" for kt, _ in all_participants)
    project = Project(name=project_name, description="")
    if has_server:
        server_participant = next(p for kt, p in all_participants if kt == "server")
        server_props_with_yaml = dict(server_participant.props)
        server_props_with_yaml.update(server_props)
        project.set_server(server_participant.name, server_participant.org, server_props_with_yaml)
    else:
        project.set_server(host, _DUMMY_ORG, server_props)

    for kit_type, participant in all_participants:
        if kit_type == "server":
            continue
        participant_props = dict(participant.props)
        if kit_type == "client":
            project.add_client(participant.name, participant.org, participant_props)
        else:
            project.add_admin(participant.name, participant.org, participant_props)

    cert_builder = PrebuiltCertBuilder(rootca_path=rootca_path, cert_map=cert_map)

    # Build the builder list from site.yaml, replacing any CertBuilder with PrebuiltCertBuilder.
    # WorkspaceBuilder and StaticFileBuilder from the YAML are honored as-is.
    # Defaults are injected only when absent from the YAML.
    has_cert = False
    all_builders = []
    for b in custom_builders:
        if isinstance(b, CertBuilder):
            all_builders.append(cert_builder)
            has_cert = True
        else:
            all_builders.append(b)

    if not has_cert:
        # Inject after WorkspaceBuilder, or at the front if none.
        ws_pos = next((i for i, b in enumerate(all_builders) if isinstance(b, WorkspaceBuilder)), -1)
        all_builders.insert(ws_pos + 1, cert_builder)

    if not any(isinstance(b, WorkspaceBuilder) for b in all_builders):
        all_builders.insert(0, WorkspaceBuilder())

    if not any(isinstance(b, StaticFileBuilder) for b in all_builders):
        cert_pos = next((i for i, b in enumerate(all_builders) if isinstance(b, PrebuiltCertBuilder)), None)
        if cert_pos is None:
            raise RuntimeError("PrebuiltCertBuilder missing from builder list; this is a bug")
        all_builders.insert(cert_pos + 1, StaticFileBuilder(scheme=scheme))

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
        return 1

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


def _safe_zip_names(zf: zipfile.ZipFile, zip_path: str):
    names = []
    seen = set()
    for info in zf.infolist():
        name = info.filename
        norm = posixpath.normpath(name)
        mode = info.external_attr >> 16
        if (
            not name
            or name == "."
            or os.path.isabs(name)
            or "\\" in name
            or norm != name
            or norm.startswith("..")
            or posixpath.basename(name) != name
            or name in seen
            or info.is_dir()
            or info.file_size > _MAX_ZIP_MEMBER_SIZE
            or stat.S_IFMT(mode) in {stat.S_IFDIR, stat.S_IFLNK}
        ):
            output_error_message(
                "INVALID_SIGNED_ZIP",
                f"Invalid path in signed zip {zip_path}: {name!r}.",
                "Signed zips must contain safe files at the archive root only.",
                None,
                exit_code=4,
            )
            continue
        if name.lower().endswith(".key"):
            output_error_message(
                "INVALID_SIGNED_ZIP",
                "Signed zip must not contain private key material.",
                "Use the local request directory that contains the private key.",
                None,
                exit_code=4,
            )
            continue
        seen.add(name)
        names.append(name)
    return names


def _read_zip_member_limited(zf: zipfile.ZipFile, name: str, zip_path: str) -> bytes:
    try:
        with zf.open(name) as member_file:
            content = member_file.read(_MAX_ZIP_MEMBER_SIZE + 1)
    except Exception as e:
        output_error_message(
            "INVALID_SIGNED_ZIP",
            f"Failed to read {name} from signed zip {zip_path}: {e}.",
            "Ask the Project Admin to regenerate the signed zip.",
            None,
            exit_code=4,
        )
        return b""
    if len(content) > _MAX_ZIP_MEMBER_SIZE:
        output_error_message(
            "INVALID_SIGNED_ZIP",
            f"Signed zip member exceeds size limit: {name}.",
            "Ask the Project Admin to regenerate the signed zip.",
            None,
            exit_code=4,
        )
        return b""
    return content


def _decode_zip_json(content: bytes, name: str, zip_path: str):
    try:
        return json.loads(content.decode("utf-8"))
    except Exception as e:
        output_error_message(
            "INVALID_SIGNED_ZIP",
            f"Failed to read {name} from signed zip {zip_path}: {e}.",
            "Ask the Project Admin to regenerate the signed zip.",
            None,
            exit_code=4,
        )
        raise


def _read_zip_json(zf: zipfile.ZipFile, name: str, zip_path: str):
    return _decode_zip_json(_read_zip_member_limited(zf, name, zip_path), name, zip_path)


def _decode_zip_yaml(content: bytes, name: str, zip_path: str):
    data = None
    try:
        data = yaml.safe_load(content.decode("utf-8"))
    except Exception as e:
        output_error_message(
            "INVALID_SIGNED_ZIP",
            f"Failed to read {name} from signed zip {zip_path}: {e}.",
            "Ask the Project Admin to regenerate the signed zip.",
            None,
            exit_code=4,
        )
        raise
    if not isinstance(data, dict):
        output_error_message(
            "INVALID_SIGNED_ZIP",
            f"{name} in signed zip must be a mapping.",
            "Ask the Project Admin to regenerate the signed zip.",
            None,
            exit_code=4,
        )
        raise ValueError(f"{name} in signed zip must be a mapping")
    return data


def _read_zip_yaml(zf: zipfile.ZipFile, name: str, zip_path: str):
    return _decode_zip_yaml(_read_zip_member_limited(zf, name, zip_path), name, zip_path)


def _hash_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _cert_public_key_sha256(cert) -> str:
    public_key_der = cert.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return _hash_bytes(public_key_der)


def _normalize_hash(value: str) -> str:
    if not isinstance(value, str):
        return ""
    return value.removeprefix("sha256:").lower()


def _verify_rootca_fingerprint_options(args, actual_fingerprint: str) -> bool:
    expected = getattr(args, "expected_rootca_fingerprint", None)
    if expected:
        normalized_expected = normalize_sha256_fingerprint(expected)
        if not normalized_expected:
            output_error_message(
                "INVALID_ROOTCA_FINGERPRINT",
                f"Invalid expected root CA SHA256 fingerprint: {expected!r}.",
                "Use SHA256:AA:BB:... or OpenSSL output such as sha256 Fingerprint=AA:BB:...",
                None,
                exit_code=4,
            )
            return False
        if normalized_expected != actual_fingerprint:
            output_error_message(
                "ROOTCA_FINGERPRINT_MISMATCH",
                "Root CA SHA256 fingerprint does not match the expected out-of-band value.",
                (
                    f"Expected {normalized_expected}; actual {actual_fingerprint}. "
                    "Verify that the signed zip came from the intended Project Admin."
                ),
                None,
                exit_code=4,
            )
            return False

    if getattr(args, "confirm_rootca", False):
        if is_json_mode():
            output_error_message(
                "INVALID_ARGS",
                "--confirm-rootca cannot be used with JSON output.",
                "Use --expected-rootca-fingerprint for non-interactive verification.",
                None,
                exit_code=4,
            )
            return False
        if not sys.stdin.isatty():
            output_error_message(
                "INVALID_ARGS",
                "--confirm-rootca requires an interactive terminal.",
                "Use --expected-rootca-fingerprint for non-interactive verification.",
                None,
                exit_code=4,
            )
            return False
        print_human("Root CA SHA256 fingerprint from signed zip:")
        print_human(actual_fingerprint)
        if not prompt_yn("Fingerprint verified out-of-band?"):
            print_human("Cancelled.")
            return False

    return True


def _expected_hash(meta: dict, *names):
    hashes = meta.get("hashes") if isinstance(meta.get("hashes"), dict) else {}
    for name in names:
        value = hashes.get(name) or hashes.get(f"{name}_sha256") or meta.get(f"{name}_sha256")
        if value:
            return _normalize_hash(value)
    return ""


def _validate_signed_metadata(signed_meta: dict, site_meta: dict, cert_name: str) -> bool:
    if not isinstance(signed_meta, dict):
        output_error_message(
            "INVALID_SIGNED_ZIP",
            "signed.json in signed zip must be a mapping.",
            "Ask the Project Admin to regenerate the signed zip.",
            None,
            exit_code=4,
        )
        return False
    if signed_meta.get("artifact_type") != "nvflare.cert.signed" or signed_meta.get("schema_version") != "1":
        output_error_message(
            "INVALID_SIGNED_ZIP",
            "Signed zip metadata has an unsupported artifact type or schema version.",
            "Ask the Project Admin to regenerate the signed zip.",
            None,
            exit_code=4,
        )
        return False

    required = ("request_id", "project", "name", "org", "kind", "cert_type", "cert_file", "rootca_file")
    missing = [field for field in required if not signed_meta.get(field)]
    if missing:
        output_error_message(
            "INVALID_SIGNED_ZIP",
            f"Signed zip metadata is missing required field(s): {', '.join(missing)}.",
            "Ask the Project Admin to regenerate the signed zip.",
            None,
            exit_code=4,
        )
        return False
    if not _validate_request_id(signed_meta["request_id"]):
        return False
    if not _validate_safe_project_name(signed_meta["project"], code="INVALID_SIGNED_ZIP"):
        return False
    if not _validate_org_name(signed_meta["org"], code="INVALID_SIGNED_ZIP"):
        return False
    cert_type = signed_meta["cert_type"]
    if cert_type not in _VALID_CERT_TYPES:
        output_error_message(
            "INVALID_SIGNED_ZIP",
            f"Invalid signed zip cert_type: {cert_type!r}.",
            "Ask the Project Admin to regenerate the signed zip.",
            None,
            exit_code=4,
        )
        return False
    if not _validate_signed_kind_cert_type(signed_meta["kind"], cert_type):
        return False
    if not _validate_participant_name(signed_meta["name"], cert_type, code="INVALID_SIGNED_ZIP"):
        return False
    if signed_meta["cert_file"] != cert_name or signed_meta["rootca_file"] != "rootCA.pem":
        output_error_message(
            "INVALID_SIGNED_ZIP",
            "Signed zip metadata file names do not match zip contents.",
            "Ask the Project Admin to regenerate the signed zip.",
            None,
            exit_code=4,
        )
        return False

    for meta_field, site_field in (("name", "name"), ("org", "org"), ("cert_type", "type"), ("project", "project")):
        if site_meta.get(site_field) != signed_meta.get(meta_field):
            output_error_message(
                "INVALID_SIGNED_ZIP",
                f"site.yaml field '{site_field}' does not match signed metadata.",
                "Ask the Project Admin to regenerate the signed zip.",
                None,
                exit_code=4,
            )
            return False
    if site_meta.get("kind") != signed_meta.get("kind"):
        output_error_message(
            "INVALID_SIGNED_ZIP",
            "site.yaml field 'kind' does not match signed metadata.",
            "Ask the Project Admin to regenerate the signed zip.",
            None,
            exit_code=4,
        )
        return False
    if (site_meta.get("cert_role") or None) != (signed_meta.get("cert_role") or None):
        output_error_message(
            "INVALID_SIGNED_ZIP",
            "site.yaml field 'cert_role' does not match signed metadata.",
            "Ask the Project Admin to regenerate the signed zip.",
            None,
            exit_code=4,
        )
        return False

    hashes = signed_meta.get("hashes")
    # The signed.zip schema requires canonical hash field names produced by cert approve.
    # Later hash verification accepts aliases only as a tolerance layer for lookup.
    required_hashes = ("csr_sha256", "site_yaml_sha256", "certificate_sha256", "rootca_sha256", "public_key_sha256")
    if not isinstance(hashes, dict) or any(not hashes.get(name) for name in required_hashes):
        output_error_message(
            "INVALID_SIGNED_ZIP",
            "Signed zip metadata is missing required hashes.",
            "Ask the Project Admin to regenerate the signed zip.",
            None,
            exit_code=4,
        )
        return False
    return True


def _validate_signed_hashes(signed_meta: dict, file_contents: dict) -> bool:
    # Hash aliases are resolved as hashes[alias] or hashes[f"{alias}_sha256"];
    # this keeps canonical keys like certificate_sha256 readable while allowing
    # older short names such as cert_sha256 during schema tightening.
    checks = [
        (("site_yaml", "site", "site.yaml"), "site.yaml"),
        (("cert", "certificate", "crt"), "cert"),
        (("rootca", "root_ca", "rootCA.pem"), "rootCA.pem"),
    ]
    for hash_names, content_name in checks:
        expected = _expected_hash(signed_meta, *hash_names)
        if not expected:
            output_error_message(
                "INVALID_SIGNED_ZIP",
                f"Missing required hash for {content_name} in signed zip metadata.",
                "Ask the Project Admin to regenerate the signed zip.",
                None,
                exit_code=4,
            )
            return False
        actual = _hash_bytes(file_contents[content_name])
        if actual != expected:
            output_error_message(
                "INVALID_SIGNED_ZIP",
                f"Hash mismatch for {content_name} in signed zip.",
                "Ask the Project Admin to regenerate the signed zip.",
                None,
                exit_code=4,
            )
            return False
    return True


def _validate_signed_public_key_hash(signed_meta: dict, cert) -> bool:
    expected = _expected_hash(signed_meta, "public_key")
    if not expected:
        output_error_message(
            "INVALID_SIGNED_ZIP",
            "Missing required public key hash in signed zip metadata.",
            "Ask the Project Admin to regenerate the signed zip.",
            None,
            exit_code=4,
        )
        return False
    public_key_der = cert.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    actual = _hash_bytes(public_key_der)
    if actual != expected:
        output_error_message(
            "INVALID_SIGNED_ZIP",
            "Public key hash mismatch for signed certificate.",
            "Ask the Project Admin to regenerate the signed zip.",
            None,
            exit_code=4,
        )
        return False
    return True


def _signed_identity_from_metadata(signed_meta: dict, site_meta: dict, cert):
    participant_meta = signed_meta.get("participant") if isinstance(signed_meta.get("participant"), dict) else {}
    name = (
        signed_meta.get("name")
        or signed_meta.get("subject_cn")
        or participant_meta.get("name")
        or site_meta.get("name")
        or _read_cert_common_name(cert)
    )
    org = signed_meta.get("org") or participant_meta.get("org") or site_meta.get("org") or _read_cert_org(cert)
    cert_type = (
        signed_meta.get("cert_type")
        or signed_meta.get("type")
        or participant_meta.get("cert_type")
        or participant_meta.get("type")
        or site_meta.get("cert_type")
        or site_meta.get("type")
    )
    project_name = (
        signed_meta.get("project")
        or signed_meta.get("project_name")
        or participant_meta.get("project")
        or participant_meta.get("project_name")
        or site_meta.get("project")
        or site_meta.get("project_name")
    )
    request_id = signed_meta.get("request_id") or participant_meta.get("request_id") or site_meta.get("request_id")
    return {
        "name": name,
        "org": org,
        "cert_type": cert_type,
        "project_name": project_name,
        "request_id": request_id,
    }


def _load_signed_zip(input_path: str):
    signed_meta = None
    site_meta = None
    cert_name = ""
    file_contents = {}
    try:
        with zipfile.ZipFile(input_path, "r") as zf:
            names = set(_safe_zip_names(zf, input_path))
            required = {"signed.json", "site.yaml", "rootCA.pem"}
            missing = required - names
            if missing:
                output_error_message(
                    "INVALID_SIGNED_ZIP",
                    f"Signed zip is missing required file(s): {', '.join(sorted(missing))}.",
                    "Ask the Project Admin to regenerate the signed zip.",
                    None,
                    exit_code=4,
                )
                return signed_meta, site_meta, cert_name, file_contents
            cert_names = sorted(n for n in names if n.endswith(".crt"))
            if len(cert_names) != 1:
                output_error_message(
                    "INVALID_SIGNED_ZIP",
                    "Signed zip must contain exactly one signed certificate.",
                    "Ask the Project Admin to regenerate the signed zip.",
                    None,
                    exit_code=4,
                )
                return signed_meta, site_meta, cert_name, file_contents
            cert_name = cert_names[0]
            expected = required | {cert_name}
            if names != expected:
                extra = ", ".join(sorted(names - expected))
                output_error_message(
                    "INVALID_SIGNED_ZIP",
                    f"Signed zip contains unexpected file(s): {extra}.",
                    "Ask the Project Admin to regenerate the signed zip.",
                    None,
                    exit_code=4,
                )
                return signed_meta, site_meta, cert_name, file_contents
            file_contents = {
                "signed.json": _read_zip_member_limited(zf, "signed.json", input_path),
                "site.yaml": _read_zip_member_limited(zf, "site.yaml", input_path),
                "rootCA.pem": _read_zip_member_limited(zf, "rootCA.pem", input_path),
                "cert": _read_zip_member_limited(zf, cert_name, input_path),
            }
            signed_meta = _decode_zip_json(file_contents["signed.json"], "signed.json", input_path)
            site_meta = _decode_zip_yaml(file_contents["site.yaml"], "site.yaml", input_path)
            if not _validate_signed_metadata(signed_meta, site_meta, cert_name):
                return None, None, None, {}
            if not _validate_signed_hashes(signed_meta, file_contents):
                return None, None, None, {}
    except zipfile.BadZipFile:
        output_error_message(
            "INVALID_SIGNED_ZIP",
            f"Not a valid signed zip: {input_path}.",
            "Provide the .signed.zip returned by 'nvflare cert approve'.",
            None,
            exit_code=4,
        )
        return None, None, None, {}
    except FileNotFoundError:
        output_error("SIGNED_ZIP_NOT_FOUND", exit_code=1, path=input_path)
        return None, None, None, {}
    except Exception as e:
        output_error_message(
            "INVALID_SIGNED_ZIP",
            f"Failed to read signed zip {input_path}: {e}.",
            "Provide the .signed.zip returned by 'nvflare cert approve'.",
            None,
            exit_code=4,
        )
        return None, None, None, {}

    return signed_meta, site_meta, cert_name, file_contents


def _audit_request_dir(request_id: str):
    if not request_id:
        return None
    base = os.path.expanduser(os.path.join("~", ".nvflare", "cert_requests"))
    candidates = [
        os.path.join(base, request_id, "audit.json"),
        os.path.join(base, request_id, "request.json"),
        os.path.join(base, f"{request_id}.json"),
    ]
    for path in candidates:
        if not os.path.isfile(path):
            continue
        try:
            data = _read_json_file_nofollow(path)
        except Exception:
            continue
        request_data = data.get("request") if isinstance(data.get("request"), dict) else {}
        for key in ("request_dir", "request_folder", "output_dir"):
            value = data.get(key)
            if value:
                request_dir = _validated_audit_request_dir(value, request_id)
                if request_dir:
                    return request_dir
            value = request_data.get(key)
            if value:
                request_dir = _validated_audit_request_dir(value, request_id)
                if request_dir:
                    return request_dir
        private_key_path = data.get("private_key_path") or data.get("key_path")
        if private_key_path:
            request_dir = _validated_audit_request_dir(os.path.dirname(private_key_path), request_id)
            if request_dir:
                return request_dir
    return None


def _validated_audit_request_dir(value: str, request_id: str):
    if not isinstance(value, str) or not value.strip():
        return None
    request_dir = os.path.abspath(os.path.expanduser(value))
    if os.path.islink(request_dir) or os.path.realpath(request_dir) != request_dir:
        return None
    request_json_path = os.path.join(request_dir, "request.json")
    if not os.path.isfile(request_json_path):
        return None
    try:
        request_meta = _read_json_file_nofollow(request_json_path)
    except Exception:
        return None
    if not isinstance(request_meta, dict) or request_meta.get("request_id") != request_id:
        return None
    return request_dir


def _has_request_material(request_dir: str, name: str) -> bool:
    return os.path.isfile(os.path.join(request_dir, f"{name}.key")) and os.path.isfile(
        os.path.join(request_dir, "request.json")
    )


def _missing_request_material(request_dir: str, name: str) -> list:
    expected = (f"{name}.key", "request.json")
    return [filename for filename in expected if not os.path.isfile(os.path.join(request_dir, filename))]


def _resolve_request_dir(args, signed_zip_path: str, identity: dict):
    if getattr(args, "request_dir", None):
        candidate = os.path.abspath(args.request_dir)
        request_id = identity.get("request_id")
        if request_id:
            validated = _validated_audit_request_dir(candidate, request_id)
            if not validated:
                missing = _missing_request_material(candidate, identity["name"]) if os.path.isdir(candidate) else []
                if missing:
                    output_error_message(
                        "REQUEST_DIR_INCOMPLETE",
                        "The specified --request-dir is missing local request material.",
                        f"Expected file(s): {', '.join(missing)}.",
                        None,
                        exit_code=1,
                    )
                    return None
                output_error_message(
                    "REQUEST_DIR_MISMATCH",
                    "The specified --request-dir does not match the signed zip request_id.",
                    "Use the directory created by 'nvflare cert request' for this signed zip.",
                    None,
                    exit_code=4,
                )
                return None
            missing = _missing_request_material(validated, identity["name"])
            if missing:
                output_error_message(
                    "REQUEST_DIR_INCOMPLETE",
                    "The specified --request-dir is missing local request material.",
                    f"Expected file(s): {', '.join(missing)}.",
                    None,
                    exit_code=1,
                )
                return None
            return validated
        if _has_request_material(candidate, identity["name"]):
            return candidate
        if os.path.isdir(candidate):
            missing = _missing_request_material(candidate, identity["name"])
            output_error_message(
                "REQUEST_DIR_INCOMPLETE",
                "The specified --request-dir is missing local request material.",
                f"Expected file(s): {', '.join(missing)}.",
                None,
                exit_code=1,
            )
        return None

    audit_dir = _audit_request_dir(identity.get("request_id"))
    if audit_dir and _has_request_material(audit_dir, identity["name"]):
        return os.path.abspath(audit_dir)

    signed_dir = os.path.dirname(os.path.abspath(signed_zip_path))
    parent = os.path.dirname(signed_dir)
    candidates = [signed_dir, os.path.join(signed_dir, identity["name"]), os.path.join(parent, identity["name"])]
    request_id = identity.get("request_id")
    for candidate in candidates:
        if request_id:
            validated = _validated_audit_request_dir(candidate, request_id)
            if validated and _has_request_material(validated, identity["name"]):
                return validated
        elif _has_request_material(candidate, identity["name"]):
            return candidate
    return None


def _read_local_request_metadata(request_dir: str) -> dict:
    request_json_path = os.path.join(request_dir, "request.json")
    if not os.path.isfile(request_json_path):
        output_error_message(
            "REQUEST_METADATA_NOT_FOUND",
            "Local request metadata was not found.",
            "Use --request-dir to point to the original request folder created by 'nvflare cert request'.",
            None,
            exit_code=1,
            detail=f"missing {request_json_path}",
        )
        return None

    try:
        request_meta = _read_json_file_nofollow(request_json_path)
    except Exception as e:
        output_error_message(
            "REQUEST_METADATA_INVALID",
            "Local request metadata could not be read.",
            "Use the original request folder created by 'nvflare cert request'.",
            None,
            exit_code=4,
            detail=str(e),
        )
        return None
    if not isinstance(request_meta, dict):
        output_error_message(
            "REQUEST_METADATA_INVALID",
            "Local request metadata must be a JSON object.",
            "Use the original request folder created by 'nvflare cert request'.",
            None,
            exit_code=4,
        )
        return None
    return request_meta


def _request_metadata_mismatch(detail: str) -> None:
    output_error_message(
        "REQUEST_METADATA_MISMATCH",
        "Local request metadata does not match the signed zip.",
        "Use the signed zip returned for this request, or point --request-dir to the matching request folder.",
        None,
        exit_code=4,
        detail=detail,
    )


def _validate_local_request_metadata(request_meta: dict, signed_meta: dict) -> bool:
    required = ("request_id", "project", "name", "org", "kind", "cert_type", "csr_sha256", "public_key_sha256")
    missing = [field for field in required if not request_meta.get(field)]
    if missing:
        _request_metadata_mismatch(f"request.json missing required field(s): {', '.join(missing)}")
        return False

    for field in ("request_id", "project", "name", "org", "kind", "cert_type", "cert_role"):
        if (request_meta.get(field) or None) != (signed_meta.get(field) or None):
            _request_metadata_mismatch(f"field {field!r} differs")
            return False

    signed_public_key_hash = _expected_hash(signed_meta, "public_key")
    if _normalize_hash(request_meta["public_key_sha256"]) != signed_public_key_hash:
        _request_metadata_mismatch("public_key_sha256 differs")
        return False

    signed_csr_hash = _expected_hash(signed_meta, "csr")
    if signed_csr_hash and _normalize_hash(request_meta["csr_sha256"]) != signed_csr_hash:
        _request_metadata_mismatch("csr_sha256 differs")
        return False
    return True


def _write_materialized_signed_files(
    request_dir: str, identity: dict, signed_meta: dict, site_meta: dict, file_contents
):
    try:
        os.makedirs(request_dir, mode=0o700, exist_ok=True)
        _write_file_nofollow(
            os.path.join(request_dir, "signed.json"), json.dumps(signed_meta, indent=2).encode("utf-8")
        )
        _write_file_nofollow(os.path.join(request_dir, "site.yaml"), yaml.safe_dump(site_meta).encode("utf-8"))
        _write_file_nofollow(os.path.join(request_dir, f"{identity['name']}.crt"), file_contents["cert"])
        _write_file_nofollow(os.path.join(request_dir, "rootCA.pem"), file_contents["rootCA.pem"])
    except OSError as e:
        output_error("OUTPUT_DIR_NOT_WRITABLE", path=request_dir, detail=str(e))
        return False
    return True


def _find_project_participant(project, name: str):
    for participant in project.get_all_participants():
        if participant.name == name:
            return participant
    return None


def _selected_project_context(args, identity: dict, kit_type: str):
    if not getattr(args, "project_file", None):
        return None, None, identity.get("project_name") or getattr(args, "project_name", None) or "project"

    project_from_yaml, custom_builders = _load_project_from_file(args.project_file)
    if project_from_yaml is None:
        return None, None, None
    signed_project = identity.get("project_name")
    if signed_project and project_from_yaml.name and signed_project != project_from_yaml.name:
        output_error_message(
            "PROJECT_IDENTITY_CONFLICT",
            f"Signed zip project {signed_project!r} conflicts with project file project {project_from_yaml.name!r}.",
            "Use a project file for the same federation as the signed zip.",
            None,
            exit_code=4,
        )
        return None, None, None

    participant = _find_project_participant(project_from_yaml, identity["name"])
    if not participant:
        if not is_json_mode():
            print_human(
                f"Warning: {identity['name']} was not found in {args.project_file} participants; "
                "using signed zip identity and project builders."
            )
        return None, custom_builders, signed_project or getattr(args, "project_name", None) or project_from_yaml.name

    if identity.get("org") and participant.org and identity["org"] != participant.org:
        output_error_message(
            "PROJECT_IDENTITY_CONFLICT",
            f"Signed zip org {identity['org']!r} conflicts with project file org {participant.org!r}.",
            "Use a project file whose participant identity matches the signed zip.",
            None,
            exit_code=4,
        )
        return None, None, None

    participant_kit_type = _participant_kit_type(participant)
    if participant_kit_type != kit_type:
        output_error_message(
            "PROJECT_IDENTITY_CONFLICT",
            f"Signed zip type {kit_type!r} conflicts with project file type {participant_kit_type!r}.",
            "Use a project file whose participant identity matches the signed zip.",
            None,
            exit_code=4,
        )
        return None, None, None

    return participant, custom_builders, signed_project or getattr(args, "project_name", None) or project_from_yaml.name


def _handle_signed_zip_package(args, scheme, host, port):
    signed_meta, site_meta, cert_name, file_contents = _load_signed_zip(args.input)
    if not signed_meta or not site_meta or not cert_name or not file_contents:
        return 1
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_cert = os.path.join(tmp_dir, cert_name)
        temp_rootca = os.path.join(tmp_dir, "rootCA.pem")
        try:
            _write_file_nofollow(temp_cert, file_contents["cert"])
            _write_file_nofollow(temp_rootca, file_contents["rootCA.pem"])
            cert = _load_crt_nofollow(temp_cert)
            # Used below to ensure signed metadata project matches the root CA subject.
            rootca = _load_crt_nofollow(temp_rootca)
        except Exception as e:
            output_error_message(
                "INVALID_SIGNED_ZIP",
                f"Failed to load certificate from signed zip: {e}.",
                "Ask the Project Admin to regenerate the signed zip.",
                None,
                exit_code=4,
            )
            return 1
        if not _validate_signed_public_key_hash(signed_meta, cert):
            return 1

        identity = _signed_identity_from_metadata(signed_meta, site_meta, cert)
        if not identity.get("name"):
            output_error_message(
                "INVALID_SIGNED_ZIP",
                "Signed zip does not identify a participant name.",
                "Ask the Project Admin to regenerate the signed zip.",
                None,
                exit_code=4,
            )
            return 1
        if cert_name != f"{identity['name']}.crt":
            output_error_message(
                "INVALID_SIGNED_ZIP",
                f"Signed zip certificate {cert_name!r} does not match participant {identity['name']!r}.",
                "Ask the Project Admin to regenerate the signed zip.",
                None,
                exit_code=4,
            )
            return 1

        if not _validate_request_id(identity.get("request_id")):
            return 1
        if not _validate_safe_project_name(identity.get("project_name"), code="INVALID_SIGNED_ZIP"):
            return 1
        if identity.get("org"):
            if not _validate_org_name(identity["org"], code="INVALID_SIGNED_ZIP"):
                return 1

        cert_kit_type = _read_cert_type_from_cert(cert)
        if not cert_kit_type or cert_kit_type not in _VALID_CERT_TYPES:
            output_error("CERT_TYPE_UNKNOWN", exit_code=1, cert=cert_name)
            return 1
        if not _validate_participant_name(identity["name"], cert_kit_type, code="INVALID_SIGNED_ZIP"):
            return 1
        if identity.get("cert_type") and identity["cert_type"] != cert_kit_type:
            output_error_message(
                "SIGNED_ZIP_IDENTITY_CONFLICT",
                f"Signed metadata type {identity['cert_type']!r} conflicts with certificate type {cert_kit_type!r}.",
                "Ask the Project Admin to regenerate the signed zip.",
                None,
                exit_code=4,
            )
            return 1
        cert_cn = _read_cert_common_name(cert)
        if cert_cn and cert_cn != identity["name"]:
            output_error_message(
                "SIGNED_ZIP_IDENTITY_CONFLICT",
                f"Signed metadata name {identity['name']!r} conflicts with certificate CN {cert_cn!r}.",
                "Ask the Project Admin to regenerate the signed zip.",
                None,
                exit_code=4,
            )
            return 1
        cert_org = _read_cert_org(cert)
        if identity.get("org") and cert_org != identity["org"]:
            output_error_message(
                "SIGNED_ZIP_IDENTITY_CONFLICT",
                f"Signed metadata org {identity['org']!r} conflicts with certificate org {cert_org!r}.",
                "Ask the Project Admin to regenerate the signed zip.",
                None,
                exit_code=4,
            )
            return 1
        root_project = _read_cert_common_name(rootca)
        if identity.get("project_name") and root_project != identity["project_name"]:
            output_error_message(
                "SIGNED_ZIP_IDENTITY_CONFLICT",
                f"Signed metadata project {identity['project_name']!r} conflicts with root CA project {root_project!r}.",
                "Ask the Project Admin to regenerate the signed zip.",
                None,
                exit_code=4,
            )
            return 1

        rootca_fingerprint_sha256 = cert_fingerprint_sha256(rootca)
        if not _verify_rootca_fingerprint_options(args, rootca_fingerprint_sha256):
            return 1

        resolved_request_dir = _resolve_request_dir(args, args.input, identity)
        if not resolved_request_dir:
            output_error_message(
                "REQUEST_DIR_NOT_FOUND",
                "Local request directory was not found.",
                "Use --request-dir to point to the original request folder created by 'nvflare cert request'.",
                None,
                exit_code=1,
            )
            return 1
        key_path = os.path.join(resolved_request_dir, f"{identity['name']}.key")
        if not os.path.isfile(key_path):
            output_error(
                "KEY_NOT_FOUND",
                exit_code=1,
                path=key_path,
                detail="Use --request-dir to point to the local request folder that contains the private key.",
            )
            return 1
        request_meta = _read_local_request_metadata(resolved_request_dir)
        if request_meta is None:
            return 1
        if not _validate_local_request_metadata(request_meta, signed_meta):
            return 1

        cert = _validate_cert_material(temp_cert, key_path, temp_rootca, validate_key_match=True)
        if cert is None:
            return 1
        expected_public_key_hash = _expected_hash(signed_meta, "public_key")
        if expected_public_key_hash and _cert_public_key_sha256(cert) != expected_public_key_hash:
            output_error_message(
                "KEY_CERT_MISMATCH",
                "The signed certificate public key does not match signed metadata.",
                "Ask the Project Admin to regenerate the signed zip.",
                None,
                exit_code=1,
            )
            return 1
        identity["org"] = identity.get("org") or _read_cert_org(cert) or _DUMMY_ORG
        participant, custom_builders, project_name = _selected_project_context(args, identity, cert_kit_type)
        if participant is None and custom_builders is None and project_name is None:
            return 1
        participant_props = dict(participant.props) if participant else {}

        if not _write_materialized_signed_files(resolved_request_dir, identity, signed_meta, site_meta, file_contents):
            return 1
        cert_path = os.path.join(resolved_request_dir, f"{identity['name']}.crt")
        rootca_path = os.path.join(resolved_request_dir, "rootCA.pem")

        result = _build_selected_participant_package(
            args=args,
            scheme=scheme,
            host=host,
            port=port,
            name=identity["name"],
            org=identity.get("org") or _DUMMY_ORG,
            kit_type=cert_kit_type,
            cert_path=cert_path,
            key_path=key_path,
            rootca_path=rootca_path,
            project_name=project_name,
            participant_props=participant_props,
            custom_builders=custom_builders,
        )
        if not isinstance(result, dict):
            return result
        result["rootca_fingerprint_sha256"] = rootca_fingerprint_sha256
    output_ok(result)
    return 0


def handle_package(args):
    """Assemble a startup kit from locally generated key + Project Admin cert + rootCA.pem."""

    # Step 1: --schema check (before any other work)
    from nvflare.tool.package.package_cli import _PACKAGE_EXAMPLES, _package_parser

    handle_schema_flag(_package_parser, "nvflare package", _PACKAGE_EXAMPLES, sys.argv[1:])

    # Step 2: Validate required args and endpoint up front (before any file I/O)
    input_path = getattr(args, "input", None)
    has_signed_zip_input = bool(input_path)
    has_project_file = bool(getattr(args, "project_file", None))
    has_dir = bool(getattr(args, "dir", None))
    has_explicit = any(getattr(args, attr, None) for attr in ("cert", "key", "rootca"))

    if has_signed_zip_input and not str(input_path).lower().endswith(".signed.zip"):
        output_error_message(
            "INVALID_ARGS",
            f"Unsupported package input: {input_path}.",
            "Provide a .signed.zip file produced by 'nvflare cert approve'.",
            None,
            exit_code=4,
        )
        return 1

    if has_signed_zip_input and any(getattr(args, attr, None) for attr in ("name", "dir", "cert", "key", "rootca")):
        output_error_message(
            "INVALID_ARGS",
            "Positional signed zip input cannot be combined with legacy material inputs.",
            "Use --request-dir to point to the local request folder for signed zip mode.",
            None,
            exit_code=4,
        )
        return 1

    if not has_signed_zip_input and not has_dir and not has_explicit:
        output_error_message(
            "INVALID_ARGS",
            "Signed zip input is required.",
            "Use: nvflare package <identity>.signed.zip -e <endpoint>",
            None,
            exit_code=4,
        )
        return 1

    has_rootca_verify_option = bool(getattr(args, "expected_rootca_fingerprint", None)) or bool(
        getattr(args, "confirm_rootca", False)
    )
    if has_rootca_verify_option and not has_signed_zip_input:
        output_error_message(
            "INVALID_ARGS",
            "Root CA fingerprint verification options require signed zip input.",
            "Use the distributed provisioning form: nvflare package <signed.zip> -e <endpoint>.",
            None,
            exit_code=4,
        )
        return 1

    if has_project_file and getattr(args, "name", None):
        output_error_message(
            "INVALID_ARGS",
            "--project-file and participant name are mutually exclusive.",
            "Use signed zip mode for the public distributed provisioning workflow.",
            None,
            exit_code=4,
        )
        return 1

    if has_project_file and any(getattr(args, attr, None) for attr in ("cert", "key", "rootca")):
        output_error_message(
            "INVALID_ARGS",
            "--project-file and explicit material files are mutually exclusive.",
            "In internal yaml mode, cert files are discovered from the material directory by participant name.",
            None,
            exit_code=4,
        )
        return 1

    if not getattr(args, "endpoint", None):
        detail = "for this command"
        output_usage_error(
            _package_parser if not is_json_mode() else None,
            f"--endpoint is required {detail}.",
            exit_code=4,
            hint="Provide the server endpoint URI, e.g. grpc://server.example.com:8002",
        )
        return 1
    try:
        scheme, host, port = _parse_endpoint(args.endpoint)
    except ValueError:
        output_error("INVALID_ENDPOINT", exit_code=4, endpoint=args.endpoint)
        return 1

    if has_signed_zip_input:
        return _handle_signed_zip_package(args, scheme, host, port)

    # -----------------------------------------------------------------------
    # Internal YAML mode: build kits for all participants defined in the yaml file
    # using a material directory with per-participant certs named by CN.
    # -----------------------------------------------------------------------
    if has_project_file:
        return _handle_package_yaml_mode(args, scheme, host, port)

    # Step 4: Resolve internal material-directory mode vs explicit material files.
    if has_dir and has_explicit:
        output_error_message(
            "INVALID_ARGS",
            "Material directory and explicit material files are mutually exclusive.",
            "Use one material resolution mode for this internal packaging operation.",
            None,
            exit_code=4,
        )
        return 1
    if not has_dir and not has_explicit:
        output_error_message(
            "INVALID_ARGS",
            "Provide either a material directory or explicit cert, key, and root CA files.",
            "Use signed zip mode for the public distributed provisioning workflow.",
            None,
            exit_code=4,
        )
        return 1

    if has_dir:
        # Auto-detect name from *.key if the internal caller did not provide it.
        if not args.name:
            args.name = _discover_name_from_dir(args.dir)
        # Resolve paths by convention: all files are named after the participant
        args.cert = os.path.join(args.dir, f"{args.name}.crt")
        args.key = os.path.join(args.dir, f"{args.name}.key")
        args.rootca = os.path.join(args.dir, "rootCA.pem")
    else:
        # Explicit internal material mode: all three paths and participant name are required.
        missing = [f"--{f}" for f in ("cert", "key", "rootca") if not getattr(args, f, None)]
        if missing:
            output_error_message(
                "INVALID_ARGS",
                f"Missing required argument(s): {', '.join(missing)}.",
                "When using explicit internal material mode, cert, key, and root CA must all be provided.",
                None,
                exit_code=4,
            )
            return 1
        if not args.name:
            output_error_message(
                "INVALID_ARGS",
                "Participant name is required when using explicit internal material mode.",
                "Use signed zip mode for the public distributed provisioning workflow.",
                None,
                exit_code=4,
            )
            return 1

    # Step 5: Guard sentinel name collision (host collision check happens after kit_type is known).
    if args.name == _DUMMY_SERVER_NAME:
        output_error_message(
            "INVALID_ARGS",
            f"Participant name {_DUMMY_SERVER_NAME!r} is reserved and cannot be used.",
            "Choose a different name for this participant.",
            None,
            exit_code=4,
        )
        return 1

    # Step 6: Validate resolved cert/key/rootca exist
    if not os.path.isfile(args.cert):
        if has_dir:
            hint = (
                f"{os.path.basename(args.cert)} is the signed certificate from the Project Admin "
                f"(different from {args.name}.csr, which is the signing request you generated). "
                "Ask your Project Admin to approve your request zip, "
                f"then place the resulting {os.path.basename(args.cert)} and rootCA.pem into {args.dir}."
            )
        else:
            hint = "Provide the signed certificate received from the Project Admin."
        output_error("CERT_NOT_FOUND", exit_code=1, path=args.cert, detail=hint)
        return 1

    if not os.path.isfile(args.key):
        hint = (
            f"Use the private key generated by 'nvflare cert request' for {args.name}."
            if has_dir
            else "Provide the private key generated by 'nvflare cert request'."
        )
        output_error("KEY_NOT_FOUND", exit_code=1, path=args.key, detail=hint)
        return 1

    if not os.path.isfile(args.rootca):
        hint = (
            f"Place the rootCA.pem from the Project Admin into {args.dir}."
            if has_dir
            else "Provide the rootCA.pem received from the Project Admin."
        )
        output_error("ROOTCA_NOT_FOUND", exit_code=1, path=args.rootca, detail=hint)
        return 1

    cert = _validate_cert_material(args.cert, args.key, args.rootca, validate_key_match=True)
    if cert is None:
        return 1
    kit_type = _read_cert_type_from_cert(cert)
    if not kit_type or kit_type not in _VALID_CERT_TYPES:
        output_error("CERT_TYPE_UNKNOWN", exit_code=1, cert=args.cert)
        return 1
    args.kit_type = kit_type

    project_name = getattr(args, "project_name", None) or "project"
    result = _build_selected_participant_package(
        args=args,
        scheme=scheme,
        host=host,
        port=port,
        name=args.name,
        org=_DUMMY_ORG,
        kit_type=args.kit_type,
        cert_path=args.cert,
        key_path=args.key,
        rootca_path=args.rootca,
        project_name=project_name,
    )
    if not isinstance(result, dict):
        return result
    output_ok(result)
    return 0
