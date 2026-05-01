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

import copy
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
from nvflare.lighter.utils import load_crt_bytes, load_yaml, verify_cert, verify_content
from nvflare.tool.cert.cert_constants import (
    ADMIN_CERT_TYPES,
    CA_INFO_FIELD,
    DEFAULT_PROVISION_VERSION,
    KIT_TYPE_TO_ROLE,
    PROVISION_VERSION_FIELD,
    ROOTCA_FINGERPRINT_FIELD,
    VALID_CERT_TYPES,
    is_valid_provision_version,
)
from nvflare.tool.cert.file_utils import read_file_nofollow as _shared_read_file_nofollow
from nvflare.tool.cert.file_utils import safe_project_name_error
from nvflare.tool.cert.file_utils import write_file_nofollow as _shared_write_file_nofollow
from nvflare.tool.cert.fingerprint import cert_fingerprint_sha256, normalize_sha256_fingerprint
from nvflare.tool.cli_output import (
    is_json_mode,
    output_error,
    output_error_message,
    output_ok,
    output_usage_error,
    print_human,
)
from nvflare.tool.cli_schema import handle_schema_flag

_VALID_SCHEMES = {"grpc", "tcp", "http"}
_VALID_CONNECTION_SECURITY = {"clear", "tls", "mtls"}
_ADMIN_ROLES = set(ADMIN_CERT_TYPES)
_VALID_CERT_TYPES = set(VALID_CERT_TYPES)
_KIT_TYPE_TO_ROLE = KIT_TYPE_TO_ROLE
_DUMMY_SERVER_NAME = "__nvflare_dummy_server__"
_DUMMY_ORG = "myorg"
_MAX_ZIP_MEMBER_SIZE = 10 * 1024 * 1024
_REQUEST_ID_PATTERN = re.compile(
    r"(?:[0-9a-fA-F]{32}|[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"
)
_SIGNED_KIND_TO_CERT_TYPE = {
    "site": "client",
    "server": "server",
}
_SIGNED_USER_CERT_TYPES = _VALID_CERT_TYPES - {"client", "server"}
_PEM_PRIVATE_KEY_MARKERS = (
    b"-----BEGIN PRIVATE KEY-----",
    b"-----BEGIN ENCRYPTED PRIVATE KEY-----",
    b"-----BEGIN RSA PRIVATE KEY-----",
    b"-----BEGIN EC PRIVATE KEY-----",
    b"-----BEGIN DSA PRIVATE KEY-----",
    b"-----BEGIN OPENSSH PRIVATE KEY-----",
)


def _reject_invalid_project_name(project_name: str, *, code: str, hint: str, field_label: str = "Project") -> None:
    output_error_message(
        code,
        f"Invalid {field_label.lower()} name: {project_name!r}.",
        hint,
        None,
        exit_code=4,
    )


def _validate_safe_project_name(
    project_name: str, *, code: str = "INVALID_PROJECT_NAME", field_label: str = "Project"
) -> bool:
    validation_error = safe_project_name_error(project_name, field_label=field_label)
    if validation_error:
        _, hint = validation_error
        _reject_invalid_project_name(project_name, code=code, hint=hint, field_label=field_label)
        return False
    return True


def _validate_provision_version(value: str, *, code: str = "INVALID_SIGNED_ZIP") -> bool:
    if is_valid_provision_version(value):
        return True
    output_error_message(
        code,
        f"Invalid provision version: {value!r}.",
        "Provision version must be exactly two digits, for example 00.",
        None,
        exit_code=4,
    )
    return False


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


_write_file_nofollow = _shared_write_file_nofollow


def _read_file_nofollow(path: str, max_size: int = _MAX_ZIP_MEMBER_SIZE) -> bytes:
    return _shared_read_file_nofollow(path, max_size)


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


def _remove_existing_path(path: str) -> None:
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    else:
        os.remove(path)


class FixedProdWorkspaceBuilder(WorkspaceBuilder):
    """Workspace builder variant that finalizes signed-zip kits into a fixed provision directory."""

    def __init__(self, target_prod_dir: str, participant_name: str, force: bool = False, exclude_names=None):
        super().__init__()
        self.target_prod_dir = target_prod_dir
        self.participant_name = participant_name
        self.force = force
        self.exclude_names = set(exclude_names or [])

    def initialize(self, project: Project, ctx):
        # Keep template loading behavior from WorkspaceBuilder without using its prod_NN incrementing state.
        ctx.load_templates(self.template_files)
        ctx[CtxKey.LAST_PROD_STAGE] = -1

    def finalize(self, project: Project, ctx):
        target_parent = os.path.dirname(self.target_prod_dir)
        os.makedirs(target_parent, exist_ok=True)
        target_exists = os.path.exists(self.target_prod_dir)
        if target_exists and not os.path.isdir(self.target_prod_dir):
            raise ValueError(f"target production path exists but is not a directory: {self.target_prod_dir}")
        os.makedirs(self.target_prod_dir, exist_ok=True)
        ctx[CtxKey.CURRENT_PROD_DIR] = self.target_prod_dir

        wip_dir = ctx.get_wip_dir()
        for name in os.listdir(wip_dir):
            if name in self.exclude_names:
                continue
            src = os.path.join(wip_dir, name)
            dst = os.path.join(self.target_prod_dir, name)
            if name == self.participant_name and os.path.exists(dst):
                if not self.force:
                    raise ValueError(f"participant output already exists: {dst}")
                _remove_existing_path(dst)
            if os.path.exists(dst):
                # Preserve existing root-level files when adding another participant to the same provision directory.
                continue
            shutil.move(src, dst)

        ctx.info(f"Generated results can be found under {self.target_prod_dir}. ")


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
        return ""
    if len(keys) > 1:
        output_error_message(
            "AMBIGUOUS_KEY",
            f"Multiple *.key files found in {work_dir}: {keys}",
            "Provide exactly one non-hidden participant key in the material directory.",
            None,
            exit_code=1,
        )
        return ""
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


def _prod_dir_rootca_fingerprints(prod_dir: str):
    fingerprints = set()
    for root, dirs, files in os.walk(prod_dir):
        dirs[:] = [d for d in dirs if not os.path.islink(os.path.join(root, d))]
        if "rootCA.pem" not in files:
            continue
        rootca_path = os.path.join(root, "rootCA.pem")
        try:
            fingerprints.add(cert_fingerprint_sha256(_load_crt_nofollow(rootca_path)))
        except Exception as e:
            output_error_message(
                "ROOTCA_LOAD_FAILED",
                f"Existing production directory contains unreadable rootCA.pem: {rootca_path}.",
                "Inspect the existing provision directory before packaging into it.",
                None,
                exit_code=4,
                detail=str(e),
            )
            return None
    return fingerprints


def _ensure_prod_dir_rootca_matches(prod_dir: str, expected_fingerprint: str) -> bool:
    if not os.path.exists(prod_dir):
        return True
    fingerprints = _prod_dir_rootca_fingerprints(prod_dir)
    if fingerprints is None:
        return False
    mismatches = sorted(f for f in fingerprints if f != expected_fingerprint)
    if mismatches:
        output_error_message(
            "ROOTCA_FINGERPRINT_MISMATCH",
            f"Existing production directory {prod_dir} uses a different root CA.",
            "Use the signed zip for the same CA, choose the matching CA provision version, or clear the stale provision directory.",
            None,
            exit_code=4,
            detail=f"expected {expected_fingerprint}; found {', '.join(mismatches)}",
        )
        return False
    return True


def _fixed_signed_prod_dir(workspace: str, project_name: str, provision_version: str):
    project_dir = _project_dir_under_workspace(workspace, project_name)
    if not project_dir:
        return None
    return os.path.join(project_dir, f"prod_{provision_version}")


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
    elif isinstance(project_dict, dict) and "participants" in project_dict and PropKey.API_VERSION not in project_dict:
        project_dict = copy.deepcopy(project_dict)
        project_dict[PropKey.API_VERSION] = 3

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


def _build_package_builders(custom_builders, cert_builder, scheme, workspace_builder=None):
    has_cert = False
    all_builders = []
    for b in custom_builders or []:
        if isinstance(b, SignatureBuilder):
            continue
        if isinstance(b, CertBuilder):
            all_builders.append(cert_builder)
            has_cert = True
        elif isinstance(b, StaticFileBuilder):
            # Keep any custom StaticFileBuilder instance from the participant
            # definition, but make its connection scheme match the approved
            # project profile.
            b.scheme = scheme
            all_builders.append(b)
        else:
            all_builders.append(b)

    if workspace_builder is not None:
        existing_workspace_builder = next((b for b in all_builders if isinstance(b, WorkspaceBuilder)), None)
        if existing_workspace_builder is not None:
            workspace_builder.template_files = existing_workspace_builder.template_files
        all_builders = [b for b in all_builders if not isinstance(b, WorkspaceBuilder)]

    if not has_cert:
        ws_pos = next((i for i, b in enumerate(all_builders) if isinstance(b, WorkspaceBuilder)), -1)
        all_builders.insert(ws_pos + 1, cert_builder)

    if workspace_builder is not None:
        all_builders.insert(0, workspace_builder)
    elif not any(isinstance(b, WorkspaceBuilder) for b in all_builders):
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
    admin_port: int = None,
    name: str,
    org: str,
    kit_type: str,
    cert_path: str,
    key_path: str,
    rootca_path: str,
    project_name: str,
    participant_props: dict = None,
    project_props: dict = None,
    custom_builders=None,
    provision_version: str = None,
    rootca_fingerprint: str = None,
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
            "Use a participant name that is distinct from the configured server hostname.",
            None,
            exit_code=4,
        )
        return 1

    workspace = os.path.abspath(getattr(args, "workspace", None) or "workspace")
    admin_port = admin_port if admin_port is not None else port
    project_name = project_name or getattr(args, "project_name", None) or "project"
    if _project_dir_under_workspace(workspace, project_name) is None:
        return 1

    target_prod_dir = None
    workspace_builder = None
    if provision_version is not None:
        if not _validate_provision_version(provision_version):
            return 1
        target_prod_dir = _fixed_signed_prod_dir(workspace, project_name, provision_version)
        if target_prod_dir is None:
            return 1
        if rootca_fingerprint and not _ensure_prod_dir_rootca_matches(target_prod_dir, rootca_fingerprint):
            return 1
        existing_path = os.path.join(target_prod_dir, name)
        if os.path.exists(existing_path) and not getattr(args, "force", False):
            output_error_message(
                "OUTPUT_DIR_EXISTS",
                f"Participant output already exists: {existing_path}.",
                "Use --force to replace this participant output. --force does not bypass root CA mismatch checks.",
                None,
                exit_code=1,
            )
            return 1
        exclude_names = {host} if kit_type != "server" else set()
        workspace_builder = FixedProdWorkspaceBuilder(
            target_prod_dir=target_prod_dir,
            participant_name=name,
            force=getattr(args, "force", False),
            exclude_names=exclude_names,
        )
    else:
        latest_prod = _latest_prod_dir(workspace, project_name)
        if latest_prod:
            existing_path = os.path.join(latest_prod, name)
            if os.path.exists(existing_path) and not getattr(args, "force", False):
                output_error("OUTPUT_DIR_EXISTS", exit_code=1, path=existing_path)
                return 1

    is_server = kit_type == "server"
    server_props = {
        PropKey.FED_LEARN_PORT: port,
        PropKey.ADMIN_PORT: admin_port,
    }

    project = Project(name=project_name, description="", props=project_props)
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
        builders=_build_package_builders(custom_builders, cert_builder, scheme, workspace_builder),
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
        if target_prod_dir is None:
            shutil.rmtree(os.path.join(prod_dir, host), ignore_errors=True)
        project.remove_server()

    return _make_package_result(prod_dir, name, kit_type, f"{scheme}://{host}:{port}")


def _safe_zip_names(zf: zipfile.ZipFile, zip_path: str):
    names = []
    seen = set()
    for info in zf.infolist():
        name = info.filename
        norm = posixpath.normpath(name)
        parts = norm.split("/")
        mode = info.external_attr >> 16
        if (
            not name
            or name == "."
            or name.endswith("/")
            or os.path.isabs(name)
            or "\\" in name
            or any(ord(ch) < 32 for ch in name)
            or norm != name
            or norm.startswith("../")
            or ".." in parts
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
            return None
        if name.lower().endswith(".key"):
            output_error_message(
                "INVALID_SIGNED_ZIP",
                "Signed zip must not contain private key material.",
                "Use the local request directory that contains the private key.",
                None,
                exit_code=4,
            )
            return None
        if name.lower().endswith(".pem") and _read_zip_member_limited(zf, name, zip_path) is None:
            return None
        seen.add(name)
        names.append(name)
    return names


def _contains_private_key_material(content: bytes) -> bool:
    return any(marker in content for marker in _PEM_PRIVATE_KEY_MARKERS)


def _read_zip_member_limited(zf: zipfile.ZipFile, name: str, zip_path: str) -> Optional[bytes]:
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
        return None
    if len(content) > _MAX_ZIP_MEMBER_SIZE:
        output_error_message(
            "INVALID_SIGNED_ZIP",
            f"Signed zip member exceeds size limit: {name}.",
            "Ask the Project Admin to regenerate the signed zip.",
            None,
            exit_code=4,
        )
        return None
    if _contains_private_key_material(content):
        output_error_message(
            "INVALID_SIGNED_ZIP",
            "Signed zip must not contain private key material.",
            "Use the local request directory that contains the private key.",
            None,
            exit_code=4,
        )
        return None
    return content


def _decode_zip_json(content: Optional[bytes], name: str, zip_path: str):
    if content is None:
        return None
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
        return None


def _read_zip_json(zf: zipfile.ZipFile, name: str, zip_path: str):
    return _decode_zip_json(_read_zip_member_limited(zf, name, zip_path), name, zip_path)


def _decode_zip_yaml(content: Optional[bytes], name: str, zip_path: str):
    if content is None:
        return None
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
        return None
    if not isinstance(data, dict):
        output_error_message(
            "INVALID_SIGNED_ZIP",
            f"{name} in signed zip must be a mapping.",
            "Ask the Project Admin to regenerate the signed zip.",
            None,
            exit_code=4,
        )
        return None
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
    return value.strip().lower().removeprefix("sha256:")


def _verify_rootca_fingerprint_options(args, actual_fingerprint: str) -> bool:
    expected = getattr(args, "expected_fingerprint", None)
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

    if not expected:
        print_human(
            "Warning: Root CA SHA256 fingerprint was not verified. " "Use --fingerprint to verify it out-of-band."
        )

    return True


def _resolve_signed_ca_info(signed_meta: dict, actual_rootca_fingerprint: str):
    ca_info = signed_meta.get(CA_INFO_FIELD)
    if ca_info is None:
        return {
            PROVISION_VERSION_FIELD: DEFAULT_PROVISION_VERSION,
            ROOTCA_FINGERPRINT_FIELD: actual_rootca_fingerprint,
        }
    if not isinstance(ca_info, dict):
        output_error_message(
            "INVALID_SIGNED_ZIP",
            "Signed zip ca_info must be a mapping.",
            "Ask the Project Admin to regenerate the signed zip.",
            None,
            exit_code=4,
        )
        return None
    provision_version = ca_info.get(PROVISION_VERSION_FIELD)
    if not _validate_provision_version(provision_version):
        return None
    signed_fingerprint = normalize_sha256_fingerprint(ca_info.get(ROOTCA_FINGERPRINT_FIELD))
    if not signed_fingerprint:
        output_error_message(
            "INVALID_ROOTCA_FINGERPRINT",
            f"Invalid signed ca_info root CA fingerprint: {ca_info.get(ROOTCA_FINGERPRINT_FIELD)!r}.",
            "Ask the Project Admin to regenerate the signed zip.",
            None,
            exit_code=4,
        )
        return None
    if signed_fingerprint != actual_rootca_fingerprint:
        output_error_message(
            "ROOTCA_FINGERPRINT_MISMATCH",
            "Signed ca_info root CA fingerprint does not match included rootCA.pem.",
            "Ask the Project Admin to regenerate the signed zip.",
            None,
            exit_code=4,
            detail=f"signed {signed_fingerprint}; included {actual_rootca_fingerprint}",
        )
        return None
    return {
        PROVISION_VERSION_FIELD: provision_version,
        ROOTCA_FINGERPRINT_FIELD: signed_fingerprint,
    }


def _expected_hash(meta: dict, *names):
    hashes = meta.get("hashes") if isinstance(meta.get("hashes"), dict) else {}
    for name in names:
        value = hashes.get(name) or hashes.get(f"{name}_sha256") or meta.get(f"{name}_sha256")
        if value:
            return _normalize_hash(value)
    return ""


def _is_project_shaped_site_meta(site_meta: dict) -> bool:
    return isinstance(site_meta, dict) and isinstance(site_meta.get("participants"), list)


def _role_to_cert_type(role: str) -> str:
    if role == "org-admin":
        return "org_admin"
    return role if role in _ADMIN_ROLES else ""


def _resolve_participant_type(participant: dict):
    """Return (kind, cert_type, cert_role) from a participant dict, or None if the type is unrecognised."""
    p_type = participant.get("type")
    if p_type == "client":
        return "site", "client", None
    if p_type == "server":
        return "server", "server", None
    if p_type == "admin":
        cert_role = _role_to_cert_type(participant.get("role"))
        return "user", cert_role, cert_role
    return None


def _site_identity_from_signed_metadata(site_meta: dict) -> dict:
    if not isinstance(site_meta, dict):
        return {}
    if _is_project_shaped_site_meta(site_meta):
        participants = site_meta.get("participants") or []
        if len(participants) != 1 or not isinstance(participants[0], dict):
            return {}
        participant = participants[0]
        resolved = _resolve_participant_type(participant)
        if resolved is None:
            return {}
        kind, cert_type, cert_role = resolved
        return {
            "project": site_meta.get("name"),
            "name": participant.get("name"),
            "org": participant.get("org"),
            "kind": kind,
            "cert_type": cert_type,
            "cert_role": cert_role,
        }
    return {
        "project": site_meta.get("project"),
        "name": site_meta.get("name"),
        "org": site_meta.get("org"),
        "kind": site_meta.get("kind"),
        "cert_type": site_meta.get("type") or site_meta.get("cert_type"),
        "cert_role": site_meta.get("cert_role"),
    }


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

    required = (
        "request_id",
        "project",
        "name",
        "org",
        "kind",
        "cert_type",
        "cert_file",
        "rootca_file",
        "scheme",
        "default_connection_security",
        "server",
    )
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
    ca_info = signed_meta.get(CA_INFO_FIELD)
    if ca_info is not None:
        if not isinstance(ca_info, dict):
            output_error_message(
                "INVALID_SIGNED_ZIP",
                "Signed zip ca_info must be a mapping.",
                "Ask the Project Admin to regenerate the signed zip.",
                None,
                exit_code=4,
            )
            return False
        if not _validate_provision_version(ca_info.get(PROVISION_VERSION_FIELD)):
            return False
        if not normalize_sha256_fingerprint(ca_info.get(ROOTCA_FINGERPRINT_FIELD)):
            output_error_message(
                "INVALID_ROOTCA_FINGERPRINT",
                f"Invalid signed ca_info root CA fingerprint: {ca_info.get(ROOTCA_FINGERPRINT_FIELD)!r}.",
                "Ask the Project Admin to regenerate the signed zip.",
                None,
                exit_code=4,
            )
            return False
    scheme = signed_meta.get("scheme")
    if scheme is not None and (not isinstance(scheme, str) or scheme.strip().lower() not in _VALID_SCHEMES):
        output_error_message(
            "INVALID_SIGNED_ZIP",
            f"Invalid signed zip scheme: {scheme!r}.",
            "Ask the Project Admin to regenerate the signed zip.",
            None,
            exit_code=4,
        )
        return False
    default_conn_sec = signed_meta.get("default_connection_security")
    if default_conn_sec is not None and (
        not isinstance(default_conn_sec, str) or default_conn_sec.strip().lower() not in _VALID_CONNECTION_SECURITY
    ):
        output_error_message(
            "INVALID_SIGNED_ZIP",
            f"Invalid signed zip default_connection_security: {default_conn_sec!r}.",
            "Ask the Project Admin to regenerate the signed zip.",
            None,
            exit_code=4,
        )
        return False
    if _signed_server_endpoint(signed_meta) is None:
        return False

    site_identity = _site_identity_from_signed_metadata(site_meta)
    if not site_identity:
        output_error_message(
            "INVALID_SIGNED_ZIP",
            "site.yaml must identify exactly one participant.",
            "Ask the Project Admin to regenerate the signed zip.",
            None,
            exit_code=4,
        )
        return False
    for field in ("name", "org", "cert_type", "project"):
        if site_identity.get(field) != signed_meta.get(field):
            output_error_message(
                "INVALID_SIGNED_ZIP",
                f"site.yaml field '{field}' does not match signed metadata.",
                "Ask the Project Admin to regenerate the signed zip.",
                None,
                exit_code=4,
            )
            return False
    if site_identity.get("kind") != signed_meta.get("kind"):
        output_error_message(
            "INVALID_SIGNED_ZIP",
            "site.yaml field 'kind' does not match signed metadata.",
            "Ask the Project Admin to regenerate the signed zip.",
            None,
            exit_code=4,
        )
        return False
    if (site_identity.get("cert_role") or None) != (signed_meta.get("cert_role") or None):
        output_error_message(
            "INVALID_SIGNED_ZIP",
            "site.yaml field 'cert_role' does not match signed metadata.",
            "Ask the Project Admin to regenerate the signed zip.",
            None,
            exit_code=4,
        )
        return False
    if _is_project_shaped_site_meta(site_meta) and "connection_security" in site_meta["participants"][0]:
        output_error_message(
            "INVALID_SIGNED_ZIP",
            "Signed zip site.yaml must not contain participant connection_security overrides.",
            "Ask the Project Admin to regenerate the signed zip.",
            None,
            exit_code=4,
        )
        return False
    if _is_project_shaped_site_meta(site_meta):
        participant = site_meta["participants"][0]
        endpoint_fields = [field for field in ("server", "fed_learn_port", "admin_port") if field in participant]
        if endpoint_fields:
            output_error_message(
                "INVALID_SIGNED_ZIP",
                f"Signed zip site.yaml must not contain endpoint field(s): {', '.join(endpoint_fields)}.",
                "Use the endpoint information stored in signed.json.",
                None,
                exit_code=4,
            )
            return False
    if _is_project_shaped_site_meta(site_meta) and PropKey.LISTENING_HOST in site_meta["participants"][0]:
        output_error_message(
            "INVALID_SIGNED_ZIP",
            "Signed zip site.yaml must not contain listening_host.",
            "Distributed provisioning does not support listener certificates yet; use centralized provisioning.",
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
    # Note: csr_sha256 is required in signed_meta["hashes"] (validated by _validate_signed_metadata)
    # but is not verified here because the CSR is not included in the signed zip.
    # Equivalent security is provided by public_key_sha256: the cert's public key is verified
    # against the CSR's public key hash in _validate_signed_public_key_hash and then against
    # the local private key via validate_key_match=True in _validate_cert_material.
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


def _verify_signed_metadata_signature(file_contents: dict) -> bool:
    try:
        rootca = load_crt_bytes(file_contents["rootCA.pem"])
        signature = file_contents["signed.json.sig"].decode("utf-8")
        verify_content(file_contents["signed.json"], signature, rootca.public_key())
        return True
    except Exception as e:
        output_error_message(
            "INVALID_SIGNED_ZIP",
            f"Signed approval metadata signature is invalid: {e}.",
            "Ask the Project Admin to regenerate the signed zip.",
            None,
            exit_code=4,
        )
        return False


def _load_signed_zip(input_path: str):
    signed_meta = None
    site_meta = None
    cert_name = ""
    file_contents = {}
    try:
        with zipfile.ZipFile(input_path, "r") as zf:
            safe_names = _safe_zip_names(zf, input_path)
            if safe_names is None:
                return signed_meta, site_meta, cert_name, file_contents
            names = set(safe_names)
            required = {"signed.json", "signed.json.sig", "site.yaml", "rootCA.pem"}
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
            for content_name, member_name in (
                ("signed.json", "signed.json"),
                ("signed.json.sig", "signed.json.sig"),
                ("site.yaml", "site.yaml"),
                ("rootCA.pem", "rootCA.pem"),
                ("cert", cert_name),
            ):
                content = _read_zip_member_limited(zf, member_name, input_path)
                if content is None:
                    return None, None, None, {}
                file_contents[content_name] = content
            if not _verify_signed_metadata_signature(file_contents):
                return None, None, None, {}
            signed_meta = _decode_zip_json(file_contents["signed.json"], "signed.json", input_path)
            site_meta = _decode_zip_yaml(file_contents["site.yaml"], "site.yaml", input_path)
            if signed_meta is None or site_meta is None:
                return None, None, None, {}
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
                if not os.path.isdir(candidate):
                    output_error_message(
                        "REQUEST_DIR_NOT_FOUND",
                        f"Request directory not found: {candidate}.",
                        "Provide the directory created by 'nvflare cert request', or omit --request-dir to auto-discover.",
                        None,
                        exit_code=4,
                    )
                    return None
                missing = _missing_request_material(candidate, identity["name"])
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
        if not os.path.isdir(candidate):
            output_error_message(
                "REQUEST_DIR_NOT_FOUND",
                f"Request directory not found: {candidate}.",
                "Provide the directory created by 'nvflare cert request', or omit --request-dir to auto-discover.",
                None,
                exit_code=4,
            )
            return None
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


def _read_yaml_file_nofollow(path: str, *, code: str, hint: str):
    if not os.path.isfile(path):
        output_error_message(
            code,
            f"Local participant definition was not found: {path}.",
            hint,
            None,
            exit_code=1,
        )
        return None
    try:
        data = yaml.safe_load(_read_file_nofollow(path).decode("utf-8"))
    except Exception as e:
        output_error_message(
            code,
            f"Local participant definition could not be read: {e}.",
            hint,
            None,
            exit_code=4,
        )
        return None
    if not isinstance(data, dict):
        output_error_message(
            code,
            "Local participant definition must be a YAML mapping.",
            hint,
            None,
            exit_code=4,
        )
        return None
    return data


def _participant_cert_identity(project_name: str, participant: dict) -> dict:
    if not isinstance(participant, dict):
        return {}
    resolved = _resolve_participant_type(participant)
    if resolved is None:
        return {}
    kind, cert_type, cert_role = resolved
    return {
        "project": project_name,
        "name": participant.get("name"),
        "org": participant.get("org"),
        "kind": kind,
        "cert_type": cert_type,
        "cert_role": cert_role,
    }


def _flat_site_to_project_dict(site_meta: dict, identity: dict) -> dict:
    kit_type = identity.get("cert_type") or site_meta.get("type")
    participant = {k: v for k, v in site_meta.items() if k not in {"project", "project_name", "kind", "cert_role"}}
    participant["name"] = site_meta.get("name") or identity.get("name")
    participant["org"] = site_meta.get("org") or identity.get("org")
    if kit_type in _ADMIN_ROLES:
        participant["type"] = "admin"
        participant[PropKey.ROLE] = _KIT_TYPE_TO_ROLE[kit_type]
    else:
        participant["type"] = kit_type
    return {
        PropKey.API_VERSION: 3,
        PropKey.NAME: site_meta.get("project") or site_meta.get("project_name") or identity.get("project"),
        "participants": [participant],
    }


def _local_project_dict_from_site(site_meta: dict, identity: dict) -> dict:
    if _is_project_shaped_site_meta(site_meta):
        project_dict = copy.deepcopy(site_meta)
        if PropKey.API_VERSION not in project_dict:
            project_dict[PropKey.API_VERSION] = 3
        return project_dict
    return _flat_site_to_project_dict(site_meta, identity)


def _find_participant_def(project_dict: dict, identity: dict):
    participants = project_dict.get("participants")
    if not isinstance(participants, list):
        return None
    for participant in participants:
        if isinstance(participant, dict) and participant.get("name") == identity.get("name"):
            return participant
    return participants[0] if len(participants) == 1 and isinstance(participants[0], dict) else None


def _find_project_participant(project: Project, name: str):
    for participant in project.get_all_participants():
        if participant.name == name:
            return participant
    return None


def _local_site_mismatch(detail: str) -> None:
    output_error_message(
        "LOCAL_SITE_MISMATCH",
        "Local participant definition does not match the signed zip.",
        "Use the original request folder created for this signed zip.",
        None,
        exit_code=4,
        detail=detail,
    )


def _local_site_unsupported_feature(detail: str) -> None:
    output_error_message(
        "LOCAL_SITE_UNSUPPORTED_FEATURE",
        "Local participant definition uses a feature unsupported by distributed provisioning.",
        "Use centralized provisioning for this participant, or remove the unsupported field and request a new signed zip.",
        None,
        exit_code=4,
        detail=detail,
    )


def _validate_local_site_identity(project_dict: dict, participant_def: dict, signed_meta: dict) -> bool:
    project_name = project_dict.get(PropKey.NAME)
    if isinstance(participant_def, dict):
        participant_type = participant_def.get("type")
        if participant_type not in {"client", "server", "admin"}:
            _local_site_mismatch(
                f"unsupported participant type {participant_type!r}; expected client, server, or admin"
            )
            return False
        if PropKey.LISTENING_HOST in participant_def:
            _local_site_unsupported_feature(
                "listening_host is not supported by distributed provisioning packaging yet; "
                "the signed zip contains only one participant certificate/key pair"
            )
            return False
    site_identity = _participant_cert_identity(project_name, participant_def)
    if not site_identity:
        _local_site_mismatch("site.yaml must identify one client, server, or admin participant")
        return False
    expected = {
        "project": signed_meta.get("project"),
        "name": signed_meta.get("name"),
        "org": signed_meta.get("org"),
        "kind": signed_meta.get("kind"),
        "cert_type": signed_meta.get("cert_type"),
        "cert_role": signed_meta.get("cert_role") or None,
    }
    for field, expected_value in expected.items():
        if (site_identity.get(field) or None) != (expected_value or None):
            _local_site_mismatch(f"field {field!r} differs")
            return False
    return True


def _load_local_participant_context(request_dir: str, identity: dict, signed_meta: dict, kit_type: str):
    site_yaml_path = os.path.join(request_dir, "site.yaml")
    site_meta = _read_yaml_file_nofollow(
        site_yaml_path,
        code="LOCAL_SITE_INVALID",
        hint="Use the original request folder created by 'nvflare cert request'.",
    )
    if site_meta is None:
        return None, None, None, None
    project_dict = _local_project_dict_from_site(site_meta, identity)
    participant_def = _find_participant_def(project_dict, identity)
    if participant_def is None:
        _local_site_mismatch(f"participant {identity.get('name')!r} was not found")
        return None, None, None, None
    if not _validate_local_site_identity(project_dict, participant_def, signed_meta):
        return None, None, None, None
    try:
        project = prepare_project(copy.deepcopy(project_dict))
        custom_builders = prepare_builders(project_dict)
    except Exception as ex:
        output_error_message(
            "LOCAL_SITE_INVALID",
            f"Invalid local participant definition: {ex}",
            "Use the original request folder created by 'nvflare cert request'.",
            None,
            exit_code=4,
        )
        return None, None, None, None
    participant = _find_project_participant(project, identity["name"])
    if not participant:
        _local_site_mismatch(f"participant {identity.get('name')!r} was not found")
        return None, None, None, None
    participant_kit_type = _participant_kit_type(participant)
    if participant_kit_type != kit_type:
        _local_site_mismatch(f"participant type {participant_kit_type!r} differs from signed type {kit_type!r}")
        return None, None, None, None
    return project, participant, custom_builders, participant_def


def _validate_signed_scheme_and_security(signed_meta: dict) -> tuple:
    scheme = signed_meta.get("scheme")
    if scheme is not None:
        scheme = scheme.strip().lower()
        if scheme not in _VALID_SCHEMES:
            output_error_message(
                "INVALID_SIGNED_ZIP",
                f"Invalid signed zip scheme: {signed_meta.get('scheme')!r}.",
                "Ask the Project Admin to regenerate the signed zip.",
                None,
                exit_code=4,
            )
            return None, None
    default_conn_sec = signed_meta.get("default_connection_security")
    if default_conn_sec is not None:
        default_conn_sec = default_conn_sec.strip().lower()
        if default_conn_sec not in _VALID_CONNECTION_SECURITY:
            output_error_message(
                "INVALID_SIGNED_ZIP",
                f"Invalid signed zip default_connection_security: {signed_meta.get('default_connection_security')!r}.",
                "Ask the Project Admin to regenerate the signed zip.",
                None,
                exit_code=4,
            )
            return None, None
    return scheme, default_conn_sec


def _port_from_value(value, label: str, code: str = "LOCAL_SITE_INVALID", hint: str = None):
    if not isinstance(value, int) or isinstance(value, bool) or not (1 <= value <= 65535):
        output_error_message(
            code,
            f"{label} must be an integer from 1 to 65535.",
            hint or "Use the original participant definition created for this request.",
            None,
            exit_code=4,
        )
        return None
    return value


def _signed_server_endpoint(signed_meta: dict):
    server = signed_meta.get("server")
    if not isinstance(server, dict):
        output_error_message(
            "INVALID_SIGNED_ZIP",
            "Signed approval metadata does not contain a server endpoint.",
            "Ask the Project Admin to regenerate the signed zip with a project profile server block.",
            None,
            exit_code=4,
        )
        return None
    host = server.get("host")
    if not isinstance(host, str) or not host.strip():
        output_error_message(
            "INVALID_SIGNED_ZIP",
            "Signed approval metadata server.host must be a non-empty string.",
            "Ask the Project Admin to regenerate the signed zip with a valid project profile server block.",
            None,
            exit_code=4,
        )
        return None
    invalid, reason = name_check(host.strip(), "host_name")
    if invalid:
        output_error_message(
            "INVALID_SIGNED_ZIP",
            f"Signed approval metadata server.host is invalid: {reason}.",
            "Ask the Project Admin to regenerate the signed zip with a valid project profile server block.",
            None,
            exit_code=4,
        )
        return None
    hint = "Ask the Project Admin to regenerate the signed zip with a valid project profile server block."
    port = _port_from_value(server.get("fed_learn_port"), "server.fed_learn_port", "INVALID_SIGNED_ZIP", hint)
    admin_port = _port_from_value(server.get("admin_port"), "server.admin_port", "INVALID_SIGNED_ZIP", hint)
    if port is None or admin_port is None:
        return None
    return host.strip(), port, admin_port


def _resolve_packaging_endpoint(args, signed_meta: dict, participant_def: dict, kit_type: str):
    signed_scheme, default_conn_sec = _validate_signed_scheme_and_security(signed_meta)
    if signed_scheme is None and signed_meta.get("scheme") is not None:
        return None
    if default_conn_sec is None and signed_meta.get("default_connection_security") is not None:
        return None

    if not signed_scheme:
        output_error_message(
            "INVALID_SIGNED_ZIP",
            "Signed approval metadata does not contain a communication scheme.",
            "Ask the Project Admin to regenerate the signed zip with a project profile.",
            None,
            exit_code=4,
        )
        return None

    scheme = signed_scheme
    signed_endpoint = _signed_server_endpoint(signed_meta)
    if signed_endpoint is None:
        return None
    host, port, admin_port = signed_endpoint
    if kit_type == "server":
        participant_name = participant_def.get("name")
        if not isinstance(participant_name, str) or not participant_name.strip():
            output_error_message(
                "LOCAL_SITE_INVALID",
                "participant name must be a non-empty string.",
                "Use the original participant definition created for this request.",
                None,
                exit_code=4,
            )
            return None
    return scheme, host, port, admin_port, default_conn_sec


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
        site_yaml_path = os.path.join(request_dir, "site.yaml")
        if not os.path.exists(site_yaml_path):
            _write_file_nofollow(site_yaml_path, yaml.safe_dump(site_meta).encode("utf-8"))
        _write_file_nofollow(os.path.join(request_dir, f"{identity['name']}.crt"), file_contents["cert"])
        _write_file_nofollow(os.path.join(request_dir, "rootCA.pem"), file_contents["rootCA.pem"])
    except OSError as e:
        output_error("OUTPUT_DIR_NOT_WRITABLE", path=request_dir, detail=str(e))
        return False
    return True


def _handle_signed_zip_package(args):
    signed_meta, site_meta, cert_name, file_contents = _load_signed_zip(args.input)
    if not signed_meta or not site_meta or not cert_name or not file_contents:
        return 1
    if getattr(args, "endpoint", None) or (getattr(args, "admin_port", None) is not None):
        output_error_message(
            "INVALID_ARGS",
            "Endpoint overrides are not supported for signed zip packaging.",
            "Use the endpoint information stored in the local participant definition, or regenerate the request.",
            None,
            exit_code=4,
        )
        return 1
    if getattr(args, "project_file", None):
        output_error_message(
            "INVALID_ARGS",
            "Project file override cannot be combined with signed zip packaging.",
            "Use the local participant definition stored in the original request folder.",
            None,
            exit_code=4,
        )
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
        ca_info = _resolve_signed_ca_info(signed_meta, rootca_fingerprint_sha256)
        if ca_info is None:
            return 1
        if not _verify_rootca_fingerprint_options(args, rootca_fingerprint_sha256):
            return 1

        resolved_request_dir = _resolve_request_dir(args, args.input, identity)
        if not resolved_request_dir:
            if getattr(args, "request_dir", None):
                return 1
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

        local_project = None
        local_participant = None
        local_custom_builders = None
        local_participant_def = None
        (
            local_project,
            local_participant,
            local_custom_builders,
            local_participant_def,
        ) = _load_local_participant_context(resolved_request_dir, identity, signed_meta, cert_kit_type)
        if local_project is None:
            return 1

        endpoint_info = _resolve_packaging_endpoint(args, signed_meta, local_participant_def or {}, cert_kit_type)
        if endpoint_info is None:
            return 1
        scheme, host, port, admin_port, default_conn_sec = endpoint_info

        identity["org"] = identity.get("org") or _read_cert_org(cert) or _DUMMY_ORG
        participant = local_participant
        custom_builders = local_custom_builders
        project_name = local_project.name
        participant_props = dict(participant.props) if participant else {}
        project_props = {}
        if default_conn_sec:
            project_props[PropKey.CONN_SECURITY] = default_conn_sec
        if cert_kit_type not in ("server",):
            participant_props.pop(PropKey.CONN_SECURITY, None)
        elif default_conn_sec and not participant_props.get(PropKey.CONN_SECURITY):
            participant_props[PropKey.CONN_SECURITY] = default_conn_sec

        if not _write_materialized_signed_files(resolved_request_dir, identity, signed_meta, site_meta, file_contents):
            return 1
        cert_path = os.path.join(resolved_request_dir, f"{identity['name']}.crt")
        rootca_path = os.path.join(resolved_request_dir, "rootCA.pem")

        result = _build_selected_participant_package(
            args=args,
            scheme=scheme,
            host=host,
            port=port,
            admin_port=admin_port,
            name=identity["name"],
            org=identity.get("org") or _DUMMY_ORG,
            kit_type=cert_kit_type,
            cert_path=cert_path,
            key_path=key_path,
            rootca_path=rootca_path,
            project_name=project_name,
            participant_props=participant_props,
            project_props=project_props,
            custom_builders=custom_builders,
            provision_version=ca_info[PROVISION_VERSION_FIELD],
            rootca_fingerprint=ca_info[ROOTCA_FINGERPRINT_FIELD],
        )
        if not isinstance(result, dict):
            return result
        result[PROVISION_VERSION_FIELD] = ca_info[PROVISION_VERSION_FIELD]
        result["rootca_fingerprint_sha256"] = rootca_fingerprint_sha256
    output_ok(result)
    return 0


def _handle_internal_material_package(args, has_dir: bool, has_explicit: bool):
    from nvflare.tool.package.package_cli import _package_parser

    if not getattr(args, "endpoint", None):
        detail = "for this command"
        output_usage_error(
            _package_parser if not is_json_mode() else None,
            f"Server endpoint URI is required {detail}.",
            exit_code=4,
            hint="Provide the server endpoint URI, e.g. grpc://server.example.com:8002",
        )
        return 1
    try:
        scheme, host, port = _parse_endpoint(args.endpoint)
    except ValueError:
        output_error("INVALID_ENDPOINT", exit_code=4, endpoint=args.endpoint)
        return 1

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
        if not args.name:
            args.name = _discover_name_from_dir(args.dir)
            if not args.name:
                return 1
        args.cert = os.path.join(args.dir, f"{args.name}.crt")
        args.key = os.path.join(args.dir, f"{args.name}.key")
        args.rootca = os.path.join(args.dir, "rootCA.pem")
    else:
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

    if args.name == _DUMMY_SERVER_NAME:
        output_error_message(
            "INVALID_ARGS",
            f"Participant name {_DUMMY_SERVER_NAME!r} is reserved and cannot be used.",
            "Choose a different name for this participant.",
            None,
            exit_code=4,
        )
        return 1

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
        admin_port=getattr(args, "admin_port", None),
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


def handle_package(args):
    """Assemble a startup kit from locally generated key + Project Admin cert + rootCA.pem."""

    # Step 1: --schema check (before any other work)
    from nvflare.tool.package.package_cli import _PACKAGE_EXAMPLES, _package_parser

    handle_schema_flag(_package_parser, "nvflare package", _PACKAGE_EXAMPLES, sys.argv[1:])

    # Step 2: Validate required args and endpoint up front (before any file I/O)
    input_path = getattr(args, "input", None)
    has_signed_zip_input = bool(input_path)
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
            "Positional signed zip input cannot be combined with material inputs.",
            "Use --request-dir to point to the local request folder for signed zip mode.",
            None,
            exit_code=4,
        )
        return 1

    if not has_signed_zip_input and not has_dir and not has_explicit:
        output_error_message(
            "INVALID_ARGS",
            "Signed zip input is required.",
            "Use: nvflare package <identity>.signed.zip",
            None,
            exit_code=4,
        )
        return 1

    has_rootca_verify_option = bool(getattr(args, "expected_fingerprint", None))
    if has_rootca_verify_option and not has_signed_zip_input:
        output_error_message(
            "INVALID_ARGS",
            "Root CA fingerprint verification options require signed zip input.",
            "Use the distributed provisioning form: nvflare package <signed.zip>.",
            None,
            exit_code=4,
        )
        return 1

    if has_signed_zip_input:
        return _handle_signed_zip_package(args)

    return _handle_internal_material_package(args, has_dir, has_explicit)
