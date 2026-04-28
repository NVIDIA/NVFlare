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

"""nvflare cert subcommand handlers: init, request, approve, and internal csr/sign helpers."""

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
import uuid
import zipfile
from typing import Optional

from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.x509.oid import NameOID

from nvflare.apis.utils.format_check import name_check
from nvflare.lighter.constants import PropKey
from nvflare.lighter.entity import participant_from_dict
from nvflare.lighter.impl.cert import CertBuilder
from nvflare.lighter.utils import generate_keys, load_private_key_file, serialize_cert, serialize_pri_key, x509_name
from nvflare.tool import cli_output
from nvflare.tool.cert.cert_constants import ADMIN_CERT_TYPES, VALID_CERT_TYPES
from nvflare.tool.cert.fingerprint import cert_fingerprint_sha256
from nvflare.tool.cli_output import (
    output_error,
    output_error_message,
    output_ok,
    output_usage_error,
    print_human,
    prompt_yn,
)
from nvflare.tool.cli_schema import handle_schema_flag

_VALID_CERT_TYPES = set(VALID_CERT_TYPES)
_VALID_SCHEMES = {"grpc", "tcp", "http"}
_VALID_CONNECTION_SECURITY = {"clear", "tls", "mtls"}
_USAGE_HINT = "Run the command with -h for usage."
_SAFE_CERT_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._@-]*")
_SAFE_PROJECT_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_REQUEST_ID_PATTERN = re.compile(
    r"(?:[0-9a-fA-F]{32}|[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"
)
_MAX_ZIP_MEMBER_SIZE = 10 * 1024 * 1024
_REQUEST_ARTIFACT_TYPE = "nvflare.cert.request"
_SIGNED_ARTIFACT_TYPE = "nvflare.cert.signed"
_ARTIFACT_VERSION = "1"
_REQUEST_KIND_TO_CERT_TYPE = {
    "site": "client",
    "server": "server",
}
_USER_ROLE_TO_CERT_TYPE = {
    "org-admin": "org_admin",
    "org_admin": "org_admin",
    "lead": "lead",
    "member": "member",
}
_REQUEST_KINDS = set(_REQUEST_KIND_TO_CERT_TYPE) | {"user"}
_USER_CERT_TYPES = set(_USER_ROLE_TO_CERT_TYPE.values())


class _UnsafeZipSourceError(Exception):
    pass


def _validate_safe_cert_name(name: str, *, field_label: str, max_length: Optional[int] = 64) -> bool:
    if not isinstance(name, str) or not name.strip():
        output_error(
            "INVALID_NAME", exit_code=4, name=name, reason=f"{field_label} must not be empty or whitespace only."
        )
        return False
    if max_length is not None and len(name) > max_length:
        output_error(
            "INVALID_NAME",
            exit_code=4,
            name=name,
            reason=f"{field_label} must be {max_length} characters or fewer.",
        )
        return False
    if os.sep in name or (os.altsep and os.altsep in name) or name.startswith("."):
        output_error(
            "INVALID_NAME",
            exit_code=4,
            name=name,
            reason=f"{field_label} must not contain path separators or start with '.'.",
        )
        return False
    if not _SAFE_CERT_NAME_PATTERN.fullmatch(name):
        output_error(
            "INVALID_NAME",
            exit_code=4,
            name=name,
            reason=f"{field_label} must match [A-Za-z0-9][A-Za-z0-9._@-]*.",
        )
        return False
    return True


def _cert_name_max_length(cert_type: str):
    # Centralized provisioning truncates long server CNs and keeps the full host
    # as default_host/SAN. Keep distributed server requests consistent with it.
    return None if cert_type == "server" else 64


def _csr_subject_name(name: str, cert_type: str) -> str:
    return name[:64] if cert_type == "server" and len(name) > 64 else name


def _validate_safe_project_name(project: str, *, field_label: str = "Project") -> bool:
    if not isinstance(project, str) or not project.strip():
        output_error(
            "INVALID_PROJECT_NAME",
            exit_code=4,
            name=project,
            reason=f"{field_label} must not be empty or whitespace only.",
        )
        return False
    if len(project) > 64:
        output_error(
            "INVALID_PROJECT_NAME",
            exit_code=4,
            name=project,
            reason=f"{field_label} must be 64 characters or fewer.",
        )
        return False
    if os.sep in project or (os.altsep and os.altsep in project) or project.startswith("."):
        output_error(
            "INVALID_PROJECT_NAME",
            exit_code=4,
            name=project,
            reason=f"{field_label} must not contain path separators or start with '.'.",
        )
        return False
    if not _SAFE_PROJECT_NAME_PATTERN.fullmatch(project):
        output_error(
            "INVALID_PROJECT_NAME",
            exit_code=4,
            name=project,
            reason=f"{field_label} must match [A-Za-z0-9][A-Za-z0-9._-]*.",
        )
        return False
    return True


def _validate_org_name(org: str) -> bool:
    if not isinstance(org, str):
        invalid, reason = True, "org must be a string"
    else:
        invalid, reason = name_check(org, "org")
    if invalid:
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail=reason,
        )
        return False
    return True


def _validate_request_kind(kind: str) -> bool:
    if kind not in _REQUEST_KINDS:
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail=f"cert request kind must be one of: {', '.join(sorted(_REQUEST_KINDS))}",
        )
        return False
    return True


def _validate_request_kind_cert_type(kind: str, cert_type: str, cert_role: str = None) -> bool:
    if not _validate_request_kind(kind):
        return False
    if kind in _REQUEST_KIND_TO_CERT_TYPE:
        expected = _REQUEST_KIND_TO_CERT_TYPE[kind]
        if cert_type != expected:
            output_error_message(
                "INVALID_ARGS",
                "Invalid arguments.",
                _USAGE_HINT,
                exit_code=4,
                detail=f"request kind '{kind}' requires cert type '{expected}'",
            )
            return False
        return True

    if cert_type not in _USER_CERT_TYPES:
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail="request kind 'user' requires cert type one of: org_admin, lead, member",
        )
        return False
    if cert_role and _USER_ROLE_TO_CERT_TYPE.get(cert_role) != cert_type:
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail="user cert_role does not match request cert_type",
        )
        return False
    return True


def _validate_identity_name(name: str, cert_type: str) -> bool:
    if cert_type == "client":
        entity_type = "client"
    elif cert_type == "server":
        entity_type = "server"
    elif cert_type in ADMIN_CERT_TYPES:
        entity_type = "admin"
    else:
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail=f"invalid cert type '{cert_type}'; valid types: {', '.join(sorted(VALID_CERT_TYPES))}",
        )
        return False

    if not isinstance(name, str):
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail=f"name must be a string for entity_type={entity_type}",
        )
        return False

    invalid, reason = name_check(name, entity_type)
    if invalid:
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail=reason,
        )
        return False
    return True


def _validate_request_id(request_id: str) -> bool:
    if not isinstance(request_id, str) or not _REQUEST_ID_PATTERN.fullmatch(request_id):
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail="request_id must be a UUID hex string.",
        )
        return False
    return True


# ---------------------------------------------------------------------------
# cert init
# ---------------------------------------------------------------------------


def _backup_existing_ca(output_dir: str) -> None:
    """Move existing CA files into <output_dir>/.bak/<timestamp>/."""
    import time

    ts = time.strftime("%Y%m%dT%H%M%S")
    bak_dir = os.path.join(output_dir, ".bak", ts)
    os.makedirs(bak_dir, mode=0o700, exist_ok=True)
    for fname in ("rootCA.pem", "rootCA.key", "ca.json"):
        src = os.path.join(output_dir, fname)
        if os.path.exists(src):
            shutil.move(src, os.path.join(bak_dir, fname))


def handle_cert_init(args):
    # 1. --schema: handled before any I/O
    import nvflare.tool.cert.cert_cli as _cert_cli

    _cert_cli._ensure_parsers_initialized()
    handle_schema_flag(
        _cert_cli._cert_init_parser,
        "nvflare cert init",
        [
            "nvflare cert init --profile project_profile.yaml -o ./ca",
            "nvflare cert init --profile project_profile.yaml -o ./ca --org NVIDIA --force",
        ],
        sys.argv[1:],
    )

    # 2. Validate required args
    profile_path = getattr(args, "profile", None)
    project = getattr(args, "project", None)
    missing_flags = [
        flag
        for flag, is_missing in (
            ("--profile", not profile_path and not project),
            ("-o/--output-dir", not args.output_dir),
        )
        if is_missing
    ]
    if missing_flags:
        output_usage_error(
            _cert_cli._cert_init_parser, f"missing required argument(s): {', '.join(missing_flags)}", exit_code=4
        )
        return 1
    project_profile_name = None
    if profile_path:
        project_profile_name = _load_project_name_from_profile(profile_path)
        if project_profile_name is None:
            return 1
        project = project_profile_name
    elif not _validate_safe_project_name(project):
        return 1

    # 3. Resolve force
    force = args.force

    # 4. Resolve and create output dir
    output_dir = os.path.abspath(args.output_dir)
    try:
        os.makedirs(output_dir, mode=0o700, exist_ok=True)
    except OSError as e:
        output_error("OUTPUT_DIR_NOT_WRITABLE", path=output_dir, detail=str(e))
        return 1

    # 5. Check write permission
    if not os.access(output_dir, os.W_OK):
        output_error("OUTPUT_DIR_NOT_WRITABLE", path=output_dir, detail="directory is not writable")
        return 1

    # 6. Check for existing rootCA.key
    ca_key_path = os.path.join(output_dir, "rootCA.key")
    if os.path.exists(ca_key_path):
        if not force:
            output_error("CA_ALREADY_EXISTS", path=output_dir)
            return 1
        # --force: back up existing files
        _backup_existing_ca(output_dir)

    # 7. Generate key pair
    try:
        pri_key, pub_key = generate_keys()
    except Exception as e:
        output_error("CERT_GENERATION_FAILED", detail=str(e))
        return 1

    # 8. Generate self-signed CA certificate
    try:
        cert = CertBuilder._generate_cert(
            subject=project,
            subject_org=args.org,
            issuer=project,  # self-signed: issuer == subject
            signing_pri_key=pri_key,
            subject_pub_key=pub_key,
            valid_days=getattr(args, "valid_days", 3650) or 3650,
            ca=True,
        )
    except Exception as e:
        output_error("CERT_GENERATION_FAILED", detail=str(e))
        return 1

    # 9. Serialize
    pem_cert = serialize_cert(cert)
    pem_key = serialize_pri_key(pri_key)

    # 10. Write files
    rootca_path = os.path.join(output_dir, "rootCA.pem")
    ca_json_path = os.path.join(output_dir, "ca.json")
    written_paths = []
    try:
        written_paths.append(rootca_path)
        _write_file_nofollow(rootca_path, pem_cert)

        written_paths.append(ca_key_path)
        _write_private_key(ca_key_path, pem_key)

        created_at = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        ca_meta = {
            "project": project,
            "created_at": created_at,
        }
        if profile_path:
            ca_meta["project_profile"] = os.path.abspath(profile_path)
        written_paths.append(ca_json_path)
        _write_json_file(ca_json_path, ca_meta)
    except OSError as e:
        for path in written_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass
        output_error("OUTPUT_DIR_NOT_WRITABLE", path=output_dir, detail=str(e))
        return 1

    # 11. Compute valid_until for output
    valid_days_actual = getattr(args, "valid_days", 3650) or 3650
    valid_until_dt = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=valid_days_actual)
    valid_until_str = valid_until_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # 12. Output result
    result = {
        "ca_cert": rootca_path,
        "project": project,
        "subject_cn": project,
        "valid_until": valid_until_str,
    }
    if project_profile_name is not None:
        result["project_profile"] = os.path.abspath(profile_path)
    output_ok(result)
    return 0


# ---------------------------------------------------------------------------
# cert csr
# ---------------------------------------------------------------------------


def _generate_csr(name: str, org: str = None, role: str = None):
    """Generate RSA private key and CSR.

    The ``role`` is embedded in the CSR's UNSTRUCTURED_NAME field as the
    site-admin-proposed type for the Project Admin to either accept explicitly
    or override explicitly when signing.

    Returns:
        (pem_private_key: bytes, pem_csr: bytes)
    """
    pri_key, _ = generate_keys()

    subject = x509_name(cn_name=name, org_name=org, role=role)

    csr = (
        x509.CertificateSigningRequestBuilder().subject_name(subject).sign(pri_key, hashes.SHA256(), default_backend())
    )

    pem_key = serialize_pri_key(pri_key)
    pem_csr = csr.public_bytes(serialization.Encoding.PEM)
    return pem_key, pem_csr


def _write_private_key(path: str, pem_bytes: bytes) -> None:
    """Write private key PEM to path with 0600 permissions set atomically at creation."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as f:
            fd = -1  # ownership transferred to f
            f.write(pem_bytes)
    except Exception:
        if fd != -1:
            os.close(fd)
        try:
            os.unlink(path)
        except OSError:
            pass
        raise


def _write_file_nofollow(path: str, content: bytes, mode: int = 0o644) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, mode)
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as f:
            fd = -1  # ownership transferred to f
            f.write(content)
    except Exception:
        if fd != -1:
            os.close(fd)
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
        raise _UnsafeZipSourceError(f"zip source too large: {path}")
    return content


def _read_text_nofollow(path: str) -> str:
    return _read_file_nofollow(path).decode("utf-8")


def _write_json_file(path: str, data: dict) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(fd, 0o600)
    except Exception:
        os.close(fd)
        try:
            os.unlink(path)
        except OSError:
            pass
        raise
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        try:
            json.dump(data, f, indent=2)
        except Exception:
            try:
                os.unlink(path)
            except OSError:
                pass
            raise


def _utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _utc_ts(dt: datetime.datetime = None) -> str:
    return (dt or _utc_now()).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: str) -> str:
    return _sha256_bytes(_read_file_nofollow(path))


def _csr_public_key_sha256(csr: x509.CertificateSigningRequest) -> str:
    public_key_der = csr.public_key().public_bytes(
        serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    return _sha256_bytes(public_key_der)


def _cert_public_key_sha256(cert: x509.Certificate) -> str:
    public_key_der = cert.public_key().public_bytes(
        serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    return _sha256_bytes(public_key_der)


def _write_yaml_file(path: str, data: dict) -> None:
    import yaml

    content = yaml.safe_dump(data, sort_keys=False).encode("utf-8")
    _write_file_nofollow(path, content)


def _load_yaml_file(path: str) -> dict:
    import yaml

    data = None
    try:
        data = yaml.safe_load(_read_text_nofollow(path))
    except Exception as e:
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail=f"failed to parse yaml {path}: {e}",
        )
        return None
    if not isinstance(data, dict):
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail=f"yaml must be a mapping: {path}",
        )
        return None
    return data


def _load_project_name_from_profile(profile_path: str) -> str:
    if not os.path.isfile(profile_path):
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail=f"project profile file not found: {profile_path}",
        )
        return None
    profile = _load_yaml_file(profile_path)
    if profile is None:
        return None
    project_name = profile.get("name")
    if not project_name:
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail="project profile missing required field: name",
        )
        return None
    if not _validate_safe_project_name(project_name, field_label="Project profile name"):
        return None
    return project_name


def _safe_zip_names(zf: zipfile.ZipFile) -> list:
    names = []
    seen = set()
    for info in zf.infolist():
        name = info.filename
        mode = info.external_attr >> 16
        normalized = posixpath.normpath(name)
        parts = normalized.split("/")
        if (
            not name
            or name == "."
            or name.endswith("/")
            or os.path.isabs(name)
            or "\\" in name
            or normalized != name
            or normalized.startswith("../")
            or ".." in parts
            or posixpath.basename(name) != name
            or name in seen
            or info.is_dir()
            or info.file_size > _MAX_ZIP_MEMBER_SIZE
            or stat.S_IFMT(mode) in {stat.S_IFDIR, stat.S_IFLNK}
        ):
            output_error_message(
                "INVALID_ARGS",
                "Invalid arguments.",
                _USAGE_HINT,
                exit_code=4,
                detail=f"unsafe zip member or path traversal: {name}",
            )
            return None
        if name.lower().endswith(".key"):
            output_error_message(
                "INVALID_ARGS",
                "Invalid arguments.",
                _USAGE_HINT,
                exit_code=4,
                detail=f"request zip must not contain private keys: {name}",
            )
            return None
        seen.add(name)
        names.append(name)
    return names


def _read_zip_member_limited(zf: zipfile.ZipFile, member: str) -> Optional[bytes]:
    with zf.open(member) as member_file:
        content = member_file.read(_MAX_ZIP_MEMBER_SIZE + 1)
    if len(content) > _MAX_ZIP_MEMBER_SIZE:
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail=f"zip member exceeds size limit: {member}",
        )
        return None
    return content


def _read_zip_source_nofollow(src_path: str) -> bytes:
    src_stat = os.lstat(src_path)
    if os.path.islink(src_path) or not stat.S_ISREG(src_stat.st_mode):
        raise _UnsafeZipSourceError(f"not a regular file: {src_path}")
    read_flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        read_flags |= os.O_NOFOLLOW
    src_fd = os.open(src_path, read_flags)
    with os.fdopen(src_fd, "rb") as src_file:
        opened_stat = os.fstat(src_file.fileno())
        if (
            src_stat.st_ino != opened_stat.st_ino
            or src_stat.st_dev != opened_stat.st_dev
            or not stat.S_ISREG(opened_stat.st_mode)
        ):
            raise _UnsafeZipSourceError(f"unsafe zip source changed while reading: {src_path}")
        if opened_stat.st_size > _MAX_ZIP_MEMBER_SIZE:
            raise _UnsafeZipSourceError(f"zip source too large: {src_path}")
        content = src_file.read(_MAX_ZIP_MEMBER_SIZE + 1)
        if len(content) > _MAX_ZIP_MEMBER_SIZE:
            raise _UnsafeZipSourceError(f"zip source too large: {src_path}")
        return content


def _write_zip_nofollow(zip_path: str, members: dict, force: bool = False) -> bool:
    if os.path.exists(zip_path) and not force:
        output_error("CERT_ALREADY_EXISTS", path=zip_path)
        return False
    parent = os.path.dirname(os.path.abspath(zip_path)) or "."
    try:
        os.makedirs(parent, mode=0o700, exist_ok=True)
    except OSError as e:
        output_error("OUTPUT_DIR_NOT_WRITABLE", path=parent, detail=str(e))
        return False

    prepared_members = []
    for arcname, src_path in members.items():
        if arcname.lower().endswith(".key"):
            output_error_message(
                "INVALID_ARGS",
                "Invalid arguments.",
                _USAGE_HINT,
                exit_code=4,
                detail=f"zip must not contain private keys: {arcname}",
            )
            return False
        try:
            prepared_members.append((arcname, _read_zip_source_nofollow(src_path)))
        except _UnsafeZipSourceError as e:
            output_error_message(
                "INVALID_ARGS",
                "Invalid arguments.",
                _USAGE_HINT,
                exit_code=4,
                detail=f"unsafe zip source: {e}",
            )
            return False
        except OSError as e:
            output_error("OUTPUT_DIR_NOT_WRITABLE", path=src_path, detail=str(e))
            return False

    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(zip_path, flags, 0o600)
        with os.fdopen(fd, "wb") as f:
            with zipfile.ZipFile(f, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for arcname, content in prepared_members:
                    zf.writestr(arcname, content)
    except Exception as e:
        try:
            if os.path.exists(zip_path) and not os.path.islink(zip_path):
                os.remove(zip_path)
        except OSError:
            pass
        output_error("OUTPUT_DIR_NOT_WRITABLE", path=zip_path, detail=str(e))
        return False
    return True


def _read_json(path: str) -> dict:
    data = None
    try:
        data = json.loads(_read_text_nofollow(path))
    except Exception as e:
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail=f"failed to parse json {path}: {e}",
        )
        return data
    if not isinstance(data, dict):
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail=f"json must be a mapping: {path}",
        )
        return None
    return data


def _audit_root(dirname: str) -> str:
    return os.path.expanduser(os.path.join("~", ".nvflare", dirname))


def _write_request_audit(request_id: str, audit_record: dict) -> str:
    audit_dir = os.path.join(_audit_root("cert_requests"), request_id)
    os.makedirs(audit_dir, mode=0o700, exist_ok=True)
    audit_path = os.path.join(audit_dir, "audit.json")
    _write_json_file(audit_path, audit_record)
    return audit_path


def _write_approve_audit(request_id: str, audit_record: dict) -> str:
    audit_dir = _audit_root("cert_approves")
    os.makedirs(audit_dir, mode=0o700, exist_ok=True)
    audit_path = os.path.join(audit_dir, f"{request_id}.json")
    _write_json_file(audit_path, audit_record)
    return audit_path


def _try_write_request_audit(request_id: str, audit_record: dict):
    try:
        return _write_request_audit(request_id, audit_record)
    except OSError as e:
        print_human(f"Warning: could not write request audit record: {e}")
        return None


def _try_write_approve_audit(request_id: str, audit_record: dict):
    try:
        return _write_approve_audit(request_id, audit_record)
    except OSError as e:
        print_human(f"Warning: could not write approval audit record: {e}")
        return None


def _backup_existing_csr(out_dir: str, name: str) -> None:
    """Move existing <name>.key and <name>.csr to .bak/<timestamp>/ before overwrite."""
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S")
    bak_dir = os.path.join(out_dir, ".bak", timestamp)
    os.makedirs(bak_dir, mode=0o700, exist_ok=True)
    for ext in ("key", "csr"):
        src = os.path.join(out_dir, f"{name}.{ext}")
        if os.path.exists(src):
            shutil.move(src, os.path.join(bak_dir, f"{name}.{ext}"))


def _load_single_site_yaml(path: str) -> dict:
    import yaml

    if not os.path.isfile(path):
        output_error("PROJECT_FILE_NOT_FOUND", path=path)
        return None
    data = None
    try:
        data = yaml.safe_load(_read_text_nofollow(path))
    except Exception as e:
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail=f"failed to parse site yaml {path}: {e}",
        )
        return None
    if not isinstance(data, dict):
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail=f"site yaml must be a mapping: {path}",
        )
        return None
    name = data.get("name")
    org = data.get("org")
    cert_type = data.get("type")
    if not name or not org or not cert_type:
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail="site yaml must contain: name, org, type",
        )
        return None
    if cert_type not in _VALID_CERT_TYPES:
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail=f"invalid cert type '{cert_type}'; valid types: {', '.join(sorted(VALID_CERT_TYPES))}",
        )
        return None
    return {"name": name, "org": org, "cert_type": cert_type}


def generate_csr_files(name: str, org: str, cert_type: str, output_dir: str, force: bool = False) -> dict:
    """Generate a participant private key and CSR using the existing cert logic."""
    name = name.strip()
    if cert_type not in _VALID_CERT_TYPES:
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail=f"invalid cert type '{cert_type}'; valid types: {', '.join(sorted(VALID_CERT_TYPES))}",
        )
        return {}
    if not _validate_safe_cert_name(name, field_label="Name", max_length=_cert_name_max_length(cert_type)):
        return {}

    out_dir = os.path.abspath(output_dir)
    try:
        os.makedirs(out_dir, mode=0o700, exist_ok=True)
    except OSError as e:
        output_error("OUTPUT_DIR_NOT_WRITABLE", path=out_dir, detail=str(e))
        return {}

    if not os.access(out_dir, os.W_OK):
        output_error("OUTPUT_DIR_NOT_WRITABLE", path=out_dir, detail="directory is not writable")
        return {}

    key_path = os.path.join(out_dir, f"{name}.key")
    csr_path = os.path.join(out_dir, f"{name}.csr")

    if os.path.exists(key_path) and not force:
        output_error("KEY_ALREADY_EXISTS", path=key_path)
        return {}

    if force and (os.path.exists(key_path) or os.path.exists(csr_path)):
        _backup_existing_csr(out_dir, name)

    try:
        pem_key, pem_csr = _generate_csr(_csr_subject_name(name, cert_type), org, cert_type)
    except Exception as e:
        output_error("CSR_GENERATION_FAILED", detail=str(e))
        return {}

    try:
        _write_private_key(key_path, pem_key)
        _write_file_nofollow(csr_path, pem_csr)
    except OSError as e:
        try:
            if os.path.exists(key_path) and not os.path.islink(key_path):
                os.remove(key_path)
        except OSError:
            pass
        output_error("OUTPUT_DIR_NOT_WRITABLE", path=out_dir, detail=str(e))
        return {}

    csr = x509.load_pem_x509_csr(pem_csr, default_backend())
    return {
        "name": name,
        "org": org,
        "cert_type": cert_type,
        "output_dir": out_dir,
        "key_path": key_path,
        "csr_path": csr_path,
        "csr_sha256": _sha256_bytes(pem_csr),
        "public_key_sha256": _csr_public_key_sha256(csr),
    }


def handle_cert_csr(args):
    if getattr(args, "schema", False):
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            "Use 'nvflare cert request --schema' for the public distributed provisioning request schema.",
            exit_code=4,
            detail="'nvflare cert csr' is not a public CLI command",
        )
        return 1

    # 2. Resolve inputs (either --project-file or explicit args)
    site = None
    if getattr(args, "project_file", None):
        # Mutual exclusivity check before touching the filesystem
        if getattr(args, "name", None) or getattr(args, "org", None) or getattr(args, "cert_type", None):
            output_error_message(
                "INVALID_ARGS",
                "Invalid arguments.",
                _USAGE_HINT,
                exit_code=4,
                detail="use either --project-file or --name/--org/--type",
            )
            return 1
        site = _load_single_site_yaml(args.project_file)
        if site is None:
            return 1

    # 3. Validate required args (-o is required in all modes; -n/-t only without --project-file)
    missing_flags = []
    if not getattr(args, "output_dir", None):
        missing_flags.append("-o/--output-dir")
    if site is None and not getattr(args, "name", None):
        missing_flags.append("-n/--name")
    if site is None and not getattr(args, "cert_type", None):
        missing_flags.append("-t/--type")
    if missing_flags:
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail=f"missing required argument(s): {', '.join(missing_flags)}",
        )
        return 1

    name = (site["name"] if site else args.name).strip()
    org = site["org"] if site else getattr(args, "org", None)
    cert_type = site["cert_type"] if site else getattr(args, "cert_type", None)
    csr_result = generate_csr_files(
        name=name,
        org=org,
        cert_type=cert_type,
        output_dir=args.output_dir,
        force=args.force,
    )
    if not csr_result:
        return 1

    # 12. Emit output
    result = {
        "name": csr_result["name"],
        "key": csr_result["key_path"],
        "csr": csr_result["csr_path"],
        "next_step": f"Send {csr_result['name']}.csr to your Project Admin for signing.",
    }
    output_ok(result)
    return 0


# ---------------------------------------------------------------------------
# cert sign
# ---------------------------------------------------------------------------


def _get_cn(name: x509.Name) -> str:
    """Extract COMMON_NAME value from an x509.Name, or empty string if absent."""
    for attr in name:
        if attr.oid == NameOID.COMMON_NAME:
            return attr.value
    return ""


def _get_csr_role(csr: x509.CertificateSigningRequest) -> str:
    role_attrs = csr.subject.get_attributes_for_oid(NameOID.UNSTRUCTURED_NAME)
    if not role_attrs:
        return ""
    return role_attrs[0].value


def _get_cert_not_valid_after(cert: x509.Certificate) -> datetime.datetime:
    try:
        return cert.not_valid_after_utc
    except AttributeError:
        not_after = cert.not_valid_after
        return (
            not_after.replace(tzinfo=datetime.timezone.utc)
            if not_after.tzinfo is None
            else not_after.astimezone(datetime.timezone.utc)
        )


def _validate_signing_ca(ca_cert: x509.Certificate, now: datetime.datetime) -> datetime.datetime:
    try:
        basic_constraints = ca_cert.extensions.get_extension_for_class(x509.BasicConstraints).value
    except x509.ExtensionNotFound:
        output_error("CERT_SIGNING_FAILED", reason="CA certificate is missing BasicConstraints")
        return None

    if not basic_constraints.ca:
        output_error("CERT_SIGNING_FAILED", reason="CA certificate is not a CA certificate")
        return None

    ca_not_valid_after = _get_cert_not_valid_after(ca_cert)
    if ca_not_valid_after <= now:
        output_error(
            "CERT_SIGNING_FAILED",
            reason=f"CA certificate expired at {ca_not_valid_after.strftime('%Y-%m-%dT%H:%M:%SZ')}",
        )
        return None

    return ca_not_valid_after


def _build_signed_cert(
    csr: x509.CertificateSigningRequest,
    ca_cert: x509.Certificate,
    ca_key,
    cert_type: str,
    now: datetime.datetime,
    not_valid_after: datetime.datetime,
    server_default_host: str = None,
    server_additional_hosts=None,
) -> x509.Certificate:
    """Build and sign a certificate from a CSR using the CA key.

    The subject is rebuilt from safe CSR fields only; UNSTRUCTURED_NAME (role) is always
    set from cert_type (the Project Admin's authoritative -t argument), never from the CSR.
    """
    subject_cn = _get_cn(csr.subject)

    if cert_type in ADMIN_CERT_TYPES:
        key_usage_kwargs = dict(
            digital_signature=True,
            content_commitment=True,
            key_encipherment=False,
            data_encipherment=False,
            key_agreement=False,
            key_cert_sign=False,
            crl_sign=False,
            encipher_only=False,
            decipher_only=False,
        )
        eku_oids = [x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH]
    else:
        # client and server
        key_usage_kwargs = dict(
            digital_signature=True,
            content_commitment=False,
            key_encipherment=True,
            data_encipherment=False,
            key_agreement=False,
            key_cert_sign=False,
            crl_sign=False,
            encipher_only=False,
            decipher_only=False,
        )
        eku_oids = [
            (
                x509.oid.ExtendedKeyUsageOID.SERVER_AUTH
                if cert_type == "server"
                else x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH
            )
        ]

    # Rebuild subject from safe OIDs only; do NOT copy CSR subject verbatim.
    _SAFE_OIDS = {
        NameOID.COMMON_NAME,
        NameOID.ORGANIZATION_NAME,
        NameOID.COUNTRY_NAME,
        NameOID.STATE_OR_PROVINCE_NAME,
        NameOID.LOCALITY_NAME,
    }
    seen_oids = set()
    subject_org = None
    for attr in csr.subject:
        if attr.oid not in _SAFE_OIDS:
            continue
        if attr.oid in seen_oids:
            raise ValueError(f"CSR contains duplicate subject attribute for OID '{attr.oid._name}'")
        seen_oids.add(attr.oid)
        if attr.oid == NameOID.ORGANIZATION_NAME:
            subject_org = attr.value

    issuer_cn = ca_cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value
    extra_extensions = [
        (x509.BasicConstraints(ca=False, path_length=None), True),
        (x509.KeyUsage(**key_usage_kwargs), True),
        (x509.ExtendedKeyUsage(eku_oids), False),
    ]
    return CertBuilder._generate_cert(
        subject=subject_cn,
        subject_org=subject_org,
        issuer=issuer_cn,
        signing_pri_key=ca_key,
        subject_pub_key=csr.public_key(),
        ca=False,
        role=cert_type,
        server_default_host=server_default_host if cert_type == "server" else None,
        server_additional_hosts=server_additional_hosts if cert_type == "server" else None,
        not_valid_before=now,
        not_valid_after=not_valid_after,
        extra_extensions=extra_extensions,
    )


def _load_and_validate_csr(csr_path: str) -> x509.CertificateSigningRequest:
    if not os.path.exists(csr_path):
        output_error("CSR_NOT_FOUND", path=csr_path)
        return None
    if not os.path.isfile(csr_path):
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail=f"-r/--csr must be a file path, not a directory: {csr_path}",
        )
        return None

    csr_data = None
    try:
        csr_data = _read_file_nofollow(csr_path)
    except Exception as e:
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail=f"failed to read CSR {csr_path}: {e}",
        )
        return None
    csr = None
    try:
        csr = x509.load_pem_x509_csr(csr_data, default_backend())
    except Exception as e:
        output_error("INVALID_CSR", path=csr_path, detail=str(e))
        return None

    if not csr.is_signature_valid:
        output_error("INVALID_CSR", path=csr_path)
        return None
    return csr


def _resolve_sign_cert_type(csr: x509.CertificateSigningRequest, cert_type: str, accept_csr_role: bool) -> str:
    if cert_type and accept_csr_role:
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail="use either -t/--type or --accept-csr-role, not both",
        )
        return None

    if accept_csr_role:
        cert_type = _get_csr_role(csr)
        if not cert_type:
            output_error_message(
                "INVALID_ARGS",
                "Invalid arguments.",
                _USAGE_HINT,
                exit_code=4,
                detail=(
                    "CSR does not contain a proposed role; provide -t/--type for this internal signing helper "
                    "or create a public request with 'nvflare cert request --participant <user.yaml>'"
                ),
            )
            return None
    elif not cert_type:
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail="specify either -t/--type to override the role or --accept-csr-role to trust the CSR role",
        )
        return None

    if cert_type not in _VALID_CERT_TYPES:
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail=f"invalid cert type '{cert_type}'; valid types: {', '.join(sorted(VALID_CERT_TYPES))}",
        )
        return None
    return cert_type


def sign_csr_files(
    csr_path: str,
    ca_dir: str,
    output_dir: str,
    cert_type: str = None,
    accept_csr_role: bool = False,
    valid_days: int = 1095,
    force: bool = False,
    csr: x509.CertificateSigningRequest = None,
    server_default_host: str = None,
    server_additional_hosts=None,
) -> dict:
    """Sign a CSR using the existing cert signing logic and write cert/rootCA files."""
    ca_key_path = os.path.join(ca_dir, "rootCA.key")
    ca_cert_path = os.path.join(ca_dir, "rootCA.pem")
    ca_json_path = os.path.join(ca_dir, "ca.json")
    for path in (ca_key_path, ca_cert_path, ca_json_path):
        if not os.path.exists(path):
            output_error("CA_NOT_FOUND", ca_dir=ca_dir)
            return None

    if csr is None:
        csr = _load_and_validate_csr(csr_path)
    if csr is None:
        return None
    cert_type = _resolve_sign_cert_type(csr, cert_type, accept_csr_role)
    if cert_type is None:
        return None

    subject_cn = _get_cn(csr.subject)
    if not _validate_safe_cert_name(subject_cn, field_label="CSR subject CN"):
        return None
    output_filename = f"{subject_cn}.crt"

    output_dir = os.path.abspath(output_dir)
    try:
        os.makedirs(output_dir, mode=0o700, exist_ok=True)
    except OSError as e:
        output_error("OUTPUT_DIR_NOT_WRITABLE", path=output_dir, detail=str(e))
        return None

    if not os.access(output_dir, os.W_OK):
        output_error("OUTPUT_DIR_NOT_WRITABLE", path=output_dir, detail="directory is not writable")
        return None

    cert_out_path = os.path.join(output_dir, output_filename)
    rootca_out_path = os.path.join(output_dir, "rootCA.pem")

    if os.path.exists(cert_out_path) and not force:
        output_error("CERT_ALREADY_EXISTS", path=cert_out_path)
        return None
    if os.path.exists(rootca_out_path) and not force:
        output_error("ROOTCA_ALREADY_EXISTS", path=rootca_out_path)
        return None

    try:
        rootca_bytes = _read_file_nofollow(ca_cert_path)
        ca_cert = x509.load_pem_x509_certificate(rootca_bytes, default_backend())
        ca_key = load_private_key_file(ca_key_path)
    except Exception as e:
        output_error("CA_LOAD_FAILED", ca_dir=ca_dir, detail=str(e))
        return None

    now = _utc_now()
    ca_not_valid_after = _validate_signing_ca(ca_cert, now)
    if ca_not_valid_after is None:
        return None

    valid_days = valid_days or 1095
    requested_not_valid_after = now + datetime.timedelta(days=valid_days)
    leaf_not_valid_after = min(requested_not_valid_after, ca_not_valid_after)
    try:
        signed_cert = _build_signed_cert(
            csr=csr,
            ca_cert=ca_cert,
            ca_key=ca_key,
            cert_type=cert_type,
            now=now,
            not_valid_after=leaf_not_valid_after,
            server_default_host=server_default_host,
            server_additional_hosts=server_additional_hosts,
        )
    except Exception as e:
        output_error("CERT_SIGNING_FAILED", reason=str(e))
        return None

    try:
        signed_cert_pem = serialize_cert(signed_cert)
        _write_file_nofollow(cert_out_path, signed_cert_pem)
        _write_file_nofollow(rootca_out_path, rootca_bytes)
    except OSError as e:
        for path in (cert_out_path, rootca_out_path):
            try:
                if os.path.exists(path) and not os.path.islink(path):
                    os.remove(path)
            except OSError:
                pass
        output_error("CERT_OUTPUT_WRITE_FAILED", path=output_dir, detail=str(e))
        return None

    try:
        valid_until_dt = signed_cert.not_valid_after_utc
    except AttributeError:
        valid_until_dt = signed_cert.not_valid_after
    valid_until = valid_until_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    return {
        "signed_cert": cert_out_path,
        "rootca": rootca_out_path,
        "subject_cn": subject_cn,
        "cert_type": cert_type,
        "serial": hex(signed_cert.serial_number),
        "valid_until": valid_until,
        "certificate": signed_cert,
        "certificate_sha256": _sha256_file(cert_out_path),
        "rootca_sha256": _sha256_file(rootca_out_path),
        "rootca_fingerprint_sha256": cert_fingerprint_sha256(ca_cert),
        "public_key_sha256": _cert_public_key_sha256(signed_cert),
    }


def handle_cert_sign(args):
    if getattr(args, "schema", False):
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            "Use 'nvflare cert approve --schema' for the public distributed provisioning approval schema.",
            exit_code=4,
            detail="'nvflare cert sign' is not a public CLI command",
        )
        return 1

    # 2. Validate required args and signer decision mode
    missing_flags = [
        flag
        for flag, attr in (
            ("-r/--csr", "csr_path"),
            ("-c/--ca-dir", "ca_dir"),
            ("-o/--output-dir", "output_dir"),
        )
        if not getattr(args, attr, None)
    ]
    if missing_flags:
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail=f"missing required argument(s): {', '.join(missing_flags)}",
        )
        return 1
    csr = _load_and_validate_csr(args.csr_path)
    if csr is None:
        return 1
    cert_type = _resolve_sign_cert_type(csr, getattr(args, "cert_type", None), getattr(args, "accept_csr_role", False))
    if cert_type is None:
        return 1
    subject_cn = _get_cn(csr.subject)
    if getattr(args, "accept_csr_role", False) and not cli_output.is_json_mode() and sys.stdin.isatty():
        if not prompt_yn(f"CSR for '{subject_cn}' proposes role '{cert_type}'. Sign using this CSR role?"):
            print_human("Cancelled.")
            return 1

    sign_result = sign_csr_files(
        csr_path=args.csr_path,
        ca_dir=args.ca_dir,
        output_dir=args.output_dir,
        cert_type=cert_type,
        accept_csr_role=False,
        valid_days=getattr(args, "valid_days", 1095),
        force=args.force,
        csr=csr,
    )
    if sign_result is None:
        return 1

    # 12. Output result
    next_step = (
        "This internal command writes a signed certificate and rootCA.pem.\n"
        "For the public distributed provisioning workflow, use:\n"
        "  nvflare cert request --participant <participant.yaml>\n"
        "  nvflare cert approve <request.zip> --ca-dir <ca-dir> --profile <project_profile.yaml>\n"
        "  nvflare package <signed.zip>"
    )
    result = {
        "signed_cert": sign_result["signed_cert"],
        "rootca": sign_result["rootca"],
        "subject_cn": sign_result["subject_cn"],
        "cert_type": sign_result["cert_type"],
        "serial": sign_result["serial"],
        "valid_until": sign_result["valid_until"],
        "next_step": next_step,
    }
    output_ok(result)
    return 0


# ---------------------------------------------------------------------------
# cert request / approve
# ---------------------------------------------------------------------------


def _normalize_cert_role(role: str) -> str:
    if not isinstance(role, str):
        return None
    return _USER_ROLE_TO_CERT_TYPE.get(role.strip())


def _valid_user_role_names() -> str:
    return ", ".join(sorted(_USER_ROLE_TO_CERT_TYPE))


def _validate_port(value, field_label: str) -> bool:
    if not isinstance(value, int) or isinstance(value, bool) or not (1 <= value <= 65535):
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail=f"{field_label} must be an integer from 1 to 65535",
        )
        return False
    return True


def _copy_mapping(value: dict) -> dict:
    return copy.deepcopy(value)


def _derive_identity_from_participant(project_name: str, participant: dict) -> dict:
    if not isinstance(participant, dict):
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail="participant definition participants[0] must be a mapping",
        )
        return None
    name = participant.get("name")
    org = participant.get("org")
    participant_type = participant.get("type")
    if not name or not org or not participant_type:
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail="participant definition participants[0] must contain: name, org, type",
        )
        return None
    if not _validate_safe_project_name(project_name):
        return None
    if not _validate_org_name(org):
        return None

    if participant_type == "client":
        identity = {"kind": "site", "name": name, "cert_role": None, "cert_type": "client"}
    elif participant_type == "server":
        identity = {"kind": "server", "name": name, "cert_role": None, "cert_type": "server"}
    elif participant_type == "admin":
        cert_role = _normalize_cert_role(participant.get("role"))
        if not cert_role:
            output_error_message(
                "INVALID_ARGS",
                "Invalid arguments.",
                _USAGE_HINT,
                exit_code=4,
                detail=f"admin participant role must be one of: {_valid_user_role_names()}",
            )
            return None
        identity = {"kind": "user", "name": name, "cert_role": cert_role, "cert_type": cert_role}
    else:
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail="participant type must be one of: client, server, admin",
        )
        return None

    name = str(identity["name"]).strip()
    if not _validate_safe_cert_name(name, field_label="Name", max_length=_cert_name_max_length(identity["cert_type"])):
        return None
    if not _validate_identity_name(name, identity["cert_type"]):
        return None
    identity["name"] = name
    return identity


def _validate_participant_connection_fields(participant: dict, identity: dict) -> bool:
    if PropKey.LISTENING_HOST in participant:
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail=(
                "listening_host is not supported by distributed provisioning yet; "
                "use centralized provisioning for third-party listener certificates"
            ),
        )
        return False

    if identity["kind"] == "server":
        if not _validate_port(participant.get("fed_learn_port"), "server fed_learn_port"):
            return False
        if not _validate_port(participant.get("admin_port"), "server admin_port"):
            return False
        conn_sec = participant.get("connection_security")
        if conn_sec is not None and (
            not isinstance(conn_sec, str) or conn_sec.strip().lower() not in _VALID_CONNECTION_SECURITY
        ):
            output_error_message(
                "INVALID_ARGS",
                "Invalid arguments.",
                _USAGE_HINT,
                exit_code=4,
                detail="server connection_security must be one of: clear, tls, mtls",
            )
            return False
        return True

    server = participant.get("server")
    if not isinstance(server, dict):
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail="client and admin participant definitions must contain a server mapping",
        )
        return False
    host = server.get("host")
    if not isinstance(host, str) or not host.strip():
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail="server.host must be a non-empty string",
        )
        return False
    if not _validate_port(server.get("fed_learn_port"), "server.fed_learn_port"):
        return False
    if not _validate_port(server.get("admin_port"), "server.admin_port"):
        return False
    return True


def _validate_participant_with_centralized_rules(participant: dict) -> bool:
    try:
        participant_from_dict(_copy_mapping(participant))
        return True
    except Exception as e:
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail=f"participant definition is invalid: {e}",
        )
        return False


def _load_participant_definition(path: str):
    data = _load_yaml_file(path)
    if data is None:
        return None, None
    project_name = data.get("name")
    participants = data.get("participants")
    if not project_name or not isinstance(participants, list):
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail="participant definition must contain top-level name and participants",
        )
        return None, None
    if len(participants) != 1:
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail="participant definition must contain exactly one participants[0] entry",
        )
        return None, None

    normalized = _copy_mapping(data)
    participant = _copy_mapping(participants[0])
    identity = _derive_identity_from_participant(project_name, participant)
    if identity is None:
        return None, None
    if identity["kind"] == "user":
        participant["role"] = identity["cert_role"]
    if not _validate_participant_with_centralized_rules(participant):
        return None, None
    if not _validate_participant_connection_fields(participant, identity):
        return None, None
    normalized["name"] = project_name
    normalized["participants"] = [participant]
    return normalized, identity


def _is_project_shaped_site_meta(site_meta: dict) -> bool:
    return isinstance(site_meta, dict) and isinstance(site_meta.get("participants"), list)


def _site_identity_from_metadata(site_meta: dict) -> dict:
    if not isinstance(site_meta, dict):
        return {}
    if _is_project_shaped_site_meta(site_meta):
        participants = site_meta.get("participants") or []
        if len(participants) != 1 or not isinstance(participants[0], dict):
            return {}
        participant = participants[0]
        project_name = site_meta.get("name")
        identity = _derive_identity_from_participant(project_name, participant)
        if identity is None:
            return {}
        return {
            "project": project_name,
            "name": identity["name"],
            "org": participant.get("org"),
            "kind": identity["kind"],
            "cert_type": identity["cert_type"],
            "cert_role": identity["cert_role"],
        }

    return {
        "project": site_meta.get("project"),
        "name": site_meta.get("name"),
        "org": site_meta.get("org"),
        "kind": site_meta.get("kind"),
        "cert_type": site_meta.get("type") or site_meta.get("cert_type"),
        "cert_role": site_meta.get("cert_role"),
    }


def _build_sanitized_approval_site(local_site: dict) -> dict:
    if not _is_project_shaped_site_meta(local_site):
        return _copy_mapping(local_site)
    sanitized = _copy_mapping(local_site)
    sanitized.pop("builders", None)
    sanitized.pop("packager", None)
    participant = sanitized["participants"][0]
    participant.pop("connection_security", None)
    return sanitized


def _server_cert_san_fields(site_meta: dict, request_meta: dict):
    if request_meta.get("cert_type") != "server" or not _is_project_shaped_site_meta(site_meta):
        return None, None
    participant = site_meta["participants"][0]
    try:
        server = participant_from_dict(_copy_mapping(participant))
    except Exception as e:
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail=f"server participant definition is invalid: {e}",
        )
        return None
    return server.get_default_host(), server.get_prop(PropKey.HOST_NAMES)


def _build_site_metadata(request_meta: dict) -> dict:
    site_meta = {
        "name": request_meta["name"],
        "org": request_meta["org"],
        "type": request_meta["cert_type"],
        "project": request_meta["project"],
        "kind": request_meta["kind"],
    }
    if request_meta.get("cert_role"):
        site_meta["cert_role"] = request_meta["cert_role"]
    return site_meta


def handle_cert_request(args):
    import nvflare.tool.cert.cert_cli as _cert_cli

    _cert_cli._ensure_parsers_initialized()
    handle_schema_flag(
        _cert_cli._cert_request_parser,
        "nvflare cert request",
        [
            "nvflare cert request --participant hospital-a.yaml",
            "nvflare cert request -p alice.yaml --out ./requests/alice",
        ],
        sys.argv[1:],
    )

    participant_path = getattr(args, "participant", None)
    if not participant_path:
        output_usage_error(
            _cert_cli._cert_request_parser,
            "missing required argument: -p/--participant",
            exit_code=4,
        )
        return 1

    # These attrs are not registered in the CLI parser; the check only fires when
    # handle_cert_request is called programmatically with a hand-built args namespace.
    conflicting = [
        flag
        for flag, attr in (("--org", "org"), ("--project", "project"), ("--name", "name"), ("--type", "cert_type"))
        if getattr(args, attr, None)
    ]
    if conflicting:
        output_usage_error(
            _cert_cli._cert_request_parser,
            f"--participant is incompatible with: {', '.join(conflicting)}",
            exit_code=4,
        )
        return 1

    local_site, identity = _load_participant_definition(participant_path)
    if local_site is None or identity is None:
        return 1
    project = local_site["name"]
    org = local_site["participants"][0]["org"]
    name = identity["name"].strip()

    request_dir = os.path.abspath(getattr(args, "output_dir", None) or os.path.join(".", name))
    if os.path.exists(request_dir) and not os.path.isdir(request_dir):
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail=f"--out must be a directory path: {request_dir}",
        )
        return 1

    request_zip_path = os.path.join(request_dir, f"{name}.request.zip")
    if os.path.exists(request_zip_path) and not getattr(args, "force", False):
        output_error("CERT_ALREADY_EXISTS", path=request_zip_path)
        return 1

    csr_result = generate_csr_files(
        name=name,
        org=org,
        cert_type=identity["cert_type"],
        output_dir=request_dir,
        force=getattr(args, "force", False),
    )
    if not csr_result:
        return 1

    request_id = uuid.uuid4().hex
    created_at = _utc_ts()
    request_meta = {
        "artifact_type": _REQUEST_ARTIFACT_TYPE,
        "schema_version": _ARTIFACT_VERSION,
        "request_id": request_id,
        "created_at": created_at,
        "project": project,
        "name": name,
        "org": org,
        "kind": identity["kind"],
        "cert_type": identity["cert_type"],
        "cert_role": identity["cert_role"],
        "csr_sha256": csr_result["csr_sha256"],
        "public_key_sha256": csr_result["public_key_sha256"],
    }
    site_meta = local_site or _build_site_metadata(request_meta)
    approval_site_meta = _build_sanitized_approval_site(site_meta)

    request_json_path = os.path.join(request_dir, "request.json")
    site_yaml_path = os.path.join(request_dir, "site.yaml")
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            approval_site_yaml_path = os.path.join(tmp_dir, "site.yaml")
            _write_yaml_file(approval_site_yaml_path, approval_site_meta)
            approval_site_yaml_sha256 = _sha256_file(approval_site_yaml_path)
            request_meta["site_yaml_sha256"] = approval_site_yaml_sha256
            _write_json_file(request_json_path, request_meta)
            _write_yaml_file(site_yaml_path, site_meta)
            if not _write_zip_nofollow(
                request_zip_path,
                {
                    "request.json": request_json_path,
                    "site.yaml": approval_site_yaml_path,
                    f"{name}.csr": csr_result["csr_path"],
                },
                force=getattr(args, "force", False),
            ):
                return 1
    except _UnsafeZipSourceError as e:
        output_error_message(
            "INVALID_ARGS",
            "Request file too large to process.",
            _USAGE_HINT,
            exit_code=4,
            detail=str(e),
        )
        return 1
    except OSError as e:
        output_error("OUTPUT_DIR_NOT_WRITABLE", path=request_dir, detail=str(e))
        return 1

    audit_record = {
        "schema_version": _ARTIFACT_VERSION,
        "request": request_meta,
        "request_dir": request_dir,
        "request_zip_path": request_zip_path,
        "private_key_path": csr_result["key_path"],
        "csr_path": csr_result["csr_path"],
        "site_yaml_path": site_yaml_path,
        "hashes": {
            "request_json_sha256": _sha256_file(request_json_path),
            "site_yaml_sha256": approval_site_yaml_sha256,
            "local_site_yaml_sha256": _sha256_file(site_yaml_path),
            "csr_sha256": _sha256_file(csr_result["csr_path"]),
            "request_zip_sha256": _sha256_file(request_zip_path),
            "public_key_sha256": csr_result["public_key_sha256"],
        },
    }
    audit_path = _try_write_request_audit(request_id, audit_record)

    output_ok(
        {
            "name": name,
            "project": project,
            "request_zip": request_zip_path,
            "request_id": request_id,
            "audit": audit_path or "(not written)",
            "next_step": f"Send {os.path.basename(request_zip_path)} to your Project Admin.",
        }
    )
    return 0


def _read_request_zip(request_zip_path: str, extract_dir: str) -> dict:
    if not os.path.exists(request_zip_path):
        output_error("REQUEST_ZIP_NOT_FOUND", path=request_zip_path)
        return None
    if not os.path.isfile(request_zip_path):
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail=f"request zip must be a file path: {request_zip_path}",
        )
        return None

    request_meta = None
    try:
        with zipfile.ZipFile(request_zip_path, "r") as zf:
            names = _safe_zip_names(zf)
            if names is None:
                return None
            if "request.json" not in names or "site.yaml" not in names:
                output_error_message(
                    "INVALID_ARGS",
                    "Invalid arguments.",
                    _USAGE_HINT,
                    exit_code=4,
                    detail="request zip must contain request.json and site.yaml",
                )
                return None
            request_json = _read_zip_member_limited(zf, "request.json")
            if request_json is None:
                return None
            request_meta = json.loads(request_json.decode("utf-8"))
            if not isinstance(request_meta, dict):
                raise ValueError("request.json must be a mapping")
            cert_type = request_meta.get("cert_type")
            name = request_meta.get("name")
            if not _validate_safe_cert_name(name, field_label="Name", max_length=_cert_name_max_length(cert_type)):
                return None
            expected = {"request.json", "site.yaml", f"{name}.csr"}
            if set(names) != expected:
                output_error_message(
                    "INVALID_ARGS",
                    "Invalid arguments.",
                    _USAGE_HINT,
                    exit_code=4,
                    detail=f"request zip must contain only: {', '.join(sorted(expected))}",
                )
                return None
            for member in expected:
                target_path = os.path.join(extract_dir, member)
                content = _read_zip_member_limited(zf, member)
                if content is None:
                    return None
                _write_file_nofollow(target_path, content)
    except zipfile.BadZipFile as e:
        output_error_message(
            "INVALID_ARGS", "Invalid arguments.", _USAGE_HINT, exit_code=4, detail=f"invalid request zip: {e}"
        )
        return None
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as e:
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail=f"invalid request metadata: {e}",
        )
        return None
    except Exception as e:
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail=f"failed to read request zip: {e}",
        )
        return None
    return request_meta


def _validate_request_metadata(
    request_meta: dict, site_meta: dict, site_yaml_path: str, csr_path: str
) -> x509.CertificateSigningRequest:
    required = (
        "artifact_type",
        "schema_version",
        "request_id",
        "project",
        "name",
        "org",
        "kind",
        "cert_type",
        "csr_sha256",
        "public_key_sha256",
        "site_yaml_sha256",
    )
    missing = [field for field in required if not request_meta.get(field)]
    if missing:
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail=f"request metadata missing required field(s): {', '.join(missing)}",
        )
        return None
    if request_meta["artifact_type"] != _REQUEST_ARTIFACT_TYPE or request_meta["schema_version"] != _ARTIFACT_VERSION:
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail="unsupported request artifact metadata",
        )
        return None
    if not _validate_request_id(request_meta["request_id"]):
        return None
    if not _validate_safe_project_name(request_meta["project"]):
        return None
    name = request_meta["name"]
    if not _validate_org_name(request_meta["org"]):
        return None
    cert_type = request_meta["cert_type"]
    if cert_type not in _VALID_CERT_TYPES:
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail=f"invalid cert type '{cert_type}'; valid types: {', '.join(sorted(VALID_CERT_TYPES))}",
        )
        return None
    if not _validate_safe_cert_name(name, field_label="Name", max_length=_cert_name_max_length(cert_type)):
        return None
    if not _validate_request_kind_cert_type(request_meta["kind"], cert_type, request_meta.get("cert_role")):
        return None
    if not _validate_identity_name(name, cert_type):
        return None

    site_identity = _site_identity_from_metadata(site_meta)
    if not site_identity:
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail="site.yaml must identify exactly one participant",
        )
        return None
    for field in ("name", "org", "cert_type", "project"):
        if site_identity.get(field) != request_meta.get(field):
            output_error_message(
                "INVALID_ARGS",
                "Invalid arguments.",
                _USAGE_HINT,
                exit_code=4,
                detail=f"site.yaml field '{field}' does not match request metadata",
            )
            return None
    if site_identity.get("kind") != request_meta.get("kind"):
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail="site.yaml field 'kind' does not match request metadata",
        )
        return None
    if (site_identity.get("cert_role") or None) != (request_meta.get("cert_role") or None):
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail="site.yaml field 'cert_role' does not match request metadata",
        )
        return None
    if _is_project_shaped_site_meta(site_meta):
        participant = site_meta["participants"][0]
        if "connection_security" in participant:
            output_error_message(
                "INVALID_ARGS",
                "Invalid arguments.",
                _USAGE_HINT,
                exit_code=4,
                detail="approval site.yaml must not contain participant connection_security overrides",
            )
            return None
        if PropKey.LISTENING_HOST in participant:
            output_error_message(
                "INVALID_ARGS",
                "Invalid arguments.",
                _USAGE_HINT,
                exit_code=4,
                detail=(
                    "approval site.yaml must not contain listening_host; "
                    "distributed provisioning does not support listener certificates yet"
                ),
            )
            return None
    if request_meta["site_yaml_sha256"] != _sha256_file(site_yaml_path):
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail="site.yaml hash does not match request metadata",
        )
        return None

    csr = _load_and_validate_csr(csr_path)
    if csr is None:
        return None
    if _get_cn(csr.subject) != _csr_subject_name(name, cert_type):
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail="CSR common name does not match request metadata",
        )
        return None
    csr_role = _get_csr_role(csr)
    if csr_role != cert_type:
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail="CSR role does not match request metadata",
        )
        return None
    org_attrs = csr.subject.get_attributes_for_oid(NameOID.ORGANIZATION_NAME)
    csr_org = org_attrs[0].value if org_attrs else None
    if csr_org != request_meta.get("org"):
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail="CSR organization does not match request metadata",
        )
        return None
    if request_meta["csr_sha256"] != _sha256_file(csr_path):
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail="CSR hash does not match request metadata",
        )
        return None
    if request_meta["public_key_sha256"] != _csr_public_key_sha256(csr):
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail="CSR public key hash does not match request metadata",
        )
        return None
    return csr


def _validate_request_project_matches_ca(ca_dir: str, project: str) -> dict:
    ca_json_path = os.path.join(ca_dir, "ca.json")
    ca_cert_path = os.path.join(ca_dir, "rootCA.pem")
    for path in (ca_json_path, ca_cert_path):
        if not os.path.exists(path):
            output_error("CA_NOT_FOUND", ca_dir=ca_dir)
            return None

    ca_meta = _read_json(ca_json_path)
    if ca_meta is None:
        return None
    ca_project = ca_meta.get("project")
    if not _validate_safe_project_name(ca_project, field_label="CA project"):
        return None
    if ca_project != project:
        output_error_message(
            "PROJECT_CA_MISMATCH",
            f"Request project {project!r} does not match CA project {ca_project!r}.",
            "Use the CA directory for the same project as the request.",
            None,
            exit_code=4,
        )
        return None

    try:
        ca_cert = x509.load_pem_x509_certificate(_read_file_nofollow(ca_cert_path), default_backend())
    except Exception as e:
        output_error("CA_LOAD_FAILED", ca_dir=ca_dir, detail=str(e))
        return None
    ca_subject = _get_cn(ca_cert.subject)
    if ca_subject != project:
        output_error_message(
            "PROJECT_CA_MISMATCH",
            f"Request project {project!r} does not match root CA subject {ca_subject!r}.",
            "Use the CA directory for the same project as the request.",
            None,
            exit_code=4,
        )
        return None
    return ca_meta


def _load_project_profile(profile_path: str, request_project: str = None) -> dict:
    if not os.path.isfile(profile_path):
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail=f"project profile file not found: {profile_path}",
        )
        return None
    profile = _load_yaml_file(profile_path)
    if profile is None:
        return None
    missing = [field for field in ("name", "scheme", "connection_security") if not profile.get(field)]
    if missing:
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail=f"project profile missing required field(s): {', '.join(missing)}",
        )
        return None
    profile_project = profile.get("name")
    if not _validate_safe_project_name(profile_project, field_label="Project profile name"):
        return None
    if request_project and profile_project != request_project:
        output_error_message(
            "PROJECT_PROFILE_MISMATCH",
            f"Request project {request_project!r} does not match project profile {profile_project!r}.",
            "Use the project_profile.yaml for the same federation as the request.",
            None,
            exit_code=4,
        )
        return None
    scheme = profile.get("scheme")
    if not isinstance(scheme, str) or scheme.strip().lower() not in _VALID_SCHEMES:
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail="project profile scheme must be one of: grpc, tcp, http",
        )
        return None
    conn_sec = profile.get("connection_security")
    if not isinstance(conn_sec, str) or conn_sec.strip().lower() not in _VALID_CONNECTION_SECURITY:
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail="project profile connection_security must be one of: clear, tls, mtls",
        )
        return None
    return {
        "name": profile_project,
        "scheme": scheme.strip().lower(),
        "default_connection_security": conn_sec.strip().lower(),
    }


def handle_cert_approve(args):
    import nvflare.tool.cert.cert_cli as _cert_cli

    _cert_cli._ensure_parsers_initialized()
    handle_schema_flag(
        _cert_cli._cert_approve_parser,
        "nvflare cert approve",
        [
            "nvflare cert approve site-3.request.zip --ca-dir ./ca --profile project_profile.yaml",
            "nvflare cert approve site-3.request.zip --ca-dir ./ca --profile project_profile.yaml"
            " --out site-3.signed.zip",
        ],
        sys.argv[1:],
    )

    missing_flags = [
        flag for flag, attr in (("REQUEST_ZIP", "request_zip"), ("--ca-dir", "ca_dir")) if not getattr(args, attr, None)
    ]
    if missing_flags:
        output_usage_error(
            _cert_cli._cert_approve_parser,
            f"missing required argument(s): {', '.join(missing_flags)}",
            exit_code=4,
        )
        return 1

    if not getattr(args, "profile", None):
        output_usage_error(
            _cert_cli._cert_approve_parser,
            "--profile is required for distributed provisioning approvals",
            exit_code=4,
        )
        return 1

    request_zip_path = os.path.abspath(args.request_zip)
    with tempfile.TemporaryDirectory() as tmp_dir:
        request_dir = os.path.join(tmp_dir, "request")
        signed_dir = os.path.join(tmp_dir, "signed")
        os.makedirs(request_dir, mode=0o700)
        os.makedirs(signed_dir, mode=0o700)

        request_meta = _read_request_zip(request_zip_path, request_dir)
        if not request_meta:
            return 1
        name = request_meta["name"]
        site_yaml_path = os.path.join(request_dir, "site.yaml")
        csr_path = os.path.join(request_dir, f"{name}.csr")
        site_meta = _load_yaml_file(site_yaml_path)
        if site_meta is None:
            return 1
        csr = _validate_request_metadata(request_meta, site_meta, site_yaml_path, csr_path)
        if csr is None:
            return 1
        ca_meta = _validate_request_project_matches_ca(args.ca_dir, request_meta["project"])
        if ca_meta is None:
            return 1
        project_profile = _load_project_profile(args.profile, request_meta["project"])
        if project_profile is None:
            return 1

        # The values used below survive the tempdir cleanup: output paths are
        # written into the final signed zip location, and metadata is copied.
        server_san_fields = _server_cert_san_fields(site_meta, request_meta)
        # Returns (None, None) for non-server certs, (host, hosts) for server certs,
        # or bare None on validation error (output_error_message already emitted).
        if server_san_fields is None:
            return 1
        server_default_host, server_additional_hosts = server_san_fields
        sign_result = sign_csr_files(
            csr_path=csr_path,
            ca_dir=args.ca_dir,
            output_dir=signed_dir,
            cert_type=request_meta["cert_type"],
            accept_csr_role=False,
            valid_days=getattr(args, "valid_days", 1095),
            force=True,
            csr=csr,
            server_default_host=server_default_host,
            server_additional_hosts=server_additional_hosts,
        )
        if sign_result is None:
            return 1

        approved_at = _utc_ts()
        signed_meta = {
            "artifact_type": _SIGNED_ARTIFACT_TYPE,
            "schema_version": _ARTIFACT_VERSION,
            "request_id": request_meta["request_id"],
            "approved_at": approved_at,
            "project": request_meta["project"],
            "name": request_meta["name"],
            "org": request_meta["org"],
            "kind": request_meta["kind"],
            "cert_type": request_meta["cert_type"],
            "certificate": {
                "serial": sign_result["serial"],
                "valid_until": sign_result["valid_until"],
            },
            "cert_file": f"{name}.crt",
            "rootca_file": "rootCA.pem",
            "hashes": {
                "csr_sha256": _sha256_file(csr_path),
                "site_yaml_sha256": _sha256_file(site_yaml_path),
                "certificate_sha256": sign_result["certificate_sha256"],
                "rootca_sha256": sign_result["rootca_sha256"],
                "public_key_sha256": _csr_public_key_sha256(csr),
            },
        }
        signed_meta["scheme"] = project_profile["scheme"]
        signed_meta["default_connection_security"] = project_profile["default_connection_security"]
        if request_meta.get("cert_role"):
            signed_meta["cert_role"] = request_meta["cert_role"]
        signed_json_path = os.path.join(signed_dir, "signed.json")
        _write_json_file(signed_json_path, signed_meta)
        signed_site_yaml_path = os.path.join(signed_dir, "site.yaml")
        _write_file_nofollow(signed_site_yaml_path, _read_file_nofollow(site_yaml_path))

        signed_zip_path = getattr(args, "signed_zip", None)
        if signed_zip_path:
            signed_zip_path = os.path.abspath(signed_zip_path)
        else:
            signed_zip_path = os.path.join(os.path.dirname(request_zip_path), f"{name}.signed.zip")
        if not _write_zip_nofollow(
            signed_zip_path,
            {
                "signed.json": signed_json_path,
                "site.yaml": signed_site_yaml_path,
                f"{name}.crt": sign_result["signed_cert"],
                "rootCA.pem": sign_result["rootca"],
            },
            force=getattr(args, "force", False),
        ):
            return 1

        audit_record = {
            "schema_version": _ARTIFACT_VERSION,
            "approval": signed_meta,
            "request": request_meta,
            "request_zip_path": request_zip_path,
            "signed_zip_path": signed_zip_path,
            "ca": {
                "ca_dir": os.path.abspath(args.ca_dir),
                "metadata": ca_meta,
                "project_profile": project_profile,
                "rootca_path": os.path.abspath(os.path.join(args.ca_dir, "rootCA.pem")),
            },
            "hashes": {
                "request_zip_sha256": _sha256_file(request_zip_path),
                "request_json_sha256": _sha256_file(os.path.join(request_dir, "request.json")),
                "site_yaml_sha256": _sha256_file(site_yaml_path),
                "csr_sha256": _sha256_file(csr_path),
                "certificate_sha256": sign_result["certificate_sha256"],
                "rootca_sha256": sign_result["rootca_sha256"],
                "signed_zip_sha256": _sha256_file(signed_zip_path),
                "public_key_sha256": sign_result["public_key_sha256"],
            },
        }
        audit_path = _try_write_approve_audit(request_meta["request_id"], audit_record)

    output_ok(
        {
            "name": name,
            "project": request_meta["project"],
            "org": request_meta["org"],
            "kind": request_meta["kind"],
            "cert_role": request_meta.get("cert_role"),
            "cert_type": request_meta["cert_type"],
            "csr_sha256": request_meta["csr_sha256"],
            "signed_zip": signed_zip_path,
            "request_id": request_meta["request_id"],
            "rootca_fingerprint_sha256": sign_result["rootca_fingerprint_sha256"],
            "audit": audit_path or "(not written)",
            "next_step": f"Return {os.path.basename(signed_zip_path)} to the requester.",
        }
    )
    return 0
