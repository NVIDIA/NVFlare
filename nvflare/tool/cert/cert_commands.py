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

"""nvflare cert subcommand handlers: init, csr, sign, request, approve."""

import datetime
import hashlib
import ipaddress
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

from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.x509.oid import NameOID

from nvflare.apis.utils.format_check import name_check
from nvflare.lighter.impl.cert import CertBuilder
from nvflare.lighter.utils import (
    generate_keys,
    load_crt,
    load_private_key_file,
    serialize_cert,
    serialize_pri_key,
    x509_name,
)
from nvflare.tool import cli_output
from nvflare.tool.cert.cert_constants import ADMIN_CERT_TYPES, VALID_CERT_TYPES

_VALID_CERT_TYPES = set(VALID_CERT_TYPES)
from nvflare.tool.cli_output import (
    output_error,
    output_error_message,
    output_ok,
    output_usage_error,
    print_human,
    prompt_yn,
)
from nvflare.tool.cli_schema import handle_schema_flag

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


def _validate_safe_cert_name(name: str, *, field_label: str) -> None:
    if not isinstance(name, str) or not name.strip():
        output_error(
            "INVALID_NAME", exit_code=4, name=name, reason=f"{field_label} must not be empty or whitespace only."
        )
        return
    if len(name) > 64:
        output_error("INVALID_NAME", exit_code=4, name=name, reason=f"{field_label} must be 64 characters or fewer.")
        return
    if os.sep in name or (os.altsep and os.altsep in name) or name.startswith("."):
        output_error(
            "INVALID_NAME",
            exit_code=4,
            name=name,
            reason=f"{field_label} must not contain path separators or start with '.'.",
        )
        return
    if not _SAFE_CERT_NAME_PATTERN.fullmatch(name):
        output_error(
            "INVALID_NAME",
            exit_code=4,
            name=name,
            reason=f"{field_label} must match [A-Za-z0-9][A-Za-z0-9._@-]*.",
        )
        return


def _validate_safe_project_name(project: str, *, field_label: str = "Project") -> None:
    if not isinstance(project, str) or not project.strip():
        output_error(
            "INVALID_PROJECT_NAME",
            exit_code=4,
            name=project,
            reason=f"{field_label} must not be empty or whitespace only.",
        )
        return
    if len(project) > 64:
        output_error(
            "INVALID_PROJECT_NAME",
            exit_code=4,
            name=project,
            reason=f"{field_label} must be 64 characters or fewer.",
        )
        return
    if os.sep in project or (os.altsep and os.altsep in project) or project.startswith("."):
        output_error(
            "INVALID_PROJECT_NAME",
            exit_code=4,
            name=project,
            reason=f"{field_label} must not contain path separators or start with '.'.",
        )
        return
    if not _SAFE_PROJECT_NAME_PATTERN.fullmatch(project):
        output_error(
            "INVALID_PROJECT_NAME",
            exit_code=4,
            name=project,
            reason=f"{field_label} must match [A-Za-z0-9][A-Za-z0-9._-]*.",
        )
        return


def _validate_org_name(org: str) -> None:
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


def _validate_request_kind(kind: str) -> None:
    if kind not in _REQUEST_KINDS:
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail="cert request kind must be one of: site, server, user",
        )


def _validate_request_kind_cert_type(kind: str, cert_type: str, cert_role: str = None) -> None:
    _validate_request_kind(kind)
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
        return

    if cert_type not in _USER_CERT_TYPES:
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail="request kind 'user' requires cert type one of: org_admin, lead, member",
        )
    if cert_role and _USER_ROLE_TO_CERT_TYPE.get(cert_role) != cert_type:
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail="user cert_role does not match request cert_type",
        )


def _validate_identity_name(name: str, cert_type: str) -> None:
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
        return

    if not isinstance(name, str):
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail=f"name must be a string for entity_type={entity_type}",
        )
        return

    invalid, reason = name_check(name, entity_type)
    if invalid:
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail=reason,
        )


def _validate_request_id(request_id: str) -> None:
    if not isinstance(request_id, str) or not _REQUEST_ID_PATTERN.fullmatch(request_id):
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail="request_id must be a UUID hex string.",
        )


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
            "nvflare cert init --project MyProject -o ./ca",
            "nvflare cert init --project MyProject -o ./ca --org NVIDIA --force",
        ],
        sys.argv[1:],
    )

    # 2. Validate required args
    missing_flags = [
        flag
        for flag, attr in (("--project", "project"), ("-o/--output-dir", "output_dir"))
        if not getattr(args, attr, None)
    ]
    if missing_flags:
        output_usage_error(
            _cert_cli._cert_init_parser, f"missing required argument(s): {', '.join(missing_flags)}", exit_code=4
        )
    _validate_safe_project_name(args.project)

    # 3. Resolve force
    force = args.force

    # 4. Resolve and create output dir
    output_dir = os.path.abspath(args.output_dir)
    try:
        os.makedirs(output_dir, mode=0o700, exist_ok=True)
    except OSError as e:
        output_error("OUTPUT_DIR_NOT_WRITABLE", path=output_dir, detail=str(e))

    # 5. Check write permission
    if not os.access(output_dir, os.W_OK):
        output_error("OUTPUT_DIR_NOT_WRITABLE", path=output_dir, detail="directory is not writable")

    # 6. Check for existing rootCA.key
    ca_key_path = os.path.join(output_dir, "rootCA.key")
    if os.path.exists(ca_key_path):
        if not force:
            output_error("CA_ALREADY_EXISTS", path=output_dir)
        # --force: back up existing files
        _backup_existing_ca(output_dir)

    # 7. Generate key pair
    try:
        pri_key, pub_key = generate_keys()
    except Exception as e:
        output_error("CERT_GENERATION_FAILED", detail=str(e))

    # 8. Generate self-signed CA certificate
    try:
        cert = CertBuilder._generate_cert(
            subject=args.project,
            subject_org=args.org,
            issuer=args.project,  # self-signed: issuer == subject
            signing_pri_key=pri_key,
            subject_pub_key=pub_key,
            valid_days=getattr(args, "valid_days", 3650) or 3650,
            ca=True,
        )
    except Exception as e:
        output_error("CERT_GENERATION_FAILED", detail=str(e))

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
            "project": args.project,
            "created_at": created_at,
        }
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

    # 11. Compute valid_until for output
    valid_days_actual = getattr(args, "valid_days", 3650) or 3650
    valid_until_dt = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=valid_days_actual)
    valid_until_str = valid_until_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # 12. Output result
    result = {
        "ca_cert": rootca_path,
        "project": args.project,
        "subject_cn": args.project,
        "valid_until": valid_until_str,
    }
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
    with os.fdopen(fd, "wb") as f:
        try:
            f.write(pem_bytes)
        except Exception:
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
    if not isinstance(data, dict):
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail=f"yaml must be a mapping: {path}",
        )
    return data


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
            or name.endswith("/")
            or name.startswith("/")
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
            continue
        if name.lower().endswith(".key"):
            output_error_message(
                "INVALID_ARGS",
                "Invalid arguments.",
                _USAGE_HINT,
                exit_code=4,
                detail=f"request zip must not contain private keys: {name}",
            )
            continue
        seen.add(name)
        names.append(name)
    return names


def _read_zip_member_limited(zf: zipfile.ZipFile, member: str) -> bytes:
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
        raise ValueError(f"zip member exceeds size limit: {member}")
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


def _write_zip_nofollow(zip_path: str, members: dict, force: bool = False) -> None:
    if os.path.exists(zip_path) and not force:
        output_error("CERT_ALREADY_EXISTS", path=zip_path)
    parent = os.path.dirname(os.path.abspath(zip_path)) or "."
    try:
        os.makedirs(parent, mode=0o700, exist_ok=True)
    except OSError as e:
        output_error("OUTPUT_DIR_NOT_WRITABLE", path=parent, detail=str(e))

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
            continue
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
        except OSError as e:
            output_error("OUTPUT_DIR_NOT_WRITABLE", path=src_path, detail=str(e))

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
    if not isinstance(data, dict):
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail=f"json must be a mapping: {path}",
        )
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
    _validate_safe_cert_name(name, field_label="Name")
    if cert_type not in _VALID_CERT_TYPES:
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail=f"invalid cert type '{cert_type}'; valid types: {', '.join(sorted(VALID_CERT_TYPES))}",
        )

    out_dir = os.path.abspath(output_dir)
    try:
        os.makedirs(out_dir, mode=0o700, exist_ok=True)
    except OSError as e:
        output_error("OUTPUT_DIR_NOT_WRITABLE", path=out_dir, detail=str(e))

    if not os.access(out_dir, os.W_OK):
        output_error("OUTPUT_DIR_NOT_WRITABLE", path=out_dir, detail="directory is not writable")

    key_path = os.path.join(out_dir, f"{name}.key")
    csr_path = os.path.join(out_dir, f"{name}.csr")

    if os.path.exists(key_path) and not force:
        output_error("KEY_ALREADY_EXISTS", path=key_path)

    if force and (os.path.exists(key_path) or os.path.exists(csr_path)):
        _backup_existing_csr(out_dir, name)

    try:
        pem_key, pem_csr = _generate_csr(name, org, cert_type)
    except Exception as e:
        output_error("CSR_GENERATION_FAILED", detail=str(e))

    try:
        _write_private_key(key_path, pem_key)
        _write_file_nofollow(csr_path, pem_csr)
    except OSError as e:
        output_error("OUTPUT_DIR_NOT_WRITABLE", path=out_dir, detail=str(e))

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
    # 1. --schema
    import nvflare.tool.cert.cert_cli as _cert_cli

    _cert_cli._ensure_parsers_initialized()
    handle_schema_flag(
        _cert_cli._cert_csr_parser,
        "nvflare cert csr",
        [
            "nvflare cert csr -n hospital-1 -t client -o ./csr",
            "nvflare cert csr -n fl-server -t server -o ./server-csr --org ACME --force",
            "nvflare cert csr --project-file site.yml -o ./csr",
        ],
        sys.argv[1:],
    )

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
        site = _load_single_site_yaml(args.project_file)

    # 3. Validate required args (-o is required in all modes; -n/-t only without --project-file)
    missing_flags = []
    if not getattr(args, "output_dir", None):
        missing_flags.append("-o/--output-dir")
    if site is None and not getattr(args, "name", None):
        missing_flags.append("-n/--name")
    if site is None and not getattr(args, "cert_type", None):
        missing_flags.append("-t/--type")
    if missing_flags:
        output_usage_error(
            _cert_cli._cert_csr_parser, f"missing required argument(s): {', '.join(missing_flags)}", exit_code=4
        )

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

    if not basic_constraints.ca:
        output_error("CERT_SIGNING_FAILED", reason="CA certificate is not a CA certificate")

    ca_not_valid_after = _get_cert_not_valid_after(ca_cert)
    if ca_not_valid_after <= now:
        output_error(
            "CERT_SIGNING_FAILED",
            reason=f"CA certificate expired at {ca_not_valid_after.strftime('%Y-%m-%dT%H:%M:%SZ')}",
        )

    return ca_not_valid_after


def _build_signed_cert(
    csr: x509.CertificateSigningRequest,
    ca_cert: x509.Certificate,
    ca_key,
    cert_type: str,
    now: datetime.datetime,
    not_valid_after: datetime.datetime,
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
    safe_attrs = []
    seen_oids = set()
    for attr in csr.subject:
        if attr.oid not in _SAFE_OIDS:
            continue
        if attr.oid in seen_oids:
            raise ValueError(f"CSR contains duplicate subject attribute for OID '{attr.oid._name}'")
        seen_oids.add(attr.oid)
        safe_attrs.append(attr)
    safe_attrs.append(x509.NameAttribute(NameOID.UNSTRUCTURED_NAME, cert_type))
    safe_subject = x509.Name(safe_attrs)
    try:
        issuer_ski = ca_cert.extensions.get_extension_for_class(x509.SubjectKeyIdentifier).value.digest
        authority_key_identifier = x509.AuthorityKeyIdentifier(
            key_identifier=issuer_ski,
            authority_cert_issuer=None,
            authority_cert_serial_number=None,
        )
    except x509.ExtensionNotFound:
        authority_key_identifier = x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key())

    builder = (
        x509.CertificateBuilder()
        .subject_name(safe_subject)
        .issuer_name(ca_cert.subject)
        .public_key(csr.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(not_valid_after)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.KeyUsage(**key_usage_kwargs), critical=True)
        .add_extension(x509.ExtendedKeyUsage(eku_oids), critical=False)
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(csr.public_key()),
            critical=False,
        )
        .add_extension(
            authority_key_identifier,
            critical=False,
        )
    )
    if cert_type == "server" and subject_cn:
        try:
            ip = ipaddress.ip_address(subject_cn)
            san_entry = x509.IPAddress(ip)
        except ValueError:
            san_entry = x509.DNSName(subject_cn)
        builder = builder.add_extension(
            x509.SubjectAlternativeName([san_entry]),
            critical=False,
        )
    return builder.sign(ca_key, hashes.SHA256(), default_backend())


def _load_and_validate_csr(csr_path: str) -> x509.CertificateSigningRequest:
    if not os.path.exists(csr_path):
        output_error("CSR_NOT_FOUND", path=csr_path)
    if not os.path.isfile(csr_path):
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail=f"-r/--csr must be a file path, not a directory: {csr_path}",
        )

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
    try:
        csr = x509.load_pem_x509_csr(csr_data, default_backend())
    except Exception as e:
        output_error("INVALID_CSR", path=csr_path, detail=str(e))

    if not csr.is_signature_valid:
        output_error("INVALID_CSR", path=csr_path)
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

    if accept_csr_role:
        cert_type = _get_csr_role(csr)
        if not cert_type:
            output_error_message(
                "INVALID_ARGS",
                "Invalid arguments.",
                _USAGE_HINT,
                exit_code=4,
                detail="CSR does not contain a proposed role; re-run with -t/--type or generate the CSR with 'cert csr -t'",
            )
    elif not cert_type:
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail="specify either -t/--type to override the role or --accept-csr-role to trust the CSR role",
        )

    if cert_type not in _VALID_CERT_TYPES:
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail=f"invalid cert type '{cert_type}'; valid types: {', '.join(sorted(VALID_CERT_TYPES))}",
        )
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
) -> dict:
    """Sign a CSR using the existing cert signing logic and write cert/rootCA files."""
    ca_key_path = os.path.join(ca_dir, "rootCA.key")
    ca_cert_path = os.path.join(ca_dir, "rootCA.pem")
    ca_json_path = os.path.join(ca_dir, "ca.json")
    for path in (ca_key_path, ca_cert_path, ca_json_path):
        if not os.path.exists(path):
            output_error("CA_NOT_FOUND", ca_dir=ca_dir)

    if csr is None:
        csr = _load_and_validate_csr(csr_path)
    cert_type = _resolve_sign_cert_type(csr, cert_type, accept_csr_role)

    subject_cn = _get_cn(csr.subject)
    _validate_safe_cert_name(subject_cn, field_label="CSR subject CN")
    output_filename = f"{subject_cn}.crt"

    output_dir = os.path.abspath(output_dir)
    try:
        os.makedirs(output_dir, mode=0o700, exist_ok=True)
    except OSError as e:
        output_error("OUTPUT_DIR_NOT_WRITABLE", path=output_dir, detail=str(e))

    if not os.access(output_dir, os.W_OK):
        output_error("OUTPUT_DIR_NOT_WRITABLE", path=output_dir, detail="directory is not writable")

    cert_out_path = os.path.join(output_dir, output_filename)
    rootca_out_path = os.path.join(output_dir, "rootCA.pem")

    if os.path.exists(cert_out_path) and not force:
        output_error("CERT_ALREADY_EXISTS", path=cert_out_path)
    if os.path.exists(rootca_out_path) and not force:
        output_error("ROOTCA_ALREADY_EXISTS", path=rootca_out_path)

    try:
        rootca_bytes = _read_file_nofollow(ca_cert_path)
        ca_cert = x509.load_pem_x509_certificate(rootca_bytes, default_backend())
        ca_key = load_private_key_file(ca_key_path)
    except Exception as e:
        output_error("CA_LOAD_FAILED", ca_dir=ca_dir, detail=str(e))

    now = _utc_now()
    ca_not_valid_after = _validate_signing_ca(ca_cert, now)

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
        )
    except Exception as e:
        output_error("CERT_SIGNING_FAILED", reason=str(e))

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
        "public_key_sha256": _cert_public_key_sha256(signed_cert),
    }


def handle_cert_sign(args):
    # 1. --schema
    import nvflare.tool.cert.cert_cli as _cert_cli

    _cert_cli._ensure_parsers_initialized()
    handle_schema_flag(
        _cert_cli._cert_sign_parser,
        "nvflare cert sign",
        [
            "nvflare cert sign -r ./hospital-1.csr -c ./ca -o ./signed --accept-csr-role",
            "nvflare cert sign -r ./alice.csr -c ./ca -o ./alice-signed -t lead",
        ],
        sys.argv[1:],
    )

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
        output_usage_error(
            _cert_cli._cert_sign_parser, f"missing required argument(s): {', '.join(missing_flags)}", exit_code=4
        )
    csr = _load_and_validate_csr(args.csr_path)
    cert_type = _resolve_sign_cert_type(csr, getattr(args, "cert_type", None), getattr(args, "accept_csr_role", False))
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

    # 12. Output result
    next_step = (
        "This internal command writes a signed certificate and rootCA.pem.\n"
        "For the public distributed provisioning workflow, use:\n"
        "  nvflare cert request ...\n"
        "  nvflare cert approve <request.zip> --ca-dir <ca-dir>\n"
        "  nvflare package <signed.zip> -e grpc://<server>:<port>"
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


def _resolve_request_identity(args) -> dict:
    kind = getattr(args, "kind", None)
    identity_args = list(getattr(args, "identity_args", None) or getattr(args, "values", []) or [])
    if kind in _REQUEST_KIND_TO_CERT_TYPE:
        if len(identity_args) != 1:
            output_error_message(
                "INVALID_ARGS",
                "Invalid arguments.",
                _USAGE_HINT,
                exit_code=4,
                detail=f"cert request {kind} requires exactly one NAME argument",
            )
        return {
            "kind": kind,
            "name": identity_args[0],
            "cert_role": None,
            "cert_type": _REQUEST_KIND_TO_CERT_TYPE[kind],
        }
    if kind == "user":
        if len(identity_args) != 2:
            output_error_message(
                "INVALID_ARGS",
                "Invalid arguments.",
                _USAGE_HINT,
                exit_code=4,
                detail="cert request user requires ROLE and NAME arguments",
            )
        role = identity_args[0]
        cert_type = _USER_ROLE_TO_CERT_TYPE.get(role)
        if not cert_type:
            output_error_message(
                "INVALID_ARGS",
                "Invalid arguments.",
                _USAGE_HINT,
                exit_code=4,
                detail="user role must be one of: org-admin, lead, member",
            )
        return {
            "kind": "user",
            "name": identity_args[1],
            "cert_role": role,
            "cert_type": cert_type,
        }
    output_error_message(
        "INVALID_ARGS",
        "Invalid arguments.",
        _USAGE_HINT,
        exit_code=4,
        detail="cert request kind must be one of: site, server, user",
    )


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
            "nvflare cert request site site-3 --org nvidia --project example_project",
            "nvflare cert request user org-admin alice@nvidia.com --org nvidia --project example_project",
        ],
        sys.argv[1:],
    )

    _validate_request_kind(getattr(args, "kind", None))
    missing_flags = [
        flag for flag, attr in (("--org", "org"), ("--project", "project")) if not getattr(args, attr, None)
    ]
    if missing_flags:
        output_usage_error(
            _cert_cli._cert_request_parser,
            f"missing required argument(s): {', '.join(missing_flags)}",
            exit_code=4,
        )
    _validate_safe_project_name(args.project)
    _validate_org_name(args.org)

    identity = _resolve_request_identity(args)
    name = identity["name"].strip()
    _validate_safe_cert_name(name, field_label="Name")
    _validate_identity_name(name, identity["cert_type"])

    request_dir = os.path.abspath(
        getattr(args, "out", None) or getattr(args, "output_dir", None) or os.path.join(".", name)
    )
    if os.path.exists(request_dir) and not os.path.isdir(request_dir):
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail=f"--out must be a directory path: {request_dir}",
        )

    request_zip_path = os.path.join(request_dir, f"{name}.request.zip")
    if os.path.exists(request_zip_path) and not getattr(args, "force", False):
        output_error("CERT_ALREADY_EXISTS", path=request_zip_path)

    csr_result = generate_csr_files(
        name=name,
        org=args.org,
        cert_type=identity["cert_type"],
        output_dir=request_dir,
        force=getattr(args, "force", False),
    )

    request_id = uuid.uuid4().hex
    created_at = _utc_ts()
    request_meta = {
        "artifact_type": _REQUEST_ARTIFACT_TYPE,
        "schema_version": _ARTIFACT_VERSION,
        "request_id": request_id,
        "created_at": created_at,
        "project": args.project,
        "name": name,
        "org": args.org,
        "kind": identity["kind"],
        "cert_type": identity["cert_type"],
        "cert_role": identity["cert_role"],
        "csr_sha256": csr_result["csr_sha256"],
        "public_key_sha256": csr_result["public_key_sha256"],
    }
    site_meta = _build_site_metadata(request_meta)

    request_json_path = os.path.join(request_dir, "request.json")
    site_yaml_path = os.path.join(request_dir, "site.yaml")
    try:
        _write_json_file(request_json_path, request_meta)
        _write_yaml_file(site_yaml_path, site_meta)
        _write_zip_nofollow(
            request_zip_path,
            {
                "request.json": request_json_path,
                "site.yaml": site_yaml_path,
                f"{name}.csr": csr_result["csr_path"],
            },
            force=getattr(args, "force", False),
        )
    except OSError as e:
        output_error("OUTPUT_DIR_NOT_WRITABLE", path=request_dir, detail=str(e))

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
            "site_yaml_sha256": _sha256_file(site_yaml_path),
            "csr_sha256": _sha256_file(csr_result["csr_path"]),
            "request_zip_sha256": _sha256_file(request_zip_path),
            "public_key_sha256": csr_result["public_key_sha256"],
        },
    }
    audit_path = _try_write_request_audit(request_id, audit_record)

    output_ok(
        {
            "name": name,
            "project": args.project,
            "request_zip": request_zip_path,
            "request_id": request_id,
            "audit": audit_path or "(not written)",
            "next_step": f"Send {os.path.basename(request_zip_path)} to your Project Admin.",
        }
    )
    return 0


def _read_request_zip(request_zip_path: str, extract_dir: str) -> dict:
    if not os.path.exists(request_zip_path):
        output_error("CSR_NOT_FOUND", path=request_zip_path)
    if not os.path.isfile(request_zip_path):
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail=f"request zip must be a file path: {request_zip_path}",
        )

    request_meta = None
    try:
        with zipfile.ZipFile(request_zip_path, "r") as zf:
            names = _safe_zip_names(zf)
            if "request.json" not in names or "site.yaml" not in names:
                output_error_message(
                    "INVALID_ARGS",
                    "Invalid arguments.",
                    _USAGE_HINT,
                    exit_code=4,
                    detail="request zip must contain request.json and site.yaml",
                )
            request_meta = json.loads(_read_zip_member_limited(zf, "request.json").decode("utf-8"))
            if not isinstance(request_meta, dict):
                raise ValueError("request.json must be a mapping")
            name = request_meta.get("name")
            _validate_safe_cert_name(name, field_label="Name")
            expected = {"request.json", "site.yaml", f"{name}.csr"}
            if set(names) != expected:
                output_error_message(
                    "INVALID_ARGS",
                    "Invalid arguments.",
                    _USAGE_HINT,
                    exit_code=4,
                    detail=f"request zip must contain only: {', '.join(sorted(expected))}",
                )
            for member in expected:
                target_path = os.path.join(extract_dir, member)
                _write_file_nofollow(target_path, _read_zip_member_limited(zf, member))
    except zipfile.BadZipFile as e:
        output_error_message(
            "INVALID_ARGS", "Invalid arguments.", _USAGE_HINT, exit_code=4, detail=f"invalid request zip: {e}"
        )
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as e:
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail=f"invalid request metadata: {e}",
        )
    except Exception as e:
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail=f"failed to read request zip: {e}",
        )
    return request_meta


def _validate_request_metadata(request_meta: dict, site_meta: dict, csr_path: str) -> x509.CertificateSigningRequest:
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
    _validate_request_id(request_meta["request_id"])
    _validate_safe_project_name(request_meta["project"])
    name = request_meta["name"]
    _validate_safe_cert_name(name, field_label="Name")
    _validate_org_name(request_meta["org"])
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
    _validate_request_kind_cert_type(request_meta["kind"], cert_type, request_meta.get("cert_role"))
    _validate_identity_name(name, cert_type)

    for meta_field, site_field in (("name", "name"), ("org", "org"), ("cert_type", "type"), ("project", "project")):
        if site_meta.get(site_field) != request_meta.get(meta_field):
            output_error_message(
                "INVALID_ARGS",
                "Invalid arguments.",
                _USAGE_HINT,
                exit_code=4,
                detail=f"site.yaml field '{site_field}' does not match request metadata",
            )
    if site_meta.get("kind") != request_meta.get("kind"):
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail="site.yaml field 'kind' does not match request metadata",
        )
    if (site_meta.get("cert_role") or None) != (request_meta.get("cert_role") or None):
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail="site.yaml field 'cert_role' does not match request metadata",
        )

    csr = _load_and_validate_csr(csr_path)
    if _get_cn(csr.subject) != name:
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail="CSR common name does not match request metadata",
        )
    csr_role = _get_csr_role(csr)
    if csr_role != cert_type:
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail="CSR role does not match request metadata",
        )
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
    if request_meta["csr_sha256"] != _sha256_file(csr_path):
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail="CSR hash does not match request metadata",
        )
    if request_meta["public_key_sha256"] != _csr_public_key_sha256(csr):
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail="CSR public key hash does not match request metadata",
        )
    return csr


def _validate_request_project_matches_ca(ca_dir: str, project: str) -> dict:
    ca_json_path = os.path.join(ca_dir, "ca.json")
    ca_cert_path = os.path.join(ca_dir, "rootCA.pem")
    for path in (ca_json_path, ca_cert_path):
        if not os.path.exists(path):
            output_error("CA_NOT_FOUND", ca_dir=ca_dir)

    ca_meta = _read_json(ca_json_path)
    ca_project = ca_meta.get("project")
    _validate_safe_project_name(ca_project, field_label="CA project")
    if ca_project != project:
        output_error_message(
            "PROJECT_CA_MISMATCH",
            f"Request project {project!r} does not match CA project {ca_project!r}.",
            "Use the CA directory for the same project as the request.",
            None,
            exit_code=4,
        )

    try:
        ca_cert = load_crt(ca_cert_path)
    except Exception as e:
        output_error("CA_LOAD_FAILED", ca_dir=ca_dir, detail=str(e))
    ca_subject = _get_cn(ca_cert.subject)
    if ca_subject != project:
        output_error_message(
            "PROJECT_CA_MISMATCH",
            f"Request project {project!r} does not match root CA subject {ca_subject!r}.",
            "Use the CA directory for the same project as the request.",
            None,
            exit_code=4,
        )
    return ca_meta


def handle_cert_approve(args):
    import nvflare.tool.cert.cert_cli as _cert_cli

    _cert_cli._ensure_parsers_initialized()
    handle_schema_flag(
        _cert_cli._cert_approve_parser,
        "nvflare cert approve",
        [
            "nvflare cert approve site-3.request.zip --ca-dir ./ca",
            "nvflare cert approve site-3.request.zip --ca-dir ./ca --out site-3.signed.zip",
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

    request_zip_path = os.path.abspath(args.request_zip)
    with tempfile.TemporaryDirectory() as tmp_dir:
        request_dir = os.path.join(tmp_dir, "request")
        signed_dir = os.path.join(tmp_dir, "signed")
        os.makedirs(request_dir, mode=0o700)
        os.makedirs(signed_dir, mode=0o700)

        request_meta = _read_request_zip(request_zip_path, request_dir)
        name = request_meta["name"]
        site_yaml_path = os.path.join(request_dir, "site.yaml")
        csr_path = os.path.join(request_dir, f"{name}.csr")
        site_meta = _load_yaml_file(site_yaml_path)
        csr = _validate_request_metadata(request_meta, site_meta, csr_path)
        ca_meta = _validate_request_project_matches_ca(args.ca_dir, request_meta["project"])

        sign_result = sign_csr_files(
            csr_path=csr_path,
            ca_dir=args.ca_dir,
            output_dir=signed_dir,
            cert_type=request_meta["cert_type"],
            accept_csr_role=False,
            valid_days=getattr(args, "valid_days", 1095),
            force=True,
            csr=csr,
        )

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
            "cert_role": request_meta.get("cert_role"),
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
        signed_json_path = os.path.join(signed_dir, "signed.json")
        _write_json_file(signed_json_path, signed_meta)
        signed_site_yaml_path = os.path.join(signed_dir, "site.yaml")
        _write_file_nofollow(signed_site_yaml_path, _read_file_nofollow(site_yaml_path))

        signed_zip_path = getattr(args, "out", None) or getattr(args, "signed_zip", None)
        if signed_zip_path:
            signed_zip_path = os.path.abspath(signed_zip_path)
        else:
            signed_zip_path = os.path.join(os.path.dirname(request_zip_path), f"{name}.signed.zip")
        _write_zip_nofollow(
            signed_zip_path,
            {
                "signed.json": signed_json_path,
                "site.yaml": signed_site_yaml_path,
                f"{name}.crt": sign_result["signed_cert"],
                "rootCA.pem": sign_result["rootca"],
            },
            force=getattr(args, "force", False),
        )

        audit_record = {
            "schema_version": _ARTIFACT_VERSION,
            "approval": signed_meta,
            "request": request_meta,
            "request_zip_path": request_zip_path,
            "signed_zip_path": signed_zip_path,
            "ca": {
                "ca_dir": os.path.abspath(args.ca_dir),
                "metadata": ca_meta,
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
            "signed_zip": signed_zip_path,
            "request_id": request_meta["request_id"],
            "audit": audit_path or "(not written)",
            "next_step": f"Return {os.path.basename(signed_zip_path)} to the requester.",
        }
    )
    return 0
