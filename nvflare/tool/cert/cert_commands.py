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

"""nvflare cert subcommand handlers: init, csr, sign."""

import datetime
import ipaddress
import json
import os
import shutil
import sys

from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.x509.oid import NameOID

from nvflare.lighter.impl.cert import CertBuilder
from nvflare.lighter.utils import (
    generate_keys,
    load_crt,
    load_private_key_file,
    serialize_cert,
    serialize_pri_key,
    x509_name,
)
from nvflare.tool.cert.cert_cli import _VALID_CERT_TYPES
from nvflare.tool.cli_output import output_error, output_error_message, output_ok, output_usage_error
from nvflare.tool.cli_schema import handle_schema_flag

_USAGE_HINT = "Run the command with -h for usage."

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
    try:
        with open(rootca_path, "wb") as f:
            f.write(pem_cert)

        _write_private_key(ca_key_path, pem_key)

        created_at = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        ca_meta = {
            "project": args.project,
            "created_at": created_at,
        }
        with open(ca_json_path, "w") as f:
            json.dump(ca_meta, f, indent=2)
    except OSError as e:
        output_error("OUTPUT_DIR_NOT_WRITABLE", path=output_dir, detail=str(e))

    # 11. Compute valid_until for output
    valid_until_dt = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=3650)
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
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(pem_bytes)


def _write_file(path: str, pem_bytes: bytes) -> None:
    with open(path, "wb") as f:
        f.write(pem_bytes)


def _backup_existing_csr(out_dir: str, name: str) -> None:
    """Move existing <name>.key and <name>.csr to .bak/<timestamp>/ before overwrite."""
    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    bak_dir = os.path.join(out_dir, ".bak", timestamp)
    os.makedirs(bak_dir, mode=0o700, exist_ok=True)
    for ext in ("key", "csr"):
        src = os.path.join(out_dir, f"{name}.{ext}")
        if os.path.exists(src):
            shutil.move(src, os.path.join(bak_dir, f"{name}.{ext}"))


def _load_single_site_yaml(path: str) -> dict:
    from nvflare.lighter.utils import load_yaml

    if not os.path.isfile(path):
        output_error("PROJECT_FILE_NOT_FOUND", path=path)
    try:
        data = load_yaml(path)
    except Exception as e:
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail=f"failed to parse site yaml {path}: {e}",
        )
    if not isinstance(data, dict):
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail=f"site yaml must be a mapping: {path}",
        )
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
    if cert_type not in _VALID_CERT_TYPES:
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail=f"invalid cert type '{cert_type}'; valid types: {', '.join(sorted(_VALID_CERT_TYPES))}",
        )
    return {"name": name, "org": org, "cert_type": cert_type}


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

    # 3. Normalize and validate name
    name = (site["name"] if site else args.name).strip()
    if not name:
        output_error("INVALID_NAME", exit_code=4, name=name, reason="Name must not be empty or whitespace only.")
    if len(name) > 64:
        output_error("INVALID_NAME", exit_code=4, name=name, reason="Name must be 64 characters or fewer.")
    if os.sep in name or (os.altsep and os.altsep in name) or name.startswith("."):
        output_error(
            "INVALID_NAME", exit_code=4, name=name, reason="Name must not contain path separators or start with '.'."
        )

    # 4. Resolve force
    force = args.force

    # 5. Resolve output dir; create if needed
    out_dir = os.path.abspath(args.output_dir)
    try:
        os.makedirs(out_dir, mode=0o700, exist_ok=True)
    except OSError as e:
        output_error("OUTPUT_DIR_NOT_WRITABLE", path=out_dir, detail=str(e))

    # 6. Check write permission
    if not os.access(out_dir, os.W_OK):
        output_error("OUTPUT_DIR_NOT_WRITABLE", path=out_dir, detail="directory is not writable")

    # 7. Resolve output file paths
    key_path = os.path.join(out_dir, f"{name}.key")
    csr_path = os.path.join(out_dir, f"{name}.csr")

    # 8. Check for existing key file
    if os.path.exists(key_path) and not force:
        output_error("KEY_ALREADY_EXISTS", path=key_path)

    # 9. Back up existing files if --force and they exist
    if force and (os.path.exists(key_path) or os.path.exists(csr_path)):
        _backup_existing_csr(out_dir, name)

    # 10. Generate key and CSR
    try:
        org = site["org"] if site else getattr(args, "org", None)
        cert_type = site["cert_type"] if site else getattr(args, "cert_type", None)
        pem_key, pem_csr = _generate_csr(name, org, cert_type)
    except Exception as e:
        output_error("CSR_GENERATION_FAILED", detail=str(e))

    # 11. Write files
    try:
        _write_private_key(key_path, pem_key)
        _write_file(csr_path, pem_csr)
    except OSError as e:
        output_error("OUTPUT_DIR_NOT_WRITABLE", path=out_dir, detail=str(e))

    # 12. Emit output
    result = {
        "name": name,
        "key": key_path,
        "csr": csr_path,
        "next_step": f"Send {name}.csr to your Project Admin for signing.",
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


def _build_signed_cert(
    csr: x509.CertificateSigningRequest,
    ca_cert: x509.Certificate,
    ca_key,
    cert_type: str,
    valid_days: int,
) -> x509.Certificate:
    """Build and sign a certificate from a CSR using the CA key.

    The subject is rebuilt from safe CSR fields only; UNSTRUCTURED_NAME (role) is always
    set from cert_type (the Project Admin's authoritative -t argument), never from the CSR.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    subject_cn = _get_cn(csr.subject)

    _ADMIN_ROLES = {"org_admin", "lead", "member"}
    if cert_type in _ADMIN_ROLES:
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

    # Rebuild subject from safe OIDs only — do NOT copy CSR subject verbatim.
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

    builder = (
        x509.CertificateBuilder()
        .subject_name(safe_subject)
        .issuer_name(ca_cert.subject)
        .public_key(csr.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=valid_days))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.KeyUsage(**key_usage_kwargs), critical=False)
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(csr.public_key()),
            critical=False,
        )
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()),
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
    if getattr(args, "cert_type", None) and getattr(args, "accept_csr_role", False):
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _USAGE_HINT,
            exit_code=4,
            detail="use either -t/--type or --accept-csr-role, not both",
        )

    # 3. Validate CSR file exists
    csr_path = args.csr_path
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

    # 4. Validate CA dir
    ca_dir = args.ca_dir
    ca_key_path = os.path.join(ca_dir, "rootCA.key")
    ca_cert_path = os.path.join(ca_dir, "rootCA.pem")
    ca_json_path = os.path.join(ca_dir, "ca.json")
    for path in (ca_key_path, ca_cert_path, ca_json_path):
        if not os.path.exists(path):
            output_error("CA_NOT_FOUND", ca_dir=ca_dir)

    # 5. Load and validate CSR
    with open(csr_path, "rb") as f:
        csr_data = f.read()
    try:
        csr = x509.load_pem_x509_csr(csr_data, default_backend())
    except Exception as e:
        output_error("INVALID_CSR", path=csr_path, detail=str(e))

    if not csr.is_signature_valid:
        output_error("INVALID_CSR", path=csr_path)

    # 6. Resolve cert type explicitly: signer either overrides with -t or accepts the CSR role.
    cert_type = getattr(args, "cert_type", None)
    if getattr(args, "accept_csr_role", False):
        # Read proposed role from CSR subject UNSTRUCTURED_NAME (set by 'cert csr -t')
        _csr_role_attrs = csr.subject.get_attributes_for_oid(NameOID.UNSTRUCTURED_NAME)
        if _csr_role_attrs:
            cert_type = _csr_role_attrs[0].value
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
            detail=f"invalid cert type '{cert_type}'; valid types: {', '.join(sorted(_VALID_CERT_TYPES))}",
        )
    subject_cn = _get_cn(csr.subject)
    if not subject_cn or not subject_cn.strip():
        output_error(
            "INVALID_NAME",
            exit_code=4,
            name=subject_cn,
            reason="CSR subject CN must not be empty or whitespace only.",
        )
    if os.sep in subject_cn or (os.altsep and os.altsep in subject_cn) or subject_cn.startswith("."):
        output_error(
            "INVALID_NAME",
            exit_code=4,
            name=subject_cn,
            reason="CSR subject CN must not contain path separators or start with '.'.",
        )
    output_filename = f"{subject_cn}.crt"

    # 7. Resolve output paths; check for existing cert
    output_dir = os.path.abspath(args.output_dir)
    try:
        os.makedirs(output_dir, mode=0o700, exist_ok=True)
    except OSError as e:
        output_error("OUTPUT_DIR_NOT_WRITABLE", path=output_dir, detail=str(e))

    if not os.access(output_dir, os.W_OK):
        output_error("OUTPUT_DIR_NOT_WRITABLE", path=output_dir, detail="directory is not writable")

    cert_out_path = os.path.join(output_dir, output_filename)
    rootca_out_path = os.path.join(output_dir, "rootCA.pem")

    if os.path.exists(cert_out_path) and not args.force:
        output_error("CERT_ALREADY_EXISTS", path=cert_out_path)

    # 8. Load CA material
    try:
        ca_cert = load_crt(ca_cert_path)
        ca_key = load_private_key_file(ca_key_path)
    except Exception as e:
        output_error("CA_NOT_FOUND", ca_dir=ca_dir)

    # 9. Build and sign the certificate
    valid_days = getattr(args, "valid_days", 1095) or 1095
    try:
        signed_cert = _build_signed_cert(
            csr=csr,
            ca_cert=ca_cert,
            ca_key=ca_key,
            cert_type=cert_type,
            valid_days=valid_days,
        )
    except Exception as e:
        output_error("CERT_SIGNING_FAILED", reason=str(e))

    # 10. Write signed cert and copy rootCA.pem
    with open(cert_out_path, "wb") as f:
        f.write(serialize_cert(signed_cert))
    shutil.copy2(ca_cert_path, rootca_out_path)

    # 11. Compute valid_until for output
    try:
        _valid_until_dt = signed_cert.not_valid_after_utc
    except AttributeError:
        _valid_until_dt = signed_cert.not_valid_after  # cryptography < 42.0
    valid_until = _valid_until_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # 12. Output result
    next_step = (
        f"Send {output_filename} and rootCA.pem to the site admin.\n"
        f"They place those files in the same directory as their {subject_cn}.key, then run:\n"
        f"  nvflare package -e grpc://<server>:<port> --dir <that-dir>"
    )
    result = {
        "signed_cert": cert_out_path,
        "rootca": rootca_out_path,
        "subject_cn": subject_cn,
        "cert_type": cert_type,
        "serial": signed_cert.serial_number,
        "valid_until": valid_until,
        "next_step": next_step,
    }
    output_ok(result)
    return 0
