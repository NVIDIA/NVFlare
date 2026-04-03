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
from nvflare.tool.cli_errors import get_error
from nvflare.tool.cli_output import output, output_error

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
    if getattr(args, "schema", False):
        import nvflare.tool.cert.cert_cli as _cert_cli
        from nvflare.tool.cli_schema import parser_to_schema

        _cert_cli._ensure_parsers_initialized()

        schema = parser_to_schema(
            _cert_cli._cert_init_parser,
            command="nvflare cert init",
            examples=[
                "nvflare cert init --project MyProject -o ./ca",
                "nvflare cert init --project MyProject -o ./ca --output json",
                "nvflare cert init --project MyProject -o ./ca --org NVIDIA --force",
            ],
        )
        print(json.dumps(schema, indent=2))
        return 0

    # 2. Validate required args
    output_fmt = getattr(args, "output_fmt", None)
    for flag, attr in (("--project", "project"), ("-o/--output-dir", "output_dir")):
        if not getattr(args, attr, None):
            message, hint = get_error("INVALID_ARGS", detail=f"{flag} is required")
            output_error("INVALID_ARGS", message, hint, output_fmt, exit_code=2)

    # 3. --output json implies --force
    json_mode = output_fmt == "json"
    force = args.force or json_mode

    # 4. Resolve and create output dir
    output_dir = os.path.abspath(args.output_dir)
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        message, hint = get_error("OUTPUT_DIR_NOT_WRITABLE", path=output_dir, detail=str(e))
        output_error("OUTPUT_DIR_NOT_WRITABLE", message, hint, output_fmt)

    # 5. Check write permission
    if not os.access(output_dir, os.W_OK):
        message, hint = get_error("OUTPUT_DIR_NOT_WRITABLE", path=output_dir, detail="directory is not writable")
        output_error("OUTPUT_DIR_NOT_WRITABLE", message, hint, output_fmt)

    # 6. Check for existing rootCA.key
    ca_key_path = os.path.join(output_dir, "rootCA.key")
    if os.path.exists(ca_key_path):
        if not force:
            message, hint = get_error("CA_ALREADY_EXISTS", path=output_dir)
            output_error("CA_ALREADY_EXISTS", message, hint, output_fmt)
        # --force or json mode: back up existing files
        _backup_existing_ca(output_dir)

    # 7. Generate key pair
    try:
        pri_key, pub_key = generate_keys()
    except Exception as e:
        message, hint = get_error("CERT_GENERATION_FAILED", detail=str(e))
        output_error("CERT_GENERATION_FAILED", message, hint, output_fmt)

    # 8. Generate self-signed CA certificate
    try:
        cert = CertBuilder._generate_cert(
            subject=args.project,
            subject_org=args.org,
            issuer=args.project,  # self-signed: issuer == subject
            signing_pri_key=pri_key,
            subject_pub_key=pub_key,
            valid_days=3650,
            ca=True,
        )
    except Exception as e:
        message, hint = get_error("CERT_GENERATION_FAILED", detail=str(e))
        output_error("CERT_GENERATION_FAILED", message, hint, output_fmt)

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

        created_at = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        ca_meta = {
            "project": args.project,
            "created_at": created_at,
            "next_serial": 2,
        }
        with open(ca_json_path, "w") as f:
            json.dump(ca_meta, f, indent=2)
    except OSError as e:
        message, hint = get_error("OUTPUT_DIR_NOT_WRITABLE", path=output_dir, detail=str(e))
        output_error("OUTPUT_DIR_NOT_WRITABLE", message, hint, output_fmt)

    # 11. Compute valid_until for output
    valid_until_dt = datetime.datetime.utcnow() + datetime.timedelta(days=3650)
    valid_until_str = valid_until_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # 12. Output result
    result = {
        "ca_cert": rootca_path,
        "ca_key": ca_key_path,
        "project": args.project,
        "subject_cn": args.project,
        "valid_until": valid_until_str,
    }
    output(result, output_fmt)
    return 0


# ---------------------------------------------------------------------------
# cert csr
# ---------------------------------------------------------------------------


def _generate_csr(name: str, org: str = None, role: str = None):
    """Generate RSA private key and CSR.

    The optional ``role`` is embedded in the CSR's UNSTRUCTURED_NAME field as a
    hint to the Project Admin.  The Project Admin's ``cert sign -t`` is always
    authoritative and may override whatever is in the CSR.

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


def handle_cert_csr(args):
    # 1. --schema
    if getattr(args, "schema", False):
        import nvflare.tool.cert.cert_cli as _cert_cli
        from nvflare.tool.cli_schema import parser_to_schema

        _cert_cli._ensure_parsers_initialized()

        schema = parser_to_schema(
            _cert_cli._cert_csr_parser,
            command="nvflare cert csr",
            examples=[
                "nvflare cert csr -n hospital-1 -o ./csr",
                "nvflare cert csr -n researcher-alice -o ./alice-csr --output json",
                "nvflare cert csr -n fl-server -o ./server-csr --org ACME --force",
            ],
        )
        print(json.dumps(schema, indent=2))
        return 0

    # 2. Validate required args
    output_fmt = getattr(args, "output_fmt", None)
    for flag, attr in (("-n/--name", "name"), ("-o/--output-dir", "output_dir")):
        if not getattr(args, attr, None):
            message, hint = get_error("INVALID_ARGS", detail=f"{flag} is required")
            output_error("INVALID_ARGS", message, hint, output_fmt, exit_code=2)

    # 3. Normalize and validate name
    name = args.name.strip()
    if len(name) > 64:
        message, hint = get_error("INVALID_NAME", name=name, reason="Name must be 64 characters or fewer.")
        output_error("INVALID_NAME", message, hint, output_fmt, exit_code=4)
    if os.sep in name or (os.altsep and os.altsep in name) or name.startswith("."):
        message, hint = get_error("INVALID_NAME", name=name, reason="Name must not contain path separators or start with '.'.")
        output_error("INVALID_NAME", message, hint, output_fmt, exit_code=4)

    # 4. --output json implies --force
    force = args.force or (output_fmt == "json")

    # 5. Resolve output dir; create if needed
    out_dir = os.path.abspath(args.output_dir)
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as e:
        message, hint = get_error("OUTPUT_DIR_NOT_WRITABLE", path=out_dir, detail=str(e))
        output_error("OUTPUT_DIR_NOT_WRITABLE", message, hint, output_fmt)

    # 6. Check write permission
    if not os.access(out_dir, os.W_OK):
        message, hint = get_error("OUTPUT_DIR_NOT_WRITABLE", path=out_dir, detail="directory is not writable")
        output_error("OUTPUT_DIR_NOT_WRITABLE", message, hint, output_fmt)

    # 7. Resolve output file paths
    key_path = os.path.join(out_dir, f"{name}.key")
    csr_path = os.path.join(out_dir, f"{name}.csr")

    # 8. Check for existing key file
    if os.path.exists(key_path) and not force:
        message, hint = get_error("KEY_ALREADY_EXISTS", path=key_path)
        output_error("KEY_ALREADY_EXISTS", message, hint, output_fmt)

    # 9. Back up existing files if --force and they exist
    if force and (os.path.exists(key_path) or os.path.exists(csr_path)):
        _backup_existing_csr(out_dir, name)

    # 10. Generate key and CSR
    try:
        pem_key, pem_csr = _generate_csr(name, getattr(args, "org", None), getattr(args, "cert_type", None))
    except Exception as e:
        message, hint = get_error("CSR_GENERATION_FAILED", detail=str(e))
        output_error("CSR_GENERATION_FAILED", message, hint, output_fmt)

    # 11. Write files
    try:
        _write_private_key(key_path, pem_key)
        _write_file(csr_path, pem_csr)
    except OSError as e:
        message, hint = get_error("OUTPUT_DIR_NOT_WRITABLE", path=out_dir, detail=str(e))
        output_error("OUTPUT_DIR_NOT_WRITABLE", message, hint, output_fmt)

    # 12. Emit output
    result = {
        "name": name,
        "key": key_path,
        "csr": csr_path,
        "next_step": f"Send {name}.csr to your Project Admin for signing.",
    }
    output(result, output_fmt)
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


def _claim_serial(ca_json_path: str) -> int:
    """Atomically claim the next serial number from ca.json and return it.

    Uses an exclusive file lock so concurrent ``nvflare cert sign`` invocations
    cannot claim the same serial number. Falls back gracefully on platforms where
    fcntl is unavailable (e.g. Windows).
    """
    try:
        import fcntl

        def _lock(f):
            fcntl.flock(f, fcntl.LOCK_EX)

    except ImportError:

        def _lock(f):
            pass

    with open(ca_json_path, "r+") as f:
        _lock(f)
        meta = json.load(f)
        serial = meta.get("next_serial", 2)
        meta["next_serial"] = serial + 1
        f.seek(0)
        f.truncate()
        json.dump(meta, f, indent=2)
    return serial


def _build_signed_cert(
    csr: x509.CertificateSigningRequest,
    ca_cert: x509.Certificate,
    ca_key,
    cert_type: str,
    valid_days: int,
    serial: int,
) -> x509.Certificate:
    """Build and sign a certificate from a CSR using the CA key.

    The subject is rebuilt from safe CSR fields only; UNSTRUCTURED_NAME (role) is always
    set from cert_type (the Project Admin's authoritative -t argument), never from the CSR.
    """
    now = datetime.datetime.utcnow()
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
    safe_attrs = [attr for attr in csr.subject if attr.oid in _SAFE_OIDS]
    safe_attrs.append(x509.NameAttribute(NameOID.UNSTRUCTURED_NAME, cert_type))
    safe_subject = x509.Name(safe_attrs)

    builder = (
        x509.CertificateBuilder()
        .subject_name(safe_subject)
        .issuer_name(ca_cert.subject)
        .public_key(csr.public_key())
        .serial_number(serial)
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
    if getattr(args, "schema", False):
        import nvflare.tool.cert.cert_cli as _cert_cli
        from nvflare.tool.cli_schema import parser_to_schema

        _cert_cli._ensure_parsers_initialized()

        schema = parser_to_schema(
            _cert_cli._cert_sign_parser,
            command="nvflare cert sign",
            examples=[
                "nvflare cert sign -r ./hospital-1.csr -c ./ca -o ./signed -t client",
                "nvflare cert sign -r ./alice.csr -c ./ca -o ./alice-signed -t lead --output json",
            ],
        )
        print(json.dumps(schema, indent=2))
        return 0

    # 2. Validate required args (-t/--type is optional; read from CSR if omitted)
    output_fmt = getattr(args, "output_fmt", None)
    for flag, attr in (
        ("-r/--csr", "csr_path"),
        ("-c/--ca-dir", "ca_dir"),
        ("-o/--output-dir", "output_dir"),
    ):
        if not getattr(args, attr, None):
            message, hint = get_error("INVALID_ARGS", detail=f"{flag} is required")
            output_error("INVALID_ARGS", message, hint, output_fmt, exit_code=2)

    # 3. --output json implies --force
    if output_fmt == "json":
        args.force = True

    # 4. Validate CSR file exists
    csr_path = args.csr_path
    if not os.path.exists(csr_path):
        message, hint = get_error("CSR_NOT_FOUND", path=csr_path)
        output_error("CSR_NOT_FOUND", message, hint, output_fmt)

    # 4. Validate CA dir
    ca_dir = args.ca_dir
    ca_key_path = os.path.join(ca_dir, "rootCA.key")
    ca_cert_path = os.path.join(ca_dir, "rootCA.pem")
    ca_json_path = os.path.join(ca_dir, "ca.json")
    for path in (ca_key_path, ca_cert_path, ca_json_path):
        if not os.path.exists(path):
            message, hint = get_error("CA_NOT_FOUND", ca_dir=ca_dir)
            output_error("CA_NOT_FOUND", message, hint, output_fmt)

    # 5. Load and validate CSR
    with open(csr_path, "rb") as f:
        csr_data = f.read()
    try:
        csr = x509.load_pem_x509_csr(csr_data, default_backend())
    except Exception as e:
        message, hint = get_error("INVALID_CSR", path=csr_path)
        output_error("INVALID_CSR", message, hint, output_fmt)

    if not csr.is_signature_valid:
        message, hint = get_error("INVALID_CSR", path=csr_path)
        output_error("INVALID_CSR", message, hint, output_fmt)

    # 6. Resolve cert type: -t is authoritative when given; otherwise read from CSR UNSTRUCTURED_NAME.
    _VALID_CERT_TYPES = {"client", "server", "org_admin", "lead", "member"}
    cert_type = getattr(args, "cert_type", None)
    if not cert_type:
        # Read proposed role from CSR subject UNSTRUCTURED_NAME (set by 'cert csr -t')
        _csr_role_attrs = csr.subject.get_attributes_for_oid(NameOID.UNSTRUCTURED_NAME)
        if _csr_role_attrs:
            cert_type = _csr_role_attrs[0].value
    if not cert_type or cert_type not in _VALID_CERT_TYPES:
        message, hint = get_error(
            "INVALID_ARGS", detail="-t/--type is required (or embed role in CSR with 'cert csr -t')"
        )
        output_error("INVALID_ARGS", message, hint, output_fmt, exit_code=2)
    subject_cn = _get_cn(csr.subject)
    if os.sep in subject_cn or (os.altsep and os.altsep in subject_cn) or subject_cn.startswith("."):
        message, hint = get_error("INVALID_NAME", name=subject_cn, reason="CSR subject CN must not contain path separators or start with '.'.")
        output_error("INVALID_NAME", message, hint, output_fmt, exit_code=4)
    output_filename = f"{subject_cn}.crt"

    # 7. Resolve output paths; check for existing cert
    output_dir = os.path.abspath(args.output_dir)
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        message, hint = get_error("OUTPUT_DIR_NOT_WRITABLE", path=output_dir, detail=str(e))
        output_error("OUTPUT_DIR_NOT_WRITABLE", message, hint, output_fmt)

    if not os.access(output_dir, os.W_OK):
        message, hint = get_error("OUTPUT_DIR_NOT_WRITABLE", path=output_dir, detail="directory is not writable")
        output_error("OUTPUT_DIR_NOT_WRITABLE", message, hint, output_fmt)

    cert_out_path = os.path.join(output_dir, output_filename)
    rootca_out_path = os.path.join(output_dir, "rootCA.pem")

    if os.path.exists(cert_out_path) and not args.force:
        message, hint = get_error("CERT_ALREADY_EXISTS", path=cert_out_path)
        output_error("CERT_ALREADY_EXISTS", message, hint, output_fmt)

    # 8. Load CA material
    try:
        ca_cert = load_crt(ca_cert_path)
        ca_key = load_private_key_file(ca_key_path)
    except Exception as e:
        message, hint = get_error("CA_NOT_FOUND", ca_dir=ca_dir)
        output_error("CA_NOT_FOUND", message, hint, output_fmt)

    # 9. Atomically claim serial from ca.json (read + increment under exclusive lock)
    audit_serial = _claim_serial(ca_json_path)

    # 10. Build and sign the certificate
    valid_days = getattr(args, "valid_days", 1095) or 1095
    try:
        signed_cert = _build_signed_cert(
            csr=csr,
            ca_cert=ca_cert,
            ca_key=ca_key,
            cert_type=cert_type,
            valid_days=valid_days,
            serial=audit_serial,
        )
    except Exception as e:
        message, hint = get_error("CERT_SIGNING_FAILED", reason=str(e))
        output_error("CERT_SIGNING_FAILED", message, hint, output_fmt)

    # 11. Write signed cert and copy rootCA.pem
    with open(cert_out_path, "wb") as f:
        f.write(serialize_cert(signed_cert))
    shutil.copy2(ca_cert_path, rootca_out_path)

    # 12. Compute valid_until for output
    try:
        _valid_until_dt = signed_cert.not_valid_after_utc
    except AttributeError:
        _valid_until_dt = signed_cert.not_valid_after  # cryptography < 42.0
    valid_until = _valid_until_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # 14. Output result
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
        "serial": audit_serial,
        "valid_until": valid_until,
        "next_step": next_step,
    }
    if not output_fmt:
        print("CSR signed successfully.")
        print(f"  Signed cert:  {cert_out_path}")
        print(f"  Root CA:      {rootca_out_path}  (also included for convenience)")
        print(f"  Subject:      {subject_cn} ({cert_type})")
        print(f"  Serial:       {audit_serial}")
        print(f"  Valid until:  {_valid_until_dt.strftime('%Y-%m-%d')}")
        print()
        print(f"Next step: Send {output_filename} and rootCA.pem to the site admin.")
        print(f"They place those files in the same directory as their {subject_cn}.key, then run:")
        print("  nvflare package -e grpc://<server>:<port> --dir <that-dir>")
    else:
        output(result, output_fmt)
    return 0
