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

"""Prepare isolated public profile inputs and temporary lab PKI on a Linux host.

The PKI is for disposable tests only. It never reads existing KBS private keys.
"""

import argparse
import datetime
import ipaddress
import os
import subprocess
import tempfile
from pathlib import Path

import yaml
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
from cvm.common.contracts import HEADER_BYTES, STORAGE_PROFILE
from cvm.common.io import write_json


def prepare(directory, *, http_only=False):
    directory = Path(directory).resolve()
    inputs = directory / "inputs"
    inputs.mkdir(exist_ok=True)
    tdx_firmware = Path(os.environ.get("CVM_TDX_FIRMWARE", inputs / "OVMF.inteltdx.fd")).resolve()
    if not http_only and not tdx_firmware.is_file():
        raise SystemExit("Provide the validated TDVF at inputs/OVMF.inteltdx.fd or set CVM_TDX_FIRMWARE")
    # These are KBS deployment keys, not vault build scratch. Keep them in a
    # protected backend directory: logind may remove user-owned /dev/shm files
    # when the last SSH session exits on a shared lab host.
    pki = Path(tempfile.mkdtemp(prefix="lab-pki-", dir=directory))
    now = datetime.datetime.now(datetime.timezone.utc)
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Disposable CVM test CA")])
    ca = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(days=3))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
        .add_extension(x509.KeyUsage(True, False, False, False, False, True, True, None, None), critical=True)
        .sign(ca_key, hashes.SHA256())
    )
    (inputs / "test-ca.pem").write_bytes(ca.public_bytes(serialization.Encoding.PEM))

    as_ca_key = ec.generate_private_key(ec.SECP256R1())
    as_ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Disposable CVM AS signing CA")])
    as_ca = (
        x509.CertificateBuilder()
        .subject_name(as_ca_name)
        .issuer_name(as_ca_name)
        .public_key(as_ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(days=3))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(as_ca_key.public_key()), critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(as_ca_key.public_key()), critical=False)
        .add_extension(x509.KeyUsage(True, False, False, False, False, True, True, None, None), critical=True)
        .sign(as_ca_key, hashes.SHA256())
    )
    (inputs / "test-as-ca.pem").write_bytes(as_ca.public_bytes(serialization.Encoding.PEM))

    def keyfile(name, key):
        path = pki / (name + ".key")
        path.write_bytes(
            key.private_bytes(
                serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
            )
        )
        path.chmod(0o600)

    keyfile("ca", ca_key)
    keyfile("as-ca", as_ca_key)
    for name in ("server", "builder", "admin", "untrusted", "as"):
        issuer, issuer_key = (as_ca, as_ca_key) if name == "as" else (ca, ca_key)
        key = ec.generate_private_key(ec.SECP256R1())
        keyfile(name, key)
        builder = (
            x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)]))
            .issuer_name(issuer.subject)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(minutes=1))
            .not_valid_after(now + datetime.timedelta(days=3))
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        )
        builder = (
            builder.add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
            .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(issuer_key.public_key()), critical=False)
            .add_extension(x509.KeyUsage(True, False, False, False, False, False, False, None, None), critical=True)
        )
        if name == "server":
            builder = builder.add_extension(
                x509.SubjectAlternativeName(
                    [
                        x509.DNSName("localhost"),
                        x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
                        x509.IPAddress(ipaddress.ip_address("10.0.2.2")),
                    ]
                ),
                critical=False,
            )
        builder = builder.add_extension(
            x509.ExtendedKeyUsage(
                [ExtendedKeyUsageOID.SERVER_AUTH if name == "server" else ExtendedKeyUsageOID.CLIENT_AUTH]
            ),
            critical=False,
        )
        cert = builder.sign(issuer_key, hashes.SHA256())
        (pki / (name + ".pem")).write_bytes(cert.public_bytes(serialization.Encoding.PEM))
        if name == "as":
            (inputs / "test-as-public.pem").write_bytes(
                key.public_key().public_bytes(
                    serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
                )
            )
    admin_key = ed25519.Ed25519PrivateKey.generate()
    keyfile("kbs-admin", admin_key)
    (pki / "kbs-admin.pub").write_bytes(
        admin_key.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    )
    if http_only:
        write_json(directory / "lab-state.json", {"pki": str(pki)})
        print("Prepared disposable HTTPS test PKI")
        return
    names = [
        "python3",
        "python3-yaml",
        "python3-cryptography",
        "initramfs-tools",
        "cryptsetup-bin",
        "nfs-common",
        "nftables",
        "e2fsprogs",
        "dmsetup",
        "iproute2",
        "chrony",
        "docker.io",
        "containerd",
        "linux-image-7.0.0-31-generic",
        "linux-modules-7.0.0-31-generic",
    ]
    pins = {}
    for name in names:
        text = subprocess.check_output(["apt-cache", "policy", name], text=True)
        version = next(
            line.split(":", 1)[1].strip() for line in text.splitlines() if line.strip().startswith("Candidate:")
        )
        assert version != "(none)", name
        pins[name] = version
    profile = {
        "profile_version": "test-cpu-2026.09",
        "gpu": "none",
        "guest_release": "26.04",
        "base_image": str(inputs / "ubuntu-26.04-server-cloudimg-amd64.img"),
        "build_firmware": "/usr/share/ovmf/OVMF.fd",
        "kbs_url": "https://10.0.2.2:19199",
        "kbs_cert": str(inputs / "test-ca.pem"),
        "as_public_key": str(inputs / "test-as-public.pem"),
        "token_algorithm": "ES256",
        "token_issuer": "CoCo-Attestation-Service",
        "bootstrap_egress": [443, 19199],
        "vcpus": 4,
        "memory_gib": 8,
        "root_overlay_max_mib": 4096,
        "root_drive_size": 8,
        "trustee_commit": "512fed65642015b849f38fb13bfdec7806639987",
        "attestation_policy_id": "default",
        "vault_header_bytes": HEADER_BYTES,
        "vault_storage_profile": STORAGE_PROFILE,
        "kernel_version": "7.0.0-31-generic",
        "python_version": pins["python3"],
        "docker_version": pins["docker.io"],
        "containerd_version": pins["containerd"],
        "cryptsetup_version": pins["cryptsetup-bin"],
        "required_system_packages": [name + "=" + version for name, version in pins.items()],
        "attestation_policy": str(directory / "config/attestation_policy.rego"),
        "reference_values": str(inputs / "test-tcb-references.json"),
        "platforms": {
            "intel_tdx": {
                "firmware": str(tdx_firmware),
                "attester": "tdx",
                "cpu_model": "host",
                "quote_generation": {"type": "vsock", "cid": 2, "port": 4050},
                "kbs_client": str(inputs / "kbs-client"),
            },
            "amd_sev_snp": {
                "firmware": "/usr/share/ovmf/OVMF.amdsev.fd",
                "attester": "snp",
                "cpu_model": "EPYC-v4",
                "kbs_client": str(inputs / "kbs-client"),
            },
        },
    }
    # Empty TCB references deliberately deny approval until actual evidence is
    # reviewed. Never copy measured guest values into an allow policy blindly.
    write_json(inputs / "test-tcb-references.json", {})
    (inputs / "lab-profile.yml").write_text(yaml.safe_dump(profile))
    write_json(directory / "lab-state.json", {"pki": str(pki), "profile": str(inputs / "lab-profile.yml")})
    print("Prepared disposable lab inputs and PKI")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--http-only", action="store_true")
    args = parser.parse_args()
    prepare(args.directory, http_only=args.http_only)
