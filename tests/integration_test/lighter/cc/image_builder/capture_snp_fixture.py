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

"""Export an EAR from upstream Trustee's signed SNP fixture, using an isolated AS."""

import argparse
import base64
import hashlib
import json
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cvm.build.config import SOURCE
from cvm.common.io import read_json, write_json
from cvm.runtime.attestation import validate_token
from cvm.trustee.provenance import provenance


def capture(source, output):
    source = Path(source).resolve()
    binary = source / "target/release/restful-as"
    upstream = provenance(source, binary)
    evidence_path = "attestation-service/tests/e2e/evidence.json"
    evidence = (source / evidence_path).read_bytes()
    report = json.loads(evidence)["attestation_report"]
    policy, platform = report["policy"], report["plat_info"]
    references = {
        "snp_launch_measurement": [bytes(report["measurement"]).hex()],
        "snp_bootloader": [report["reported_tcb"]["bootloader"]],
        "snp_tee_svn": [report["reported_tcb"]["tee"]],
        "snp_snp_svn": [report["reported_tcb"]["snp"]],
        "snp_microcode": [report["reported_tcb"]["microcode"]],
        "snp_smt_enabled": bool(platform & 1),
        "snp_tsme_enabled": bool(platform & 2),
        "snp_guest_abi_major": (policy >> 8) & 255,
        "snp_guest_abi_minor": policy & 255,
        "snp_single_socket": bool(policy & (1 << 20)),
        "snp_smt_allowed": bool(policy & (1 << 16)),
    }
    # These are test-only approvals for upstream's archived public evidence.
    # No reference values or policies in an existing deployment are changed.
    with tempfile.TemporaryDirectory(prefix="snp-fixture-") as temporary:
        root = Path(temporary)
        storage = root / "storage"
        policies = storage / "attestation_service_policy"
        policies.mkdir(parents=True)
        (policies / "default_cpu.rego").write_bytes((SOURCE / "config/attestation_policy.rego").read_bytes())
        references["cvm_reference_expiry"] = {name: time.time() + 600 for name in references}
        for name, value in references.items():
            write_json(
                storage / "reference_value" / name,
                {"version": "0.1.0", "name": name, "value": value, "expiration": "2099-01-01T00:00:00Z"},
            )
        key = ec.generate_private_key(ec.SECP256R1())
        (root / "as.key").write_bytes(
            key.private_bytes(
                serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
            )
        )
        public = key.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
        (root / "as.pem").write_bytes(public)
        config = {
            "storage_backend": {"storage_type": "local_fs", "backends": {"local_fs": {"dir_path": str(storage)}}},
            "attestation_token_broker": {
                "duration_min": 5,
                "issuer_name": "CoCo-Attestation-Service",
                "signer": {"key_path": str(root / "as.key")},
            },
        }
        write_json(root / "config.json", config)
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        endpoint = f"http://127.0.0.1:{port}"
        with (root / "as.log").open("wb") as log:
            process = subprocess.Popen(
                [str(binary), "-s", f"127.0.0.1:{port}", "-c", str(root / "config.json")], stdout=log, stderr=log
            )
            try:
                for _ in range(100):
                    if process.poll() is not None:
                        raise RuntimeError("Fixture AS exited: " + (root / "as.log").read_text())
                    try:
                        with urllib.request.urlopen(endpoint + "/.well-known/jwks.json", timeout=1):
                            break
                    except urllib.error.URLError:
                        time.sleep(0.1)
                request = {
                    "verification_requests": [
                        {"tee": "snp", "evidence": base64.urlsafe_b64encode(evidence).decode().rstrip("=")}
                    ],
                    "policy_ids": ["default"],
                }
                call = urllib.request.Request(
                    endpoint + "/attestation",
                    data=json.dumps(request).encode(),
                    headers={"Content-Type": "application/json"},
                )
                try:
                    with urllib.request.urlopen(call, timeout=60) as response:
                        token = response.read()
                except urllib.error.HTTPError as error:
                    raise RuntimeError(error.read().decode()) from None
                claims = validate_token(
                    token,
                    {
                        "as_public_key": str(root / "as.pem"),
                        "token_algorithm": "ES256",
                        "token_issuer": "CoCo-Attestation-Service",
                        "platform": "amd_sev_snp",
                        "attestation_policy_id": "default",
                    },
                    bytes(report["host_data"]),
                )
                fixture = {
                    **upstream,
                    "evidence_source": f"https://github.com/confidential-containers/trustee/blob/{upstream['trustee_commit']}/{evidence_path}",
                    "evidence_sha256": hashlib.sha256(evidence).hexdigest(),
                    "as_policy_sha256": hashlib.sha256(
                        (SOURCE / "config/attestation_policy.rego").read_bytes()
                    ).hexdigest(),
                    "references": {name: value for name, value in references.items() if name != "cvm_reference_expiry"},
                    "token": token.decode(),
                    "as_public_key": public.decode(),
                    "issued_at": claims["iat"],
                }
                write_json(output, fixture)
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
    assert read_json(output)["source_clean"]
    print("Captured and signature-verified a positive SNP EAR from unmodified Trustee.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trustee_source", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    capture(arguments.trustee_source, arguments.output)
