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

"""Run an isolated, deny-by-default lab upstream Trustee.

Requires unmodified CoCo Trustee v0.22.0 binaries on
the lab host, and the disposable PKI from prepare_lab.py. Never touches another
KBS deployment or its credentials.
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cvm.build.config import SOURCE
from cvm.common.io import canonical, read_json, write_json
from cvm.trustee.client import encode
from cvm.trustee.provenance import provenance


def main(directory):
    directory = Path(directory).resolve()
    pki = Path(read_json(directory / "lab-state.json")["pki"])
    kbs_port = int(os.environ.get("CVM_LAB_KBS_PORT", "19199"))
    state = directory / "lab-kbs"
    state.mkdir(mode=0o700, exist_ok=True)
    storage = state / "storage"
    policies = storage / "attestation_service_policy"
    policies.mkdir(parents=True, exist_ok=True)
    strict = (directory / "config/attestation_policy.rego").read_bytes()
    (policies / "default_cpu.rego").write_bytes(strict)
    from cvm.common.gpu_policy import render

    (policies / "default_gpu.rego").write_text(render(read_json(directory / "config/gpu_policy.json")))
    for device in ("switch", "ppcie"):
        (policies / ("default_" + device + ".rego")).write_text(
            'package policy\nimport rego.v1\ntrust_claims := {"hardware": 97}\n'
        )
    (storage / "reference_value").mkdir(exist_ok=True)
    resource_policy = storage / "kbs/resource-policy.rego"
    resource_policy.parent.mkdir(exist_ok=True)
    if not resource_policy.exists():
        resource_policy.write_text("package policy\nimport rego.v1\ndefault allow := false\n")
    (pki / "as-chain.pem").write_bytes(
        (pki / "as.pem").read_bytes() + (directory / "inputs/test-as-ca.pem").read_bytes()
    )
    (storage / "repository").mkdir(mode=0o700, exist_ok=True)
    config = {
        "http_server": {
            "sockets": [f"127.0.0.1:{kbs_port}"],
            "insecure_http": False,
            "private_key": str(pki / "server.key"),
            "certificate": str(pki / "server.pem"),
        },
        "attestation_token": {
            "insecure_header_jwk": False,
            "trusted_certs_paths": [str(directory / "inputs/test-as-ca.pem")],
        },
        "admin": {
            "authorization_mode": "AuthenticatedAuthorization",
            "authentication": {
                "bearer_jwt": {
                    "identity_providers": [
                        {
                            "issuer": "cvm-builder",
                            "audience": "coco-trustee",
                            "public_key_uri": str(pki / "kbs-admin.pub"),
                        }
                    ]
                }
            },
            "authorization": {
                "regex_acl": {
                    "acls": [
                        {
                            "role": "cvm-policy",
                            "allowed_endpoints": "^/kbs/v0/(resource-policy|reference-value/[^/]+)$",
                        },
                        {
                            "role": "cvm-resources",
                            "allowed_endpoints": "^/kbs/v0/resource/keys/[^/]+/[^/]+$",
                        },
                    ]
                }
            },
        },
        "storage_backend": {
            "storage_type": "local_fs",
            "backends": {"local_fs": {"dir_path": str(storage)}},
        },
        "session_storage_type": "memory",
        "attestation_service": {
            "type": "coco_as_builtin",
            "verifier_config": read_json(SOURCE / "trustee/kbs.json")["attestation_service"]["verifier_config"],
            "attestation_token_broker": {
                "duration_min": 5,
                "issuer_name": "CoCo-Attestation-Service",
                "signer": {"key_path": str(pki / "as.key"), "cert_path": str(pki / "as-chain.pem")},
            },
        },
        "plugins": [{"name": "resource", "storage_backend_type": "kvstorage"}],
    }
    write_json(state / "kbs.json", config)
    # Only the lab issuer has the signing key; builders get a fixed-role token.
    key = serialization.load_pem_private_key((pki / "kbs-admin.key").read_bytes(), None)
    now = int(time.time())
    claims = {
        "iat": now,
        "nbf": now - 5,
        "exp": now + 3600,
        "role": "cvm-resources",
        "iss": "cvm-builder",
        "aud": "coco-trustee",
    }
    body = encode(canonical({"alg": "EdDSA", "typ": "JWT"})) + "." + encode(canonical(claims))
    token_path = pki / "resource-token.jwt"
    token_path.write_text(body + "." + encode(key.sign(body.encode())))
    token_path.chmod(0o600)
    write_json(
        state / "resources.json",
        {
            "url": f"https://127.0.0.1:{kbs_port}",
            "ca": str(directory / "inputs/test-ca.pem"),
            "admin_token_file": str(token_path),
        },
    )
    config_admin = {
        "url": f"https://127.0.0.1:{kbs_port}",
        "ca": str(directory / "inputs/test-ca.pem"),
        "admin_private_key": str(pki / "kbs-admin.key"),
        "storage_directory": str(storage),
        "state": str(state / "admin"),
        "deployment_receipt": str(state / "deployment-receipt.json"),
        "trustee_binary": str(directory / "trustee-source/target/release/kbs"),
        "trustee_build": str(state / "trustee_build.json"),
    }
    write_json(state / "admin.json", config_admin)
    write_json(
        state / "trustee_build.json",
        provenance(directory / "trustee-source", directory / "trustee-source/target/release/kbs"),
    )
    processes = []
    stopping = False

    def terminate(signum, frame):
        nonlocal stopping
        stopping = True

    previous = {number: signal.signal(number, terminate) for number in (signal.SIGTERM, signal.SIGHUP)}
    try:
        log = (state / "kbs.log").open("ab")
        processes.append(
            subprocess.Popen(
                [
                    "sudo",
                    "-n",
                    "env",
                    "RUST_LOG=warn",
                    str(directory / "trustee-source/target/release/kbs"),
                    "--config-file",
                    str(state / "kbs.json"),
                ],
                stdout=log,
                stderr=log,
                env=dict(os.environ, RUST_LOG="warn"),
            )
        )
        write_json(state / "pids.json", {"kbs": processes[0].pid})
        print("Isolated upstream Trustee started", flush=True)
        while not stopping and all(process.poll() is None for process in processes):
            time.sleep(1)
        if stopping:
            return
        raise RuntimeError("Lab service exited; inspect its protected log")
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        for number, handler in previous.items():
            signal.signal(number, handler)


if __name__ == "__main__":
    main(sys.argv[1])
