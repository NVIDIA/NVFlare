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

"""Real HTTPS contracts with disposable signed EAR fixtures (not hardware proof)."""

import base64
import json
import os
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa, utils
from cvm.build.config import SOURCE
from cvm.common.contracts import resource_path
from cvm.common.errors import BuildError
from cvm.common.io import canonical, read_json, write_json
from cvm.common.linux import memory_file, run
from cvm.common.policy import compose
from cvm.runtime.attestation import unb64url, validate_token
from cvm.trustee.admin import read_resource_policy, verify_readback
from cvm.trustee.client import api, delete_resource, encode, upload_resource


@unittest.skipUnless(os.environ.get("CVM_HTTP_TESTS") == "1", "Opt-in isolated lab HTTPS tests")
class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = Path(os.environ.get("CVM_LAB_DIRECTORY", SOURCE)).resolve()
        cls.pki = Path(read_json(cls.directory / "lab-state.json")["pki"])
        cls.admin = read_json(cls.directory / "lab-kbs/admin.json")
        cls.resources = read_json(cls.directory / "lab-kbs/resources.json")
        cls.original_policy = read_resource_policy(cls.admin)
        cls.build_id = "http-test-" + os.urandom(8).hex()
        cls.manifest = {
            "build_id": cls.build_id,
            "platform": "intel_tdx",
            "attestation_policy_id": "default",
            "measurements": {"mr_td": "1" * 96, "rtmr_0": "0" * 96, "rtmr_1": "2" * 96, "rtmr_2": "3" * 96},
        }
        policy = compose([cls.manifest]).encode()
        api(cls.admin, "POST", "resource-policy", canonical({"policy": encode(policy)}))
        verify_readback(policy, read_resource_policy(cls.admin))
        cls.paths = []

    @classmethod
    def tearDownClass(cls):
        for path in cls.paths:
            delete_resource(cls.resources, path)
        api(cls.admin, "POST", "resource-policy", canonical({"policy": encode(cls.original_policy)}))
        assert read_resource_policy(cls.admin) == cls.original_policy

    def create(self):
        digest = os.urandom(32)
        path = resource_path(self.build_id, "intel_tdx", digest)
        secret = os.urandom(64)
        upload_resource(self.resources, path, secret)
        self.paths.append(path)
        return digest, path, secret

    def token(self, digest, tee, *, status="affirming", policy="default", expired=False, signer="as", gpu_submods=None):
        key = serialization.load_pem_private_key((self.pki / (signer + ".key")).read_bytes(), None)
        public = key.public_key().public_numbers()
        certs = [
            (self.pki / (signer + ".pem")).read_bytes(),
            (
                (self.directory / "inputs/test-as-ca.pem").read_bytes()
                if signer == "as"
                else Path(self.admin["ca"]).read_bytes()
            ),
        ]
        jwk = {
            "kty": "EC",
            "crv": "P-256",
            "alg": "ES256",
            "x": encode(public.x.to_bytes(32, "big")),
            "y": encode(public.y.to_bytes(32, "big")),
            "x5c": [
                base64.b64encode(x509.load_pem_x509_certificate(cert).public_bytes(serialization.Encoding.DER)).decode()
                for cert in certs
            ],
        }
        public_tee = tee.public_key().public_numbers()
        evidence = {
            "init_data": (digest + bytes(16)).hex(),
            "tdx": {"quote": {"body": self.manifest["measurements"]}, "td_attributes": {"debug": False}},
            "runtime_data_claims": {
                "tee-pubkey": {
                    "kty": "RSA",
                    "alg": "RSA-OAEP-256",
                    "n": encode(public_tee.n.to_bytes(256, "big")),
                    "e": encode(public_tee.e.to_bytes(3, "big")),
                }
            },
        }
        claims = {
            "iat": int(time.time()),
            "exp": int(time.time()) + (-60 if expired else 120),
            "submods": {
                "cpu0": {
                    "ear.appraisal-policy-id": policy,
                    "ear.status": status,
                    "ear.trustworthiness-vector": {"executables": 3, "hardware": 2, "configuration": 2},
                    "ear.veraison.annotated-evidence": evidence,
                }
            },
        }
        claims["submods"].update(gpu_submods or {})
        body = encode(canonical({"alg": "ES256", "jwk": jwk})) + "." + encode(canonical(claims))
        r, s = utils.decode_dss_signature(key.sign(body.encode(), ec.ECDSA(hashes.SHA256())))
        return (body + "." + encode(r.to_bytes(32, "big") + s.to_bytes(32, "big"))).encode()

    def retrieve(self, digest, path, **claims):
        tee = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        pem = tee.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL, serialization.NoEncryption()
        )
        cpu_token = self.token(digest, tee, **claims)
        with memory_file(pem) as private, memory_file(cpu_token) as token:
            value = run(
                [
                    self.directory / "inputs/kbs-client",
                    "--url",
                    self.admin["url"],
                    "--cert-file",
                    self.admin["ca"],
                    "get-resource",
                    "--path",
                    path,
                    "--tee-key-file",
                    f"/proc/self/fd/{private}",
                    "--attestation-token",
                    f"/proc/self/fd/{token}",
                ],
                pass_fds=(private, token),
                timeout=30,
                env=dict(os.environ, RUST_LOG="off"),
            )
            return base64.b64decode(value.strip(), validate=True)

    def test_real_policy_path_and_encrypted_resource_response(self):
        digest, path, secret = self.create()
        self.assertEqual(self.retrieve(digest, path), secret)

    def test_named_reference_api_uses_upstream_local_fs_records(self):
        name = "cvm_http_test_reference"
        target = Path(self.admin["storage_directory"]) / "reference_value" / name
        value = ["public-fixture"]
        write_json(target, {"version": "0.1.0", "name": name, "expiration": "2099-01-01T00:00:00Z", "value": value})
        try:
            self.assertEqual(json.loads(api(self.admin, "GET", "reference-value/" + name)), value)
            with self.assertRaises(BuildError):
                api(self.admin, "GET", "reference-value/absent-fixture")
        finally:
            target.unlink()

    def test_upstream_default_cpu_policy_rejects_sample_evidence(self):
        # The stock minimal CLI uses sample evidence on the non-TEE backend host.
        # This proves the actual AS loaded our policy, not hardware acceptance.
        token = run(
            [
                self.directory / "inputs/kbs-client",
                "--url",
                self.admin["url"],
                "--cert-file",
                self.admin["ca"],
                "attest",
            ],
            timeout=30,
            env=dict(os.environ, RUST_LOG="off"),
        ).strip()
        claims = json.loads(unb64url(token.decode().split(".")[1]))
        self.assertEqual(claims["submods"]["cpu0"]["ear.appraisal-policy-id"], "default")
        config = {
            "token_algorithm": "ES256",
            "token_issuer": "CoCo-Attestation-Service",
            "as_public_key": str(self.directory / "inputs/test-as-public.pem"),
            "attestation_policy_id": "default",
        }
        with self.assertRaisesRegex(BuildError, "CPU appraisal denied"):
            validate_token(token, config, bytes(32))

    def test_cross_vault_access_denied(self):
        a, _, _ = self.create()
        _, b_path, _ = self.create()
        with self.assertRaises(BuildError):
            self.retrieve(a, b_path)

    def test_negative_wrong_policy_and_expired_ear_denied(self):
        digest, path, _ = self.create()
        for claims in ({"status": "contraindicated"}, {"policy": "unselected"}, {"expired": True}):
            with self.subTest(claims=claims), self.assertRaises(BuildError):
                self.retrieve(digest, path, **claims)

    def test_transport_identities_cannot_forge_attestation(self):
        digest, path, _ = self.create()
        for signer in ("builder", "admin", "server", "untrusted"):
            with self.subTest(signer=signer), self.assertRaises(BuildError):
                self.retrieve(digest, path, signer=signer)

    def test_gpu_keys_require_composite_ear_and_matching_policy(self):
        from common.test_gpu_composite import gpu_submod, invalid_submods

        digest, path, secret = self.create()
        gpu_manifest = dict(self.manifest, contract={"gpu": "nvidia_cc", "gpu_count": 2})
        composed = compose([gpu_manifest]).encode()
        api(self.admin, "POST", "resource-policy", canonical({"policy": encode(composed)}))
        verify_readback(composed, read_resource_policy(self.admin))
        try:
            policy = self.manifest["attestation_policy_id"]
            valid = {f"gpu{i}": gpu_submod(policy, i) for i in range(2)}
            self.assertEqual(self.retrieve(digest, path, gpu_submods=valid), secret)
            for submods in invalid_submods(policy, 2):
                with self.subTest(submods=submods), self.assertRaises(BuildError):
                    self.retrieve(digest, path, gpu_submods=submods)
        finally:
            api(self.admin, "POST", "resource-policy", canonical({"policy": encode(compose([self.manifest]).encode())}))

    def test_native_upload_retries_and_replacement_semantics(self):
        digest, path, secret = self.create()
        upload_resource(self.resources, path, secret)
        self.assertEqual(self.retrieve(digest, path), secret)
        replacement = os.urandom(64)
        upload_resource(self.resources, path, replacement)
        self.assertEqual(self.retrieve(digest, path), replacement)

    def test_resource_token_cannot_change_policy(self):
        for endpoint in ("resource-policy", "attestation-policy", "reference-value"):
            with self.subTest(endpoint=endpoint), self.assertRaises(BuildError):
                api(self.resources, "POST", endpoint, canonical({"policy": encode(b"allow := true")}))

    def test_forged_resource_token_cannot_upload(self):
        import tempfile

        path = resource_path(self.build_id, "intel_tdx", os.urandom(32))
        token = Path(self.resources["admin_token_file"]).read_text()
        header, payload, signature = token.split(".")
        claims = json.loads(unb64url(payload))
        claims["role"] = "cvm-policy"
        with tempfile.TemporaryDirectory() as directory:
            token_path = Path(directory) / "forged.jwt"
            token_path.write_text(header + "." + encode(canonical(claims)) + "." + signature)
            with self.assertRaises(BuildError):
                upload_resource(dict(self.resources, admin_token_file=str(token_path)), path, os.urandom(64))

    def test_native_delete_denies_release_but_does_not_tombstone(self):
        digest, path, secret = self.create()
        delete_resource(self.resources, path)
        with self.assertRaises(BuildError):
            self.retrieve(digest, path)
        # This is deliberately native CoCo behavior. Operators must fence uploads
        # and preserve deletions across restore, rather than relying on tombstones.
        upload_resource(self.resources, path, secret)
        self.assertEqual(self.retrieve(digest, path), secret)

    def test_policy_role_cannot_mutate_resources_or_replace_as(self):
        digest, path, secret = self.create()
        policy_before = read_resource_policy(self.admin)
        for method, expected in (("POST", 401), ("PUT", 401), ("DELETE", 401)):
            # Preserve the real HTTP code while using the same signed admin
            # request and TLS verification as production administration.
            codes = []
            original_open = urllib.request.OpenerDirector.open

            def observe(opener, *args, **kwargs):
                try:
                    return original_open(opener, *args, **kwargs)
                except urllib.error.HTTPError as error:
                    codes.append(error.code)
                    raise

            from unittest.mock import patch

            with patch.object(urllib.request.OpenerDirector, "open", observe), self.assertRaises(BuildError):
                api(self.admin, method, "resource/" + path, b"x" * 64)
            self.assertEqual(codes, [expected])
            self.assertEqual(self.retrieve(digest, path), secret)
            self.assertEqual(read_resource_policy(self.admin), policy_before)
        with self.assertRaises(BuildError):
            api(
                self.admin,
                "POST",
                "attestation-policy",
                canonical(
                    {
                        "policy_id": "default_cpu",
                        "type": "rego",
                        "policy": encode(b"package policy\ndefault executables = 3\n"),
                    }
                ),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
