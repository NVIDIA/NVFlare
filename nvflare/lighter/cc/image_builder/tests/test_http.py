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
import os
import ssl
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from builder.admin import api, encode, verify_readback
from builder.common import BuildError, canonical, memory_file, read_json, resource_path, run, write_json
from builder.key_service import ResourceStore, request
from builder.policy import compose
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa, utils


@unittest.skipUnless(os.environ.get("CVM_HTTP_TESTS") == "1", "Opt-in isolated lab HTTPS tests")
class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = Path(os.environ.get("CVM_LAB_DIRECTORY", Path(__file__).resolve().parent.parent)).resolve()
        cls.pki = Path(read_json(cls.directory / "lab-state.json")["pki"])
        cls.admin = read_json(cls.directory / "lab-kbs/admin.json")
        cls.store = ResourceStore(cls.admin["resources"], cls.admin["key_service_state"])
        cls.original_policy = api(cls.admin, "GET", "resource-policy")
        cls.original_ids = read_json(cls.store.state / "approved-bundles.json")
        cls.build_id = "http-test-" + os.urandom(8).hex()
        cls.manifest = {
            "build_id": cls.build_id,
            "platform": "intel_tdx",
            "attestation_policy_id": "cvm-v2-test-r1",
            "measurements": {"mr_td": "1" * 96, "rtmr_0": "0" * 96, "rtmr_1": "2" * 96, "rtmr_2": "3" * 96},
        }
        policy = compose([cls.manifest]).encode()
        api(cls.admin, "POST", "resource-policy", canonical({"policy": encode(policy)}))
        verify_readback(policy, api(cls.admin, "GET", "resource-policy"))
        write_json(
            cls.store.state / "approved-bundles.json", {"build_ids": cls.original_ids["build_ids"] + [cls.build_id]}
        )
        cls.paths = []

    @classmethod
    def tearDownClass(cls):
        for path in cls.paths:
            request(cls.key_service("admin"), "DELETE", path)
        write_json(cls.store.state / "approved-bundles.json", cls.original_ids)
        api(cls.admin, "POST", "resource-policy", canonical({"policy": cls.original_policy.decode()}))
        assert api(cls.admin, "GET", "resource-policy") == cls.original_policy

    @classmethod
    def key_service(cls, role="builder"):
        return {
            "url": "https://127.0.0.1:19200",
            "ca": cls.admin["ca"],
            "cert": str(cls.pki / (role + ".pem")),
            "key": str(cls.pki / (role + ".key")),
        }

    def create(self):
        digest = os.urandom(32)
        path = resource_path(self.build_id, "intel_tdx", digest)
        secret = os.urandom(64)
        request(self.key_service(), "PUT", path, secret)
        self.paths.append(path)
        return digest, path, secret

    def token(self, digest, tee, *, status="affirming", policy="cvm-v2-test-r1", expired=False):
        key = serialization.load_pem_private_key((self.pki / "as.key").read_bytes(), None)
        public = key.public_key().public_numbers()
        certs = [(self.pki / "as.pem").read_bytes(), Path(self.admin["ca"]).read_bytes()]
        jwk = {
            "kty": "EC",
            "crv": "P-256",
            "alg": "ES256",
            "x": encode(public.x.to_bytes(32, "big")),
            "y": encode(public.y.to_bytes(32, "big")),
            # This baseline's EAR broker and verifier both use URL_SAFE_NO_PAD
            # here (unlike JOSE's standard-base64 x5c). Match its pinned wire contract.
            "x5c": [
                encode(x509.load_pem_x509_certificate(cert).public_bytes(serialization.Encoding.DER)) for cert in certs
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
        body = encode(canonical({"alg": "ES256", "jwk": jwk})) + "." + encode(canonical(claims))
        r, s = utils.decode_dss_signature(key.sign(body.encode(), ec.ECDSA(hashes.SHA256())))
        return (body + "." + encode(r.to_bytes(32, "big") + s.to_bytes(32, "big"))).encode()

    def retrieve(self, digest, path, **claims):
        tee = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        pem = tee.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL, serialization.NoEncryption()
        )
        with memory_file(pem) as private, memory_file(self.token(digest, tee, **claims)) as token:
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

    def test_cross_vault_access_denied(self):
        a, _, _ = self.create()
        _, b_path, _ = self.create()
        with self.assertRaises(BuildError):
            self.retrieve(a, b_path)

    def test_negative_wrong_policy_and_expired_ear_denied(self):
        digest, path, _ = self.create()
        for claims in ({"status": "contraindicated"}, {"policy": "default"}, {"expired": True}):
            with self.subTest(claims=claims), self.assertRaises(BuildError):
                self.retrieve(digest, path, **claims)

    def test_idempotent_upload_and_conflict(self):
        _, path, secret = self.create()
        request(self.key_service(), "PUT", path, secret)
        with self.assertRaises(BuildError):
            request(self.key_service(), "PUT", path, os.urandom(64))

    def test_builder_cannot_revoke_or_change_policy(self):
        _, path, _ = self.create()
        with self.assertRaises(BuildError):
            request(self.key_service(), "DELETE", path)
        context = ssl.create_default_context(cafile=self.admin["ca"])
        context.load_cert_chain(self.key_service()["cert"], self.key_service()["key"])
        for endpoint in ("resource-policy", "attestation-policy", "reference-value"):
            payload = canonical(
                {
                    "policy_id": "cvm-v2-test-r1_cpu",
                    "type": "rego",
                    "policy": encode(b"package policy\ndefault executables = 3\n"),
                }
            )
            req = urllib.request.Request(self.admin["url"] + "/kbs/v0/" + endpoint, data=payload, method="POST")
            with self.subTest(endpoint=endpoint), self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(req, context=context)
            self.assertIn(error.exception.code, (401, 403))
            error.exception.close()

    def test_unknown_client_certificate_is_denied(self):
        path = resource_path(self.build_id, "intel_tdx", os.urandom(32))
        with self.assertRaises(BuildError):
            request(self.key_service("untrusted"), "PUT", path, os.urandom(64))

    def test_revoked_key_is_unavailable_and_cannot_be_recreated(self):
        digest, path, secret = self.create()
        request(self.key_service("admin"), "DELETE", path)
        with self.assertRaises(BuildError):
            self.retrieve(digest, path)
        with self.assertRaises(BuildError):
            request(self.key_service(), "PUT", path, secret)

    def test_native_resource_mutation_and_as_replacement_denied(self):
        _, path, _ = self.create()
        with self.assertRaises(BuildError):
            api(self.admin, "POST", "resource/" + path, b"x" * 64)
        with self.assertRaises(BuildError):
            api(
                self.admin,
                "POST",
                "attestation-policy",
                canonical(
                    {
                        "policy_id": "cvm-v2-test-r1_cpu",
                        "type": "rego",
                        "policy": encode(b"package policy\ndefault executables = 3\n"),
                    }
                ),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
