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

"""Trustee EAR verification and guest-held TEE-key proof of possession.

The REST response includes a private key. It is consumed in memory only and
must never be forwarded as a CC token, persisted, or included in exceptions.
"""

import json
import secrets
import threading
import time
from urllib.parse import urlsplit

import jwt
import requests
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa

from .cc_authorizer import CCAuthorizer, CCTokenGenerateError, CCTokenVerifyError

COCO_NAMESPACE = "x-trustee-coco"
TRUST_VECTOR = {
    "executables": 3,
    "hardware": 2,
    "configuration": 3,
    "file-system": 0,
    "instance-identity": 0,
    "runtime-opaque": 0,
    "storage-opaque": 0,
    "sourced-data": 0,
}
EAT_PROFILE = "tag:github.com,2024:confidential-containers/Trustee"
MAX_TOKEN_BYTES = 2 * 1024 * 1024


class CoCoAuthorizer(CCAuthorizer):
    def __init__(
        self,
        trustee_public_key,
        audience,
        site_name=None,
        expected_workloads=None,
        token_url="http://127.0.0.1:8006/aa/token",
        max_token_age_seconds=300,
    ):
        self.trustee_key = serialization.load_pem_public_key(trustee_public_key.encode())
        if not isinstance(self.trustee_key, ec.EllipticCurvePublicKey) or not isinstance(
            self.trustee_key.curve, ec.SECP256R1
        ):
            raise ValueError("Trustee EAR signing key must be a pinned P-256 public key")
        if not isinstance(audience, str) or not audience:
            raise ValueError("A project-specific audience is required")
        if type(max_token_age_seconds) is not int or not 1 <= max_token_age_seconds <= 300:
            raise ValueError("max_token_age_seconds must be 1..300")
        url = urlsplit(token_url)
        if (
            url.scheme != "http"
            or url.hostname != "127.0.0.1"
            or url.path != "/aa/token"
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError("token_url must address the guest-local 127.0.0.1 /aa/token API")
        self.audience = audience
        self.site_name = site_name
        self.expected_workloads = expected_workloads or {}
        self.token_url = token_url
        self.max_age = max_token_age_seconds
        self.seen = {}
        self.lock = threading.Lock()

    def get_namespace(self):
        return COCO_NAMESPACE

    @staticmethod
    def _algorithm(key):
        if isinstance(key, (rsa.RSAPrivateKey, rsa.RSAPublicKey)) and key.key_size >= 2048:
            return "RS256"
        if isinstance(key, (ec.EllipticCurvePrivateKey, ec.EllipticCurvePublicKey)) and isinstance(
            key.curve, ec.SECP256R1
        ):
            return "ES256"
        raise ValueError("Unsupported TEE proof key")

    def _ear(self, token):
        if not isinstance(token, str) or len(token) > MAX_TOKEN_BYTES:
            raise ValueError("Invalid EAR size")
        # Never trust jku/x5u/x5c from the JWT header, nor the service TLS cert
        # as a substitute for the independently authenticated AS signing key.
        claims = jwt.decode(
            token, self.trustee_key, algorithms=["ES256"], options={"require": ["iat", "exp"], "verify_aud": False}
        )
        now = time.time()
        if (
            type(claims["iat"]) is not int
            or type(claims["exp"]) is not int
            or not 0 <= now - claims["iat"] <= self.max_age
            or claims["exp"] <= claims["iat"]
            or claims.get("eat_profile") != EAT_PROFILE
        ):
            raise ValueError("Invalid EAR freshness or profile")
        submods = claims.get("submods", {})
        if set(submods) != {"cpu0", "gpu0"}:
            raise ValueError("Both CPU and GPU appraisals are required")
        for submod in submods.values():
            if json.dumps(submod.get("ear.trustworthiness-vector"), sort_keys=True) != json.dumps(
                TRUST_VECTOR, sort_keys=True
            ):
                raise ValueError("CPU/GPU appraisal failed")
        evidence = submods["cpu0"]["ear.veraison.annotated-evidence"]
        jwk = evidence["runtime_data_claims"]["tee-pubkey"]
        # KBS encryption JWKs may advertise RSA-OAEP. Use only their public
        # key coordinates, selecting a signing algorithm locally.
        if jwk.get("kty") == "RSA":
            key = jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps({k: jwk[k] for k in ("kty", "n", "e")}))
        elif jwk.get("kty") == "EC":
            key = jwt.algorithms.ECAlgorithm.from_jwk(json.dumps({k: jwk[k] for k in ("kty", "crv", "x", "y")}))
        else:
            raise ValueError("Unsupported attested TEE key")
        self._algorithm(key)
        return claims, key

    def _get_guest_token(self):
        with requests.Session() as session:
            session.trust_env = False  # never send the private key via a proxy
            with session.get(
                self.token_url, params={"token_type": "kbs"}, timeout=(3, 10), allow_redirects=False, stream=True
            ) as response:
                if response.status_code != 200:
                    raise ValueError("Guest token API failed")
                chunks, size = [], 0
                for chunk in response.iter_content(65536):
                    size += len(chunk)
                    if size > MAX_TOKEN_BYTES:
                        raise ValueError("Guest token API response too large")
                    chunks.append(chunk)
                return json.loads(b"".join(chunks))

    def generate(self):
        if not self.site_name:
            raise CCTokenGenerateError("Verifier-only CoCo authorizer cannot generate tokens")
        try:
            reply = self._get_guest_token()
            ear = reply["token"]
            _, public = self._ear(ear)
            private = serialization.load_pem_private_key(reply["tee_keypair"].encode(), password=None)
            if private.public_key().public_numbers() != public.public_numbers():
                raise ValueError("TEE private key does not match attested public key")
            now = int(time.time())
            # The private key never leaves this process. Proof lifetime is
            # independent of EAR lifetime (AA may cache EAR until expiration).
            return jwt.encode(
                {
                    "ear": ear,
                    "sub": self.site_name,
                    "aud": self.audience,
                    "iat": now,
                    "exp": now + 60,
                    "jti": secrets.token_hex(24),
                },
                private,
                algorithm=self._algorithm(private),
            )
        except Exception:
            raise CCTokenGenerateError("Unable to generate a verified CoCo attestation proof") from None

    def verify(self, token):
        # Use verify_for_site at the protocol boundary to bind the authenticated
        # FL peer, not merely a self-declared subject.
        raise CCTokenVerifyError("CoCo verification requires the expected FL site identity")

    def verify_for_site(self, token, site):
        try:
            if site not in self.expected_workloads or not isinstance(token, str) or len(token) > MAX_TOKEN_BYTES:
                raise ValueError("Unapproved site or invalid proof size")
            untrusted = jwt.decode(token, options={"verify_signature": False})
            claims, public = self._ear(untrusted["ear"])
            proof = jwt.decode(
                token,
                public,
                algorithms=[self._algorithm(public)],
                audience=self.audience,
                options={"require": ["sub", "iat", "exp", "jti", "aud"]},
            )
            now = time.time()
            if (
                proof["sub"] != site
                or type(proof["iat"]) is not int
                or type(proof["exp"]) is not int
                or not 0 <= now - proof["iat"] <= 60
                or not 0 < proof["exp"] - proof["iat"] <= 60
                or not isinstance(proof["jti"], str)
                or len(proof["jti"]) != 48
            ):
                raise ValueError("Invalid proof identity or freshness")
            expected = self.expected_workloads[site]
            cpu = claims["submods"]["cpu0"]
            evidence = cpu["ear.veraison.annotated-evidence"]
            if evidence["init_data"] != expected["init_data"]:
                raise ValueError("Unapproved measured workload")
            image, args = expected["image"], expected["args"]
            if image not in cpu["ear.trustee.identifiers"]["validated"]["container_images"]:
                raise ValueError("Image signature identity is not validated")
            containers = evidence["init_data_claims"]["agent_policy_claims"]["containers"]
            if not any(
                c.get("OCI", {}).get("Annotations", {}).get("io.kubernetes.cri.image-name") == image
                and c.get("OCI", {}).get("Process", {}).get("Args") == args
                for c in containers
            ):
                raise ValueError("Workload command or image mismatch")
            with self.lock:
                self.seen = {k: expiry for k, expiry in self.seen.items() if expiry > now}
                key = (site, proof["jti"])
                if key in self.seen or len(self.seen) >= 10000:
                    raise ValueError("Replayed proof or replay cache full")
                self.seen[key] = proof["exp"]
            return True
        except Exception:
            raise CCTokenVerifyError("CoCo attestation proof was rejected") from None
