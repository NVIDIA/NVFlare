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
import logging
import math
import random
import secrets
import threading
import time
from urllib.parse import urlsplit

import jwt
import requests
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa

from .cc_authorizer import CCAuthorizer, CCTokenGenerateError

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


class _TemporaryTokenError(CCTokenGenerateError):
    """A guest API transport failure that may safely be retried."""


class CoCoAuthorizer(CCAuthorizer):
    def __init__(
        self,
        trustee_public_key,
        audience,
        site_name=None,
        token_url="http://127.0.0.1:8006/aa/token",
        max_token_age_seconds=300,
        proof_lifetime_seconds=300,
        ear_leeway_seconds=180,
        retry_max_attempts=10,
        retry_initial_delay=1.0,
        retry_max_delay=15.0,
        retry_backoff_multiplier=2.0,
        retry_jitter_ratio=0.5,
    ):
        """Configure EAR freshness and the generated/accepted outer proof lifetime separately.

        proof_lifetime_seconds must be a positive integer. Verifiers reject proofs
        older than this limit or with a declared lifetime exceeding it. EAR expiry
        and max_token_age_seconds are enforced independently.
        ear_leeway_seconds (0..180) applies PyJWT clock-skew tolerance to EAR
        iat, exp and nbf checks. The maximum EAR age and outer proof checks
        remain independent of this allowance.
        retry_max_attempts (1..100) includes the first attempt and is used only
        by generate_with_retry(), never by single-attempt generate().
        Retry delays are in seconds. Each wait is sampled between
        delay * (1 - retry_jitter_ratio) and delay; delay grows by
        retry_backoff_multiplier up to retry_max_delay.
        """
        self.trustee_key = serialization.load_pem_public_key(trustee_public_key.encode())
        if not isinstance(self.trustee_key, ec.EllipticCurvePublicKey) or not isinstance(
            self.trustee_key.curve, ec.SECP256R1
        ):
            raise ValueError("Trustee EAR signing key must be a pinned P-256 public key")
        if not isinstance(audience, str) or not audience:
            raise ValueError("A project-specific audience is required")
        if type(max_token_age_seconds) is not int or not 1 <= max_token_age_seconds <= 300:
            raise ValueError("max_token_age_seconds must be 1..300")
        if type(proof_lifetime_seconds) is not int or proof_lifetime_seconds <= 0:
            raise ValueError("proof_lifetime_seconds must be a positive integer")
        if type(ear_leeway_seconds) is not int or not 0 <= ear_leeway_seconds <= 180:
            raise ValueError("ear_leeway_seconds must be 0..180")
        if type(retry_max_attempts) is not int or not 1 <= retry_max_attempts <= 100:
            raise ValueError("retry_max_attempts must be 1..100")
        for name, value in (
            ("retry_initial_delay", retry_initial_delay),
            ("retry_max_delay", retry_max_delay),
            ("retry_backoff_multiplier", retry_backoff_multiplier),
            ("retry_jitter_ratio", retry_jitter_ratio),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if retry_initial_delay <= 0 or retry_max_delay < retry_initial_delay:
            raise ValueError("Require 0 < retry_initial_delay <= retry_max_delay")
        if retry_backoff_multiplier < 1:
            raise ValueError("retry_backoff_multiplier must be >= 1")
        if not 0 <= retry_jitter_ratio <= 1:
            raise ValueError("retry_jitter_ratio must be 0..1")
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
        self.token_url = token_url
        self.max_age = max_token_age_seconds
        self.proof_lifetime_seconds = proof_lifetime_seconds
        self.ear_leeway_seconds = ear_leeway_seconds
        self.seen = {}
        self.lock = threading.Lock()
        self.retry_max_attempts = retry_max_attempts
        self.retry_initial_delay = retry_initial_delay
        self.retry_max_delay = retry_max_delay
        self.retry_backoff_multiplier = retry_backoff_multiplier
        self.retry_jitter_ratio = retry_jitter_ratio
        self._generation_lock = threading.Lock()
        self._generation_context = threading.local()

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
        # Match the tested EAR clock-skew allowance for iat, exp and nbf.
        # This leeway is not applied to the outer proof's decode.
        claims = jwt.decode(
            token,
            self.trustee_key,
            algorithms=["ES256"],
            options={"require": ["iat", "exp"], "verify_aud": False},
            leeway=self.ear_leeway_seconds,
        )
        now = time.time()
        if (
            type(claims["iat"]) is not int
            or type(claims["exp"]) is not int
            or not -self.ear_leeway_seconds <= now - claims["iat"] <= self.max_age
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
        remaining = getattr(self._generation_context, "deadline", float("inf")) - time.monotonic()
        if remaining <= 0:
            raise CCTokenGenerateError("Token generation deadline exhausted")
        with requests.Session() as session:
            session.trust_env = False  # never send the private key via a proxy
            with session.get(
                self.token_url,
                params={"token_type": "kbs"},
                timeout=(min(3, remaining), min(10, remaining)),
                allow_redirects=False,
                stream=True,
            ) as response:
                if response.status_code in (429, 502, 503, 504):
                    raise _TemporaryTokenError("Guest token API temporarily unavailable")
                if response.status_code != 200:
                    raise ValueError("Guest token API failed")
                chunks, size = [], 0
                for chunk in response.iter_content(65536):
                    if time.monotonic() >= getattr(self._generation_context, "deadline", float("inf")):
                        raise CCTokenGenerateError("Token generation deadline exhausted")
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
                    "exp": now + self.proof_lifetime_seconds,
                    "jti": secrets.token_hex(24),
                },
                private,
                algorithm=self._algorithm(private),
            )
        except (_TemporaryTokenError, requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            raise _TemporaryTokenError("Guest token API temporarily unavailable") from None
        except Exception:
            raise CCTokenGenerateError("Unable to generate a verified CoCo attestation proof") from None

    def generate_with_retry(self, timeout, cancel_event):
        """Retry transport failures without extending token validity.

        A single guarded worker bounds the caller's wait even if HTTP stalls.
        Python cannot forcibly terminate an in-flight request: its worker keeps
        the lock until it exits, preventing abandoned workers from accumulating.
        Direct generate() remains a single-attempt API.
        """
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("Token generation timeout must be finite and positive")
        deadline = time.monotonic() + timeout
        stopped, done = threading.Event(), threading.Event()
        outcome = {}

        def check_budget():
            if cancel_event.is_set() or stopped.is_set():
                raise CCTokenGenerateError("Token generation cancelled")
            if time.monotonic() >= deadline:
                raise CCTokenGenerateError("Token generation deadline exhausted")

        def wait(delay):
            until = min(deadline, time.monotonic() + delay)
            while time.monotonic() < until:
                check_budget()
                stopped.wait(min(0.05, max(0, until - time.monotonic())))
            check_budget()

        def run():
            self._generation_context.deadline = deadline
            try:
                delay = self.retry_initial_delay
                for attempt in range(1, self.retry_max_attempts + 1):
                    check_budget()
                    try:
                        token = self.generate()
                        check_budget()
                        outcome["token"] = token
                        return
                    except _TemporaryTokenError:
                        check_budget()
                        if attempt == self.retry_max_attempts:
                            raise CCTokenGenerateError("Token generation retry attempts exhausted") from None
                        logging.getLogger(__name__).warning(
                            "Temporary guest token API failure; retrying after attempt %d", attempt
                        )
                        wait(random.uniform(delay * (1 - self.retry_jitter_ratio), delay))
                        delay = min(self.retry_max_delay, delay * self.retry_backoff_multiplier)
            except CCTokenGenerateError as error:
                outcome["error"] = CCTokenGenerateError(str(error))
            except Exception:
                outcome["error"] = CCTokenGenerateError("Unable to generate a verified CoCo attestation proof")
            finally:
                self._generation_lock.release()
                done.set()

        # Waiting for another request also consumes this request's budget.
        while True:
            check_budget()
            if self._generation_lock.acquire(blocking=False):
                break
            wait(0.05)
        worker = threading.Thread(target=run, daemon=True, name="CoCo-TokenGeneration")
        try:
            worker.start()
        except Exception:
            self._generation_lock.release()
            raise CCTokenGenerateError("Unable to start token generation") from None
        try:
            while not done.wait(min(0.05, max(0, deadline - time.monotonic()))):
                check_budget()
            check_budget()
            if "error" in outcome:
                raise outcome["error"]
            return outcome["token"]
        finally:
            stopped.set()

    def verify(self, token):
        """Verify proof validity only; use verify_for_site at an FL peer boundary."""
        return self._verify(token)

    def verify_for_site(self, token, site_name):
        """Bind the signed subject to the independently authenticated FL peer."""
        if not isinstance(site_name, str) or not site_name:
            return False
        return self._verify(token, expected_site=site_name)

    def _verify(self, token, expected_site=None):
        try:
            if not isinstance(token, str) or len(token) > MAX_TOKEN_BYTES:
                return False
            untrusted = jwt.decode(token, options={"verify_signature": False})
            _, public = self._ear(untrusted["ear"])
            proof = jwt.decode(
                token,
                public,
                algorithms=[self._algorithm(public)],
                audience=self.audience,
                options={"require": ["sub", "iat", "exp", "jti", "aud"]},
            )
            now = time.time()
            if (
                not isinstance(proof["sub"], str)
                or not proof["sub"]
                or (expected_site is not None and proof["sub"] != expected_site)
                or type(proof["iat"]) is not int
                or type(proof["exp"]) is not int
                or not 0 <= now - proof["iat"] <= self.proof_lifetime_seconds
                or not 0 < proof["exp"] - proof["iat"] <= self.proof_lifetime_seconds
                or not isinstance(proof["jti"], str)
                or len(proof["jti"]) != 48
            ):
                raise ValueError("Invalid proof identity or freshness")
            with self.lock:
                self.seen = {k: expiry for k, expiry in self.seen.items() if expiry > now}
                key = (proof["sub"], proof["jti"])
                if key in self.seen or len(self.seen) >= 10000:
                    raise ValueError("Replayed proof or replay cache full")
                self.seen[key] = proof["exp"]
            return True
        except Exception:
            return False
