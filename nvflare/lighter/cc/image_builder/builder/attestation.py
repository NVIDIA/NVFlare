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

"""Fresh KBS attestation and resource authorization; tokens and keys stay in RAM."""

import base64
import contextlib
import json
import os
import time
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, padding, rsa, utils

from .common import BuildError, memory_file, require, resource_path, run

ATTESTATION_BUDGET_SECONDS = 60


def unb64url(value):
    require(
        isinstance(value, str) and value and all(c.isalnum() or c in "-_" for c in value) and value.isascii(),
        "Invalid token encoding",
    )
    result = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    require(base64.urlsafe_b64encode(result).decode().rstrip("=") == value, "Noncanonical token encoding")
    return result


def validate_token(token, config, digest, *, now=None):
    """Pin the signing key/algorithm on the verified root, never trust JWT headers as keys."""
    try:
        header64, payload64, signature64 = token.decode("ascii").split(".")
        header = json.loads(unb64url(header64))
        require(header.get("alg") == config["token_algorithm"] and not header.get("crit"), "Unexpected token algorithm")
        key = serialization.load_pem_public_key(Path(config["as_public_key"]).read_bytes())
        signed = (header64 + "." + payload64).encode()
        signature = unb64url(signature64)
        if header["alg"] == "RS256":
            require(isinstance(key, rsa.RSAPublicKey) and key.key_size >= 2048, "Invalid RSA verification key")
            key.verify(signature, signed, padding.PKCS1v15(), hashes.SHA256())
        elif header["alg"] == "ES256":
            require(
                isinstance(key, ec.EllipticCurvePublicKey)
                and isinstance(key.curve, ec.SECP256R1)
                and len(signature) == 64,
                "Invalid EC signature",
            )
            der = utils.encode_dss_signature(
                int.from_bytes(signature[:32], "big"), int.from_bytes(signature[32:], "big")
            )
            key.verify(der, signed, ec.ECDSA(hashes.SHA256()))
        elif header["alg"] == "EdDSA":
            require(isinstance(key, ed25519.Ed25519PublicKey), "Invalid Ed25519 verification key")
            key.verify(signature, signed)
        else:
            raise BuildError("Unsupported token algorithm")
        claims = json.loads(unb64url(payload64))
        now = time.time() if now is None else now
        require(type(claims.get("exp")) in (int, float) and now < claims["exp"], "Expired appraisal")
        require(
            type(claims.get("iat")) in (int, float) and now - 300 <= claims["iat"] <= now + 30, "Appraisal is not fresh"
        )
        require(claims.get("nbf", 0) <= now + 30, "Appraisal is not yet valid")
        require(claims.get("iss") == config["token_issuer"], "Wrong appraisal issuer")
        cpu = claims["submods"]["cpu0"]
        require(cpu["ear.appraisal-policy-id"] == config["attestation_policy_id"], "Wrong appraisal policy")
        require(cpu["ear.status"] == "affirming", "CPU appraisal denied")
        tv = cpu["ear.trustworthiness-vector"]
        require(
            all(
                type(tv.get(k)) is int and tv[k] == v
                for k, v in {"executables": 3, "hardware": 2, "configuration": 2}.items()
            ),
            "CPU trust vector denied",
        )
        evidence = cpu["ear.veraison.annotated-evidence"]
        if config["platform"] == "intel_tdx":
            require(evidence["init_data"] == (digest + bytes(16)).hex(), "Appraised TDX binding mismatch")
            require(evidence["tdx"]["td_attributes"]["debug"] is False, "TDX debug is enabled")
        else:
            require(evidence["init_data"] == base64.b64encode(digest).decode(), "Appraised SNP binding mismatch")
            require(
                evidence["snp"]["policy_debug_allowed"] is False and evidence["snp"]["policy_migrate_ma"] is False,
                "SNP debug or migration is enabled",
            )
        if config.get("gpu") == "nvidia_cc":
            from .gpu_policy import validate_submods

            validate_submods(claims["submods"], config["gpu_count"], config["attestation_policy_id"])
        return claims
    except (ValueError, KeyError, TypeError, InvalidSignature, UnicodeError):
        raise BuildError("Invalid or unauthenticated KBS appraisal") from None


@contextlib.contextmanager
def authorized_key(config, digest, *, budget=None):
    maximum = 240 if config.get("gpu") == "nvidia_cc" else ATTESTATION_BUDGET_SECONDS
    budget = maximum if budget is None else budget
    require(type(budget) in (int, float) and 0 < budget <= maximum, "Invalid attestation budget")
    deadline = time.monotonic() + budget

    def remaining():
        value = deadline - time.monotonic()
        require(value > 0, "Attestation exceeded its total time budget")
        return value

    # kbs-client binds its fresh challenge and this ephemeral public key to the
    # report. The same private key must decrypt the authorized resource response.
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    # The pinned Rust client accepts RSA keys in PKCS#1 (EC uses PKCS#8).
    pem = private.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL, serialization.NoEncryption()
    )
    command = [config["kbs_client"], "--url", config["kbs_url"], "--cert-file", config["kbs_cert"]]
    environment = dict(os.environ, RUST_LOG="off")
    with memory_file(pem) as tee_key:
        token = run(
            command + ["attest", "--tee-key-file", f"/proc/self/fd/{tee_key}"],
            pass_fds=(tee_key,),
            timeout=remaining(),
            env=environment,
        ).strip()
        validate_token(token, config, digest)
        with memory_file(token) as token_fd:
            path = resource_path(config["build_id"], config["platform"], digest)
            encoded = run(
                command
                + [
                    "get-resource",
                    "--path",
                    path,
                    "--tee-key-file",
                    f"/proc/self/fd/{tee_key}",
                    "--attestation-token",
                    f"/proc/self/fd/{token_fd}",
                ],
                pass_fds=(tee_key, token_fd),
                timeout=remaining(),
                env=environment,
            ).strip()
            try:
                key = base64.b64decode(encoded, validate=True)
            except ValueError:
                raise BuildError("Invalid resource encoding") from None
            require(len(key) == 64 and base64.b64encode(key) == encoded, "KBS returned an invalid vault secret")
            with memory_file(key) as key_fd:
                yield key_fd
