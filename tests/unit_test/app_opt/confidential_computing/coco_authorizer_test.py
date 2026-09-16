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

import copy
import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from unittest.mock import patch

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa

from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_exception import NotAuthenticated
from nvflare.app_opt.confidential_computing.cc_authorizer import CCTokenGenerateError
from nvflare.app_opt.confidential_computing.cc_manager import CC_INFO, CC_NAMESPACE, CC_TOKEN, CCManager
from nvflare.app_opt.confidential_computing.coco_authorizer import EAT_PROFILE, TRUST_VECTOR, CoCoAuthorizer


@pytest.fixture(params=["rsa", "ec"])
def material(request):
    signer = ec.generate_private_key(ec.SECP256R1())
    key = (
        rsa.generate_private_key(public_exponent=65537, key_size=2048)
        if request.param == "rsa"
        else ec.generate_private_key(ec.SECP256R1())
    )
    algorithm = jwt.algorithms.RSAAlgorithm if request.param == "rsa" else jwt.algorithms.ECAlgorithm
    jwk = json.loads(algorithm.to_jwk(key.public_key()))
    if request.param == "rsa":
        jwk["alg"] = "RSA-OAEP-256"
    expected = {"init_data": "a" * 64, "image": "registry.example/workload@sha256:" + "b" * 64, "args": ["/start"]}
    claims = {
        "iat": int(time.time()),
        "exp": int(time.time()) + 300,
        "eat_profile": EAT_PROFILE,
        "submods": {
            "cpu0": {
                "ear.trustworthiness-vector": copy.deepcopy(TRUST_VECTOR),
                "ear.veraison.annotated-evidence": {
                    "runtime_data_claims": {"tee-pubkey": jwk},
                    "init_data": expected["init_data"],
                    "init_data_claims": {
                        "agent_policy_claims": {
                            "containers": [
                                {
                                    "OCI": {
                                        "Annotations": {"io.kubernetes.cri.image-name": expected["image"]},
                                        "Process": {"Args": expected["args"]},
                                    }
                                }
                            ]
                        }
                    },
                },
                "ear.trustee.identifiers": {"validated": {"container_images": [expected["image"]]}},
            },
            "gpu0": {"ear.trustworthiness-vector": copy.deepcopy(TRUST_VECTOR)},
        },
    }
    args = {
        "trustee_public_key": signer.public_key()
        .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        .decode(),
        "audience": "nvflare-coco:test",
    }
    client = CoCoAuthorizer(**args, site_name="site-1")
    verifier = CoCoAuthorizer(**args)

    def generate(**overrides):
        configured_client = CoCoAuthorizer(**args, site_name="site-1", **overrides) if overrides else client
        reply = {
            "token": jwt.encode(claims, signer, algorithm="ES256"),
            "tee_keypair": key.private_bytes(
                serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
            ).decode(),
        }
        with patch.object(configured_client, "_get_guest_token", return_value=reply):
            return configured_client.generate()

    return claims, client, verifier, generate, key


def test_valid_proof_and_single_use(material):
    _, _, verifier, generate, _ = material
    token = generate()
    assert "PRIVATE KEY" not in token
    assert verifier.verify(token)
    assert not verifier.verify(token)
    assert verifier.verify(generate())
    with pytest.raises(CCTokenGenerateError):
        verifier.generate()


@contextmanager
def clock_at(now):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.fromtimestamp(now, tz=tz)

    # Exercise both PyJWT's time validation and the authorizer's own age checks.
    with (
        patch("jwt.api_jwt.datetime", Clock),
        patch("nvflare.app_opt.confidential_computing.coco_authorizer.time.time", return_value=now),
    ):
        yield


@pytest.mark.parametrize("lag", [0, 90, 180])
def test_cold_guest_generates_proof_verified_by_synchronized_server(material, lag):
    claims, _, verifier, generate, _ = material
    server_now = int(datetime.now(timezone.utc).timestamp())
    claims.update(iat=server_now, exp=server_now + 300)
    with clock_at(server_now - lag):
        token = generate()
    with clock_at(server_now):
        assert verifier.verify_for_site(token, "site-1")
        assert not verifier.verify_for_site(token, "site-1")


@pytest.mark.parametrize("leeway", [0, 30, 180])
def test_ear_iat_leeway_boundary(material, leeway):
    claims, _, verifier, generate, _ = material
    now = int(time.time())
    claims.update(iat=now + leeway, exp=now + leeway + 300)
    verifier.ear_leeway_seconds = leeway
    with clock_at(now):
        token = generate(ear_leeway_seconds=leeway)
        assert verifier.verify(token)
        # The verifier independently enforces its own allowance.
        if leeway:
            verifier.ear_leeway_seconds = leeway - 1
            assert not verifier.verify(generate(ear_leeway_seconds=leeway))
        claims["iat"] += 1
        with pytest.raises(CCTokenGenerateError):
            generate(ear_leeway_seconds=leeway)


@pytest.mark.parametrize("leeway", [-1, 181, True, False, 1.5, "180", None])
def test_invalid_ear_leeway_rejected(material, leeway):
    _, _, _, generate, _ = material
    with pytest.raises(ValueError, match="ear_leeway_seconds must be 0..180"):
        generate(ear_leeway_seconds=leeway)


@pytest.mark.parametrize("failure", ["expired", "not_yet_valid", "old", "string_iat", "bool_iat", "float_iat"])
def test_ear_leeway_rejects_outside_window_and_invalid_claims(material, failure):
    claims, _, _, generate, _ = material
    now = int(time.time())
    claims.update(iat=now, exp=now + 300)
    if failure == "expired":
        claims.update(iat=now - 200, exp=now - 180)
    elif failure == "not_yet_valid":
        claims["nbf"] = now + 181
    elif failure == "old":
        claims["iat"] = now - 301
    else:
        claims["iat"] = {"string_iat": str(now), "bool_iat": True, "float_iat": float(now)}[failure]
    with clock_at(now), pytest.raises(CCTokenGenerateError):
        generate()


@pytest.mark.parametrize(
    "claim,offset,accepted", [("exp", -179, True), ("exp", -180, False), ("nbf", 180, True), ("nbf", 181, False)]
)
def test_ear_decode_leeway_covers_exp_and_nbf(material, claim, offset, accepted):
    claims, _, verifier, generate, _ = material
    now = int(time.time())
    claims.update(iat=now - 200 if claim == "exp" else now, exp=now + 300)
    claims[claim] = now + offset
    with clock_at(now):
        if accepted:
            assert verifier.verify(generate())
        else:
            with pytest.raises(CCTokenGenerateError):
                generate()


@pytest.mark.parametrize("claim,offset", [("exp", 0), ("nbf", 1)])
def test_zero_ear_leeway_restores_strict_exp_and_nbf(material, claim, offset):
    claims, _, _, generate, _ = material
    now = int(time.time())
    claims.update(iat=now - 10, exp=now + 300)
    claims[claim] = now + offset
    with clock_at(now), pytest.raises(CCTokenGenerateError):
        generate(ear_leeway_seconds=0)


@pytest.mark.parametrize("lifetime", [None, 60, 600])
def test_configurable_proof_lifetime(material, lifetime):
    _, client, _, generate, _ = material
    options = {} if lifetime is None else {"proof_lifetime_seconds": lifetime}
    token = generate(**options)
    proof = jwt.decode(token, options={"verify_signature": False})
    expected = 300 if lifetime is None else lifetime
    assert proof["exp"] - proof["iat"] == expected
    public = client.trustee_key.public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    ).decode()
    verifier = CoCoAuthorizer(public, client.audience, **options)
    assert verifier.proof_lifetime_seconds == expected
    assert verifier.verify_for_site(token, "site-1")
    assert not verifier.verify_for_site(token, "site-1")


@pytest.mark.parametrize("lifetime", [0, -1, True, False, 1.5, "300", None])
def test_invalid_proof_lifetime_rejected(lifetime):
    public = (
        ec.generate_private_key(ec.SECP256R1())
        .public_key()
        .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        .decode()
    )
    with pytest.raises(ValueError, match="proof_lifetime_seconds must be a positive integer"):
        CoCoAuthorizer(public, "test", proof_lifetime_seconds=lifetime)


@pytest.mark.parametrize("failure", ["too_long", "expired", "future"])
def test_configured_verifier_preserves_time_checks(material, failure):
    _, client, _, generate, key = material
    public = client.trustee_key.public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    ).decode()
    verifier = CoCoAuthorizer(public, client.audience, proof_lifetime_seconds=60)
    proof = jwt.decode(generate(proof_lifetime_seconds=60), options={"verify_signature": False})
    if failure == "too_long":
        proof["exp"] = proof["iat"] + 61
    elif failure == "expired":
        proof["iat"] -= 61
        proof["exp"] -= 61
    else:
        proof["iat"] += 30
        proof["exp"] += 30
    token = jwt.encode(proof, key, algorithm=CoCoAuthorizer._algorithm(key))
    assert not verifier.verify(token)


def test_longer_proof_does_not_extend_ear_freshness(material):
    claims, _, _, generate, _ = material
    claims["iat"] -= 301
    with pytest.raises(CCTokenGenerateError):
        generate(proof_lifetime_seconds=600)


def test_peer_bound_proof(material):
    _, _, verifier, generate, _ = material
    token = generate()
    assert not verifier.verify_for_site(token, "site-2")
    assert not verifier.verify_for_site(token, "")
    assert verifier.verify_for_site(token, "site-1")
    assert not verifier.verify_for_site(token, "site-1")


def test_registration_rejects_other_clients_real_signed_proof(material):
    _, _, verifier, generate, _ = material
    token = generate()
    manager = CCManager([], ["coco"], cc_enabled_sites=["site-1", "site-2"])
    manager.cc_verifiers = {verifier.get_namespace(): verifier}
    peer = FLContext()
    context = FLContext()
    context.set_peer_context(peer)
    for site in ("site-2", "site-1"):
        context.set_prop(FLContextKey.CLIENT_NAME, site)
        peer.set_prop(CC_INFO, {site: [{CC_NAMESPACE: verifier.get_namespace(), CC_TOKEN: token}]})
        if site == "site-2":
            with pytest.raises(NotAuthenticated):
                manager._validate_client_tokens(context)
        else:
            manager._validate_client_tokens(context)


@pytest.mark.parametrize(
    "failure", ["gpu_missing", "gpu_failed", "cpu_failed", "extra_submod", "old", "expired", "wrong_key"]
)
def test_bad_ear_rejected(material, failure):
    claims, client, _, generate, _ = material
    if failure == "gpu_missing":
        claims["submods"].pop("gpu0")
    elif failure == "gpu_failed":
        claims["submods"]["gpu0"]["ear.trustworthiness-vector"]["hardware"] = 97
    elif failure == "cpu_failed":
        claims["submods"]["cpu0"]["ear.trustworthiness-vector"]["configuration"] = 33
    elif failure == "extra_submod":
        claims["submods"]["cpu1"] = claims["submods"]["cpu0"]
    elif failure == "old":
        claims["iat"] -= 301
    elif failure == "expired":
        claims["iat"] -= 400
        claims["exp"] -= 301
    else:
        client.trustee_key = ec.generate_private_key(ec.SECP256R1()).public_key()
    with pytest.raises(CCTokenGenerateError, match="Unable to generate"):
        generate()


@pytest.mark.parametrize("failure", ["subject", "audience", "signature", "time", "ear_signature"])
def test_invalid_proof_rejected(material, failure):
    _, _, verifier, generate, key = material
    token = generate()
    if failure == "audience":
        verifier.audience = "nvflare-coco:another-project"
    else:
        proof = jwt.decode(token, options={"verify_signature": False})
        if failure == "signature":
            key = ec.generate_private_key(ec.SECP256R1())
        elif failure == "subject":
            proof["sub"] = ""
        elif failure == "time":
            proof["iat"] -= verifier.proof_lifetime_seconds + 1
            proof["exp"] -= verifier.proof_lifetime_seconds + 1
        else:
            claims = jwt.decode(proof["ear"], options={"verify_signature": False})
            proof["ear"] = jwt.encode(claims, ec.generate_private_key(ec.SECP256R1()), algorithm="ES256")
        token = jwt.encode(proof, key, algorithm=CoCoAuthorizer._algorithm(key))
    assert not verifier.verify(token)


def test_guest_api_redacted_failure(material):
    _, client, _, _, _ = material
    with patch.object(client, "_get_guest_token", side_effect=ValueError("PRIVATE KEY SECRET")):
        with pytest.raises(CCTokenGenerateError) as error:
            client.generate()
    assert "SECRET" not in str(error.value)


def test_guest_api_key_mismatch(material):
    _, client, _, generate, _ = material
    ear = jwt.decode(generate(), options={"verify_signature": False})["ear"]
    wrong_key = (
        ec.generate_private_key(ec.SECP256R1())
        .private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
        .decode()
    )
    with patch.object(client, "_get_guest_token", return_value={"token": ear, "tee_keypair": wrong_key}):
        with pytest.raises(CCTokenGenerateError):
            client.generate()


def test_guest_api_disables_proxies_and_redirects(material):
    _, client, _, _, _ = material
    with patch("nvflare.app_opt.confidential_computing.coco_authorizer.requests.Session") as factory:
        session = factory.return_value.__enter__.return_value
        response = session.get.return_value.__enter__.return_value
        response.status_code = 200
        response.iter_content.return_value = [b'{"token": "fixture"}']
        assert client._get_guest_token() == {"token": "fixture"}
        assert session.trust_env is False
        assert session.get.call_args.kwargs["allow_redirects"] is False
        assert session.get.call_args.kwargs["params"] == {"token_type": "kbs"}
        response.status_code = 302
        with pytest.raises(ValueError):
            client._get_guest_token()


@pytest.mark.parametrize(
    "url",
    [
        "http://host/aa/token",
        "https://127.0.0.1/aa/token",
        "http://user@127.0.0.1/aa/token",
        "http://127.0.0.1/aa/token?token_type=other",
    ],
)
def test_nonlocal_or_ambiguous_endpoint_rejected(material, url):
    _, client, _, _, _ = material
    public = client.trustee_key.public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    ).decode()
    with pytest.raises(ValueError):
        CoCoAuthorizer(public, "test", token_url=url)
