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
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import jwt
import pytest
import requests
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


def guest_reply(material):
    _, _, _, generate, key = material
    return {
        "token": jwt.decode(generate(), options={"verify_signature": False})["ear"],
        "tee_keypair": key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
        ).decode(),
    }


@pytest.mark.parametrize("failure", [requests.exceptions.ConnectionError, requests.exceptions.Timeout])
def test_retry_transport_failure_then_verify_fresh_proofs(material, failure, caplog):
    reply = guest_reply(material)
    _, client, verifier, _, _ = material
    with patch.object(client, "_get_guest_token", side_effect=[failure("PRIVATE SECRET"), reply, reply]) as get:
        with patch("nvflare.app_opt.confidential_computing.coco_authorizer.random.uniform", return_value=0):
            first = client.generate_with_retry(2, threading.Event())
            second = client.generate_with_retry(2, threading.Event())
    assert get.call_count == 3
    assert first != second
    assert verifier.verify_for_site(first, "site-1")
    assert verifier.verify_for_site(second, "site-1")
    assert not verifier.verify(first)
    assert "PRIVATE SECRET" not in caplog.text


@pytest.mark.parametrize("status", [429, 502, 503, 504, 400, 401, 403, 404, 500, 302])
def test_retry_http_status_allowlist(material, status):
    reply = guest_reply(material)
    _, client, verifier, _, _ = material
    with patch("nvflare.app_opt.confidential_computing.coco_authorizer.requests.Session") as factory:
        sessions = []
        for code in (status, 200):
            session = MagicMock()
            response = session.__enter__.return_value.get.return_value.__enter__.return_value
            response.status_code = code
            response.iter_content.return_value = [json.dumps(reply).encode()]
            sessions.append(session)
        factory.side_effect = sessions
        with patch("nvflare.app_opt.confidential_computing.coco_authorizer.random.uniform", return_value=0):
            if status in (429, 502, 503, 504):
                assert verifier.verify(client.generate_with_retry(2, threading.Event()))
                assert factory.call_count == 2
            else:
                with pytest.raises(CCTokenGenerateError):
                    client.generate_with_retry(2, threading.Event())
                assert factory.call_count == 1


@pytest.mark.parametrize("failure", ["signature", "appraisal", "key", "malformed"])
def test_retry_does_not_retry_invalid_attestation(material, failure):
    reply = guest_reply(material)
    _, client, _, _, _ = material
    if failure in ("signature", "appraisal"):
        claims = jwt.decode(reply["token"], options={"verify_signature": False})
        if failure == "appraisal":
            claims["submods"]["gpu0"]["ear.trustworthiness-vector"]["hardware"] = 0
            # Exercise appraisal rejection with a trusted signer for this test.
            signer = ec.generate_private_key(ec.SECP256R1())
            client.trustee_key = signer.public_key()
        else:
            signer = ec.generate_private_key(ec.SECP256R1())
        reply["token"] = jwt.encode(claims, signer, algorithm="ES256")
    elif failure == "key":
        reply["tee_keypair"] = "PRIVATE SECRET"
    else:
        reply = {}
    with patch.object(client, "_get_guest_token", return_value=reply) as get:
        with pytest.raises(CCTokenGenerateError) as error:
            client.generate_with_retry(2, threading.Event())
    assert get.call_count == 1
    assert "SECRET" not in str(error.value)


def test_retry_attempt_limit(material):
    _, client, _, _, _ = material
    with patch.object(client, "_get_guest_token", side_effect=requests.exceptions.Timeout("SECRET")) as get:
        with patch("nvflare.app_opt.confidential_computing.coco_authorizer.random.uniform", return_value=0) as jitter:
            with pytest.raises(CCTokenGenerateError, match="retry attempts exhausted"):
                client.generate_with_retry(2, threading.Event())
    assert get.call_count == 10
    assert [call.args for call in jitter.call_args_list] == [(0.5, 1), (1, 2), (2, 4), (4, 8)] + [(7.5, 15)] * 5


@pytest.mark.parametrize("ratio", [0, 0.25, 1])
def test_custom_retry_backoff(material, ratio):
    _, client, _, _, _ = material
    public = client.trustee_key.public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    ).decode()
    client = CoCoAuthorizer(
        public,
        "test",
        site_name="site-1",
        retry_max_attempts=5,
        retry_initial_delay=2,
        retry_max_delay=10,
        retry_backoff_multiplier=3,
        retry_jitter_ratio=ratio,
    )
    with patch.object(client, "_get_guest_token", side_effect=requests.exceptions.Timeout()) as get:
        with patch("nvflare.app_opt.confidential_computing.coco_authorizer.random.uniform", return_value=0) as jitter:
            with pytest.raises(CCTokenGenerateError, match="retry attempts exhausted"):
                client.generate_with_retry(2, threading.Event())
    assert get.call_count == 5
    assert [call.args for call in jitter.call_args_list] == [(delay * (1 - ratio), delay) for delay in (2, 6, 10, 10)]


@pytest.mark.parametrize(
    "name",
    ["retry_initial_delay", "retry_max_delay", "retry_backoff_multiplier", "retry_jitter_ratio"],
)
@pytest.mark.parametrize("value", [True, "1", None, float("nan"), float("inf")])
def test_invalid_retry_backoff_type(material, name, value):
    _, _, _, generate, _ = material
    with pytest.raises(ValueError, match=name):
        generate(**{name: value})


@pytest.mark.parametrize(
    "options",
    [
        {"retry_initial_delay": 0},
        {"retry_initial_delay": -1},
        {"retry_max_delay": 0},
        {"retry_initial_delay": 20, "retry_max_delay": 15},
        {"retry_backoff_multiplier": 0.5},
        {"retry_jitter_ratio": -0.1},
        {"retry_jitter_ratio": 1.1},
    ],
)
def test_invalid_retry_backoff_range(material, options):
    _, _, _, generate, _ = material
    with pytest.raises(ValueError, match="retry_"):
        generate(**options)


def test_retry_cancellation_during_backoff(material):
    _, client, _, _, _ = material
    cancel = threading.Event()

    def cancel_during_backoff(*args):
        cancel.set()
        return 1

    with patch.object(client, "_get_guest_token", side_effect=requests.exceptions.Timeout()) as get:
        with patch(
            "nvflare.app_opt.confidential_computing.coco_authorizer.random.uniform", side_effect=cancel_during_backoff
        ):
            with pytest.raises(CCTokenGenerateError, match="cancelled"):
                client.generate_with_retry(2, cancel)
    assert get.call_count == 1


def test_retry_deadline_bounds_stalled_request_and_prevents_overlapping_workers(material):
    reply = guest_reply(material)
    _, client, verifier, _, _ = material
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    def stalled_request():
        started.set()
        try:
            assert release.wait(5)
            return reply
        finally:
            finished.set()

    with patch.object(client, "_get_guest_token", side_effect=stalled_request) as get:
        try:
            start = time.monotonic()
            with pytest.raises(CCTokenGenerateError, match="deadline exhausted"):
                client.generate_with_retry(0.1, threading.Event())
            assert time.monotonic() - start < 1
            assert started.is_set()
            with pytest.raises(CCTokenGenerateError, match="deadline exhausted"):
                client.generate_with_retry(0.1, threading.Event())
            assert get.call_count == 1
        finally:
            release.set()
            assert finished.wait(2)
            assert client._generation_lock.acquire(timeout=2)
            client._generation_lock.release()
    # The late result is discarded, and a later request can recover normally.
    with patch.object(client, "_get_guest_token", return_value=reply) as get:
        assert verifier.verify(client.generate_with_retry(2, threading.Event()))
        get.assert_called_once()


def test_retry_precancelled_request_does_not_call_guest_api(material):
    _, client, _, _, _ = material
    cancel = threading.Event()
    cancel.set()
    with patch.object(client, "_get_guest_token") as get:
        with pytest.raises(CCTokenGenerateError, match="cancelled"):
            client.generate_with_retry(2, cancel)
        get.assert_not_called()


def test_retry_discards_success_after_cancellation(material):
    reply = guest_reply(material)
    _, client, _, _, _ = material
    cancel = threading.Event()

    def cancelled_response():
        cancel.set()
        return reply

    with patch.object(client, "_get_guest_token", side_effect=cancelled_response) as get:
        with pytest.raises(CCTokenGenerateError, match="cancelled"):
            client.generate_with_retry(2, cancel)
        assert client._generation_lock.acquire(timeout=2)
        client._generation_lock.release()
        get.assert_called_once()


def test_guest_api_timeouts_capped_by_remaining_budget(material):
    _, client, _, _, _ = material
    client._generation_context.deadline = time.monotonic() + 1
    with patch("nvflare.app_opt.confidential_computing.coco_authorizer.requests.Session") as factory:
        session = factory.return_value.__enter__.return_value
        response = session.get.return_value.__enter__.return_value
        response.status_code = 200
        response.iter_content.return_value = [b'{"token": "fixture"}']
        assert client._get_guest_token() == {"token": "fixture"}
        connect, read = session.get.call_args.kwargs["timeout"]
        assert 0 < connect <= 1
        assert 0 < read <= 1


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), True, "30", None])
def test_retry_invalid_deadline(material, timeout):
    _, client, _, _, _ = material
    with pytest.raises(ValueError, match="finite and positive"):
        client.generate_with_retry(timeout, threading.Event())


@pytest.mark.parametrize("attempts", [0, -1, 101, True, 1.5, "5"])
def test_retry_invalid_attempt_limit(material, attempts):
    _, _, _, generate, _ = material
    with pytest.raises(ValueError, match="retry_max_attempts"):
        generate(retry_max_attempts=attempts)


def test_manager_registration_and_refresh_retry_real_authorizer(material):
    reply = guest_reply(material)
    _, client, verifier, _, _ = material
    manager = CCManager([], [])
    manager.cc_issuers = {client: 300}
    manager.site_name = "site-1"
    context = FLContext()
    with patch.object(context, "get_identity_name", return_value="site-1"):
        with patch.object(client, "_get_guest_token", side_effect=[requests.exceptions.Timeout(), reply] * 2):
            with patch("nvflare.app_opt.confidential_computing.coco_authorizer.random.uniform", return_value=0):
                manager._generate_and_attach_tokens(context)
                registration = context.get_prop(CC_INFO)["site-1"][0][CC_TOKEN]
                refreshed = manager._generate_fresh_tokens_for_validation()[0][CC_TOKEN]
    assert verifier.verify_for_site(registration, "site-1")
    assert verifier.verify_for_site(refreshed, "site-1")


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
