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

import base64
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

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_exception import NotAuthenticated
from nvflare.app_opt.confidential_computing.cc_authorizer import CCTokenGenerateError
from nvflare.app_opt.confidential_computing.cc_manager import CC_INFO, CC_NAMESPACE, CC_TOKEN, CCManager
from nvflare.app_opt.confidential_computing.coco_authorizer import (
    CPU_TRUST_VECTORS,
    EAT_PROFILE,
    TRUST_VECTOR,
    CoCoAuthorizer,
)
from nvflare.app_opt.confidential_computing.trustee_claims import normalized_init_data


def test_approved_trust_vectors_match_trustee_policy_contract():
    expected = {
        "executables": 3,
        "hardware": 2,
        "configuration": 2,
        "file-system": 0,
        "instance-identity": 0,
        "runtime-opaque": 0,
        "storage-opaque": 0,
        "sourced-data": 0,
    }
    assert TRUST_VECTOR == expected
    assert CPU_TRUST_VECTORS == {"snp": expected, "tdx": expected}


@pytest.fixture(params=[("rsa", "snp"), ("ec", "snp"), ("rsa", "tdx"), ("ec", "tdx")])
def material(request):
    key_type, cpu_type = request.param
    signer = ec.generate_private_key(ec.SECP256R1())
    key = (
        rsa.generate_private_key(public_exponent=65537, key_size=2048)
        if key_type == "rsa"
        else ec.generate_private_key(ec.SECP256R1())
    )
    algorithm = jwt.algorithms.RSAAlgorithm if key_type == "rsa" else jwt.algorithms.ECAlgorithm
    jwk = json.loads(algorithm.to_jwk(key.public_key()))
    if key_type == "rsa":
        jwk["alg"] = "RSA-OAEP-256"
    expected = {"init_data": "a" * 64, "image": "registry.example/workload@sha256:" + "b" * 64, "args": ["/start"]}
    # Match the pinned Trustee TDX claims.rs / AS flattening contract. These
    # synthetic values exercise the real claim shape, not hardware validation.
    init_data = expected["init_data"] + ("0" * 32 if cpu_type == "tdx" else "")
    claims = {
        "iat": int(time.time()),
        "exp": int(time.time()) + 300,
        "eat_profile": EAT_PROFILE,
        "submods": {
            "cpu0": {
                "ear.trustworthiness-vector": copy.deepcopy(CPU_TRUST_VECTORS[cpu_type]),
                "ear.veraison.annotated-evidence": {
                    # AS-signed evidence identifies the appraised CPU platform.
                    cpu_type: (
                        {"measurement": "c" * 96}
                        if cpu_type == "snp"
                        else {
                            "quote": {
                                "header": {"version": "0400", "tee_type": "81000000"},
                                "body": {
                                    "mr_td": "d" * 96,
                                    "mr_config_id": init_data,
                                    "rtmr_0": "0" * 96,
                                    "rtmr_1": "1" * 96,
                                    "rtmr_2": "2" * 96,
                                    "rtmr_3": "3" * 96,
                                    "td_attributes": "0000000000000000",
                                },
                            },
                            "td_attributes": {"debug": False},
                        }
                    ),
                    "runtime_data_claims": {"tee-pubkey": jwk},
                    "init_data": init_data,
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


@pytest.mark.parametrize("gpu", [False, True])
def test_valid_proof_and_single_use(material, gpu):
    claims, _, verifier, generate, _ = material
    if not gpu:
        claims["submods"].pop("gpu0")
    token = generate()
    proof = jwt.decode(token, options={"verify_signature": False})
    ear = jwt.decode(proof["ear"], options={"verify_signature": False})
    assert set(ear["submods"]) == ({"cpu0", "gpu0"} if gpu else {"cpu0"})
    assert "PRIVATE KEY" not in token
    assert verifier.verify(token)
    assert not verifier.verify(token)
    assert verifier.verify(generate())
    with pytest.raises(CCTokenGenerateError):
        verifier.generate()


@pytest.mark.parametrize("audience", [None, "other", "expected"])
def test_optional_ear_audience_is_verified(material, audience):
    claims, _, verifier, generate, _ = material
    if audience is not None:
        claims["aud"] = audience
    verifier.ear_audience = "expected"
    assert verifier.verify(generate()) is (audience == "expected")


@pytest.mark.parametrize("material", [("rsa", "snp"), ("ec", "snp")], indirect=True)
@pytest.mark.parametrize("change", ["none", "init_data", "measurement", "missing", "site"])
def test_optional_workload_constraints_are_verified(material, change):
    claims, _, verifier, generate, _ = material
    evidence = claims["submods"]["cpu0"]["ear.veraison.annotated-evidence"]
    evidence["snp"] = {"measurement": "b" * 96}
    verifier.workload_constraints = {"site-1": {"init_data": "a" * 64, "measurement": "b" * 96}}
    if change == "init_data":
        evidence["init_data"] = "c" * 64
    elif change == "measurement":
        evidence["snp"]["measurement"] = "c" * 96
    elif change == "missing":
        evidence["snp"] = {"policy": {"debug": False}}
    elif change == "site":
        verifier.workload_constraints = {"site-2": {"init_data": "a" * 64}}
    assert verifier.verify_for_site(generate(), "site-1") is (change == "none")


@pytest.mark.parametrize("change", ["none", "init_data", "missing", "site"])
def test_optional_init_data_constraints_are_verified_for_both_cpu_types(material, change):
    claims, _, verifier, generate, _ = material
    evidence = claims["submods"]["cpu0"]["ear.veraison.annotated-evidence"]
    verifier.workload_constraints = {"site-1": {"init_data": "a" * 64}}
    if change == "init_data":
        evidence["init_data"] = "c" * 64
    elif change == "missing":
        del evidence["init_data"]
    elif change == "site":
        verifier.workload_constraints = {"site-2": {"init_data": "a" * 64}}
    assert verifier.verify_for_site(generate(), "site-1") is (change == "none")


@pytest.mark.parametrize("material", [("rsa", "tdx"), ("ec", "tdx")], indirect=True)
def test_tdx_cannot_satisfy_snp_measurement_constraint(material):
    claims, _, verifier, generate, _ = material
    evidence = claims["submods"]["cpu0"]["ear.veraison.annotated-evidence"]
    verifier.workload_constraints = {"site-1": {"measurement": evidence["tdx"]["quote"]["body"]["mr_td"]}}
    assert not verifier.verify_for_site(generate(), "site-1")


@pytest.mark.parametrize(
    "pins",
    [
        {},
        [],
        {"": {"init_data": "a" * 64}},
        {"site-1": {}},
        {"site-1": {"other": "a" * 64}},
        {"site-1": {"init_data": "not-hex"}},
        {"site-1": {"cpu_tee": "sample"}},
        {"site-1": {"cpu_tee": True}},
        {"site-1": {"cpu_tee": "TDX"}},
        {"site-1": {"gpu_required": None}},
        {"site-1": {"gpu_required": 0}},
        {"site-1": {"gpu_required": 1}},
        {"site-1": {"gpu_required": "true"}},
        {"site-1": {"gpu_required": []}},
        {"site-1": {"tdx_mr_td": "a" * 64}},
        {"site-1": {"tdx_rtmr_0": "A" * 96}},
        {"site-1": {"tdx_rtmr_1": None}},
        {"site-1": {"tdx_rtmr_4": "a" * 96}},
        {"site-1": {"measurement": "a" * 96, "tdx_mr_td": "a" * 96}},
        {"site-1": {"cpu_tee": "tdx", "measurement": "a" * 96}},
        {"site-1": {"cpu_tee": "snp", "tdx_rtmr_1": "a" * 96}},
    ],
)
def test_malformed_workload_constraints_fail_at_construction(material, pins):
    _, _, _, generate, _ = material
    with pytest.raises(ValueError):
        generate(workload_constraints=pins)


@pytest.mark.parametrize("cpu_tee", ["snp", "tdx"])
def test_cpu_tee_constraint_pins_signed_platform(material, cpu_tee):
    claims, _, verifier, generate, _ = material
    evidence = claims["submods"]["cpu0"]["ear.veraison.annotated-evidence"]
    verifier.workload_constraints = {"site-1": {"cpu_tee": cpu_tee}}
    assert verifier.verify_for_site(generate(), "site-1") is (cpu_tee in evidence)


@pytest.mark.parametrize("gpu_required", [False, True])
@pytest.mark.parametrize("gpu_present", [False, True])
def test_site_gpu_requirement_matches_signed_appraisals(material, gpu_required, gpu_present):
    claims, _, verifier, generate, _ = material
    if not gpu_present:
        claims["submods"].pop("gpu0")
    verifier.workload_constraints = {"site-1": {"gpu_required": gpu_required}}
    token = generate(workload_constraints=verifier.workload_constraints)
    assert verifier.verify_for_site(token, "site-1") is (gpu_present == gpu_required)


def test_gpu_required_site_cannot_use_another_sites_cpu_only_permission(material):
    claims, _, verifier, generate, _ = material
    claims["submods"].pop("gpu0")
    verifier.workload_constraints = {
        "site-1": {"gpu_required": True},
        "site-2": {"gpu_required": False},
    }
    token = generate()
    assert not verifier.verify_for_site(token, "site-1")
    assert not verifier.verify_for_site(token, "site-2")


def test_typed_constraints_are_accepted_at_construction(material):
    claims, _, _, generate, _ = material
    evidence = claims["submods"]["cpu0"]["ear.veraison.annotated-evidence"]
    if "snp" in evidence:
        pins = {"cpu_tee": "snp", "init_data": "a" * 64, "measurement": evidence["snp"]["measurement"]}
    else:
        body = evidence["tdx"]["quote"]["body"]
        pins = {
            "cpu_tee": "tdx",
            "init_data": "a" * 64,
            **{f"tdx_{name}": body[name] for name in ("mr_td", "rtmr_0", "rtmr_1", "rtmr_2", "rtmr_3")},
        }
    pins["gpu_required"] = True
    assert generate(workload_constraints={"site-1": pins})


@pytest.mark.parametrize("material", [("rsa", "tdx"), ("ec", "tdx")], indirect=True)
@pytest.mark.parametrize("gpu", [False, True])
@pytest.mark.parametrize("change", ["none", "mr_td", "rtmr_0", "rtmr_1", "rtmr_2", "rtmr_3", "missing", "malformed"])
def test_tdx_typed_measurement_constraints(material, gpu, change):
    claims, _, verifier, generate, _ = material
    if not gpu:
        claims["submods"].pop("gpu0")
    evidence = claims["submods"]["cpu0"]["ear.veraison.annotated-evidence"]
    body = evidence["tdx"]["quote"]["body"]
    verifier.workload_constraints = {
        "site-1": {
            "cpu_tee": "tdx",
            "init_data": "a" * 64,
            **{f"tdx_{name}": body[name] for name in ("mr_td", "rtmr_0", "rtmr_1", "rtmr_2", "rtmr_3")},
        }
    }
    if change == "missing":
        body.pop("rtmr_2")
    elif change == "malformed":
        evidence["tdx"]["quote"] = []
    elif change != "none":
        body[change] = "f" * 96
    assert verifier.verify_for_site(generate(), "site-1") is (change == "none")


@pytest.mark.parametrize("material", [("rsa", "snp"), ("ec", "snp")], indirect=True)
@pytest.mark.parametrize("pin", ["tdx_mr_td", "tdx_rtmr_0", "tdx_rtmr_1", "tdx_rtmr_2", "tdx_rtmr_3"])
def test_snp_cannot_satisfy_tdx_measurement_constraint(material, pin):
    claims, _, verifier, generate, _ = material
    evidence = claims["submods"]["cpu0"]["ear.veraison.annotated-evidence"]
    verifier.workload_constraints = {"site-1": {pin: evidence["snp"]["measurement"]}}
    assert not verifier.verify_for_site(generate(), "site-1")


@pytest.mark.parametrize("material", [("rsa", "tdx"), ("ec", "tdx")], indirect=True)
@pytest.mark.parametrize(
    "change", ["padding", "short", "long", "upper", "not_hex", "null", "body_mismatch", "body_missing", "body_type"]
)
def test_tdx_init_data_pin_rejects_malformed_or_inconsistent_binding(material, change):
    claims, _, verifier, generate, _ = material
    evidence = claims["submods"]["cpu0"]["ear.veraison.annotated-evidence"]
    body = evidence["tdx"]["quote"]["body"]
    verifier.workload_constraints = {"site-1": {"init_data": "a" * 64}}
    if change == "body_mismatch":
        body["mr_config_id"] = "b" * 64 + "0" * 32
    elif change == "body_missing":
        body.pop("mr_config_id")
    elif change == "body_type":
        evidence["tdx"]["quote"]["body"] = []
    else:
        value = {
            "padding": "a" * 64 + "0" * 31 + "1",
            "short": "a" * 64,
            "long": "a" * 64 + "0" * 64,
            "upper": "A" * 64 + "0" * 32,
            "not_hex": "z" * 64 + "0" * 32,
            "null": None,
        }[change]
        evidence["init_data"] = body["mr_config_id"] = value
    assert not verifier.verify_for_site(generate(), "site-1")


@pytest.mark.parametrize("material", [("rsa", "tdx"), ("ec", "tdx")], indirect=True)
@pytest.mark.parametrize("change", ["mr_td", "rtmr_1", "init_data"])
def test_tdx_pin_cannot_be_satisfied_by_modifying_ear_and_resigning_outer_proof(material, change):
    _, _, verifier, generate, key = material
    proof = jwt.decode(generate(), options={"verify_signature": False})
    header, _, signature = proof["ear"].split(".")
    claims = jwt.decode(proof["ear"], options={"verify_signature": False})
    evidence = claims["submods"]["cpu0"]["ear.veraison.annotated-evidence"]
    body = evidence["tdx"]["quote"]["body"]
    if change == "init_data":
        evidence["init_data"] = body["mr_config_id"] = "b" * 64 + "0" * 32
        verifier.workload_constraints = {"site-1": {"init_data": "b" * 64}}
    else:
        body[change] = "b" * 96
        verifier.workload_constraints = {"site-1": {f"tdx_{change}": "b" * 96}}
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=").decode()
    proof["ear"] = ".".join((header, payload, signature))
    token = jwt.encode(proof, key, algorithm=CoCoAuthorizer._algorithm(key))
    assert not verifier.verify_for_site(token, "site-1")


@pytest.mark.parametrize("material", [("rsa", "snp"), ("ec", "snp")], indirect=True)
def test_snp_init_data_does_not_accept_tdx_padding(material):
    claims, _, verifier, generate, _ = material
    evidence = claims["submods"]["cpu0"]["ear.veraison.annotated-evidence"]
    verifier.workload_constraints = {"site-1": {"init_data": "a" * 64}}
    evidence["init_data"] += "0" * 32
    assert not verifier.verify_for_site(generate(), "site-1")


@pytest.mark.parametrize("change", ["none", "unknown", "ambiguous"])
def test_init_data_normalizer_requires_explicit_unambiguous_cpu_evidence(material, change):
    claims, _, _, _, _ = material
    evidence = copy.deepcopy(claims["submods"]["cpu0"]["ear.veraison.annotated-evidence"])
    if change == "none":
        assert normalized_init_data(evidence) == "a" * 64
    else:
        if change == "unknown":
            evidence["untrusted"] = {"measurement": "a" * 96}
        else:
            evidence["tdx" if "snp" in evidence else "snp"] = {"measurement": "a" * 96}
        with pytest.raises(ValueError):
            normalized_init_data(evidence)


def test_replay_cache_is_process_local_not_a_challenge_protocol(material):
    _, _, verifier, generate, _ = material
    token = generate()
    assert verifier.verify(token)
    assert not verifier.verify(token)
    # An explicit restart loses replay history. FL mTLS, protected client keys,
    # subject binding and expiry remain required; do not claim nonce freshness.
    verifier.seen.clear()
    assert verifier.verify(token)


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


@pytest.mark.parametrize("lag", [0, 2, 90, 180])
def test_protected_server_proof_verified_by_lagging_client(material, lag):
    claims, client, verifier, _, _ = material
    now = int(time.time())
    claims.update(iat=now, exp=now + 300)
    public = client.trustee_key.public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    ).decode()
    server = CoCoAuthorizer(public, client.audience, site_name="server")
    # Both signatures are real; only the guest API and the two peer clocks are mocked.
    with clock_at(now), patch.object(server, "_get_guest_token", return_value=guest_reply(material)):
        token = server.generate()
    with clock_at(now - lag):
        assert verifier.proof_iat_leeway_seconds == 180
        assert verifier.verify_for_site(token, "server")
        assert not verifier.verify_for_site(token, "server")


@pytest.mark.parametrize("leeway", [0, 30, 180])
def test_proof_iat_leeway_boundary(material, leeway):
    claims, client, _, generate, _ = material
    now = int(time.time())
    claims.update(iat=now, exp=now + 300)
    public = client.trustee_key.public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    ).decode()
    verifier = CoCoAuthorizer(public, client.audience, proof_iat_leeway_seconds=leeway)
    with clock_at(now + leeway):
        token = generate()
    with clock_at(now):
        assert verifier.verify_for_site(token, "site-1")
    with clock_at(now + leeway + 1):
        outside_window = generate()
    with clock_at(now):
        assert not verifier.verify_for_site(outside_window, "site-1")


@pytest.mark.parametrize(
    "future_claim,ear_leeway,proof_leeway,accepted",
    [("proof", 0, 180, True), ("proof", 180, 0, False), ("ear", 180, 0, True), ("ear", 0, 180, False)],
)
def test_ear_and_proof_clock_skew_allowances_are_independent(
    material, future_claim, ear_leeway, proof_leeway, accepted
):
    claims, client, _, generate, _ = material
    now = int(time.time())
    claims.update(iat=now + (90 if future_claim == "ear" else 0), exp=now + 300)
    public = client.trustee_key.public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    ).decode()
    verifier = CoCoAuthorizer(
        public, client.audience, ear_leeway_seconds=ear_leeway, proof_iat_leeway_seconds=proof_leeway
    )
    with clock_at(now + (90 if future_claim == "proof" else 0)):
        token = generate()
    with clock_at(now):
        assert verifier.verify_for_site(token, "site-1") is accepted


@pytest.mark.parametrize("leeway", [-1, 181, True, False, 1.5, "180", None])
def test_invalid_proof_iat_leeway_rejected(material, leeway):
    _, _, _, generate, _ = material
    with pytest.raises(ValueError, match="proof_iat_leeway_seconds must be 0..180"):
        generate(proof_iat_leeway_seconds=leeway)


@pytest.mark.parametrize("kind", ["missing", "null", "true", "false", "string", "float", "list", "object"])
def test_proof_iat_requires_an_integer_with_skew_tolerance(material, kind):
    _, _, verifier, generate, key = material
    proof = jwt.decode(generate(), options={"verify_signature": False})
    valid_iat = proof.pop("iat")
    if kind != "missing":
        proof["iat"] = {
            "null": None,
            "true": True,
            "false": False,
            "string": str(valid_iat),
            "float": float(valid_iat),
            "list": [],
            "object": {},
        }[kind]
    token = jwt.encode(proof, key, algorithm=CoCoAuthorizer._algorithm(key))
    assert not verifier.verify(token)


@pytest.mark.parametrize(
    "claim,offset,accepted",
    [("exp", -1, False), ("exp", 0, False), ("exp", 1, True), ("nbf", 1, False), ("nbf", 0, True)],
)
def test_proof_iat_leeway_does_not_relax_expiry_or_not_before(material, claim, offset, accepted):
    claims, _, verifier, generate, key = material
    now = int(time.time())
    claims.update(iat=now, exp=now + 300)
    with clock_at(now):
        proof = jwt.decode(generate(), options={"verify_signature": False})
        # Keep the signed lifetime and EAR valid so only the target claim decides.
        proof["iat"] = now - 10
        proof[claim] = now + offset
        if claim == "nbf":
            proof["exp"] = now + 290
        token = jwt.encode(proof, key, algorithm=CoCoAuthorizer._algorithm(key))
        assert verifier.verify(token) is accepted


def test_future_dated_proof_replay_stays_blocked_until_strict_expiry(material):
    claims, _, verifier, generate, _ = material
    now = int(time.time())
    claims.update(iat=now, exp=now + 600)
    with clock_at(now + 90):
        token = generate(proof_lifetime_seconds=60)
    proof = jwt.decode(token, options={"verify_signature": False})
    cache_key = (proof["sub"], proof["jti"])
    with clock_at(now):
        assert verifier.verify(token)
        assert verifier.seen[cache_key] == now + 150
    for offset in (0, 89, 90, 149):
        with clock_at(now + offset):
            assert not verifier.verify(token)
            assert verifier.seen[cache_key] == now + 150
    with clock_at(now + 150):
        assert not verifier.verify(token)
        # Even losing replay state must not make an exactly expired proof valid.
        verifier.seen.clear()
        assert not verifier.verify(token)


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
        proof["iat"] += verifier.proof_iat_leeway_seconds + 1
        proof["exp"] += verifier.proof_iat_leeway_seconds + 1
    token = jwt.encode(proof, key, algorithm=CoCoAuthorizer._algorithm(key))
    assert not verifier.verify(token)


def test_longer_proof_does_not_extend_ear_freshness(material):
    claims, _, _, generate, _ = material
    claims["iat"] -= 301
    with pytest.raises(CCTokenGenerateError):
        generate(proof_lifetime_seconds=600)


@pytest.mark.parametrize("gpu", [False, True])
def test_peer_bound_proof(material, gpu):
    claims, _, verifier, generate, _ = material
    if not gpu:
        claims["submods"].pop("gpu0")
    token = generate()
    assert not verifier.verify_for_site(token, "site-2")
    assert not verifier.verify_for_site(token, "")
    assert verifier.verify_for_site(token, "site-1")
    assert not verifier.verify_for_site(token, "site-1")


@pytest.mark.parametrize("case", ["valid", "hostname", "signature", "ear_signature"])
def test_server_proof_is_verified_locally_with_logical_identity(material, case):
    _, client, verifier, _, key = material
    public = client.trustee_key.public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    ).decode()
    server = CoCoAuthorizer(public, client.audience, site_name="server")
    # The fixture signs an EAR locally; no real guest or hardware is involved.
    with patch.object(server, "_get_guest_token", return_value=guest_reply(material)) as guest:
        token = server.generate()
    guest.assert_called_once()
    proof = jwt.decode(token, options={"verify_signature": False})
    assert proof["sub"] == "server"

    with (
        patch.object(verifier, "_get_guest_token") as verifier_guest,
        patch("nvflare.app_opt.confidential_computing.coco_authorizer.requests.Session") as session,
    ):
        assert not verifier.verify_for_site(token, "server.example.com")
        if case != "valid":
            if case == "hostname":
                proof["sub"] = "server.example.com"
            elif case == "signature":
                key = ec.generate_private_key(ec.SECP256R1())
            else:
                claims = jwt.decode(proof["ear"], options={"verify_signature": False})
                proof["ear"] = jwt.encode(claims, ec.generate_private_key(ec.SECP256R1()), algorithm="ES256")
            token = jwt.encode(proof, key, algorithm=CoCoAuthorizer._algorithm(key))
        assert verifier.verify_for_site(token, "server") is (case == "valid")
        with pytest.raises(CCTokenGenerateError, match="Verifier-only"):
            verifier.generate()
        verifier_guest.assert_not_called()
        session.assert_not_called()


@pytest.mark.parametrize("gpu", [False, True])
def test_registration_rejects_other_clients_real_signed_proof(material, gpu):
    claims, _, verifier, generate, _ = material
    if not gpu:
        claims["submods"].pop("gpu0")
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
                manager.handle_event(EventType.CLIENT_REGISTER_RECEIVED, context)
        else:
            manager.handle_event(EventType.CLIENT_REGISTER_RECEIVED, context)


@pytest.mark.parametrize(
    "failure", ["cpu_missing", "gpu_failed", "cpu_failed", "extra_submod", "old", "expired", "wrong_key"]
)
def test_bad_ear_rejected(material, failure):
    claims, client, _, generate, _ = material
    if failure == "cpu_missing":
        claims["submods"].pop("cpu0")
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


@pytest.mark.parametrize("gpu", [False, True])
@pytest.mark.parametrize("failure", ["subject", "audience", "signature", "time", "ear_signature"])
def test_invalid_proof_rejected(material, failure, gpu):
    claims, _, verifier, generate, key = material
    if not gpu:
        claims["submods"].pop("gpu0")
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


@pytest.mark.parametrize(
    "submods", [None, [], ["cpu0"], "cpu0", {}, {"gpu0": {}}, {"cpu0": None}, {"cpu0": []}, {"cpu1": {}}]
)
def test_malformed_signed_submodules_rejected(material, submods):
    claims, client, verifier, generate, key = material
    claims["submods"] = submods
    with pytest.raises(CCTokenGenerateError):
        generate()
    # Bypass only the generating client's appraisal to test the independent
    # verifier with an AS-signed EAR and a genuinely signed outer proof.
    with patch.object(client, "_ear", return_value=(claims, key.public_key())):
        token = generate()
    assert not verifier.verify(token)


@pytest.mark.parametrize("gpu", [False, True])
@pytest.mark.parametrize("failure", ["cpu_failed", "cpu_vector_missing", "cpu_key_missing", "extra_submod"])
def test_signed_invalid_cpu_evidence_rejected_in_both_modes(material, gpu, failure):
    claims, client, verifier, generate, key = material
    if not gpu:
        claims["submods"].pop("gpu0")
    cpu = claims["submods"]["cpu0"]
    if failure == "cpu_failed":
        cpu["ear.trustworthiness-vector"]["hardware"] = 97
    elif failure == "cpu_vector_missing":
        cpu.pop("ear.trustworthiness-vector")
    elif failure == "cpu_key_missing":
        cpu["ear.veraison.annotated-evidence"]["runtime_data_claims"].pop("tee-pubkey")
    else:
        claims["submods"]["gpu1"] = copy.deepcopy(cpu)
    with pytest.raises(CCTokenGenerateError):
        generate()
    with patch.object(client, "_ear", return_value=(claims, key.public_key())):
        token = generate()
    assert not verifier.verify(token)


@pytest.mark.parametrize("gpu_claim", [None, [], {}, {"ear.trustworthiness-vector": {**TRUST_VECTOR, "hardware": 97}}])
def test_present_invalid_gpu_never_falls_back_to_cpu_only(material, gpu_claim):
    claims, client, verifier, generate, key = material
    claims["submods"]["gpu0"] = gpu_claim
    with pytest.raises(CCTokenGenerateError):
        generate()
    with patch.object(client, "_ear", return_value=(claims, key.public_key())):
        token = generate()
    assert not verifier.verify(token)


def test_removing_gpu_from_signed_ear_rejected_even_with_resigned_outer_proof(material):
    _, _, verifier, generate, key = material
    proof = jwt.decode(generate(), options={"verify_signature": False})
    header, payload, signature = proof["ear"].split(".")
    claims = jwt.decode(proof["ear"], options={"verify_signature": False})
    claims["submods"].pop("gpu0")
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=").decode()
    proof["ear"] = ".".join((header, payload, signature))
    token = jwt.encode(proof, key, algorithm=CoCoAuthorizer._algorithm(key))
    assert not verifier.verify(token)


@pytest.mark.parametrize("failure", ["missing", "unknown", "ambiguous", "null", "empty", "list"])
def test_cpu_type_must_be_supported_unambiguous_signed_evidence(material, failure):
    claims, client, verifier, generate, key = material
    evidence = claims["submods"]["cpu0"]["ear.veraison.annotated-evidence"]
    cpu_type = "snp" if "snp" in evidence else "tdx"
    if failure == "missing":
        evidence.pop(cpu_type)
    elif failure == "unknown":
        evidence["sample"] = evidence.pop(cpu_type)
    elif failure == "ambiguous":
        evidence["tdx" if cpu_type == "snp" else "snp"] = copy.deepcopy(evidence[cpu_type])
    else:
        evidence[cpu_type] = {"null": None, "empty": {}, "list": []}[failure]
    with pytest.raises(CCTokenGenerateError):
        generate()
    with patch.object(client, "_ear", return_value=(claims, key.public_key())):
        token = generate()
    assert not verifier.verify(token)


@pytest.mark.parametrize("submod", ["cpu0", "gpu0"])
@pytest.mark.parametrize("field,value", [("configuration", 0), ("configuration", True), ("executables", 4)])
def test_platform_vectors_do_not_accept_generic_success_threshold(material, submod, field, value):
    claims, client, verifier, generate, key = material
    claims["submods"][submod]["ear.trustworthiness-vector"][field] = value
    with pytest.raises(CCTokenGenerateError):
        generate()
    with patch.object(client, "_ear", return_value=(claims, key.public_key())):
        token = generate()
    assert not verifier.verify(token)


@pytest.mark.parametrize("submod", ["cpu0", "gpu0"])
def test_configuration_three_is_not_an_approved_trustee_claim(material, submod):
    claims, client, verifier, generate, key = material
    claims["submods"][submod]["ear.trustworthiness-vector"]["configuration"] = 3
    with pytest.raises(CCTokenGenerateError):
        generate()
    with patch.object(client, "_ear", return_value=(claims, key.public_key())):
        token = generate()
    assert not verifier.verify(token)


def test_changing_cpu_type_without_as_signature_is_rejected(material):
    _, _, verifier, generate, key = material
    proof = jwt.decode(generate(), options={"verify_signature": False})
    header, _, signature = proof["ear"].split(".")
    claims = jwt.decode(proof["ear"], options={"verify_signature": False})
    cpu = claims["submods"]["cpu0"]
    evidence = cpu["ear.veraison.annotated-evidence"]
    old_type = "snp" if "snp" in evidence else "tdx"
    new_type = "tdx" if old_type == "snp" else "snp"
    evidence[new_type] = evidence.pop(old_type)
    cpu["ear.trustworthiness-vector"] = CPU_TRUST_VECTORS[new_type]
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=").decode()
    proof["ear"] = ".".join((header, payload, signature))
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
