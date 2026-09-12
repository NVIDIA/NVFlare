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
from unittest.mock import patch

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa

from nvflare.app_opt.confidential_computing.cc_authorizer import CCTokenGenerateError
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

    def generate():
        reply = {
            "token": jwt.encode(claims, signer, algorithm="ES256"),
            "tee_keypair": key.private_bytes(
                serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
            ).decode(),
        }
        with patch.object(client, "_get_guest_token", return_value=reply):
            return client.generate()

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
            proof["iat"] -= 120
            proof["exp"] -= 120
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
