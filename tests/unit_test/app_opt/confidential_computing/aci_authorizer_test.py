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

import time
from types import SimpleNamespace
from unittest.mock import Mock

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from nvflare.app_opt.confidential_computing.aci_authorizer import ACIAuthorizer

APPROVED_POLICY_HASH = "ab" * 32


@pytest.fixture(scope="module")
def signing_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture
def authorizer(monkeypatch, signing_key):
    # Replace only remote signing-key discovery. Signature and temporal checks
    # below run through the production jwt.decode implementation.
    jwks_client = Mock()
    jwks_client.get_signing_key_from_jwt.return_value = SimpleNamespace(key=signing_key.public_key())
    monkeypatch.setattr(jwt, "PyJWKClient", Mock(return_value=jwks_client))
    return ACIAuthorizer(expected_policy_hash=APPROVED_POLICY_HASH)


def signed_token(signing_key, *, hostdata=APPROVED_POLICY_HASH, expires_in=300, include_hostdata=True):
    now = int(time.time())
    claims = {"iat": now - 10, "nbf": now - 10, "exp": now + expires_in}
    if include_hostdata:
        claims["x-ms-sevsnpvm-hostdata"] = hostdata
    return jwt.encode(claims, signing_key, algorithm="RS256")


@pytest.mark.parametrize("expected", [None, "", "ab" * 31, "ab" * 33, "g" * 64, " ab" * 32, 123, True])
def test_requires_approved_policy_hash(expected):
    with pytest.raises(ValueError, match="expected_policy_hash"):
        ACIAuthorizer(expected_policy_hash=expected)


def test_requires_policy_hash_when_omitted():
    with pytest.raises(ValueError, match="expected_policy_hash"):
        ACIAuthorizer()


def test_normalizes_configured_policy_hash():
    authorizer = ACIAuthorizer(expected_policy_hash=APPROVED_POLICY_HASH.upper())
    assert authorizer.expected_policy_hash == APPROVED_POLICY_HASH


@pytest.mark.parametrize("hostdata", [APPROVED_POLICY_HASH, APPROVED_POLICY_HASH.upper()])
def test_accepts_approved_policy(authorizer, signing_key, hostdata):
    assert authorizer.verify(signed_token(signing_key, hostdata=hostdata)) is True


@pytest.mark.parametrize("hostdata", ["cd" * 32, None, "", "ab" * 31, "ab" * 33, "g" * 64, [], {}, 123, True])
def test_rejects_different_or_malformed_policy(authorizer, signing_key, hostdata):
    assert authorizer.verify(signed_token(signing_key, hostdata=hostdata)) is False


def test_rejects_missing_policy(authorizer, signing_key):
    assert authorizer.verify(signed_token(signing_key, include_hostdata=False)) is False


def test_rejects_expired_evidence_for_approved_policy(authorizer, signing_key):
    assert authorizer.verify(signed_token(signing_key, expires_in=-1)) is False


def test_rejects_invalid_signature_for_approved_policy(authorizer, signing_key):
    token = signed_token(signing_key)
    header, claims, signature = token.split(".")
    # Change the leading signature byte so the signed payload stays intact.
    signature = ("A" if signature[0] != "A" else "B") + signature[1:]
    assert authorizer.verify(".".join((header, claims, signature))) is False


def test_rejects_malformed_token(authorizer):
    assert authorizer.verify("not-an-attestation-token") is False
