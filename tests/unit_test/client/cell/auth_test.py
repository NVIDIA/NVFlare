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

import dataclasses
import hashlib

import pytest

from nvflare.client.cell.auth import (
    TokenScope,
    combine_nonces,
    compute_hello_proof,
    generate_nonce,
    generate_session_token,
    token_digest,
    verify_hello_proof,
)
from nvflare.client.cell.defs import PROTOCOL_VERSION


def _make_scope(**overrides) -> TokenScope:
    kwargs = dict(
        job_id="job-1234",
        site_name="site-1",
        attach_id="attach-abcd",
        target_fqcn="site-1.job-1234",
        trainer_fqcn="site-1.-client_api_attach-abcd",
        rank_policy="0",
        protocol_version=PROTOCOL_VERSION,
    )
    kwargs.update(overrides)
    return TokenScope(**kwargs)


class TestTokenAndDigest:
    def test_digest_is_stable_sha256_of_token(self):
        token = generate_session_token()
        expected = hashlib.sha256(token.encode("utf-8")).hexdigest()
        assert token_digest(token) == expected
        assert token_digest(token) == token_digest(token)

    def test_token_entropy_and_uniqueness(self):
        tokens = {generate_session_token() for _ in range(100)}
        assert len(tokens) == 100
        # default 32 bytes of entropy -> 64 hex chars
        assert all(len(t) == 64 for t in tokens)

    def test_nonce_uniqueness(self):
        nonces = {generate_nonce() for _ in range(100)}
        assert len(nonces) == 100


class TestHelloProof:
    def test_proof_round_trip(self):
        token = generate_session_token()
        nonce = generate_nonce()
        scope = _make_scope()
        proof = compute_hello_proof(token, nonce, scope)
        assert verify_hello_proof(token, nonce, scope, proof)

    def test_tampered_nonce_fails(self):
        token = generate_session_token()
        nonce = generate_nonce()
        scope = _make_scope()
        proof = compute_hello_proof(token, nonce, scope)
        assert not verify_hello_proof(token, generate_nonce(), scope, proof)

    @pytest.mark.parametrize(
        "field,value",
        [
            ("job_id", "job-9999"),
            ("site_name", "site-2"),
            ("attach_id", "attach-zzzz"),
            ("target_fqcn", "site-2.job-9999"),
            ("trainer_fqcn", "site-1.-client_api_other"),
            ("rank_policy", "1"),
        ],
    )
    def test_tampered_scope_field_fails(self, field, value):
        token = generate_session_token()
        nonce = generate_nonce()
        scope = _make_scope()
        proof = compute_hello_proof(token, nonce, scope)
        tampered = dataclasses.replace(scope, **{field: value})
        assert not verify_hello_proof(token, nonce, tampered, proof)

    def test_wrong_protocol_version_fails(self):
        token = generate_session_token()
        nonce = generate_nonce()
        scope = _make_scope()
        proof = compute_hello_proof(token, nonce, scope)
        skewed = dataclasses.replace(scope, protocol_version=PROTOCOL_VERSION + 1)
        assert not verify_hello_proof(token, nonce, skewed, proof)

    def test_wrong_token_fails(self):
        nonce = generate_nonce()
        scope = _make_scope()
        proof = compute_hello_proof(generate_session_token(), nonce, scope)
        assert not verify_hello_proof(generate_session_token(), nonce, scope, proof)

    def test_empty_or_non_str_proof_fails(self):
        token = generate_session_token()
        nonce = generate_nonce()
        scope = _make_scope()
        assert not verify_hello_proof(token, nonce, scope, "")
        assert not verify_hello_proof(token, nonce, scope, None)

    def test_serialization_is_unambiguous_across_field_boundaries(self):
        # shifting bytes between adjacent fields must not produce the same proof
        token = generate_session_token()
        nonce = generate_nonce()
        scope_a = _make_scope(job_id="ab", site_name="c")
        scope_b = _make_scope(job_id="a", site_name="bc")
        assert compute_hello_proof(token, nonce, scope_a) != compute_hello_proof(token, nonce, scope_b)

    def test_non_ascii_proof_returns_false_without_raising(self):
        # hmac.compare_digest raises TypeError on a non-ASCII str; verify must return False.
        token = generate_session_token()
        nonce = generate_nonce()
        scope = _make_scope()
        assert verify_hello_proof(token, nonce, scope, "deadbeefÿ") is False
        assert verify_hello_proof(token, nonce, scope, "中文") is False

    def test_cross_typed_scope_produces_different_proof_and_fails(self):
        # protocol_version=1 (int) and protocol_version="1" (str) must not be interchangeable.
        token = generate_session_token()
        nonce = generate_nonce()
        scope_int = _make_scope(protocol_version=1)
        scope_str = _make_scope(protocol_version="1")
        proof_int = compute_hello_proof(token, nonce, scope_int)
        proof_str = compute_hello_proof(token, nonce, scope_str)
        assert proof_int != proof_str
        # a proof computed over the int scope must not verify for the str scope, and vice versa
        assert not verify_hello_proof(token, nonce, scope_str, proof_int)
        assert not verify_hello_proof(token, nonce, scope_int, proof_str)


class TestOneRoundProof:
    """The one-round variant folds the proof into HELLO over a combined executor+trainer nonce
    and uses the module-level functions directly (no SessionTokenManager / issue_nonce)."""

    def test_one_round_round_trip(self):
        token = generate_session_token()
        scope = _make_scope()
        executor_nonce = generate_nonce()  # delivered in bootstrap config
        trainer_nonce = generate_nonce()  # contributed by the trainer
        combined = combine_nonces(executor_nonce, trainer_nonce)
        proof = compute_hello_proof(token, combined, scope)
        assert verify_hello_proof(token, combined, scope, proof)

    def test_one_round_tampered_trainer_nonce_fails(self):
        token = generate_session_token()
        scope = _make_scope()
        executor_nonce = generate_nonce()
        trainer_nonce = generate_nonce()
        proof = compute_hello_proof(token, combine_nonces(executor_nonce, trainer_nonce), scope)
        forged = combine_nonces(executor_nonce, generate_nonce())
        assert not verify_hello_proof(token, forged, scope, proof)

    def test_one_round_tampered_executor_nonce_fails(self):
        token = generate_session_token()
        scope = _make_scope()
        executor_nonce = generate_nonce()
        trainer_nonce = generate_nonce()
        proof = compute_hello_proof(token, combine_nonces(executor_nonce, trainer_nonce), scope)
        forged = combine_nonces(generate_nonce(), trainer_nonce)
        assert not verify_hello_proof(token, forged, scope, proof)

    def test_combine_nonces_is_unambiguous(self):
        # the split point between the two nonces must not be shiftable
        assert combine_nonces("ab", "c") != combine_nonces("a", "bc")
