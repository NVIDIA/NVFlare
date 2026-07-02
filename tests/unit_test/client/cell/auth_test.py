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
    MIN_TOKEN_HEX_CHARS,
    SessionTokenManager,
    TokenScope,
    combine_nonces,
    compute_hello_proof,
    generate_nonce,
    generate_session_token,
    token_digest,
    verify_hello_proof,
)
from nvflare.client.cell.defs import PROTOCOL_VERSION


class _FakeClock:
    """Injectable monotonic clock for expiry tests (no sleeps)."""

    def __init__(self, start: float = 1000.0):
        self.now = start

    def advance(self, seconds: float):
        self.now += seconds

    def __call__(self) -> float:
        return self.now


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

    def test_manager_digest_matches_token_digest(self):
        mgr = SessionTokenManager(scope=_make_scope())
        assert mgr.digest == token_digest(mgr.token)

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


class TestSessionTokenManager:
    def test_successful_attach(self):
        scope = _make_scope()
        mgr = SessionTokenManager(scope=scope)
        nonce = mgr.issue_nonce()
        proof = compute_hello_proof(mgr.token, nonce, scope)
        assert mgr.verify_proof(nonce, scope, proof)
        assert mgr.session_active

    def test_consumed_nonce_rejected(self):
        scope = _make_scope()
        mgr = SessionTokenManager(scope=scope)
        nonce = mgr.issue_nonce()
        proof = compute_hello_proof(mgr.token, nonce, scope)
        assert mgr.verify_proof(nonce, scope, proof)
        # the same nonce (already consumed) must be rejected even with a valid proof
        assert not mgr.verify_proof(nonce, scope, proof)

    def test_failed_attempt_also_consumes_nonce(self):
        scope = _make_scope()
        mgr = SessionTokenManager(scope=scope)
        nonce = mgr.issue_nonce()
        assert not mgr.verify_proof(nonce, scope, "bad-proof")
        # nonce was consumed by the failed attempt; a now-valid proof must still fail
        proof = compute_hello_proof(mgr.token, nonce, scope)
        assert not mgr.verify_proof(nonce, scope, proof)

    def test_replayed_proof_against_fresh_nonce_fails(self):
        scope = _make_scope()
        mgr = SessionTokenManager(scope=scope)
        nonce1 = mgr.issue_nonce()
        observed_proof = compute_hello_proof(mgr.token, nonce1, scope)
        # attacker observed the proof for nonce1 but is challenged with a fresh nonce
        nonce2 = mgr.issue_nonce()
        assert not mgr.verify_proof(nonce2, scope, observed_proof)

    def test_unissued_nonce_rejected(self):
        scope = _make_scope()
        mgr = SessionTokenManager(scope=scope)
        nonce = generate_nonce()  # not issued by the manager
        proof = compute_hello_proof(mgr.token, nonce, scope)
        assert not mgr.verify_proof(nonce, scope, proof)

    def test_scope_mismatch_rejected(self):
        scope = _make_scope()
        mgr = SessionTokenManager(scope=scope)
        nonce = mgr.issue_nonce()
        wrong_scope = _make_scope(attach_id="attach-other")
        # proof is internally consistent with the wrong scope, but scope != expected scope
        proof = compute_hello_proof(mgr.token, nonce, wrong_scope)
        assert not mgr.verify_proof(nonce, wrong_scope, proof)
        assert not mgr.session_active

    def test_wrong_protocol_version_rejected(self):
        scope = _make_scope()
        mgr = SessionTokenManager(scope=scope)
        nonce = mgr.issue_nonce()
        skewed = dataclasses.replace(scope, protocol_version=PROTOCOL_VERSION + 1)
        proof = compute_hello_proof(mgr.token, nonce, skewed)
        assert not mgr.verify_proof(nonce, skewed, proof)

    def test_expiry_after_attach_timeout(self):
        clock = _FakeClock()
        scope = _make_scope()
        mgr = SessionTokenManager(scope=scope, attach_timeout=30.0, clock=clock)
        nonce = mgr.issue_nonce()
        proof = compute_hello_proof(mgr.token, nonce, scope)
        clock.advance(30.1)
        assert mgr.is_expired()
        assert not mgr.verify_proof(nonce, scope, proof)
        # issuing a fresh nonce after expiry is refused
        with pytest.raises(RuntimeError):
            mgr.issue_nonce()

    def test_attach_within_timeout_succeeds(self):
        clock = _FakeClock()
        scope = _make_scope()
        mgr = SessionTokenManager(scope=scope, attach_timeout=30.0, clock=clock)
        nonce = mgr.issue_nonce()
        proof = compute_hello_proof(mgr.token, nonce, scope)
        clock.advance(29.9)
        assert not mgr.is_expired()
        assert mgr.verify_proof(nonce, scope, proof)

    def test_active_session_does_not_expire(self):
        clock = _FakeClock()
        scope = _make_scope()
        mgr = SessionTokenManager(scope=scope, attach_timeout=30.0, clock=clock)
        nonce = mgr.issue_nonce()
        proof = compute_hello_proof(mgr.token, nonce, scope)
        assert mgr.verify_proof(nonce, scope, proof)
        clock.advance(3600)
        # after attach, the session is governed by heartbeat/job lifetime, not attach_timeout
        assert not mgr.is_expired()

    def test_single_session_enforced(self):
        scope = _make_scope()
        mgr = SessionTokenManager(scope=scope)
        # both nonces are issued before the session becomes active (issuing while a session is
        # active is refused; see test_issue_nonce_refused_while_session_active)
        nonce1 = mgr.issue_nonce()
        nonce2 = mgr.issue_nonce()
        proof1 = compute_hello_proof(mgr.token, nonce1, scope)
        proof2 = compute_hello_proof(mgr.token, nonce2, scope)
        assert mgr.verify_proof(nonce1, scope, proof1)
        # a second, otherwise fully valid attach attempt must be rejected
        assert not mgr.verify_proof(nonce2, scope, proof2)
        assert mgr.session_active
        # and no further nonce can be issued while the session is active
        with pytest.raises(RuntimeError):
            mgr.issue_nonce()

    def test_invalidate_kills_future_verification(self):
        scope = _make_scope()
        mgr = SessionTokenManager(scope=scope)
        # a nonce issued before invalidate must not verify afterwards
        nonce = mgr.issue_nonce()
        proof = compute_hello_proof(mgr.token, nonce, scope)
        mgr.invalidate()
        assert mgr.invalidated
        assert not mgr.session_active
        assert not mgr.verify_proof(nonce, scope, proof)
        # and issuing a new nonce after invalidate is refused
        with pytest.raises(RuntimeError):
            mgr.issue_nonce()

    def test_invalidate_after_attach_ends_session(self):
        scope = _make_scope()
        mgr = SessionTokenManager(scope=scope)
        nonce = mgr.issue_nonce()
        proof = compute_hello_proof(mgr.token, nonce, scope)
        assert mgr.verify_proof(nonce, scope, proof)
        mgr.invalidate()
        assert not mgr.session_active
        # after invalidate no fresh nonce can be issued and verification stays dead
        with pytest.raises(RuntimeError):
            mgr.issue_nonce()

    def test_provided_token_is_used(self):
        token = generate_session_token()
        mgr = SessionTokenManager(scope=_make_scope(), token=token)
        assert mgr.token == token
        assert mgr.digest == token_digest(token)

    def test_invalid_attach_timeout_rejected(self):
        with pytest.raises(ValueError):
            SessionTokenManager(scope=_make_scope(), attach_timeout=0)
        with pytest.raises(ValueError):
            SessionTokenManager(scope=_make_scope(), attach_timeout=-1.0)

    def test_invalid_scope_type_rejected(self):
        with pytest.raises(TypeError):
            SessionTokenManager(scope={"job_id": "job-1234"})

    def test_non_ascii_proof_consumes_nonce_and_fails_without_raising(self):
        scope = _make_scope()
        mgr = SessionTokenManager(scope=scope)
        nonce = mgr.issue_nonce()
        # a non-ASCII proof must fail (not raise) and still consume the nonce
        assert mgr.verify_proof(nonce, scope, "deadbeefÿ") is False
        assert not mgr.session_active
        # the nonce was consumed; a now-valid proof over the same nonce must still fail
        proof = compute_hello_proof(mgr.token, nonce, scope)
        assert not mgr.verify_proof(nonce, scope, proof)

    def test_issue_nonce_refused_while_session_active(self):
        scope = _make_scope()
        mgr = SessionTokenManager(scope=scope)
        nonce = mgr.issue_nonce()
        assert mgr.verify_proof(nonce, scope, compute_hello_proof(mgr.token, nonce, scope))
        assert mgr.session_active
        with pytest.raises(RuntimeError):
            mgr.issue_nonce()

    def test_issue_nonce_refused_after_invalidate(self):
        mgr = SessionTokenManager(scope=_make_scope())
        mgr.invalidate()
        with pytest.raises(RuntimeError):
            mgr.issue_nonce()

    def test_issue_nonce_refused_after_expiry(self):
        clock = _FakeClock()
        mgr = SessionTokenManager(scope=_make_scope(), attach_timeout=30.0, clock=clock)
        clock.advance(30.1)
        with pytest.raises(RuntimeError):
            mgr.issue_nonce()

    def test_pending_nonces_are_bounded_oldest_evicted(self):
        scope = _make_scope()
        mgr = SessionTokenManager(scope=scope)
        cap = SessionTokenManager.MAX_PENDING_NONCES
        first = mgr.issue_nonce()
        # overflow the cap; the first (oldest) nonce must be evicted
        for _ in range(cap + 50):
            mgr.issue_nonce()
        # the pending set never exceeds the cap (a HELLO flood cannot grow memory unboundedly)
        assert len(mgr._pending_nonces) == cap
        # the evicted oldest nonce no longer verifies even with a correct proof
        assert not mgr.verify_proof(first, scope, compute_hello_proof(mgr.token, first, scope))

    def test_hello_flood_keeps_memory_bounded(self):
        mgr = SessionTokenManager(scope=_make_scope())
        cap = SessionTokenManager.MAX_PENDING_NONCES
        for _ in range(cap * 3):
            mgr.issue_nonce()
            assert len(mgr._pending_nonces) <= cap

    def test_nonce_cannot_be_redeemed_after_attach_window(self):
        clock = _FakeClock()
        scope = _make_scope()
        mgr = SessionTokenManager(scope=scope, attach_timeout=30.0, clock=clock)
        nonce = mgr.issue_nonce()
        proof = compute_hello_proof(mgr.token, nonce, scope)
        # advance past the attach window; the manager is expired so the nonce cannot be redeemed
        clock.advance(30.1)
        assert not mgr.verify_proof(nonce, scope, proof)

    def test_short_token_rejected(self):
        with pytest.raises(ValueError):
            SessionTokenManager(scope=_make_scope(), token="a" * (MIN_TOKEN_HEX_CHARS - 1))

    def test_minimum_length_token_accepted(self):
        token = "a" * MIN_TOKEN_HEX_CHARS
        mgr = SessionTokenManager(scope=_make_scope(), token=token)
        assert mgr.token == token

    def test_non_str_token_rejected(self):
        with pytest.raises(TypeError):
            SessionTokenManager(scope=_make_scope(), token=12345678901234567890123456789012)
