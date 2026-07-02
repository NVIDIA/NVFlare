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
"""Session-token auth primitives for the Client API control protocol.

Implements the attach auth contract of the Client API Execution Modes design
(docs/design/client_api_execution_modes.md, "Appendix B: Attach Topology and Auth"):

- The executor generates a high-entropy session token per attach id / launch. The raw token
  is held only in memory for the session lifetime; anything persisted stores only its digest.
- Token proof is challenge-response, not bearer presentation: HELLO -> HELLO_CHALLENGE (nonce)
  -> HELLO_PROOF (HMAC keyed by the token over the nonce and the full token scope). The raw
  token never crosses the wire, so observing attach traffic does not permit replay.
- The token is scoped to (job id, site name, attach id, target FQCN, trainer FQCN, rank
  policy, protocol version), is single-session by default, and expires if the trainer does
  not attach before attach_timeout.
- Two proof paths share the same HMAC primitive:
    * The interactive HELLO_CHALLENGE / HELLO_PROOF path goes through SessionTokenManager,
      which issues a self-generated single-use nonce (issue_nonce) and verifies the proof
      against it (verify_proof).
    * The one-round variant (Appendix B) folds the proof into HELLO, computed over an
      executor nonce delivered in the bootstrap config combined with a trainer-supplied
      nonce. This path does NOT go through SessionTokenManager.verify_proof (there is no
      self-issued nonce); it uses the module-level verify_hello_proof directly over a
      combined nonce (see combine_nonces).

This module is part of interface freeze #1. It is a pure library: no Cell/cellnet imports,
no file I/O, no logging side effects.
"""

import dataclasses
import hashlib
import hmac
import secrets
import threading
import time
from collections import OrderedDict
from typing import Callable, Optional

from .defs import PROTOCOL_VERSION

# Domain-separation tag for the HELLO proof HMAC, versioned with the proof format.
_PROOF_DOMAIN_TAG = "nvflare.client_api.hello_proof.v1"

# Default sizes (bytes of entropy) for generated secrets.
DEFAULT_TOKEN_BYTES = 32
DEFAULT_NONCE_BYTES = 16

# Minimum accepted length (hex chars) for a caller-supplied token: 16 bytes of entropy.
MIN_TOKEN_HEX_CHARS = 32


@dataclasses.dataclass(frozen=True)
class TokenScope:
    """The scope a session token is bound to.

    A token is valid only for this exact scope; the executor accepts an attach only when the
    proof, the presented scope, and its own expected scope all match.

    Attributes:
        job_id: id of the job the token was issued for.
        site_name: name of the site (FL client) the token was issued for.
        attach_id: attach/session id the token was issued for.
        target_fqcn: FQCN of the CJ/job cell the trainer must talk to.
        trainer_fqcn: FQCN the trainer cell must bind (routing name, not identity).
        rank_policy: allowed rank policy for the session (e.g. which rank may attach).
        protocol_version: Client API control protocol version.
    """

    job_id: str
    site_name: str
    attach_id: str
    target_fqcn: str
    trainer_fqcn: str
    rank_policy: str
    protocol_version: int = PROTOCOL_VERSION


def generate_session_token(num_bytes: int = DEFAULT_TOKEN_BYTES) -> str:
    """Generate a new high-entropy session token.

    Args:
        num_bytes: bytes of entropy in the token.

    Returns:
        The token as a hex string. The raw token must be held only in memory; anything
        persisted must store only its digest (see token_digest).
    """
    return secrets.token_hex(num_bytes)


def generate_nonce(num_bytes: int = DEFAULT_NONCE_BYTES) -> str:
    """Generate a new single-use challenge nonce as a hex string."""
    return secrets.token_hex(num_bytes)


def token_digest(token: str) -> str:
    """Compute the persistable digest (SHA-256 hex) of a session token."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _canonical_proof_message(nonce: str, scope: TokenScope) -> bytes:
    """Serialize (nonce + all scope fields) canonically, unambiguously, and type-unambiguously.

    Each item is netstring-encoded (length-prefixed), so no combination of field values can
    produce the same byte stream as a different combination (no concatenation ambiguity).
    Each scope field additionally carries an explicit type tag (its Python type name) and is
    rendered with repr(), so values that differ only by type -- e.g. protocol_version=1 (int)
    vs protocol_version="1" (str) -- serialize differently and cannot be substituted for one
    another (no cross-type proof forgery). Field order is fixed: domain tag, nonce, then
    TokenScope fields in declaration order.
    """
    items = [_PROOF_DOMAIN_TAG, nonce]
    for f in dataclasses.fields(scope):
        value = getattr(scope, f.name)
        items.append(f"{type(value).__name__}:{value!r}")

    buf = bytearray()
    for item in items:
        encoded = item.encode("utf-8")
        buf += f"{len(encoded)}:".encode("utf-8")
        buf += encoded
        buf += b","
    return bytes(buf)


def compute_hello_proof(token: str, nonce: str, scope: TokenScope) -> str:
    """Compute the HELLO_PROOF value: HMAC-SHA256 keyed by the token over (nonce, scope).

    The proof covers the challenge nonce and the full token scope (attach id, job id, site
    name, trainer FQCN, target FQCN, rank policy, protocol version), so a proof observed on
    the wire is useless for any other nonce, scope, or protocol version.

    Args:
        token: the raw session token (the HMAC key; never sent on the wire).
        nonce: the challenge nonce from HELLO_CHALLENGE.
        scope: the token scope the proof is computed over.

    Returns:
        The proof as a hex string.
    """
    return hmac.new(token.encode("utf-8"), _canonical_proof_message(nonce, scope), hashlib.sha256).hexdigest()


def verify_hello_proof(token: str, nonce: str, scope: TokenScope, proof: str) -> bool:
    """Verify a HELLO_PROOF value in constant time.

    Args:
        token: the raw session token.
        nonce: the challenge nonce the proof must cover.
        scope: the token scope the proof must cover.
        proof: the presented proof (hex string).

    Returns:
        True if the proof is valid for (token, nonce, scope). Never raises for a
        malformed/attacker-supplied proof: a non-str, empty, or non-ASCII proof returns False.
    """
    if not isinstance(proof, str) or not proof:
        return False
    try:
        expected = compute_hello_proof(token, nonce, scope)
        # Encode both operands to bytes before compare_digest: hmac.compare_digest raises
        # TypeError on a non-ASCII str, so a non-ASCII (attacker-supplied) proof would raise
        # instead of failing. Comparing bytes makes any malformed proof simply mismatch.
        return hmac.compare_digest(expected.encode("utf-8"), proof.encode("utf-8"))
    except (TypeError, ValueError):
        return False


def combine_nonces(executor_nonce: str, trainer_nonce: str) -> str:
    """Combine an executor nonce with a trainer nonce for the one-round HELLO proof variant.

    In the one-round variant (Appendix B), the executor delivers a nonce in the bootstrap
    config and the trainer contributes its own nonce; the proof is computed over both so
    neither side alone fixes the challenge. The combination is length-prefixed so the pair is
    unambiguous regardless of nonce contents. Both sides must combine identically, then feed
    the result as the ``nonce`` argument to compute_hello_proof / verify_hello_proof -- this
    path does not use SessionTokenManager (there is no self-issued nonce).
    """
    if not isinstance(executor_nonce, str) or not isinstance(trainer_nonce, str):
        raise TypeError("executor_nonce and trainer_nonce must both be str")
    return f"{len(executor_nonce)}:{executor_nonce}:{trainer_nonce}"


class SessionTokenManager:
    """Executor-side manager for one session token and its challenge-response verification.

    Holds the raw token in memory for the session lifetime (plus its digest for anything
    persisted), issues single-use challenge nonces, and verifies HELLO_PROOF messages while
    enforcing the auth contract:

    - a nonce is consumed by its first verification attempt (success or failure), so a
      captured proof cannot be replayed and a consumed nonce cannot be reused;
    - the presented scope must equal the expected scope exactly;
    - the token expires if no session is established within attach_timeout of creation;
    - the token is single-session: while a session is active, further attach attempts are
      rejected;
    - after invalidate(), all verification fails permanently (reconnect requires a fresh
      token through a new manager);
    - issue_nonce() refuses (raises RuntimeError) once the manager is invalidated, expired, or
      a session is active, and the set of outstanding pending nonces is bounded by
      MAX_PENDING_NONCES (oldest evicted first) so a HELLO flood cannot grow memory without
      bound.

    Binding the accepted session to the peer's FQCN (recording/enforcing the bound peer) is
    intentionally out of scope here and handled separately (see AT-2).

    The clock is injectable for testability; it must be a monotonic seconds counter.
    Thread-safe.
    """

    # Upper bound on outstanding (issued-but-unconsumed) nonces; oldest are evicted past this.
    MAX_PENDING_NONCES = 1024

    def __init__(
        self,
        scope: TokenScope,
        attach_timeout: Optional[float] = None,
        token: Optional[str] = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        """Create a manager for one session token.

        Args:
            scope: the scope the token is bound to.
            attach_timeout: seconds from creation within which the trainer must attach;
                None means no attach expiry.
            token: the raw token to manage; a new high-entropy token is generated if None.
            clock: monotonic seconds source (injectable for tests).
        """
        if attach_timeout is not None and attach_timeout <= 0:
            raise ValueError(f"attach_timeout must be positive but got {attach_timeout}")
        if not isinstance(scope, TokenScope):
            raise TypeError(f"scope must be TokenScope but got {type(scope)}")
        if token is not None:
            if not isinstance(token, str):
                raise TypeError(f"token must be str but got {type(token)}")
            if len(token) < MIN_TOKEN_HEX_CHARS:
                raise ValueError(
                    f"token must have at least {MIN_TOKEN_HEX_CHARS} chars "
                    f"(>= 16 bytes of entropy) but got {len(token)}"
                )

        self._scope = scope
        self._token = token if token else generate_session_token()
        self._digest = token_digest(self._token)
        self._attach_timeout = attach_timeout
        self._clock = clock
        self._created_at = clock()
        # Insertion-ordered map nonce -> issued_at, so the oldest pending nonce is evictable
        # in O(1) when the cap is exceeded, and each nonce carries an issuance time for TTL.
        self._pending_nonces = OrderedDict()
        self._session_active = False
        self._invalidated = False
        self._lock = threading.Lock()

    @property
    def token(self) -> str:
        """The raw session token (in-memory only; never persist or send on the wire)."""
        return self._token

    @property
    def digest(self) -> str:
        """The persistable SHA-256 hex digest of the token."""
        return self._digest

    @property
    def scope(self) -> TokenScope:
        """The scope this token is bound to."""
        return self._scope

    @property
    def session_active(self) -> bool:
        """Whether a session is currently bound to this token."""
        with self._lock:
            return self._session_active

    @property
    def invalidated(self) -> bool:
        """Whether this token has been explicitly invalidated."""
        with self._lock:
            return self._invalidated

    def is_expired(self) -> bool:
        """Whether the attach window has closed without a session being established."""
        with self._lock:
            return self._is_expired()

    def _is_expired(self) -> bool:
        if self._attach_timeout is None or self._session_active:
            return False
        return (self._clock() - self._created_at) > self._attach_timeout

    def issue_nonce(self) -> str:
        """Issue a fresh single-use challenge nonce for HELLO_CHALLENGE.

        Refuses to issue once the manager is no longer in the pre-attach state, so a stale or
        flooded caller cannot accumulate nonces after the session has been settled:

        Raises:
            RuntimeError: if the token has been invalidated, the attach window has expired, or
                a session is already active.

        The number of outstanding (issued-but-unconsumed) nonces is capped at
        MAX_PENDING_NONCES; issuing past the cap evicts the oldest pending nonce so a HELLO
        flood cannot grow memory without bound.
        """
        with self._lock:
            if self._invalidated:
                raise RuntimeError("cannot issue nonce: session token has been invalidated")
            if self._session_active:
                raise RuntimeError("cannot issue nonce: a session is already active")
            if self._is_expired():
                raise RuntimeError("cannot issue nonce: attach window has expired")
            nonce = generate_nonce()
            self._pending_nonces[nonce] = self._clock()
            # bound memory: evict oldest outstanding nonces once the cap is exceeded
            while len(self._pending_nonces) > self.MAX_PENDING_NONCES:
                self._pending_nonces.popitem(last=False)
            return nonce

    def verify_proof(self, nonce: str, scope: TokenScope, proof: str) -> bool:
        """Verify a HELLO_PROOF attach attempt; on success the session becomes active.

        The nonce is consumed by this call whether or not verification succeeds. The proof is
        verified against the manager's EXPECTED scope (self._scope), not the caller-supplied
        scope: a wrong presented scope therefore yields a proof mismatch in constant time
        (hmac.compare_digest) rather than leaking, via a short-circuiting tuple compare, which
        scope field mismatched -- and the accepted proof is bound to the scope the executor
        expects. (Recording/enforcing the bound peer FQCN is out of scope here; see AT-2.)

        Args:
            nonce: the challenge nonce the proof claims to answer; must have been issued by
                issue_nonce() and not yet consumed.
            scope: the scope presented by the attaching trainer.
            proof: the presented proof (hex string).

        Returns:
            True only if the nonce is valid and unconsumed, the token is not invalidated or
            expired, no session is already active, the proof verifies against the expected
            scope, and (redundantly) the presented scope equals the expected scope.
        """
        if not isinstance(nonce, str):
            return False
        with self._lock:
            # single-use: consumed by any verification attempt (success or failure)
            issued = self._pending_nonces.pop(nonce, None) is not None

            if self._invalidated:
                return False
            if not issued:
                return False
            # An old nonce cannot be redeemed after the attach window: _is_expired() covers
            # it (a nonce's issue time is >= the manager's creation time under a monotonic
            # clock, so the manager-level window always trips first) -- no separate per-nonce
            # TTL is needed while no session is active.
            if self._is_expired():
                return False
            if self._session_active:
                # single-session: reject further attach attempts while a session is active
                return False
            # constant-time proof check bound to the EXPECTED scope
            if not verify_hello_proof(self._token, nonce, self._scope, proof):
                return False
            # redundant defense-in-depth guard (the proof is already bound to self._scope)
            if scope != self._scope:
                return False

            self._session_active = True
            return True

    def invalidate(self) -> None:
        """Invalidate the token and session permanently; all future verification fails."""
        with self._lock:
            self._invalidated = True
            self._session_active = False
            self._pending_nonces.clear()
