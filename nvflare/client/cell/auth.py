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

Implements the token/proof primitives of the Client API Execution Modes design
(docs/design/client_api_execution_modes.md, "Control Protocol" HELLO handshake and
"Appendix B: Attach Topology and Auth"):

- Token/nonce generation and the persistable digest: the raw token is held only in memory;
  anything persisted stores only its digest (token_digest).
- The HELLO proof is challenge-response, not bearer presentation: the proof is an HMAC keyed
  by the token over a challenge nonce and the full token scope, so the raw token never crosses
  the wire and an observed proof is useless for any other nonce/scope (compute_hello_proof /
  verify_hello_proof).
- TokenScope binds a token to (job id, site name, attach id, target FQCN, trainer FQCN, rank
  policy, protocol version).
- One-round variant (Appendix B, for local/confidential channels such as an external_process
  trainer on localhost): the proof is computed over an executor nonce combined with a
  trainer-supplied nonce (combine_nonces) and verified with the module-level verify_hello_proof.

Scope note: this module is the shared, stateless proof toolkit consumed by external_process
(EP-3) and attach (AT-2). The stateful, executor-side session manager -- single-use nonce
issuance, attach-window expiry, single-session enforcement, invalidation -- is an attach-mode
requirement (NVFlare does not own the trainer process there) and lands with the attach backend
(AT-2), not here. external_process launches the trainer itself on localhost and needs only the
lightweight launch-token proof these functions provide.

This module is part of interface freeze #1. It is a pure library: no Cell/cellnet imports,
no file I/O, no logging side effects.
"""

import dataclasses
import hashlib
import hmac
import secrets

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
