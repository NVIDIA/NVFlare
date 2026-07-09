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
"""Normalized aggregate terminal outcome for DownloadService transactions.

TransactionDoneStatus.FINISHED only means a transaction reached its receiver count:
a receiver that FAILED still counts toward num_receivers, and the transaction_done_cb
carries no per-receiver outcomes. The Client API payload lifecycle
(docs/design/client_api_execution_modes.md, "Terminal transfer outcome") requires the
distinction between "every expected receiver succeeded" and "the transaction merely
terminated". This module provides that distinction additively: TransactionDoneStatus
values and the transaction_done_cb contract are unchanged.

Receiver truth wins over known termination mechanics: a transaction whose expected
receivers all succeeded resolves COMPLETED even if it was terminated by routine
cleanup (delete_transaction) or a late timeout. Everything else fails closed —
including a FINISHED transaction with no objects (a mid-assembly race must not
certify success) and any unknown/future termination status (validated before
receiver truth is considered).

Outcome status values reuse the TransferProgressState terminal vocabulary
(completed / failed / aborted) rather than introducing another status set.

Semantics with the full payload layer (receiver-confirmed completion, per-receiver
budgets, awaitable transfer facade):
- per-receiver statuses are receiver-confirmed where the receiver supports it (a served
  EOF is provisional until the receiver confirms its finalization succeeded); legacy
  receivers remain producer-served — both skews and the runtime kill-switch degrade to
  producer-served semantics (download_service.py, receiver-confirmed completion);
- expected receiver identities and per-receiver acquire/idle budgets bound the outcome's
  resolution time (a stalled or never-pulling receiver is finalized FAILED without
  waiting the whole-transaction TTL); min_receivers surfaces the optional k-of-N quorum
  via quorum_met while `completed` stays the strict all-receivers certificate;
- the outcome covers the refs present at termination; adding objects to a
  transaction after receivers already finished the earlier ones is not supported.
"""

import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional, Sequence, Tuple

from nvflare.fuel.f3.streaming.transfer_progress import TransferProgressState


class DownloadStatus:
    """Constants for object download status."""

    SUCCESS = "success"
    FAILED = "failed"


class TransactionDoneStatus:
    """Constants for transaction completion status."""

    FINISHED = "finished"
    TIMEOUT = "timeout"
    DELETED = "deleted"


def terminal_state_for_done_status(done_status: str) -> Optional[str]:
    """Map a TransactionDoneStatus value to the TransferProgressState terminal vocabulary."""
    if done_status == TransactionDoneStatus.FINISHED:
        return TransferProgressState.COMPLETED
    if done_status == TransactionDoneStatus.TIMEOUT:
        return TransferProgressState.FAILED
    if done_status == TransactionDoneStatus.DELETED:
        return TransferProgressState.ABORTED
    return None


class TransferOutcomeReason:
    """Constants explaining how a TransferOutcome status was determined."""

    ALL_RECEIVERS_SUCCEEDED = "all_receivers_succeeded"
    RECEIVER_FAILED = "receiver_failed"
    NO_OBJECTS = "no_objects"
    TIMEOUT = "timeout"
    DELETED = "deleted"
    UNKNOWN_RECEIVER_COUNT = "unknown_receiver_count"
    UNKNOWN_DONE_STATUS = "unknown_done_status"


@dataclass(frozen=True)
class RefOutcome:
    """Per-object terminal outcome.

    receiver_statuses maps receiver FQCN to a DownloadStatus value (success / failed) --
    receiver-confirmed where the receiver supports it, producer-served for legacy peers
    or when the receiver-confirm kill-switch is off, and budget-FAILED for receivers that
    exhausted their acquire/idle budget. It is deep-frozen at construction (a
    MappingProxyType over a private copy): the same instance is recorded in the
    service outcome table and handed to outcome_cb consumers, so a callback must
    not be able to mutate the recorded per-receiver truth. If outcomes ever cross
    a process boundary, the serializer must materialize it (dict(...)).
    """

    ref_id: str
    receiver_statuses: Mapping[str, str]

    def __post_init__(self):
        # frozen=True only blocks attribute rebinding; freeze the container too
        object.__setattr__(self, "receiver_statuses", MappingProxyType(dict(self.receiver_statuses)))


@dataclass(frozen=True)
class TransferOutcome:
    """Aggregate terminal outcome of one download transaction.

    status is a TransferProgressState terminal value: COMPLETED only when every
    expected receiver of every ref succeeded; FAILED on any receiver failure,
    missing receiver, no objects, timeout, or unknown receiver count; ABORTED on
    explicit deletion before full success. done_status carries the raw
    TransactionDoneStatus for callers that need the untranslated termination cause.
    """

    tx_id: str
    status: str  # a TransferProgressState terminal value
    reason: str  # a TransferOutcomeReason value
    done_status: str  # the raw TransactionDoneStatus value
    num_receivers: int
    refs: Tuple[RefOutcome, ...]
    timestamp: float
    # optional k-of-N quorum declared by the workflow: informational for quorum_met;
    # `completed` deliberately stays the strict all-receivers certificate
    min_receivers: Optional[int] = None
    # expected receiver identities declared by the workflow. When present, completion and
    # quorum are judged against THESE identities: a status from an unexpected receiver can
    # neither complete the transfer nor count toward the quorum.
    receiver_ids: Optional[Tuple[str, ...]] = None

    def __post_init__(self):
        # frozen=True only blocks attribute rebinding; freeze the containers too so
        # outcome_cb consumers cannot mutate the recorded outcome
        object.__setattr__(self, "refs", tuple(self.refs))
        if self.receiver_ids is not None:
            object.__setattr__(self, "receiver_ids", tuple(self.receiver_ids))

    @property
    def completed(self) -> bool:
        return self.status == TransferProgressState.COMPLETED

    @property
    def quorum_met(self) -> bool:
        """True if every ref reached at least min_receivers confirmed successes.

        The k-of-N surface for fan-out workflows (min_responses-style policies): `completed`
        stays the strict all-receivers certificate; a workflow that accepts partial fan-out
        checks quorum_met (or thresholds refs itself). Falls back to `completed` when no
        min_receivers was declared; fails closed with no refs.
        """
        if self.min_receivers is None:
            return self.completed
        if not self.refs:
            return False
        # a receiver counts toward the quorum only if it succeeded on EVERY ref: counting
        # per-ref successes independently would let different receiver subsets satisfy each
        # ref while no single receiver holds the complete payload
        quorum_receivers = None
        for r in self.refs:
            ref_successes = {rcv for rcv, v in r.receiver_statuses.items() if v == DownloadStatus.SUCCESS}
            quorum_receivers = ref_successes if quorum_receivers is None else quorum_receivers & ref_successes
        if self.receiver_ids is not None:
            # only declared receivers count toward the quorum
            quorum_receivers &= set(self.receiver_ids)
        return len(quorum_receivers) >= self.min_receivers

    def expired(self, now: float, ttl: float) -> bool:
        return now - self.timestamp > ttl


def _all_receivers_succeeded(num_receivers: int, refs: Sequence[RefOutcome], receiver_ids=None) -> bool:
    if num_receivers <= 0 or not refs:
        return False
    if receiver_ids:
        # identity mode: every DECLARED receiver must have succeeded on every ref.
        # Statuses from unexpected receivers are ignored -- they can never certify
        # a transfer that a declared receiver did not actually get.
        for r in refs:
            for expected in receiver_ids:
                if r.receiver_statuses.get(expected) != DownloadStatus.SUCCESS:
                    return False
        return True
    for r in refs:
        if len(r.receiver_statuses) < num_receivers:
            return False
        if any(s != DownloadStatus.SUCCESS for s in r.receiver_statuses.values()):
            return False
    return True


_KNOWN_DONE_STATUSES = (
    TransactionDoneStatus.FINISHED,
    TransactionDoneStatus.TIMEOUT,
    TransactionDoneStatus.DELETED,
)


def compute_transfer_outcome(
    tx_id: str,
    done_status: str,
    num_receivers: int,
    refs: Sequence[RefOutcome],
    timestamp: Optional[float] = None,
    min_receivers: Optional[int] = None,
    receiver_ids=None,
) -> TransferOutcome:
    """Compute the aggregate terminal outcome for a terminated transaction.

    The termination status is validated first: an unknown/future done_status fails
    closed even if every receiver succeeded. For known statuses, receiver truth
    wins: if every expected receiver of every ref succeeded, the outcome is
    COMPLETED regardless of how the transaction terminated (FINISHED, or routine
    cleanup via DELETED, or a late TIMEOUT). Otherwise the outcome fails closed
    based on the termination cause.

    Args:
        tx_id: ID of the terminated transaction.
        done_status: the TransactionDoneStatus value the transaction terminated with.
        num_receivers: the transaction's expected receiver count (0 means unknown).
        refs: per-ref receiver statuses snapshotted at termination.
        timestamp: termination time; defaults to now.

    Returns: a TransferOutcome.
    """
    if timestamp is None:
        timestamp = time.time()

    if done_status not in _KNOWN_DONE_STATUSES:
        status, reason = TransferProgressState.FAILED, TransferOutcomeReason.UNKNOWN_DONE_STATUS
    elif _all_receivers_succeeded(num_receivers, refs, receiver_ids):
        status, reason = TransferProgressState.COMPLETED, TransferOutcomeReason.ALL_RECEIVERS_SUCCEEDED
    elif done_status == TransactionDoneStatus.DELETED:
        status, reason = TransferProgressState.ABORTED, TransferOutcomeReason.DELETED
    elif done_status == TransactionDoneStatus.TIMEOUT:
        status, reason = TransferProgressState.FAILED, TransferOutcomeReason.TIMEOUT
    else:
        # FINISHED without full receiver success
        if num_receivers <= 0:
            # unknown receiver count: all-receivers-success can never be certified
            status, reason = TransferProgressState.FAILED, TransferOutcomeReason.UNKNOWN_RECEIVER_COUNT
        elif not refs:
            # a FINISHED transaction with no objects must not certify success
            status, reason = TransferProgressState.FAILED, TransferOutcomeReason.NO_OBJECTS
        else:
            status, reason = TransferProgressState.FAILED, TransferOutcomeReason.RECEIVER_FAILED

    return TransferOutcome(
        tx_id=tx_id,
        status=status,
        reason=reason,
        done_status=done_status,
        num_receivers=num_receivers,
        refs=refs,
        timestamp=timestamp,
        min_receivers=min_receivers,
        receiver_ids=receiver_ids,
    )
