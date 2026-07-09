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

Known limits, resolved by later PRs of the same design (see
docs/design/client_api_execution_modes_plan.md):
- per-receiver statuses are producer-served (recorded when produce() returns EOF),
  not receiver-confirmed — a receiver-side finalization failure after the last chunk
  is not visible here until receiver-confirmed completion (plan PR F3-2) lands;
- num_receivers is a count without receiver identity — expected-receiver identity
  checks arrive with per-receiver budgets (plan PR F3-3);
- the outcome covers the refs present at termination; adding objects to a
  transaction after receivers already finished the earlier ones is not supported.
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

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

    receiver_statuses maps receiver FQCN to a DownloadStatus value (success / failed)
    as recorded by the producer side. Treat the dict as read-only: the same instance
    is shared with every consumer of the outcome.
    """

    ref_id: str
    receiver_statuses: Dict[str, str]


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
    refs: List[RefOutcome]
    timestamp: float

    @property
    def completed(self) -> bool:
        return self.status == TransferProgressState.COMPLETED

    def expired(self, now: float, ttl: float) -> bool:
        return now - self.timestamp > ttl


def _all_receivers_succeeded(num_receivers: int, refs: List[RefOutcome]) -> bool:
    if num_receivers <= 0 or not refs:
        return False
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
    refs: List[RefOutcome],
    timestamp: Optional[float] = None,
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
    elif _all_receivers_succeeded(num_receivers, refs):
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
    )
