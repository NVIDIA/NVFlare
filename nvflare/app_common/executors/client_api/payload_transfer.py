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

"""Payload-transfer seam for the Client API Cell backends.

============================ CONTRACT SEAM — READ FIRST ============================
This module is the ONLY place the Client API backends touch payload transfer, and it
is coded STRICTLY against docs/design/f3_backend_interface_contract.md — the interface
contract of the F3 payload layer rework (PR #4865: attempt-scoped tx_ids, TransferWaiter,
per-receiver budgets, receiver-confirmed outcomes; surface cross-checked against the PR
head on 2026-07-09). That PR is NOT yet merged. What main already has: outcome_cb and the
frozen TransferOutcome (`.completed` is the strict all-receivers certificate). What #4865
adds — and what this seam therefore gates behind payload_layer_available(), raising
PayloadLayerUnavailable until it lands — is the producer-side identity/budget surface
(receiver_ids / min_receivers / receiver_acquire_timeout / receiver_idle_timeout) and the
awaitable TransferWaiter facade (`downloader.get_waiter()` /
`DownloadService.get_transfer_waiter(tx_id)`; the contract doc names the accessor
`get_transfer_waiter()` on the downloader — the PR spells it `get_waiter()`, and this seam
follows the code). Nothing here reaches into nvflare/fuel/f3/streaming internals: only the
contract's public primitives (ObjectDownloader, TransferWaiter,
DownloadService.delete_transaction, download_object, Downloadable, Consumer) are used.
Anything not in the contract is implemented here as backend policy.
====================================================================================

Contract obligations honored here (see the contract doc for the full text):

- ``outcome.completed is True`` iff every declared receiver certified storage — the
  "returns == delivered" primitive. Acting on it (releasing the task payload, stopping
  the producer) is safe.
- ``waiter.wait`` never hangs but MAY return ``None`` (timeout or service shutdown).
  Every caller of :meth:`TaskPayloadAttempt.wait` must handle the ``None`` arm; this
  module treats ``None`` as "not delivered".
- tx_ids are attempt-scoped and never reused: a retry is a NEW attempt (new
  ``TaskPayloadAttempt``) after the failed one was terminated. Cross-attempt identity
  (``transfer_id``) is minted and enforced by the backend, never by this layer.
"""

from typing import Any, List, Optional, Tuple

from nvflare.apis.signal import Signal
from nvflare.fuel.f3.streaming.download_service import (
    Consumer,
    Downloadable,
    DownloadService,
    ProduceRC,
    download_object,
)
from nvflare.fuel.f3.streaming.obj_downloader import ObjectDownloader
from nvflare.fuel.utils.log_utils import get_obj_logger

PAYLOAD_LAYER_HINT = (
    "the F3 payload layer rework (PR #4865: attempt-scoped tx_ids, TransferWaiter, per-receiver "
    "budgets) is required for external_process payload transfer but is not available in this build"
)

# --- Backend transfer policy (contract: "Size budgets to the payload; the platform
# deliberately ships no default values" — budget policy is the BACKEND's, so these are
# named backend constants, not platform defaults). All values activate with #4865. ---

# Whole-attempt inactivity clock (required by the contract): resets on any transfer
# activity, so it bounds a silent attempt, not the total transfer duration.
TASK_TRANSFER_INACTIVITY_TIMEOUT = 600.0
# Per-receiver acquire budget: how long the trainer has to issue its first pull after
# TASK_READY. The trainer pulls immediately on receipt (before training), so a short
# window fails fast when no consumer ever shows up.
TASK_ACQUIRE_TIMEOUT = 60.0
# Per-receiver idle budget: progress timeout once the trainer's pull has started.
TASK_IDLE_TIMEOUT = 300.0
# Result-up pull: timeout per chunk request (chunk-level retry with backoff is built into
# download_object per the contract).
RESULT_PULL_PER_REQUEST_TIMEOUT = 60.0


class PayloadTransferError(Exception):
    """A payload transfer failed terminally (attempt failed, pull failed, or bad payload)."""


class PayloadLayerUnavailable(PayloadTransferError):
    """The #4865 payload-layer primitives this seam is coded against are not in this build."""


def payload_layer_available() -> bool:
    """True when the #4865 contract surface (TransferWaiter et al.) is present.

    Detection is by the contract's own vocabulary (the transfer-waiter accessor), not by
    streaming internals; when #4865 merges this returns True with no change here.
    """
    return hasattr(DownloadService, "get_transfer_waiter")


class ShareableDownloadable(Downloadable):
    """One-shot Downloadable for a task Shareable.

    V1 task payloads are one lifecycle unit of one object: produce() hands the whole
    object in a single OK round, then answers EOF. Chunked/tensor payloads arrive with
    the trainer-engine PR via the contract's add_file/add_tensors and need no change to
    the attempt lifecycle below.
    """

    _DONE_STATE = {"done": True}

    def produce(self, state: dict, requester: str) -> Tuple[str, Any, dict]:
        if state and state.get("done"):
            return ProduceRC.EOF, None, {}
        return ProduceRC.OK, self.base_obj, dict(self._DONE_STATE)

    def release(self):
        # let the GC reclaim the task payload as soon as the attempt settles
        self.base_obj = None


class _ObjectConsumer(Consumer):
    """Collects a one-shot object pull (the receive side of ShareableDownloadable)."""

    def __init__(self):
        super().__init__()
        self.obj: Any = None
        self.completed = False
        self.failure_reason: Optional[str] = None

    def consume(self, ref_id: str, state: dict, data: Any) -> dict:
        self.obj = data
        return state if isinstance(state, dict) else {}

    def download_completed(self, ref_id: str):
        self.completed = True

    def download_failed(self, ref_id: str, reason: str):
        self.failure_reason = reason


class TaskPayloadAttempt:
    """One producer-side transfer attempt of a task payload to one trainer.

    Contract mapping (f3_backend_interface_contract.md):

    - one attempt == one ObjectDownloader == one fresh attempt-scoped tx_id; a retry is a
      NEW TaskPayloadAttempt created by the backend after terminate() on this one — one
      live attempt per logical transfer is the BACKEND's contract, enforced there.
    - the declared receiver identity (``receiver_ids=(trainer_fqcn,)``) enables the acquire
      budget and identity-checked completion.
    - ``completed()``/``failed()`` are latched from the exception-isolated ``outcome_cb`` at
      settlement, so the backend's wait loop can poll them without blocking on the waiter.
    - ``wait()`` is the event-driven verdict: it never hangs, and returns ``None`` on
      timeout or service shutdown — callers MUST treat ``None`` as not-delivered.
    """

    def __init__(self, cell, obj: Any, receiver_fqcn: str):
        if not payload_layer_available():
            raise PayloadLayerUnavailable(PAYLOAD_LAYER_HINT)
        self.logger = get_obj_logger(self)
        self.receiver_fqcn = receiver_fqcn
        self._outcome = None
        self._downloader = ObjectDownloader(
            cell=cell,
            timeout=TASK_TRANSFER_INACTIVITY_TIMEOUT,
            num_receivers=1,
            receiver_ids=(receiver_fqcn,),
            receiver_acquire_timeout=TASK_ACQUIRE_TIMEOUT,
            receiver_idle_timeout=TASK_IDLE_TIMEOUT,
            outcome_cb=self._on_outcome,
        )
        self.tx_id = self._downloader.tx_id
        self.ref_id = self._downloader.add_object(ShareableDownloadable(obj))

    def _on_outcome(self, outcome):
        # contract: outcome_cb is exception-isolated and fires after settlement; the
        # outcome is deep-frozen, so holding it is safe
        self._outcome = outcome

    def completed(self) -> bool:
        outcome = self._outcome
        return outcome is not None and bool(outcome.completed)

    def failed(self) -> bool:
        outcome = self._outcome
        return outcome is not None and not outcome.completed

    def failure_reason(self) -> Optional[str]:
        outcome = self._outcome
        if outcome is None or outcome.completed:
            return None
        return f"task payload transfer failed: {getattr(outcome, 'reason', None) or 'receiver did not certify storage'}"

    def wait(self, timeout: Optional[float], linger: Optional[float] = None) -> bool:
        """Awaits the attempt verdict. True iff every declared receiver certified storage.

        The ``None`` arm of the contract (`waiter.wait` resolves ``None`` on timeout or
        service shutdown) is folded into False here: not-certified == not-delivered.
        """
        waiter = self._get_waiter()
        outcome = waiter.wait(timeout=timeout, linger=linger)
        if outcome is None:
            # contract None arm: timeout or service shutdown — never certified
            self.logger.warning(f"task payload attempt {self.tx_id}: waiter resolved None (timeout or shutdown)")
            return False
        return bool(outcome.completed)

    def _get_waiter(self):
        # #4865 spells the accessor downloader.get_waiter(); the DownloadService accessor is
        # the equivalent fallback should the merged shape keep only one of the two
        getter = getattr(self._downloader, "get_waiter", None)
        if getter is not None:
            return getter()
        return DownloadService.get_transfer_waiter(self.tx_id)

    def terminate(self):
        """Terminates the attempt if it is still live (idempotent).

        Contract: before retrying (or abandoning) a logical transfer, the backend must
        terminate the failed attempt — DownloadService.delete_transaction is the
        contract-listed termination and is a no-op for an already-settled attempt.
        """
        try:
            DownloadService.delete_transaction(self.tx_id)
        except Exception as e:
            # settlement is exception-proof on the service side; never let cleanup
            # propagate into the backend's task path
            self.logger.error(f"failed to terminate task payload attempt {self.tx_id}: {e}")


def fetch_result_payload(
    cell,
    from_fqcn: str,
    ref_ids: List[str],
    abort_signal: Optional[Signal] = None,
) -> List[Any]:
    """Pulls result payload objects from the trainer-side producer (consumer side).

    Contract: ``download_object`` has chunk-level retry with backoff built in, and a
    failure in the consumer's ``download_completed`` is confirmed to the producer as
    FAILED — that is the receiver-truth mechanism, so no extra acknowledgment is sent.

    Args:
        cell: the CJ cell.
        from_fqcn: the trainer cell FQCN (the producer of the result attempt).
        ref_ids: the download ref ids from the RESULT_READY manifest.
        abort_signal: optional abort signal observed between chunk requests.

    Returns:
        The pulled objects, in ref_ids order.

    Raises:
        PayloadTransferError: if any pull fails terminally.
    """
    results = []
    for ref_id in ref_ids:
        consumer = _ObjectConsumer()
        download_object(
            from_fqcn=from_fqcn,
            ref_id=ref_id,
            per_request_timeout=RESULT_PULL_PER_REQUEST_TIMEOUT,
            cell=cell,
            consumer=consumer,
            abort_signal=abort_signal,
        )
        if not consumer.completed:
            raise PayloadTransferError(
                f"failed to pull result payload ref {ref_id} from {from_fqcn}: "
                f"{consumer.failure_reason or 'download did not complete'}"
            )
        results.append(consumer.obj)
    return results
