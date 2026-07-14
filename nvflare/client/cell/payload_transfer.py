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

"""Payload seam of the Client API Cell protocol, shared by both sides of the wire.

Design: docs/design/client_api_execution_modes.md ("Payload Lifecycle State Machine").
Control messages (TASK_READY / RESULT_READY) carry only download ref ids; the bytes move
through explicit F3 DownloadService transactions (nvflare/fuel/f3/streaming) — never on
the control path. This module maps the protocol's task/result lifecycle onto those
primitives for its two symmetric consumers, each a producer in one direction and a
consumer in the other:

- the CJ backend (external_process_backend): TaskPayloadAttempt for task-down,
  fetch_result_payload for result-up;
- the trainer engine (nvflare/client/cell/api): fetch_result_payload for task-down,
  TaskPayloadAttempt for result-up.

Contract points consumed from the layer (see download_service.py):

- ``TransferOutcome.completed is True`` iff every declared receiver certified storage —
  the "returns == delivered" primitive. Acting on it (releasing the payload, stopping the
  producer) is safe; a control ack (TASK_ACCEPTED / RESULT_ACCEPTED) never is.
- ``waiter.wait`` never hangs but MAY return ``None`` (timeout or service shutdown); this
  module folds ``None`` into "not delivered".
- tx_ids are attempt-scoped and never reused: a retry is a NEW attempt (new
  ``TaskPayloadAttempt``) after the failed one was terminated. Cross-attempt identity
  (``transfer_id``) is minted and enforced by the caller, never by this layer.

Budget policy (the platform deliberately ships no defaults; sizing is this seam's job):

The one-shot ``ShareableDownloadable`` serves the whole Shareable in a single OK round —
but large tensors inside it do NOT ride that reply. FOBS encode hands them to the
via-downloader decomposers, which create INNER DownloadService transactions whose chunk
pulls move the actual bytes; those pulls run between this (outer) transaction's OK round
and its EOF round, so the outer transaction is legitimately inactive for the entire
tensor transfer, however long that takes. Two consequences:

- **No per-receiver idle budget.** A fixed idle wall around an unbounded healthy quiet
  period manufactures failures for large models (it would expire mid-transfer while the
  inner transactions are progressing normally). Liveness is enforced elsewhere: the
  acquire budget fails fast when no consumer ever shows up, the inner transactions carry
  their own progress-based inactivity timeouts (a wedged pull settles there and is
  confirmed back to this transaction as a receiver failure), and the attempt owner
  terminates the attempt on task end / abort / teardown.
- **TRANSFER_TTL is a leak backstop, not a liveness bound.** Owner-side ``terminate()``
  in the callers' finally/teardown paths is the primary lifetime bound; the TTL only
  reclaims an attempt whose owner failed to. It is sized so it cannot fire under a
  healthy large transfer on the V1 same-host deployment.

The endgame that removes the backstop entirely is decomposing tensors as refs on THIS
transaction (add_tensors/add_file, trainer-engine PR): every tensor chunk pull then
resets these clocks and the transfer becomes progress-bounded end to end.
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

# Whole-transaction inactivity backstop. NOT a liveness bound (see the module docstring):
# it must exceed the worst-case HEALTHY quiet period of the outer transaction, which spans
# the entire inner (via-downloader) tensor transfer. The owner's terminate() bounds real
# lifetimes long before this fires.
TRANSFER_TTL = 3600.0
# Per-receiver acquire budget: how long the consumer has to issue its first pull after the
# control message. Both directions pull eagerly on receipt (the trainer engine materializes
# the task payload as soon as TASK_READY arrives; the CJ pulls as soon as RESULT_READY is
# accepted), so a short window fails fast when no consumer ever shows up.
TASK_ACQUIRE_TIMEOUT = 60.0
# Consumer-side pull: timeout per chunk request (chunk-level retry with backoff is built
# into download_object).
RESULT_PULL_PER_REQUEST_TIMEOUT = 60.0


class PayloadTransferError(Exception):
    """A payload transfer failed terminally (attempt failed, pull failed, or bad payload)."""


class ShareableDownloadable(Downloadable):
    """One-shot Downloadable for a task/result Shareable.

    V1 payloads are one lifecycle unit of one object: produce() hands the whole object in
    a single OK round, then answers EOF. Tensors inside the object move through the inner
    via-downloader transactions created at encode time (see the module docstring).
    """

    _DONE_STATE = {"done": True}

    def produce(self, state: dict, requester: str) -> Tuple[str, Any, dict]:
        if state and state.get("done"):
            return ProduceRC.EOF, None, {}
        return ProduceRC.OK, self.base_obj, dict(self._DONE_STATE)

    def release(self):
        # let the GC reclaim the payload as soon as the attempt settles
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
    """One producer-side transfer attempt of a payload Shareable to one receiver.

    - one attempt == one ObjectDownloader == one fresh attempt-scoped tx_id; a retry is a
      NEW TaskPayloadAttempt created by the caller after terminate() on this one — one
      live attempt per logical transfer is the CALLER's contract, enforced there.
    - the declared receiver identity (``receiver_ids=(receiver_fqcn,)``) enables the
      acquire budget and identity-checked completion certification.
    - ``completed()``/``failed()`` are non-blocking reads of the transaction's terminal
      outcome (through its TransferWaiter), so the backend's wait loop can poll them
      alongside process death and abort without blocking.
    - ``wait()`` is the event-driven verdict: it never hangs, and folds the waiter's
      ``None`` arm (timeout or service shutdown) into "not delivered".
    """

    def __init__(self, cell, obj: Any, receiver_fqcn: str):
        self.logger = get_obj_logger(self)
        self.receiver_fqcn = receiver_fqcn
        self._downloader = ObjectDownloader(
            cell=cell,
            timeout=TRANSFER_TTL,
            num_receivers=1,
            receiver_ids=(receiver_fqcn,),
            receiver_acquire_timeout=TASK_ACQUIRE_TIMEOUT,
            # receiver_idle_timeout deliberately unset: the healthy quiet period of the
            # one-shot round spans the whole inner tensor transfer (module docstring)
        )
        self.tx_id = self._downloader.tx_id
        self._waiter = self._downloader.get_waiter()
        try:
            self.ref_id = self._downloader.add_object(ShareableDownloadable(obj))
        except BaseException:
            # the transaction is already registered with the service; a failing add_object
            # must not leak it until the TTL reclaims it
            self.terminate()
            raise

    def completed(self) -> bool:
        outcome = self._waiter.outcome
        return outcome is not None and bool(outcome.completed)

    def failed(self) -> bool:
        return self._waiter.done() and not self.completed()

    def failure_reason(self) -> Optional[str]:
        if not self._waiter.done():
            return None
        outcome = self._waiter.outcome
        if outcome is None:
            # terminally resolved with nothing recorded: service shutdown
            return "payload transfer failed: service shut down before the attempt settled"
        if outcome.completed:
            return None
        return f"payload transfer failed: {getattr(outcome, 'reason', None) or 'receiver did not certify storage'}"

    def wait(self, timeout: Optional[float], linger: Optional[float] = None) -> bool:
        """Awaits the attempt verdict. True iff every declared receiver certified storage."""
        outcome = self._waiter.wait(timeout=timeout, linger=linger)
        if outcome is None:
            # waiter None arm: timeout or service shutdown — never certified
            self.logger.warning(f"payload attempt {self.tx_id}: waiter resolved None (timeout or shutdown)")
            return False
        return bool(outcome.completed)

    def terminate(self):
        """Terminates the attempt if it is still live (idempotent).

        Before retrying (or abandoning) a logical transfer, the caller must terminate the
        failed attempt — DownloadService.delete_transaction is a no-op for an
        already-settled attempt.
        """
        try:
            DownloadService.delete_transaction(self.tx_id)
        except Exception as e:
            # settlement is exception-proof on the service side; never let cleanup
            # propagate into the caller's task path
            self.logger.error(f"failed to terminate payload attempt {self.tx_id}: {e}")


def fetch_result_payload(
    cell,
    from_fqcn: str,
    ref_ids: List[str],
    abort_signal: Optional[Signal] = None,
) -> List[Any]:
    """Pulls payload objects from the producer side (the consumer side of the seam).

    ``download_object`` has chunk-level retry with backoff built in, and a failure in the
    consumer's ``download_completed`` is confirmed to the producer as FAILED — that is the
    receiver-truth mechanism, so no extra acknowledgment is sent.

    Args:
        cell: this side's cell.
        from_fqcn: the producer cell FQCN.
        ref_ids: the download ref ids from the control-message manifest.
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
                f"failed to pull payload ref {ref_id} from {from_fqcn}: "
                f"{consumer.failure_reason or 'download did not complete'}"
            )
        results.append(consumer.obj)
    return results
