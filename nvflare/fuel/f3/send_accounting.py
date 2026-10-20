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

"""Process-local accounting context for one logical F3 send.

The context deliberately contains no wire representation.  A trusted caller
attaches it to a :class:`Message`, and the F3 send path carries the same Python
object through explicit local clones.  Serialization still sees only the
message headers and payload.

The accounting object is structural so this package does not depend on the
private resource-statistics implementation.  Its traffic-class value is also
opaque here: classification belongs to the trusted semantic caller.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Callable, Collection, Hashable, Optional, Protocol

log = logging.getLogger(__name__)
_CALL_FAILED = object()
SOURCE_CONTRIBUTION_CONTEXT_KEY = "source_contribution_context"
MAX_SOURCE_CONTRIBUTION_GROUPS = 256
# One logical send/result keeps at most this many small chunk identities and
# delegated-consumer mappings. Saturation stops accounting, never delivery.
MAX_ACCOUNTING_ENTRIES = 16_384
_MAX_CONSUMER_NAME_LENGTH = 512
_MAX_PAYLOAD_BYTES = (1 << 128) - 1


class LogicalSendAccounting(Protocol):
    """Callback surface supplied by the job-scoped accounting owner."""

    def try_begin(self, traffic_class: object, recipient_name: str) -> Optional[object]:
        pass

    def complete_remote_accepted(self, admission: object, payload_bytes: int) -> bool:
        pass

    def add_accepted_payload_bytes(self, admission: object, payload_bytes: int) -> bool:
        pass

    def abandon(self, admission: object) -> bool:
        pass

    def mark_incomplete(self, admission: object) -> bool:
        pass

    def mark_counter_gap(self) -> bool:
        pass


@dataclass
class _DestinationState:
    owner_admission_attempted: bool = False
    begun: bool = False
    admission: Optional[object] = None
    main_accepted: bool = False
    main_payload_bytes: int = 0
    completed: bool = False
    abandoned: bool = False
    incomplete: bool = False
    pending_transactions: set[str] = field(default_factory=set)
    successful_transactions: set[str] = field(default_factory=set)
    inflight_contributions: set[Hashable] = field(default_factory=set)
    accepted_contributions: set[Hashable] = field(default_factory=set)
    contribution_lengths: dict[Hashable, int] = field(default_factory=dict)
    queued_contributions: list[tuple[Hashable, int]] = field(default_factory=list)
    pending_external_contributions: set[Hashable] = field(default_factory=set)


class _SendAttempt:
    """One transport attempt admitted by a process-local context."""

    def __init__(self, accepted_cb, abandoned_cb):
        self._accepted_cb = accepted_cb
        self._abandoned_cb = abandoned_cb
        self._lock = threading.Lock()
        self._finished = False

    def accepted(self, payload_bytes: int) -> None:
        with self._lock:
            if self._finished:
                return
            self._finished = True
        self._accepted_cb(payload_bytes)

    def abandon(self) -> None:
        with self._lock:
            if self._finished:
                return
            self._finished = True
        self._abandoned_cb()


class _OobContributionContext:
    """A byte-only DownloadService response tied to a parent logical send."""

    def __init__(
        self,
        parent: "LogicalSendContext",
        destination: str,
        contribution_id: Hashable,
        payload_bytes: int,
        actual_destination: Optional[str] = None,
    ):
        self._parent = parent
        self._destination = destination
        self._contribution_id = contribution_id
        self._payload_bytes = payload_bytes
        self._actual_destination = actual_destination or destination

    def is_origin(self, origin: str) -> bool:
        return self._parent.is_origin(origin)

    def try_begin(self, origin: str, destination: str) -> Optional[_SendAttempt]:
        if destination != self._actual_destination:
            return None
        return self._parent._try_begin_contribution(
            origin=origin,
            destination=self._destination,
            contribution_id=self._contribution_id,
            payload_bytes=self._payload_bytes,
        )

    def mark_counter_gap(self) -> None:
        self._parent.mark_counter_gap()


class LogicalSendContext:
    """Origin-owned state for one logical message to one final destination."""

    def __init__(self, accounting: LogicalSendAccounting, traffic_class: object, recipient_name: str):
        self._accounting = accounting
        self._traffic_class = traffic_class
        self._recipient_name = recipient_name
        self._origin: Optional[str] = None
        self._destination: Optional[str] = None
        self._lock = threading.RLock()
        self._destinations: dict[str, _DestinationState] = {}
        self._transaction_receivers: dict[str, Optional[frozenset[str]]] = {}
        self._declared_transaction_receivers: dict[str, Optional[frozenset[str]]] = {}
        self._transaction_results: dict[str, frozenset[str]] = {}
        self._consumer_resolver: Optional[Callable] = None
        self._consumer_destinations: dict[tuple[str, str], str] = {}
        self._external_source_handler: Optional[Callable] = None
        self._external_sources: set[tuple[str, str]] = set()
        self._external_results: dict[tuple[Hashable, str], tuple[int, bool]] = {}
        self._metadata_count = 0
        self._metadata_saturated = False

    def set_consumer_resolver(self, resolver: Callable) -> None:
        """Set a trusted consumer-to-logical-destination resolver, not a lifetime rule.

        The callback receives the actual Cell identity and its authenticated
        request. It must validate job/site membership itself; this layer never
        treats a matching string prefix as authorization. Descendant consumers
        are revalidated on every pull; a cached destination is not cached proof.
        """

        with self._lock:
            self._consumer_resolver = resolver

    def resolve_oob_consumer(self, transaction_id: str, consumer: str, request) -> Optional[str]:
        """Recheck each mapped consumer without changing download admission."""

        with self._lock:
            key = (transaction_id, consumer)
            receivers = self._transaction_receivers.get(transaction_id)
            destination = self._destination
            state = self._destinations.get(destination)
            if self._metadata_saturated or (
                state is not None and (state.incomplete or state.abandoned or state.completed)
            ):
                return None
            resolver = self._consumer_resolver
        resolved = consumer if receivers is not None and consumer in receivers else None
        if resolved is None and receivers is not None and resolver is not None:
            try:
                resolved = resolver(consumer, request)
            except BaseException:
                # The callback may inspect credentials. Never log its exception
                # text, which could contain that process-local proof.
                log.warning("logical-send consumer resolution failed")
        valid = (
            isinstance(resolved, str)
            and receivers is not None
            and resolved in receivers
            and (destination is None or resolved == destination)
        )
        with self._lock:
            cached = self._consumer_destinations.get(key)
            state = self._destinations.get(destination)
            if (
                not valid
                or len(consumer) > _MAX_CONSUMER_NAME_LENGTH
                or (cached is not None and cached != resolved)
                or self._metadata_saturated
                or (state is not None and (state.incomplete or state.abandoned or state.completed))
            ):
                self._consumer_destinations.pop(key, None)
                rejected = True
            elif consumer != resolved and cached is None:
                if self._metadata_count >= MAX_ACCOUNTING_ENTRIES:
                    self._metadata_saturated = True
                    rejected = True
                else:
                    self._consumer_destinations[key] = resolved
                    self._metadata_count += 1
                    rejected = False
            else:
                rejected = False
        if rejected:
            self.mark_counter_gap()
            if destination is not None:
                # DownloadService continues serving regardless of this result.
                # Fail this observation now, including on an EOF pull: another
                # concurrent valid pull must not restore proof for this request.
                self._mark_incomplete(destination)
            return None
        return resolved

    def set_external_source_handler(self, handler: Callable) -> None:
        """Install an optional local adapter for a source owned by another process."""

        with self._lock:
            self._external_source_handler = handler

    def register_external_source(self, source_fqcn: str, source_ref: str) -> None:
        key = (source_fqcn, source_ref)
        with self._lock:
            if key in self._external_sources:
                return
            self._external_sources.add(key)
            handler = self._external_source_handler
        try:
            if handler is None or handler(self, source_fqcn, source_ref) is False:
                self.mark_counter_gap()
        except BaseException as ex:
            log.warning("external source registration failed: %s", ex)
            self.mark_counter_gap()

    def register_external_contribution(self, key: Hashable, destination: str) -> bool:
        with self._lock:
            state = self._destinations.setdefault(destination, _DestinationState())
            if state.begun or state.completed or state.abandoned or state.incomplete:
                self.mark_counter_gap()
                return False
            state.pending_external_contributions.add(key)
        return True

    def settle_external_contribution(self, key: Hashable, destination: str, payload_bytes: int, complete: bool) -> None:
        valid = (
            isinstance(payload_bytes, int)
            and not isinstance(payload_bytes, bool)
            and 0 <= payload_bytes <= _MAX_PAYLOAD_BYTES
            and isinstance(complete, bool)
        )
        result = (payload_bytes, complete)
        with self._lock:
            result_key = (key, destination)
            previous = self._external_results.get(result_key)
            if previous is not None:
                if not valid or previous != result:
                    self._mark_incomplete(destination)
                return
            state = self._destinations.setdefault(destination, _DestinationState())
            if key not in state.pending_external_contributions:
                self._mark_incomplete(destination)
                return
            if state.completed or state.abandoned or state.incomplete:
                state.pending_external_contributions.discard(key)
                return
            if not valid or not complete:
                self._mark_incomplete(destination)
                state.pending_external_contributions.discard(key)
                return
            contribution_id = ("external", key)
            if not self._reserve_contribution_locked(destination, state, contribution_id, payload_bytes):
                state.pending_external_contributions.discard(key)
                return
            self._external_results[result_key] = result
            if state.admission is None:
                state.queued_contributions.append((contribution_id, payload_bytes))
                state.inflight_contributions.add(contribution_id)
                state.pending_external_contributions.discard(key)
                return
        added = self._add_payload_bytes(destination, payload_bytes)
        with self._lock:
            state.pending_external_contributions.discard(key)
        if added:
            self._complete_if_ready(destination)

    def is_origin(self, origin: str) -> bool:
        """Whether ``origin`` may originate this still-local context."""

        with self._lock:
            return self._origin is None or self._origin == origin

    def prepare_destination(self, origin: str, destination: str) -> bool:
        """Bind accounting before encoding without changing download lifetime."""

        with self._lock:
            if not origin or not destination or self._origin not in {None, origin}:
                self.mark_counter_gap()
                return False
            if self._destination not in {None, destination}:
                self.mark_counter_gap()
                return False
            self._origin = origin
            self._destination = destination
        return True

    def mark_counter_gap(self) -> None:
        """Record a pre-admission instrumentation failure without raising."""

        self._safe_call("mark_counter_gap")

    def pre_admit(self, origin: str, destination: str) -> bool:
        """Admit one known destination before a semantic callback returns.

        A task-response callback returns a ``Message`` to CellNet, which starts
        the actual transport only after the callback has returned.  Pre-admitting
        that one known destination closes the small teardown race between those
        two stages.  It does not count anything: the transport still supplies
        the post-FOBS size and terminal acceptance outcome later.

        The transport ``begun`` flag intentionally remains false here.  FOBS may
        therefore register out-of-band transactions between this method and
        :meth:`try_begin`, exactly as it does on the ordinary lazy path.
        """

        if not origin or not destination:
            self.mark_counter_gap()
            return False

        with self._lock:
            if self._origin is None:
                self._origin = origin
            elif self._origin != origin:
                self.mark_counter_gap()
                return False
            if self._destination is None:
                self._destination = destination
            elif self._destination != destination:
                self.mark_counter_gap()
                return False

            state = self._destinations.setdefault(destination, _DestinationState())
            if state.owner_admission_attempted:
                if state.admission is None:
                    self.mark_counter_gap()
                    return False
                return True
            if state.begun:
                self.mark_counter_gap()
                return False
            state.owner_admission_attempted = True

        admission = self._safe_call("try_begin", self._traffic_class, self._recipient_name, failed_value=_CALL_FAILED)
        if admission is _CALL_FAILED or admission is None:
            self.mark_counter_gap()
            return False

        with self._lock:
            state.admission = admission
        return True

    def abandon_before_send(self, origin: str, destination: str) -> bool:
        """Release a matching pre-admission when transport will not begin.

        Do not create an observation: a send with no pre-admission must
        leave even a closed or frozen counter unchanged. The real stream and
        DownloadService transactions continue using their existing lifecycle.
        """

        with self._lock:
            state = self._destinations.get(destination)
            if (
                self._origin != origin
                or self._destination != destination
                or state is None
                or state.admission is None
                or state.begun
                or state.completed
                or state.abandoned
                or state.incomplete
            ):
                return False
            state.abandoned = True
            admission = state.admission
        return bool(self._safe_call("abandon", admission))

    def abandon_excluded_delivery(self, origin: str, destination: str) -> bool:
        """Release an unused pre-admission for an excluded local delivery."""

        return self.abandon_before_send(origin, destination)

    def try_begin(self, origin: str, destination: str) -> Optional[_SendAttempt]:
        """Admit the main logical send once for this origin and destination."""

        if not origin or not destination:
            return None

        with self._lock:
            if self._origin is None:
                self._origin = origin
            elif self._origin != origin:
                # A process-local direct delivery can retain the Python attribute.
                # Only the Cell that first originated this logical send may count it.
                return None
            if self._destination is None:
                self._destination = destination
            elif self._destination != destination:
                # A context has one trusted participant name. A second final
                # destination cannot safely reuse that attribution.
                self.mark_counter_gap()
                return None

            state = self._destinations.setdefault(destination, _DestinationState())
            if state.begun or state.completed or state.abandoned:
                return None
            state.begun = True
            owner_admission_attempted = state.owner_admission_attempted
            admission = state.admission
            if not owner_admission_attempted:
                state.owner_admission_attempted = True
            # A logical recipient is distinct from DownloadService's actual
            # consumers. Unknown lifetime identities never need to be replaced.
            for tx_id, receivers in self._transaction_receivers.items():
                if receivers is None:
                    self._transaction_receivers[tx_id] = frozenset((destination,))
            state.pending_transactions = {
                tx_id
                for tx_id, receivers in self._transaction_receivers.items()
                if receivers is not None and destination in receivers and tx_id not in self._transaction_results
            }
            unknown_transaction = any(receivers is None for receivers in self._transaction_receivers.values())
            destination_has_transaction = any(
                receivers is not None and destination in receivers for receivers in self._transaction_receivers.values()
            )
            unbound_destination = bool(self._transaction_receivers) and not destination_has_transaction

        if not owner_admission_attempted:
            admission = self._safe_call(
                "try_begin", self._traffic_class, self._recipient_name, failed_value=_CALL_FAILED
            )
            if admission is _CALL_FAILED or admission is None:
                self.mark_counter_gap()
                return None
            with self._lock:
                state.admission = admission
        elif admission is None:
            # A failed pre-admission already marked the aggregate partial.  Do
            # not retry after the semantic callback/cutoff boundary.
            return None

        with self._lock:
            queued_contributions = tuple(state.queued_contributions)
            state.queued_contributions.clear()
            already_failed = any(
                tx_id in self._transaction_results
                and destination not in self._transaction_results[tx_id]
                and self._transaction_receivers.get(tx_id) is not None
                and destination in self._transaction_receivers[tx_id]
                for tx_id in self._transaction_results
            )
            should_mark_incomplete = state.incomplete or unknown_transaction or unbound_destination or already_failed
            if should_mark_incomplete:
                state.incomplete = True

        if should_mark_incomplete:
            self._safe_call("mark_incomplete", admission)
            return None

        for contribution_id, payload_bytes in queued_contributions:
            added = self._add_payload_bytes(destination, payload_bytes)
            with self._lock:
                state.inflight_contributions.discard(contribution_id)
            if not added:
                return None

        return _SendAttempt(
            accepted_cb=lambda payload_bytes: self._main_accepted(destination, payload_bytes),
            abandoned_cb=lambda: self._main_abandoned(destination),
        )

    def register_oob_transaction(self, transaction_id: str, receiver_ids: Optional[Collection[str]]) -> None:
        """Register a FOBS DownloadService transaction before transport admission."""

        try:
            receivers = None if not receiver_ids else frozenset(str(receiver) for receiver in receiver_ids)
        except BaseException as ex:
            log.warning("logical-send accounting could not normalize DownloadService receivers: %s", ex)
            receivers = None

        incomplete_destinations = []
        duplicate = False
        with self._lock:
            existing = self._declared_transaction_receivers.get(transaction_id)
            if transaction_id in self._declared_transaction_receivers:
                if existing != receivers:
                    incomplete_destinations.extend(self._destinations)
                duplicate = True
            else:
                self._declared_transaction_receivers[transaction_id] = receivers
                logical_receivers = frozenset((self._destination,)) if self._destination else receivers
                self._transaction_receivers[transaction_id] = logical_receivers
                if logical_receivers is None:
                    # Transport binds its destination before admission. Until
                    # then there is no accounting recipient, but lifetime stays
                    # count-based and collection may still succeed.
                    pass
                else:
                    for destination, state in self._destinations.items():
                        if destination in logical_receivers:
                            if state.begun:
                                # FOBS finalizers run during encoding, before transport
                                # admission.  A later registration cannot be proven complete.
                                incomplete_destinations.append(destination)
                            else:
                                state.pending_transactions.add(transaction_id)

        for destination in incomplete_destinations:
            self._mark_incomplete(destination)
        if duplicate:
            return

    def make_oob_contribution_context(
        self,
        transaction_id: str,
        destination: str,
        contribution_id: Hashable,
        payload_bytes: int,
    ) -> Optional[_OobContributionContext]:
        """Create a byte-only context for one DownloadService data response."""

        if (
            isinstance(payload_bytes, bool)
            or not isinstance(payload_bytes, int)
            or not 0 <= payload_bytes <= _MAX_PAYLOAD_BYTES
        ):
            self.mark_oob_incomplete(destination)
            return None
        with self._lock:
            receivers = self._transaction_receivers.get(transaction_id)
            logical_destination = self._consumer_destinations.get((transaction_id, destination), destination)
            if receivers is None or logical_destination not in receivers:
                return None
        return _OobContributionContext(
            self, logical_destination, contribution_id, payload_bytes, actual_destination=destination
        )

    def mark_oob_incomplete(self, destination: str) -> None:
        """Fail safe when a DownloadService contribution cannot be identified."""

        with self._lock:
            logical_destination = self._destination or destination
        self._mark_incomplete(logical_destination)

    def settle_oob_transaction(self, transaction_id: str, successful_receivers: Collection[str]) -> None:
        """Settle one registered transaction using its per-receiver terminal truth."""

        try:
            actual_successful = frozenset(str(receiver) for receiver in successful_receivers)
            with self._lock:
                successful = frozenset(
                    self._consumer_destinations.get((transaction_id, receiver), receiver)
                    for receiver in actual_successful
                )
        except BaseException as ex:
            log.warning("logical-send accounting could not normalize DownloadService outcome: %s", ex)
            successful = frozenset()

        to_incomplete = []
        to_complete = []
        with self._lock:
            if transaction_id in self._transaction_results:
                return
            self._transaction_results[transaction_id] = successful
            receivers = self._transaction_receivers.get(transaction_id)
            if receivers is None:
                to_incomplete.extend(self._destinations)
            else:
                for destination in receivers:
                    state = self._destinations.setdefault(destination, _DestinationState())
                    state.pending_transactions.discard(transaction_id)
                    if destination in successful:
                        state.successful_transactions.add(transaction_id)
                        completion = self._take_completion_locked(destination, state)
                        if completion:
                            to_complete.append(completion)
                    else:
                        to_incomplete.append(destination)

        for destination in to_incomplete:
            self._mark_incomplete(destination)
        for admission, payload_bytes in to_complete:
            self._safe_call("complete_remote_accepted", admission, payload_bytes)

    def _try_begin_contribution(
        self,
        origin: str,
        destination: str,
        contribution_id: Hashable,
        payload_bytes: int,
    ) -> Optional[_SendAttempt]:
        with self._lock:
            if self._origin is None or self._origin != origin:
                return None
            state = self._destinations.setdefault(destination, _DestinationState())
            previous = state.contribution_lengths.get(contribution_id)
            if previous is not None and previous != payload_bytes:
                self._mark_incomplete(destination)
                return None
            if state.completed or state.abandoned or state.incomplete:
                return None
            if not self._reserve_contribution_locked(destination, state, contribution_id, payload_bytes):
                return None
            if contribution_id in state.accepted_contributions or contribution_id in state.inflight_contributions:
                return None
            state.inflight_contributions.add(contribution_id)

        return _SendAttempt(
            accepted_cb=lambda _encoded_size: self._contribution_accepted(destination, contribution_id, payload_bytes),
            abandoned_cb=lambda: self._contribution_abandoned(destination, contribution_id),
        )

    def _reserve_contribution_locked(self, destination, state, contribution_id, payload_bytes) -> bool:
        previous = state.contribution_lengths.get(contribution_id)
        if previous is not None:
            if previous != payload_bytes:
                self._mark_incomplete(destination)
                return False
            return True
        if self._metadata_saturated or self._metadata_count >= MAX_ACCOUNTING_ENTRIES:
            self._metadata_saturated = True
            self._mark_incomplete(destination)
            return False
        state.contribution_lengths[contribution_id] = payload_bytes
        self._metadata_count += 1
        return True

    def _main_accepted(self, destination: str, payload_bytes: int) -> None:
        if isinstance(payload_bytes, bool) or not isinstance(payload_bytes, int) or payload_bytes < 0:
            self._mark_incomplete(destination)
            return

        completion = None
        with self._lock:
            state = self._destinations[destination]
            if state.completed or state.abandoned or state.incomplete:
                return
            state.main_accepted = True
            state.main_payload_bytes = payload_bytes
            completion = self._take_completion_locked(destination, state)

        if completion:
            self._safe_call("complete_remote_accepted", *completion)

    def _main_abandoned(self, destination: str) -> None:
        admission = None
        with self._lock:
            state = self._destinations[destination]
            if state.completed or state.abandoned or state.incomplete:
                return
            state.abandoned = True
            admission = state.admission
        if admission is not None:
            self._safe_call("abandon", admission)

    def _contribution_accepted(self, destination: str, contribution_id: Hashable, payload_bytes: int) -> None:
        queue_only = False
        with self._lock:
            state = self._destinations[destination]
            if state.completed or state.abandoned or state.incomplete:
                state.inflight_contributions.discard(contribution_id)
                return
            if contribution_id in state.accepted_contributions:
                state.inflight_contributions.discard(contribution_id)
                return
            state.accepted_contributions.add(contribution_id)
            if state.admission is None:
                state.queued_contributions.append((contribution_id, payload_bytes))
                queue_only = True
        if queue_only:
            return

        added = self._add_payload_bytes(destination, payload_bytes)
        with self._lock:
            state.inflight_contributions.discard(contribution_id)
        if added:
            self._complete_if_ready(destination)

    def _contribution_abandoned(self, destination: str, contribution_id: Hashable) -> None:
        admission = None
        with self._lock:
            state = self._destinations[destination]
            state.inflight_contributions.discard(contribution_id)
            if state.completed or state.abandoned or state.incomplete:
                return
            state.incomplete = True
            admission = state.admission
        if admission is not None:
            self._safe_call("mark_incomplete", admission)

    def _add_payload_bytes(self, destination: str, payload_bytes: int) -> bool:
        with self._lock:
            state = self._destinations[destination]
            if state.completed or state.abandoned or state.incomplete or state.admission is None:
                return False
            admission = state.admission
        result = self._safe_call("add_accepted_payload_bytes", admission, payload_bytes, failed_value=_CALL_FAILED)
        if result is not True:
            self._mark_incomplete(destination)
            return False
        return True

    def _complete_if_ready(self, destination: str) -> None:
        completion = None
        with self._lock:
            state = self._destinations[destination]
            completion = self._take_completion_locked(destination, state)
        if completion:
            self._safe_call("complete_remote_accepted", *completion)

    def _take_completion_locked(self, destination: str, state: _DestinationState):
        if not self._ready_to_complete_locked(destination, state):
            return None
        state.completed = True
        return state.admission, state.main_payload_bytes

    def _mark_incomplete(self, destination: str) -> None:
        admission = None
        mark_counter_gap = False
        with self._lock:
            state = self._destinations.setdefault(destination, _DestinationState())
            if state.abandoned or state.incomplete:
                return
            if state.completed:
                # The operation has already been published.  A valid FOBS path
                # registers before admission, so late evidence can only be
                # represented as an aggregate counter gap.
                mark_counter_gap = True
            else:
                state.incomplete = True
                admission = state.admission
        if mark_counter_gap:
            self._safe_call("mark_counter_gap")
            return
        if admission is not None:
            self._safe_call("mark_incomplete", admission)

    def _ready_to_complete_locked(self, destination: str, state: _DestinationState) -> bool:
        if (
            not state.main_accepted
            or state.admission is None
            or state.completed
            or state.abandoned
            or state.incomplete
            or state.pending_transactions
            or state.inflight_contributions
            or state.pending_external_contributions
        ):
            return False

        for transaction_id, receivers in self._transaction_receivers.items():
            if receivers is None:
                return False
            if destination in receivers and transaction_id not in state.successful_transactions:
                return False
        return True

    def _safe_call(self, method_name: str, *args, failed_value=None):
        try:
            method = getattr(self._accounting, method_name)
            return method(*args)
        except BaseException as ex:
            # Accounting is observational.  Even a malformed callback must not
            # change the transport result or suppress delivery.
            log.warning("logical-send accounting callback %s failed: %s", method_name, ex)
            return failed_value


@dataclass
class _SourceGroup:
    payload_bytes: int = 0
    seen: dict[Hashable, int] = field(default_factory=dict)
    accepted: set[Hashable] = field(default_factory=set)
    inflight: set[Hashable] = field(default_factory=set)
    settled: bool = False
    successful: bool = False


class _SourceContributionContext:
    """One source data reply; it adds bytes but never admits a public message."""

    def __init__(self, owner, key, contribution_id, payload_bytes):
        self._owner = owner
        self._key = key
        self._contribution_id = contribution_id
        self._payload_bytes = payload_bytes

    def is_origin(self, origin: str) -> bool:
        return self._owner.is_origin(origin)

    def try_begin(self, origin: str, destination: str) -> Optional[_SendAttempt]:
        if not self.is_origin(origin) or destination != self._key[1]:
            return None
        return self._owner._begin(self._key, self._contribution_id, self._payload_bytes)

    def mark_counter_gap(self) -> None:
        self._owner.mark_counter_gap()


class SourceContributionAccumulator:
    """Bounded, source-only accepted tensor bytes for one managed trainer result.

    Serialization registers source references; existing transport callbacks
    observe accepted chunks. Snapshot does not wait, retain model data, or
    count an internal trainer-to-client message. It is a site self-report.
    """

    def __init__(
        self, origin: str, expected_recipients: Collection[str], max_groups: int = MAX_SOURCE_CONTRIBUTION_GROUPS
    ):
        self._origin = origin
        self._recipients = frozenset(expected_recipients)
        self._max_groups = min(max_groups, MAX_SOURCE_CONTRIBUTION_GROUPS)
        self._lock = threading.RLock()
        self._transactions: dict[str, frozenset[str]] = {}
        self._references: dict[str, str] = {}
        self._groups: dict[tuple[str, str], _SourceGroup] = {}
        self._chunk_count = 0
        self._incomplete = not bool(self._recipients)
        self._frozen: Optional[tuple] = None

    def is_origin(self, origin: str) -> bool:
        return origin == self._origin

    def mark_counter_gap(self) -> None:
        with self._lock:
            if self._frozen is None:
                self._incomplete = True

    def register_oob_transaction(self, transaction_id: str, receiver_ids: Optional[Collection[str]]) -> None:
        # The managed task supplies trusted final accounting recipients. A
        # count-based transaction may intentionally leave its lifetime receiver
        # identities unknown; that does not erase the accounting destination or
        # authorize us to change the transaction's completion policy.
        recipients = self._recipients if receiver_ids is None else frozenset(receiver_ids)
        with self._lock:
            if self._frozen is not None:
                return
            if recipients != self._recipients or transaction_id in self._transactions:
                self._incomplete = True
                return
            self._transactions[transaction_id] = recipients

    def register_source_reference(self, transaction_id: str, source_ref: str) -> None:
        with self._lock:
            if self._frozen is not None:
                return
            if transaction_id not in self._transactions:
                self._incomplete = True
                return
            previous = self._references.get(source_ref)
            if previous is not None:
                if previous != transaction_id:
                    self._incomplete = True
                return
            if len(self._groups) + len(self._recipients) > self._max_groups:
                self._incomplete = True
                return
            self._references[source_ref] = transaction_id
            for recipient in self._recipients:
                self._groups[(source_ref, recipient)] = _SourceGroup()

    def resolve_oob_consumer(self, transaction_id: str, consumer: str, _request) -> Optional[str]:
        with self._lock:
            if consumer in self._transactions.get(transaction_id, ()):
                return consumer
            self._incomplete = True
        return None

    def make_oob_contribution_context(
        self, transaction_id: str, destination: str, contribution_id: Hashable, payload_bytes: int
    ) -> Optional[_SourceContributionContext]:
        if (
            not isinstance(payload_bytes, int)
            or isinstance(payload_bytes, bool)
            or not 0 <= payload_bytes <= _MAX_PAYLOAD_BYTES
            or not isinstance(contribution_id, tuple)
            or len(contribution_id) < 2
        ):
            self.mark_counter_gap()
            return None
        source_ref = contribution_id[1]
        key = (source_ref, destination)
        with self._lock:
            if self._frozen is not None:
                return None
            if self._references.get(source_ref) != transaction_id or key not in self._groups:
                self._incomplete = True
                return None
        return _SourceContributionContext(self, key, contribution_id, payload_bytes)

    def _begin(self, key, contribution_id, payload_bytes) -> Optional[_SendAttempt]:
        with self._lock:
            if self._frozen is not None or self._incomplete:
                return None
            group = self._groups[key]
            previous = group.seen.get(contribution_id)
            if previous is not None and previous != payload_bytes:
                self._incomplete = True
                return None
            if contribution_id in group.accepted or contribution_id in group.inflight:
                return None
            if previous is None:
                if self._chunk_count >= MAX_ACCOUNTING_ENTRIES:
                    self._incomplete = True
                    return None
                group.seen[contribution_id] = payload_bytes
                self._chunk_count += 1
            group.inflight.add(contribution_id)
        return _SendAttempt(
            accepted_cb=lambda _encoded_bytes: self._accepted(key, contribution_id, payload_bytes),
            abandoned_cb=lambda: self._abandoned(key, contribution_id),
        )

    def _accepted(self, key, contribution_id, payload_bytes) -> None:
        with self._lock:
            if self._frozen is not None:
                return
            group = self._groups[key]
            group.inflight.discard(contribution_id)
            if contribution_id in group.accepted:
                return
            if group.payload_bytes > _MAX_PAYLOAD_BYTES - payload_bytes:
                self._incomplete = True
                return
            group.accepted.add(contribution_id)
            group.payload_bytes += payload_bytes

    def _abandoned(self, key, contribution_id) -> None:
        with self._lock:
            if self._frozen is None:
                self._groups[key].inflight.discard(contribution_id)

    def mark_oob_incomplete(self, _destination: str) -> None:
        self.mark_counter_gap()

    def settle_oob_transaction(self, transaction_id: str, successful_receivers: Collection[str]) -> None:
        successful = frozenset(successful_receivers)
        with self._lock:
            if self._frozen is not None:
                return
            for key, group in self._groups.items():
                if self._references[key[0]] == transaction_id:
                    succeeded = key[1] in successful
                    if group.settled and group.successful != succeeded:
                        self._incomplete = True
                    group.settled = True
                    group.successful = succeeded

    def snapshot(self) -> dict:
        """Freeze once at the existing send barrier; late callbacks cannot rewrite it."""

        with self._lock:
            if self._frozen is None:
                self._frozen = tuple(
                    (
                        source_ref,
                        recipient,
                        str(group.payload_bytes),
                        bool(
                            not self._incomplete
                            and group.settled
                            and group.successful
                            and not group.inflight
                            and len(group.accepted) == len(group.seen)
                        ),
                    )
                    for (source_ref, recipient), group in sorted(self._groups.items())
                )
            return {
                "version": 1,
                "groups": [
                    {
                        "source_ref": source_ref,
                        "recipient_fqcn": recipient,
                        "accepted_payload_bytes": payload_bytes,
                        "complete": complete,
                    }
                    for source_ref, recipient, payload_bytes, complete in self._frozen
                ],
            }


def attach_logical_send_context(
    message, accounting: LogicalSendAccounting, traffic_class: object, recipient_name: str
) -> LogicalSendContext:
    """Attach one trusted, process-local logical-send context to ``message``."""

    context = LogicalSendContext(accounting=accounting, traffic_class=traffic_class, recipient_name=recipient_name)
    message.set_logical_send_context(context)
    return context
