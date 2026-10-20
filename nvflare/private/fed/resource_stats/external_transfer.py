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

"""Join managed trainer source bytes to the client job's logical result.

This bounded, process-local bridge does not collect another participant report.
Only lazy references actually serialized into the outgoing result contribute.
The existing source-settlement exchange supplies an immutable final summary;
missing or invalid summaries leave accounting partial, not job execution failed.
"""

from __future__ import annotations

import hmac
import json
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field

from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, MessagePropKey
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.private.defs import CellMessageHeaderKeys
from nvflare.private.fed.utils.identity_utils import TokenVerifier, load_cert_file

from .communication_limits import U128_MAX
from .f3_job_counter import get_job_f3_counter

MAX_SUMMARY_BYTES = 64 * 1024
MAX_GROUPS = 256
MAX_RESULTS = 64
_DECIMAL = re.compile(r"(?:0|[1-9][0-9]{0,38})\Z")


@dataclass
class _Result:
    key: tuple
    refs: frozenset[str]
    recipients: frozenset[str]
    supported: bool
    summary: dict | None = None
    contexts: dict = field(default_factory=dict)
    invalid: bool = False


class ExternalTransferManager:
    """One manager per active client-job counter; no timers or disk writes."""

    def __init__(self, counter):
        self.counter = counter
        self._lock = threading.RLock()
        self._results = OrderedDict()
        self._sources = {}

    def _active(self):
        return self.counter.state != "frozen"

    def _gap(self):
        self.counter.mark_counter_gap()

    @staticmethod
    def _identifier(value):
        return (
            isinstance(value, str)
            and 0 < len(value) <= 512
            and value.isascii()
            and not any(ord(c) < 33 or ord(c) == 127 for c in value)
        )

    def register_result_source(self, session_id, task_id, source_fqcn, source_refs, recipient_fqcns, bridge_supported):
        """Register identities already validated by the managed ClientAPI session."""

        if not self._active():
            return
        refs, recipients = frozenset(source_refs), frozenset(recipient_fqcns)
        key = (session_id, task_id, source_fqcn)
        if (
            not all(self._identifier(v) for v in (*key, *refs, *recipients))
            or not refs
            or not recipients
            or len(refs) * len(recipients) > MAX_GROUPS
            or not isinstance(bridge_supported, bool)
        ):
            self._gap()
            return
        with self._lock:
            previous = self._results.get(key)
            if previous is not None:
                if (previous.refs, previous.recipients, previous.supported) != (refs, recipients, bridge_supported):
                    self._invalidate(previous)
                return
            if any((source_fqcn, ref) in self._sources for ref in refs):
                self._gap()
                return
            while len(self._results) >= MAX_RESULTS:
                old_key, old = self._results.popitem(last=False)
                if any(group not in (old.summary or {}) for group in old.contexts):
                    self._invalidate(old)
                for ref in old.refs:
                    self._sources.pop((old_key[2], ref), None)
            self._results[key] = _Result(key, refs, recipients, bridge_supported)
            for ref in refs:
                self._sources[(source_fqcn, ref)] = key

    def bind_context(self, context, destination):
        """Install the serializer hook before forwarding a result to the server."""

        context.set_external_source_handler(lambda ctx, source, ref: self._forwarded(ctx, source, ref, destination))

    def _forwarded(self, context, source, ref, destination):
        if not self._active():
            return False
        with self._lock:
            result_key = self._sources.get((source, ref))
            result = self._results.get(result_key)
            group = (ref, destination)
            contribution_key = (result_key, ref, destination)
            if not context.register_external_contribution(contribution_key, destination):
                return False
            if result is None or destination not in result.recipients:
                context.settle_external_contribution(contribution_key, destination, 0, False)
                self._gap()
                return False
            previous = result.contexts.get(group)
            if previous is not None and previous is not context:
                context.settle_external_contribution(contribution_key, destination, 0, False)
                self._invalidate(result)
                return False
            result.contexts[group] = context
            if result.invalid or not result.supported:
                context.settle_external_contribution(contribution_key, destination, 0, False)
            elif result.summary is not None:
                amount, complete = result.summary.get(group, (0, False))
                context.settle_external_contribution(contribution_key, destination, amount, complete)
            return True

    def _invalidate(self, result):
        self._gap()
        result.invalid = True
        # Existing admissions remain partial; never overwrite an accepted total.
        for (ref, destination), context in result.contexts.items():
            context.mark_counter_gap()
            context.settle_external_contribution((result.key, ref, destination), destination, 0, False)

    def _validate(self, block, result):
        if not isinstance(block, dict) or set(block) != {"version", "groups"} or type(block["version"]) is not int:
            raise ValueError("invalid summary shape")
        groups = block["groups"]
        if block["version"] != 1 or not isinstance(groups, list) or len(groups) > MAX_GROUPS:
            raise ValueError("unsupported or oversized summary")
        validated = {}
        for group in groups:
            if not isinstance(group, dict) or set(group) != {
                "source_ref",
                "recipient_fqcn",
                "accepted_payload_bytes",
                "complete",
            }:
                raise ValueError("invalid group shape")
            ref, recipient, amount = group["source_ref"], group["recipient_fqcn"], group["accepted_payload_bytes"]
            if (
                not self._identifier(ref)
                or not self._identifier(recipient)
                or ref not in result.refs
                or recipient not in result.recipients
                or (ref, recipient) in validated
                or not isinstance(amount, str)
                or _DECIMAL.fullmatch(amount) is None
                or int(amount) > U128_MAX
                or type(group["complete"]) is not bool
            ):
                raise ValueError("invalid group value")
            validated[(ref, recipient)] = (int(amount), group["complete"])
        if len(json.dumps(block, separators=(",", ":")).encode("utf-8")) > MAX_SUMMARY_BYTES:
            raise ValueError("oversized summary")
        return validated

    def accept_summary(self, session_id, task_id, source_fqcn, block):
        """Best-effort optional data: rejecting it must not reject settlement."""

        if not self._active():
            return False
        with self._lock:
            key = (session_id, task_id, source_fqcn)
            result = self._results.get(key)
            if result is None:
                self._gap()
                return False
            try:
                summary = self._validate(block, result)
            except (ValueError, TypeError, OverflowError):
                self._invalidate(result)
                return False
            if not result.supported or result.invalid:
                self._invalidate(result)
                return False
            if result.summary is not None:
                if result.summary != summary:
                    self._invalidate(result)
                    return False
                return True
            result.summary = summary
            for (ref, destination), context in result.contexts.items():
                amount, complete = summary.get((ref, destination), (0, False))
                context.settle_external_contribution((key, ref, destination), destination, amount, complete)
            return True


_lock = threading.Lock()
_manager = None


def load_task_token_verifier(cert_file):
    """Best-effort public-key loading; never load a site or job private key."""

    if not isinstance(cert_file, str) or not cert_file:
        return None
    try:
        return TokenVerifier(load_cert_file(cert_file))
    except Exception:
        # Do not log credential-bearing exception text or block job startup.
        return None


def _credential(value, limit=512):
    return isinstance(value, str) and 0 < len(value) <= limit and value != "NA" and value.isascii()


def _valid_fqcn(value):
    return _credential(value) and value == value.strip() and not FQCN.validate(value)


def _verified_task_credentials(server, client, request, logical_destination, participant_name, server_job_fqcn):
    """Verify once for this task; a site token is not trainer attestation."""

    if server is None or server.secure_train is not True or client is None or request is None:
        return None
    verifier = getattr(server, "resource_stats_token_verifier", None)
    if not isinstance(verifier, TokenVerifier) or not _valid_fqcn(server_job_fqcn):
        return None
    job_parts = FQCN.split(server_job_fqcn)
    if len(job_parts) != 2 or job_parts[0] != FQCN.ROOT_SERVER:
        return None
    name = request.get_header(CellMessageHeaderKeys.CLIENT_NAME)
    token = request.get_header(CellMessageHeaderKeys.TOKEN)
    signature = request.get_header(CellMessageHeaderKeys.TOKEN_SIGNATURE)
    if not (_credential(name) and _credential(token) and _credential(signature, 8192)):
        return None
    site_fqcn = client.get_fqcn()
    if (
        name != participant_name
        or name != client.name
        or token != client.token
        or server.client_manager.clients.get(token) is not client
        or not _valid_fqcn(site_fqcn)
        or FQCN.get_root(site_fqcn) == FQCN.ROOT_SERVER
        or request.get_header(MessageHeaderKey.ORIGIN) != logical_destination
        or logical_destination != FQCN.join((site_fqcn, job_parts[1]))
        or verifier.verify(name, token, signature) is not True
    ):
        return None
    return name, token, signature, site_fqcn


def _delegated_credentials_match(proof, server, request):
    """Recheck the bounded bearer tuple and current registration on each pull."""

    name, token, signature, site_fqcn = proof
    client = server.client_manager.clients.get(token)
    if client is None or client.name != name or client.token != token or client.get_fqcn() != site_fqcn:
        return False
    incoming = (
        request.get_header(CellMessageHeaderKeys.CLIENT_NAME),
        request.get_header(CellMessageHeaderKeys.TOKEN),
        request.get_header(CellMessageHeaderKeys.TOKEN_SIGNATURE),
    )
    if not all(_credential(value, limit) for value, limit in zip(incoming, (512, 512, 8192))):
        return False
    return all(hmac.compare_digest(expected, actual) for expected, actual in zip(proof[:3], incoming))


def make_authenticated_consumer_resolver(
    logical_destination, participant_name, *, server=None, client=None, task_request=None, server_job_fqcn=None
):
    """Credit a same-job trainer using direct TLS or existing site delegation.

    The optional signed tuple is verified at task binding and matched on every
    descendant pull. Missing proof leaves accounting partial without changing
    authentication admission, source lifetime, or delivery.
    """

    try:
        proof = _verified_task_credentials(
            server, client, task_request, logical_destination, participant_name, server_job_fqcn
        )
    except Exception:
        proof = None

    def resolve(consumer, request):
        if not isinstance(consumer, str) or not isinstance(logical_destination, str):
            return None
        if consumer == logical_destination:
            return logical_destination
        if not _valid_fqcn(consumer) or not FQCN.is_ancestor(logical_destination, consumer):
            return None
        try:
            endpoint = request.get_prop(MessagePropKey.ENDPOINT)
            common_name = request.get_prop(MessagePropKey.COMMON_NAME)
            if endpoint is not None and endpoint.name == consumer and common_name == participant_name:
                return logical_destination
            if proof is not None and _delegated_credentials_match(proof, server, request):
                return logical_destination
        except Exception:
            # An accounting verifier is not an authentication admission filter.
            return None
        return None

    return resolve


def get_external_transfer_manager(cell):
    """Return the bridge only where an active job-process counter exists."""

    global _manager
    counter = get_job_f3_counter()
    if counter is None or counter.state == "frozen" or cell is None:
        return None
    with _lock:
        if _manager is None or _manager.counter is not counter:
            _manager = ExternalTransferManager(counter)
        return _manager
