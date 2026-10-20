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

from types import SimpleNamespace

import pytest

from nvflare.apis.client import Client
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, MessagePropKey
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.send_accounting import LogicalSendContext
from nvflare.lighter.utils import Identity, generate_cert, generate_keys, serialize_cert, sign_content
from nvflare.private.defs import CellMessageHeaderKeys
from nvflare.private.fed.resource_stats.external_transfer import (
    MAX_RESULTS,
    ExternalTransferManager,
    load_task_token_verifier,
    make_authenticated_consumer_resolver,
)
from nvflare.private.fed.resource_stats.f3_counter import F3Counter, F3TrafficClass
from nvflare.private.fed.utils.identity_utils import TokenVerifier

SOURCE = "site-1.job-1.trainer"
DEST = "server.job-1"


def summary(ref="ref-1", amount="1024", complete=True):
    return {
        "version": 1,
        "groups": [{"source_ref": ref, "recipient_fqcn": DEST, "accepted_payload_bytes": amount, "complete": complete}],
    }


def setup(supported=True, refs=("ref-1",)):
    counter = F3Counter()
    manager = ExternalTransferManager(counter)
    manager.register_result_source("session", "task", SOURCE, refs, (DEST,), supported)
    context = LogicalSendContext(counter, F3TrafficClass.TASK_RESULT, "server")
    manager.bind_context(context, DEST)
    context.prepare_destination("site-1.job-1", DEST)
    return counter, manager, context


def send(context):
    attempt = context.try_begin("site-1.job-1", DEST)
    assert attempt is not None
    attempt.accepted(100)


@pytest.mark.parametrize("early", [False, True])
def test_summary_joins_one_logical_message_in_either_order(early):
    counter, manager, context = setup()
    if early:
        assert manager.accept_summary("session", "task", SOURCE, summary())
    context.register_external_source(SOURCE, "ref-1")
    send(context)
    if not early:
        assert counter.pending_count == 1
        assert manager.accept_summary("session", "task", SOURCE, summary())
    assert manager.accept_summary("session", "task", SOURCE, summary())  # identical replay
    assert counter.freeze() == {"sent_to": [{"participant_name": "server", "payload_bytes": "1124", "messages": "1"}]}


def test_unforwarded_filtered_source_is_not_counted():
    counter, manager, context = setup(refs=("ref-1", "filtered-ref"))
    block = summary()
    block["groups"].extend(summary("filtered-ref", "999999")["groups"])
    assert manager.accept_summary("session", "task", SOURCE, block)
    context.register_external_source(SOURCE, "ref-1")
    send(context)
    assert counter.freeze()["sent_to"][0]["payload_bytes"] == "1124"


@pytest.mark.parametrize("supported", [True, False])
def test_missing_or_unsupported_bridge_never_claims_complete(supported):
    counter, _, context = setup(supported)
    context.register_external_source(SOURCE, "ref-1")
    if supported:
        send(context)
    else:
        # Unsupported optional accounting does not create a publishable send.
        assert context.try_begin("site-1.job-1", DEST) is None
    assert counter.freeze()["status"] == "partial"


@pytest.mark.parametrize(
    "block",
    [
        None,
        {},
        {"version": True, "groups": []},
        summary("unknown"),
        summary(amount="01"),
        summary(amount=str(2**128)),
        summary(complete=1),
        {"version": 1, "groups": summary()["groups"] * 2},
    ],
)
def test_invalid_blocks_leave_pending_result_partial(block):
    counter, manager, context = setup()
    context.register_external_source(SOURCE, "ref-1")
    send(context)
    assert not manager.accept_summary("session", "task", SOURCE, block)
    assert counter.pending_count == 0  # Known-invalid accounting must not spend the drain budget.
    assert counter.freeze()["status"] == "partial"


def test_conflicting_replay_keeps_prior_bytes_but_marks_partial():
    counter, manager, context = setup()
    context.register_external_source(SOURCE, "ref-1")
    send(context)
    assert manager.accept_summary("session", "task", SOURCE, summary())
    assert not manager.accept_summary("session", "task", SOURCE, summary(amount="1025"))
    result = counter.freeze()
    assert result["status"] == "partial"
    assert result["sent_to"][0]["payload_bytes"] == "1124"


def test_summary_failure_and_missing_group_do_not_publish_envelope_only():
    for block in (summary(complete=False), {"version": 1, "groups": []}):
        counter, manager, context = setup()
        context.register_external_source(SOURCE, "ref-1")
        send(context)
        assert manager.accept_summary("session", "task", SOURCE, block)
        result = counter.freeze()
        assert result["status"] == "partial"
        assert result["sent_to"] == []


def test_late_summary_cannot_change_frozen_report():
    counter, manager, context = setup()
    context.register_external_source(SOURCE, "ref-1")
    send(context)
    frozen = counter.freeze()
    assert not manager.accept_summary("session", "task", SOURCE, summary())
    assert counter.snapshot() == frozen


def test_recent_state_is_bounded_and_unresolved_eviction_partial():
    counter, manager, context = setup()
    context.register_external_source(SOURCE, "ref-1")
    send(context)
    for i in range(MAX_RESULTS):
        manager.register_result_source("session", f"task-{i}", SOURCE, (f"ref-{i + 2}",), (DEST,), True)
    assert len(manager._results) == MAX_RESULTS
    assert len(manager._sources) == MAX_RESULTS
    assert counter.freeze()["status"] == "partial"


@pytest.mark.parametrize(
    "consumer,peer,cn,expected",
    [
        (SOURCE, SOURCE, "site-1", "site-1.job-1"),
        (SOURCE, "site-1", "site-1", None),
        (SOURCE, SOURCE, "site-2", None),
        ("site-1.job-2.trainer", "site-1.job-2.trainer", "site-1", None),
        ("site-1.job-10.trainer", "site-1.job-10.trainer", "site-1", None),
        ("site-2.job-1.trainer", "site-2.job-1.trainer", "site-2", None),
        (SOURCE, None, None, None),
    ],
)
def test_consumer_binding_requires_direct_authenticated_same_job(consumer, peer, cn, expected):
    request = Message(headers={MessagePropKey.COMMON_NAME: "site-1", MessagePropKey.ENDPOINT: SOURCE})
    if peer is not None:
        request.set_prop(MessagePropKey.ENDPOINT, SimpleNamespace(name=peer))
    if cn is not None:
        request.set_prop(MessagePropKey.COMMON_NAME, cn)
    resolver = make_authenticated_consumer_resolver("site-1.job-1", "site-1")
    assert resolver(consumer, request) == expected


@pytest.fixture(scope="module")
def task_signing_identity():
    root_key, root_public_key = generate_keys()
    root_identity = Identity("root", "nvidia")
    root_cert = generate_cert(root_identity, root_identity, root_key, root_public_key, ca=True)
    server_key, server_public_key = generate_keys()
    server_cert = generate_cert(Identity("server", "nvidia"), root_identity, root_key, server_public_key)
    job_key, job_public_key = generate_keys()
    job_cert = generate_cert(Identity("server", "nvidia"), root_identity, root_key, job_public_key)
    return SimpleNamespace(
        root_cert=root_cert,
        server_key=server_key,
        server_cert=server_cert,
        site_signature=sign_content("site-1" + "registered-token-1", server_key),
        job_key=job_key,
        job_cert=job_cert,
    )


def _task_credentials(signing_identity):
    name, token = "site-1", "registered-token-1"
    return {
        CellMessageHeaderKeys.CLIENT_NAME: name,
        CellMessageHeaderKeys.TOKEN: token,
        CellMessageHeaderKeys.TOKEN_SIGNATURE: signing_identity.site_signature,
    }


def _delegated_session(signing_identity):
    headers = _task_credentials(signing_identity)
    client = Client("site-1", "registered-token-1")
    server = SimpleNamespace(
        secure_train=True,
        resource_stats_token_verifier=TokenVerifier(signing_identity.server_cert),
        client_manager=SimpleNamespace(clients={client.token: client}),
    )
    request = Message(headers={**headers, MessageHeaderKey.ORIGIN: "site-1.job-1"})
    return server, client, request


def _task_resolver(server, client, task_request, logical_destination="site-1.job-1", server_job_fqcn=DEST):
    return make_authenticated_consumer_resolver(
        logical_destination,
        "site-1",
        server=server,
        client=client,
        task_request=task_request,
        server_job_fqcn=server_job_fqcn,
    )


def _relay_download(headers):
    request = Message(headers={**headers, MessageHeaderKey.ORIGIN: SOURCE})
    # These are trustworthy local transport properties, but they identify the
    # relay rather than the external trainer that consumes the tensor.
    request.set_prop(MessagePropKey.ENDPOINT, SimpleNamespace(name="server"))
    request.set_prop(MessagePropKey.COMMON_NAME, "server")
    return request


def test_signed_site_delegation_maps_relayed_trainer_without_direct_tls(task_signing_identity, monkeypatch):
    server, client, task = _delegated_session(task_signing_identity)
    calls = []
    verify = server.resource_stats_token_verifier.verify

    def verify_once(name, token, signature):
        calls.append((name, token, signature))
        return verify(name, token, signature)

    monkeypatch.setattr(server.resource_stats_token_verifier, "verify", verify_once)
    resolver = _task_resolver(server, client, task)
    download = _relay_download(_task_credentials(task_signing_identity))
    assert resolver(SOURCE, download) == "site-1.job-1"
    assert resolver(SOURCE, download) == "site-1.job-1"
    assert len(calls) == 1  # Crypto at binding; bounded tuple checks at each pull.


@pytest.mark.parametrize(
    "field,value",
    [
        (CellMessageHeaderKeys.CLIENT_NAME, "site-2"),
        (CellMessageHeaderKeys.CLIENT_NAME, None),
        (CellMessageHeaderKeys.TOKEN, "another-token"),
        (CellMessageHeaderKeys.TOKEN, "NA"),
        (CellMessageHeaderKeys.TOKEN, "t" * 513),
        (CellMessageHeaderKeys.TOKEN, b"registered-token-1"),
        (CellMessageHeaderKeys.TOKEN_SIGNATURE, "another-signature"),
        (CellMessageHeaderKeys.TOKEN_SIGNATURE, None),
        (CellMessageHeaderKeys.TOKEN_SIGNATURE, "NA"),
        (CellMessageHeaderKeys.TOKEN_SIGNATURE, "s" * 8193),
        (CellMessageHeaderKeys.TOKEN_SIGNATURE, "non-ascii-\u2603"),
        (CellMessageHeaderKeys.TOKEN_SIGNATURE, True),
    ],
)
def test_delegated_download_requires_exact_bounded_proof(task_signing_identity, field, value):
    server, client, task = _delegated_session(task_signing_identity)
    resolver = _task_resolver(server, client, task)
    headers = _task_credentials(task_signing_identity)
    headers[field] = value
    assert resolver(SOURCE, _relay_download(headers)) is None


@pytest.mark.parametrize(
    "consumer",
    [
        "site-2.job-1.trainer",
        "site-1.job-2.trainer",
        "site-1.job-10.trainer",
        "site-1.job-1-other.trainer",
        "site-1.job-1..trainer",
        "site-1.job-1.trainer ",
        "site-1.-client_api_trainer",
    ],
)
def test_delegated_proof_cannot_credit_different_site_or_job(task_signing_identity, consumer):
    server, client, task = _delegated_session(task_signing_identity)
    resolver = _task_resolver(server, client, task)
    assert resolver(consumer, _relay_download(_task_credentials(task_signing_identity))) is None


@pytest.mark.parametrize("change", ["removed", "name", "token", "fqcn"])
def test_delegated_download_rechecks_current_registration(task_signing_identity, change):
    server, client, task = _delegated_session(task_signing_identity)
    resolver = _task_resolver(server, client, task)
    download = _relay_download(_task_credentials(task_signing_identity))
    assert resolver(SOURCE, download) == "site-1.job-1"
    if change == "removed":
        server.client_manager.clients.clear()
    elif change == "name":
        client.name = "site-2"
    elif change == "token":
        client.token = "rotated-token"
    else:
        client.set_fqcn("another-relay.site-1")
    assert resolver(SOURCE, download) is None


@pytest.mark.parametrize(
    "invalid_binding",
    [
        "insecure",
        "missing-verifier",
        "fake-verifier",
        "wrong-certificate",
        "invalid-signature",
        "unregistered-client",
        "wrong-token",
        "wrong-name",
        "wrong-origin",
        "wrong-job",
        "invalid-server-job",
        "missing-site-origin",
    ],
)
def test_invalid_task_binding_disables_only_delegated_mapping(task_signing_identity, invalid_binding):
    server, client, task = _delegated_session(task_signing_identity)
    job_fqcn = DEST
    if invalid_binding == "insecure":
        server.secure_train = False
    elif invalid_binding == "missing-verifier":
        server.resource_stats_token_verifier = None
    elif invalid_binding == "fake-verifier":
        server.resource_stats_token_verifier = SimpleNamespace(verify=lambda *_args: True)
    elif invalid_binding == "wrong-certificate":
        server.resource_stats_token_verifier = TokenVerifier(task_signing_identity.job_cert)
    elif invalid_binding == "invalid-signature":
        task.set_header(
            CellMessageHeaderKeys.TOKEN_SIGNATURE,
            sign_content(client.name + client.token, task_signing_identity.job_key),
        )
    elif invalid_binding == "unregistered-client":
        server.client_manager.clients.clear()
    elif invalid_binding == "wrong-token":
        task.set_header(CellMessageHeaderKeys.TOKEN, "unregistered-token")
    elif invalid_binding == "wrong-name":
        task.set_header(CellMessageHeaderKeys.CLIENT_NAME, "site-2")
    elif invalid_binding == "wrong-origin":
        task.set_header(MessageHeaderKey.ORIGIN, "site-1.job-2")
    elif invalid_binding == "wrong-job":
        job_fqcn = "server.job-2"
    elif invalid_binding == "invalid-server-job":
        job_fqcn = "server.job-1.extra"
    else:
        client.set_fqcn(None)
    resolver = _task_resolver(server, client, task, server_job_fqcn=job_fqcn)
    download = _relay_download(_task_credentials(task_signing_identity))
    assert resolver(SOURCE, download) is None
    # Existing direct mTLS evidence still works when optional delegation fails.
    download.set_prop(MessagePropKey.ENDPOINT, SimpleNamespace(name=SOURCE))
    download.set_prop(MessagePropKey.COMMON_NAME, "site-1")
    assert resolver(SOURCE, download) == "site-1.job-1"


def test_task_binding_uses_registered_hierarchical_site_origin(task_signing_identity):
    server, client, task = _delegated_session(task_signing_identity)
    client.set_fqcn("relay.site-1")
    destination = "relay.site-1.job-1"
    task.set_header(MessageHeaderKey.ORIGIN, destination)
    resolver = _task_resolver(server, client, task, logical_destination=destination)
    assert resolver(destination + ".trainer", _relay_download(_task_credentials(task_signing_identity))) == destination


def test_delegated_resolver_errors_are_optional_and_do_not_log_proof(task_signing_identity, monkeypatch, caplog):
    server, client, task = _delegated_session(task_signing_identity)

    def raises_with_secret(*_args):
        raise ValueError("private bearer token must not appear")

    monkeypatch.setattr(server.resource_stats_token_verifier, "verify", raises_with_secret)
    resolver = _task_resolver(server, client, task)
    assert resolver(SOURCE, _relay_download(_task_credentials(task_signing_identity))) is None
    assert "private bearer token" not in caplog.text


def test_public_certificate_loader_uses_site_public_key_not_job_key(task_signing_identity, tmp_path):
    cert_file = tmp_path / "configured-parent.crt"
    cert_file.write_bytes(serialize_cert(task_signing_identity.server_cert))
    verifier = load_task_token_verifier(str(cert_file))
    assert isinstance(verifier, TokenVerifier)
    signature = _task_credentials(task_signing_identity)[CellMessageHeaderKeys.TOKEN_SIGNATURE]
    assert verifier.verify("site-1", "registered-token-1", signature) is True
    assert TokenVerifier(task_signing_identity.job_cert).verify("site-1", "registered-token-1", signature) is False
    # Neither a parent nor job private key is present beside the public cert.
    assert list(tmp_path.iterdir()) == [cert_file]


@pytest.mark.parametrize("cert_file", [None, "", 1, "not-a-real-certificate.crt"])
def test_public_certificate_loader_failures_are_best_effort(cert_file):
    assert load_task_token_verifier(cert_file) is None


def test_public_certificate_loader_invalid_content_is_best_effort(tmp_path):
    cert_file = tmp_path / "invalid.crt"
    cert_file.write_bytes(b"not a certificate")
    assert load_task_token_verifier(str(cert_file)) is None
