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

import datetime
import logging
from types import SimpleNamespace

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from nvflare.apis.fl_constant import ConnectionSecurity
from nvflare.fuel.f3.cellnet.core_cell import CoreCell
from nvflare.fuel.f3.cellnet.credential_manager import CERT_CONTENT, CredentialManager
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, MessageType, ReturnCode
from nvflare.fuel.f3.cellnet.identity import ADMIN_LISTENER_KEY, CellIdentityResolver
from nvflare.fuel.f3.cellnet.utils import make_reply
from nvflare.fuel.f3.comm_error import CommError
from nvflare.fuel.f3.drivers.driver_params import DriverParams
from nvflare.fuel.f3.endpoint import Endpoint
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.sfm.conn_manager import ConnManager
from nvflare.fuel.f3.sfm.constants import HandshakeKeys
from nvflare.fuel.f3.sfm.sfm_conn import SfmConnection
from nvflare.fuel.utils.constants import Mode


class _FakeConnection:
    def __init__(self, peer_cn, conn_security=ConnectionSecurity.MTLS, mode=Mode.PASSIVE, admin_listener=False):
        self.name = "CN-test"
        self.closed = False
        self.connector = SimpleNamespace(
            mode=mode,
            params={
                DriverParams.SECURE: True,
                DriverParams.CONNECTION_SECURITY: conn_security,
            },
        )
        if admin_listener:
            self.connector.params[ADMIN_LISTENER_KEY] = "true"
        self.conn_props = {}
        if peer_cn is not None:
            self.conn_props[DriverParams.PEER_CN.value] = peer_cn

    def get_conn_properties(self):
        return self.conn_props

    def close(self):
        self.closed = True


def _cert_pem(common_name: str):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=1))
        .sign(key, hashes.SHA256())
    )
    return cert.public_bytes(serialization.Encoding.PEM)


def _conn_manager(local_fqcn="server", identity_map=None):
    resolver = CellIdentityResolver(local_fqcn=local_fqcn, prefix_identity_map=identity_map)
    return ConnManager(Endpoint(local_fqcn), identity_resolver=resolver)


class _NoOpStats:
    def increment(self, *args, **kwargs):
        pass

    def record_value(self, *args, **kwargs):
        pass


class _EmptyRegistry:
    def find(self, *args, **kwargs):
        return None


def _receive_test_core_cell(reply):
    cell = CoreCell.__new__(CoreCell)
    cell.my_info = SimpleNamespace(fqcn="server")
    cell.logger = logging.getLogger(__name__)
    cell.received_msg_counter_pool = _NoOpStats()
    cell.sent_msg_counter_pool = _NoOpStats()
    cell.received_msg_size_pool = _NoOpStats()
    cell.msg_travel_stats_pool = _NoOpStats()
    cell.in_filter_reg = _EmptyRegistry()
    cell.out_reply_filter_reg = _EmptyRegistry()
    cell.message_interceptor = None
    cell.sent_reply = None
    cell._process_request = lambda origin, message: reply
    cell._send_reply = lambda reply, endpoint: setattr(cell, "sent_reply", reply)
    return cell


def test_identity_resolver_maps_job_cell_to_configured_parent_identity():
    resolver = CellIdentityResolver(local_fqcn="site-1", prefix_identity_map={"site-1": "site-1"})

    assert resolver.resolve("site-1.job-123") == "site-1"


def test_identity_resolver_maps_relay_child_to_child_identity_without_configured_prefix():
    resolver = CellIdentityResolver(local_fqcn="relay-1", exact_identity_map={"relay-1": "relay-1"})

    assert resolver.resolve("relay-1.site-1") == "site-1"


def test_identity_resolver_maps_nested_relay_child_to_child_identity_despite_parent_prefix():
    resolver = CellIdentityResolver(local_fqcn="relay-1.relay-2", prefix_identity_map={"relay-1": "relay-1"})

    assert resolver.resolve("relay-1.relay-2.relay-3") == "relay-3"
    resolver.require_match("relay-1.relay-2.relay-3", "relay-3", "connection relay-3")
    with pytest.raises(ValueError, match="relay-1"):
        resolver.require_match("relay-1.relay-2.relay-3", "relay-1", "connection relay-3")


def test_identity_resolver_uses_configured_identity_for_nested_relay_child():
    resolver = CellIdentityResolver(
        local_fqcn="relay-1.relay-2",
        prefix_identity_map={
            "relay-1": "relay-1",
            "relay-1.relay-2.relay-3": "custom-relay-3",
        },
    )

    assert resolver.resolve("relay-1.relay-2.relay-3") == "custom-relay-3"


def test_identity_resolver_rejects_unresolvable_endpoint_identity():
    resolver = CellIdentityResolver(local_fqcn="server")

    with pytest.raises(ValueError, match="does not resolve"):
        resolver.require_match(".", "site-1", "connection test")


def test_identity_resolver_accepts_authenticated_admin_client_identity():
    resolver = CellIdentityResolver(local_fqcn="server")

    resolver.require_match("_admin_9af49fef-235f-41bd-9296-12fd09eacb2a", "admin@nvidia.com", "connection admin")


def test_identity_resolver_rejects_admin_like_endpoint_without_authenticated_identity():
    resolver = CellIdentityResolver(local_fqcn="server")

    with pytest.raises(ValueError, match="authenticated mTLS peer common name"):
        resolver.require_match("_admin_9af49fef-235f-41bd-9296-12fd09eacb2a", None, "connection admin")

    with pytest.raises(ValueError, match="requires identity"):
        resolver.require_match("_admin_not-a-uuid", "admin@nvidia.com", "connection admin")


def test_identity_resolver_maps_hierarchical_cell_pipe_cell_to_owner_identity():
    # current CellPipe naming: <site>.<token>.<mode> resolves via the FQCN hierarchy
    resolver = CellIdentityResolver(local_fqcn="server", prefix_identity_map={"site-1": "site-1"})

    assert resolver.resolve("site-1.8cb50f16-8158-46f6-a8d7-ec85b1f06c53.active") == "site-1"
    assert resolver.resolve("site-1.8cb50f16-8158-46f6-a8d7-ec85b1f06c53.passive") == "site-1"


def test_identity_resolver_maps_hierarchical_cell_pipe_cell_behind_relay_to_owner_identity():
    resolver = CellIdentityResolver(local_fqcn="relay-1", exact_identity_map={"relay-1": "relay-1"})

    assert resolver.resolve("relay-1.site-1.job-123.active") == "site-1"


def test_identity_resolver_maps_legacy_cell_pipe_alias_to_owner_identity():
    # CellPipe cells from older NVFlare versions use underscore alias names
    resolver = CellIdentityResolver(local_fqcn="server", prefix_identity_map={"site-1": "site-1"})

    assert resolver.resolve("site-1_8cb50f16-8158-46f6-a8d7-ec85b1f06c53_active") == "site-1"
    assert resolver.resolve("site-1_8cb50f16-8158-46f6-a8d7-ec85b1f06c53_passive") == "site-1"


def test_identity_resolver_maps_cell_pipe_alias_to_configured_owner_identity():
    resolver = CellIdentityResolver(local_fqcn="server", prefix_identity_map={"site-1": "custom-site-cn"})

    assert resolver.resolve("site-1_job-123_active") == "custom-site-cn"


def test_identity_resolver_maps_cell_pipe_alias_owner_with_underscores_from_the_right():
    resolver = CellIdentityResolver(local_fqcn="server")

    # The runtime id cannot contain "_", so the only valid owner is "site-a_x"
    assert resolver.resolve("site-a_x_job-123_active") == "site-a_x"


def test_identity_resolver_maps_dotted_cell_pipe_alias_to_owner_identity():
    resolver = CellIdentityResolver(local_fqcn="site-1")

    assert resolver.resolve("site-1.site-1_job-123_passive") == "site-1"


def test_identity_resolver_does_not_treat_unconstrained_names_as_cell_pipe_aliases():
    resolver = CellIdentityResolver(local_fqcn="server")

    # Too few segments, empty runtime id, or an unknown mode are not aliases
    assert resolver.resolve("site-1_active") == "site-1_active"
    assert resolver.resolve("site-1__active") == "site-1__active"
    assert resolver.resolve("site-1_job-123_idle") == "site-1_job-123_idle"


def test_mtls_handshake_accepts_job_cell_with_parent_cert_identity():
    manager = _conn_manager(identity_map={"site-1": "site-1"})
    conn = _FakeConnection(peer_cn="site-1")
    sfm_conn = SfmConnection(conn, Endpoint("server"))

    manager.update_endpoint(sfm_conn, {HandshakeKeys.ENDPOINT_NAME: "site-1.job-123"})

    assert "site-1.job-123" in manager.sfm_endpoints
    assert not conn.closed


def test_mtls_handshake_accepts_configured_auth_identity_for_site_cert_cn_mismatch():
    manager = _conn_manager(identity_map={"site-1": "custom-site-cn"})
    conn = _FakeConnection(peer_cn="custom-site-cn")
    sfm_conn = SfmConnection(conn, Endpoint("server"))

    manager.update_endpoint(sfm_conn, {HandshakeKeys.ENDPOINT_NAME: "site-1.job-123"})

    assert "site-1.job-123" in manager.sfm_endpoints
    assert not conn.closed


def test_mtls_handshake_rejects_spoofed_endpoint_identity():
    manager = _conn_manager(identity_map={"site-1": "site-1"})
    conn = _FakeConnection(peer_cn="attacker")
    sfm_conn = SfmConnection(conn, Endpoint("server"))

    with pytest.raises(CommError) as ex:
        manager.update_endpoint(sfm_conn, {HandshakeKeys.ENDPOINT_NAME: "site-1.job-123"})

    assert ex.value.code == CommError.BAD_DATA
    assert "site-1.job-123" not in manager.sfm_endpoints
    assert conn.closed


def test_mtls_handshake_accepts_hierarchical_cell_pipe_cell_with_site_cert_identity():
    manager = _conn_manager(identity_map={"site-1": "site-1"})
    conn = _FakeConnection(peer_cn="site-1")
    sfm_conn = SfmConnection(conn, Endpoint("server"))

    manager.update_endpoint(sfm_conn, {HandshakeKeys.ENDPOINT_NAME: "site-1.job-123.active"})

    assert "site-1.job-123.active" in manager.sfm_endpoints
    assert not conn.closed


def test_mtls_handshake_accepts_legacy_cell_pipe_alias_with_site_cert_identity():
    manager = _conn_manager(identity_map={"site-1": "site-1"})
    conn = _FakeConnection(peer_cn="site-1")
    sfm_conn = SfmConnection(conn, Endpoint("server"))

    manager.update_endpoint(sfm_conn, {HandshakeKeys.ENDPOINT_NAME: "site-1_job-123_active"})

    assert "site-1_job-123_active" in manager.sfm_endpoints
    assert not conn.closed


def test_mtls_handshake_rejects_spoofed_cell_pipe_alias_identity():
    manager = _conn_manager(identity_map={"site-1": "site-1", "site-a": "site-a"})
    conn = _FakeConnection(peer_cn="attacker")
    sfm_conn = SfmConnection(conn, Endpoint("server"))

    with pytest.raises(CommError) as ex:
        manager.update_endpoint(sfm_conn, {HandshakeKeys.ENDPOINT_NAME: "site-1_job-123_active"})

    assert ex.value.code == CommError.BAD_DATA
    assert conn.closed

    # An ambiguous alias resolves only to its right-anchored owner, never a shorter site
    conn = _FakeConnection(peer_cn="site-a")
    sfm_conn = SfmConnection(conn, Endpoint("server"))
    with pytest.raises(CommError) as ex:
        manager.update_endpoint(sfm_conn, {HandshakeKeys.ENDPOINT_NAME: "site-a_x_job-123_active"})

    assert ex.value.code == CommError.BAD_DATA
    assert conn.closed


def test_mtls_handshake_rejects_invalid_endpoint_fqcn():
    manager = _conn_manager(identity_map={"server": "server"})
    conn = _FakeConnection(peer_cn="server")
    sfm_conn = SfmConnection(conn, Endpoint("server"))

    with pytest.raises(CommError) as ex:
        manager.update_endpoint(sfm_conn, {HandshakeKeys.ENDPOINT_NAME: "server."})

    assert ex.value.code == CommError.BAD_DATA
    assert "server." not in manager.sfm_endpoints
    assert conn.closed


def test_mtls_handshake_rejects_admin_endpoint_on_non_admin_listener():
    manager = _conn_manager()
    conn = _FakeConnection(peer_cn="admin@nvidia.com")
    sfm_conn = SfmConnection(conn, Endpoint("server"))

    with pytest.raises(CommError) as ex:
        manager.update_endpoint(sfm_conn, {HandshakeKeys.ENDPOINT_NAME: "_admin_9af49fef-235f-41bd-9296-12fd09eacb2a"})

    assert ex.value.code == CommError.BAD_DATA
    assert "_admin_9af49fef-235f-41bd-9296-12fd09eacb2a" not in manager.sfm_endpoints
    assert conn.closed


def test_mtls_handshake_accepts_admin_endpoint_on_admin_listener():
    manager = _conn_manager()
    conn = _FakeConnection(peer_cn="admin@nvidia.com", admin_listener=True)
    sfm_conn = SfmConnection(conn, Endpoint("server"))

    manager.update_endpoint(sfm_conn, {HandshakeKeys.ENDPOINT_NAME: "_admin_9af49fef-235f-41bd-9296-12fd09eacb2a"})

    assert "_admin_9af49fef-235f-41bd-9296-12fd09eacb2a" in manager.sfm_endpoints
    assert not conn.closed


def test_tls_handshake_does_not_enforce_peer_identity():
    manager = _conn_manager(identity_map={"site-1": "site-1"})
    conn = _FakeConnection(peer_cn=None, conn_security=ConnectionSecurity.TLS)
    sfm_conn = SfmConnection(conn, Endpoint("server"))

    manager.update_endpoint(sfm_conn, {HandshakeKeys.ENDPOINT_NAME: "site-1.job-123"})

    assert "site-1.job-123" in manager.sfm_endpoints
    assert not conn.closed


@pytest.mark.parametrize("peer_cn", [None, "N/A"])
def test_mtls_active_connection_without_peer_cn_is_accepted(peer_cn):
    # On the active (connecting) side, some drivers (e.g. gRPC) do not expose the
    # peer CN (it is None or "N/A"). Enforcement must be skipped there so the
    # connecting peer can register the endpoint it dialed out to; otherwise the
    # default gRPC mTLS client->server handshake would always be rejected.
    manager = _conn_manager(local_fqcn="site-1", identity_map={"server": "server"})
    conn = _FakeConnection(peer_cn=peer_cn, mode=Mode.ACTIVE)
    sfm_conn = SfmConnection(conn, Endpoint("site-1"))

    manager.update_endpoint(sfm_conn, {HandshakeKeys.ENDPOINT_NAME: "server"})

    assert "server" in manager.sfm_endpoints
    assert not conn.closed


def test_mtls_active_connection_with_peer_cn_enforces_endpoint_identity():
    manager = _conn_manager(local_fqcn="site-1", identity_map={"server": "server"})
    conn = _FakeConnection(peer_cn="attacker", mode=Mode.ACTIVE)
    sfm_conn = SfmConnection(conn, Endpoint("site-1"))

    with pytest.raises(CommError) as ex:
        manager.update_endpoint(sfm_conn, {HandshakeKeys.ENDPOINT_NAME: "server"})

    assert ex.value.code == CommError.BAD_DATA
    assert "server" not in manager.sfm_endpoints
    assert conn.closed


def test_mtls_certificate_cache_rejects_cert_identity_mismatch():
    resolver = CellIdentityResolver(local_fqcn="server", prefix_identity_map={"site-1": "site-1"})
    manager = CredentialManager(Endpoint("server"), identity_resolver=resolver, enforce_identity=True)
    message = Message(
        headers={MessageHeaderKey.ORIGIN: "site-1.job-123"},
        payload={CERT_CONTENT: _cert_pem("attacker")},
    )

    with pytest.raises(RuntimeError, match="attacker"):
        manager.process_response(message)

    assert "site-1.job-123" not in manager.cert_cache


def test_mtls_certificate_request_rejects_spoofed_origin_identity():
    resolver = CellIdentityResolver(local_fqcn="server", prefix_identity_map={"site-1": "site-1"})
    manager = CredentialManager(Endpoint("server"), identity_resolver=resolver, enforce_identity=True)
    manager.local_cert = _cert_pem("server")
    manager.ca_cert = manager.local_cert
    message = Message(
        headers={
            MessageHeaderKey.ORIGIN: "site-1.job-123",
            MessageHeaderKey.DESTINATION: "server",
        },
        payload={CERT_CONTENT: _cert_pem("attacker")},
    )

    with pytest.raises(RuntimeError, match="attacker"):
        manager.process_request(message)

    assert "site-1.job-123" not in manager.cert_cache


def test_mtls_certificate_cache_accepts_job_cell_parent_cert_identity():
    resolver = CellIdentityResolver(local_fqcn="server", prefix_identity_map={"site-1": "site-1"})
    manager = CredentialManager(Endpoint("server"), identity_resolver=resolver, enforce_identity=True)
    cert = _cert_pem("site-1")
    message = Message(
        headers={MessageHeaderKey.ORIGIN: "site-1.job-123"},
        payload={CERT_CONTENT: cert},
    )

    assert manager.process_response(message) == cert
    assert manager.cert_cache["site-1.job-123"] == cert


def test_mtls_certificate_cache_accepts_configured_auth_identity_for_site_cert_cn_mismatch():
    resolver = CellIdentityResolver(local_fqcn="server", prefix_identity_map={"site-1": "custom-site-cn"})
    manager = CredentialManager(Endpoint("server"), identity_resolver=resolver, enforce_identity=True)
    cert = _cert_pem("custom-site-cn")
    message = Message(
        headers={MessageHeaderKey.ORIGIN: "site-1.job-123"},
        payload={CERT_CONTENT: cert},
    )

    assert manager.process_response(message) == cert
    assert manager.cert_cache["site-1.job-123"] == cert


def test_certificate_cache_access_uses_lock():
    class _TrackingLock:
        def __init__(self):
            self.enter_count = 0

        def __enter__(self):
            self.enter_count += 1

        def __exit__(self, exc_type, exc_value, traceback):
            pass

    manager = CredentialManager(Endpoint("server"))
    manager.cell_cipher = object()
    manager.lock = _TrackingLock()
    cert = _cert_pem("site-1")

    manager._cache_cert("site-1", cert)
    assert manager.get_certificate("site-1") == cert
    assert manager.lock.enter_count == 2


def test_failed_certificate_exchange_request_closes_connection_after_reply():
    cell = _receive_test_core_cell(make_reply(ReturnCode.PROCESS_EXCEPTION))
    conn = _FakeConnection(peer_cn="site-1")
    message = Message(
        headers={
            MessageHeaderKey.MSG_TYPE: MessageType.REQ,
            MessageHeaderKey.ORIGIN: "site-1",
            MessageHeaderKey.DESTINATION: "server",
            MessageHeaderKey.FROM_CELL: "site-1",
            MessageHeaderKey.TO_CELL: "server",
            MessageHeaderKey.CHANNEL: "credential_manager",
            MessageHeaderKey.TOPIC: "key_exchange",
            MessageHeaderKey.REPLY_EXPECTED: True,
            MessageHeaderKey.REQ_ID: "req-1",
        }
    )

    CoreCell._process_received_msg(cell, Endpoint("site-1"), conn, message)

    assert cell.sent_reply is not None
    assert conn.closed
