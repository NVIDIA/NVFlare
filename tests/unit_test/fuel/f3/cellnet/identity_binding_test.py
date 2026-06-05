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
from types import SimpleNamespace

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from nvflare.apis.fl_constant import ConnectionSecurity
from nvflare.fuel.f3.cellnet.credential_manager import CERT_CONTENT, CredentialManager
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.cellnet.identity import CellIdentityResolver
from nvflare.fuel.f3.comm_error import CommError
from nvflare.fuel.f3.drivers.driver_params import DriverParams
from nvflare.fuel.f3.endpoint import Endpoint
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.sfm.conn_manager import ConnManager
from nvflare.fuel.f3.sfm.constants import HandshakeKeys
from nvflare.fuel.f3.sfm.sfm_conn import SfmConnection
from nvflare.fuel.utils.constants import Mode


class _FakeConnection:
    def __init__(self, peer_cn, conn_security=ConnectionSecurity.MTLS, mode=Mode.PASSIVE):
        self.name = "CN-test"
        self.closed = False
        self.connector = SimpleNamespace(
            mode=mode,
            params={
                DriverParams.SECURE: True,
                DriverParams.CONNECTION_SECURITY: conn_security,
            },
        )
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
