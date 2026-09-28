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

import ssl
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from nvflare.apis.fl_constant import FLContextKey, SecureTrainConst
from nvflare.apis.fl_context import FLContext
from nvflare.fuel.f3.cellnet.cell_cipher import InvalidCertChain
from nvflare.fuel.f3.cellnet.connector_manager import ConnectorManager
from nvflare.fuel.f3.cellnet.core_cell import CertificateExchanger
from nvflare.fuel.f3.cellnet.credential_manager import CredentialManager
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.comm_error import CommError
from nvflare.fuel.f3.drivers.connector_info import Mode
from nvflare.fuel.f3.drivers.grpc import utils as grpc_utils
from nvflare.fuel.f3.drivers.net_utils import get_ssl_context
from nvflare.fuel.f3.drivers.tcp_driver import TcpDriver
from nvflare.fuel.f3.endpoint import Endpoint
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.sfm.conn_manager import ConnManager
from nvflare.fuel.sec.security_content_service import LoadResult, SecurityContentManager, SecurityContentService
from nvflare.lighter.utils import (
    Identity,
    generate_cert,
    generate_keys,
    serialize_cert,
    serialize_pri_key,
    sign_folders,
)
from nvflare.private.fed.app.deployer.server_deployer import ServerDeployer
from nvflare.private.fed.server.cred_keeper import CredKeeper
from nvflare.security.certificate_renewal import watch_credentials


def issue(root_key, root_cert, key, name, seconds=3600, *, start=-60, san=None, eku=None, ca=False, aki=True):
    now = datetime.now(timezone.utc)
    builder = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)]))
        .issuer_name(root_cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now + timedelta(seconds=start))
        .not_valid_after(now + timedelta(seconds=seconds))
        .add_extension(x509.BasicConstraints(ca=ca, path_length=None), critical=True)
        .add_extension(x509.KeyUsage(True, False, True, False, False, ca, ca, False, False), critical=True)
        .add_extension(
            x509.ExtendedKeyUsage(
                eku if eku is not None else [ExtendedKeyUsageOID.SERVER_AUTH, ExtendedKeyUsageOID.CLIENT_AUTH]
            ),
            critical=False,
        )
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        .add_extension(x509.SubjectAlternativeName([x509.DNSName(san or name)]), critical=False)
    )
    if aki:
        builder = builder.add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(root_key.public_key()), critical=False
        )
    return serialize_cert(builder.sign(root_key, hashes.SHA256()))


def pki(directory, seconds=3600):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    root_key, root_pub = generate_keys()
    root = generate_cert(Identity("root"), Identity("root"), root_key, root_pub, ca=True)
    (directory / "rootCA.pem").write_bytes(serialize_cert(root))
    key, _ = generate_keys()
    (directory / "server.key").write_bytes(serialize_pri_key(key))
    (directory / "server.crt").write_bytes(issue(root_key, root, key, "localhost", seconds))
    params = {
        "ca_cert": str(directory / "rootCA.pem"),
        "server_cert": str(directory / "server.crt"),
        "server_key": str(directory / "server.key"),
        "connection_security": "mtls",
    }
    return params, root_key, root, key


@pytest.fixture
def enrolled(tmp_path):
    params, root_key, root, key = pki(tmp_path)
    params["certificate_renewal"] = True
    credential = watch_credentials(params, "localhost")[0]
    return params, credential, root_key, root, key


def test_same_key_renewal_is_loaded_without_waiting_for_watcher(enrolled):
    params, credential, root_key, root, key = enrolled
    manager = CredentialManager(Endpoint("server", conn_props=params))
    keeper = CredKeeper()
    ctx = FLContext()
    ctx.set_prop(
        FLContextKey.SERVER_CONFIG,
        [
            {
                SecureTrainConst.SSL_CERT: params["server_cert"],
                SecureTrainConst.PRIVATE_KEY: params["server_key"],
            }
        ],
    )
    with patch("nvflare.private.fed.server.cred_keeper.CommConfigurator") as config:
        config.return_value.certificate_renewal_enabled.return_value = True
        old = keeper.get_id_asserter(ctx)
        assert keeper.get_id_asserter(ctx) is old
        renewed = issue(root_key, root, key, "localhost", seconds=7200)
        Path(params["server_cert"]).write_bytes(renewed)
        assert keeper.get_id_asserter(ctx).cert_data == renewed
        assert old.cert_data != renewed
        Path(params["server_cert"]).write_bytes(b"incomplete")
        with pytest.raises(ValueError):
            keeper.get_id_asserter(ctx)
        Path(params["server_cert"]).write_bytes(renewed)
    assert manager.create_request()["cert_content"] == renewed
    assert (
        manager.process_request(
            Message(
                {
                    MessageHeaderKey.ORIGIN: "peer",
                    MessageHeaderKey.DESTINATION: "server",
                },
                {"cert_content": renewed},
            )
        )["cert_content"]
        == renewed
    )
    assert credential.refresh()
    assert credential.status()["last_observed"] is not None
    assert not credential.refresh()


@pytest.mark.parametrize("failure", ["identity", "key", "untrusted", "expired", "future", "ca", "malformed"])
def test_invalid_file_is_reported_without_restoring_it(enrolled, failure):
    params, credential, root_key, root, key = enrolled
    kwargs = {"seconds": 7200}
    name = "localhost"
    if failure == "identity":
        name = "intruder"
    elif failure == "key":
        key, _ = generate_keys()
    elif failure == "untrusted":
        root_key, _ = generate_keys()
    elif failure == "expired":
        kwargs["seconds"] = -1
    elif failure == "future":
        kwargs["start"] = 600
    elif failure == "ca":
        kwargs["ca"] = True
    data = b"incomplete" if failure == "malformed" else issue(root_key, root, key, name, **kwargs)
    Path(params["server_cert"]).write_bytes(data)
    assert not credential.refresh()
    assert credential.status()["error"]
    assert Path(params["server_cert"]).read_bytes() == data


def test_malformed_update_breaks_new_loads_and_valid_replacement_recovers(enrolled):
    params, credential, root_key, root, key = enrolled
    params = dict(params, scheme="stcp")
    Path(params["server_cert"]).write_bytes(b"incomplete")
    assert not credential.refresh()
    with pytest.raises(ssl.SSLError):
        get_ssl_context(params, True)
    Path(params["server_cert"]).write_bytes(issue(root_key, root, key, "localhost", 7200))
    assert get_ssl_context(params, True)
    assert credential.refresh()
    assert credential.status()["error"] is None


def test_restoring_original_file_clears_diagnostic(enrolled):
    params, credential, *_ = enrolled
    path = Path(params["server_cert"])
    original = path.read_bytes()
    path.write_bytes(b"incomplete")
    assert not credential.refresh()
    path.write_bytes(original)
    assert credential.refresh()
    assert credential.status()["error"] is None


def test_watcher_detects_file_changes_even_when_leaf_is_unchanged(enrolled):
    params, credential, *_ = enrolled
    path = Path(params["server_cert"])
    path.write_bytes(path.read_bytes() + b"\n")
    assert credential.refresh()
    assert not credential.refresh()


def test_changed_key_requires_restart_of_watcher(enrolled):
    params, credential, root_key, root, _ = enrolled
    replacement = generate_keys()[0]
    Path(params["server_key"]).write_bytes(serialize_pri_key(replacement))
    Path(params["server_cert"]).write_bytes(issue(root_key, root, replacement, "localhost", 7200))
    assert not credential.refresh()
    assert "restart" in credential.status()["error"]
    assert watch_credentials(params, "localhost")[0].public_key != credential.public_key


def test_missing_initial_credential_is_actionable(tmp_path):
    params, *_ = pki(tmp_path)
    Path(params["server_cert"]).unlink()
    with pytest.raises(ValueError, match="Enroll first"):
        watch_credentials(params, "localhost")


@pytest.mark.parametrize("missing, message", [("server_key", "private-key"), ("ca_cert", "CA certificate")])
def test_missing_credential_path_is_actionable(enrolled, missing, message):
    params, *_ = enrolled
    del params[missing]
    with pytest.raises(ValueError, match=f"Enroll first:.*{message} path is missing"):
        watch_credentials(params, "localhost")


@pytest.mark.parametrize("signed", [False, True])
def test_signed_startup_kit_rejects_renewal(enrolled, tmp_path, monkeypatch, signed):
    params, _, root_key, *_ = enrolled
    if signed:
        sign_folders(str(tmp_path), root_key, signature_file="signature.json")
    manager = SecurityContentManager(str(tmp_path))
    monkeypatch.setattr(SecurityContentService, "security_content_manager", manager)
    if signed:
        assert manager.load_content("server.crt")[1] == LoadResult.OK
        with pytest.raises(ValueError, match="certificate_renewal.*signed HE/CC"):
            watch_credentials(params, "localhost")
    else:
        assert len(watch_credentials(params, "localhost")) == 1


def _handshake_peer_certificate(server_context, client_context, hostname):
    server_in, server_out, client_in, client_out = (ssl.MemoryBIO() for _ in range(4))
    server = server_context.wrap_bio(server_in, server_out, server_side=True)
    client = client_context.wrap_bio(client_in, client_out, server_hostname=hostname)
    done = set()
    for _ in range(20):
        for connection, outbound, incoming in ((client, client_out, server_in), (server, server_out, client_in)):
            if connection not in done:
                try:
                    connection.do_handshake()
                    done.add(connection)
                except ssl.SSLWantReadError:
                    pass
            data = outbound.read()
            if data:
                incoming.write(data)
        if len(done) == 2:
            return client.getpeercert(binary_form=True)
    raise AssertionError("TLS handshake did not complete")


@pytest.mark.parametrize("version", [ssl.TLSVersion.TLSv1_2, ssl.TLSVersion.TLSv1_3])
@pytest.mark.parametrize("hostname", [None, "localhost"])
def test_tls_handshake_reloads_certificate_without_mutating_connector_params(enrolled, version, hostname):
    params, _, root_key, root, key = enrolled
    params = dict(params, scheme="stcp")
    context = get_ssl_context(params, True)
    params["implemented_conn_sec"] = "unchanged by callback"
    before = dict(params)
    client = ssl.create_default_context(cafile=params["ca_cert"])
    client.check_hostname = False
    client.minimum_version = client.maximum_version = version
    client.load_cert_chain(params["server_cert"], params["server_key"])
    original = _handshake_peer_certificate(context, client, hostname)
    renewed = issue(root_key, root, key, "localhost", seconds=7200)
    Path(params["server_cert"]).write_bytes(renewed)
    expected = x509.load_pem_x509_certificate(renewed).public_bytes(serialization.Encoding.DER)
    assert expected != original
    assert _handshake_peer_certificate(context, client, hostname) == expected
    assert params == before
    # The listener's original context still holds the old cert; the callback selects a fresh one.
    context.sni_callback = None
    assert _handshake_peer_certificate(context, client, hostname) == original


@pytest.mark.parametrize("role, renewal", [("client", False), ("server", False), ("server", True)])
def test_grpc_does_not_substitute_opposite_role_credentials(enrolled, role, renewal):
    params, *_ = enrolled
    other = "server" if role == "client" else "client"
    params = {
        "ca_cert": params["ca_cert"],
        other + "_cert": params["server_cert"],
        other + "_key": params["server_key"],
        "certificate_renewal": renewal,
    }
    with (
        patch.object(grpc_utils.grpc, "ssl_channel_credentials") as client,
        patch.object(grpc_utils.grpc, "ssl_server_credentials") as server,
        patch.object(grpc_utils.grpc, "dynamic_ssl_server_credentials"),
        patch.object(grpc_utils.grpc, "ssl_server_certificate_configuration") as dynamic,
    ):
        if role == "client":
            grpc_utils.get_grpc_client_credentials(params)
            assert client.call_args.kwargs["certificate_chain"] is None
            assert client.call_args.kwargs["private_key"] is None
        else:
            grpc_utils.get_grpc_server_credentials(params)
            assert (dynamic if renewal else server).call_args.args[0] == [(None, None)]


def test_connection_override_cannot_downgrade_renewable_mtls(enrolled):
    params, *_ = enrolled
    manager = ConnManager(Endpoint("server", conn_props=params))
    try:
        with pytest.raises(CommError, match="end-to-end mTLS"):
            manager.add_connector(
                TcpDriver(),
                dict(params, scheme="stcp", url="stcp://localhost:1234", connection_security="tls"),
                Mode.ACTIVE,
            )
    finally:
        manager.stop()


def test_status_reports_earliest_chain_expiry(tmp_path):
    params, root_key, root, key = pki(tmp_path)
    Path(params["server_cert"]).write_bytes(issue(root_key, root, key, "localhost", seconds=400 * 86400))
    credential = watch_credentials(params, "localhost")[0]
    assert credential.expires == root.not_valid_after_utc.timestamp()


@pytest.mark.parametrize("scheme", ["tcp", "stcp", "grpcs", "https"])
@pytest.mark.parametrize("explicit", [None, "mtls"])
def test_internal_security_default_is_unchanged(scheme, explicit):
    config = MagicMock()
    config.get_config.return_value = (
        {"internal": {"scheme": scheme, "resources": {"connection_security": explicit}}} if explicit else {}
    )
    config.get_internal_connection_scheme.return_value = scheme
    manager = ConnectorManager(MagicMock(), True, config)
    assert manager.int_resources["connection_security"] == (explicit or "clear")


@pytest.mark.parametrize("explicit", [None, "configured-server"])
@pytest.mark.parametrize("host", ["server.example", "0", "0.0.0.0"])
def test_bind_override_does_not_change_expected_identity(explicit, host):
    deployer = ServerDeployer()
    config = {"service": {"target": f"{host}:8002"}}
    if explicit:
        config["auth_identity"] = explicit
    deployer.server_config = [config]
    deployer.host = "0.0.0.0"
    with (
        patch("nvflare.private.fed.app.deployer.server_deployer.CommConfigurator") as configurator,
        patch("nvflare.private.fed.app.deployer.server_deployer.FederatedServer"),
    ):
        configurator.return_value.certificate_renewal_enabled.return_value = True
        deployed, _ = deployer.create_fl_server(MagicMock())
    assert deployed["service"]["target"] == "0.0.0.0:8002"
    assert deployed.get("auth_identity") == (explicit or (host if host == "server.example" else None))


@pytest.mark.parametrize("managed", [False, True])
def test_cached_peer_refresh_is_throttled_and_does_not_parse_on_lookup(enrolled, managed):
    params, credential, root_key, root, key = enrolled
    if not managed:
        params = dict(params, certificate_renewal=False)
    manager = CredentialManager(Endpoint("server", conn_props=params))
    cert = issue(root_key, root, key, "localhost", seconds=20)
    now = datetime.now(timezone.utc).timestamp()
    with patch("nvflare.fuel.f3.cellnet.credential_manager.time.time", return_value=now):
        manager._cache_cert("peer", cert)
    with patch("nvflare.fuel.f3.cellnet.credential_manager.time.time", return_value=now + 6):
        with patch("nvflare.fuel.f3.cellnet.credential_manager.x509.load_pem_x509_certificates") as parse:
            assert manager.get_certificate("peer") is None
            assert manager.get_certificate("peer") == cert
            parse.assert_not_called()
        manager._cache_cert("peer", cert)  # issuer has not renewed yet
        assert manager.get_certificate("peer") == cert


def test_unmanaged_client_refreshes_renewed_server_certificate_for_encrypted_reply(enrolled, tmp_path):
    params, credential, root_key, root, server_key = enrolled
    server = CredentialManager(Endpoint("server", conn_props=params))
    old_cert = Path(params["server_cert"]).read_bytes()
    old_expiry = credential.expires
    client_key, _ = generate_keys()
    client_cert = issue(root_key, root, client_key, "client", seconds=7200)
    (tmp_path / "client.crt").write_bytes(client_cert)
    (tmp_path / "client.key").write_bytes(serialize_pri_key(client_key))
    client = CredentialManager(
        Endpoint(
            "client",
            conn_props={
                "ca_cert": params["ca_cert"],
                "client_cert": str(tmp_path / "client.crt"),
                "client_key": str(tmp_path / "client.key"),
            },
        )
    )
    assert not client._renewal
    cell = MagicMock()
    cell.send_request.side_effect = lambda *args: Message({MessageHeaderKey.ORIGIN: "server"}, server.create_request())
    exchanger = CertificateExchanger(cell, client)
    assert exchanger.get_certificate("server") == old_cert
    assert server.decrypt(client_cert, client.encrypt(old_cert, b"request")) == b"request"
    renewed = issue(root_key, root, server_key, "localhost", seconds=7200)
    Path(params["server_cert"]).write_bytes(renewed)
    assert credential.refresh()
    after_expiry = datetime.fromtimestamp(old_expiry + 2, timezone.utc)
    with (
        patch("nvflare.fuel.f3.cellnet.credential_manager.time.time", return_value=after_expiry.timestamp()),
        patch("nvflare.lighter.utils._utc_now", return_value=after_expiry),
    ):
        # The first reply has no cached decryption key, so it validates the peer chain.
        cipher = server.encrypt(client_cert, b"reply")
        with pytest.raises(InvalidCertChain):
            client.decrypt(old_cert, cipher)
        assert client.decrypt(exchanger.get_certificate("server"), cipher) == b"reply"
        assert client.get_certificate("server") == renewed
    assert cell.send_request.call_count == 2
