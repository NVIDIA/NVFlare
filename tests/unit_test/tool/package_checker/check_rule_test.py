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

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from cryptography import x509
from cryptography.x509.oid import ExtendedKeyUsageOID

from nvflare.lighter.utils import Identity, generate_cert, generate_keys, serialize_cert, serialize_pri_key
from nvflare.tool.package_checker import ClientPackageChecker, ServerPackageChecker
from nvflare.tool.package_checker.check_rule import CHECK_PASSED, CheckEndpointCredentials, CheckServerAvailable
from nvflare.tool.package_checker.package_checker import CheckStatus


@pytest.fixture(scope="module")
def issuer():
    root_key, root_pub = generate_keys()
    root = generate_cert(Identity("root"), Identity("root"), root_key, root_pub, ca=True)
    key, pub = generate_keys()
    intermediate = generate_cert(Identity("issuer"), root.subject, root_key, pub, ca=True)
    return root, intermediate, key


@pytest.mark.parametrize(
    "role,condition,expected",
    [
        ("server", "good", CHECK_PASSED),
        ("client", "good", CHECK_PASSED),
        ("client", "no EKU", CHECK_PASSED),
        ("client", "missing cert", "startup/identity.pem: No such file"),
        ("server", "missing key", "startup/identity.key: No such file"),
        ("server", "malformed cert", "startup/identity.pem:"),
        ("client", "malformed key", "startup/identity.key:"),
        ("server", "wrong key", "startup/identity.key does not match"),
        ("client", "expired", "startup/identity.pem expired"),
        ("server", "future", "startup/identity.pem is not valid until"),
        ("client", "wrong root", "failed chain validation against startup/trust.pem"),
        ("server", "wrong EKU", "cannot be used for serverAuth"),
        ("client", "wrong EKU", "cannot be used for clientAuth"),
    ],
)
def test_endpoint_credentials(tmp_path, issuer, role, condition, expected):
    root, intermediate, signing_key = issuer
    key, pub = generate_keys()
    now = datetime.now(timezone.utc)
    day = timedelta(days=1)
    server_usage = (role == "server") != (condition == "wrong EKU")
    eku = ExtendedKeyUsageOID.SERVER_AUTH if server_usage else ExtendedKeyUsageOID.CLIENT_AUTH
    leaf = generate_cert(
        Identity("endpoint"),
        intermediate.subject,
        signing_key,
        pub,
        not_valid_before=now + day if condition == "future" else now - 2 * day,
        not_valid_after=now - day if condition == "expired" else now + 2 * day,
        extra_extensions=[] if condition == "no EKU" else [(x509.ExtendedKeyUsage([eku]), False)],
    )
    startup = tmp_path / "startup"
    startup.mkdir()
    config = {"ssl_cert": "identity.pem", "ssl_private_key": "identity.key", "ssl_root_cert": "trust.pem"}
    fed_config = {"servers": [config]} if role == "server" else {"client": config}
    (startup / f"fed_{role}.json").write_text(json.dumps(fed_config))
    trust = (
        generate_cert(Identity("other"), Identity("other"), key, pub, ca=True) if condition == "wrong root" else root
    )
    (startup / "trust.pem").write_bytes(serialize_cert(trust))
    if condition != "missing cert":
        chain = serialize_cert(leaf) + serialize_cert(intermediate)
        (startup / "identity.pem").write_bytes(b"bad PEM" if condition == "malformed cert" else chain)
    if condition != "missing key":
        pem = serialize_pri_key(signing_key if condition == "wrong key" else key)
        (startup / "identity.key").write_bytes(b"bad PEM" if condition == "malformed key" else pem)

    result = CheckEndpointCredentials(role)(str(tmp_path))
    assert expected in result.problem


@pytest.mark.parametrize("security", ["clear", "tls", "mtls"])
def test_missing_client_identity_stops_before_connectivity_and_dry_run(tmp_path, issuer, security):
    startup = tmp_path / "startup"
    startup.mkdir()
    config = {"ssl_cert": "client.crt", "ssl_private_key": "client.key", "ssl_root_cert": "rootCA.pem"}
    (startup / "rootCA.pem").write_bytes(serialize_cert(issuer[0]))
    (startup / "fed_client.json").write_text(json.dumps({"client": dict(config, connection_security=security)}))
    checker = ClientPackageChecker()
    checker.init(str(tmp_path))
    with patch.object(CheckServerAvailable, "__call__") as probe, patch.object(checker, "check_dry_run") as dry_run:
        assert checker.check() == CheckStatus.FAIL
    assert "startup/client.crt" in checker.report[str(tmp_path)][0][1]
    probe.assert_not_called()
    dry_run.assert_not_called()


@pytest.mark.parametrize("scheme", ["atcp", "satcp"])
def test_server_accepts_async_scheme_and_checks_credentials(tmp_path, scheme):
    startup = tmp_path / "startup"
    startup.mkdir()
    server = {
        "service": {"scheme": scheme, "target": "localhost:8002"},
        "admin_port": 8003,
        "ssl_root_cert": "missing.pem",
    }
    (startup / "fed_server.json").write_text(json.dumps({"servers": [server]}))
    checker = ServerPackageChecker()
    checker.init(str(tmp_path))
    with (
        patch("nvflare.tool.package_checker.check_rule.try_bind_address", return_value=None),
        patch.object(checker, "check_dry_run") as dry_run,
    ):
        assert checker.check() == CheckStatus.FAIL
    assert "startup/missing.pem" in checker.report[str(tmp_path)][0][1]
    dry_run.assert_not_called()
