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

"""Unit tests for nvflare package command handler."""

import json
import os
import shutil
import stat
import types

import pytest

from nvflare.lighter.utils import Identity, generate_cert, generate_keys, serialize_cert, serialize_pri_key
from nvflare.tool.package.package_commands import (
    _discover_name_from_dir,
    _make_fed_admin_json,
    _make_fed_client_json,
    _make_fed_server_json,
    _parse_endpoint,
    handle_package,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ca(tmp_dir):
    """Generate a self-signed CA and write rootCA.pem + ca.key into tmp_dir."""
    ca_key, ca_pub = generate_keys()
    ca_id = Identity("test-ca", "TestOrg")
    ca_cert = generate_cert(ca_id, ca_id, ca_key, ca_pub, ca=True)
    rootca_path = os.path.join(tmp_dir, "rootCA.pem")
    with open(rootca_path, "wb") as f:
        f.write(serialize_cert(ca_cert))
    return ca_key, ca_cert, rootca_path


def _make_signed_cert(ca_key, ca_cert, name, tmp_dir, cert_filename):
    """Generate a key + CA-signed cert, write them into tmp_dir."""
    key, pub = generate_keys()
    subj = Identity(name, "TestOrg")
    issuer = Identity(
        ca_cert.subject.get_attributes_for_oid(
            __import__("cryptography.x509.oid", fromlist=["NameOID"]).NameOID.COMMON_NAME
        )[0].value,
        "TestOrg",
    )
    cert = generate_cert(subj, issuer, ca_key, pub)
    key_path = os.path.join(tmp_dir, f"{name}.key")
    cert_path = os.path.join(tmp_dir, cert_filename)
    with open(key_path, "wb") as f:
        f.write(serialize_pri_key(key))
    with open(cert_path, "wb") as f:
        f.write(serialize_cert(cert))
    return key_path, cert_path


def _make_args(**kwargs):
    """Build a minimal namespace for handle_package."""
    defaults = dict(
        kit_type="client",
        endpoint=None,
        name=None,
        dir=None,
        cert=None,
        key=None,
        rootca=None,
        output_dir=None,
        server_name=None,
        admin_port=None,
        require_signed_jobs=None,
        force=True,
        output_fmt=None,
        schema=False,
    )
    defaults.update(kwargs)
    return types.SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# Unit tests: _parse_endpoint
# ---------------------------------------------------------------------------


class TestParseEndpoint:
    def test_grpc_valid(self):
        scheme, host, port = _parse_endpoint("grpc://server.example.com:8002")
        assert scheme == "grpc"
        assert host == "server.example.com"
        assert port == 8002

    def test_tcp_valid(self):
        scheme, host, port = _parse_endpoint("tcp://192.168.1.10:9000")
        assert scheme == "tcp"
        assert host == "192.168.1.10"
        assert port == 9000

    def test_invalid_scheme_raises(self):
        with pytest.raises(ValueError):
            _parse_endpoint("http://server:8002")

    def test_missing_port_raises(self):
        with pytest.raises(ValueError):
            _parse_endpoint("grpc://server")

    def test_empty_host_raises(self):
        with pytest.raises(ValueError):
            _parse_endpoint("grpc://:8002")


# ---------------------------------------------------------------------------
# Unit tests: _discover_name_from_dir
# ---------------------------------------------------------------------------


class TestDiscoverNameFromDir:
    def test_single_key_detected(self, tmp_path):
        (tmp_path / "alice.key").write_text("dummy")
        name = _discover_name_from_dir(str(tmp_path), None)
        assert name == "alice"

    def test_no_key_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            _discover_name_from_dir(str(tmp_path), None)

    def test_multiple_keys_exits(self, tmp_path):
        (tmp_path / "alice.key").write_text("dummy")
        (tmp_path / "bob.key").write_text("dummy")
        with pytest.raises(SystemExit):
            _discover_name_from_dir(str(tmp_path), None)


# ---------------------------------------------------------------------------
# Unit tests: config JSON builders
# ---------------------------------------------------------------------------


class TestMakeFedClientJson:
    def test_structure(self):
        cfg = _make_fed_client_json("hospital-1", "grpc", "server.example.com", 8002, 8003, "fl-server")
        assert cfg["format_version"] == 2
        assert cfg["servers"][0]["name"] == "fl-server"
        assert cfg["servers"][0]["service"]["scheme"] == "grpc"
        assert cfg["client"]["fqsn"] == "hospital-1"
        assert cfg["client"]["connection_security"] == "mtls"
        assert cfg["overseer_agent"]["args"]["sp_end_point"] == "server.example.com:8002:8003"

    def test_admin_port_default_offset(self):
        cfg = _make_fed_client_json("h1", "grpc", "srv", 8002, 8003, "srv")
        assert "8002:8003" in cfg["overseer_agent"]["args"]["sp_end_point"]


class TestMakeFedServerJson:
    def test_structure(self):
        cfg = _make_fed_server_json("fl-server", "grpc", 8002, 8003, True)
        assert cfg["format_version"] == 2
        assert cfg["require_signed_jobs"] is True
        assert cfg["servers"][0]["service"]["target"] == "0.0.0.0:8002"
        assert cfg["servers"][0]["admin_port"] == 8003
        assert cfg["overseer_agent"]["args"]["sp_end_point"] == "fl-server:8002:8003"

    def test_require_signed_jobs_false(self):
        cfg = _make_fed_server_json("srv", "grpc", 8002, 8003, False)
        assert cfg["require_signed_jobs"] is False


class TestMakeFedAdminJson:
    def test_structure(self):
        cfg = _make_fed_admin_json("admin-alice", "fl-server", "grpc", "server.example.com", 8003)
        adm = cfg["admin"]
        assert adm["username"] == "admin-alice"
        assert adm["server_identity"] == "fl-server"
        assert adm["host"] == "server.example.com"
        assert adm["port"] == 8003
        assert adm["connection_security"] == "mtls"


# ---------------------------------------------------------------------------
# Integration tests: full kit assembly
# ---------------------------------------------------------------------------


@pytest.fixture()
def cert_env(tmp_path):
    """Provide a temporary directory with a CA, signed cert, and key."""
    ca_key, ca_cert, rootca_path = _make_ca(str(tmp_path))
    return dict(tmp_path=tmp_path, ca_key=ca_key, ca_cert=ca_cert, rootca_path=rootca_path)


class TestClientKitAssembly:
    def test_basic_explicit_mode(self, cert_env, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        rootca = cert_env["rootca_path"]

        key_path, cert_path = _make_signed_cert(ca_key, ca_cert, "hospital-1", str(work), "client.crt")

        out_dir = str(tmp_path / "output")
        args = _make_args(
            kit_type="client",
            name="hospital-1",
            endpoint="grpc://server.example.com:8002",
            cert=cert_path,
            key=key_path,
            rootca=rootca,
            output_dir=out_dir,
            force=True,
        )
        handle_package(args)

        startup = os.path.join(out_dir, "startup")
        assert os.path.isfile(os.path.join(startup, "client.crt"))
        assert os.path.isfile(os.path.join(startup, "client.key"))
        assert os.path.isfile(os.path.join(startup, "rootCA.pem"))
        assert os.path.isfile(os.path.join(startup, "fed_client.json"))
        assert os.path.isfile(os.path.join(startup, "start.sh"))
        assert os.path.isfile(os.path.join(startup, "sub_start.sh"))
        assert os.path.isfile(os.path.join(startup, "stop_fl.sh"))

        local = os.path.join(out_dir, "local")
        assert os.path.isfile(os.path.join(local, "resources.json.default"))
        assert os.path.isfile(os.path.join(local, "log_config.json.default"))
        assert os.path.isfile(os.path.join(local, "privacy.json.sample"))
        assert os.path.isfile(os.path.join(local, "authorization.json.default"))

    def test_key_permissions(self, cert_env, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        rootca = cert_env["rootca_path"]

        key_path, cert_path = _make_signed_cert(ca_key, ca_cert, "hospital-1", str(work), "client.crt")
        out_dir = str(tmp_path / "output")
        args = _make_args(
            kit_type="client",
            name="hospital-1",
            endpoint="grpc://server.example.com:8002",
            cert=cert_path,
            key=key_path,
            rootca=rootca,
            output_dir=out_dir,
        )
        handle_package(args)

        key_file = os.path.join(out_dir, "startup", "client.key")
        file_mode = stat.S_IMODE(os.stat(key_file).st_mode)
        assert file_mode == 0o600

    def test_fed_client_json_content(self, cert_env, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        rootca = cert_env["rootca_path"]

        key_path, cert_path = _make_signed_cert(ca_key, ca_cert, "hospital-1", str(work), "client.crt")
        out_dir = str(tmp_path / "output")
        args = _make_args(
            kit_type="client",
            name="hospital-1",
            endpoint="grpc://server.example.com:8002",
            cert=cert_path,
            key=key_path,
            rootca=rootca,
            output_dir=out_dir,
            server_name="fl-server",
        )
        handle_package(args)

        with open(os.path.join(out_dir, "startup", "fed_client.json")) as f:
            cfg = json.load(f)
        assert cfg["servers"][0]["name"] == "fl-server"
        assert cfg["client"]["fqsn"] == "hospital-1"
        assert "8002:8003" in cfg["overseer_agent"]["args"]["sp_end_point"]

    def test_dir_mode_auto_discovery(self, cert_env, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        rootca_src = cert_env["rootca_path"]

        key_path, cert_path = _make_signed_cert(ca_key, ca_cert, "hospital-1", str(work), "client.crt")
        shutil.copy2(rootca_src, str(work / "rootCA.pem"))

        out_dir = str(tmp_path / "output")
        args = _make_args(
            kit_type="client",
            endpoint="grpc://server.example.com:8002",
            dir=str(work),
            output_dir=out_dir,
        )
        handle_package(args)

        assert args.name == "hospital-1"
        assert os.path.isfile(os.path.join(out_dir, "startup", "fed_client.json"))

    def test_dir_mode_name_explicit(self, cert_env, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        rootca_src = cert_env["rootca_path"]

        key_path, cert_path = _make_signed_cert(ca_key, ca_cert, "hospital-1", str(work), "client.crt")
        shutil.copy2(rootca_src, str(work / "rootCA.pem"))

        out_dir = str(tmp_path / "output")
        args = _make_args(
            kit_type="client",
            name="hospital-1",
            endpoint="grpc://server.example.com:8002",
            dir=str(work),
            output_dir=out_dir,
        )
        handle_package(args)

        assert os.path.isfile(os.path.join(out_dir, "startup", "fed_client.json"))


class TestServerKitAssembly:
    def test_basic_explicit_mode(self, cert_env, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        rootca = cert_env["rootca_path"]

        key_path, cert_path = _make_signed_cert(ca_key, ca_cert, "fl-server", str(work), "server.crt")
        out_dir = str(tmp_path / "output")
        args = _make_args(
            kit_type="server",
            name="fl-server",
            endpoint="grpc://0.0.0.0:8002",
            cert=cert_path,
            key=key_path,
            rootca=rootca,
            output_dir=out_dir,
        )
        handle_package(args)

        startup = os.path.join(out_dir, "startup")
        assert os.path.isfile(os.path.join(startup, "server.crt"))
        assert os.path.isfile(os.path.join(startup, "server.key"))
        assert os.path.isfile(os.path.join(startup, "rootCA.pem"))
        assert os.path.isfile(os.path.join(startup, "fed_server.json"))
        assert os.path.isfile(os.path.join(startup, "authorization.json"))
        assert os.path.isfile(os.path.join(startup, "start.sh"))
        assert os.path.isfile(os.path.join(startup, "sub_start.sh"))
        assert os.path.isfile(os.path.join(startup, "stop_fl.sh"))

        local = os.path.join(out_dir, "local")
        assert os.path.isfile(os.path.join(local, "resources.json.default"))
        assert os.path.isfile(os.path.join(local, "log_config.json.default"))
        assert os.path.isfile(os.path.join(local, "privacy.json.sample"))

    def test_fed_server_json_content(self, cert_env, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        rootca = cert_env["rootca_path"]

        key_path, cert_path = _make_signed_cert(ca_key, ca_cert, "fl-server", str(work), "server.crt")
        out_dir = str(tmp_path / "output")
        args = _make_args(
            kit_type="server",
            name="fl-server",
            endpoint="grpc://0.0.0.0:8002",
            cert=cert_path,
            key=key_path,
            rootca=rootca,
            output_dir=out_dir,
            require_signed_jobs=False,
        )
        handle_package(args)

        with open(os.path.join(out_dir, "startup", "fed_server.json")) as f:
            cfg = json.load(f)
        assert cfg["require_signed_jobs"] is False
        assert cfg["servers"][0]["service"]["target"] == "0.0.0.0:8002"
        assert cfg["servers"][0]["admin_port"] == 8003

    def test_key_permissions(self, cert_env, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        rootca = cert_env["rootca_path"]

        key_path, cert_path = _make_signed_cert(ca_key, ca_cert, "fl-server", str(work), "server.crt")
        out_dir = str(tmp_path / "output")
        args = _make_args(
            kit_type="server",
            name="fl-server",
            endpoint="grpc://0.0.0.0:8002",
            cert=cert_path,
            key=key_path,
            rootca=rootca,
            output_dir=out_dir,
        )
        handle_package(args)

        key_file = os.path.join(out_dir, "startup", "server.key")
        file_mode = stat.S_IMODE(os.stat(key_file).st_mode)
        assert file_mode == 0o600


class TestAdminKitAssembly:
    @pytest.mark.parametrize("kit_type", ["org_admin", "lead", "member"])
    def test_basic_explicit_mode(self, kit_type, cert_env, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        rootca = cert_env["rootca_path"]

        key_path, cert_path = _make_signed_cert(ca_key, ca_cert, "admin-alice", str(work), f"{kit_type}.crt")
        out_dir = str(tmp_path / "output")
        args = _make_args(
            kit_type=kit_type,
            name="admin-alice",
            endpoint="grpc://server.example.com:8002",
            cert=cert_path,
            key=key_path,
            rootca=rootca,
            output_dir=out_dir,
        )
        handle_package(args)

        startup = os.path.join(out_dir, "startup")
        assert os.path.isfile(os.path.join(startup, "client.crt"))
        assert os.path.isfile(os.path.join(startup, "client.key"))
        assert os.path.isfile(os.path.join(startup, "rootCA.pem"))
        assert os.path.isfile(os.path.join(startup, "fed_admin.json"))
        assert os.path.isfile(os.path.join(startup, "fl_admin.sh"))

        local = os.path.join(out_dir, "local")
        assert os.path.isfile(os.path.join(local, "resources.json.default"))

    def test_fed_admin_json_content(self, cert_env, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        rootca = cert_env["rootca_path"]

        key_path, cert_path = _make_signed_cert(ca_key, ca_cert, "admin-alice", str(work), "lead.crt")
        out_dir = str(tmp_path / "output")
        args = _make_args(
            kit_type="lead",
            name="admin-alice",
            endpoint="grpc://server.example.com:8002",
            cert=cert_path,
            key=key_path,
            rootca=rootca,
            output_dir=out_dir,
            server_name="fl-server",
            admin_port=8003,
        )
        handle_package(args)

        with open(os.path.join(out_dir, "startup", "fed_admin.json")) as f:
            cfg = json.load(f)
        assert cfg["admin"]["username"] == "admin-alice"
        assert cfg["admin"]["server_identity"] == "fl-server"
        assert cfg["admin"]["port"] == 8003

    def test_dir_mode(self, cert_env, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        rootca_src = cert_env["rootca_path"]

        key_path, cert_path = _make_signed_cert(ca_key, ca_cert, "admin-alice", str(work), "lead.crt")
        shutil.copy2(rootca_src, str(work / "rootCA.pem"))

        out_dir = str(tmp_path / "output")
        args = _make_args(
            kit_type="lead",
            endpoint="grpc://server.example.com:8002",
            dir=str(work),
            output_dir=out_dir,
        )
        handle_package(args)

        assert args.name == "admin-alice"
        assert os.path.isfile(os.path.join(out_dir, "startup", "fed_admin.json"))


# ---------------------------------------------------------------------------
# Error condition tests
# ---------------------------------------------------------------------------


class TestErrorConditions:
    def test_missing_cert_file(self, cert_env, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        rootca = cert_env["rootca_path"]

        key_path, _ = _make_signed_cert(ca_key, ca_cert, "hospital-1", str(work), "client.crt")
        out_dir = str(tmp_path / "output")
        args = _make_args(
            kit_type="client",
            name="hospital-1",
            endpoint="grpc://server.example.com:8002",
            cert=str(tmp_path / "nonexistent.crt"),
            key=key_path,
            rootca=rootca,
            output_dir=out_dir,
        )
        with pytest.raises(SystemExit) as exc_info:
            handle_package(args)
        assert exc_info.value.code == 1

    def test_missing_key_file(self, cert_env, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        rootca = cert_env["rootca_path"]

        _, cert_path = _make_signed_cert(ca_key, ca_cert, "hospital-1", str(work), "client.crt")
        out_dir = str(tmp_path / "output")
        args = _make_args(
            kit_type="client",
            name="hospital-1",
            endpoint="grpc://server.example.com:8002",
            cert=cert_path,
            key=str(tmp_path / "nonexistent.key"),
            rootca=rootca,
            output_dir=out_dir,
        )
        with pytest.raises(SystemExit) as exc_info:
            handle_package(args)
        assert exc_info.value.code == 1

    def test_missing_rootca_file(self, cert_env, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        rootca = cert_env["rootca_path"]

        key_path, cert_path = _make_signed_cert(ca_key, ca_cert, "hospital-1", str(work), "client.crt")
        out_dir = str(tmp_path / "output")
        args = _make_args(
            kit_type="client",
            name="hospital-1",
            endpoint="grpc://server.example.com:8002",
            cert=cert_path,
            key=key_path,
            rootca=str(tmp_path / "nonexistent.pem"),
            output_dir=out_dir,
        )
        with pytest.raises(SystemExit) as exc_info:
            handle_package(args)
        assert exc_info.value.code == 1

    def test_invalid_endpoint(self, cert_env, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        rootca = cert_env["rootca_path"]

        key_path, cert_path = _make_signed_cert(ca_key, ca_cert, "hospital-1", str(work), "client.crt")
        out_dir = str(tmp_path / "output")
        args = _make_args(
            kit_type="client",
            name="hospital-1",
            endpoint="http://server.example.com:8002",  # invalid scheme
            cert=cert_path,
            key=key_path,
            rootca=rootca,
            output_dir=out_dir,
        )
        with pytest.raises(SystemExit) as exc_info:
            handle_package(args)
        assert exc_info.value.code == 4

    def test_dir_and_explicit_mutually_exclusive(self, cert_env, tmp_path):
        args = _make_args(
            kit_type="client",
            name="hospital-1",
            endpoint="grpc://server.example.com:8002",
            dir=str(tmp_path),
            cert=str(tmp_path / "some.crt"),
        )
        with pytest.raises(SystemExit) as exc_info:
            handle_package(args)
        assert exc_info.value.code == 4

    def test_neither_dir_nor_explicit(self, tmp_path):
        args = _make_args(
            kit_type="client",
            name="hospital-1",
            endpoint="grpc://server.example.com:8002",
        )
        with pytest.raises(SystemExit) as exc_info:
            handle_package(args)
        assert exc_info.value.code == 4

    def test_require_signed_jobs_on_non_server(self, cert_env, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        rootca = cert_env["rootca_path"]

        key_path, cert_path = _make_signed_cert(ca_key, ca_cert, "hospital-1", str(work), "client.crt")
        args = _make_args(
            kit_type="client",
            name="hospital-1",
            endpoint="grpc://server.example.com:8002",
            cert=cert_path,
            key=key_path,
            rootca=rootca,
            require_signed_jobs=True,
        )
        with pytest.raises(SystemExit) as exc_info:
            handle_package(args)
        assert exc_info.value.code == 4

    def test_output_dir_exists_no_force(self, cert_env, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        rootca = cert_env["rootca_path"]

        key_path, cert_path = _make_signed_cert(ca_key, ca_cert, "hospital-1", str(work), "client.crt")
        out_dir = tmp_path / "output"
        out_dir.mkdir()  # already exists

        args = _make_args(
            kit_type="client",
            name="hospital-1",
            endpoint="grpc://server.example.com:8002",
            cert=cert_path,
            key=key_path,
            rootca=rootca,
            output_dir=str(out_dir),
            force=False,
        )
        with pytest.raises(SystemExit) as exc_info:
            handle_package(args)
        assert exc_info.value.code == 1

    def test_cert_not_signed_by_rootca(self, tmp_path):
        """Cert signed by a different CA than the provided rootCA should fail."""
        # CA 1 (real signer)
        ca1_dir = str(tmp_path / "ca1")
        os.makedirs(ca1_dir, exist_ok=True)
        ca_key_1, ca_cert_1, rootca_1 = _make_ca(ca1_dir)

        # CA 2 (different CA whose rootCA.pem we'll provide)
        ca2_dir = str(tmp_path / "ca2")
        os.makedirs(ca2_dir, exist_ok=True)
        ca_key_2, ca_cert_2, rootca_2 = _make_ca(ca2_dir)

        # Sign cert with CA1 but provide rootCA of CA2
        work = tmp_path / "work"
        work.mkdir()
        key_path, cert_path = _make_signed_cert(ca_key_1, ca_cert_1, "hospital-1", str(work), "client.crt")
        out_dir = str(tmp_path / "output")
        args = _make_args(
            kit_type="client",
            name="hospital-1",
            endpoint="grpc://server.example.com:8002",
            cert=cert_path,
            key=key_path,
            rootca=rootca_2,  # wrong CA
            output_dir=out_dir,
        )
        with pytest.raises(SystemExit) as exc_info:
            handle_package(args)
        assert exc_info.value.code == 1

    def test_missing_endpoint_for_client(self, cert_env, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        rootca = cert_env["rootca_path"]

        key_path, cert_path = _make_signed_cert(ca_key, ca_cert, "hospital-1", str(work), "client.crt")
        args = _make_args(
            kit_type="client",
            name="hospital-1",
            endpoint=None,
            cert=cert_path,
            key=key_path,
            rootca=rootca,
        )
        with pytest.raises(SystemExit) as exc_info:
            handle_package(args)
        assert exc_info.value.code == 4

    def test_missing_endpoint_for_server(self, cert_env, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        rootca = cert_env["rootca_path"]

        key_path, cert_path = _make_signed_cert(ca_key, ca_cert, "fl-server", str(work), "server.crt")
        args = _make_args(
            kit_type="server",
            name="fl-server",
            endpoint=None,
            cert=cert_path,
            key=key_path,
            rootca=rootca,
        )
        with pytest.raises(SystemExit) as exc_info:
            handle_package(args)
        assert exc_info.value.code == 4
