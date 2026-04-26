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

import datetime
import hashlib
import json
import os
import platform
import shutil
import stat
import types
import unittest.mock
import zipfile

import pytest
from cryptography.hazmat.primitives import serialization

from nvflare.lighter.constants import CtxKey, PropKey
from nvflare.lighter.entity import Project
from nvflare.lighter.impl.cert import CertBuilder
from nvflare.lighter.impl.signature import SignatureBuilder
from nvflare.lighter.impl.static_file import StaticFileBuilder
from nvflare.lighter.impl.workspace import WorkspaceBuilder
from nvflare.lighter.provisioner import Provisioner
from nvflare.lighter.utils import Identity, generate_cert, generate_keys, load_crt, serialize_cert, serialize_pri_key
from nvflare.tool import cli_output
from nvflare.tool.package.package_commands import (
    _DUMMY_SERVER_NAME,
    PrebuiltCertBuilder,
    _discover_name_from_dir,
    _parse_endpoint,
    _read_zip_json,
    _resolve_request_dir,
    _safe_zip_names,
    _validate_cert_material,
    _validate_signed_public_key_hash,
    _write_file_nofollow,
    handle_package,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ca(tmp_dir, name="test-ca", org="TestOrg"):
    """Generate a self-signed CA and write rootCA.pem + ca.key into tmp_dir."""
    ca_key, ca_pub = generate_keys()
    ca_id = Identity(name, org)
    ca_cert = generate_cert(ca_id, ca_id, ca_key, ca_pub, ca=True)
    rootca_path = os.path.join(tmp_dir, "rootCA.pem")
    with open(rootca_path, "wb") as f:
        f.write(serialize_cert(ca_cert))
    return ca_key, ca_cert, rootca_path


def _make_signed_cert(ca_key, ca_cert, name, tmp_dir, cert_filename, role=None, org="TestOrg"):
    """Generate a key + CA-signed cert, write them into tmp_dir.

    Pass ``role`` to embed it as UNSTRUCTURED_NAME in the cert subject (simulates
    certs produced by ``nvflare cert sign -t <role>``).
    """
    key, pub = generate_keys()
    subj = Identity(name, org, role=role)
    issuer = Identity(
        ca_cert.subject.get_attributes_for_oid(
            __import__("cryptography.x509.oid", fromlist=["NameOID"]).NameOID.COMMON_NAME
        )[0].value,
        "TestOrg",
    )
    cert = generate_cert(subj, issuer, ca_key, pub)
    key_path = os.path.join(tmp_dir, f"{name}.key")
    cert_path = os.path.join(tmp_dir, cert_filename)
    _write_file_nofollow(key_path, serialize_pri_key(key), mode=0o600)
    _write_file_nofollow(cert_path, serialize_cert(cert), mode=0o644)
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
        workspace=None,
        project_name="testproject",
        admin_port=None,
        force=True,
        output_fmt=None,
        schema=False,
        project_file=None,
    )
    defaults.update(kwargs)
    return types.SimpleNamespace(**defaults)


def _kit_dir(workspace, project_name, name):
    """Return the expected kit output dir: workspace/project_name/prod_00/name."""
    return os.path.join(workspace, project_name, "prod_00", name)


class _FakeBuilderCtx:
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def get_kit_dir(self, participant):
        return os.path.join(self.root_dir, participant.name)


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

    def test_http_valid(self):
        scheme, host, port = _parse_endpoint("http://server:8002")
        assert scheme == "http"

    def test_invalid_scheme_raises(self):
        with pytest.raises(ValueError):
            _parse_endpoint("https://server:8002")

    def test_missing_port_raises(self):
        with pytest.raises(ValueError):
            _parse_endpoint("grpc://server")

    def test_empty_host_raises(self):
        with pytest.raises(ValueError):
            _parse_endpoint("grpc://:8002")

    def test_port_zero_raises(self):
        with pytest.raises(ValueError):
            _parse_endpoint("grpc://server:0")


class TestPrebuiltCertBuilder:
    def test_cert_write_rejects_preexisting_symlink(self, tmp_path):
        ca_key, ca_cert, rootca_path = _make_ca(str(tmp_path))
        key_path, cert_path = _make_signed_cert(
            ca_key,
            ca_cert,
            "hospital-1",
            str(tmp_path),
            "hospital-1.crt",
            role="client",
        )
        project = Project("pkgtest", "")
        project.add_client("hospital-1", "myorg", {})
        ctx = _FakeBuilderCtx(str(tmp_path / "kits"))
        kit_dir = tmp_path / "kits" / "hospital-1"
        kit_dir.mkdir(parents=True)
        cert_redirect = tmp_path / "cert-redirect"
        cert_redirect.write_text("cert target")
        try:
            (kit_dir / "client.crt").symlink_to(cert_redirect)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks are not supported on this platform")
        builder = PrebuiltCertBuilder(
            cert_path=cert_path,
            key_path=key_path,
            rootca_path=rootca_path,
            target_name="hospital-1",
        )

        with pytest.raises(OSError):
            builder.build(project, ctx)
        assert cert_redirect.read_text() == "cert target"

    def test_rootca_write_rejects_preexisting_symlink(self, tmp_path):
        ca_key, ca_cert, rootca_path = _make_ca(str(tmp_path))
        key_path, cert_path = _make_signed_cert(
            ca_key,
            ca_cert,
            "hospital-1",
            str(tmp_path),
            "hospital-1.crt",
            role="client",
        )
        project = Project("pkgtest", "")
        project.add_client("hospital-1", "myorg", {})
        ctx = _FakeBuilderCtx(str(tmp_path / "kits"))
        kit_dir = tmp_path / "kits" / "hospital-1"
        kit_dir.mkdir(parents=True)
        rootca_redirect = tmp_path / "rootca-redirect"
        rootca_redirect.write_text("rootca target")
        try:
            (kit_dir / "rootCA.pem").symlink_to(rootca_redirect)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks are not supported on this platform")
        builder = PrebuiltCertBuilder(
            cert_path=cert_path,
            key_path=key_path,
            rootca_path=rootca_path,
            target_name="hospital-1",
        )

        with pytest.raises(OSError):
            builder.build(project, ctx)
        assert rootca_redirect.read_text() == "rootca target"

    @pytest.mark.skipif(
        not hasattr(os, "symlink") or not hasattr(os, "O_NOFOLLOW"), reason="nofollow symlink support required"
    )
    def test_source_key_symlink_is_rejected(self, tmp_path):
        ca_key, ca_cert, rootca_path = _make_ca(str(tmp_path))
        key_path, cert_path = _make_signed_cert(
            ca_key,
            ca_cert,
            "hospital-1",
            str(tmp_path),
            "hospital-1.crt",
            role="client",
        )
        key_link = tmp_path / "hospital-1-link.key"
        os.symlink(key_path, str(key_link))
        project = Project("pkgtest", "")
        project.add_client("hospital-1", "myorg", {})
        ctx = _FakeBuilderCtx(str(tmp_path / "kits"))
        builder = PrebuiltCertBuilder(
            cert_path=cert_path,
            key_path=str(key_link),
            rootca_path=rootca_path,
            target_name="hospital-1",
        )

        with pytest.raises(OSError):
            builder.build(project, ctx)

    def test_mismatched_target_name_raises_instead_of_silent_empty_kit(self, tmp_path):
        ca_key, ca_cert, rootca_path = _make_ca(str(tmp_path))
        key_path, cert_path = _make_signed_cert(
            ca_key,
            ca_cert,
            "hospital-1",
            str(tmp_path),
            "hospital-1.crt",
            role="client",
        )
        project = Project("pkgtest", "")
        project.add_client("hospital-1", "myorg", {})
        ctx = _FakeBuilderCtx(str(tmp_path / "kits"))
        builder = PrebuiltCertBuilder(
            cert_path=cert_path,
            key_path=key_path,
            rootca_path=rootca_path,
            target_name="typoed-name",
        )

        with pytest.raises(ValueError, match="no participant kit was built"):
            builder.build(project, ctx)

    def test_port_above_max_raises(self):
        with pytest.raises(ValueError):
            _parse_endpoint("grpc://server:65536")

    def test_port_max_valid(self):
        _, _, port = _parse_endpoint("grpc://server:65535")
        assert port == 65535

    def test_port_min_valid(self):
        _, _, port = _parse_endpoint("grpc://server:1")
        assert port == 1


class TestSignedZipHelpers:
    class _FailingFdOpen:
        def __init__(self, fd):
            self.fd = fd

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            os.close(self.fd)

        def write(self, _content):
            raise OSError("disk full")

    @pytest.mark.skipif(platform.system() == "Windows", reason="unlinking open files differs on Windows")
    def test_write_file_nofollow_removes_created_file_when_write_fails(self, tmp_path):
        path = tmp_path / "rootCA.pem"

        def failing_fdopen(fd, _mode):
            return self._FailingFdOpen(fd)

        with unittest.mock.patch("nvflare.tool.package.package_commands.os.fdopen", side_effect=failing_fdopen):
            with pytest.raises(OSError):
                _write_file_nofollow(str(path), b"public cert")

        assert not path.exists()

    def test_safe_zip_names_does_not_append_directory_when_error_is_mocked(self, tmp_path):
        zip_path = tmp_path / "signed.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("nested/", b"")
            zf.writestr("signed.json", "{}")

        with zipfile.ZipFile(zip_path, "r") as zf:
            with unittest.mock.patch("nvflare.tool.package.package_commands.output_error_message") as error:
                names = _safe_zip_names(zf, str(zip_path))

        error.assert_called_once()
        assert "nested/" not in names
        assert names == ["signed.json"]

    def test_read_zip_json_raises_after_error_when_error_is_mocked(self, tmp_path):
        zip_path = tmp_path / "signed.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("signed.json", "not json")

        with zipfile.ZipFile(zip_path, "r") as zf:
            with unittest.mock.patch("nvflare.tool.package.package_commands.output_error_message") as error:
                with pytest.raises(json.JSONDecodeError):
                    _read_zip_json(zf, "signed.json", str(zip_path))

        error.assert_called_once()

    def test_resolve_request_dir_rejects_fallback_candidate_with_mismatched_request_id(self, tmp_path):
        signed_dir = tmp_path / "signed"
        signed_dir.mkdir()
        candidate = signed_dir / "site-3"
        candidate.mkdir()
        _write_file_nofollow(candidate / "site-3.key", b"key", mode=0o600)
        (candidate / "request.json").write_text(json.dumps({"request_id": "2" * 32}))
        args = types.SimpleNamespace(request_dir=None)
        identity = {"name": "site-3", "request_id": "1" * 32}

        assert _resolve_request_dir(args, str(signed_dir / "site-3.signed.zip"), identity) is None


def test_missing_endpoint_human_mode_no_help_dump(capsys, monkeypatch, tmp_path):
    import argparse

    from nvflare.tool.package.package_cli import def_package_cli_parser

    monkeypatch.setattr(cli_output, "_output_format", "txt")
    root = argparse.ArgumentParser(prog="nvflare")
    subs = root.add_subparsers()
    def_package_cli_parser(subs)
    args = _make_args(
        endpoint=None,
        dir=str(tmp_path),
        workspace=str(tmp_path / "ws"),
    )

    with pytest.raises(SystemExit) as exc_info:
        handle_package(args)
    assert exc_info.value.code == 4
    captured = capsys.readouterr()
    assert "--endpoint is required" in captured.err
    assert "Code: INVALID_ARGS (exit 4)" in captured.err
    assert "usage:" in captured.err


def test_package_help_includes_working_examples():
    import argparse

    from nvflare.tool.package.package_cli import def_package_cli_parser

    root = argparse.ArgumentParser(prog="nvflare")
    subs = root.add_subparsers()
    parser = def_package_cli_parser(subs)["package"]

    help_text = parser.format_help()

    assert "Examples:" in help_text
    assert "nvflare package hospital-1.signed.zip -e grpc://fl-server:8002" in help_text
    assert "--project-file ./project.yaml --request-dir ./hospital-1" in help_text
    assert "--cert ./signed/hospital-1/hospital-1.crt" not in help_text
    assert "--key ./csr/hospital-1.key" not in help_text
    assert "--rootca ./signed/hospital-1/rootCA.pem" not in help_text


def test_package_schema_uses_shared_examples(capsys):
    import argparse

    from nvflare.tool.cli_schema import handle_schema_flag
    from nvflare.tool.package.package_cli import _PACKAGE_EXAMPLES, def_package_cli_parser

    root = argparse.ArgumentParser(prog="nvflare")
    subs = root.add_subparsers()
    parser = def_package_cli_parser(subs)["package"]

    with pytest.raises(SystemExit) as exc_info:
        handle_schema_flag(parser, "nvflare package", _PACKAGE_EXAMPLES, ["--schema"])
    assert exc_info.value.code == 0

    schema = json.loads(capsys.readouterr().out)
    assert schema["command"] == "nvflare package"
    assert schema["examples"] == _PACKAGE_EXAMPLES


# ---------------------------------------------------------------------------
# Unit tests: _discover_name_from_dir
# ---------------------------------------------------------------------------


class TestDiscoverNameFromDir:
    def test_single_key_detected(self, tmp_path):
        (tmp_path / "alice.key").write_text("dummy")
        name = _discover_name_from_dir(str(tmp_path))
        assert name == "alice"

    def test_hidden_key_ignored_when_visible_key_exists(self, tmp_path):
        (tmp_path / ".hidden.key").write_text("dummy")
        (tmp_path / "alice.key").write_text("dummy")
        name = _discover_name_from_dir(str(tmp_path))
        assert name == "alice"

    def test_only_hidden_key_exits(self, tmp_path):
        (tmp_path / ".hidden.key").write_text("dummy")
        with pytest.raises(SystemExit):
            _discover_name_from_dir(str(tmp_path))

    def test_no_key_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            _discover_name_from_dir(str(tmp_path))

    def test_multiple_keys_exits(self, tmp_path):
        (tmp_path / "alice.key").write_text("dummy")
        (tmp_path / "bob.key").write_text("dummy")
        with pytest.raises(SystemExit):
            _discover_name_from_dir(str(tmp_path))


def test_package_compat_output_alias_sets_output_format(tmp_path):
    import argparse

    from nvflare.tool.package.package_cli import def_package_cli_parser, handle_package_cmd

    root = argparse.ArgumentParser(prog="nvflare")
    subs = root.add_subparsers(dest="sub_command")
    def_package_cli_parser(subs)
    args = root.parse_args(["package", "-e", "grpc://fl-server:8002", "--dir", str(tmp_path), "--output", "json"])

    with unittest.mock.patch("nvflare.tool.cli_output.set_output_format") as set_output_format:
        with unittest.mock.patch("nvflare.tool.package.package_commands.handle_package") as handle_package:
            handle_package_cmd(args)

    set_output_format.assert_called_once_with("json")
    handle_package.assert_called_once_with(args)


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

        key_path, cert_path = _make_signed_cert(
            ca_key, ca_cert, "hospital-1", str(work), "hospital-1.crt", role="client"
        )
        ws = str(tmp_path / "ws")
        args = _make_args(
            kit_type="client",
            name="hospital-1",
            endpoint="grpc://server.example.com:8002",
            cert=cert_path,
            key=key_path,
            rootca=rootca,
            workspace=ws,
        )
        handle_package(args)

        out_dir = _kit_dir(ws, "testproject", "hospital-1")
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

        assert os.path.isdir(os.path.join(out_dir, "transfer"))

    def test_output_path_structure(self, cert_env, tmp_path):
        """Output is workspace/<project_name>/prod_00/<name>/."""
        work = tmp_path / "work"
        work.mkdir()
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        rootca = cert_env["rootca_path"]

        key_path, cert_path = _make_signed_cert(
            ca_key, ca_cert, "hospital-1", str(work), "hospital-1.crt", role="client"
        )
        ws = str(tmp_path / "ws")
        args = _make_args(
            kit_type="client",
            name="hospital-1",
            endpoint="grpc://server.example.com:8002",
            cert=cert_path,
            key=key_path,
            rootca=rootca,
            workspace=ws,
            project_name="myfl",
        )
        handle_package(args)

        expected = os.path.join(ws, "myfl", "prod_00", "hospital-1")
        assert os.path.isdir(expected)
        assert os.path.isfile(os.path.join(expected, "startup", "fed_client.json"))

    def test_second_participant_goes_to_next_prod_dir(self, cert_env, tmp_path):
        """Each provisioner run creates a new prod_NN; second participant goes to prod_01."""
        work = tmp_path / "work"
        work.mkdir()
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        rootca = cert_env["rootca_path"]
        ws = str(tmp_path / "ws")

        key1, cert1 = _make_signed_cert(ca_key, ca_cert, "site-1", str(work), "site-1.crt", role="client")
        key2, cert2 = _make_signed_cert(ca_key, ca_cert, "site-2", str(work), "site-2.crt", role="client")

        args1 = _make_args(
            kit_type="client",
            name="site-1",
            endpoint="grpc://server:8002",
            cert=cert1,
            key=key1,
            rootca=rootca,
            workspace=ws,
            project_name="myproject",
        )
        args2 = _make_args(
            kit_type="client",
            name="site-2",
            endpoint="grpc://server:8002",
            cert=cert2,
            key=key2,
            rootca=rootca,
            workspace=ws,
            project_name="myproject",
        )
        handle_package(args1)
        handle_package(args2)

        # First participant lands in prod_00, second in prod_01 (WorkspaceBuilder always increments)
        assert os.path.isdir(os.path.join(ws, "myproject", "prod_00", "site-1"))
        assert os.path.isdir(os.path.join(ws, "myproject", "prod_01", "site-2"))

    def test_default_workspace_name(self, cert_env, tmp_path, monkeypatch):
        """If workspace is None, defaults to 'workspace' in cwd."""
        work = tmp_path / "work"
        work.mkdir()
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        rootca = cert_env["rootca_path"]

        key_path, cert_path = _make_signed_cert(
            ca_key, ca_cert, "hospital-1", str(work), "hospital-1.crt", role="client"
        )
        monkeypatch.chdir(str(tmp_path))
        args = _make_args(
            kit_type="client",
            name="hospital-1",
            endpoint="grpc://server.example.com:8002",
            cert=cert_path,
            key=key_path,
            rootca=rootca,
            workspace=None,  # not set → defaults to "workspace"
        )
        handle_package(args)
        assert os.path.isdir(os.path.join("workspace", "testproject", "prod_00", "hospital-1"))

    def test_key_permissions(self, cert_env, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        rootca = cert_env["rootca_path"]

        key_path, cert_path = _make_signed_cert(
            ca_key, ca_cert, "hospital-1", str(work), "hospital-1.crt", role="client"
        )
        ws = str(tmp_path / "ws")
        args = _make_args(
            kit_type="client",
            name="hospital-1",
            endpoint="grpc://server.example.com:8002",
            cert=cert_path,
            key=key_path,
            rootca=rootca,
            workspace=ws,
        )
        handle_package(args)

        out_dir = _kit_dir(ws, "testproject", "hospital-1")
        key_file = os.path.join(out_dir, "startup", "client.key")
        file_mode = stat.S_IMODE(os.stat(key_file).st_mode)
        assert file_mode == 0o600

    def test_fed_client_json_content(self, cert_env, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        rootca = cert_env["rootca_path"]

        key_path, cert_path = _make_signed_cert(
            ca_key, ca_cert, "hospital-1", str(work), "hospital-1.crt", role="client"
        )
        ws = str(tmp_path / "ws")
        args = _make_args(
            kit_type="client",
            name="hospital-1",
            endpoint="grpc://server.example.com:8002",
            cert=cert_path,
            key=key_path,
            rootca=rootca,
            workspace=ws,
        )
        handle_package(args)

        out_dir = _kit_dir(ws, "testproject", "hospital-1")
        with open(os.path.join(out_dir, "startup", "fed_client.json")) as f:
            cfg = json.load(f)
        assert cfg["servers"][0]["name"] == "testproject"  # project_name used
        assert cfg["servers"][0]["service"]["scheme"] == "grpc"
        assert cfg["client"]["fqsn"] == "hospital-1"
        assert "8002:8002" in cfg["overseer_agent"]["args"]["sp_end_point"]

    def test_dir_mode_auto_discovery(self, cert_env, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        rootca_src = cert_env["rootca_path"]

        key_path, cert_path = _make_signed_cert(
            ca_key, ca_cert, "hospital-1", str(work), "hospital-1.crt", role="client"
        )
        shutil.copy2(rootca_src, str(work / "rootCA.pem"))

        ws = str(tmp_path / "ws")
        args = _make_args(
            kit_type="client",
            endpoint="grpc://server.example.com:8002",
            dir=str(work),
            workspace=ws,
        )
        handle_package(args)

        assert args.name == "hospital-1"
        out_dir = _kit_dir(ws, "testproject", "hospital-1")
        assert os.path.isfile(os.path.join(out_dir, "startup", "fed_client.json"))

    def test_dir_mode_name_explicit(self, cert_env, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        rootca_src = cert_env["rootca_path"]

        key_path, cert_path = _make_signed_cert(
            ca_key, ca_cert, "hospital-1", str(work), "hospital-1.crt", role="client"
        )
        shutil.copy2(rootca_src, str(work / "rootCA.pem"))

        ws = str(tmp_path / "ws")
        args = _make_args(
            kit_type="client",
            name="hospital-1",
            endpoint="grpc://server.example.com:8002",
            dir=str(work),
            workspace=ws,
        )
        handle_package(args)

        out_dir = _kit_dir(ws, "testproject", "hospital-1")
        assert os.path.isfile(os.path.join(out_dir, "startup", "fed_client.json"))


class TestServerKitAssembly:
    def test_basic_explicit_mode(self, cert_env, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        rootca = cert_env["rootca_path"]

        key_path, cert_path = _make_signed_cert(ca_key, ca_cert, "fl-server", str(work), "fl-server.crt", role="server")
        ws = str(tmp_path / "ws")
        args = _make_args(
            kit_type="server",
            name="fl-server",
            endpoint="grpc://0.0.0.0:8002",
            cert=cert_path,
            key=key_path,
            rootca=rootca,
            workspace=ws,
        )
        handle_package(args)

        out_dir = _kit_dir(ws, "testproject", "fl-server")
        startup = os.path.join(out_dir, "startup")
        assert os.path.isfile(os.path.join(startup, "server.crt"))
        assert os.path.isfile(os.path.join(startup, "server.key"))
        assert os.path.isfile(os.path.join(startup, "rootCA.pem"))
        assert os.path.isfile(os.path.join(startup, "fed_server.json"))
        assert os.path.isfile(os.path.join(startup, "start.sh"))
        assert os.path.isfile(os.path.join(startup, "sub_start.sh"))
        assert os.path.isfile(os.path.join(startup, "stop_fl.sh"))
        # authorization.json goes in local/ as .default (matches provisioner, not active in startup/)
        assert not os.path.isfile(os.path.join(startup, "authorization.json"))

        local = os.path.join(out_dir, "local")
        assert os.path.isfile(os.path.join(local, "resources.json.default"))
        assert os.path.isfile(os.path.join(local, "log_config.json.default"))
        assert os.path.isfile(os.path.join(local, "privacy.json.sample"))
        assert os.path.isfile(os.path.join(local, "authorization.json.default"))

        assert os.path.isdir(os.path.join(out_dir, "transfer"))

    def test_fed_server_json_content(self, cert_env, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        rootca = cert_env["rootca_path"]

        key_path, cert_path = _make_signed_cert(ca_key, ca_cert, "fl-server", str(work), "fl-server.crt", role="server")
        ws = str(tmp_path / "ws")
        args = _make_args(
            kit_type="server",
            name="fl-server",
            endpoint="grpc://fl-server:8002",
            cert=cert_path,
            key=key_path,
            rootca=rootca,
            workspace=ws,
        )
        handle_package(args)

        out_dir = _kit_dir(ws, "testproject", "fl-server")
        with open(os.path.join(out_dir, "startup", "fed_server.json")) as f:
            cfg = json.load(f)
        assert cfg["servers"][0]["service"]["scheme"] == "grpc"
        # target is server.name:port (server.name == args.name for server kit)
        assert cfg["servers"][0]["service"]["target"] == "fl-server:8002"
        assert cfg["servers"][0]["admin_port"] == 8002

    def test_key_permissions(self, cert_env, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        rootca = cert_env["rootca_path"]

        key_path, cert_path = _make_signed_cert(ca_key, ca_cert, "fl-server", str(work), "fl-server.crt", role="server")
        ws = str(tmp_path / "ws")
        args = _make_args(
            kit_type="server",
            name="fl-server",
            endpoint="grpc://0.0.0.0:8002",
            cert=cert_path,
            key=key_path,
            rootca=rootca,
            workspace=ws,
        )
        handle_package(args)

        out_dir = _kit_dir(ws, "testproject", "fl-server")
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

        key_path, cert_path = _make_signed_cert(
            ca_key, ca_cert, "alice@myorg.com", str(work), "alice@myorg.com.crt", role=kit_type
        )
        ws = str(tmp_path / "ws")
        args = _make_args(
            kit_type=kit_type,
            name="alice@myorg.com",
            endpoint="grpc://server.example.com:8002",
            cert=cert_path,
            key=key_path,
            rootca=rootca,
            workspace=ws,
        )
        handle_package(args)

        out_dir = _kit_dir(ws, "testproject", "alice@myorg.com")
        startup = os.path.join(out_dir, "startup")
        assert os.path.isfile(os.path.join(startup, "client.crt"))
        assert os.path.isfile(os.path.join(startup, "client.key"))
        assert os.path.isfile(os.path.join(startup, "rootCA.pem"))
        assert os.path.isfile(os.path.join(startup, "fed_admin.json"))
        assert os.path.isfile(os.path.join(startup, "fl_admin.sh"))

        local = os.path.join(out_dir, "local")
        assert os.path.isfile(os.path.join(local, "resources.json.default"))

        assert os.path.isdir(os.path.join(out_dir, "transfer"))

    def test_fed_admin_json_content(self, cert_env, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        rootca = cert_env["rootca_path"]

        key_path, cert_path = _make_signed_cert(
            ca_key, ca_cert, "alice@myorg.com", str(work), "alice@myorg.com.crt", role="lead"
        )
        ws = str(tmp_path / "ws")
        args = _make_args(
            kit_type="lead",
            name="alice@myorg.com",
            endpoint="grpc://server.example.com:8002",
            cert=cert_path,
            key=key_path,
            rootca=rootca,
            workspace=ws,
            admin_port=8003,
        )
        handle_package(args)

        out_dir = _kit_dir(ws, "testproject", "alice@myorg.com")
        with open(os.path.join(out_dir, "startup", "fed_admin.json")) as f:
            cfg = json.load(f)
        assert cfg["admin"]["username"] == "alice@myorg.com"
        assert cfg["admin"]["server_identity"] == "server.example.com"  # derived from endpoint hostname
        assert cfg["admin"]["port"] == 8003

    def test_dir_mode(self, cert_env, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        rootca_src = cert_env["rootca_path"]

        # Key filename must match the email-format name for auto-discovery to produce a valid admin name
        key_path, cert_path = _make_signed_cert(
            ca_key, ca_cert, "alice@myorg.com", str(work), "alice@myorg.com.crt", role="lead"
        )
        shutil.copy2(rootca_src, str(work / "rootCA.pem"))

        ws = str(tmp_path / "ws")
        args = _make_args(
            kit_type="lead",
            endpoint="grpc://server.example.com:8002",
            dir=str(work),
            workspace=ws,
        )
        handle_package(args)

        assert args.name == "alice@myorg.com"
        out_dir = _kit_dir(ws, "testproject", "alice@myorg.com")
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

        key_path, _ = _make_signed_cert(ca_key, ca_cert, "hospital-1", str(work), "hospital-1.crt", role="client")
        args = _make_args(
            kit_type="client",
            name="hospital-1",
            endpoint="grpc://server.example.com:8002",
            cert=str(tmp_path / "nonexistent.crt"),
            key=key_path,
            rootca=rootca,
            workspace=str(tmp_path / "ws"),
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

        _, cert_path = _make_signed_cert(ca_key, ca_cert, "hospital-1", str(work), "hospital-1.crt", role="client")
        args = _make_args(
            kit_type="client",
            name="hospital-1",
            endpoint="grpc://server.example.com:8002",
            cert=cert_path,
            key=str(tmp_path / "nonexistent.key"),
            rootca=rootca,
            workspace=str(tmp_path / "ws"),
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

        key_path, cert_path = _make_signed_cert(
            ca_key, ca_cert, "hospital-1", str(work), "hospital-1.crt", role="client"
        )
        args = _make_args(
            kit_type="client",
            name="hospital-1",
            endpoint="grpc://server.example.com:8002",
            cert=cert_path,
            key=key_path,
            rootca=str(tmp_path / "nonexistent.pem"),
            workspace=str(tmp_path / "ws"),
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

        key_path, cert_path = _make_signed_cert(
            ca_key, ca_cert, "hospital-1", str(work), "hospital-1.crt", role="client"
        )
        args = _make_args(
            kit_type="client",
            name="hospital-1",
            endpoint="https://server.example.com:8002",  # invalid scheme
            cert=cert_path,
            key=key_path,
            rootca=rootca,
            workspace=str(tmp_path / "ws"),
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

    def test_output_dir_exists_no_force(self, cert_env, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        rootca = cert_env["rootca_path"]

        key_path, cert_path = _make_signed_cert(
            ca_key, ca_cert, "hospital-1", str(work), "hospital-1.crt", role="client"
        )
        ws = str(tmp_path / "ws")
        # Pre-create the participant directory inside prod_00 to trigger the exists check
        os.makedirs(os.path.join(ws, "testproject", "prod_00", "hospital-1"), exist_ok=True)

        args = _make_args(
            kit_type="client",
            name="hospital-1",
            endpoint="grpc://server.example.com:8002",
            cert=cert_path,
            key=key_path,
            rootca=rootca,
            workspace=ws,
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
        key_path, cert_path = _make_signed_cert(
            ca_key_1, ca_cert_1, "hospital-1", str(work), "hospital-1.crt", role="client"
        )
        args = _make_args(
            kit_type="client",
            name="hospital-1",
            endpoint="grpc://server.example.com:8002",
            cert=cert_path,
            key=key_path,
            rootca=rootca_2,  # wrong CA
            workspace=str(tmp_path / "ws"),
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

        key_path, cert_path = _make_signed_cert(
            ca_key, ca_cert, "hospital-1", str(work), "hospital-1.crt", role="client"
        )
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

        key_path, cert_path = _make_signed_cert(ca_key, ca_cert, "fl-server", str(work), "fl-server.crt", role="server")
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

    def test_missing_name_explicit_mode(self, cert_env, tmp_path):
        """Explicit mode without -n must fail, not produce a broken kit."""
        work = tmp_path / "work"
        work.mkdir()
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        rootca = cert_env["rootca_path"]

        key_path, cert_path = _make_signed_cert(
            ca_key, ca_cert, "hospital-1", str(work), "hospital-1.crt", role="client"
        )
        args = _make_args(
            kit_type="client",
            name=None,  # omitted
            endpoint="grpc://server.example.com:8002",
            cert=cert_path,
            key=key_path,
            rootca=rootca,
            workspace=str(tmp_path / "ws"),
        )
        with pytest.raises(SystemExit) as exc_info:
            handle_package(args)
        assert exc_info.value.code == 4

    def test_missing_name_explicit_mode_no_workspace(self, cert_env, tmp_path):
        """Explicit mode without -n and without workspace must fail cleanly."""
        work = tmp_path / "work"
        work.mkdir()
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        rootca = cert_env["rootca_path"]

        key_path, cert_path = _make_signed_cert(
            ca_key, ca_cert, "hospital-1", str(work), "hospital-1.crt", role="client"
        )
        args = _make_args(
            kit_type="client",
            name=None,  # omitted
            endpoint="grpc://server.example.com:8002",
            cert=cert_path,
            key=key_path,
            rootca=rootca,
            workspace=None,
        )
        with pytest.raises(SystemExit) as exc_info:
            handle_package(args)
        assert exc_info.value.code == 4

    @pytest.mark.parametrize("kit_type", ["client", "server"])
    def test_server_identity_derived_from_endpoint(self, kit_type, cert_env, tmp_path):
        """Server identity is always the endpoint hostname — no --server-name argument needed."""
        work = tmp_path / "work"
        work.mkdir()
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        rootca = cert_env["rootca_path"]

        # Use distinct participant name to avoid name collision with the server placeholder
        participant_name = "fl-server" if kit_type == "server" else "hospital-1"
        key_path, cert_path = _make_signed_cert(
            ca_key, ca_cert, participant_name, str(work), f"{participant_name}.crt", role=kit_type
        )
        args = _make_args(
            kit_type=kit_type,
            name=participant_name,
            endpoint="grpc://fl-server:8002",
            cert=cert_path,
            key=key_path,
            rootca=rootca,
            workspace=str(tmp_path / "ws"),
        )
        handle_package(args)  # must not raise for any kit type

    def test_server_identity_derived_from_endpoint_admin(self, cert_env, tmp_path):
        """Admin server identity is always the endpoint hostname — no --server-name argument needed."""
        work = tmp_path / "work"
        work.mkdir()
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        rootca = cert_env["rootca_path"]

        key_path, cert_path = _make_signed_cert(
            ca_key, ca_cert, "alice@myorg.com", str(work), "alice@myorg.com.crt", role="lead"
        )
        args = _make_args(
            kit_type="lead",
            name="alice@myorg.com",
            endpoint="grpc://fl-server:8002",
            cert=cert_path,
            key=key_path,
            rootca=rootca,
            workspace=str(tmp_path / "ws"),
        )
        handle_package(args)  # must not raise

    def test_endpoint_validated_before_file_checks(self, cert_env, tmp_path):
        """Invalid endpoint should error with exit_code=4 before file-not-found (exit_code=1)."""
        work = tmp_path / "work"
        work.mkdir()
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        rootca = cert_env["rootca_path"]

        key_path, cert_path = _make_signed_cert(
            ca_key, ca_cert, "hospital-1", str(work), "hospital-1.crt", role="client"
        )
        # Use invalid endpoint AND valid files — endpoint error should fire first
        args = _make_args(
            kit_type="client",
            name="hospital-1",
            endpoint="https://server.example.com:8002",  # invalid
            cert=cert_path,
            key=key_path,
            rootca=rootca,
            workspace=str(tmp_path / "ws"),
        )
        with pytest.raises(SystemExit) as exc_info:
            handle_package(args)
        assert exc_info.value.code == 4  # endpoint error, not file-not-found (1)

    def test_dir_mode_cert_not_found_hint_mentions_dir(self, cert_env, tmp_path, capsys):
        """CERT_NOT_FOUND hint in --dir mode should mention the directory and explain .crt vs .csr."""
        work = tmp_path / "work"
        work.mkdir()
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        rootca_src = cert_env["rootca_path"]

        # Only put key + rootCA — no .crt (simulates user who has CSR but hasn't gotten cert back)
        key_path, _ = _make_signed_cert(ca_key, ca_cert, "hospital-1", str(work), "hospital-1.crt", role="client")
        # Remove the cert file to simulate missing cert scenario
        os.remove(str(work / "hospital-1.crt"))
        shutil.copy2(rootca_src, str(work / "rootCA.pem"))

        args = _make_args(
            kit_type="client",
            name="hospital-1",
            endpoint="grpc://server.example.com:8002",
            dir=str(work),
            workspace=str(tmp_path / "ws"),
        )
        with pytest.raises(SystemExit) as exc_info:
            handle_package(args)
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        # Hint should mention the dir and distinguish .crt from .csr
        assert str(work) in captured.out or str(work) in captured.err
        assert ".csr" in captured.out or ".csr" in captured.err


# ---------------------------------------------------------------------------
# New behavior tests
# ---------------------------------------------------------------------------


class TestUX2BindTargetVsIdentity:
    """UX2: standard StaticFileBuilder always uses server.name for target and identity."""

    def test_server_kit_0000_endpoint_name_used_for_target(self, cert_env, tmp_path):
        """Even when endpoint host is 0.0.0.0, fed_server.json target uses server.name (args.name),
        not the raw endpoint host.  Both target and sp_end_point use the same identity."""
        work = tmp_path / "work"
        work.mkdir()
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        rootca = cert_env["rootca_path"]

        key_path, cert_path = _make_signed_cert(ca_key, ca_cert, "fl-server", str(work), "fl-server.crt", role="server")
        ws = str(tmp_path / "ws")
        args = _make_args(
            kit_type="server",
            name="fl-server",
            endpoint="grpc://0.0.0.0:8002",
            cert=cert_path,
            key=key_path,
            rootca=rootca,
            workspace=ws,
        )
        handle_package(args)

        out_dir = _kit_dir(ws, "testproject", "fl-server")
        with open(os.path.join(out_dir, "startup", "fed_server.json")) as f:
            cfg = json.load(f)
        # target and sp_end_point must both use server.name (fl-server), not the raw endpoint host
        assert cfg["servers"][0]["service"]["target"] == "fl-server:8002"
        sp = cfg["overseer_agent"]["args"]["sp_end_point"]
        assert sp.startswith("fl-server:")

    def test_server_kit_hostname_endpoint_target_is_hostname(self, cert_env, tmp_path):
        """When endpoint host is a regular hostname, target equals that hostname."""
        work = tmp_path / "work"
        work.mkdir()
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        rootca = cert_env["rootca_path"]

        key_path, cert_path = _make_signed_cert(ca_key, ca_cert, "fl-server", str(work), "fl-server.crt", role="server")
        ws = str(tmp_path / "ws")
        args = _make_args(
            kit_type="server",
            name="fl-server",
            endpoint="grpc://fl-server:8002",
            cert=cert_path,
            key=key_path,
            rootca=rootca,
            workspace=ws,
        )
        handle_package(args)

        out_dir = _kit_dir(ws, "testproject", "fl-server")
        with open(os.path.join(out_dir, "startup", "fed_server.json")) as f:
            cfg = json.load(f)
        assert cfg["servers"][0]["service"]["target"] == "fl-server:8002"


class TestUX3AdminEmailValidation:
    """UX3: admin kit types must have email-format names."""

    @pytest.mark.parametrize("kit_type", ["org_admin", "lead", "member"])
    def test_non_email_name_rejected(self, kit_type, cert_env, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        rootca = cert_env["rootca_path"]

        key_path, cert_path = _make_signed_cert(
            ca_key, ca_cert, "admin-alice", str(work), "admin-alice.crt", role=kit_type
        )
        args = _make_args(
            kit_type=kit_type,
            name="admin-alice",  # not an email
            endpoint="grpc://server.example.com:8002",
            cert=cert_path,
            key=key_path,
            rootca=rootca,
            workspace=str(tmp_path / "ws"),
        )
        with pytest.raises(SystemExit) as exc_info:
            handle_package(args)
        assert exc_info.value.code == 4

    @pytest.mark.parametrize("kit_type", ["org_admin", "lead", "member"])
    def test_email_name_accepted(self, kit_type, cert_env, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        rootca = cert_env["rootca_path"]

        key_path, cert_path = _make_signed_cert(
            ca_key, ca_cert, "alice@myorg.com", str(work), "alice@myorg.com.crt", role=kit_type
        )
        ws = str(tmp_path / "ws")
        args = _make_args(
            kit_type=kit_type,
            name="alice@myorg.com",
            endpoint="grpc://server.example.com:8002",
            cert=cert_path,
            key=key_path,
            rootca=rootca,
            workspace=ws,
        )
        handle_package(args)  # must not raise
        out_dir = _kit_dir(ws, "testproject", "alice@myorg.com")
        assert os.path.isfile(os.path.join(out_dir, "startup", "fed_admin.json"))


class TestA4bNameCollisionGuard:
    """A4b: participant name must not collide with endpoint hostname or sentinel."""

    def test_client_name_equals_server_hostname_rejected(self, cert_env, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        rootca = cert_env["rootca_path"]

        key_path, cert_path = _make_signed_cert(ca_key, ca_cert, "fl-server", str(work), "fl-server.crt", role="client")
        args = _make_args(
            kit_type="client",
            name="fl-server",  # same as endpoint hostname — collision
            endpoint="grpc://fl-server:8002",
            cert=cert_path,
            key=key_path,
            rootca=rootca,
            workspace=str(tmp_path / "ws"),
        )
        with pytest.raises(SystemExit) as exc_info:
            handle_package(args)
        assert exc_info.value.code == 4

    def test_internal_dummy_server_name_rejected(self, cert_env, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        rootca = cert_env["rootca_path"]

        key_path, cert_path = _make_signed_cert(
            ca_key, ca_cert, _DUMMY_SERVER_NAME, str(work), f"{_DUMMY_SERVER_NAME}.crt", role="client"
        )
        args = _make_args(
            kit_type="client",
            name=_DUMMY_SERVER_NAME,
            endpoint="grpc://fl-server:8002",
            cert=cert_path,
            key=key_path,
            rootca=rootca,
            workspace=str(tmp_path / "ws"),
        )
        with pytest.raises(SystemExit) as exc_info:
            handle_package(args)
        assert exc_info.value.code == 4

    def test_literal_dummy_server_name_is_allowed_for_real_participant(self, cert_env, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        rootca = cert_env["rootca_path"]

        key_path, cert_path = _make_signed_cert(
            ca_key, ca_cert, "dummy-server", str(work), "dummy-server.crt", role="client"
        )
        ws = str(tmp_path / "ws")
        args = _make_args(
            kit_type="client",
            name="dummy-server",
            endpoint="grpc://fl-server:8002",
            cert=cert_path,
            key=key_path,
            rootca=rootca,
            workspace=ws,
        )
        handle_package(args)
        assert os.path.isdir(os.path.join(ws, "testproject", "prod_00", "dummy-server"))

    def test_server_kit_name_equals_endpoint_host_is_allowed(self, cert_env, tmp_path):
        """For server kits, name == endpoint hostname is the expected normal case."""
        work = tmp_path / "work"
        work.mkdir()
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        rootca = cert_env["rootca_path"]

        key_path, cert_path = _make_signed_cert(ca_key, ca_cert, "fl-server", str(work), "fl-server.crt", role="server")
        ws = str(tmp_path / "ws")
        args = _make_args(
            kit_type="server",
            name="fl-server",
            endpoint="grpc://fl-server:8002",
            cert=cert_path,
            key=key_path,
            rootca=rootca,
            workspace=ws,
        )
        handle_package(args)  # must not raise
        out_dir = _kit_dir(ws, "testproject", "fl-server")
        assert os.path.isfile(os.path.join(out_dir, "startup", "fed_server.json"))

    def test_admin_name_equals_endpoint_host_rejected(self, cert_env, tmp_path):
        """Admin name == endpoint hostname must also be rejected (non-server collision guard)."""
        work = tmp_path / "work"
        work.mkdir()
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        rootca = cert_env["rootca_path"]

        # Use an email that starts with the server hostname — but the exact-match collision guard
        # fires on the full name, not substring.  We need the name to be exactly == hostname.
        # Hostnames are not email format, so this would also be caught by the email validator
        # for admin types.  Test that the email validator fires for a non-email name first.
        key_path, cert_path = _make_signed_cert(
            ca_key, ca_cert, "server.example.com", str(work), "server.example.com.crt", role="lead"
        )
        args = _make_args(
            kit_type="lead",
            name="server.example.com",  # not email format
            endpoint="grpc://server.example.com:8002",
            cert=cert_path,
            key=key_path,
            rootca=rootca,
            workspace=str(tmp_path / "ws"),
        )
        with pytest.raises(SystemExit) as exc_info:
            handle_package(args)
        assert exc_info.value.code == 4


class TestS5RootCaPermissions:
    """S5: rootCA.pem must have 0o644 permissions after copying."""

    def test_rootca_permissions(self, cert_env, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        rootca = cert_env["rootca_path"]

        key_path, cert_path = _make_signed_cert(
            ca_key, ca_cert, "hospital-1", str(work), "hospital-1.crt", role="client"
        )
        ws = str(tmp_path / "ws")
        args = _make_args(
            kit_type="client",
            name="hospital-1",
            endpoint="grpc://server.example.com:8002",
            cert=cert_path,
            key=key_path,
            rootca=rootca,
            workspace=ws,
        )
        handle_package(args)

        out_dir = _kit_dir(ws, "testproject", "hospital-1")
        rootca_file = os.path.join(out_dir, "startup", "rootCA.pem")
        file_mode = stat.S_IMODE(os.stat(rootca_file).st_mode)
        assert file_mode == 0o644


# ---------------------------------------------------------------------------
# Additional coverage tests
# ---------------------------------------------------------------------------


class TestMissingType:
    """When -t is omitted and cert has no UNSTRUCTURED_NAME, CERT_TYPE_UNKNOWN is raised (exit 1)."""

    def test_missing_kit_type_cert_no_role_exits_1(self, cert_env, tmp_path):
        """No -t, cert has no UNSTRUCTURED_NAME → CERT_TYPE_UNKNOWN (exit 1)."""
        work = tmp_path / "work"
        work.mkdir()
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        rootca = cert_env["rootca_path"]

        key_path, cert_path = _make_signed_cert(ca_key, ca_cert, "hospital-1", str(work), "hospital-1.crt")
        args = _make_args(
            kit_type=None,  # not provided, and cert has no UNSTRUCTURED_NAME
            name="hospital-1",
            endpoint="grpc://server.example.com:8002",
            cert=cert_path,
            key=key_path,
            rootca=rootca,
            workspace=str(tmp_path / "ws"),
        )
        with pytest.raises(SystemExit) as exc_info:
            handle_package(args)
        assert exc_info.value.code == 1


class TestNoExtraServerDirForNonServerKit:
    """After packaging a non-server kit, the server placeholder dir must be removed."""

    def test_client_kit_no_server_dir_in_prod(self, cert_env, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        rootca = cert_env["rootca_path"]

        key_path, cert_path = _make_signed_cert(
            ca_key, ca_cert, "hospital-1", str(work), "hospital-1.crt", role="client"
        )
        ws = str(tmp_path / "ws")
        args = _make_args(
            kit_type="client",
            name="hospital-1",
            endpoint="grpc://server.example.com:8002",
            cert=cert_path,
            key=key_path,
            rootca=rootca,
            workspace=ws,
        )
        handle_package(args)

        prod_dir = os.path.join(ws, "testproject", "prod_00")
        entries = os.listdir(prod_dir)
        # Only the client dir should be present — server placeholder removed
        assert "hospital-1" in entries
        assert "server.example.com" not in entries
        assert len(entries) == 1

    def test_admin_kit_no_server_dir_in_prod(self, cert_env, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        rootca = cert_env["rootca_path"]

        key_path, cert_path = _make_signed_cert(
            ca_key, ca_cert, "alice@myorg.com", str(work), "alice@myorg.com.crt", role="lead"
        )
        ws = str(tmp_path / "ws")
        args = _make_args(
            kit_type="lead",
            name="alice@myorg.com",
            endpoint="grpc://server.example.com:8002",
            cert=cert_path,
            key=key_path,
            rootca=rootca,
            workspace=ws,
        )
        handle_package(args)

        prod_dir = os.path.join(ws, "testproject", "prod_00")
        entries = os.listdir(prod_dir)
        assert "alice@myorg.com" in entries
        assert "server.example.com" not in entries
        assert len(entries) == 1


class TestCertExpired:
    """Expired certificate must be rejected with exit code 1."""

    def test_expired_cert_exits_1(self, cert_env, tmp_path):
        """Patch package cert loading to return a cert whose expiry is in the past."""
        work = tmp_path / "work"
        work.mkdir()
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        rootca = cert_env["rootca_path"]

        key_path, cert_path = _make_signed_cert(
            ca_key, ca_cert, "hospital-1", str(work), "hospital-1.crt", role="client"
        )

        past = datetime.datetime(2000, 1, 1, tzinfo=datetime.timezone.utc)

        import nvflare.tool.package.package_commands as package_commands

        real_load_crt = package_commands._load_crt_nofollow

        def patched_load_crt(path):
            cert = real_load_crt(path)
            if path == cert_path:
                # Return a mock that looks like a cert but has expired
                mock_cert = unittest.mock.MagicMock(wraps=cert)
                mock_cert.not_valid_after_utc = past
                return mock_cert
            return cert

        with unittest.mock.patch(
            "nvflare.tool.package.package_commands._load_crt_nofollow", side_effect=patched_load_crt
        ):
            args = _make_args(
                kit_type="client",
                name="hospital-1",
                endpoint="grpc://server.example.com:8002",
                cert=cert_path,
                key=key_path,
                rootca=rootca,
                workspace=str(tmp_path / "ws"),
            )
            with pytest.raises(SystemExit) as exc_info:
                handle_package(args)
            assert exc_info.value.code == 1


class TestDirModeAdminEmailValidation:
    """In --dir mode, auto-discovered name that is not email format must be rejected for admin kits."""

    @pytest.mark.parametrize("kit_type", ["org_admin", "lead", "member"])
    def test_non_email_key_filename_rejected_for_admin_kit(self, kit_type, cert_env, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        rootca_src = cert_env["rootca_path"]

        # Key is named after a non-email participant — auto-discovery will produce non-email name
        key_path, cert_path = _make_signed_cert(
            ca_key, ca_cert, "admin-alice", str(work), "admin-alice.crt", role=kit_type
        )
        shutil.copy2(rootca_src, str(work / "rootCA.pem"))

        args = _make_args(
            kit_type=kit_type,
            endpoint="grpc://server.example.com:8002",
            dir=str(work),
            workspace=str(tmp_path / "ws"),
        )
        with pytest.raises(SystemExit) as exc_info:
            handle_package(args)
        assert exc_info.value.code == 4


# ---------------------------------------------------------------------------
# Regression tests: bugs that were found and fixed
# ---------------------------------------------------------------------------


class TestPartialExplicitModeArgs:
    """Partial explicit-mode args must produce INVALID_ARGS (exit 4), not TypeError."""

    @pytest.mark.parametrize(
        "present",
        [
            {"cert": "x.crt"},
            {"key": "x.key"},
            {"rootca": "rootCA.pem"},
            {"cert": "x.crt", "key": "x.key"},
            {"cert": "x.crt", "rootca": "rootCA.pem"},
            {"key": "x.key", "rootca": "rootCA.pem"},
        ],
    )
    def test_partial_explicit_exits_4_not_type_error(self, present, tmp_path):
        """Partial --cert/--key/--rootca must exit with code 4, not crash."""
        kwargs = dict(
            kit_type="client",
            name="hospital-1",
            endpoint="grpc://fl-server:8002",
            workspace=str(tmp_path / "ws"),
        )
        kwargs.update(present)
        args = _make_args(**kwargs)
        with pytest.raises(SystemExit) as exc_info:
            handle_package(args)
        assert exc_info.value.code == 4


class TestNextStepOutputPath:
    """next_step in output must contain the full kit directory path."""

    def test_server_next_step_contains_full_output_dir(self, cert_env, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        rootca = cert_env["rootca_path"]

        key_path, cert_path = _make_signed_cert(ca_key, ca_cert, "fl-server", str(work), "fl-server.crt", role="server")
        ws = str(tmp_path / "ws")
        args = _make_args(
            kit_type="server",
            name="fl-server",
            endpoint="grpc://fl-server:8002",
            cert=cert_path,
            key=key_path,
            rootca=rootca,
            workspace=ws,
        )
        import unittest.mock

        captured = {}

        def _capture(result):
            captured.update(result)

        with unittest.mock.patch("nvflare.tool.package.package_commands.output_ok", side_effect=_capture):
            handle_package(args)

        expected_dir = _kit_dir(ws, "testproject", "fl-server")
        assert (
            expected_dir in captured["next_step"]
        ), f"next_step {captured['next_step']!r} does not contain full output_dir {expected_dir!r}"

    def test_client_next_step_contains_full_output_dir(self, cert_env, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        rootca = cert_env["rootca_path"]

        key_path, cert_path = _make_signed_cert(
            ca_key, ca_cert, "hospital-1", str(work), "hospital-1.crt", role="client"
        )
        ws = str(tmp_path / "ws")
        args = _make_args(
            kit_type="client",
            name="hospital-1",
            endpoint="grpc://fl-server:8002",
            cert=cert_path,
            key=key_path,
            rootca=rootca,
            workspace=ws,
        )

        captured = {}

        def _capture(result):
            captured.update(result)

        with unittest.mock.patch("nvflare.tool.package.package_commands.output_ok", side_effect=_capture):
            handle_package(args)

        expected_dir = _kit_dir(ws, "testproject", "hospital-1")
        assert expected_dir in captured["next_step"]

    def test_admin_next_step_uses_fl_admin_sh(self, cert_env, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        rootca = cert_env["rootca_path"]

        key_path, cert_path = _make_signed_cert(
            ca_key, ca_cert, "alice@myorg.com", str(work), "alice@myorg.com.crt", role="lead"
        )
        ws = str(tmp_path / "ws")
        args = _make_args(
            kit_type="lead",
            name="alice@myorg.com",
            endpoint="grpc://fl-server:8002",
            cert=cert_path,
            key=key_path,
            rootca=rootca,
            workspace=ws,
        )

        captured = {}

        def _capture(result):
            captured.update(result)

        with unittest.mock.patch("nvflare.tool.package.package_commands.output_ok", side_effect=_capture):
            handle_package(args)

        expected_dir = _kit_dir(ws, "testproject", "alice@myorg.com")
        assert expected_dir in captured["next_step"]
        assert "fl_admin.sh" in captured["next_step"]


# ---------------------------------------------------------------------------
# Parity tests: nvflare provision vs nvflare package
# ---------------------------------------------------------------------------

# Files whose binary content is expected to differ between provision and package
# because they are derived from different CA/key material.
_CERT_EXTENSIONS = frozenset([".crt", ".key", ".pem"])

# Subdirectories expected inside every participant kit (excluding the root level).
_EXPECTED_SUBDIRS = {"startup", "local", "transfer"}

# JSON config files present in startup/ for each participant type.
_JSON_FILES = {
    "server": "fed_server.json",
    "client": "fed_client.json",
    "admin": "fed_admin.json",
}

# Cert-related keys inside JSON configs — content differs by design (different CAs).
_CERT_VALUE_KEYS = frozenset(["ssl_private_key", "ssl_cert", "ssl_root_cert", "client_key", "client_cert", "ca_cert"])


def _provision_project(workspace: str) -> str:
    """Run the standard provisioner and return the prod_NN directory path.

    The org ``"myorg"`` is used for all participants so that ``sub_start.sh``
    (which embeds ``org_name``) matches the ``handle_package`` output, which
    also hard-codes ``"myorg"``.
    """
    project = Project("testparity", "")
    project.set_server(
        "fl-server",
        "myorg",
        {
            PropKey.FED_LEARN_PORT: 8002,
            PropKey.ADMIN_PORT: 8003,
        },
    )
    project.add_client("hospital-1", "myorg", {})
    project.add_admin("admin@myorg.com", "myorg", {PropKey.ROLE: "org_admin"})

    provisioner = Provisioner(workspace, [WorkspaceBuilder(), CertBuilder(), StaticFileBuilder(scheme="grpc")])
    ctx = provisioner.provision(project)
    assert not ctx.get(CtxKey.BUILD_ERROR), "Provisioner reported a build error"
    return ctx[CtxKey.CURRENT_PROD_DIR]


def _run_package_server(ca_key, ca_cert, workspace: str, work_dir: str) -> str:
    """Assemble a server kit via handle_package and return the participant directory."""
    key_path, cert_path = _make_signed_cert(ca_key, ca_cert, "fl-server", work_dir, "fl-server.crt", role="server")
    rootca = os.path.join(work_dir, "rootCA.pem")
    args = _make_args(
        kit_type="server",
        name="fl-server",
        endpoint="grpc://fl-server:8002",
        cert=cert_path,
        key=key_path,
        rootca=rootca,
        workspace=workspace,
        project_name="testparity",
        admin_port=8003,
    )
    handle_package(args)
    return os.path.join(workspace, "testparity", "prod_00", "fl-server")


def _run_package_client(ca_key, ca_cert, workspace: str, work_dir: str) -> str:
    """Assemble a client kit via handle_package and return the participant directory."""
    key_path, cert_path = _make_signed_cert(ca_key, ca_cert, "hospital-1", work_dir, "hospital-1.crt", role="client")
    rootca = os.path.join(work_dir, "rootCA.pem")
    args = _make_args(
        kit_type="client",
        name="hospital-1",
        endpoint="grpc://fl-server:8002",
        cert=cert_path,
        key=key_path,
        rootca=rootca,
        workspace=workspace,
        project_name="testparity",
        admin_port=8003,
    )
    handle_package(args)
    return os.path.join(workspace, "testparity", "prod_00", "hospital-1")


def _run_package_admin(ca_key, ca_cert, workspace: str, work_dir: str) -> str:
    """Assemble an admin kit via handle_package and return the participant directory."""
    key_path, cert_path = _make_signed_cert(
        ca_key, ca_cert, "admin@myorg.com", work_dir, "admin@myorg.com.crt", role="org_admin"
    )
    rootca = os.path.join(work_dir, "rootCA.pem")
    args = _make_args(
        kit_type="org_admin",
        name="admin@myorg.com",
        endpoint="grpc://fl-server:8002",
        cert=cert_path,
        key=key_path,
        rootca=rootca,
        workspace=workspace,
        project_name="testparity",
        admin_port=8003,
    )
    handle_package(args)
    return os.path.join(workspace, "testparity", "prod_00", "admin@myorg.com")


def _file_tree(kit_dir: str) -> dict:
    """Return {relative_path: absolute_path} for all files under kit_dir."""
    result = {}
    for root, _dirs, files in os.walk(kit_dir):
        for f in files:
            abs_path = os.path.join(root, f)
            rel = os.path.relpath(abs_path, kit_dir)
            result[rel] = abs_path
    return result


def _compare_kit_dirs(prov_dir: str, pkg_dir: str, participant_type: str) -> list:
    """Compare provision and package kit directories structurally.

    Returns a list of difference description strings (empty means identical).
    Asserts are raised for structural differences; content differences for
    cert material are silently skipped.

    Args:
        prov_dir: path to the provisioner-generated participant kit directory.
        pkg_dir: path to the handle_package-generated participant kit directory.
        participant_type: one of ``"server"``, ``"client"``, ``"admin"``.

    Returns:
        List of documented intentional differences (for informational purposes).
    """
    intentional_diffs = []

    # 1. Both directories must exist.
    assert os.path.isdir(prov_dir), f"Provision kit dir missing: {prov_dir}"
    assert os.path.isdir(pkg_dir), f"Package kit dir missing: {pkg_dir}"

    # 2. Expected subdirectory structure.
    for subdir in _EXPECTED_SUBDIRS:
        assert os.path.isdir(os.path.join(prov_dir, subdir)), f"prov missing subdir: {subdir}"
        assert os.path.isdir(os.path.join(pkg_dir, subdir)), f"pkg missing subdir: {subdir}"

    # 3. File sets must match exactly.
    prov_files = set(_file_tree(prov_dir).keys())
    pkg_files = set(_file_tree(pkg_dir).keys())
    prov_only = prov_files - pkg_files
    pkg_only = pkg_files - prov_files
    assert not prov_only, f"Files only in provision output: {sorted(prov_only)}"
    assert not pkg_only, f"Files only in package output: {sorted(pkg_only)}"

    # 4. Per-file content comparison.
    prov_tree = _file_tree(prov_dir)
    pkg_tree = _file_tree(pkg_dir)

    for rel in sorted(prov_files):
        ext = os.path.splitext(rel)[1]

        # 4a. Cert content differs by design (different CA/key material per run).
        if ext in _CERT_EXTENSIONS:
            intentional_diffs.append(f"cert-content-differs: {rel}")
            continue

        with open(prov_tree[rel], "rb") as f:
            prov_content = f.read()
        with open(pkg_tree[rel], "rb") as f:
            pkg_content = f.read()

        # 4b. JSON files: compare keys and non-cert values.
        if rel.endswith(".json"):
            prov_obj = json.loads(prov_content)
            pkg_obj = json.loads(pkg_content)
            assert set(prov_obj.keys()) == set(
                pkg_obj.keys()
            ), f"{rel}: top-level key mismatch — prov={sorted(prov_obj.keys())} pkg={sorted(pkg_obj.keys())}"
            for key in prov_obj:
                if key in _CERT_VALUE_KEYS:
                    # Cert-path values may differ in content; both must be str.
                    assert isinstance(prov_obj[key], str), f"{rel}[{key}] prov value is not str"
                    assert isinstance(pkg_obj[key], str), f"{rel}[{key}] pkg value is not str"
                    intentional_diffs.append(f"cert-value-skipped: {rel}[{key}]")
                else:
                    assert prov_obj[key] == pkg_obj[key], f"{rel}[{key}]: prov={prov_obj[key]!r} pkg={pkg_obj[key]!r}"
            continue

        # 4c. All other files must be byte-for-byte identical.
        assert prov_content == pkg_content, f"Content mismatch for {rel}"

    # 5. Key file permissions.
    #    - ``handle_package`` (PrebuiltCertBuilder) always writes 0o600.
    #    - Standard CertBuilder (used by the provisioner) does not chmod keys; they
    #      are created at the OS default (typically 0o644).
    #    We assert ``package`` output is always 0o600 (security requirement), and
    #    document the provisioner behaviour as an intentional known difference.
    for rel in sorted(prov_files):
        if os.path.splitext(rel)[1] == ".key":
            prov_mode = stat.S_IMODE(os.stat(prov_tree[rel]).st_mode)
            pkg_mode = stat.S_IMODE(os.stat(pkg_tree[rel]).st_mode)
            assert pkg_mode == 0o600, f"pkg  {rel} permissions {oct(pkg_mode)} != 0o600"
            if prov_mode != 0o600:
                # CertBuilder does not restrict key permissions — document but don't fail.
                intentional_diffs.append(f"key-perm-prov-{oct(prov_mode)}-pkg-0o600: {rel}")

    # 6. rootCA.pem must be readable (0o644) in both outputs.
    for rel in sorted(prov_files):
        if rel.endswith("rootCA.pem"):
            prov_mode = stat.S_IMODE(os.stat(prov_tree[rel]).st_mode)
            pkg_mode = stat.S_IMODE(os.stat(pkg_tree[rel]).st_mode)
            assert prov_mode == 0o644, f"prov {rel} permissions {oct(prov_mode)} != 0o644"
            assert pkg_mode == 0o644, f"pkg  {rel} permissions {oct(pkg_mode)} != 0o644"

    return intentional_diffs


@pytest.fixture()
def parity_env(tmp_path):
    """Shared environment for all parity tests: one CA + rootCA written to work_dir."""
    work_dir = str(tmp_path / "work")
    os.makedirs(work_dir)
    ca_key, ca_cert, rootca_path = _make_ca(work_dir)
    # _make_ca writes rootCA.pem into work_dir — package helpers read it from there.
    return dict(
        work_dir=work_dir,
        ca_key=ca_key,
        ca_cert=ca_cert,
        rootca_path=rootca_path,
        tmp_path=tmp_path,
    )


class TestProvisionPackageParity:
    """Assert that 'nvflare package' output is structurally identical to 'nvflare provision' output.

    Both paths provision a project named ``testparity`` with:
      - server:  fl-server  (grpc, ports 8002/8003)
      - client:  hospital-1
      - admin:   admin@myorg.com  (org_admin role)

    The org is set to ``"myorg"`` on both sides so that sub_start.sh (which embeds
    ``org_name``) is identical.  Cert content differs by design because each run
    uses a freshly generated CA.
    """

    # Prefixes of entries in the ``diffs`` list that are documented intentional
    # differences — not failures.  Any diff entry *not* matching one of these
    # prefixes causes the test to fail.
    _KNOWN_DIFF_PREFIXES = (
        # Cert binary content is always different (different CA/key per run).
        "cert-content-differs:",
        # Cert-related JSON values are strings in both; actual content differs.
        "cert-value-skipped:",
        # CertBuilder does not chmod key files; PrebuiltCertBuilder always sets 0o600.
        "key-perm-prov-",
    )

    @staticmethod
    def _unexpected_diffs(diffs: list) -> list:
        return [d for d in diffs if not any(d.startswith(p) for p in TestProvisionPackageParity._KNOWN_DIFF_PREFIXES)]

    def test_server_kit_parity(self, parity_env, tmp_path):
        """Server startup kit from provision and package must be structurally identical."""
        prov_prod = _provision_project(str(tmp_path / "prov_ws"))
        prov_server_dir = os.path.join(prov_prod, "fl-server")

        pkg_server_dir = _run_package_server(
            parity_env["ca_key"],
            parity_env["ca_cert"],
            str(tmp_path / "pkg_ws"),
            parity_env["work_dir"],
        )

        diffs = _compare_kit_dirs(prov_server_dir, pkg_server_dir, "server")
        unexpected = self._unexpected_diffs(diffs)
        assert not unexpected, f"Unexpected differences in server kit: {unexpected}"

    def test_client_kit_parity(self, parity_env, tmp_path):
        """Client startup kit from provision and package must be structurally identical."""
        prov_prod = _provision_project(str(tmp_path / "prov_ws"))
        prov_client_dir = os.path.join(prov_prod, "hospital-1")

        pkg_client_dir = _run_package_client(
            parity_env["ca_key"],
            parity_env["ca_cert"],
            str(tmp_path / "pkg_ws"),
            parity_env["work_dir"],
        )

        diffs = _compare_kit_dirs(prov_client_dir, pkg_client_dir, "client")
        unexpected = self._unexpected_diffs(diffs)
        assert not unexpected, f"Unexpected differences in client kit: {unexpected}"

    def test_admin_kit_parity(self, parity_env, tmp_path):
        """Admin startup kit from provision and package must be structurally identical."""
        prov_prod = _provision_project(str(tmp_path / "prov_ws"))
        prov_admin_dir = os.path.join(prov_prod, "admin@myorg.com")

        pkg_admin_dir = _run_package_admin(
            parity_env["ca_key"],
            parity_env["ca_cert"],
            str(tmp_path / "pkg_ws"),
            parity_env["work_dir"],
        )

        diffs = _compare_kit_dirs(prov_admin_dir, pkg_admin_dir, "admin")
        unexpected = self._unexpected_diffs(diffs)
        assert not unexpected, f"Unexpected differences in admin kit: {unexpected}"

    def test_all_three_participants_parity(self, parity_env, tmp_path):
        """All three participant kits pass the parity check in one provisioner run."""
        prov_prod = _provision_project(str(tmp_path / "prov_ws"))

        pkg_server_dir = _run_package_server(
            parity_env["ca_key"],
            parity_env["ca_cert"],
            str(tmp_path / "pkg_s"),
            parity_env["work_dir"],
        )
        pkg_client_dir = _run_package_client(
            parity_env["ca_key"],
            parity_env["ca_cert"],
            str(tmp_path / "pkg_c"),
            parity_env["work_dir"],
        )
        pkg_admin_dir = _run_package_admin(
            parity_env["ca_key"],
            parity_env["ca_cert"],
            str(tmp_path / "pkg_a"),
            parity_env["work_dir"],
        )

        for participant, prov_name, pkg_dir in [
            ("server", "fl-server", pkg_server_dir),
            ("client", "hospital-1", pkg_client_dir),
            ("admin", "admin@myorg.com", pkg_admin_dir),
        ]:
            prov_dir = os.path.join(prov_prod, prov_name)
            diffs = _compare_kit_dirs(prov_dir, pkg_dir, participant)
            unexpected = self._unexpected_diffs(diffs)
            assert not unexpected, f"Unexpected differences in {participant} kit: {unexpected}"


# ---------------------------------------------------------------------------
# Yaml mode tests: --project-file
# ---------------------------------------------------------------------------


def _write_project_yaml(path, participants, project_name="myproject", builders=None):
    """Write a minimal api_version-3 project yaml to *path*.

    ``builders`` is an optional list of dicts with ``path`` and optional ``args`` keys,
    e.g. ``[{"path": "nvflare.lighter.impl.workspace.WorkspaceBuilder"}]``.
    """
    lines = [
        "api_version: 3",
        f"name: {project_name}",
        'description: ""',
        "participants:",
    ]
    for p in participants:
        lines.append(f"  - name: {p['name']}")
        lines.append(f"    type: {p['type']}")
        lines.append(f"    org: {p.get('org', 'myorg')}")
        if "role" in p:
            lines.append(f"    role: {p['role']}")
    if builders:
        lines.append("builders:")
        for b in builders:
            lines.append(f"  - path: {b['path']}")
            if b.get("args"):
                lines.append("    args:")
                for k, v in b["args"].items():
                    value = f'"{v}"' if isinstance(v, str) else v
                    lines.append(f"      {k}: {value}")
    path.write_text("\n".join(lines) + "\n")


def _write_single_site_yaml(path, data):
    lines = []
    for k, v in data.items():
        if isinstance(v, dict):
            lines.append(f"{k}:")
            for dk, dv in v.items():
                value = f'"{dv}"' if isinstance(dv, str) else dv
                lines.append(f"  {dk}: {value}")
        else:
            value = f'"{v}"' if isinstance(v, str) else v
            lines.append(f"{k}: {value}")
    path.write_text("\n".join(lines) + "\n")


class TestYamlMode:
    """Tests for _handle_package_yaml_mode (invoked via handle_package --project-file)."""

    # ------------------------------------------------------------------
    # 1. All participants land in the SAME prod_NN directory
    # ------------------------------------------------------------------
    def test_all_participants_in_same_prod_dir(self, cert_env, tmp_path):
        """Two clients from one --project-file call must both be in prod_00, not prod_01."""
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        cert_dir = tmp_path / "certs"
        cert_dir.mkdir()
        shutil.copy2(cert_env["rootca_path"], str(cert_dir / "rootCA.pem"))

        _make_signed_cert(ca_key, ca_cert, "hospital-1", str(cert_dir), "hospital-1.crt", role="client")
        _make_signed_cert(ca_key, ca_cert, "hospital-2", str(cert_dir), "hospital-2.crt", role="client")

        project_yaml = tmp_path / "project.yaml"
        _write_project_yaml(
            project_yaml,
            [
                {"name": "hospital-1", "type": "client"},
                {"name": "hospital-2", "type": "client"},
            ],
        )

        ws = str(tmp_path / "ws")
        args = _make_args(
            kit_type=None,
            endpoint="grpc://fl-server:8002",
            dir=str(cert_dir),
            workspace=ws,
            project_name="myproject",
            project_file=str(project_yaml),
        )
        handle_package(args)

        prod_dir = os.path.join(ws, "myproject", "prod_00")
        assert os.path.isdir(os.path.join(prod_dir, "hospital-1")), "hospital-1 not in prod_00"
        assert os.path.isdir(os.path.join(prod_dir, "hospital-2")), "hospital-2 not in prod_00"
        # Neither participant should appear in a separate prod_01
        assert not os.path.isdir(os.path.join(ws, "myproject", "prod_01")), "unexpected prod_01 created"

    def test_single_site_yaml_preserves_listening_host_for_comm_config(self, cert_env, tmp_path):
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        cert_dir = tmp_path / "certs"
        cert_dir.mkdir()
        shutil.copy2(cert_env["rootca_path"], str(cert_dir / "rootCA.pem"))
        _make_signed_cert(ca_key, ca_cert, "hospital-1", str(cert_dir), "hospital-1.crt", role="client")

        site_yaml = tmp_path / "site.yaml"
        _write_single_site_yaml(
            site_yaml,
            {
                "name": "hospital-1",
                "type": "client",
                "org": "myorg",
                "listening_host": {
                    "scheme": "grpc",
                    "default_host": "hospital-1",
                    "port": 9001,
                    "connection_security": "mtls",
                },
            },
        )

        ws = str(tmp_path / "ws")
        args = _make_args(
            kit_type=None,
            endpoint="grpc://fl-server:8002",
            dir=str(cert_dir),
            workspace=ws,
            project_name="myproject",
            project_file=str(site_yaml),
        )
        handle_package(args)

        comm_config = os.path.join(ws, "myproject", "prod_00", "hospital-1", "local", "comm_config.json")
        assert os.path.isfile(comm_config)
        with open(comm_config) as f:
            cfg = json.load(f)
        assert cfg["internal"]["scheme"] == "grpc"
        assert cfg["internal"]["resources"]["host"] == "hospital-1"
        assert cfg["internal"]["resources"]["port"] == 9001
        assert cfg["internal"]["resources"]["connection_security"] == "mtls"

    def test_yaml_server_participant_preserves_listening_host_for_comm_config(self, cert_env, tmp_path):
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        cert_dir = tmp_path / "certs"
        cert_dir.mkdir()
        shutil.copy2(cert_env["rootca_path"], str(cert_dir / "rootCA.pem"))
        _make_signed_cert(ca_key, ca_cert, "fl-server", str(cert_dir), "fl-server.crt", role="server")

        project_yaml = tmp_path / "project.yaml"
        _write_project_yaml(
            project_yaml,
            [
                {
                    "name": "fl-server",
                    "type": "server",
                    "listening_host": {
                        "scheme": "grpc",
                        "default_host": "fl-server",
                        "port": 8102,
                        "connection_security": "mtls",
                    },
                }
            ],
        )

        # _write_project_yaml only handles flat keys, so patch in nested listening_host.
        project_yaml.write_text(
            "\n".join(
                [
                    "api_version: 3",
                    "name: myproject",
                    'description: ""',
                    "participants:",
                    "  - name: fl-server",
                    "    type: server",
                    "    org: myorg",
                    "    listening_host:",
                    "      scheme: grpc",
                    "      default_host: fl-server",
                    "      port: 8102",
                    "      connection_security: mtls",
                ]
            )
            + "\n"
        )

        ws = str(tmp_path / "ws")
        args = _make_args(
            kit_type=None,
            endpoint="grpc://fl-server:8002",
            dir=str(cert_dir),
            workspace=ws,
            project_name="myproject",
            project_file=str(project_yaml),
        )
        handle_package(args)

        comm_config = os.path.join(ws, "myproject", "prod_00", "fl-server", "local", "comm_config.json")
        assert os.path.isfile(comm_config)
        with open(comm_config) as f:
            cfg = json.load(f)
        assert cfg["internal"]["scheme"] == "grpc"
        assert cfg["internal"]["resources"]["host"] == "fl-server"
        assert cfg["internal"]["resources"]["port"] == 8102
        assert cfg["internal"]["resources"]["connection_security"] == "mtls"

    # ------------------------------------------------------------------
    # 2. -t client filter builds only clients
    # ------------------------------------------------------------------
    def test_type_filter_client_only(self, cert_env, tmp_path):
        """-t client with a yaml containing server + clients should only build clients."""
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        cert_dir = tmp_path / "certs"
        cert_dir.mkdir()
        shutil.copy2(cert_env["rootca_path"], str(cert_dir / "rootCA.pem"))

        _make_signed_cert(ca_key, ca_cert, "hospital-1", str(cert_dir), "hospital-1.crt", role="client")
        _make_signed_cert(ca_key, ca_cert, "hospital-2", str(cert_dir), "hospital-2.crt", role="client")
        # Server cert exists in dir (not needed because filter excludes server)
        _make_signed_cert(ca_key, ca_cert, "fl-server", str(cert_dir), "fl-server.crt", role="server")

        project_yaml = tmp_path / "project.yaml"
        _write_project_yaml(
            project_yaml,
            [
                {"name": "fl-server", "type": "server"},
                {"name": "hospital-1", "type": "client"},
                {"name": "hospital-2", "type": "client"},
            ],
        )

        ws = str(tmp_path / "ws")
        args = _make_args(
            kit_type="client",
            endpoint="grpc://fl-server:8002",
            dir=str(cert_dir),
            workspace=ws,
            project_name="myproject",
            project_file=str(project_yaml),
        )
        handle_package(args)

        prod_dir = os.path.join(ws, "myproject", "prod_00")
        assert os.path.isdir(os.path.join(prod_dir, "hospital-1"))
        assert os.path.isdir(os.path.join(prod_dir, "hospital-2"))
        # Server kit must NOT be present
        assert not os.path.isdir(os.path.join(prod_dir, "fl-server")), "server dir unexpectedly present"

    # ------------------------------------------------------------------
    # 3. -t server on a clients-only yaml → NO_PARTICIPANTS error
    # ------------------------------------------------------------------
    def test_type_filter_no_match_exits(self, cert_env, tmp_path):
        """-t server on a yaml with only clients must exit with NO_PARTICIPANTS."""
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        cert_dir = tmp_path / "certs"
        cert_dir.mkdir()
        shutil.copy2(cert_env["rootca_path"], str(cert_dir / "rootCA.pem"))
        _make_signed_cert(ca_key, ca_cert, "hospital-1", str(cert_dir), "hospital-1.crt", role="client")

        project_yaml = tmp_path / "project.yaml"
        _write_project_yaml(project_yaml, [{"name": "hospital-1", "type": "client"}])

        args = _make_args(
            kit_type="server",
            endpoint="grpc://fl-server:8002",
            dir=str(cert_dir),
            workspace=str(tmp_path / "ws"),
            project_name="myproject",
            project_file=str(project_yaml),
        )
        with pytest.raises(SystemExit) as exc_info:
            handle_package(args)
        assert exc_info.value.code == 1

    # ------------------------------------------------------------------
    # 4. Non-existent project file → PROJECT_FILE_NOT_FOUND (exit 1)
    # ------------------------------------------------------------------
    def test_project_file_not_found_exits(self, cert_env, tmp_path):
        """Passing a non-existent --project-file must exit with code 1."""
        cert_dir = tmp_path / "certs"
        cert_dir.mkdir()
        shutil.copy2(cert_env["rootca_path"], str(cert_dir / "rootCA.pem"))

        args = _make_args(
            kit_type=None,
            endpoint="grpc://fl-server:8002",
            dir=str(cert_dir),
            workspace=str(tmp_path / "ws"),
            project_file=str(tmp_path / "nonexistent.yaml"),
        )
        with pytest.raises(SystemExit) as exc_info:
            handle_package(args)
        assert exc_info.value.code == 1

    # ------------------------------------------------------------------
    # 5. relay participant in yaml → UNSUPPORTED_TOPOLOGY (exit 4)
    # ------------------------------------------------------------------
    def test_relay_participant_rejected(self, cert_env, tmp_path):
        """A yaml containing a relay participant must exit with code 4."""
        cert_dir = tmp_path / "certs"
        cert_dir.mkdir()
        shutil.copy2(cert_env["rootca_path"], str(cert_dir / "rootCA.pem"))

        project_yaml = tmp_path / "project.yaml"
        _write_project_yaml(
            project_yaml,
            [
                {"name": "hospital-1", "type": "client"},
                {"name": "relay-1", "type": "relay"},
            ],
        )

        args = _make_args(
            kit_type=None,
            endpoint="grpc://fl-server:8002",
            dir=str(cert_dir),
            workspace=str(tmp_path / "ws"),
            project_file=str(project_yaml),
        )
        with pytest.raises(SystemExit) as exc_info:
            handle_package(args)
        assert exc_info.value.code == 4

    # ------------------------------------------------------------------
    # 6. Cert file missing for a participant → CERT_NOT_FOUND (exit 1)
    # ------------------------------------------------------------------
    def test_missing_cert_exits(self, cert_env, tmp_path):
        """If a participant cert file is absent, must exit with code 1."""
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        cert_dir = tmp_path / "certs"
        cert_dir.mkdir()
        shutil.copy2(cert_env["rootca_path"], str(cert_dir / "rootCA.pem"))
        # Write the key but NOT the cert
        _make_signed_cert(ca_key, ca_cert, "hospital-1", str(cert_dir), "hospital-1.crt", role="client")
        os.remove(str(cert_dir / "hospital-1.crt"))

        project_yaml = tmp_path / "project.yaml"
        _write_project_yaml(project_yaml, [{"name": "hospital-1", "type": "client"}])

        args = _make_args(
            kit_type=None,
            endpoint="grpc://fl-server:8002",
            dir=str(cert_dir),
            workspace=str(tmp_path / "ws"),
            project_file=str(project_yaml),
        )
        with pytest.raises(SystemExit) as exc_info:
            handle_package(args)
        assert exc_info.value.code == 1

    # ------------------------------------------------------------------
    # 7. rootCA.pem missing from --dir → ROOTCA_NOT_FOUND (exit 1)
    # ------------------------------------------------------------------
    def test_missing_rootca_exits(self, cert_env, tmp_path):
        """If rootCA.pem is absent from --dir, must exit with code 1."""
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        cert_dir = tmp_path / "certs"
        cert_dir.mkdir()
        # Deliberately do NOT write rootCA.pem
        _make_signed_cert(ca_key, ca_cert, "hospital-1", str(cert_dir), "hospital-1.crt", role="client")

        project_yaml = tmp_path / "project.yaml"
        _write_project_yaml(project_yaml, [{"name": "hospital-1", "type": "client"}])

        args = _make_args(
            kit_type=None,
            endpoint="grpc://fl-server:8002",
            dir=str(cert_dir),
            workspace=str(tmp_path / "ws"),
            project_file=str(project_yaml),
        )
        with pytest.raises(SystemExit) as exc_info:
            handle_package(args)
        assert exc_info.value.code == 1

    # ------------------------------------------------------------------
    # 8. --project-file and -n/--name are mutually exclusive
    # ------------------------------------------------------------------
    def test_project_name_and_file_mutually_exclusive(self, cert_env, tmp_path):
        """Setting both --project-file and -n must exit with INVALID_ARGS (exit 4)."""
        cert_dir = tmp_path / "certs"
        cert_dir.mkdir()
        shutil.copy2(cert_env["rootca_path"], str(cert_dir / "rootCA.pem"))

        project_yaml = tmp_path / "project.yaml"
        _write_project_yaml(project_yaml, [{"name": "hospital-1", "type": "client"}])

        args = _make_args(
            kit_type=None,
            name="hospital-1",  # mutually exclusive with project_file
            endpoint="grpc://fl-server:8002",
            dir=str(cert_dir),
            workspace=str(tmp_path / "ws"),
            project_file=str(project_yaml),
        )
        with pytest.raises(SystemExit) as exc_info:
            handle_package(args)
        assert exc_info.value.code == 4

    # ------------------------------------------------------------------
    # 9. --dir is required in yaml mode
    # ------------------------------------------------------------------
    def test_dir_required_in_yaml_mode(self, cert_env, tmp_path):
        """--project-file without --dir must exit with INVALID_ARGS (exit 4)."""
        project_yaml = tmp_path / "project.yaml"
        _write_project_yaml(project_yaml, [{"name": "hospital-1", "type": "client"}])

        args = _make_args(
            kit_type=None,
            endpoint="grpc://fl-server:8002",
            dir=None,  # not provided
            workspace=str(tmp_path / "ws"),
            project_file=str(project_yaml),
        )
        with pytest.raises(SystemExit) as exc_info:
            handle_package(args)
        assert exc_info.value.code == 4

    # ------------------------------------------------------------------
    # 10. No dummy server dir in prod when yaml has no server participant
    # ------------------------------------------------------------------
    def test_no_dummy_server_dir_in_prod(self, cert_env, tmp_path):
        """When yaml has only clients (no server), the dummy server dir must be absent from prod."""
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        cert_dir = tmp_path / "certs"
        cert_dir.mkdir()
        shutil.copy2(cert_env["rootca_path"], str(cert_dir / "rootCA.pem"))
        _make_signed_cert(ca_key, ca_cert, "hospital-1", str(cert_dir), "hospital-1.crt", role="client")

        project_yaml = tmp_path / "project.yaml"
        _write_project_yaml(project_yaml, [{"name": "hospital-1", "type": "client"}])

        ws = str(tmp_path / "ws")
        args = _make_args(
            kit_type=None,
            endpoint="grpc://fl-server:8002",
            dir=str(cert_dir),
            workspace=ws,
            project_name="myproject",
            project_file=str(project_yaml),
        )
        handle_package(args)

        prod_dir = os.path.join(ws, "myproject", "prod_00")
        entries = os.listdir(prod_dir)
        assert "hospital-1" in entries, "client kit not found in prod_00"
        # The dummy server directory (named after the endpoint host) must have been removed
        assert "fl-server" not in entries, "dummy server dir unexpectedly present in prod_00"
        assert len(entries) == 1, f"unexpected extra entries in prod_00: {entries}"

    # ------------------------------------------------------------------
    # 11. Malformed rootCA.pem in YAML mode → structured error, not traceback
    # ------------------------------------------------------------------
    def test_malformed_rootca_exits_with_structured_error(self, cert_env, tmp_path):
        """A corrupt rootCA.pem must produce a structured CLI error (exit 1), not a raw traceback."""
        cert_dir = tmp_path / "certs"
        cert_dir.mkdir()
        # Write an invalid (non-PEM) rootCA.pem
        (cert_dir / "rootCA.pem").write_text("this is not a valid certificate\n")

        project_yaml = tmp_path / "project.yaml"
        _write_project_yaml(project_yaml, [{"name": "hospital-1", "type": "client"}])

        args = _make_args(
            kit_type=None,
            endpoint="grpc://fl-server:8002",
            dir=str(cert_dir),
            workspace=str(tmp_path / "ws"),
            project_file=str(project_yaml),
        )
        with pytest.raises(SystemExit) as exc_info:
            handle_package(args)
        assert exc_info.value.code == 1

    # ------------------------------------------------------------------
    # 12. Empty participant list after filter → NO_PARTICIPANTS (exit 1)
    # ------------------------------------------------------------------
    def test_empty_participants_after_filter_exits(self, cert_env, tmp_path):
        """yaml with only clients filtered by -t server → no participants → exit 1."""
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        cert_dir = tmp_path / "certs"
        cert_dir.mkdir()
        shutil.copy2(cert_env["rootca_path"], str(cert_dir / "rootCA.pem"))
        _make_signed_cert(ca_key, ca_cert, "hospital-1", str(cert_dir), "hospital-1.crt", role="client")

        project_yaml = tmp_path / "project.yaml"
        _write_project_yaml(project_yaml, [{"name": "hospital-1", "type": "client"}])

        args = _make_args(
            kit_type="server",  # filter that matches nothing
            endpoint="grpc://fl-server:8002",
            dir=str(cert_dir),
            workspace=str(tmp_path / "ws"),
            project_file=str(project_yaml),
        )
        with pytest.raises(SystemExit) as exc_info:
            handle_package(args)
        assert exc_info.value.code == 1

    # ------------------------------------------------------------------
    # 13. Mixed types (client + admin user) all land in same prod_NN
    # ------------------------------------------------------------------
    def test_mixed_types_all_in_same_prod(self, cert_env, tmp_path):
        """client + lead admin user kits must both appear in prod_00.

        Note: server kits are only built when -t server is explicit; clients
        and admin users are built together when no type filter is given.
        """
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        cert_dir = tmp_path / "certs"
        cert_dir.mkdir()
        shutil.copy2(cert_env["rootca_path"], str(cert_dir / "rootCA.pem"))
        _make_signed_cert(ca_key, ca_cert, "hospital-1", str(cert_dir), "hospital-1.crt", role="client")
        _make_signed_cert(ca_key, ca_cert, "alice@hospital.com", str(cert_dir), "alice@hospital.com.crt", role="lead")

        project_yaml = tmp_path / "project.yaml"
        _write_project_yaml(
            project_yaml,
            [
                {"name": "hospital-1", "type": "client"},
                {"name": "alice@hospital.com", "type": "admin", "role": "lead"},
            ],
        )

        ws = str(tmp_path / "ws")
        args = _make_args(
            kit_type=None,
            endpoint="grpc://fl-server:8002",
            dir=str(cert_dir),
            workspace=ws,
            project_name="myproject",
            project_file=str(project_yaml),
        )
        handle_package(args)

        prod_dir = os.path.join(ws, "myproject", "prod_00")
        assert os.path.isdir(os.path.join(prod_dir, "hospital-1"))
        assert os.path.isdir(os.path.join(prod_dir, "alice@hospital.com"))
        assert not os.path.isdir(os.path.join(ws, "myproject", "prod_01"))

    # ------------------------------------------------------------------
    # 14. --project-name override is reflected in output path
    # ------------------------------------------------------------------
    def test_project_name_override_in_output_path(self, cert_env, tmp_path):
        """--project-name sets the project directory in the workspace output path."""
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        cert_dir = tmp_path / "certs"
        cert_dir.mkdir()
        shutil.copy2(cert_env["rootca_path"], str(cert_dir / "rootCA.pem"))
        _make_signed_cert(ca_key, ca_cert, "hospital-1", str(cert_dir), "hospital-1.crt", role="client")

        project_yaml = tmp_path / "project.yaml"
        _write_project_yaml(project_yaml, [{"name": "hospital-1", "type": "client"}])

        ws = str(tmp_path / "ws")
        args = _make_args(
            kit_type=None,
            endpoint="grpc://fl-server:8002",
            dir=str(cert_dir),
            workspace=ws,
            project_name="custom-project",
            project_file=str(project_yaml),
        )
        handle_package(args)

        assert os.path.isdir(os.path.join(ws, "custom-project", "prod_00", "hospital-1"))
        assert not os.path.isdir(os.path.join(ws, "myproject")), "yaml project name must not override --project-name"

    # ------------------------------------------------------------------
    # 15. --force creates new prod_NN when participant already exists
    # ------------------------------------------------------------------
    def test_force_creates_new_prod_dir(self, cert_env, tmp_path):
        """Re-packaging with --force when prod_00 exists must create prod_01."""
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        cert_dir = tmp_path / "certs"
        cert_dir.mkdir()
        shutil.copy2(cert_env["rootca_path"], str(cert_dir / "rootCA.pem"))
        _make_signed_cert(ca_key, ca_cert, "hospital-1", str(cert_dir), "hospital-1.crt", role="client")

        project_yaml = tmp_path / "project.yaml"
        _write_project_yaml(project_yaml, [{"name": "hospital-1", "type": "client"}])

        ws = str(tmp_path / "ws")
        base_args = dict(
            kit_type=None,
            endpoint="grpc://fl-server:8002",
            dir=str(cert_dir),
            workspace=ws,
            project_name="myproject",
            project_file=str(project_yaml),
        )

        # First run → prod_00
        handle_package(_make_args(**base_args))
        assert os.path.isdir(os.path.join(ws, "myproject", "prod_00", "hospital-1"))

        # Second run with force=True → prod_01
        handle_package(_make_args(force=True, **base_args))
        assert os.path.isdir(os.path.join(ws, "myproject", "prod_01", "hospital-1"))

    # ------------------------------------------------------------------
    # 16. YAML that already includes WorkspaceBuilder must not double-run finalize
    # ------------------------------------------------------------------
    def test_yaml_with_workspace_builder_does_not_fail(self, cert_env, tmp_path):
        """If project.yaml already lists WorkspaceBuilder, the dedup filter must prevent
        BUILD_FAILED from finalize() running twice."""
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        cert_dir = tmp_path / "certs"
        cert_dir.mkdir()
        shutil.copy2(cert_env["rootca_path"], str(cert_dir / "rootCA.pem"))
        _make_signed_cert(ca_key, ca_cert, "hospital-1", str(cert_dir), "hospital-1.crt", role="client")

        # Write a yaml that includes WorkspaceBuilder in the builders list (common in
        # project.yaml files that were used with nvflare provision).
        project_yaml = tmp_path / "project.yaml"
        project_yaml.write_text(
            "api_version: 3\n"
            "name: myproject\n"
            'description: ""\n'
            "participants:\n"
            "  - name: hospital-1\n"
            "    type: client\n"
            "    org: myorg\n"
            "builders:\n"
            "  - path: nvflare.lighter.impl.workspace.WorkspaceBuilder\n"
            "  - path: nvflare.lighter.impl.static_file.StaticFileBuilder\n"
        )

        ws = str(tmp_path / "ws")
        args = _make_args(
            kit_type=None,
            endpoint="grpc://fl-server:8002",
            dir=str(cert_dir),
            workspace=ws,
            project_name="myproject",
            project_file=str(project_yaml),
        )
        # Must not raise SystemExit (BUILD_FAILED)
        handle_package(args)
        assert os.path.isdir(os.path.join(ws, "myproject", "prod_00", "hospital-1"))


# ---------------------------------------------------------------------------
# TestKitTypeFromCert: -t derived from cert UNSTRUCTURED_NAME
# ---------------------------------------------------------------------------


class TestKitTypeFromCert:
    """Kit type is derived from the signed cert when -t is omitted."""

    def test_client_kit_type_from_cert(self, cert_env, tmp_path):
        """Cert with UNSTRUCTURED_NAME=client → client kit assembled without -t."""
        work = tmp_path / "work"
        work.mkdir()
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        rootca = cert_env["rootca_path"]

        key_path, cert_path = _make_signed_cert(
            ca_key, ca_cert, "hospital-1", str(work), "hospital-1.crt", role="client"
        )
        ws = str(tmp_path / "ws")
        args = _make_args(
            kit_type=None,  # not provided — derived from cert
            name="hospital-1",
            endpoint="grpc://server.example.com:8002",
            cert=cert_path,
            key=key_path,
            rootca=rootca,
            workspace=ws,
        )
        handle_package(args)

        out_dir = _kit_dir(ws, "testproject", "hospital-1")
        assert os.path.isfile(os.path.join(out_dir, "startup", "fed_client.json"))
        assert os.path.isfile(os.path.join(out_dir, "startup", "start.sh"))

    def test_server_kit_type_from_cert(self, cert_env, tmp_path):
        """Cert with UNSTRUCTURED_NAME=server → server kit assembled without -t."""
        work = tmp_path / "work"
        work.mkdir()
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        rootca = cert_env["rootca_path"]

        key_path, cert_path = _make_signed_cert(ca_key, ca_cert, "fl-server", str(work), "fl-server.crt", role="server")
        ws = str(tmp_path / "ws")
        args = _make_args(
            kit_type=None,
            name="fl-server",
            endpoint="grpc://fl-server:8002",
            cert=cert_path,
            key=key_path,
            rootca=rootca,
            workspace=ws,
        )
        handle_package(args)

        out_dir = _kit_dir(ws, "testproject", "fl-server")
        assert os.path.isfile(os.path.join(out_dir, "startup", "fed_server.json"))
        assert os.path.isfile(os.path.join(out_dir, "startup", "start.sh"))

    def test_admin_kit_type_from_cert(self, cert_env, tmp_path):
        """Cert with UNSTRUCTURED_NAME=lead → admin kit assembled without -t."""
        work = tmp_path / "work"
        work.mkdir()
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        rootca = cert_env["rootca_path"]

        key_path, cert_path = _make_signed_cert(
            ca_key, ca_cert, "alice@myorg.com", str(work), "alice@myorg.com.crt", role="lead"
        )
        ws = str(tmp_path / "ws")
        args = _make_args(
            kit_type=None,
            name="alice@myorg.com",
            endpoint="grpc://fl-server:8002",
            cert=cert_path,
            key=key_path,
            rootca=rootca,
            workspace=ws,
        )
        handle_package(args)

        out_dir = _kit_dir(ws, "testproject", "alice@myorg.com")
        assert os.path.isfile(os.path.join(out_dir, "startup", "fed_admin.json"))
        assert os.path.isfile(os.path.join(out_dir, "startup", "fl_admin.sh"))

    def test_explicit_t_overrides_cert_role(self, cert_env, tmp_path):
        """Explicit -t overrides what is in the cert's UNSTRUCTURED_NAME."""
        work = tmp_path / "work"
        work.mkdir()
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        rootca = cert_env["rootca_path"]

        # Cert says 'member' but we override with explicit kit_type='lead'
        key_path, cert_path = _make_signed_cert(
            ca_key, ca_cert, "alice@myorg.com", str(work), "alice@myorg.com.crt", role="member"
        )
        ws = str(tmp_path / "ws")
        args = _make_args(
            kit_type="lead",  # explicit override
            name="alice@myorg.com",
            endpoint="grpc://fl-server:8002",
            cert=cert_path,
            key=key_path,
            rootca=rootca,
            workspace=ws,
        )
        handle_package(args)

        out_dir = _kit_dir(ws, "testproject", "alice@myorg.com")
        assert os.path.isfile(os.path.join(out_dir, "startup", "fed_admin.json"))

    def test_dir_mode_kit_type_from_cert(self, cert_env, tmp_path):
        """--dir mode: kit type derived from cert when -t not given."""
        work = tmp_path / "work"
        work.mkdir()
        ca_key = cert_env["ca_key"]
        ca_cert = cert_env["ca_cert"]
        rootca_src = cert_env["rootca_path"]

        _make_signed_cert(ca_key, ca_cert, "hospital-1", str(work), "hospital-1.crt", role="client")
        shutil.copy2(rootca_src, str(work / "rootCA.pem"))

        ws = str(tmp_path / "ws")
        args = _make_args(
            kit_type=None,
            endpoint="grpc://fl-server:8002",
            dir=str(work),
            workspace=ws,
        )
        handle_package(args)

        out_dir = _kit_dir(ws, "testproject", "hospital-1")
        assert os.path.isfile(os.path.join(out_dir, "startup", "fed_client.json"))


# ---------------------------------------------------------------------------
# Builder pipeline assembly tests (yaml mode)
# ---------------------------------------------------------------------------


def _capture_builders(cert_env, tmp_path, yaml_builders=None):
    """Run handle_package in YAML mode and return the builders list passed to Provisioner.

    ``yaml_builders`` is the optional list of builder dicts to embed in the project YAML.
    """
    ca_key = cert_env["ca_key"]
    ca_cert = cert_env["ca_cert"]
    cert_dir = tmp_path / "certs"
    cert_dir.mkdir()
    shutil.copy2(cert_env["rootca_path"], str(cert_dir / "rootCA.pem"))
    _make_signed_cert(ca_key, ca_cert, "hospital-1", str(cert_dir), "hospital-1.crt", role="client")

    project_yaml = tmp_path / "project.yaml"
    _write_project_yaml(
        project_yaml,
        [{"name": "hospital-1", "type": "client"}],
        builders=yaml_builders,
    )

    captured = []
    original_init = Provisioner.__init__

    def _capturing_init(self, root_dir, builders):
        captured.extend(builders)
        original_init(self, root_dir, builders)

    with unittest.mock.patch.object(Provisioner, "__init__", _capturing_init):
        args = _make_args(
            kit_type=None,
            endpoint="grpc://fl-server:8002",
            dir=str(cert_dir),
            workspace=str(tmp_path / "ws"),
            project_name="myproject",
            project_file=str(project_yaml),
        )
        handle_package(args)

    return captured


class TestYamlModeBuilderPipeline:
    """Verify that the builder pipeline in yaml mode honors YAML-declared builders."""

    @pytest.fixture
    def cert_env(self, tmp_path):
        ca_key, ca_cert, rootca_path = _make_ca(str(tmp_path))
        return {"ca_key": ca_key, "ca_cert": ca_cert, "rootca_path": rootca_path}

    def test_no_yaml_builders_injects_defaults(self, cert_env, tmp_path):
        """No builders in YAML → WorkspaceBuilder, PrebuiltCertBuilder, StaticFileBuilder injected."""
        builders = _capture_builders(cert_env, tmp_path, yaml_builders=None)
        types_ = [type(b) for b in builders]
        assert WorkspaceBuilder in types_
        assert PrebuiltCertBuilder in types_
        assert StaticFileBuilder in types_

    def test_yaml_workspace_builder_not_duplicated(self, cert_env, tmp_path):
        """WorkspaceBuilder in YAML is used; no second default WorkspaceBuilder added."""
        builders = _capture_builders(
            cert_env,
            tmp_path,
            yaml_builders=[{"path": "nvflare.lighter.impl.workspace.WorkspaceBuilder"}],
        )
        assert sum(isinstance(b, WorkspaceBuilder) for b in builders) == 1

    def test_yaml_static_file_builder_not_duplicated(self, cert_env, tmp_path):
        """StaticFileBuilder in YAML is used; no second default StaticFileBuilder added."""
        builders = _capture_builders(
            cert_env,
            tmp_path,
            yaml_builders=[
                {"path": "nvflare.lighter.impl.workspace.WorkspaceBuilder"},
                {"path": "nvflare.lighter.impl.static_file.StaticFileBuilder", "args": {"scheme": "grpc"}},
            ],
        )
        assert sum(isinstance(b, StaticFileBuilder) for b in builders) == 1

    def test_yaml_cert_builder_replaced_by_prebuilt(self, cert_env, tmp_path):
        """CertBuilder in YAML is replaced by PrebuiltCertBuilder; no CertBuilder remains."""
        builders = _capture_builders(
            cert_env,
            tmp_path,
            yaml_builders=[
                {"path": "nvflare.lighter.impl.workspace.WorkspaceBuilder"},
                {"path": "nvflare.lighter.impl.cert.CertBuilder"},
                {"path": "nvflare.lighter.impl.static_file.StaticFileBuilder", "args": {"scheme": "grpc"}},
            ],
        )
        assert not any(isinstance(b, CertBuilder) for b in builders), "CertBuilder must be replaced"
        assert any(isinstance(b, PrebuiltCertBuilder) for b in builders)

    def test_yaml_static_file_builder_custom_scheme_preserved(self, cert_env, tmp_path):
        """StaticFileBuilder from YAML keeps its scheme arg, not overridden by --endpoint scheme."""
        builders = _capture_builders(
            cert_env,
            tmp_path,
            yaml_builders=[
                {"path": "nvflare.lighter.impl.workspace.WorkspaceBuilder"},
                {"path": "nvflare.lighter.impl.static_file.StaticFileBuilder", "args": {"scheme": "tcp"}},
            ],
        )
        static = next(b for b in builders if isinstance(b, StaticFileBuilder))
        assert static.scheme == "tcp"

    def test_prebuilt_cert_builder_positioned_after_workspace(self, cert_env, tmp_path):
        """PrebuiltCertBuilder comes after WorkspaceBuilder in the assembled list."""
        builders = _capture_builders(cert_env, tmp_path, yaml_builders=None)
        types_ = [type(b) for b in builders]
        ws_idx = types_.index(WorkspaceBuilder)
        cert_idx = types_.index(PrebuiltCertBuilder)
        assert ws_idx < cert_idx


# ---------------------------------------------------------------------------
# Distributed provisioning signed-zip package mode
# ---------------------------------------------------------------------------


def _signed_zip_sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _signed_zip_sha256_file(path) -> str:
    with open(path, "rb") as f:
        return _signed_zip_sha256_bytes(f.read())


def _public_key_sha256_from_cert(cert_path) -> str:
    cert = load_crt(str(cert_path))
    public_key_der = cert.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return _signed_zip_sha256_bytes(public_key_der)


def _make_signed_zip(
    tmp_path,
    *,
    name="site-3",
    org="nvidia",
    kind="site",
    cert_type="client",
    project_name="example_project",
    request_id="11111111111111111111111111111111",
    ca_project=None,
    cert_org=None,
    include_key=False,
    traversal=False,
    cert_member=None,
    key_member=None,
    hash_mismatch=False,
):
    request_dir = tmp_path / name
    request_dir.mkdir(exist_ok=True)
    ca_dir = tmp_path / "ca"
    ca_dir.mkdir(exist_ok=True)
    ca_key, ca_cert, rootca_path = _make_ca(str(ca_dir), name=ca_project or project_name)
    key_path, cert_path = _make_signed_cert(
        ca_key, ca_cert, name, str(request_dir), f"{name}.crt", role=cert_type, org=cert_org or org
    )
    rootca_copy = request_dir / "rootCA.pem"
    shutil.copy2(rootca_path, str(rootca_copy))
    site_yaml_path = request_dir / "site.yaml"
    request_json_path = request_dir / "request.json"
    signed_json_path = request_dir / "signed.json"
    site_yaml_path.write_text(f"name: {name}\norg: {org}\nkind: {kind}\ntype: {cert_type}\nproject: {project_name}\n")
    cert_sha256 = "0" * 64 if hash_mismatch else _signed_zip_sha256_file(cert_path)
    public_key_sha256 = _public_key_sha256_from_cert(cert_path)
    request_json_path.write_text(
        json.dumps(
            {
                "artifact_type": "nvflare.cert.request",
                "schema_version": "1",
                "request_id": request_id,
                "created_at": "2026-04-24T00:00:00Z",
                "project": project_name,
                "name": name,
                "org": org,
                "kind": kind,
                "cert_type": cert_type,
                "cert_role": None,
                "csr_sha256": "1" * 64,
                "public_key_sha256": public_key_sha256,
            },
            sort_keys=True,
        )
    )
    signed_json_path.write_text(
        json.dumps(
            {
                "artifact_type": "nvflare.cert.signed",
                "schema_version": "1",
                "request_id": request_id,
                "approved_at": "2026-04-24T00:00:00Z",
                "project": project_name,
                "name": name,
                "org": org,
                "kind": kind,
                "cert_type": cert_type,
                "cert_role": None,
                "certificate": {
                    "serial": "01",
                    "valid_until": "2029-04-24T00:00:00Z",
                },
                "cert_file": f"{name}.crt",
                "rootca_file": "rootCA.pem",
                "hashes": {
                    "csr_sha256": "1" * 64,
                    "site_yaml_sha256": _signed_zip_sha256_file(site_yaml_path),
                    "certificate_sha256": cert_sha256,
                    "rootca_sha256": _signed_zip_sha256_file(rootca_copy),
                    "public_key_sha256": public_key_sha256,
                },
            },
            sort_keys=True,
        )
    )
    signed_zip = request_dir / f"{name}.signed.zip"
    with zipfile.ZipFile(signed_zip, "w") as zf:
        zf.write(signed_json_path, "signed.json")
        zf.write(site_yaml_path, "site.yaml")
        with open(cert_path, "rb") as cert_file:
            zf.writestr(cert_member or (f"../{name}.crt" if traversal else f"{name}.crt"), cert_file.read())
        zf.write(rootca_copy, "rootCA.pem")
        if include_key:
            zf.write(key_path, key_member or f"{name}.key")
    os.remove(cert_path)
    os.remove(rootca_copy)
    os.remove(signed_json_path)
    return signed_zip, request_dir, key_path


def _rewrite_signed_zip_metadata(signed_zip, mutate):
    with zipfile.ZipFile(signed_zip, "r") as zf:
        members = {name: zf.read(name) for name in zf.namelist()}
    metadata = json.loads(members["signed.json"])
    mutate(metadata)
    members["signed.json"] = json.dumps(metadata, sort_keys=True).encode("utf-8")
    with zipfile.ZipFile(signed_zip, "w") as zf:
        for name, content in members.items():
            zf.writestr(name, content)


def _signed_zip_args(signed_zip, tmp_path, **kwargs):
    defaults = dict(
        input=str(signed_zip),
        endpoint="grpc://fl-server:8002",
        workspace=str(tmp_path / "ws"),
        force=True,
        project_file=None,
        request_dir=None,
        project_name=None,
        dir=None,
        cert=None,
        key=None,
        rootca=None,
        name=None,
        kit_type=None,
    )
    defaults.update(kwargs)
    return _make_args(**defaults)


def _participant_names(project):
    return sorted(
        p.name for p in project.get_all_participants() if p.name != "fl-server" and p.name != _DUMMY_SERVER_NAME
    )


def _run_signed_zip_with_captured_provisioner(args, prod_dir):
    captured = {}

    def _capture_init(self, root_dir, builders):
        captured["root_dir"] = root_dir
        captured["builders"] = builders

    def _capture_provision(self, project):
        captured["project"] = project
        os.makedirs(prod_dir, exist_ok=True)
        return {CtxKey.CURRENT_PROD_DIR: str(prod_dir)}

    with unittest.mock.patch.object(Provisioner, "__init__", _capture_init):
        with unittest.mock.patch.object(Provisioner, "provision", _capture_provision):
            handle_package(args)
    return captured


class TestSignedZipPackagePublicSurface:
    def test_package_help_and_schema_lead_with_signed_zip_mode(self, capsys):
        import argparse

        from nvflare.tool.cli_schema import handle_schema_flag
        from nvflare.tool.package.package_cli import _PACKAGE_EXAMPLES, def_package_cli_parser

        root = argparse.ArgumentParser(prog="nvflare")
        subs = root.add_subparsers()
        parser = def_package_cli_parser(subs)["package"]

        help_text = parser.format_help()
        assert ".signed.zip" in help_text
        assert "nvflare package" in help_text
        assert "Custom builders are honored" in help_text
        assert "are ignored" not in help_text
        assert "--cert ./signed/hospital-1/hospital-1.crt" not in help_text
        assert "--key ./csr/hospital-1.key" not in help_text

        with pytest.raises(SystemExit) as exc_info:
            handle_schema_flag(parser, "nvflare package", _PACKAGE_EXAMPLES, ["--schema"])
        assert exc_info.value.code == 0
        schema_text = capsys.readouterr().out
        assert ".signed.zip" in schema_text
        assert "--cert" not in schema_text
        assert "--rootca" not in schema_text


class TestSignedZipPackageMode:
    def test_signed_zip_finds_sibling_private_key_and_uses_default_project_model(self, tmp_path):
        signed_zip, request_dir, _ = _make_signed_zip(tmp_path)
        sibling_signed_zip = tmp_path / f"{request_dir.name}.signed.zip"
        shutil.move(str(signed_zip), str(sibling_signed_zip))
        args = _signed_zip_args(sibling_signed_zip, tmp_path)

        handle_package(args)

        kit_dir = os.path.join(str(tmp_path / "ws"), "example_project", "prod_00", "site-3")
        assert os.path.isfile(os.path.join(kit_dir, "startup", "client.key"))
        assert os.path.isfile(os.path.join(kit_dir, "startup", "client.crt"))
        assert os.path.isfile(os.path.join(kit_dir, "startup", "rootCA.pem"))
        assert not os.path.exists(os.path.join(str(tmp_path / "ws"), "example_project", "prod_00", "fl-server"))

    def test_signed_zip_input_suffix_is_case_insensitive(self, tmp_path):
        signed_zip, request_dir, _ = _make_signed_zip(tmp_path)
        uppercase_signed_zip = request_dir / "site-3.SIGNED.ZIP"
        shutil.move(str(signed_zip), str(uppercase_signed_zip))
        args = _signed_zip_args(uppercase_signed_zip, tmp_path, request_dir=str(request_dir))

        handle_package(args)

        kit_dir = os.path.join(str(tmp_path / "ws"), "example_project", "prod_00", "site-3")
        assert os.path.isfile(os.path.join(kit_dir, "startup", "client.key"))

    def test_signed_zip_ignores_stale_audit_request_dir_and_uses_sibling_request_folder(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        signed_zip, request_dir, _ = _make_signed_zip(tmp_path)
        sibling_signed_zip = tmp_path / f"{request_dir.name}.signed.zip"
        shutil.move(str(signed_zip), str(sibling_signed_zip))
        request_id = "11111111111111111111111111111111"
        stale_audit_dir = tmp_path / "home" / ".nvflare" / "cert_requests" / request_id
        stale_audit_dir.mkdir(parents=True)
        (stale_audit_dir / "audit.json").write_text(
            json.dumps({"schema_version": "1", "request_dir": str(tmp_path / "missing-request-dir")})
        )
        args = _signed_zip_args(sibling_signed_zip, tmp_path)

        handle_package(args)

        kit_dir = os.path.join(str(tmp_path / "ws"), "example_project", "prod_00", "site-3")
        assert os.path.isfile(os.path.join(kit_dir, "startup", "client.key"))

    def test_signed_zip_ignores_audit_request_dir_with_wrong_request_id(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        signed_zip, request_dir, _ = _make_signed_zip(tmp_path)
        sibling_signed_zip = tmp_path / f"{request_dir.name}.signed.zip"
        shutil.move(str(signed_zip), str(sibling_signed_zip))
        tampered_request_dir = tmp_path / "tampered-site-3"
        tampered_request_dir.mkdir()
        other_key, _other_pub = generate_keys()
        _write_file_nofollow(tampered_request_dir / "site-3.key", serialize_pri_key(other_key), mode=0o600)
        (tampered_request_dir / "request.json").write_text(
            json.dumps(
                {
                    "request_id": "22222222222222222222222222222222",
                    "project": "example_project",
                    "name": "site-3",
                    "org": "nvidia",
                    "kind": "site",
                    "cert_type": "client",
                    "csr_sha256": "1" * 64,
                    "public_key_sha256": "1" * 64,
                }
            )
        )
        audit_dir = tmp_path / "home" / ".nvflare" / "cert_requests" / "11111111111111111111111111111111"
        audit_dir.mkdir(parents=True)
        (audit_dir / "audit.json").write_text(
            json.dumps({"schema_version": "1", "request_dir": str(tampered_request_dir)})
        )
        args = _signed_zip_args(sibling_signed_zip, tmp_path)

        handle_package(args)

        kit_dir = os.path.join(str(tmp_path / "ws"), "example_project", "prod_00", "site-3")
        assert os.path.isfile(os.path.join(kit_dir, "startup", "client.key"))

    @pytest.mark.skipif(
        not hasattr(os, "symlink") or not hasattr(os, "O_NOFOLLOW"), reason="nofollow symlink support required"
    )
    def test_signed_zip_ignores_symlinked_audit_record(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        signed_zip, request_dir, _ = _make_signed_zip(tmp_path)
        sibling_signed_zip = tmp_path / f"{request_dir.name}.signed.zip"
        shutil.move(str(signed_zip), str(sibling_signed_zip))
        audit_dir = tmp_path / "home" / ".nvflare" / "cert_requests" / "11111111111111111111111111111111"
        audit_dir.mkdir(parents=True)
        real_audit = tmp_path / "audit.json"
        real_audit.write_text(json.dumps({"schema_version": "1", "request_dir": str(tmp_path / "missing")}))
        os.symlink(str(real_audit), str(audit_dir / "audit.json"))
        args = _signed_zip_args(sibling_signed_zip, tmp_path)

        handle_package(args)

        kit_dir = os.path.join(str(tmp_path / "ws"), "example_project", "prod_00", "site-3")
        assert os.path.isfile(os.path.join(kit_dir, "startup", "client.key"))

    def test_signed_zip_skips_incomplete_candidate_request_dir(self, tmp_path):
        signed_zip, request_dir, _ = _make_signed_zip(tmp_path)
        signed_container = tmp_path / "signed"
        signed_container.mkdir()
        sibling_signed_zip = signed_container / signed_zip.name
        shutil.move(str(signed_zip), str(sibling_signed_zip))
        (signed_container / "site-3.key").write_text("stray incomplete key")

        args = _signed_zip_args(sibling_signed_zip, tmp_path)

        handle_package(args)

        kit_dir = os.path.join(str(tmp_path / "ws"), "example_project", "prod_00", "site-3")
        assert os.path.isfile(os.path.join(kit_dir, "startup", "client.key"))
        assert request_dir == tmp_path / "site-3"

    def test_signed_zip_without_discoverable_request_dir_returns_targeted_error(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        signed_zip, _request_dir, _ = _make_signed_zip(tmp_path)
        signed_container = tmp_path / "elsewhere" / "signed"
        signed_container.mkdir(parents=True)
        remote_signed_zip = signed_container / signed_zip.name
        shutil.move(str(signed_zip), str(remote_signed_zip))
        args = _signed_zip_args(remote_signed_zip, tmp_path)

        with pytest.raises(SystemExit) as exc_info:
            handle_package(args)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "REQUEST_DIR_NOT_FOUND" in captured.err
        assert "--request-dir" in captured.err

    @pytest.mark.parametrize(
        "zip_kwargs",
        [
            {"include_key": True},
            {"traversal": True},
            {"cert_member": "./site-3.crt"},
            {"cert_member": "site//site-3.crt"},
            {"include_key": True, "key_member": "site-3.KEY"},
            {"hash_mismatch": True},
        ],
    )
    def test_signed_zip_rejects_unsafe_or_tampered_contents(self, tmp_path, capsys, monkeypatch, zip_kwargs):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        signed_zip, _, _ = _make_signed_zip(tmp_path, **zip_kwargs)
        args = _signed_zip_args(signed_zip, tmp_path)

        with pytest.raises(SystemExit) as exc_info:
            handle_package(args)

        assert exc_info.value.code != 0
        capsys.readouterr()

    @pytest.mark.parametrize("large_member", ["signed.json", "site-3.crt"])
    def test_signed_zip_rejects_member_exceeding_read_limit(self, tmp_path, capsys, monkeypatch, large_member):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        signed_zip, request_dir, _ = _make_signed_zip(tmp_path)
        with zipfile.ZipFile(signed_zip, "r") as zf:
            contents = {name: zf.read(name) for name in zf.namelist()}
        contents[large_member] = b"x" * 513
        with zipfile.ZipFile(signed_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for name, content in contents.items():
                zf.writestr(name, content)
        monkeypatch.setattr("nvflare.tool.package.package_commands._MAX_ZIP_MEMBER_SIZE", 512)
        monkeypatch.setattr(
            "nvflare.tool.package.package_commands._safe_zip_names",
            lambda _zf, _zip_path: ["signed.json", "site.yaml", "rootCA.pem", "site-3.crt"],
        )
        args = _signed_zip_args(signed_zip, tmp_path, request_dir=str(request_dir))

        with pytest.raises(SystemExit) as exc_info:
            handle_package(args)

        assert exc_info.value.code == 4
        assert "Signed zip member exceeds size limit" in capsys.readouterr().err

    def test_signed_zip_missing_private_key_does_not_materialize_files(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        signed_zip, request_dir, key_path = _make_signed_zip(tmp_path)
        os.remove(key_path)
        existing_site_yaml = request_dir / "site.yaml"
        existing_site_yaml.write_text("existing local request site.yaml")
        args = _signed_zip_args(signed_zip, tmp_path, request_dir=str(request_dir))

        with pytest.raises(SystemExit) as exc_info:
            handle_package(args)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "KEY_NOT_FOUND" in captured.err
        assert existing_site_yaml.read_text() == "existing local request site.yaml"
        for name in ("signed.json", "site-3.crt", "rootCA.pem"):
            assert not (request_dir / name).exists()

    @pytest.mark.parametrize(
        "field,value",
        [
            ("artifact_type", "nvflare.cert.request"),
            ("schema_version", "2"),
        ],
    )
    def test_signed_zip_rejects_unsupported_metadata_schema(self, tmp_path, capsys, monkeypatch, field, value):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        signed_zip, request_dir, _ = _make_signed_zip(tmp_path)

        def _mutate(metadata):
            metadata[field] = value

        _rewrite_signed_zip_metadata(signed_zip, _mutate)
        args = _signed_zip_args(signed_zip, tmp_path, request_dir=str(request_dir))

        with pytest.raises(SystemExit) as exc_info:
            handle_package(args)

        assert exc_info.value.code == 4
        assert "unsupported artifact type or schema version" in capsys.readouterr().err

    @pytest.mark.parametrize("hash_name", ["csr_sha256", "certificate_sha256"])
    def test_signed_zip_rejects_missing_required_hash(self, tmp_path, capsys, monkeypatch, hash_name):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        signed_zip, request_dir, _ = _make_signed_zip(tmp_path)

        def _mutate(metadata):
            del metadata["hashes"][hash_name]

        _rewrite_signed_zip_metadata(signed_zip, _mutate)
        args = _signed_zip_args(signed_zip, tmp_path, request_dir=str(request_dir))

        with pytest.raises(SystemExit) as exc_info:
            handle_package(args)

        assert exc_info.value.code == 4
        assert "missing required hashes" in capsys.readouterr().err

    def test_signed_public_key_hash_helper_requires_hash(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        ca_dir = tmp_path / "ca"
        ca_dir.mkdir()
        ca_key, ca_cert, _ = _make_ca(str(ca_dir))
        _key_path, cert_path = _make_signed_cert(ca_key, ca_cert, "site-3", str(tmp_path), "site-3.crt", role="client")
        cert = load_crt(cert_path)

        with pytest.raises(SystemExit) as exc_info:
            _validate_signed_public_key_hash({}, cert)

        assert exc_info.value.code == 4
        assert "Missing required public key hash" in capsys.readouterr().err

    @pytest.mark.parametrize(
        "field,value,expected",
        [
            ("project", 123, "Invalid project name"),
            ("name", 123, "Invalid participant name"),
            ("org", 123, "Invalid org name"),
        ],
    )
    def test_signed_zip_rejects_non_string_identity_metadata(
        self, tmp_path, capsys, monkeypatch, field, value, expected
    ):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        signed_zip, request_dir, _ = _make_signed_zip(tmp_path)

        def _mutate(metadata):
            metadata[field] = value

        _rewrite_signed_zip_metadata(signed_zip, _mutate)
        args = _signed_zip_args(signed_zip, tmp_path, request_dir=str(request_dir))

        with pytest.raises(SystemExit) as exc_info:
            handle_package(args)

        assert exc_info.value.code == 4
        assert expected in capsys.readouterr().err

    @pytest.mark.parametrize(
        "zip_kwargs, expected",
        [
            ({"project_name": "../escape"}, "Invalid project name"),
            ({"request_id": "../escape"}, "Invalid request_id"),
            ({"org": "bad-org"}, "Invalid org name"),
            ({"kind": "workspace"}, "Invalid signed zip kind/cert_type combination"),
            ({"cert_org": "other_org"}, "conflicts with certificate org"),
            ({"ca_project": "other_project"}, "conflicts with root CA project"),
        ],
    )
    def test_signed_zip_rejects_identity_metadata_conflicts(self, tmp_path, capsys, monkeypatch, zip_kwargs, expected):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        signed_zip, request_dir, _ = _make_signed_zip(tmp_path, **zip_kwargs)
        args = _signed_zip_args(signed_zip, tmp_path, request_dir=str(request_dir))

        with pytest.raises(SystemExit) as exc_info:
            handle_package(args)

        assert exc_info.value.code == 4
        assert expected in capsys.readouterr().err

    def test_signed_zip_key_mismatch_does_not_materialize_files(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        signed_zip, request_dir, key_path = _make_signed_zip(tmp_path)
        other_key, _other_pub = generate_keys()
        with open(key_path, "wb") as f:
            f.write(serialize_pri_key(other_key))
        existing_site_yaml = request_dir / "site.yaml"
        existing_site_yaml.write_text("existing local request site.yaml")
        args = _signed_zip_args(signed_zip, tmp_path, request_dir=str(request_dir))

        with pytest.raises(SystemExit) as exc_info:
            handle_package(args)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "KEY_CERT_MISMATCH" in captured.err
        assert existing_site_yaml.read_text() == "existing local request site.yaml"
        for name in ("signed.json", "site-3.crt", "rootCA.pem"):
            assert not (request_dir / name).exists()

    @pytest.mark.skipif(
        not hasattr(os, "symlink") or not hasattr(os, "O_NOFOLLOW"), reason="nofollow symlink support required"
    )
    def test_signed_zip_rejects_symlinked_local_private_key(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        signed_zip, request_dir, key_path = _make_signed_zip(tmp_path)
        real_key_path = request_dir / "real-site-3.key"
        os.rename(key_path, real_key_path)
        os.symlink(str(real_key_path), key_path)
        existing_site_yaml = request_dir / "site.yaml"
        existing_site_yaml.write_text("existing local request site.yaml")
        args = _signed_zip_args(signed_zip, tmp_path, request_dir=str(request_dir))

        with pytest.raises(SystemExit) as exc_info:
            handle_package(args)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "KEY_INVALID" in captured.err
        assert existing_site_yaml.read_text() == "existing local request site.yaml"
        for name in ("signed.json", "site-3.crt", "rootCA.pem"):
            assert not (request_dir / name).exists()

    def test_validate_cert_material_invalid_key_does_not_fall_through_when_error_is_mocked(self, tmp_path, monkeypatch):
        ca_dir = tmp_path / "ca"
        ca_dir.mkdir()
        ca_key, ca_cert, rootca_path = _make_ca(str(ca_dir))
        key_path, cert_path = _make_signed_cert(ca_key, ca_cert, "site-3", str(tmp_path), "site-3.crt", role="client")
        with open(key_path, "wb") as f:
            f.write(b"not a private key")
        error = unittest.mock.Mock()
        monkeypatch.setattr("nvflare.tool.package.package_commands.output_error_message", error)

        cert = _validate_cert_material(cert_path, key_path, rootca_path, validate_key_match=True)

        assert cert is None
        error.assert_called_once()
        assert error.call_args.args[0] == "KEY_INVALID"

    def test_signed_zip_rejects_local_request_metadata_mismatch(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        signed_zip, request_dir, _ = _make_signed_zip(tmp_path)
        request_json_path = request_dir / "request.json"
        request_json = json.loads(request_json_path.read_text())
        request_json["request_id"] = "2" * 32
        request_json_path.write_text(json.dumps(request_json, sort_keys=True))
        args = _signed_zip_args(signed_zip, tmp_path, request_dir=str(request_dir))

        with pytest.raises(SystemExit) as exc_info:
            handle_package(args)

        assert exc_info.value.code == 4
        captured = capsys.readouterr()
        assert "REQUEST_DIR_MISMATCH" in captured.err
        for name in ("signed.json", "site-3.crt", "rootCA.pem"):
            assert not (request_dir / name).exists()

    def test_signed_zip_read_error_returns_cli_error(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        signed_zip, request_dir, _ = _make_signed_zip(tmp_path)
        args = _signed_zip_args(signed_zip, tmp_path, request_dir=str(request_dir))

        with unittest.mock.patch(
            "nvflare.tool.package.package_commands.zipfile.ZipFile",
            side_effect=PermissionError("blocked"),
        ):
            with pytest.raises(SystemExit) as exc_info:
                handle_package(args)

        assert exc_info.value.code == 4
        captured = capsys.readouterr()
        assert "INVALID_SIGNED_ZIP" in captured.err
        assert "blocked" in captured.err

    def test_single_site_project_file_uses_project_field(self, tmp_path):
        signed_zip, request_dir, _ = _make_signed_zip(tmp_path)
        site_yaml = tmp_path / "site.yaml"
        site_yaml.write_text("name: site-3\norg: nvidia\ntype: client\nproject: example_project\n")
        args = _signed_zip_args(signed_zip, tmp_path, project_file=str(site_yaml), request_dir=str(request_dir))

        captured = _run_signed_zip_with_captured_provisioner(args, tmp_path / "ws" / "example_project" / "prod_00")

        assert captured["project"].name == "example_project"
        assert _participant_names(captured["project"]) == ["site-3"]

    def test_project_file_custom_builders_apply_to_signed_identity_only_and_endpoint_scheme_wins(self, tmp_path):
        signed_zip, request_dir, _ = _make_signed_zip(tmp_path)
        project_yaml = tmp_path / "project.yaml"
        project_yaml.write_text(
            "api_version: 3\n"
            "name: example_project\n"
            'description: ""\n'
            "participants:\n"
            "  - name: site-3\n"
            "    type: client\n"
            "    org: nvidia\n"
            "    listening_host:\n"
            "      scheme: grpc\n"
            "      default_host: site-3.local\n"
            "      port: 9001\n"
            "  - name: site-4\n"
            "    type: client\n"
            "    org: nvidia\n"
            "builders:\n"
            "  - path: nvflare.lighter.impl.workspace.WorkspaceBuilder\n"
            "  - path: nvflare.lighter.impl.static_file.StaticFileBuilder\n"
            "    args:\n"
            "      scheme: tcp\n"
            "  - path: nvflare.lighter.impl.signature.SignatureBuilder\n"
        )
        args = _signed_zip_args(signed_zip, tmp_path, project_file=str(project_yaml), request_dir=str(request_dir))

        captured = _run_signed_zip_with_captured_provisioner(args, tmp_path / "ws" / "example_project" / "prod_00")

        assert _participant_names(captured["project"]) == ["site-3"]
        participant = captured["project"].get_clients()[0]
        assert participant.org == "nvidia"
        assert participant.props["listening_host"]["default_host"] == "site-3.local"
        assert any(isinstance(b, StaticFileBuilder) and b.scheme == "grpc" for b in captured["builders"])
        assert not any(isinstance(b, SignatureBuilder) for b in captured["builders"])

    def test_project_file_missing_signed_participant_warns_and_builds_signed_identity(
        self, tmp_path, capsys, monkeypatch
    ):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        signed_zip, request_dir, _ = _make_signed_zip(tmp_path)
        project_yaml = tmp_path / "project.yaml"
        _write_project_yaml(project_yaml, [{"name": "site-4", "type": "client", "org": "nvidia"}], "example_project")
        args = _signed_zip_args(signed_zip, tmp_path, project_file=str(project_yaml), request_dir=str(request_dir))

        captured = _run_signed_zip_with_captured_provisioner(args, tmp_path / "ws" / "example_project" / "prod_00")

        assert _participant_names(captured["project"]) == ["site-3"]
        captured_output = capsys.readouterr()
        combined = captured_output.out + captured_output.err
        assert "Warning" in combined
        assert "site-3 was not found" in combined

    def test_project_file_identity_conflict_fails_before_build(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        signed_zip, request_dir, _ = _make_signed_zip(tmp_path)
        project_yaml = tmp_path / "project.yaml"
        _write_project_yaml(project_yaml, [{"name": "site-3", "type": "client", "org": "other-org"}], "example_project")
        args = _signed_zip_args(signed_zip, tmp_path, project_file=str(project_yaml), request_dir=str(request_dir))

        with unittest.mock.patch.object(Provisioner, "provision") as provision:
            with pytest.raises(SystemExit) as exc_info:
                handle_package(args)

        assert exc_info.value.code != 0
        provision.assert_not_called()

    def test_signed_zip_builds_one_participant_when_project_file_lists_many(self, tmp_path):
        signed_zip, request_dir, _ = _make_signed_zip(tmp_path)
        project_yaml = tmp_path / "project.yaml"
        _write_project_yaml(
            project_yaml,
            [
                {"name": "site-3", "type": "client", "org": "nvidia"},
                {"name": "site-4", "type": "client", "org": "nvidia"},
                {"name": "alice@nvidia.com", "type": "admin", "role": "lead", "org": "nvidia"},
            ],
            "example_project",
        )
        args = _signed_zip_args(signed_zip, tmp_path, project_file=str(project_yaml), request_dir=str(request_dir))

        captured = _run_signed_zip_with_captured_provisioner(args, tmp_path / "ws" / "example_project" / "prod_00")

        assert _participant_names(captured["project"]) == ["site-3"]
