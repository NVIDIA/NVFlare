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

import copy
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
import yaml
from cryptography.hazmat.primitives import serialization

from nvflare.lighter.constants import CtxKey, PropKey
from nvflare.lighter.entity import Project
from nvflare.lighter.impl.cert import CertBuilder
from nvflare.lighter.impl.signature import SignatureBuilder
from nvflare.lighter.impl.static_file import StaticFileBuilder
from nvflare.lighter.impl.workspace import WorkspaceBuilder
from nvflare.lighter.provisioner import Provisioner
from nvflare.lighter.utils import (
    Identity,
    generate_cert,
    generate_keys,
    load_crt,
    load_crt_bytes,
    serialize_cert,
    serialize_pri_key,
    sign_content,
)
from nvflare.tool import cli_output
from nvflare.tool.cert.cert_constants import (
    CA_INFO_FIELD,
    DEFAULT_PROVISION_VERSION,
    PROVISION_VERSION_FIELD,
    ROOTCA_FINGERPRINT_FIELD,
)
from nvflare.tool.cert.fingerprint import cert_fingerprint_sha256, normalize_sha256_fingerprint
from nvflare.tool.package.package_commands import (
    _DUMMY_SERVER_NAME,
    FixedProdWorkspaceBuilder,
    PrebuiltCertBuilder,
    _build_package_builders,
    _discover_name_from_dir,
    _flat_site_to_project_dict,
    _load_signed_zip,
    _normalize_hash,
    _parse_endpoint,
    _read_local_request_metadata,
    _read_zip_json,
    _read_zip_member_limited,
    _resolve_request_dir,
    _safe_zip_names,
    _validate_cert_material,
    _validate_local_request_metadata,
    _validate_local_site_identity,
    _validate_safe_project_name,
    _validate_signed_hashes,
    _validate_signed_metadata,
    _validate_signed_public_key_hash,
    _validated_audit_request_dir,
    _write_file_nofollow,
    _write_materialized_signed_files,
    handle_package,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SIGNED_SERVER_ENDPOINT = {
    "host": "server1.hospital-central.org",
    "fed_learn_port": 8002,
    "admin_port": 8003,
}


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
    certs produced by the internal signing helper with an explicit role).
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
        input=None,
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
        request_dir=None,
        expected_fingerprint=None,
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
    def test_validate_safe_project_name_supports_context_label(self):
        with unittest.mock.patch("nvflare.tool.package.package_commands.output_error_message") as error:
            valid = _validate_safe_project_name("../escape", field_label="Signed zip project")

        assert valid is False
        error.assert_called_once()
        assert "signed zip project name" in error.call_args.args[1]

    def test_flat_site_project_dict_uses_identity_project_fallback(self):
        project_dict = _flat_site_to_project_dict(
            {
                "name": "site-3",
                "org": "nvidia",
                "type": "client",
            },
            {
                "project": "example_project",
                "name": "site-3",
                "org": "nvidia",
                "cert_type": "client",
            },
        )

        assert project_dict[PropKey.NAME] == "example_project"
        assert project_dict["participants"][0]["name"] == "site-3"

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
        assert names is None

    def test_safe_zip_names_does_not_append_mode_bit_directory_when_error_is_mocked(self, tmp_path):
        zip_path = tmp_path / "signed.zip"
        dir_info = zipfile.ZipInfo("directory")
        dir_info.external_attr = stat.S_IFDIR << 16
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr(dir_info, b"")
            zf.writestr("signed.json", "{}")

        with zipfile.ZipFile(zip_path, "r") as zf:
            with unittest.mock.patch("nvflare.tool.package.package_commands.output_error_message") as error:
                names = _safe_zip_names(zf, str(zip_path))

        error.assert_called_once()
        assert names is None

    def test_safe_zip_names_rejects_control_characters_when_error_is_mocked(self, tmp_path):
        zip_path = tmp_path / "signed.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("signed.json", "{}")
            zf.writestr("bad\nname", "cert")

        with zipfile.ZipFile(zip_path, "r") as zf:
            with unittest.mock.patch("nvflare.tool.package.package_commands.output_error_message") as error:
                names = _safe_zip_names(zf, str(zip_path))

        error.assert_called_once()
        assert names is None

    def test_safe_zip_names_rejects_private_key_pem_content_when_error_is_mocked(self, tmp_path):
        zip_path = tmp_path / "signed.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("signed.json", "{}")
            zf.writestr("private.pem", "-----BEGIN PRIVATE KEY-----\nsecret\n-----END PRIVATE KEY-----\n")

        with zipfile.ZipFile(zip_path, "r") as zf:
            with unittest.mock.patch("nvflare.tool.package.package_commands.output_error_message") as error:
                names = _safe_zip_names(zf, str(zip_path))

        error.assert_called_once()
        assert "private key" in str(error.call_args)
        assert names is None

    def test_read_zip_json_returns_none_after_error_when_error_is_mocked(self, tmp_path):
        zip_path = tmp_path / "signed.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("signed.json", "not json")

        with zipfile.ZipFile(zip_path, "r") as zf:
            with unittest.mock.patch("nvflare.tool.package.package_commands.output_error_message") as error:
                result = _read_zip_json(zf, "signed.json", str(zip_path))

        assert result is None
        error.assert_called_once()

    def test_validate_signed_metadata_returns_after_non_dict_when_error_is_mocked(self):
        with unittest.mock.patch("nvflare.tool.package.package_commands.output_error_message") as error:
            valid = _validate_signed_metadata(None, {}, "site-3.crt")

        error.assert_called_once()
        assert valid is False
        assert "must be a mapping" in error.call_args.args[1]

    def test_validate_signed_metadata_returns_after_missing_fields_when_error_is_mocked(self):
        with unittest.mock.patch("nvflare.tool.package.package_commands.output_error_message") as error:
            valid = _validate_signed_metadata(
                {"artifact_type": "nvflare.cert.signed", "schema_version": "1"}, {}, "site-3.crt"
            )

        error.assert_called_once()
        assert valid is False
        assert "missing required field" in error.call_args.args[1]

    def test_validate_signed_metadata_returns_after_file_mismatch_when_error_is_mocked(self):
        signed_meta = {
            "artifact_type": "nvflare.cert.signed",
            "schema_version": "1",
            "request_id": "1" * 32,
            "project": "example_project",
            "name": "site-3",
            "org": "nvidia",
            "kind": "site",
            "cert_type": "client",
            "cert_file": "other.crt",
            "rootca_file": "rootCA.pem",
            "scheme": "grpc",
            "default_connection_security": "tls",
            "server": _SIGNED_SERVER_ENDPOINT,
            "hashes": {
                "csr_sha256": "1" * 64,
                "site_yaml_sha256": "1" * 64,
                "certificate_sha256": "1" * 64,
                "rootca_sha256": "1" * 64,
                "public_key_sha256": "1" * 64,
            },
        }
        site_meta = {
            "project": "example_project",
            "name": "site-3",
            "org": "nvidia",
            "kind": "site",
            "type": "client",
        }

        with unittest.mock.patch("nvflare.tool.package.package_commands.output_error_message") as error:
            valid = _validate_signed_metadata(signed_meta, site_meta, "site-3.crt")

        error.assert_called_once()
        assert valid is False
        assert "file names do not match" in error.call_args.args[1]

    def test_validate_signed_metadata_returns_after_invalid_org_when_error_is_mocked(self):
        signed_meta = {
            "artifact_type": "nvflare.cert.signed",
            "schema_version": "1",
            "request_id": "1" * 32,
            "project": "example_project",
            "name": "site-3",
            "org": "bad-org",
            "kind": "site",
            "cert_type": "client",
            "cert_file": "site-3.crt",
            "rootca_file": "rootCA.pem",
            "scheme": "grpc",
            "default_connection_security": "tls",
            "server": _SIGNED_SERVER_ENDPOINT,
            "hashes": {
                "csr_sha256": "1" * 64,
                "site_yaml_sha256": "1" * 64,
                "certificate_sha256": "1" * 64,
                "rootca_sha256": "1" * 64,
                "public_key_sha256": "1" * 64,
            },
        }

        with unittest.mock.patch("nvflare.tool.package.package_commands.output_error_message") as error:
            valid = _validate_signed_metadata(signed_meta, {}, "site-3.crt")

        error.assert_called_once()
        assert valid is False
        assert "Invalid org name" in error.call_args.args[1]

    def test_validate_signed_metadata_returns_after_invalid_cert_type_when_error_is_mocked(self):
        signed_meta = {
            "artifact_type": "nvflare.cert.signed",
            "schema_version": "1",
            "request_id": "1" * 32,
            "project": "example_project",
            "name": "site-3",
            "org": "nvidia",
            "kind": "site",
            "cert_type": "workspace",
            "cert_file": "site-3.crt",
            "rootca_file": "rootCA.pem",
            "scheme": "grpc",
            "default_connection_security": "tls",
            "server": _SIGNED_SERVER_ENDPOINT,
        }
        with unittest.mock.patch("nvflare.tool.package.package_commands.output_error_message") as error:
            valid = _validate_signed_metadata(signed_meta, {}, "site-3.crt")

        error.assert_called_once()
        assert valid is False
        assert "Invalid signed zip cert_type" in error.call_args.args[1]

    def test_validate_signed_metadata_returns_after_invalid_kind_when_error_is_mocked(self):
        signed_meta = {
            "artifact_type": "nvflare.cert.signed",
            "schema_version": "1",
            "request_id": "1" * 32,
            "project": "example_project",
            "name": "site-3",
            "org": "nvidia",
            "kind": "workspace",
            "cert_type": "client",
            "cert_file": "site-3.crt",
            "rootca_file": "rootCA.pem",
            "scheme": "grpc",
            "default_connection_security": "tls",
            "server": _SIGNED_SERVER_ENDPOINT,
            "hashes": {
                "csr_sha256": "1" * 64,
                "site_yaml_sha256": "1" * 64,
                "certificate_sha256": "1" * 64,
                "rootca_sha256": "1" * 64,
                "public_key_sha256": "1" * 64,
            },
        }

        with unittest.mock.patch("nvflare.tool.package.package_commands.output_error_message") as error:
            valid = _validate_signed_metadata(signed_meta, {}, "site-3.crt")

        error.assert_called_once()
        assert valid is False
        assert "Invalid signed zip kind/cert_type combination" in error.call_args.args[1]

    def test_validate_signed_metadata_rejects_listening_host_when_error_is_mocked(self):
        signed_meta = {
            "artifact_type": "nvflare.cert.signed",
            "schema_version": "1",
            "request_id": "1" * 32,
            "project": "example_project",
            "name": "site-3",
            "org": "nvidia",
            "kind": "site",
            "cert_type": "client",
            "cert_file": "site-3.crt",
            "rootca_file": "rootCA.pem",
            "scheme": "grpc",
            "default_connection_security": "tls",
            "server": _SIGNED_SERVER_ENDPOINT,
            "hashes": {
                "csr_sha256": "1" * 64,
                "site_yaml_sha256": "1" * 64,
                "certificate_sha256": "1" * 64,
                "rootca_sha256": "1" * 64,
                "public_key_sha256": "1" * 64,
            },
        }
        site_meta = {
            "name": "example_project",
            "participants": [
                {
                    "name": "site-3",
                    "type": "client",
                    "org": "nvidia",
                    "listening_host": "site-3.internal",
                }
            ],
        }

        with unittest.mock.patch("nvflare.tool.package.package_commands.output_error_message") as error:
            valid = _validate_signed_metadata(signed_meta, site_meta, "site-3.crt")

        error.assert_called_once()
        assert valid is False
        assert "listening_host" in error.call_args.args[1]

    def test_validate_signed_hashes_does_not_emit_spurious_mismatch_when_hash_missing(self):
        with unittest.mock.patch("nvflare.tool.package.package_commands.output_error_message") as error:
            valid = _validate_signed_hashes(
                {"hashes": {}}, {"site.yaml": b"site", "cert": b"cert", "rootCA.pem": b"root"}
            )

        assert valid is False
        error.assert_called_once()
        assert "Missing required hash" in error.call_args.args[1]

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

    def test_resolve_request_dir_rejects_explicit_candidate_without_material_when_request_id_absent(self, tmp_path):
        request_dir = tmp_path / "empty"
        request_dir.mkdir()
        args = types.SimpleNamespace(request_dir=str(request_dir))
        identity = {"name": "site-3"}

        with pytest.raises(SystemExit) as exc_info:
            _resolve_request_dir(args, str(tmp_path / "site-3.signed.zip"), identity)

        assert exc_info.value.code == 1

    def test_validate_local_site_identity_reports_unsupported_participant_type(self):
        project_dict = {PropKey.NAME: "hospital_federation"}
        participant = {"name": "hospital-a", "type": "workspace", "org": "hospital_alpha"}
        signed_meta = {
            "project": "hospital_federation",
            "name": "hospital-a",
            "org": "hospital_alpha",
            "kind": "site",
            "cert_type": "client",
            "cert_role": None,
        }

        with unittest.mock.patch("nvflare.tool.package.package_commands.output_error_message") as error:
            valid = _validate_local_site_identity(project_dict, participant, signed_meta)

        assert valid is False
        error.assert_called_once()
        assert "unsupported participant type 'workspace'" in error.call_args.kwargs["detail"]


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
    assert "Server endpoint URI is required" in captured.err
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
    assert "nvflare package hospital-1.signed.zip --fingerprint <expected_fingerprint>" in help_text
    assert "nvflare package hospital-1.signed.zip --request-dir ./hospital-1" in help_text
    assert "--cert ./signed/hospital-1/hospital-1.crt" not in help_text
    assert "--key ./csr/hospital-1.key" not in help_text
    assert "--rootca ./signed/hospital-1/rootCA.pem" not in help_text


@pytest.mark.parametrize(
    "old_args",
    [
        ["--admin-port", "8003"],
        ["--dir", "VALUE"],
        ["--cert", "VALUE"],
        ["--key", "VALUE"],
        ["--rootca", "VALUE"],
        ["--project-name", "VALUE"],
        ["-n", "VALUE"],
        ["--confirm-rootca"],
    ],
)
def test_package_parser_rejects_removed_low_level_options(old_args, tmp_path):
    import argparse

    from nvflare.tool.package.package_cli import def_package_cli_parser

    root = argparse.ArgumentParser(prog="nvflare")
    subs = root.add_subparsers(dest="sub_command")
    def_package_cli_parser(subs)

    argv = ["package", "hospital-1.signed.zip"] + [str(tmp_path) if value == "VALUE" else value for value in old_args]

    with pytest.raises(SystemExit) as exc_info:
        root.parse_args(argv)

    assert exc_info.value.code == 2


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
    input_arg = next(arg for arg in schema["args"] if arg["name"] == "input")
    assert input_arg["required"] is True
    assert "nargs" not in input_arg


def test_package_help_shows_signed_zip_input_required():
    import argparse

    from nvflare.tool.package.package_cli import def_package_cli_parser

    root = argparse.ArgumentParser(prog="nvflare")
    subs = root.add_subparsers()
    parser = def_package_cli_parser(subs)["package"]

    help_text = parser.format_help()

    assert " input" in help_text
    assert "[input]" not in help_text


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

    def test_no_key_returns_after_error_when_error_is_mocked(self, tmp_path):
        with unittest.mock.patch("nvflare.tool.package.package_commands.output_error") as error:
            name = _discover_name_from_dir(str(tmp_path))

        assert name == ""
        error.assert_called_once()

    def test_multiple_keys_returns_after_error_when_error_is_mocked(self, tmp_path):
        (tmp_path / "alice.key").write_text("dummy")
        (tmp_path / "bob.key").write_text("dummy")

        with unittest.mock.patch("nvflare.tool.package.package_commands.output_error_message") as error:
            name = _discover_name_from_dir(str(tmp_path))

        assert name == ""
        error.assert_called_once()


def test_package_compat_output_alias_sets_output_format(tmp_path):
    import argparse

    from nvflare.tool.package.package_cli import def_package_cli_parser, handle_package_cmd

    root = argparse.ArgumentParser(prog="nvflare")
    subs = root.add_subparsers(dest="sub_command")
    def_package_cli_parser(subs)
    args = root.parse_args(["package", "hospital-1.signed.zip", "--output", "json"])

    with unittest.mock.patch("nvflare.tool.cli_output.set_output_format") as set_output_format:
        with unittest.mock.patch(
            "nvflare.tool.package.package_commands.handle_package", return_value=1
        ) as handle_package:
            rc = handle_package_cmd(args)

    set_output_format.assert_called_once_with("json")
    handle_package.assert_called_once_with(args)
    assert rc == 1


# ---------------------------------------------------------------------------
# Integration tests: full kit assembly
# ---------------------------------------------------------------------------


@pytest.fixture()
def cert_env(tmp_path):
    """Provide a temporary directory with a CA, signed cert, and key."""
    ca_key, ca_cert, rootca_path = _make_ca(str(tmp_path))
    return dict(tmp_path=tmp_path, ca_key=ca_key, ca_cert=ca_cert, rootca_path=rootca_path)


class TestClientKitAssembly:
    def test_explicit_mode_returns_build_error_without_success_output(self, cert_env, tmp_path, monkeypatch):
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
            rootca=rootca,
            workspace=str(tmp_path / "ws"),
        )
        output_ok = unittest.mock.Mock()
        monkeypatch.setattr("nvflare.tool.package.package_commands.output_ok", output_ok)
        monkeypatch.setattr(
            "nvflare.tool.package.package_commands._build_selected_participant_package", lambda **kwargs: 1
        )

        result = handle_package(args)

        assert result == 1
        output_ok.assert_not_called()

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
        assert cfg["servers"][0]["service"]["target"] == "server.example.com:8002"

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
        not the raw endpoint host."""
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
        # target must use server.name (fl-server), not the raw endpoint host
        assert cfg["servers"][0]["service"]["target"] == "fl-server:8002"

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
    """When kit type cannot be derived from the cert, CERT_TYPE_UNKNOWN is raised (exit 1)."""

    def test_missing_kit_type_cert_no_role_exits_1(self, cert_env, tmp_path):
        """No explicit kit type and cert has no UNSTRUCTURED_NAME -> CERT_TYPE_UNKNOWN (exit 1)."""
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
# TestKitTypeFromCert: kit type derived from cert UNSTRUCTURED_NAME
# ---------------------------------------------------------------------------


class TestKitTypeFromCert:
    """Kit type is derived from the signed cert when no explicit type is provided."""

    def test_client_kit_type_from_cert(self, cert_env, tmp_path):
        """Cert with UNSTRUCTURED_NAME=client -> client kit assembled without explicit type."""
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
        """Cert with UNSTRUCTURED_NAME=server -> server kit assembled without explicit type."""
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
        """Cert with UNSTRUCTURED_NAME=lead -> admin kit assembled without explicit type."""
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

    def test_explicit_type_overrides_cert_role(self, cert_env, tmp_path):
        """Explicit internal kit_type overrides what is in the cert's UNSTRUCTURED_NAME."""
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
        """--dir mode: kit type derived from cert when no explicit type is provided."""
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


_SIGNED_ZIP_CA_KEYS = {}


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
    ca_key=None,
    ca_cert=None,
    rootca_path=None,
    provision_version=None,
):
    request_dir = tmp_path / name
    request_dir.mkdir(exist_ok=True)
    if ca_key is None or ca_cert is None or rootca_path is None:
        ca_dir = tmp_path / "ca"
        ca_dir.mkdir(exist_ok=True)
        ca_key, ca_cert, rootca_path = _make_ca(str(ca_dir), name=ca_project or project_name)
    key_path, cert_path = _make_signed_cert(
        ca_key, ca_cert, name, str(request_dir), f"{name}.crt", role=cert_type, org=cert_org or org
    )
    rootca_copy = request_dir / "rootCA.pem"
    shutil.copy2(rootca_path, str(rootca_copy))
    participant_type = "server" if kind == "server" else "client" if cert_type == "client" else "admin"
    participant = {
        "name": name,
        "type": participant_type,
        "org": org,
    }
    if participant_type == "server":
        participant.update(
            {
                "fed_learn_port": 8002,
                "admin_port": 8003,
            }
        )
    elif participant_type == "admin":
        participant["role"] = cert_type
    local_site = {
        "name": project_name,
        "participants": [participant],
    }
    approval_site = _approval_site_from_participant_definition(local_site)

    site_yaml_path = request_dir / "site.yaml"
    approval_site_yaml_path = request_dir / "approval-site.yaml"
    request_json_path = request_dir / "request.json"
    signed_json_path = request_dir / "signed.json"
    site_yaml_path.write_text(yaml.safe_dump(local_site, sort_keys=False))
    approval_site_yaml_path.write_text(yaml.safe_dump(approval_site, sort_keys=False))
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
                "site_yaml_sha256": _signed_zip_sha256_file(approval_site_yaml_path),
                "csr_sha256": "1" * 64,
                "public_key_sha256": public_key_sha256,
            },
            sort_keys=True,
        )
    )
    signed_json = {
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
        "scheme": "grpc",
        "default_connection_security": "tls",
        "server": {
            "host": "fl-server",
            "fed_learn_port": 8002,
            "admin_port": 8003,
        },
        "certificate": {
            "serial": "01",
            "valid_until": "2029-04-24T00:00:00Z",
        },
        "cert_file": f"{name}.crt",
        "rootca_file": "rootCA.pem",
        "hashes": {
            "csr_sha256": "1" * 64,
            "site_yaml_sha256": _signed_zip_sha256_file(approval_site_yaml_path),
            "certificate_sha256": cert_sha256,
            "rootca_sha256": _signed_zip_sha256_file(rootca_copy),
            "public_key_sha256": public_key_sha256,
        },
    }
    if provision_version is not None:
        signed_json[CA_INFO_FIELD] = {
            PROVISION_VERSION_FIELD: provision_version,
            ROOTCA_FINGERPRINT_FIELD: cert_fingerprint_sha256(load_crt(str(rootca_copy))),
        }
    signed_json_path.write_text(json.dumps(signed_json, sort_keys=True))
    signed_sig_path = request_dir / "signed.json.sig"
    signed_sig_path.write_text(sign_content(signed_json_path.read_bytes(), ca_key))
    signed_zip = request_dir / f"{name}.signed.zip"
    with zipfile.ZipFile(signed_zip, "w") as zf:
        zf.write(signed_json_path, "signed.json")
        zf.write(signed_sig_path, "signed.json.sig")
        zf.write(approval_site_yaml_path, "site.yaml")
        with open(cert_path, "rb") as cert_file:
            zf.writestr(cert_member or (f"../{name}.crt" if traversal else f"{name}.crt"), cert_file.read())
        zf.write(rootca_copy, "rootCA.pem")
        if include_key:
            zf.write(key_path, key_member or f"{name}.key")
    os.remove(cert_path)
    os.remove(rootca_copy)
    os.remove(signed_json_path)
    os.remove(signed_sig_path)
    os.remove(approval_site_yaml_path)
    _SIGNED_ZIP_CA_KEYS[str(signed_zip)] = ca_key
    return signed_zip, request_dir, key_path


def _approval_site_from_participant_definition(participant_definition: dict) -> dict:
    approval_site = {
        "name": participant_definition["name"],
        "participants": [],
    }
    if participant_definition.get("description"):
        approval_site["description"] = participant_definition["description"]
    for participant in participant_definition["participants"]:
        approval_participant = dict(participant)
        approval_participant.pop("connection_security", None)
        approval_participant.pop("server", None)
        approval_participant.pop("fed_learn_port", None)
        approval_participant.pop("admin_port", None)
        approval_site["participants"].append(approval_participant)
    return approval_site


def _participant_request_identity(participant_definition: dict) -> dict:
    participant = participant_definition["participants"][0]
    if participant["type"] == "server":
        kind = "server"
        cert_type = "server"
        cert_role = None
    elif participant["type"] == "admin":
        kind = "user"
        cert_type = participant["role"]
        cert_role = participant["role"]
    else:
        kind = "site"
        cert_type = "client"
        cert_role = None
    return {
        "project": participant_definition["name"],
        "name": participant["name"],
        "org": participant["org"],
        "kind": kind,
        "cert_type": cert_type,
        "cert_role": cert_role,
    }


def _make_v2_signed_zip(
    tmp_path,
    participant_definition: dict,
    *,
    scheme="grpc",
    default_connection_security="tls",
    server_endpoint=None,
    request_id="11111111111111111111111111111111",
):
    identity = _participant_request_identity(participant_definition)
    name = identity["name"]
    request_dir = tmp_path / name
    request_dir.mkdir(exist_ok=True)
    ca_dir = tmp_path / f"ca-{name.replace('@', '_').replace('.', '_')}"
    ca_dir.mkdir(exist_ok=True)
    ca_key, ca_cert, rootca_path = _make_ca(str(ca_dir), name=identity["project"], org=identity["org"])
    key_path, cert_path = _make_signed_cert(
        ca_key,
        ca_cert,
        name,
        str(request_dir),
        f"{name}.crt",
        role=identity["cert_type"],
        org=identity["org"],
    )
    rootca_copy = request_dir / "rootCA.pem"
    shutil.copy2(rootca_path, str(rootca_copy))

    local_site_yaml_path = request_dir / "site.yaml"
    request_json_path = request_dir / "request.json"
    signed_json_path = request_dir / "signed.json"
    approval_site_yaml_path = request_dir / "approval-site.yaml"
    with local_site_yaml_path.open("w") as f:
        yaml.safe_dump(participant_definition, f, sort_keys=False)
    approval_site = _approval_site_from_participant_definition(participant_definition)
    with approval_site_yaml_path.open("w") as f:
        yaml.safe_dump(approval_site, f, sort_keys=False)

    public_key_sha256 = _public_key_sha256_from_cert(cert_path)
    if server_endpoint is None:
        server_endpoint = _SIGNED_SERVER_ENDPOINT
    request_json = {
        "artifact_type": "nvflare.cert.request",
        "schema_version": "1",
        "request_id": request_id,
        "created_at": "2026-04-24T00:00:00Z",
        "project": identity["project"],
        "name": name,
        "org": identity["org"],
        "kind": identity["kind"],
        "cert_type": identity["cert_type"],
        "cert_role": identity["cert_role"],
        "csr_sha256": "1" * 64,
        "public_key_sha256": public_key_sha256,
    }
    request_json_path.write_text(json.dumps(request_json, sort_keys=True))
    signed_json_path.write_text(
        json.dumps(
            {
                "artifact_type": "nvflare.cert.signed",
                "schema_version": "1",
                "request_id": request_id,
                "approved_at": "2026-04-24T00:00:00Z",
                "project": identity["project"],
                "name": name,
                "org": identity["org"],
                "kind": identity["kind"],
                "cert_type": identity["cert_type"],
                "cert_role": identity["cert_role"],
                "scheme": scheme,
                "default_connection_security": default_connection_security,
                "server": server_endpoint,
                "certificate": {
                    "serial": "01",
                    "valid_until": "2029-04-24T00:00:00Z",
                },
                "cert_file": f"{name}.crt",
                "rootca_file": "rootCA.pem",
                "hashes": {
                    "csr_sha256": "1" * 64,
                    "site_yaml_sha256": _signed_zip_sha256_file(approval_site_yaml_path),
                    "certificate_sha256": _signed_zip_sha256_file(cert_path),
                    "rootca_sha256": _signed_zip_sha256_file(rootca_copy),
                    "public_key_sha256": public_key_sha256,
                },
            },
            sort_keys=True,
        )
    )
    signed_sig_path = request_dir / "signed.json.sig"
    signed_sig_path.write_text(sign_content(signed_json_path.read_bytes(), ca_key))
    signed_zip = request_dir / f"{name}.signed.zip"
    with zipfile.ZipFile(signed_zip, "w") as zf:
        zf.write(signed_json_path, "signed.json")
        zf.write(signed_sig_path, "signed.json.sig")
        zf.write(approval_site_yaml_path, "site.yaml")
        zf.write(cert_path, f"{name}.crt")
        zf.write(rootca_copy, "rootCA.pem")
    os.remove(cert_path)
    os.remove(rootca_copy)
    os.remove(signed_json_path)
    os.remove(signed_sig_path)
    os.remove(approval_site_yaml_path)
    _SIGNED_ZIP_CA_KEYS[str(signed_zip)] = ca_key
    return signed_zip, request_dir, key_path


def _rewrite_signed_zip_metadata(signed_zip, mutate):
    with zipfile.ZipFile(signed_zip, "r") as zf:
        members = {name: zf.read(name) for name in zf.namelist()}
    metadata = json.loads(members["signed.json"])
    mutate(metadata)
    members["signed.json"] = json.dumps(metadata, sort_keys=True).encode("utf-8")
    ca_key = _SIGNED_ZIP_CA_KEYS.get(str(signed_zip))
    if ca_key is None:
        raise RuntimeError(f"missing CA key for signed zip test fixture: {signed_zip}")
    members["signed.json.sig"] = sign_content(members["signed.json"], ca_key).encode("utf-8")
    with zipfile.ZipFile(signed_zip, "w") as zf:
        for name, content in members.items():
            zf.writestr(name, content)


def _signed_zip_args(signed_zip, tmp_path, **kwargs):
    defaults = dict(
        input=str(signed_zip),
        endpoint=None,
        workspace=str(tmp_path / "ws"),
        force=True,
        request_dir=None,
        project_name=None,
        dir=None,
        cert=None,
        key=None,
        rootca=None,
        name=None,
        kit_type=None,
        expected_fingerprint=None,
    )
    defaults.update(kwargs)
    return _make_args(**defaults)


def _rootca_fingerprint_from_signed_zip(signed_zip) -> str:
    with zipfile.ZipFile(signed_zip, "r") as zf:
        return cert_fingerprint_sha256(load_crt_bytes(zf.read("rootCA.pem")))


def _make_shared_package_ca(tmp_path, project_name="example_project", org="nvidia"):
    ca_dir = tmp_path / "shared-ca"
    ca_dir.mkdir(exist_ok=True)
    return _make_ca(str(ca_dir), name=project_name, org=org)


def _participant_names(project):
    return sorted(
        p.name
        for p in project.get_all_participants()
        if p.type != "server" and p.name != "fl-server" and p.name != _DUMMY_SERVER_NAME
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


class TestDistributedProvisioningV2PackageMode:
    def test_profile_based_signed_zip_rejects_endpoint_override(self, tmp_path):
        participant_definition = {
            "name": "hospital_federation",
            "participants": [
                {
                    "name": "hospital-a",
                    "type": "client",
                    "org": "hospital_alpha",
                }
            ],
        }
        signed_zip, request_dir, _ = _make_v2_signed_zip(
            tmp_path, participant_definition, scheme="grpc", default_connection_security="tls"
        )
        args = _signed_zip_args(
            signed_zip,
            tmp_path,
            endpoint="grpc://other-server.example.com:9000",
            request_dir=str(request_dir),
        )

        with pytest.raises(SystemExit) as exc_info:
            handle_package(args)

        assert exc_info.value.code == 4

    def test_signed_zip_rejects_project_file_override(self, tmp_path):
        participant_definition = {
            "name": "hospital_federation",
            "participants": [
                {
                    "name": "hospital-a",
                    "type": "client",
                    "org": "hospital_alpha",
                }
            ],
        }
        signed_zip, request_dir, _ = _make_v2_signed_zip(
            tmp_path, participant_definition, scheme="grpc", default_connection_security="tls"
        )
        args = _signed_zip_args(
            signed_zip,
            tmp_path,
            project_file=str(tmp_path / "other-project.yaml"),
            request_dir=str(request_dir),
        )

        with pytest.raises(SystemExit) as exc_info:
            handle_package(args)

        assert exc_info.value.code == 4

    def test_signed_zip_validates_local_identity_before_endpoint_resolution(self, tmp_path):
        participant_definition = {
            "name": "hospital_federation",
            "participants": [
                {
                    "name": "hospital-a",
                    "type": "client",
                    "org": "hospital_alpha",
                }
            ],
        }
        signed_zip, request_dir, _ = _make_v2_signed_zip(
            tmp_path, participant_definition, scheme="grpc", default_connection_security="tls"
        )
        tampered_site = copy.deepcopy(participant_definition)
        tampered_site["participants"][0]["org"] = "other_org"
        (request_dir / "site.yaml").write_text(yaml.safe_dump(tampered_site, sort_keys=False))
        args = _signed_zip_args(
            signed_zip,
            tmp_path,
            request_dir=str(request_dir),
        )

        with unittest.mock.patch("nvflare.tool.package.package_commands._resolve_packaging_endpoint") as resolve:
            with pytest.raises(SystemExit) as exc_info:
                handle_package(args)

        assert exc_info.value.code == 4
        resolve.assert_not_called()

    def test_client_signed_zip_uses_signed_server_endpoint_without_local_server_block(self, tmp_path):
        participant_definition = {
            "name": "hospital_federation",
            "participants": [
                {
                    "name": "hospital-a",
                    "type": "client",
                    "org": "hospital_alpha",
                }
            ],
        }
        signed_zip, request_dir, _ = _make_v2_signed_zip(
            tmp_path,
            participant_definition,
            scheme="grpc",
            default_connection_security="tls",
            server_endpoint={
                "host": "profile-server.hospital-central.org",
                "fed_learn_port": 9002,
                "admin_port": 9003,
            },
        )
        args = _signed_zip_args(signed_zip, tmp_path, request_dir=str(request_dir))

        handle_package(args)

        kit_dir = os.path.join(str(tmp_path / "ws"), "hospital_federation", "prod_00", "hospital-a")
        with open(os.path.join(kit_dir, "startup", "fed_client.json")) as f:
            cfg = json.load(f)
        assert cfg["servers"][0]["identity"] == "profile-server.hospital-central.org"
        assert cfg["servers"][0]["service"]["target"] == "profile-server.hospital-central.org:9002"
        assert yaml.safe_load((request_dir / "site.yaml").read_text()) == participant_definition

    def test_server_signed_zip_uses_signed_endpoint_without_local_endpoint_fields(self, tmp_path):
        participant_definition = {
            "name": "hospital_federation",
            "participants": [
                {
                    "name": "server1.hospital-central.org",
                    "type": "server",
                    "org": "hospital_central",
                    "connection_security": "clear",
                }
            ],
        }
        signed_zip, request_dir, _ = _make_v2_signed_zip(
            tmp_path,
            participant_definition,
            scheme="grpc",
            default_connection_security="tls",
            server_endpoint={
                "host": "server1.hospital-central.org",
                "fed_learn_port": 8002,
                "admin_port": 8003,
            },
        )
        args = _signed_zip_args(signed_zip, tmp_path, request_dir=str(request_dir))

        handle_package(args)

        kit_dir = os.path.join(str(tmp_path / "ws"), "hospital_federation", "prod_00", "server1.hospital-central.org")
        with open(os.path.join(kit_dir, "startup", "fed_server.json")) as f:
            cfg = json.load(f)
        assert cfg["servers"][0]["service"]["target"] == "server1.hospital-central.org:8002"
        assert cfg["servers"][0]["admin_port"] == 8003
        assert cfg["servers"][0]["connection_security"] == "clear"

    def test_signed_zip_relies_on_single_public_key_hash_validator(self, tmp_path):
        participant_definition = {
            "name": "hospital_federation",
            "participants": [
                {
                    "name": "hospital-a",
                    "type": "client",
                    "org": "hospital_alpha",
                }
            ],
        }
        signed_zip, request_dir, _ = _make_v2_signed_zip(
            tmp_path, participant_definition, scheme="grpc", default_connection_security="tls"
        )
        expected = _rootca_fingerprint_from_signed_zip(signed_zip)
        args = _signed_zip_args(
            signed_zip,
            tmp_path,
            request_dir=str(request_dir),
            expected_fingerprint=expected,
        )

        with unittest.mock.patch(
            "nvflare.tool.package.package_commands._validate_signed_public_key_hash", return_value=True
        ) as validate_hash:
            with unittest.mock.patch(
                "nvflare.tool.package.package_commands._cert_public_key_sha256",
                side_effect=AssertionError("duplicate public-key hash check"),
            ):
                handle_package(args)

        validate_hash.assert_called_once()

    def test_signed_zip_rejects_local_listening_host_until_listener_cert_supported(self, tmp_path, capsys):
        participant_definition = {
            "name": "hospital_federation",
            "participants": [
                {
                    "name": "hospital-a",
                    "type": "client",
                    "org": "hospital_alpha",
                }
            ],
        }
        signed_zip, request_dir, _ = _make_v2_signed_zip(
            tmp_path, participant_definition, scheme="grpc", default_connection_security="tls"
        )
        local_site = copy.deepcopy(participant_definition)
        local_site["participants"][0]["listening_host"] = "hospital-a.internal"
        (request_dir / "site.yaml").write_text(yaml.safe_dump(local_site, sort_keys=False))
        args = _signed_zip_args(
            signed_zip,
            tmp_path,
            request_dir=str(request_dir),
        )

        with pytest.raises(SystemExit) as exc_info:
            handle_package(args)

        assert exc_info.value.code == 4
        captured = capsys.readouterr()
        assert "listening_host" in captured.err
        assert "not supported" in captured.err

    def test_client_signed_zip_uses_profile_defaults_and_signed_endpoint_without_override_args(self, tmp_path):
        participant_definition = {
            "name": "hospital_federation",
            "participants": [
                {
                    "name": "hospital-a",
                    "type": "client",
                    "org": "hospital_alpha",
                }
            ],
        }
        signed_zip, request_dir, _ = _make_v2_signed_zip(
            tmp_path, participant_definition, scheme="grpc", default_connection_security="tls"
        )
        args = _signed_zip_args(
            signed_zip,
            tmp_path,
            endpoint=None,
            request_dir=str(request_dir),
        )

        handle_package(args)

        kit_dir = os.path.join(str(tmp_path / "ws"), "hospital_federation", "prod_00", "hospital-a")
        with open(os.path.join(kit_dir, "startup", "fed_client.json")) as f:
            cfg = json.load(f)
        assert cfg["servers"][0]["service"]["scheme"] == "grpc"
        assert cfg["servers"][0]["identity"] == "server1.hospital-central.org"
        assert cfg["client"]["connection_security"] == "tls"
        assert cfg["servers"][0]["service"]["target"] == "server1.hospital-central.org:8002"
        assert yaml.safe_load((request_dir / "site.yaml").read_text()) == participant_definition

    def test_user_signed_zip_uses_profile_defaults_and_signed_endpoint_without_override_args(self, tmp_path):
        participant_definition = {
            "name": "hospital_federation",
            "participants": [
                {
                    "name": "alice@hospital-alpha.org",
                    "type": "admin",
                    "org": "hospital_alpha",
                    "role": "lead",
                }
            ],
        }
        signed_zip, request_dir, _ = _make_v2_signed_zip(
            tmp_path, participant_definition, scheme="grpc", default_connection_security="tls"
        )
        args = _signed_zip_args(
            signed_zip,
            tmp_path,
            endpoint=None,
            request_dir=str(request_dir),
        )

        handle_package(args)

        kit_dir = os.path.join(str(tmp_path / "ws"), "hospital_federation", "prod_00", "alice@hospital-alpha.org")
        with open(os.path.join(kit_dir, "startup", "fed_admin.json")) as f:
            cfg = json.load(f)
        assert cfg["admin"]["scheme"] == "grpc"
        assert cfg["admin"]["host"] == "server1.hospital-central.org"
        assert cfg["admin"]["port"] == 8003
        assert cfg["admin"]["connection_security"] == "tls"
        assert yaml.safe_load((request_dir / "site.yaml").read_text()) == participant_definition

    @pytest.mark.parametrize(
        "local_override, expected_connection_security",
        [
            ("clear", "clear"),
            (None, "tls"),
        ],
    )
    def test_server_signed_zip_uses_local_connection_security_override_then_profile_default(
        self, tmp_path, local_override, expected_connection_security
    ):
        server = {
            "name": "server1.hospital-central.org",
            "type": "server",
            "org": "hospital_central",
        }
        if local_override:
            server["connection_security"] = local_override
        participant_definition = {
            "name": "hospital_federation",
            "participants": [server],
        }
        signed_zip, request_dir, _ = _make_v2_signed_zip(
            tmp_path, participant_definition, scheme="grpc", default_connection_security="tls"
        )
        args = _signed_zip_args(
            signed_zip,
            tmp_path,
            endpoint=None,
            request_dir=str(request_dir),
        )

        handle_package(args)

        kit_dir = os.path.join(str(tmp_path / "ws"), "hospital_federation", "prod_00", "server1.hospital-central.org")
        with open(os.path.join(kit_dir, "startup", "fed_server.json")) as f:
            cfg = json.load(f)
        assert cfg["servers"][0]["service"]["scheme"] == "grpc"
        assert cfg["servers"][0]["service"]["target"] == "server1.hospital-central.org:8002"
        assert cfg["servers"][0]["admin_port"] == 8003
        assert cfg["servers"][0]["connection_security"] == expected_connection_security
        assert yaml.safe_load((request_dir / "site.yaml").read_text()) == participant_definition

    def test_server_package_ignores_signed_default_connection_security_when_local_override_exists(self, tmp_path):
        server = {
            "name": "server1.hospital-central.org",
            "type": "server",
            "org": "hospital_central",
            "connection_security": "clear",
        }
        participant_definition = {
            "name": "hospital_federation",
            "participants": [server],
        }
        signed_zip, request_dir, _ = _make_v2_signed_zip(
            tmp_path, participant_definition, scheme="grpc", default_connection_security="mtls"
        )
        with zipfile.ZipFile(signed_zip) as zf:
            signed_meta = json.loads(zf.read("signed.json"))
            signed_site = yaml.safe_load(zf.read("site.yaml"))
        assert signed_meta["default_connection_security"] == "mtls"
        assert "connection_security" not in signed_site["participants"][0]

        args = _signed_zip_args(
            signed_zip,
            tmp_path,
            endpoint=None,
            request_dir=str(request_dir),
        )

        handle_package(args)

        kit_dir = os.path.join(str(tmp_path / "ws"), "hospital_federation", "prod_00", "server1.hospital-central.org")
        with open(os.path.join(kit_dir, "startup", "fed_server.json")) as f:
            cfg = json.load(f)
        assert cfg["servers"][0]["connection_security"] == "clear"
        assert yaml.safe_load((request_dir / "site.yaml").read_text()) == participant_definition

    def test_signed_zip_uses_builders_from_local_participant_definition(self, tmp_path):
        participant_definition = {
            "name": "hospital_federation",
            "participants": [
                {
                    "name": "hospital-a",
                    "type": "client",
                    "org": "hospital_alpha",
                }
            ],
            "builders": [
                {"path": "nvflare.lighter.impl.workspace.WorkspaceBuilder"},
                {
                    "path": "nvflare.lighter.impl.static_file.StaticFileBuilder",
                    "args": {"config_folder": "custom_config"},
                },
                {"path": "nvflare.lighter.impl.signature.SignatureBuilder"},
            ],
        }
        signed_zip, request_dir, _ = _make_v2_signed_zip(
            tmp_path, participant_definition, scheme="grpc", default_connection_security="tls"
        )
        args = _signed_zip_args(
            signed_zip,
            tmp_path,
            endpoint=None,
            request_dir=str(request_dir),
        )

        captured = _run_signed_zip_with_captured_provisioner(args, tmp_path / "ws" / "hospital_federation" / "prod_00")

        assert captured["project"].name == "hospital_federation"
        assert _participant_names(captured["project"]) == ["hospital-a"]
        assert sum(isinstance(b, WorkspaceBuilder) for b in captured["builders"]) == 1
        assert any(isinstance(b, PrebuiltCertBuilder) for b in captured["builders"])
        static_builder = next(b for b in captured["builders"] if isinstance(b, StaticFileBuilder))
        assert static_builder.config_folder == "custom_config"
        assert static_builder.scheme == "grpc"
        assert not any(isinstance(b, SignatureBuilder) for b in captured["builders"])
        assert yaml.safe_load((request_dir / "site.yaml").read_text()) == participant_definition


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
        assert "--fingerprint" in help_text
        assert "--expected-fingerprint" in help_text
        assert "--confirm-rootca" not in help_text
        assert "Custom builders are honored" in help_text
        assert "are ignored" not in help_text
        assert "--cert ./signed/hospital-1/hospital-1.crt" not in help_text
        assert "--key ./csr/hospital-1.key" not in help_text

        with pytest.raises(SystemExit) as exc_info:
            handle_schema_flag(parser, "nvflare package", _PACKAGE_EXAMPLES, ["--schema"])
        assert exc_info.value.code == 0
        schema_text = capsys.readouterr().out
        assert ".signed.zip" in schema_text
        assert "--fingerprint" in schema_text
        assert "--expected-fingerprint" in schema_text
        assert "--confirm-rootca" not in schema_text
        assert "--cert" not in schema_text
        assert "--rootca" not in schema_text


class TestSignedZipPackageMode:
    @pytest.mark.parametrize(
        "value, expected",
        [
            ("sha256:aabbcc", "aabbcc"),
            ("SHA256:AABBCC", "aabbcc"),
            ("  SHA256:AABBCC  ", "aabbcc"),
            ("AABBCC", "aabbcc"),
            (None, ""),
        ],
    )
    def test_normalize_hash_accepts_case_insensitive_prefix(self, value, expected):
        assert _normalize_hash(value) == expected

    @pytest.mark.parametrize(
        "value",
        [
            "sha256 Fingerprint=aa:bb:cc:dd:ee:ff:00:11:22:33:44:55:66:77:88:99:aa:bb:cc:dd:ee:ff:00:11:22:33:44:55:66:77:88:99",
            "SHA256:AA:BB:CC:DD:EE:FF:00:11:22:33:44:55:66:77:88:99:AA:BB:CC:DD:EE:FF:00:11:22:33:44:55:66:77:88:99",
            "aabbccddeeff00112233445566778899aabbccddeeff00112233445566778899",
        ],
    )
    def test_normalize_rootca_fingerprint_accepts_common_forms(self, value):
        assert normalize_sha256_fingerprint(value) == (
            "SHA256:AA:BB:CC:DD:EE:FF:00:11:22:33:44:55:66:77:88:99:" "AA:BB:CC:DD:EE:FF:00:11:22:33:44:55:66:77:88:99"
        )

    @pytest.mark.parametrize(
        "value",
        [
            "SHA256:AA:BB",
            "SHA256:AA:BB:CC:DD:EE:FF:00:11:22:33:44:55:66:77:88:99:AA:BB:CC:DD:EE:FF:00:11:22:33:44:55:66:77:88:ZZ",
            "AA:BB:CC:DD:EE:FF:00:11:22:33:44:55:66:77:88:99:AA:BB:CC:DD:EE:FF:00:11:22:33:44:55:66:77:88:99:",
        ],
    )
    def test_normalize_rootca_fingerprint_rejects_invalid_forms(self, value):
        assert normalize_sha256_fingerprint(value) == ""

    def test_signed_zip_output_includes_rootca_fingerprint_by_default(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        signed_zip, request_dir, _ = _make_signed_zip(tmp_path)
        expected = _rootca_fingerprint_from_signed_zip(signed_zip)
        args = _signed_zip_args(signed_zip, tmp_path, request_dir=str(request_dir))

        handle_package(args)

        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert "rootca_fingerprint_sha256" in combined
        assert expected in combined

    def test_signed_zip_ca_info_provision_version_selects_prod_dir_out_of_order(self, tmp_path):
        ca_key, ca_cert, rootca_path = _make_shared_package_ca(tmp_path)
        signed_01, request_01, _ = _make_signed_zip(
            tmp_path,
            name="site-1",
            request_id="1" * 32,
            ca_key=ca_key,
            ca_cert=ca_cert,
            rootca_path=rootca_path,
            provision_version="01",
        )
        signed_00, request_00, _ = _make_signed_zip(
            tmp_path,
            name="site-0",
            request_id="2" * 32,
            ca_key=ca_key,
            ca_cert=ca_cert,
            rootca_path=rootca_path,
            provision_version="00",
        )

        handle_package(_signed_zip_args(signed_01, tmp_path, request_dir=str(request_01)))
        handle_package(_signed_zip_args(signed_00, tmp_path, request_dir=str(request_00)))

        workspace = str(tmp_path / "ws")
        assert os.path.isdir(os.path.join(workspace, "example_project", "prod_01", "site-1"))
        assert os.path.isdir(os.path.join(workspace, "example_project", "prod_00", "site-0"))
        assert not os.path.exists(os.path.join(workspace, "example_project", "prod_00", "site-1"))

    def test_signed_zip_same_ca_info_version_adds_participants_to_same_prod_dir(self, tmp_path):
        ca_key, ca_cert, rootca_path = _make_shared_package_ca(tmp_path)
        signed_a, request_a, _ = _make_signed_zip(
            tmp_path,
            name="site-a",
            request_id="a" * 32,
            ca_key=ca_key,
            ca_cert=ca_cert,
            rootca_path=rootca_path,
            provision_version="00",
        )
        signed_b, request_b, _ = _make_signed_zip(
            tmp_path,
            name="site-b",
            request_id="b" * 32,
            ca_key=ca_key,
            ca_cert=ca_cert,
            rootca_path=rootca_path,
            provision_version="00",
        )

        handle_package(_signed_zip_args(signed_a, tmp_path, request_dir=str(request_a)))
        handle_package(_signed_zip_args(signed_b, tmp_path, request_dir=str(request_b)))

        workspace = str(tmp_path / "ws")
        assert os.path.isdir(os.path.join(workspace, "example_project", "prod_00", "site-a"))
        assert os.path.isdir(os.path.join(workspace, "example_project", "prod_00", "site-b"))
        assert not os.path.exists(os.path.join(workspace, "example_project", "prod_01"))

    def test_signed_zip_missing_ca_info_defaults_to_prod_00(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "json")
        ca_key, ca_cert, rootca_path = _make_shared_package_ca(tmp_path)
        signed_a, request_a, _ = _make_signed_zip(
            tmp_path,
            name="site-a",
            request_id="a" * 32,
            ca_key=ca_key,
            ca_cert=ca_cert,
            rootca_path=rootca_path,
        )
        signed_b, request_b, _ = _make_signed_zip(
            tmp_path,
            name="site-b",
            request_id="b" * 32,
            ca_key=ca_key,
            ca_cert=ca_cert,
            rootca_path=rootca_path,
        )

        handle_package(_signed_zip_args(signed_a, tmp_path, request_dir=str(request_a)))
        first_output = json.loads(capsys.readouterr().out)
        handle_package(_signed_zip_args(signed_b, tmp_path, request_dir=str(request_b)))

        workspace = str(tmp_path / "ws")
        assert os.path.isdir(os.path.join(workspace, "example_project", "prod_00", "site-a"))
        assert os.path.isdir(os.path.join(workspace, "example_project", "prod_00", "site-b"))
        assert first_output["data"]["provision_version"] == DEFAULT_PROVISION_VERSION
        assert first_output["data"]["rootca_fingerprint_sha256"] == _rootca_fingerprint_from_signed_zip(signed_a)
        assert not os.path.exists(os.path.join(workspace, "example_project", "prod_01"))

    def test_signed_zip_rejects_malformed_ca_info_provision_version(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        signed_zip, request_dir, _ = _make_signed_zip(tmp_path, provision_version="00")
        _rewrite_signed_zip_metadata(
            signed_zip,
            lambda meta: meta[CA_INFO_FIELD].update({PROVISION_VERSION_FIELD: "abc"}),
        )

        with pytest.raises(SystemExit) as exc_info:
            handle_package(_signed_zip_args(signed_zip, tmp_path, request_dir=str(request_dir)))

        assert exc_info.value.code == 4
        err = capsys.readouterr().err
        assert "INVALID_SIGNED_ZIP" in err
        assert "Invalid provision version" in err

    def test_signed_zip_rejects_non_mapping_ca_info(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        signed_zip, request_dir, _ = _make_signed_zip(tmp_path, provision_version="00")
        _rewrite_signed_zip_metadata(
            signed_zip,
            lambda meta: meta.update({CA_INFO_FIELD: ["not", "a", "dict"]}),
        )

        with pytest.raises(SystemExit) as exc_info:
            handle_package(_signed_zip_args(signed_zip, tmp_path, request_dir=str(request_dir)))

        assert exc_info.value.code == 4
        err = capsys.readouterr().err
        assert "INVALID_SIGNED_ZIP" in err
        assert "ca_info must be a mapping" in err

    def test_signed_zip_rejects_malformed_ca_info_rootca_fingerprint(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        signed_zip, request_dir, _ = _make_signed_zip(tmp_path, provision_version="00")
        _rewrite_signed_zip_metadata(
            signed_zip,
            lambda meta: meta[CA_INFO_FIELD].update({ROOTCA_FINGERPRINT_FIELD: "not-a-fingerprint"}),
        )

        with pytest.raises(SystemExit) as exc_info:
            handle_package(_signed_zip_args(signed_zip, tmp_path, request_dir=str(request_dir)))

        assert exc_info.value.code == 4
        err = capsys.readouterr().err
        assert "INVALID_ROOTCA_FINGERPRINT" in err
        assert "Invalid signed ca_info root CA fingerprint" in err

    def test_signed_zip_rejects_ca_info_rootca_fingerprint_mismatch(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        signed_zip, request_dir, _ = _make_signed_zip(tmp_path, provision_version="00")
        mismatched_fingerprint = "SHA256:" + ":".join(["00"] * 32)
        _rewrite_signed_zip_metadata(
            signed_zip,
            lambda meta: meta[CA_INFO_FIELD].update({ROOTCA_FINGERPRINT_FIELD: mismatched_fingerprint}),
        )

        with pytest.raises(SystemExit) as exc_info:
            handle_package(_signed_zip_args(signed_zip, tmp_path, request_dir=str(request_dir)))

        assert exc_info.value.code == 4
        err = capsys.readouterr().err
        assert "ROOTCA_FINGERPRINT_MISMATCH" in err
        assert "does not match included rootCA.pem" in err

    def test_signed_zip_same_prod_dir_rejects_different_rootca(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        signed_a, request_a, _ = _make_signed_zip(
            tmp_path,
            name="site-a",
            request_id="a" * 32,
            provision_version="00",
        )
        signed_b, request_b, _ = _make_signed_zip(
            tmp_path,
            name="site-b",
            request_id="b" * 32,
            provision_version="00",
        )

        handle_package(_signed_zip_args(signed_a, tmp_path, request_dir=str(request_a)))
        capsys.readouterr()
        with pytest.raises(SystemExit) as exc_info:
            handle_package(_signed_zip_args(signed_b, tmp_path, request_dir=str(request_b)))

        assert exc_info.value.code != 0
        err = capsys.readouterr().err.lower()
        assert "root" in err
        assert "ca" in err
        workspace = str(tmp_path / "ws")
        assert os.path.isdir(os.path.join(workspace, "example_project", "prod_00", "site-a"))
        assert not os.path.exists(os.path.join(workspace, "example_project", "prod_00", "site-b"))
        assert not os.path.exists(os.path.join(workspace, "example_project", "prod_01", "site-b"))

    def test_signed_zip_same_prod_dir_reports_unreadable_rootca_load_failure(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        ca_key, ca_cert, rootca_path = _make_shared_package_ca(tmp_path)
        signed_a, request_a, _ = _make_signed_zip(
            tmp_path,
            name="site-a",
            request_id="a" * 32,
            ca_key=ca_key,
            ca_cert=ca_cert,
            rootca_path=rootca_path,
            provision_version="00",
        )
        signed_b, request_b, _ = _make_signed_zip(
            tmp_path,
            name="site-b",
            request_id="b" * 32,
            ca_key=ca_key,
            ca_cert=ca_cert,
            rootca_path=rootca_path,
            provision_version="00",
        )

        handle_package(_signed_zip_args(signed_a, tmp_path, request_dir=str(request_a)))
        capsys.readouterr()
        rootca_path = tmp_path / "ws" / "example_project" / "prod_00" / "site-a" / "startup" / "rootCA.pem"
        rootca_path.write_text("not a certificate")

        with pytest.raises(SystemExit) as exc_info:
            handle_package(_signed_zip_args(signed_b, tmp_path, request_dir=str(request_b)))

        assert exc_info.value.code == 4
        err = capsys.readouterr().err
        assert "ROOTCA_LOAD_FAILED" in err
        assert "unreadable rootCA.pem" in err

    def test_signed_zip_existing_participant_requires_force_in_ca_info_prod_dir(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        signed_zip, request_dir, _ = _make_signed_zip(
            tmp_path,
            request_id="a" * 32,
            provision_version="00",
        )

        handle_package(_signed_zip_args(signed_zip, tmp_path, request_dir=str(request_dir)))
        capsys.readouterr()
        with pytest.raises(SystemExit) as exc_info:
            handle_package(_signed_zip_args(signed_zip, tmp_path, request_dir=str(request_dir), force=False))

        assert exc_info.value.code != 0
        assert "exists" in capsys.readouterr().err.lower()
        workspace = str(tmp_path / "ws")
        assert os.path.isdir(os.path.join(workspace, "example_project", "prod_00", "site-3"))

        handle_package(_signed_zip_args(signed_zip, tmp_path, request_dir=str(request_dir), force=True))

        assert os.path.isdir(os.path.join(workspace, "example_project", "prod_00", "site-3"))
        assert not os.path.exists(os.path.join(workspace, "example_project", "prod_01", "site-3"))

    def test_signed_zip_toctou_existing_participant_reports_output_exists_without_removing_prod_dir(
        self, tmp_path, capsys, monkeypatch
    ):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        signed_zip, request_dir, _ = _make_signed_zip(
            tmp_path,
            request_id="a" * 32,
            provision_version="00",
        )
        original_finalize = FixedProdWorkspaceBuilder.finalize
        sentinel_path = tmp_path / "ws" / "example_project" / "prod_00" / "site-3" / "sentinel.txt"

        def _finalize_with_race(builder, project, ctx):
            os.makedirs(str(sentinel_path.parent), exist_ok=True)
            sentinel_path.write_text("existing participant output")
            return original_finalize(builder, project, ctx)

        monkeypatch.setattr(FixedProdWorkspaceBuilder, "finalize", _finalize_with_race)

        with pytest.raises(SystemExit) as exc_info:
            handle_package(_signed_zip_args(signed_zip, tmp_path, request_dir=str(request_dir), force=False))

        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "OUTPUT_DIR_EXISTS" in err
        assert "Participant output already exists" in err
        assert sentinel_path.exists()

    def test_build_package_builders_replaces_workspace_builder_and_preserves_template_files(self, tmp_path):
        original_workspace_builder = WorkspaceBuilder(template_file="custom_template.yml")
        fixed_workspace_builder = FixedProdWorkspaceBuilder(
            target_prod_dir=str(tmp_path / "workspace" / "project" / "prod_00"),
            participant_name="site-1",
        )
        cert_builder = PrebuiltCertBuilder(
            cert_path=str(tmp_path / "site-1.crt"),
            key_path=str(tmp_path / "site-1.key"),
            rootca_path=str(tmp_path / "rootCA.pem"),
            target_name="site-1",
        )

        builders = _build_package_builders([original_workspace_builder], cert_builder, "grpc", fixed_workspace_builder)

        assert builders[0] is fixed_workspace_builder
        assert builders[1] is cert_builder
        assert fixed_workspace_builder.template_files == "custom_template.yml"
        assert original_workspace_builder not in builders

    def test_signed_zip_returns_build_error_without_success_output(self, tmp_path, monkeypatch):
        signed_zip, request_dir, _ = _make_signed_zip(tmp_path)
        args = _signed_zip_args(signed_zip, tmp_path, request_dir=str(request_dir))
        output_ok = unittest.mock.Mock()
        monkeypatch.setattr("nvflare.tool.package.package_commands.output_ok", output_ok)
        monkeypatch.setattr(
            "nvflare.tool.package.package_commands._build_selected_participant_package", lambda **kwargs: 1
        )

        result = handle_package(args)

        assert result == 1
        output_ok.assert_not_called()

    def test_signed_zip_member_limited_returns_none_when_error_is_mocked(self, tmp_path, monkeypatch):
        signed_zip = tmp_path / "site-3.signed.zip"
        with zipfile.ZipFile(signed_zip, "w") as zf:
            zf.writestr("signed.json", b"x" * 513)
        monkeypatch.setattr("nvflare.tool.package.package_commands._MAX_ZIP_MEMBER_SIZE", 512)

        with zipfile.ZipFile(signed_zip, "r") as zf:
            with unittest.mock.patch("nvflare.tool.package.package_commands.output_error_message") as output_error:
                content = _read_zip_member_limited(zf, "signed.json", str(signed_zip))

        assert content is None
        output_error.assert_called_once()

    def test_write_materialized_signed_files_returns_false_when_error_is_mocked(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "nvflare.tool.package.package_commands._write_file_nofollow",
            unittest.mock.Mock(side_effect=OSError("blocked")),
        )
        error = unittest.mock.Mock()
        monkeypatch.setattr("nvflare.tool.package.package_commands.output_error", error)

        result = _write_materialized_signed_files(
            str(tmp_path / "site-3"),
            {"name": "site-3"},
            {"signed": True},
            {"name": "site-3"},
            {"cert": b"cert", "rootCA.pem": b"rootca"},
        )

        assert result is False
        assert error.call_args.args[0] == "OUTPUT_DIR_NOT_WRITABLE"

    def test_signed_zip_returns_when_materialized_files_fail(self, tmp_path, monkeypatch):
        signed_zip, request_dir, _ = _make_signed_zip(tmp_path)
        args = _signed_zip_args(signed_zip, tmp_path, request_dir=str(request_dir))
        build = unittest.mock.Mock()
        monkeypatch.setattr(
            "nvflare.tool.package.package_commands._write_materialized_signed_files", lambda *a, **k: False
        )
        monkeypatch.setattr("nvflare.tool.package.package_commands._build_selected_participant_package", build)

        result = handle_package(args)

        assert result == 1
        build.assert_not_called()

    def test_signed_zip_accepts_expected_fingerprint_without_prompting(self, tmp_path, monkeypatch):
        signed_zip, request_dir, _ = _make_signed_zip(tmp_path)
        expected = _rootca_fingerprint_from_signed_zip(signed_zip).replace("SHA256:", "sha256 Fingerprint=")
        args = _signed_zip_args(
            signed_zip,
            tmp_path,
            request_dir=str(request_dir),
            expected_fingerprint=expected,
        )

        handle_package(args)

    def test_signed_zip_warns_when_rootca_fingerprint_is_not_verified(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "json")
        signed_zip, request_dir, _ = _make_signed_zip(tmp_path)
        args = _signed_zip_args(signed_zip, tmp_path, request_dir=str(request_dir))

        assert handle_package(args) == 0

        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert payload["status"] == "ok"
        assert "Root CA SHA256 fingerprint was not verified" in captured.err

    def test_signed_zip_rejects_mismatched_expected_fingerprint(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        signed_zip, request_dir, _ = _make_signed_zip(tmp_path)
        args = _signed_zip_args(
            signed_zip,
            tmp_path,
            request_dir=str(request_dir),
            expected_fingerprint="SHA256:" + ":".join(["00"] * 32),
        )

        with pytest.raises(SystemExit) as exc_info:
            handle_package(args)

        assert exc_info.value.code == 4
        assert "ROOTCA_FINGERPRINT_MISMATCH" in capsys.readouterr().err
        assert not (request_dir / "signed.json").exists()
        assert not (request_dir / "site-3.crt").exists()
        assert not (request_dir / "rootCA.pem").exists()

    def test_signed_zip_rejects_invalid_expected_fingerprint(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        signed_zip, request_dir, _ = _make_signed_zip(tmp_path)
        args = _signed_zip_args(
            signed_zip,
            tmp_path,
            request_dir=str(request_dir),
            expected_fingerprint="not-a-fingerprint",
        )

        with pytest.raises(SystemExit) as exc_info:
            handle_package(args)

        assert exc_info.value.code == 4
        assert "INVALID_ROOTCA_FINGERPRINT" in capsys.readouterr().err

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

    @pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlink support required")
    def test_signed_zip_ignores_symlinked_audit_request_dir(self, tmp_path, monkeypatch):
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
                    "request_id": "11111111111111111111111111111111",
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
        request_link = tmp_path / "request-link"
        os.symlink(str(tampered_request_dir), str(request_link))
        audit_dir = tmp_path / "home" / ".nvflare" / "cert_requests" / "11111111111111111111111111111111"
        audit_dir.mkdir(parents=True)
        (audit_dir / "audit.json").write_text(json.dumps({"schema_version": "1", "request_dir": str(request_link)}))
        args = _signed_zip_args(sibling_signed_zip, tmp_path)

        handle_package(args)

        kit_dir = os.path.join(str(tmp_path / "ws"), "example_project", "prod_00", "site-3")
        assert os.path.isfile(os.path.join(kit_dir, "startup", "client.key"))

    @pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlink support required")
    def test_audit_request_dir_rejects_symlinked_parent(self, tmp_path):
        real_parent = tmp_path / "real-parent"
        request_dir = real_parent / "site-3"
        request_dir.mkdir(parents=True)
        (request_dir / "request.json").write_text(json.dumps({"request_id": "1" * 32}))
        link_parent = tmp_path / "link-parent"
        os.symlink(str(real_parent), str(link_parent))

        result = _validated_audit_request_dir(str(link_parent / "site-3"), "1" * 32)

        assert result is None

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

    def test_signed_zip_explicit_request_dir_missing_key_returns_targeted_error(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        signed_zip, request_dir, _ = _make_signed_zip(tmp_path)
        incomplete_dir = tmp_path / "incomplete-request"
        incomplete_dir.mkdir()
        shutil.copy2(str(request_dir / "request.json"), str(incomplete_dir / "request.json"))
        args = _signed_zip_args(signed_zip, tmp_path, request_dir=str(incomplete_dir))

        with pytest.raises(SystemExit) as exc_info:
            handle_package(args)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "REQUEST_DIR_INCOMPLETE" in captured.err
        assert "site-3.key" in captured.err
        assert "REQUEST_DIR_NOT_FOUND" not in captured.err

    def test_signed_zip_explicit_request_dir_error_emits_once_when_error_is_mocked(self, tmp_path):
        signed_zip, request_dir, _ = _make_signed_zip(tmp_path)
        incomplete_dir = tmp_path / "incomplete-request"
        incomplete_dir.mkdir()
        shutil.copy2(str(request_dir / "request.json"), str(incomplete_dir / "request.json"))
        args = _signed_zip_args(signed_zip, tmp_path, request_dir=str(incomplete_dir))

        with unittest.mock.patch("nvflare.tool.package.package_commands.output_error_message") as output_error:
            result = handle_package(args)

        assert result == 1
        output_error.assert_called_once()
        assert output_error.call_args.args[0] == "REQUEST_DIR_INCOMPLETE"

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

    def test_signed_zip_rejects_missing_signed_metadata_signature(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        signed_zip, request_dir, _ = _make_signed_zip(tmp_path)
        with zipfile.ZipFile(signed_zip, "r") as zf:
            contents = {name: zf.read(name) for name in zf.namelist() if name != "signed.json.sig"}
        with zipfile.ZipFile(signed_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for name, content in contents.items():
                zf.writestr(name, content)
        args = _signed_zip_args(signed_zip, tmp_path, request_dir=str(request_dir))

        with pytest.raises(SystemExit) as exc_info:
            handle_package(args)

        assert exc_info.value.code == 4
        assert "missing required file(s): signed.json.sig" in capsys.readouterr().err
        assert not (request_dir / "signed.json").exists()

    def test_signed_zip_rejects_tampered_signed_metadata_signature(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        signed_zip, request_dir, _ = _make_signed_zip(tmp_path)
        with zipfile.ZipFile(signed_zip, "r") as zf:
            contents = {name: zf.read(name) for name in zf.namelist()}
        signed_meta = json.loads(contents["signed.json"])
        signed_meta["server"]["host"] = "attacker.example.com"
        contents["signed.json"] = json.dumps(signed_meta, sort_keys=True).encode("utf-8")
        with zipfile.ZipFile(signed_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for name, content in contents.items():
                zf.writestr(name, content)
        args = _signed_zip_args(signed_zip, tmp_path, request_dir=str(request_dir))

        with pytest.raises(SystemExit) as exc_info:
            handle_package(args)

        assert exc_info.value.code == 4
        assert "Signed approval metadata signature is invalid" in capsys.readouterr().err
        assert not (request_dir / "signed.json").exists()

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
            lambda _zf, _zip_path: ["signed.json", "signed.json.sig", "site.yaml", "rootCA.pem", "site-3.crt"],
        )
        args = _signed_zip_args(signed_zip, tmp_path, request_dir=str(request_dir))

        with pytest.raises(SystemExit) as exc_info:
            handle_package(args)

        assert exc_info.value.code == 4
        assert "Signed zip member exceeds size limit" in capsys.readouterr().err

    def test_load_signed_zip_member_size_error_emits_once_when_error_is_mocked(self, tmp_path, monkeypatch):
        signed_zip, _request_dir, _ = _make_signed_zip(tmp_path)
        with zipfile.ZipFile(signed_zip, "r") as zf:
            contents = {name: zf.read(name) for name in zf.namelist()}
        contents["signed.json"] = b"x" * 513
        with zipfile.ZipFile(signed_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for name, content in contents.items():
                zf.writestr(name, content)
        monkeypatch.setattr("nvflare.tool.package.package_commands._MAX_ZIP_MEMBER_SIZE", 512)
        monkeypatch.setattr(
            "nvflare.tool.package.package_commands._safe_zip_names",
            lambda _zf, _zip_path: ["signed.json", "signed.json.sig", "site.yaml", "rootCA.pem", "site-3.crt"],
        )

        with unittest.mock.patch("nvflare.tool.package.package_commands.output_error_message") as output_error:
            signed_meta, site_meta, cert_name, file_contents = _load_signed_zip(str(signed_zip))

        output_error.assert_called_once()
        assert "Signed zip member exceeds size limit" in output_error.call_args.args[1]
        assert signed_meta is None
        assert site_meta is None
        assert cert_name is None
        assert file_contents == {}

    def test_missing_signed_zip_uses_registered_error_in_dev_mode(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        monkeypatch.setenv("NVFLARE_DEV", "1")
        args = _signed_zip_args(tmp_path / "missing.signed.zip", tmp_path)

        with pytest.raises(SystemExit) as exc_info:
            handle_package(args)

        assert exc_info.value.code == 1
        assert "Signed zip not found" in capsys.readouterr().err

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
        assert "REQUEST_DIR_INCOMPLETE" in captured.err
        assert "site-3.key" in captured.err
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

    def test_signed_zip_missing_cert_returns_cleanly_when_error_is_mocked(self, tmp_path):
        signed_zip, request_dir, _ = _make_signed_zip(tmp_path)
        with zipfile.ZipFile(signed_zip, "r") as zf:
            members = {name: zf.read(name) for name in zf.namelist() if not name.endswith(".crt")}
        with zipfile.ZipFile(signed_zip, "w") as zf:
            for name, content in members.items():
                zf.writestr(name, content)
        args = _signed_zip_args(signed_zip, tmp_path, request_dir=str(request_dir))

        with unittest.mock.patch("nvflare.tool.package.package_commands.output_error_message") as output_error:
            rc = handle_package(args)

        assert rc == 1
        output_error.assert_called_once()
        assert "exactly one signed certificate" in output_error.call_args.args[1]

    def test_load_signed_zip_returns_after_missing_cert_when_error_is_mocked(self, tmp_path):
        signed_zip, _request_dir, _ = _make_signed_zip(tmp_path)
        with zipfile.ZipFile(signed_zip, "r") as zf:
            members = {name: zf.read(name) for name in zf.namelist() if not name.endswith(".crt")}
        with zipfile.ZipFile(signed_zip, "w") as zf:
            for name, content in members.items():
                zf.writestr(name, content)

        with unittest.mock.patch("nvflare.tool.package.package_commands.output_error_message") as output_error:
            signed_meta, site_meta, cert_name, file_contents = _load_signed_zip(str(signed_zip))

        output_error.assert_called_once()
        assert signed_meta is None
        assert site_meta is None
        assert cert_name == ""
        assert file_contents == {}

    def test_signed_zip_cert_load_failure_returns_cleanly_when_error_is_mocked(self, tmp_path):
        signed_zip, request_dir, _ = _make_signed_zip(tmp_path)
        args = _signed_zip_args(signed_zip, tmp_path, request_dir=str(request_dir))

        with unittest.mock.patch(
            "nvflare.tool.package.package_commands._load_crt_nofollow", side_effect=ValueError("bad cert")
        ):
            with unittest.mock.patch("nvflare.tool.package.package_commands.output_error_message") as output_error:
                rc = handle_package(args)

        assert rc == 1
        output_error.assert_called_once()
        assert "Failed to load certificate from signed zip" in output_error.call_args.args[1]

    def test_signed_zip_missing_identity_name_returns_cleanly_when_error_is_mocked(self, tmp_path):
        signed_zip, request_dir, _ = _make_signed_zip(tmp_path)
        args = _signed_zip_args(signed_zip, tmp_path, request_dir=str(request_dir))

        with unittest.mock.patch(
            "nvflare.tool.package.package_commands._signed_identity_from_metadata", return_value={"name": None}
        ):
            with unittest.mock.patch("nvflare.tool.package.package_commands.output_error_message") as output_error:
                rc = handle_package(args)

        assert rc == 1
        output_error.assert_called_once()
        assert "does not identify a participant name" in output_error.call_args.args[1]

    def test_signed_zip_cert_name_mismatch_returns_cleanly_when_error_is_mocked(self, tmp_path):
        signed_zip, request_dir, _ = _make_signed_zip(tmp_path)

        def _mutate(metadata):
            metadata["name"] = "other-site"

        _rewrite_signed_zip_metadata(signed_zip, _mutate)
        args = _signed_zip_args(signed_zip, tmp_path, request_dir=str(request_dir))

        with unittest.mock.patch("nvflare.tool.package.package_commands._validate_signed_metadata"):
            with unittest.mock.patch("nvflare.tool.package.package_commands.output_error_message") as output_error:
                rc = handle_package(args)

        assert rc == 1
        output_error.assert_called_once()
        assert "does not match participant" in output_error.call_args.args[1]

    def test_read_local_request_metadata_returns_none_after_invalid_json_when_error_is_mocked(self, tmp_path):
        request_dir = tmp_path / "site-3"
        request_dir.mkdir()
        (request_dir / "request.json").write_text("not json")

        with unittest.mock.patch("nvflare.tool.package.package_commands.output_error_message") as output_error:
            request_meta = _read_local_request_metadata(str(request_dir))

        assert request_meta is None
        output_error.assert_called_once()
        assert output_error.call_args.args[0] == "REQUEST_METADATA_INVALID"

    def test_validate_local_request_metadata_returns_after_missing_hash_when_error_is_mocked(self):
        request_meta = {
            "request_id": "1" * 32,
            "project": "example_project",
            "name": "site-3",
            "org": "nvidia",
            "kind": "site",
            "cert_type": "client",
            "csr_sha256": "1" * 64,
        }
        signed_meta = {
            "request_id": "1" * 32,
            "project": "example_project",
            "name": "site-3",
            "org": "nvidia",
            "kind": "site",
            "cert_type": "client",
            "cert_role": None,
            "hashes": {
                "csr_sha256": "1" * 64,
                "public_key_sha256": "1" * 64,
            },
        }

        with unittest.mock.patch("nvflare.tool.package.package_commands._request_metadata_mismatch") as mismatch:
            valid = _validate_local_request_metadata(request_meta, signed_meta)

        mismatch.assert_called_once()
        assert valid is False
        assert "public_key_sha256" in mismatch.call_args.args[0]

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

    def test_signed_public_key_hash_helper_returns_after_missing_hash_when_error_is_mocked(self, tmp_path):
        ca_dir = tmp_path / "ca"
        ca_dir.mkdir()
        ca_key, ca_cert, _ = _make_ca(str(ca_dir))
        _key_path, cert_path = _make_signed_cert(ca_key, ca_cert, "site-3", str(tmp_path), "site-3.crt", role="client")
        cert = load_crt(cert_path)

        with unittest.mock.patch("nvflare.tool.package.package_commands.output_error_message") as output_error:
            valid = _validate_signed_public_key_hash({}, cert)

        output_error.assert_called_once()
        assert valid is False
        assert "Missing required public key hash" in output_error.call_args.args[1]

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

    def test_signed_zip_unknown_cert_type_returns_after_error_when_error_is_mocked(self, tmp_path):
        signed_zip, request_dir, _ = _make_signed_zip(tmp_path)
        args = _signed_zip_args(signed_zip, tmp_path, request_dir=str(request_dir))

        with unittest.mock.patch("nvflare.tool.package.package_commands._read_cert_type_from_cert", return_value=""):
            with unittest.mock.patch("nvflare.tool.package.package_commands.output_error") as output_error:
                with unittest.mock.patch(
                    "nvflare.tool.package.package_commands.output_error_message"
                ) as output_error_message:
                    rc = handle_package(args)

        assert rc == 1
        output_error.assert_called_once()
        assert output_error.call_args.args[0] == "CERT_TYPE_UNKNOWN"
        output_error_message.assert_not_called()

    def test_signed_zip_public_key_validator_failure_returns_before_provision(self, tmp_path):
        signed_zip, request_dir, _ = _make_signed_zip(tmp_path)
        args = _signed_zip_args(signed_zip, tmp_path, request_dir=str(request_dir))

        with unittest.mock.patch(
            "nvflare.tool.package.package_commands._validate_signed_public_key_hash", return_value=False
        ) as validate_hash:
            with unittest.mock.patch.object(Provisioner, "provision") as provision:
                rc = handle_package(args)

        assert rc == 1
        validate_hash.assert_called_once()
        provision.assert_not_called()

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

    def test_validate_cert_material_expired_cert_does_not_check_key_when_error_is_mocked(self, tmp_path, monkeypatch):
        ca_dir = tmp_path / "ca"
        ca_dir.mkdir()
        ca_key, ca_cert, rootca_path = _make_ca(str(ca_dir))
        key_path, cert_path = _make_signed_cert(ca_key, ca_cert, "site-3", str(tmp_path), "site-3.crt", role="client")

        import nvflare.tool.package.package_commands as package_commands

        real_load_crt = package_commands._load_crt_nofollow
        past = datetime.datetime(2000, 1, 1, tzinfo=datetime.timezone.utc)

        def patched_load_crt(path):
            cert = real_load_crt(path)
            if path == cert_path:
                mock_cert = unittest.mock.MagicMock(wraps=cert)
                mock_cert.not_valid_after_utc = past
                return mock_cert
            return cert

        error = unittest.mock.Mock()
        monkeypatch.setattr("nvflare.tool.package.package_commands.output_error", error)
        monkeypatch.setattr("nvflare.tool.package.package_commands._load_crt_nofollow", patched_load_crt)
        monkeypatch.setattr("nvflare.tool.package.package_commands.verify_cert", lambda _cert, _ca_public_key: None)
        key_loader = unittest.mock.Mock()
        monkeypatch.setattr("nvflare.tool.package.package_commands._load_private_key_nofollow", key_loader)

        cert = _validate_cert_material(cert_path, key_path, rootca_path, validate_key_match=True)

        assert cert is None
        error.assert_called_once()
        assert error.call_args.args[0] == "CERT_EXPIRED"
        key_loader.assert_not_called()

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
