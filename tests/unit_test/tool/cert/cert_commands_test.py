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

"""Unit tests for nvflare cert command handlers: init, csr, sign."""

import argparse
import datetime
import hashlib
import json
import os
import platform
import stat
import sys
import zipfile
from unittest.mock import patch

import pytest
import yaml
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.extensions import BasicConstraints
from cryptography.x509.oid import NameOID

# Ensure parsers are initialized by importing cert_cli (registers module-level parser refs)
import nvflare.tool.cert.cert_cli  # noqa: F401
from nvflare.lighter.impl.cert import CertBuilder
from nvflare.lighter.utils import load_crt, load_private_key_file, serialize_cert
from nvflare.tool import cli_output
from nvflare.tool.cert.cert_commands import (
    _generate_csr,
    _load_and_validate_csr,
    _load_single_site_yaml,
    _load_yaml_file,
    _read_request_zip,
    _read_zip_member_limited,
    _resolve_sign_cert_type,
    _safe_zip_names,
    _server_cert_san_fields,
    _UnsafeZipSourceError,
    _validate_identity_name,
    _validate_request_id,
    _validate_request_kind_cert_type,
    _validate_request_metadata,
    _validate_request_project_matches_ca,
    _validate_safe_cert_name,
    _validate_safe_project_name,
    _validate_signing_ca,
    _write_file_nofollow,
    _write_json_file,
    _write_private_key,
    _write_zip_nofollow,
    generate_csr_files,
    handle_cert_csr,
    handle_cert_init,
    handle_cert_request,
    handle_cert_sign,
    sign_csr_files,
)
from nvflare.tool.cert.fingerprint import cert_fingerprint_sha256

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _init_args(**kwargs):
    defaults = dict(
        profile=None,
        output_dir=None,
        org=None,
        valid_days=3650,
        force=False,
        schema=False,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _csr_args(**kwargs):
    defaults = dict(
        name="hospital-1",
        output_dir=None,
        org=None,
        cert_type=None,
        project_file=None,
        force=False,
        schema=False,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _request_args(**kwargs):
    defaults = dict(
        participant=None,
        output_dir=None,
        org=None,
        project=None,
        name=None,
        cert_type=None,
        force=False,
        schema=False,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _sign_args(**kwargs):
    defaults = dict(
        csr_path=None,
        ca_dir=None,
        output_dir=None,
        cert_type=None,
        accept_csr_role=False,
        valid_days=1095,
        force=False,
        schema=False,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _make_profile(directory, project="example_project") -> str:
    """Write a minimal project_profile.yaml and return its path string."""
    path = os.path.join(str(directory), "project_profile.yaml")
    with open(path, "w") as f:
        f.write(yaml.safe_dump({"name": project}, sort_keys=False))
    return path


def _run_init(tmp_path, project="TestProject", **kwargs):
    """Run cert init in tmp_path; returns return code.

    Creates a temporary project_profile.yaml with the given project name so
    tests do not need to manage profile files manually.
    """
    profile_path = tmp_path / "project_profile.yaml"
    if not profile_path.exists():
        _write_participant_definition(profile_path, {"name": project})
    kwargs.setdefault("profile", str(profile_path))
    args = _init_args(output_dir=str(tmp_path), **kwargs)
    return handle_cert_init(args)


def _run_csr(tmp_path, name="hospital-1", **kwargs):
    kwargs.setdefault("cert_type", "client")
    args = _csr_args(name=name, output_dir=str(tmp_path), **kwargs)
    return handle_cert_csr(args)


def _load_csr_file(path):
    with open(path, "rb") as f:
        return x509.load_pem_x509_csr(f.read(), default_backend())


@pytest.mark.skipif(platform.system() == "Windows", reason="chmod not meaningful on Windows")
def test_write_file_nofollow_sets_requested_mode_despite_umask(tmp_path):
    path = tmp_path / "rootCA.pem"
    old_umask = os.umask(0o077)
    try:
        _write_file_nofollow(str(path), b"public cert", mode=0o644)
    finally:
        os.umask(old_umask)

    assert stat.S_IMODE(os.stat(path).st_mode) == 0o644


def test_write_json_file_removes_created_file_when_fchmod_fails(tmp_path):
    path = tmp_path / "request.json"
    with patch("nvflare.tool.cert.cert_commands.os.fchmod", side_effect=OSError("chmod failed")):
        with pytest.raises(OSError):
            _write_json_file(str(path), {"ok": True})

    assert not path.exists()


def test_write_json_file_removes_created_file_when_json_dump_fails(tmp_path):
    path = tmp_path / "request.json"
    with patch("nvflare.tool.cert.cert_commands.json.dump", side_effect=OSError("disk full")):
        with pytest.raises(OSError):
            _write_json_file(str(path), {"ok": True})

    assert not path.exists()


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
def test_write_file_nofollow_removes_created_file_when_write_fails(tmp_path):
    path = tmp_path / "rootCA.pem"

    def failing_fdopen(fd, _mode):
        return _FailingFdOpen(fd)

    with patch("nvflare.tool.cert.cert_commands.os.fdopen", side_effect=failing_fdopen):
        with pytest.raises(OSError):
            _write_file_nofollow(str(path), b"public cert")

    assert not path.exists()


@pytest.mark.skipif(platform.system() == "Windows", reason="unlinking open files differs on Windows")
def test_write_private_key_removes_created_file_when_write_fails(tmp_path):
    path = tmp_path / "site-3.key"

    def failing_fdopen(fd, _mode):
        return _FailingFdOpen(fd)

    with patch("nvflare.tool.cert.cert_commands.os.fdopen", side_effect=failing_fdopen):
        with pytest.raises(OSError):
            _write_private_key(str(path), b"private key")

    assert not path.exists()


def test_write_private_key_forces_owner_only_mode(tmp_path):
    path = tmp_path / "site-3.key"

    _write_private_key(str(path), b"private key")

    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600


# ---------------------------------------------------------------------------
# cert init tests
# ---------------------------------------------------------------------------


class TestCertValidationHelpers:
    def test_safe_cert_name_returns_after_length_error_when_error_is_mocked(self):
        with patch("nvflare.tool.cert.cert_commands.output_error") as output_error:
            valid = _validate_safe_cert_name("a" * 65, field_label="Name")

        output_error.assert_called_once()
        assert valid is False
        assert "64 characters" in output_error.call_args.kwargs["reason"]

    def test_safe_project_name_returns_after_separator_error_when_error_is_mocked(self):
        with patch("nvflare.tool.cert.cert_commands.output_error") as output_error:
            valid = _validate_safe_project_name("../escape")

        output_error.assert_called_once()
        assert valid is False
        assert "path separators" in output_error.call_args.kwargs["reason"]

    def test_safe_cert_name_returns_after_pattern_error_when_error_is_mocked(self):
        with patch("nvflare.tool.cert.cert_commands.output_error") as output_error:
            valid = _validate_safe_cert_name("bad:name", field_label="Name")

        output_error.assert_called_once()
        assert valid is False
        assert "must match" in output_error.call_args.kwargs["reason"]

    def test_safe_project_name_returns_after_pattern_error_when_error_is_mocked(self):
        with patch("nvflare.tool.cert.cert_commands.output_error") as output_error:
            valid = _validate_safe_project_name("bad:name")

        output_error.assert_called_once()
        assert valid is False
        assert "must match" in output_error.call_args.kwargs["reason"]

    def test_request_id_returns_false_after_error_when_error_is_mocked(self):
        with patch("nvflare.tool.cert.cert_commands.output_error_message") as output_error:
            valid = _validate_request_id("bad-request-id")

        output_error.assert_called_once()
        assert valid is False

    def test_identity_name_returns_false_after_error_when_error_is_mocked(self):
        with patch("nvflare.tool.cert.cert_commands.output_error_message") as output_error:
            valid = _validate_identity_name("not-an-email", "lead")

        output_error.assert_called_once()
        assert valid is False

    def test_safe_cert_name_returns_false_after_error_when_error_is_mocked(self):
        with patch("nvflare.tool.cert.cert_commands.output_error") as output_error:
            valid = _validate_safe_cert_name("../site-3", field_label="Name")

        output_error.assert_called_once()
        assert valid is False

    def test_request_zip_not_file_returns_none_when_error_is_mocked(self, tmp_path):
        request_zip = tmp_path / "site-3.request.zip"
        request_zip.mkdir()
        extract_dir = tmp_path / "extract"
        extract_dir.mkdir()

        with patch("nvflare.tool.cert.cert_commands.output_error_message") as output_error:
            request_meta = _read_request_zip(str(request_zip), str(extract_dir))

        output_error.assert_called_once()
        assert request_meta is None

    def test_request_zip_invalid_name_returns_none_when_error_is_mocked(self, tmp_path):
        request_zip = tmp_path / "site-3.request.zip"
        with zipfile.ZipFile(request_zip, "w") as zf:
            zf.writestr("request.json", json.dumps({"name": "bad:name"}))
            zf.writestr("site.yaml", "name: site-3\norg: nvidia\ntype: client\n")
            zf.writestr("bad:name.csr", "csr")
        extract_dir = tmp_path / "extract"
        extract_dir.mkdir()

        with patch("nvflare.tool.cert.cert_commands.output_error") as output_error:
            request_meta = _read_request_zip(str(request_zip), str(extract_dir))

        output_error.assert_called_once()
        assert request_meta is None

    def test_single_site_yaml_returns_after_invalid_type_when_error_is_mocked(self, tmp_path):
        site_yaml = tmp_path / "site.yaml"
        site_yaml.write_text("name: site-3\norg: nvidia\ntype: workspace\n")
        with patch("nvflare.tool.cert.cert_commands.output_error_message") as output_error:
            result = _load_single_site_yaml(str(site_yaml))

        assert result is None
        output_error.assert_called_once()
        assert "invalid cert type" in output_error.call_args.kwargs["detail"]

    def test_request_zip_safe_names_do_not_return_unsafe_entries_when_error_is_mocked(self, tmp_path):
        zip_path = tmp_path / "request.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("request.json", "{}")
            zf.writestr("../escape.csr", "csr")
            zf.writestr("site-3.key", "key")

        with zipfile.ZipFile(zip_path, "r") as zf:
            with patch("nvflare.tool.cert.cert_commands.output_error_message") as output_error:
                names = _safe_zip_names(zf)

        output_error.assert_called_once()
        assert names is None

    def test_request_metadata_returns_after_missing_fields_when_error_is_mocked(self, tmp_path):
        with patch("nvflare.tool.cert.cert_commands.output_error_message") as output_error:
            result = _validate_request_metadata(
                {"name": "site-3"}, {}, str(tmp_path / "site.yaml"), str(tmp_path / "site-3.csr")
            )

        assert result is None
        output_error.assert_called_once()
        assert "missing required field" in output_error.call_args.kwargs["detail"]

    def test_request_metadata_returns_after_artifact_mismatch_when_error_is_mocked(self, tmp_path):
        request_meta = {
            "artifact_type": "other",
            "schema_version": "1",
            "request_id": "1" * 32,
            "project": "example_project",
            "name": "site-3",
            "org": "nvidia",
            "kind": "site",
            "cert_type": "client",
            "csr_sha256": "1" * 64,
            "public_key_sha256": "1" * 64,
            "site_yaml_sha256": "1" * 64,
        }
        with patch("nvflare.tool.cert.cert_commands.output_error_message") as output_error:
            result = _validate_request_metadata(
                request_meta, {}, str(tmp_path / "site.yaml"), str(tmp_path / "site-3.csr")
            )

        assert result is None
        output_error.assert_called_once()
        assert "unsupported request artifact metadata" in output_error.call_args.kwargs["detail"]

    def test_request_metadata_returns_after_invalid_cert_type_when_error_is_mocked(self, tmp_path):
        request_meta = {
            "artifact_type": "nvflare.cert.request",
            "schema_version": "1",
            "request_id": "1" * 32,
            "project": "example_project",
            "name": "site-3",
            "org": "nvidia",
            "kind": "site",
            "cert_type": "workspace",
            "csr_sha256": "1" * 64,
            "public_key_sha256": "1" * 64,
            "site_yaml_sha256": "1" * 64,
        }
        with patch("nvflare.tool.cert.cert_commands.output_error_message") as output_error:
            result = _validate_request_metadata(
                request_meta, {}, str(tmp_path / "site.yaml"), str(tmp_path / "site-3.csr")
            )

        assert result is None
        output_error.assert_called_once()
        assert "invalid cert type" in output_error.call_args.kwargs["detail"]

    def test_write_zip_returns_after_private_key_member_when_error_is_mocked(self, tmp_path):
        key_path = tmp_path / "site-3.key"
        key_path.write_text("private key")
        request_path = tmp_path / "request.json"
        request_path.write_text("{}")
        zip_path = tmp_path / "request.zip"

        with patch("nvflare.tool.cert.cert_commands.output_error_message") as output_error:
            with patch("nvflare.tool.cert.cert_commands._read_zip_source_nofollow") as read_source:
                result = _write_zip_nofollow(
                    str(zip_path), {"site-3.key": str(key_path), "request.json": str(request_path)}
                )

        assert result is False
        output_error.assert_called_once()
        read_source.assert_not_called()
        assert not zip_path.exists()

    def test_write_zip_does_not_overwrite_existing_zip_when_error_is_mocked(self, tmp_path):
        source_path = tmp_path / "request.json"
        source_path.write_text('{"new": true}')
        zip_path = tmp_path / "request.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("sentinel.txt", "keep")

        with patch("nvflare.tool.cert.cert_commands.output_error") as output_error:
            result = _write_zip_nofollow(str(zip_path), {"request.json": str(source_path)}, force=False)

        assert result is False
        output_error.assert_called_once()
        with zipfile.ZipFile(zip_path, "r") as zf:
            assert zf.namelist() == ["sentinel.txt"]
            assert zf.read("sentinel.txt") == b"keep"

    def test_write_zip_returns_after_makedirs_error_when_error_is_mocked(self, tmp_path):
        source_path = tmp_path / "request.json"
        source_path.write_text("{}")
        zip_path = tmp_path / "missing-parent" / "request.zip"

        with patch("nvflare.tool.cert.cert_commands.os.makedirs", side_effect=OSError("blocked")):
            with patch("nvflare.tool.cert.cert_commands.output_error") as output_error:
                with patch("nvflare.tool.cert.cert_commands._read_zip_source_nofollow") as read_source:
                    result = _write_zip_nofollow(str(zip_path), {"request.json": str(source_path)}, force=False)

        assert result is False
        output_error.assert_called_once()
        assert output_error.call_args.args[0] == "OUTPUT_DIR_NOT_WRITABLE"
        read_source.assert_not_called()
        assert not zip_path.exists()

    def test_write_zip_returns_after_unsafe_source_when_error_is_mocked(self, tmp_path):
        source_path = tmp_path / "request.json"
        source_path.write_text("{}")
        zip_path = tmp_path / "request.zip"

        with patch("nvflare.tool.cert.cert_commands.output_error_message") as output_error:
            with patch(
                "nvflare.tool.cert.cert_commands._read_zip_source_nofollow",
                side_effect=_UnsafeZipSourceError("not a regular file"),
            ):
                result = _write_zip_nofollow(str(zip_path), {"request.json": str(source_path)}, force=False)

        assert result is False
        output_error.assert_called_once()
        assert "unsafe zip source" in output_error.call_args.kwargs["detail"]
        assert not zip_path.exists()

    def test_write_zip_returns_false_after_write_error_when_error_is_mocked(self, tmp_path):
        source_path = tmp_path / "request.json"
        source_path.write_text("{}")
        zip_path = tmp_path / "request.zip"

        with patch("zipfile.ZipFile.writestr", side_effect=OSError("disk full")):
            with patch("nvflare.tool.cert.cert_commands.output_error") as output_error:
                result = _write_zip_nofollow(str(zip_path), {"request.json": str(source_path)}, force=False)

        assert result is False
        output_error.assert_called_once()
        assert output_error.call_args.args[0] == "OUTPUT_DIR_NOT_WRITABLE"
        assert not zip_path.exists()

    def test_generate_csr_removes_private_key_if_csr_write_fails(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        with patch("nvflare.tool.cert.cert_commands._write_file_nofollow", side_effect=OSError("disk full")):
            with pytest.raises(SystemExit) as exc_info:
                generate_csr_files("site-3", "nvidia", "client", str(tmp_path))

        assert exc_info.value.code == 1
        assert not (tmp_path / "site-3.key").exists()

    def test_load_csr_returns_after_read_error_when_error_is_mocked(self, tmp_path):
        csr_path = tmp_path / "site-3.csr"
        csr_path.write_text("placeholder")

        with patch("nvflare.tool.cert.cert_commands._read_file_nofollow", side_effect=OSError("blocked")):
            with patch("nvflare.tool.cert.cert_commands.output_error_message") as output_error:
                csr = _load_and_validate_csr(str(csr_path))

        assert csr is None
        output_error.assert_called_once()
        assert "failed to read CSR" in output_error.call_args.kwargs["detail"]

    def test_load_yaml_file_returns_none_after_non_dict_when_error_is_mocked(self, tmp_path):
        site_yaml = tmp_path / "site.yaml"
        site_yaml.write_text("- not\n- a\n- mapping\n")

        with patch("nvflare.tool.cert.cert_commands.output_error_message") as output_error:
            data = _load_yaml_file(str(site_yaml))

        assert data is None
        output_error.assert_called_once()
        assert "yaml must be a mapping" in output_error.call_args.kwargs["detail"]

    def test_load_yaml_file_returns_after_parse_error_when_error_is_mocked(self, tmp_path):
        site_yaml = tmp_path / "site.yaml"
        site_yaml.write_text("name: [unterminated\n")

        with patch("nvflare.tool.cert.cert_commands.output_error_message") as output_error:
            data = _load_yaml_file(str(site_yaml))

        assert data is None
        output_error.assert_called_once()
        assert "failed to parse yaml" in output_error.call_args.kwargs["detail"]

    def test_generate_csr_returns_empty_after_invalid_type_when_error_is_mocked(self, tmp_path):
        with patch("nvflare.tool.cert.cert_commands.output_error_message") as output_error:
            with patch("nvflare.tool.cert.cert_commands._generate_csr") as generate_csr:
                result = generate_csr_files("site-3", "nvidia", "bad_type", str(tmp_path))

        assert result == {}
        output_error.assert_called_once()
        generate_csr.assert_not_called()

    def test_generate_csr_existing_key_does_not_overwrite_when_error_is_mocked(self, tmp_path):
        key_path = tmp_path / "site-3.key"
        key_path.write_text("existing key")

        with patch("nvflare.tool.cert.cert_commands.output_error") as output_error:
            with patch("nvflare.tool.cert.cert_commands._generate_csr") as generate_csr:
                result = generate_csr_files("site-3", "nvidia", "client", str(tmp_path))

        assert result == {}
        output_error.assert_called_once()
        assert output_error.call_args.args[0] == "KEY_ALREADY_EXISTS"
        generate_csr.assert_not_called()
        assert key_path.read_text() == "existing key"

    def test_validate_signing_ca_returns_none_when_basic_constraints_missing_and_error_is_mocked(self):
        class _Extensions:
            def get_extension_for_class(self, _extension_type):
                raise x509.ExtensionNotFound("missing", x509.BasicConstraints)

        ca_cert = argparse.Namespace(extensions=_Extensions())
        now = datetime.datetime.now(datetime.timezone.utc)

        with patch("nvflare.tool.cert.cert_commands.output_error") as output_error:
            result = _validate_signing_ca(ca_cert, now)

        assert result is None
        output_error.assert_called_once()
        assert output_error.call_args.args[0] == "CERT_SIGNING_FAILED"

    def test_read_zip_member_limited_returns_none_when_error_is_mocked(self, tmp_path, monkeypatch):
        zip_path = tmp_path / "request.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("request.json", b"x" * 513)
        monkeypatch.setattr("nvflare.tool.cert.cert_commands._MAX_ZIP_MEMBER_SIZE", 512)

        with zipfile.ZipFile(zip_path, "r") as zf:
            with patch("nvflare.tool.cert.cert_commands.output_error_message") as output_error:
                content = _read_zip_member_limited(zf, "request.json")

        assert content is None
        output_error.assert_called_once()

    def test_request_project_ca_uses_nofollow_read_for_rootca(self, tmp_path):
        ca_dir = tmp_path / "ca"
        handle_cert_init(
            _init_args(profile=_make_profile(tmp_path, "example_project"), org="nvidia", output_dir=str(ca_dir))
        )
        rootca_path = str(ca_dir / "rootCA.pem")

        from nvflare.tool.cert import cert_commands

        with patch(
            "nvflare.tool.cert.cert_commands._read_file_nofollow", wraps=cert_commands._read_file_nofollow
        ) as read_file:
            with patch("nvflare.tool.cert.cert_commands.output_error") as output_error:
                ca_meta = _validate_request_project_matches_ca(str(ca_dir), "example_project")

        assert ca_meta["project"] == "example_project"
        output_error.assert_not_called()
        read_file.assert_any_call(rootca_path)

    def test_request_project_ca_mismatch_returns_none_when_error_is_mocked(self, tmp_path):
        ca_dir = tmp_path / "ca"
        handle_cert_init(
            _init_args(profile=_make_profile(tmp_path, "example_project"), org="nvidia", output_dir=str(ca_dir))
        )

        with patch("nvflare.tool.cert.cert_commands.output_error_message") as output_error:
            ca_meta = _validate_request_project_matches_ca(str(ca_dir), "other_project")

        assert ca_meta is None
        output_error.assert_called_once()
        assert "does not match CA project" in output_error.call_args.args[1]


class TestCertInit:
    def test_basic_init(self, tmp_path):
        rc = _run_init(tmp_path)
        assert rc == 0
        assert (tmp_path / "rootCA.pem").exists()
        assert (tmp_path / "rootCA.key").exists()
        assert (tmp_path / "ca.json").exists()

    @pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlink support required")
    def test_init_rootca_symlink_destination_is_rejected(self, tmp_path):
        outside_target = tmp_path / "outside-rootca.pem"
        outside_target.write_text("sentinel")
        os.symlink(str(outside_target), str(tmp_path / "rootCA.pem"))

        with pytest.raises(SystemExit) as exc_info:
            _run_init(tmp_path)

        assert exc_info.value.code == 1
        assert outside_target.read_text() == "sentinel"
        assert not (tmp_path / "rootCA.key").exists()
        assert not (tmp_path / "ca.json").exists()

    @pytest.mark.skipif(platform.system() == "Windows", reason="chmod not meaningful on Windows")
    def test_ca_key_permissions(self, tmp_path):
        _run_init(tmp_path)
        key_path = tmp_path / "rootCA.key"
        mode = stat.S_IMODE(os.stat(str(key_path)).st_mode)
        assert mode == 0o600

    @pytest.mark.skipif(platform.system() == "Windows", reason="chmod not meaningful on Windows")
    def test_ca_json_permissions(self, tmp_path):
        _run_init(tmp_path)
        ca_json_path = tmp_path / "ca.json"
        mode = stat.S_IMODE(os.stat(str(ca_json_path)).st_mode)
        assert mode == 0o600

    def test_ca_json_content(self, tmp_path):
        _run_init(tmp_path, project="MyProject")
        with open(str(tmp_path / "ca.json")) as f:
            meta = json.load(f)
        assert meta["project"] == "MyProject"
        assert "created_at" in meta

    def test_init_uses_project_profile_name(self, tmp_path):
        profile_path = tmp_path / "project_profile.yaml"
        ca_dir = tmp_path / "ca"
        _write_participant_definition(profile_path, {"name": "ProfileProject"})

        rc = handle_cert_init(_init_args(profile=str(profile_path), output_dir=str(ca_dir)))

        assert rc == 0
        with open(str(ca_dir / "ca.json")) as f:
            meta = json.load(f)
        assert meta["project"] == "ProfileProject"
        assert meta["project_profile"] == os.path.abspath(str(profile_path))

    def test_existing_ca_no_force(self, tmp_path):
        # Pre-create rootCA.key
        (tmp_path / "rootCA.key").write_bytes(b"fake-key")
        with pytest.raises(SystemExit) as exc_info:
            _run_init(tmp_path)
        assert exc_info.value.code == 1

    def test_force_overwrites_and_backs_up(self, tmp_path):
        # First init
        _run_init(tmp_path)
        original_key = (tmp_path / "rootCA.key").read_bytes()
        # Force re-init
        rc = _run_init(tmp_path, force=True)
        assert rc == 0
        new_key = (tmp_path / "rootCA.key").read_bytes()
        # A new key pair is generated on force re-init
        assert original_key != new_key
        # .bak directory should exist
        bak_dirs = list((tmp_path / ".bak").iterdir())
        assert len(bak_dirs) >= 1

    def test_rootca_is_valid_ca_cert(self, tmp_path):
        _run_init(tmp_path)
        cert = load_crt(str(tmp_path / "rootCA.pem"))
        bc = cert.extensions.get_extension_for_class(BasicConstraints)
        assert bc.value.ca is True

    def test_rootca_is_self_signed(self, tmp_path):
        _run_init(tmp_path)
        cert = load_crt(str(tmp_path / "rootCA.pem"))
        # For a self-signed cert, subject CN == issuer CN
        subject_cn = cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value
        issuer_cn = cert.issuer.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value
        assert subject_cn == issuer_cn

    def test_org_in_cert(self, tmp_path):
        _run_init(tmp_path, org="NVIDIA")
        cert = load_crt(str(tmp_path / "rootCA.pem"))
        org_attrs = cert.subject.get_attributes_for_oid(NameOID.ORGANIZATION_NAME)
        assert len(org_attrs) == 1
        assert org_attrs[0].value == "NVIDIA"

    def test_schema_output(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["nvflare", "cert", "init", "--schema"])
        with pytest.raises(SystemExit) as exc_info:
            handle_cert_init(_init_args())
        assert exc_info.value.code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["schema_version"] == "1"
        assert data["command"] == "nvflare cert init"
        assert len(data["args"]) > 0
        args_by_name = {arg["name"]: arg for arg in data["args"]}
        assert args_by_name["--profile"]["required"] is True
        assert args_by_name["--output-dir"]["required"] is True

    def test_missing_required_args_show_help_and_missing_flags(self, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        with pytest.raises(SystemExit) as exc_info:
            handle_cert_init(_init_args(output_dir=None))
        assert exc_info.value.code == 4
        captured = capsys.readouterr()
        assert "INVALID_ARGS" in captured.err
        assert "missing required argument(s): --profile, -o/--output-dir" in captured.err
        assert "usage:" in captured.err

    def test_agent_mode_json_envelope(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "json")
        profile_path = tmp_path / "project_profile.yaml"
        _write_participant_definition(profile_path, {"name": "TestProject"})
        rc = handle_cert_init(_init_args(profile=str(profile_path), output_dir=str(tmp_path)))
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["schema_version"] == "1"
        assert data["status"] == "ok"
        assert data["exit_code"] == 0
        assert "ca_cert" in data["data"]
        assert "ca_key" not in data["data"]
        assert "project" in data["data"]
        assert "valid_until" in data["data"]

    def test_force_flag_overwrites_existing_ca(self, tmp_path):
        (tmp_path / "rootCA.key").write_bytes(b"old-key")
        rc = handle_cert_init(_init_args(profile=_make_profile(tmp_path), output_dir=str(tmp_path), force=True))
        assert rc == 0
        assert (tmp_path / "rootCA.key").exists()

    def test_output_dir_created(self, tmp_path):
        new_dir = str(tmp_path / "new" / "subdir")
        args = _init_args(profile=_make_profile(tmp_path), output_dir=new_dir)
        rc = handle_cert_init(args)
        assert rc == 0
        assert os.path.exists(new_dir)
        assert os.path.exists(os.path.join(new_dir, "rootCA.pem"))

    def test_init_returns_after_makedirs_error_when_error_is_mocked(self, tmp_path):
        with patch("nvflare.tool.cert.cert_commands.os.makedirs", side_effect=OSError("blocked")):
            with patch("nvflare.tool.cert.cert_commands.output_error") as output_error:
                with patch("nvflare.tool.cert.cert_commands.generate_keys") as generate_keys:
                    rc = handle_cert_init(_init_args(profile=_make_profile(tmp_path), output_dir=str(tmp_path / "ca")))

        assert rc == 1
        output_error.assert_called_once()
        assert output_error.call_args.args[0] == "OUTPUT_DIR_NOT_WRITABLE"
        generate_keys.assert_not_called()

    def test_init_returns_after_not_writable_when_error_is_mocked(self, tmp_path):
        ca_dir = tmp_path / "ca"
        ca_dir.mkdir()
        with patch("nvflare.tool.cert.cert_commands.os.access", return_value=False):
            with patch("nvflare.tool.cert.cert_commands.output_error") as output_error:
                with patch("nvflare.tool.cert.cert_commands.generate_keys") as generate_keys:
                    rc = handle_cert_init(_init_args(profile=_make_profile(tmp_path), output_dir=str(ca_dir)))

        assert rc == 1
        output_error.assert_called_once()
        assert output_error.call_args.args[0] == "OUTPUT_DIR_NOT_WRITABLE"
        generate_keys.assert_not_called()

    def test_init_cleans_up_partial_output_on_write_failure(self, tmp_path, monkeypatch):
        import nvflare.tool.cert.cert_commands as cert_commands

        def _fail_write_private_key(path, pem_bytes):
            raise OSError("simulated key write failure")

        monkeypatch.setattr(cert_commands, "_write_private_key", _fail_write_private_key)

        with pytest.raises(SystemExit) as exc_info:
            handle_cert_init(_init_args(profile=_make_profile(tmp_path), output_dir=str(tmp_path)))
        assert exc_info.value.code == 1
        assert not (tmp_path / "rootCA.pem").exists()
        assert not (tmp_path / "rootCA.key").exists()
        assert not (tmp_path / "ca.json").exists()

    def test_init_cleans_up_partial_key_file_on_mid_write_failure(self, tmp_path, monkeypatch):
        import nvflare.tool.cert.cert_commands as cert_commands

        def _partial_write_private_key(path, pem_bytes):
            with open(path, "wb") as f:
                f.write(pem_bytes[:32])
            raise OSError("disk full during key write")

        monkeypatch.setattr(cert_commands, "_write_private_key", _partial_write_private_key)

        with pytest.raises(SystemExit) as exc_info:
            handle_cert_init(_init_args(profile=_make_profile(tmp_path), output_dir=str(tmp_path)))
        assert exc_info.value.code == 1
        assert not (tmp_path / "rootCA.pem").exists()
        assert not (tmp_path / "rootCA.key").exists()
        assert not (tmp_path / "ca.json").exists()

    @pytest.mark.skipif(platform.system() == "Windows", reason="directory chmod semantics differ on Windows")
    def test_output_dir_permissions(self, tmp_path):
        new_dir = str(tmp_path / "secure" / "ca")
        rc = handle_cert_init(_init_args(profile=_make_profile(tmp_path), output_dir=new_dir))
        assert rc == 0
        mode = stat.S_IMODE(os.stat(new_dir).st_mode)
        assert mode == 0o700

    def test_ca_cert_subject_cn_matches_name(self, tmp_path):
        _run_init(tmp_path, project="FederationX")
        cert = load_crt(str(tmp_path / "rootCA.pem"))
        cn = cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value
        assert cn == "FederationX"

    def test_ca_cert_validity_approx_10_years(self, tmp_path):
        _run_init(tmp_path)
        cert = load_crt(str(tmp_path / "rootCA.pem"))
        try:
            delta = cert.not_valid_after_utc - cert.not_valid_before_utc
        except AttributeError:
            delta = cert.not_valid_after - cert.not_valid_before  # cryptography < 42.0
        # Should be ~3650 days
        assert delta.days >= 3640

    def test_ca_cert_valid_days_custom(self, tmp_path):
        _run_init(tmp_path, valid_days=90)
        cert = load_crt(str(tmp_path / "rootCA.pem"))
        try:
            delta = cert.not_valid_after_utc - cert.not_valid_before_utc
        except AttributeError:
            delta = cert.not_valid_after - cert.not_valid_before  # cryptography < 42.0
        assert 88 <= delta.days <= 92


# ---------------------------------------------------------------------------
# cert csr tests
# ---------------------------------------------------------------------------


class TestCertCsr:
    def test_basic_csr(self, tmp_path):
        rc = _run_csr(tmp_path, name="hospital-1")
        assert rc == 0
        assert (tmp_path / "hospital-1.key").exists()
        assert (tmp_path / "hospital-1.csr").exists()

    @pytest.mark.skipif(platform.system() == "Windows", reason="chmod not meaningful on Windows")
    def test_key_permissions(self, tmp_path):
        _run_csr(tmp_path, name="h1")
        key_path = str(tmp_path / "h1.key")
        mode = stat.S_IMODE(os.stat(key_path).st_mode)
        assert mode == 0o600

    @pytest.mark.skipif(
        not hasattr(os, "symlink") or not hasattr(os, "O_NOFOLLOW"), reason="nofollow symlink support required"
    )
    def test_csr_symlink_destination_is_rejected(self, tmp_path):
        outside_target = tmp_path / "outside.csr"
        outside_target.write_text("sentinel")
        os.symlink(str(outside_target), str(tmp_path / "h1.csr"))

        with pytest.raises(SystemExit) as exc_info:
            _run_csr(tmp_path, name="h1")

        assert exc_info.value.code == 1
        assert outside_target.read_text() == "sentinel"

    def test_no_cert_generated(self, tmp_path):
        _run_csr(tmp_path, name="h1")
        assert not (tmp_path / "h1.crt").exists()

    def test_csr_valid_x509(self, tmp_path):
        _run_csr(tmp_path, name="h1")
        csr = _load_csr_file(str(tmp_path / "h1.csr"))
        assert csr.is_signature_valid

    def test_csr_cn_matches_name(self, tmp_path):
        _run_csr(tmp_path, name="hospital-1")
        csr = _load_csr_file(str(tmp_path / "hospital-1.csr"))
        cn = csr.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value
        assert cn == "hospital-1"

    def test_email_style_name_is_allowed(self, tmp_path):
        rc = _run_csr(tmp_path, name="admin@nvidia.com")
        assert rc == 0
        assert (tmp_path / "admin@nvidia.com.key").exists()
        assert (tmp_path / "admin@nvidia.com.csr").exists()

    def test_whitespace_only_name_is_rejected(self, tmp_path):
        with pytest.raises(SystemExit) as exc_info:
            handle_cert_csr(_csr_args(name="   ", output_dir=str(tmp_path), cert_type="client"))
        assert exc_info.value.code == 4

    def test_csr_role_embedded_in_subject(self, tmp_path):
        """cert csr requires a proposed role and embeds it in the CSR."""
        _run_csr(tmp_path, name="h1")
        csr = _load_csr_file(str(tmp_path / "h1.csr"))
        role_attrs = csr.subject.get_attributes_for_oid(NameOID.UNSTRUCTURED_NAME)
        assert len(role_attrs) == 1
        assert role_attrs[0].value == "client"

    def test_org_in_subject(self, tmp_path):
        args = _csr_args(name="h1", output_dir=str(tmp_path), org="ACME", cert_type="client")
        handle_cert_csr(args)
        csr = _load_csr_file(str(tmp_path / "h1.csr"))
        org_attrs = csr.subject.get_attributes_for_oid(NameOID.ORGANIZATION_NAME)
        assert len(org_attrs) == 1
        assert org_attrs[0].value == "ACME"

    def test_no_org_in_subject(self, tmp_path):
        _run_csr(tmp_path, name="h1")
        csr = _load_csr_file(str(tmp_path / "h1.csr"))
        org_attrs = csr.subject.get_attributes_for_oid(NameOID.ORGANIZATION_NAME)
        assert len(org_attrs) == 0

    def test_existing_key_no_force(self, tmp_path):
        (tmp_path / "h1.key").write_bytes(b"old-key")
        with pytest.raises(SystemExit) as exc_info:
            _run_csr(tmp_path, name="h1")
        assert exc_info.value.code == 1

    def test_existing_key_with_force(self, tmp_path):
        (tmp_path / "h1.key").write_bytes(b"old-key")
        rc = _run_csr(tmp_path, name="h1", force=True)
        assert rc == 0
        # New key should be valid PEM
        key_bytes = (tmp_path / "h1.key").read_bytes()
        assert b"PRIVATE KEY" in key_bytes
        # Old key should be in .bak
        bak_dirs = list((tmp_path / ".bak").iterdir())
        assert len(bak_dirs) >= 1

    def test_force_backs_up_both_files(self, tmp_path):
        (tmp_path / "h1.key").write_bytes(b"old-key")
        (tmp_path / "h1.csr").write_bytes(b"old-csr")
        _run_csr(tmp_path, name="h1", force=True)
        bak_dirs = list((tmp_path / ".bak").iterdir())
        assert len(bak_dirs) >= 1
        bak_dir = bak_dirs[0]
        assert (bak_dir / "h1.key").exists()
        assert (bak_dir / "h1.csr").exists()

    def test_force_flag_overwrites_existing_key(self, tmp_path, capsys):
        (tmp_path / "h1.key").write_bytes(b"old-key")
        rc = handle_cert_csr(_csr_args(name="h1", output_dir=str(tmp_path), cert_type="client", force=True))
        assert rc == 0
        capsys.readouterr()  # discard output
        assert b"PRIVATE KEY" in (tmp_path / "h1.key").read_bytes()

    def test_output_dir_created(self, tmp_path):
        new_dir = str(tmp_path / "newdir")
        rc = handle_cert_csr(_csr_args(name="h1", output_dir=new_dir, cert_type="client"))
        assert rc == 0
        assert os.path.exists(new_dir)

    @pytest.mark.skipif(platform.system() == "Windows", reason="directory chmod semantics differ on Windows")
    def test_output_dir_permissions(self, tmp_path):
        new_dir = str(tmp_path / "secure-csr")
        rc = handle_cert_csr(_csr_args(name="h1", output_dir=new_dir, cert_type="client"))
        assert rc == 0
        mode = stat.S_IMODE(os.stat(new_dir).st_mode)
        assert mode == 0o700

    def test_agent_mode_json_envelope(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "json")
        handle_cert_csr(_csr_args(name="h1", output_dir=str(tmp_path), cert_type="client"))
        data = json.loads(capsys.readouterr().out)
        assert data["schema_version"] == "1"
        assert data["status"] == "ok"
        assert data["exit_code"] == 0
        assert "key" in data["data"]
        assert "csr" in data["data"]

    def test_agent_mode_no_key_material_in_output(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "json")
        handle_cert_csr(_csr_args(name="h1", output_dir=str(tmp_path), cert_type="client"))
        out = capsys.readouterr().out
        assert "BEGIN RSA PRIVATE KEY" not in out
        assert "BEGIN PRIVATE KEY" not in out

    def test_missing_required_args_show_help_and_missing_flags(self, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        with pytest.raises(SystemExit) as exc_info:
            handle_cert_csr(_csr_args(name=None, output_dir=None))
        assert exc_info.value.code == 4
        captured = capsys.readouterr()
        assert "INVALID_ARGS" in captured.err
        assert "missing required argument(s): -o/--output-dir, -n/--name, -t/--type" in captured.err

    def test_missing_required_args_return_when_error_is_mocked(self, tmp_path):
        with patch("nvflare.tool.cert.cert_commands.output_error_message") as output_error:
            with patch("nvflare.tool.cert.cert_commands.generate_csr_files") as generate_csr:
                result = handle_cert_csr(_csr_args(name=None, output_dir=None))

        assert result == 1
        output_error.assert_called_once()
        generate_csr.assert_not_called()

    def test_empty_csr_result_returns_before_success_output(self, tmp_path):
        with patch("nvflare.tool.cert.cert_commands.generate_csr_files", return_value={}):
            with patch("nvflare.tool.cert.cert_commands.output_ok") as output_ok:
                result = handle_cert_csr(_csr_args(name="site-3", output_dir=str(tmp_path), cert_type="client"))

        assert result == 1
        output_ok.assert_not_called()


# ---------------------------------------------------------------------------
# cert sign tests
# ---------------------------------------------------------------------------


def _setup_ca(tmp_path):
    """Run cert init and return ca_dir path."""
    ca_dir = str(tmp_path / "ca")
    args = _init_args(profile=_make_profile(tmp_path), output_dir=ca_dir)
    handle_cert_init(args)
    return ca_dir


def _setup_csr(tmp_path, name="hospital-1"):
    """Run cert csr and return csr_path."""
    csr_dir = str(tmp_path / "csr")
    os.makedirs(csr_dir, exist_ok=True)
    args = _csr_args(name=name, output_dir=csr_dir, cert_type="client")
    handle_cert_csr(args)
    return os.path.join(csr_dir, f"{name}.csr")


def _public_key_der(public_key) -> bytes:
    return public_key.public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)


def _san_dns_and_ips(cert: x509.Certificate) -> tuple:
    san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    return san.get_values_for_type(x509.DNSName), [str(ip) for ip in san.get_values_for_type(x509.IPAddress)]


def _overwrite_ca_cert(ca_dir: str, not_before, not_after, ca: bool = True) -> None:
    ca_cert_path = os.path.join(ca_dir, "rootCA.pem")
    ca_key_path = os.path.join(ca_dir, "rootCA.key")
    original_ca = load_crt(ca_cert_path)
    ca_key = load_private_key_file(ca_key_path)

    cert = (
        x509.CertificateBuilder()
        .subject_name(original_ca.subject)
        .issuer_name(original_ca.subject)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(not_before)
        .not_valid_after(not_after)
        .add_extension(x509.BasicConstraints(ca=ca, path_length=None), critical=True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False)
        .sign(ca_key, hashes.SHA256(), default_backend())
    )

    with open(ca_cert_path, "wb") as f:
        f.write(serialize_cert(cert))


class TestCertSign:
    def test_basic_sign(self, tmp_path):
        ca_dir = _setup_ca(tmp_path)
        csr_path = _setup_csr(tmp_path)
        out_dir = str(tmp_path / "signed")
        args = _sign_args(csr_path=csr_path, ca_dir=ca_dir, output_dir=out_dir, cert_type="client")
        rc = handle_cert_sign(args)
        assert rc == 0

    def test_sign_reuses_loaded_csr(self, tmp_path):
        ca_dir = _setup_ca(tmp_path)
        csr_path = _setup_csr(tmp_path)
        out_dir = str(tmp_path / "signed")
        args = _sign_args(csr_path=csr_path, ca_dir=ca_dir, output_dir=out_dir, cert_type="client")

        from nvflare.tool.cert import cert_commands

        original_load_csr = cert_commands._load_and_validate_csr
        with patch("nvflare.tool.cert.cert_commands._load_and_validate_csr", wraps=original_load_csr) as load_csr:
            rc = handle_cert_sign(args)

        assert rc == 0
        assert load_csr.call_count == 1

    def test_sign_output_files(self, tmp_path):
        ca_dir = _setup_ca(tmp_path)
        csr_path = _setup_csr(tmp_path)  # name="hospital-1"
        out_dir = str(tmp_path / "signed")
        args = _sign_args(csr_path=csr_path, ca_dir=ca_dir, output_dir=out_dir, cert_type="client")
        handle_cert_sign(args)
        assert os.path.exists(os.path.join(out_dir, "hospital-1.crt"))
        assert os.path.exists(os.path.join(out_dir, "rootCA.pem"))

    @pytest.mark.skipif(platform.system() == "Windows", reason="directory chmod semantics differ on Windows")
    def test_sign_output_dir_permissions(self, tmp_path):
        ca_dir = _setup_ca(tmp_path)
        csr_path = _setup_csr(tmp_path)
        out_dir = str(tmp_path / "signed-secure")
        args = _sign_args(csr_path=csr_path, ca_dir=ca_dir, output_dir=out_dir, cert_type="client")
        handle_cert_sign(args)
        mode = stat.S_IMODE(os.stat(out_dir).st_mode)
        assert mode == 0o700

    def test_sign_does_not_mutate_ca_json(self, tmp_path):
        ca_dir = _setup_ca(tmp_path)
        csr_path = _setup_csr(tmp_path)
        out_dir = str(tmp_path / "signed")
        ca_json_path = os.path.join(ca_dir, "ca.json")
        with open(ca_json_path) as _f:
            meta_before = json.load(_f)
        args = _sign_args(csr_path=csr_path, ca_dir=ca_dir, output_dir=out_dir, cert_type="client")
        handle_cert_sign(args)
        with open(ca_json_path) as _f:
            meta_after = json.load(_f)
        assert meta_after == meta_before

    def test_sign_cert_is_valid_x509(self, tmp_path):
        ca_dir = _setup_ca(tmp_path)
        csr_path = _setup_csr(tmp_path)
        out_dir = str(tmp_path / "signed")
        args = _sign_args(csr_path=csr_path, ca_dir=ca_dir, output_dir=out_dir, cert_type="client")
        handle_cert_sign(args)
        cert = load_crt(os.path.join(out_dir, "hospital-1.crt"))
        assert cert is not None

    def test_sign_cert_not_ca(self, tmp_path):
        ca_dir = _setup_ca(tmp_path)
        csr_path = _setup_csr(tmp_path)
        out_dir = str(tmp_path / "signed")
        args = _sign_args(csr_path=csr_path, ca_dir=ca_dir, output_dir=out_dir, cert_type="client")
        handle_cert_sign(args)
        cert = load_crt(os.path.join(out_dir, "hospital-1.crt"))
        bc = cert.extensions.get_extension_for_class(BasicConstraints)
        assert bc.value.ca is False

    def test_sign_leaf_key_usage_is_critical(self, tmp_path):
        ca_dir = _setup_ca(tmp_path)
        csr_path = _setup_csr(tmp_path)
        out_dir = str(tmp_path / "signed")
        args = _sign_args(csr_path=csr_path, ca_dir=ca_dir, output_dir=out_dir, cert_type="client")
        handle_cert_sign(args)
        cert = load_crt(os.path.join(out_dir, "hospital-1.crt"))
        key_usage = cert.extensions.get_extension_for_class(x509.KeyUsage)
        assert key_usage.critical is True

    def test_sign_cert_type_authoritative(self, tmp_path):
        """The -t arg controls UNSTRUCTURED_NAME in signed cert; filename uses participant CN."""
        ca_dir = _setup_ca(tmp_path)
        csr_path = _setup_csr(tmp_path, name="alice")
        out_dir = str(tmp_path / "signed")
        args = _sign_args(csr_path=csr_path, ca_dir=ca_dir, output_dir=out_dir, cert_type="lead")
        handle_cert_sign(args)
        cert = load_crt(os.path.join(out_dir, "alice.crt"))
        role_attrs = cert.subject.get_attributes_for_oid(NameOID.UNSTRUCTURED_NAME)
        assert len(role_attrs) == 1
        assert role_attrs[0].value == "lead"

    def test_sign_output_filename_from_participant(self, tmp_path):
        ca_dir = _setup_ca(tmp_path)
        csr_path = _setup_csr(tmp_path, name="srv")
        out_dir = str(tmp_path / "signed")
        args = _sign_args(csr_path=csr_path, ca_dir=ca_dir, output_dir=out_dir, cert_type="server")
        handle_cert_sign(args)
        assert os.path.exists(os.path.join(out_dir, "srv.crt"))

    def test_sign_existing_cert_no_force(self, tmp_path):
        ca_dir = _setup_ca(tmp_path)
        csr_path = _setup_csr(tmp_path)
        out_dir = str(tmp_path / "signed")
        os.makedirs(out_dir, exist_ok=True)
        # Pre-create the cert file (named after participant)
        with open(os.path.join(out_dir, "hospital-1.crt"), "w"):
            pass
        args = _sign_args(csr_path=csr_path, ca_dir=ca_dir, output_dir=out_dir, cert_type="client")
        with pytest.raises(SystemExit) as exc_info:
            handle_cert_sign(args)
        assert exc_info.value.code == 1

    def test_sign_existing_cert_with_force(self, tmp_path):
        ca_dir = _setup_ca(tmp_path)
        csr_path = _setup_csr(tmp_path)
        out_dir = str(tmp_path / "signed")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "hospital-1.crt"), "w"):
            pass
        args = _sign_args(csr_path=csr_path, ca_dir=ca_dir, output_dir=out_dir, cert_type="client", force=True)
        rc = handle_cert_sign(args)
        assert rc == 0

    def test_sign_csr_not_found(self, tmp_path):
        ca_dir = _setup_ca(tmp_path)
        out_dir = str(tmp_path / "signed")
        args = _sign_args(
            csr_path="/nonexistent/path/to.csr",
            ca_dir=ca_dir,
            output_dir=out_dir,
            cert_type="client",
        )
        with pytest.raises(SystemExit) as exc_info:
            handle_cert_sign(args)
        assert exc_info.value.code == 1

    def test_sign_invalid_csr_includes_parse_detail(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        ca_dir = _setup_ca(tmp_path)
        bad_csr_path = str(tmp_path / "bad.csr")
        with open(bad_csr_path, "wb") as f:
            f.write(b"not a csr")
        out_dir = str(tmp_path / "signed")
        args = _sign_args(csr_path=bad_csr_path, ca_dir=ca_dir, output_dir=out_dir, cert_type="client")

        with pytest.raises(SystemExit) as exc_info:
            handle_cert_sign(args)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "bad.csr" in captured.err
        assert "INVALID_CSR" in captured.err
        assert "Unable to load PEM file" in captured.err or "MismatchedTags" in captured.err

    def test_sign_csr_path_must_be_file(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        ca_dir = _setup_ca(tmp_path)
        csr_dir = str(tmp_path / "csr_dir")
        os.makedirs(csr_dir, exist_ok=True)
        out_dir = str(tmp_path / "signed")
        args = _sign_args(csr_path=csr_dir, ca_dir=ca_dir, output_dir=out_dir, cert_type="client")
        with pytest.raises(SystemExit) as exc_info:
            handle_cert_sign(args)
        assert exc_info.value.code == 4
        captured = capsys.readouterr()
        assert "must be a file path, not a directory" in captured.err
        assert "INTERNAL_ERROR" not in captured.err

    @pytest.mark.skipif(
        not hasattr(os, "symlink") or not hasattr(os, "O_NOFOLLOW"), reason="nofollow symlink support required"
    )
    def test_load_csr_symlink_source_is_rejected(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        csr_path = _setup_csr(tmp_path)
        link_path = tmp_path / "linked.csr"
        os.symlink(csr_path, str(link_path))

        with pytest.raises(SystemExit) as exc_info:
            _load_and_validate_csr(str(link_path))

        assert exc_info.value.code == 4
        captured = capsys.readouterr()
        assert "failed to read CSR" in captured.err

    def test_sign_ca_dir_invalid(self, tmp_path):
        csr_path = _setup_csr(tmp_path)
        out_dir = str(tmp_path / "signed")
        args = _sign_args(
            csr_path=csr_path,
            ca_dir=str(tmp_path / "nonexistent_ca"),
            output_dir=out_dir,
            cert_type="client",
        )
        with pytest.raises(SystemExit) as exc_info:
            handle_cert_sign(args)
        assert exc_info.value.code == 1

    def test_sign_ca_load_failure_reports_specific_error(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        ca_dir = _setup_ca(tmp_path)
        csr_path = _setup_csr(tmp_path)
        with open(os.path.join(ca_dir, "rootCA.key"), "wb") as f:
            f.write(b"not a private key")
        out_dir = str(tmp_path / "signed")
        args = _sign_args(csr_path=csr_path, ca_dir=ca_dir, output_dir=out_dir, cert_type="client")

        with pytest.raises(SystemExit) as exc_info:
            handle_cert_sign(args)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "CA_LOAD_FAILED" in captured.err
        assert ca_dir in captured.err

    def test_sign_agent_mode_json_envelope(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "json")
        ca_dir = _setup_ca(tmp_path)
        csr_path = _setup_csr(tmp_path)
        capsys.readouterr()  # discard setup output
        out_dir = str(tmp_path / "signed")
        args = _sign_args(csr_path=csr_path, ca_dir=ca_dir, output_dir=out_dir, cert_type="client")
        rc = handle_cert_sign(args)
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["schema_version"] == "1"
        assert data["status"] == "ok"
        assert data["exit_code"] == 0
        assert "signed_cert" in data["data"]
        assert "rootca" in data["data"]
        assert "serial" in data["data"]
        assert isinstance(data["data"]["serial"], str)
        assert data["data"]["serial"].startswith("0x")
        assert "--dir" not in data["data"]["next_step"]
        assert "nvflare package <signed.zip>" in data["data"]["next_step"]

    def test_sign_force_overwrites_existing_cert(self, tmp_path):
        ca_dir = _setup_ca(tmp_path)
        csr_path = _setup_csr(tmp_path)
        out_dir = str(tmp_path / "signed")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "hospital-1.crt"), "w"):
            pass
        args = _sign_args(csr_path=csr_path, ca_dir=ca_dir, output_dir=out_dir, cert_type="client", force=True)
        assert handle_cert_sign(args) == 0

    def test_missing_required_args_show_help_and_missing_flags(self, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        with pytest.raises(SystemExit) as exc_info:
            handle_cert_sign(_sign_args(csr_path=None, ca_dir=None, output_dir=None))
        assert exc_info.value.code == 4
        captured = capsys.readouterr()
        assert "INVALID_ARGS" in captured.err
        assert "missing required argument(s): -r/--csr, -c/--ca-dir, -o/--output-dir" in captured.err

    def test_missing_required_args_return_when_error_is_mocked(self):
        with patch("nvflare.tool.cert.cert_commands.output_error_message") as output_error:
            with patch("nvflare.tool.cert.cert_commands._load_and_validate_csr") as load_csr:
                result = handle_cert_sign(_sign_args(csr_path=None, ca_dir=None, output_dir=None))

        assert result == 1
        output_error.assert_called_once()
        load_csr.assert_not_called()

    def test_sign_existing_cert_does_not_overwrite_when_error_is_mocked(self, tmp_path):
        ca_dir = _setup_ca(tmp_path)
        csr_path = _setup_csr(tmp_path)
        out_dir = tmp_path / "signed"
        out_dir.mkdir()
        cert_path = out_dir / "hospital-1.crt"
        cert_path.write_text("existing cert")

        with patch("nvflare.tool.cert.cert_commands.output_error") as output_error:
            with patch("nvflare.tool.cert.cert_commands._build_signed_cert") as build_cert:
                result = sign_csr_files(csr_path=csr_path, ca_dir=ca_dir, output_dir=str(out_dir), cert_type="client")

        assert result is None
        output_error.assert_called_once()
        assert output_error.call_args.args[0] == "CERT_ALREADY_EXISTS"
        build_cert.assert_not_called()
        assert cert_path.read_text() == "existing cert"

    def test_sign_returns_when_ca_validation_returns_none(self, tmp_path):
        ca_dir = _setup_ca(tmp_path)
        csr_path = _setup_csr(tmp_path)
        out_dir = tmp_path / "signed"

        with patch("nvflare.tool.cert.cert_commands._validate_signing_ca", return_value=None):
            with patch("nvflare.tool.cert.cert_commands._build_signed_cert") as build_cert:
                result = sign_csr_files(csr_path=csr_path, ca_dir=ca_dir, output_dir=str(out_dir), cert_type="client")

        assert result is None
        build_cert.assert_not_called()

    def test_sign_rootca_copied(self, tmp_path):
        ca_dir = _setup_ca(tmp_path)
        csr_path = _setup_csr(tmp_path)
        out_dir = str(tmp_path / "signed")
        args = _sign_args(csr_path=csr_path, ca_dir=ca_dir, output_dir=out_dir, cert_type="client")
        handle_cert_sign(args)
        # rootCA.pem in output dir should match the one in ca_dir
        orig = open(os.path.join(ca_dir, "rootCA.pem"), "rb").read()
        copy = open(os.path.join(out_dir, "rootCA.pem"), "rb").read()
        assert orig == copy

    @pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlink support required")
    def test_sign_rootca_symlink_destination_is_rejected(self, tmp_path):
        ca_dir = _setup_ca(tmp_path)
        csr_path = _setup_csr(tmp_path)
        out_dir = tmp_path / "signed"
        out_dir.mkdir()

        outside_target = tmp_path / "outside-rootca.pem"
        outside_target.write_text("sentinel")
        os.symlink(str(outside_target), str(out_dir / "rootCA.pem"))

        args = _sign_args(csr_path=csr_path, ca_dir=ca_dir, output_dir=str(out_dir), cert_type="client")
        with pytest.raises(SystemExit) as exc_info:
            handle_cert_sign(args)
        assert exc_info.value.code == 1
        assert outside_target.read_text() == "sentinel"
        assert not (out_dir / "hospital-1.crt").exists()

    @pytest.mark.skipif(
        not hasattr(os, "symlink") or not hasattr(os, "O_NOFOLLOW"), reason="nofollow symlink support required"
    )
    def test_sign_rootca_symlink_source_is_rejected(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        ca_dir = _setup_ca(tmp_path)
        csr_path = _setup_csr(tmp_path)
        rootca_path = os.path.join(ca_dir, "rootCA.pem")
        real_rootca_path = os.path.join(ca_dir, "rootCA-real.pem")
        os.replace(rootca_path, real_rootca_path)
        os.symlink(real_rootca_path, rootca_path)
        out_dir = tmp_path / "signed"

        with pytest.raises(SystemExit) as exc_info:
            handle_cert_sign(_sign_args(csr_path=csr_path, ca_dir=ca_dir, output_dir=str(out_dir), cert_type="client"))

        assert exc_info.value.code == 1
        assert "CA_LOAD_FAILED" in capsys.readouterr().err
        assert not (out_dir / "hospital-1.crt").exists()

    @pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlink support required")
    def test_sign_cert_symlink_destination_is_rejected(self, tmp_path):
        ca_dir = _setup_ca(tmp_path)
        csr_path = _setup_csr(tmp_path)
        out_dir = tmp_path / "signed"
        out_dir.mkdir()

        outside_target = tmp_path / "outside-cert.crt"
        outside_target.write_text("sentinel")
        os.symlink(str(outside_target), str(out_dir / "hospital-1.crt"))

        args = _sign_args(csr_path=csr_path, ca_dir=ca_dir, output_dir=str(out_dir), cert_type="client")
        with pytest.raises(SystemExit) as exc_info:
            handle_cert_sign(args)
        assert exc_info.value.code == 1
        assert outside_target.read_text() == "sentinel"
        assert not (out_dir / "rootCA.pem").exists()

    def test_sign_refuses_to_overwrite_existing_rootca_without_force(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        ca_dir = _setup_ca(tmp_path)
        csr_path = _setup_csr(tmp_path)
        out_dir = tmp_path / "signed"
        out_dir.mkdir()
        (out_dir / "rootCA.pem").write_text("foreign-ca")

        with pytest.raises(SystemExit) as exc_info:
            handle_cert_sign(_sign_args(csr_path=csr_path, ca_dir=ca_dir, output_dir=str(out_dir), cert_type="client"))

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "ROOTCA_ALREADY_EXISTS" in captured.err
        assert str(out_dir / "rootCA.pem") in captured.err

    def test_sign_cleans_up_partial_output_when_rootca_copy_fails(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        ca_dir = _setup_ca(tmp_path)
        csr_path = _setup_csr(tmp_path)
        out_dir = tmp_path / "signed"

        with patch("nvflare.tool.cert.cert_commands._write_file_nofollow", side_effect=OSError("disk full")):
            with pytest.raises(SystemExit) as exc_info:
                handle_cert_sign(
                    _sign_args(csr_path=csr_path, ca_dir=ca_dir, output_dir=str(out_dir), cert_type="client")
                )

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "CERT_OUTPUT_WRITE_FAILED" in captured.err
        assert not (out_dir / "hospital-1.crt").exists()
        assert not (out_dir / "rootCA.pem").exists()

    def test_sign_aki_matches_issuer_cert(self, tmp_path):
        ca_dir = _setup_ca(tmp_path)
        csr_path = _setup_csr(tmp_path, name="site-1")
        out_dir = str(tmp_path / "signed")
        args = _sign_args(csr_path=csr_path, ca_dir=ca_dir, output_dir=out_dir, cert_type="client")

        handle_cert_sign(args)

        ca_cert = load_crt(os.path.join(ca_dir, "rootCA.pem"))
        issued_cert = load_crt(os.path.join(out_dir, "site-1.crt"))

        issuer_ski = ca_cert.extensions.get_extension_for_class(x509.SubjectKeyIdentifier).value.digest
        issued_aki = issued_cert.extensions.get_extension_for_class(x509.AuthorityKeyIdentifier).value

        assert issued_aki.key_identifier == issuer_ski

    def test_sign_multiple_certs_use_distinct_serial_numbers(self, tmp_path):
        ca_dir = _setup_ca(tmp_path)
        csr1 = _setup_csr(tmp_path / "csr1", name="client1")
        csr2 = _setup_csr(tmp_path / "csr2", name="client2")

        out1 = str(tmp_path / "signed1")
        out2 = str(tmp_path / "signed2")

        args1 = _sign_args(csr_path=csr1, ca_dir=ca_dir, output_dir=out1, cert_type="client")
        handle_cert_sign(args1)
        cert1 = load_crt(os.path.join(out1, "client1.crt"))

        args2 = _sign_args(csr_path=csr2, ca_dir=ca_dir, output_dir=out2, cert_type="client")
        handle_cert_sign(args2)
        cert2 = load_crt(os.path.join(out2, "client2.crt"))
        assert cert1.serial_number != cert2.serial_number

    def test_sign_admin_role_cert(self, tmp_path):
        ca_dir = _setup_ca(tmp_path)
        csr_path = _setup_csr(tmp_path, name="bob")
        out_dir = str(tmp_path / "signed")
        args = _sign_args(csr_path=csr_path, ca_dir=ca_dir, output_dir=out_dir, cert_type="org_admin")
        rc = handle_cert_sign(args)
        assert rc == 0
        assert os.path.exists(os.path.join(out_dir, "bob.crt"))
        # Admin-role certs must have content_commitment (non-repudiation) set to True.
        # This is required so that job submissions signed with admin certs cannot be repudiated.
        # Regression guard: _build_signed_cert must use content_commitment=True for admin roles.
        cert = load_crt(os.path.join(out_dir, "bob.crt"))
        key_usage = cert.extensions.get_extension_for_class(x509.KeyUsage)
        assert key_usage.value.content_commitment is True
        eku = cert.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value
        assert list(eku) == [x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH]

    def test_sign_server_cert_sets_server_auth_eku(self, tmp_path):
        ca_dir = _setup_ca(tmp_path)
        csr_dir = str(tmp_path / "csr-server")
        os.makedirs(csr_dir, exist_ok=True)
        args = _csr_args(name="fl-server", output_dir=csr_dir, cert_type="server")
        handle_cert_csr(args)
        csr_path = os.path.join(csr_dir, "fl-server.csr")

        out_dir = str(tmp_path / "signed-server")
        sign_args = _sign_args(csr_path=csr_path, ca_dir=ca_dir, output_dir=out_dir, cert_type="server")
        handle_cert_sign(sign_args)

        cert = load_crt(os.path.join(out_dir, "fl-server.crt"))
        eku = cert.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value
        assert list(eku) == [x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]

    def test_sign_uses_centralized_cert_generation_for_common_x509_content(self, tmp_path):
        ca_dir = _setup_ca(tmp_path)
        csr_dir = str(tmp_path / "csr-server")
        os.makedirs(csr_dir, exist_ok=True)
        handle_cert_csr(_csr_args(name="fl-server", org="hospital-central", output_dir=csr_dir, cert_type="server"))
        csr_path = os.path.join(csr_dir, "fl-server.csr")
        out_dir = str(tmp_path / "signed-server")
        server_default_host = "server-public.hospital-central.org"
        server_additional_hosts = ["server1.hospital-central.org", "10.0.1.50"]

        original_generate_cert = CertBuilder._generate_cert
        with patch(
            "nvflare.tool.cert.cert_commands.CertBuilder._generate_cert",
            wraps=original_generate_cert,
        ) as generate_cert:
            result = sign_csr_files(
                csr_path=csr_path,
                ca_dir=ca_dir,
                output_dir=out_dir,
                cert_type="server",
                server_default_host=server_default_host,
                server_additional_hosts=server_additional_hosts,
            )

        assert result is not None
        generate_cert.assert_called_once()
        _, call_kwargs = generate_cert.call_args
        assert call_kwargs["subject"] == "fl-server"
        assert call_kwargs["subject_org"] == "hospital-central"
        assert call_kwargs["role"] == "server"
        assert call_kwargs["server_default_host"] == server_default_host
        assert call_kwargs["server_additional_hosts"] == server_additional_hosts
        assert len(call_kwargs["extra_extensions"]) == 3

        distributed_cert = load_crt(os.path.join(out_dir, "fl-server.crt"))
        ca_cert = load_crt(os.path.join(ca_dir, "rootCA.pem"))
        ca_key = load_private_key_file(os.path.join(ca_dir, "rootCA.key"))
        csr = _load_and_validate_csr(csr_path)
        centralized_equivalent = original_generate_cert(
            subject="fl-server",
            subject_org="hospital-central",
            issuer=ca_cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value,
            signing_pri_key=ca_key,
            subject_pub_key=csr.public_key(),
            role="server",
            server_default_host=server_default_host,
            server_additional_hosts=server_additional_hosts,
        )

        assert distributed_cert.subject == centralized_equivalent.subject
        assert distributed_cert.issuer == centralized_equivalent.issuer
        assert _public_key_der(distributed_cert.public_key()) == _public_key_der(centralized_equivalent.public_key())
        assert _san_dns_and_ips(distributed_cert) == _san_dns_and_ips(centralized_equivalent)
        assert (
            distributed_cert.extensions.get_extension_for_class(x509.SubjectKeyIdentifier).value.digest
            == centralized_equivalent.extensions.get_extension_for_class(x509.SubjectKeyIdentifier).value.digest
        )
        assert (
            distributed_cert.extensions.get_extension_for_class(x509.AuthorityKeyIdentifier).value.key_identifier
            == centralized_equivalent.extensions.get_extension_for_class(
                x509.AuthorityKeyIdentifier
            ).value.key_identifier
        )

    def test_valid_days_custom(self, tmp_path):
        """--valid-days controls certificate not_valid_after."""
        import datetime

        ca_dir = _setup_ca(tmp_path)
        csr_path = _setup_csr(tmp_path, name="site-1")
        out_dir = str(tmp_path / "signed")
        args = _sign_args(csr_path=csr_path, ca_dir=ca_dir, output_dir=out_dir, cert_type="client", valid_days=90)
        handle_cert_sign(args)

        cert = load_crt(os.path.join(out_dir, "site-1.crt"))
        try:
            not_after = cert.not_valid_after_utc
            now = datetime.datetime.now(datetime.timezone.utc)
        except AttributeError:
            not_after = cert.not_valid_after
            now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
        days_remaining = (not_after - now).days
        # Should be ~90 days; give ±2 days tolerance for test timing
        assert 88 <= days_remaining <= 92

    def test_valid_days_default_is_1095(self, tmp_path):
        """Default cert validity is 3 years (1095 days)."""
        import datetime

        ca_dir = _setup_ca(tmp_path)
        csr_path = _setup_csr(tmp_path, name="site-1")
        out_dir = str(tmp_path / "signed")
        args = _sign_args(csr_path=csr_path, ca_dir=ca_dir, output_dir=out_dir, cert_type="client")
        handle_cert_sign(args)

        cert = load_crt(os.path.join(out_dir, "site-1.crt"))
        try:
            not_after = cert.not_valid_after_utc
            now = datetime.datetime.now(datetime.timezone.utc)
        except AttributeError:
            not_after = cert.not_valid_after
            now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
        days_remaining = (not_after - now).days
        assert 1093 <= days_remaining <= 1097

    def test_sign_rejects_expired_ca(self, tmp_path, capsys, monkeypatch):
        import datetime

        monkeypatch.setattr(cli_output, "_output_format", "txt")
        ca_dir = _setup_ca(tmp_path)
        now = datetime.datetime.now(datetime.timezone.utc)
        _overwrite_ca_cert(ca_dir, now - datetime.timedelta(days=2), now - datetime.timedelta(days=1), ca=True)

        csr_path = _setup_csr(tmp_path, name="site-1")
        out_dir = str(tmp_path / "signed")
        args = _sign_args(csr_path=csr_path, ca_dir=ca_dir, output_dir=out_dir, cert_type="client")
        with pytest.raises(SystemExit) as exc_info:
            handle_cert_sign(args)
        assert exc_info.value.code == 1
        assert "expired" in capsys.readouterr().err

    def test_sign_rejects_non_ca_issuer_cert(self, tmp_path, capsys, monkeypatch):
        import datetime

        monkeypatch.setattr(cli_output, "_output_format", "txt")
        ca_dir = _setup_ca(tmp_path)
        now = datetime.datetime.now(datetime.timezone.utc)
        _overwrite_ca_cert(ca_dir, now - datetime.timedelta(days=1), now + datetime.timedelta(days=30), ca=False)

        csr_path = _setup_csr(tmp_path, name="site-1")
        out_dir = str(tmp_path / "signed")
        args = _sign_args(csr_path=csr_path, ca_dir=ca_dir, output_dir=out_dir, cert_type="client")
        with pytest.raises(SystemExit) as exc_info:
            handle_cert_sign(args)
        assert exc_info.value.code == 1
        assert "not a CA certificate" in capsys.readouterr().err

    def test_leaf_validity_capped_by_ca_expiry(self, tmp_path):
        import datetime

        ca_dir = _setup_ca(tmp_path)
        now = datetime.datetime.now(datetime.timezone.utc)
        capped_ca_expiry = now + datetime.timedelta(days=30)
        _overwrite_ca_cert(ca_dir, now - datetime.timedelta(days=1), capped_ca_expiry, ca=True)

        csr_path = _setup_csr(tmp_path, name="site-1")
        out_dir = str(tmp_path / "signed")
        args = _sign_args(csr_path=csr_path, ca_dir=ca_dir, output_dir=out_dir, cert_type="client", valid_days=9999)
        handle_cert_sign(args)

        cert = load_crt(os.path.join(out_dir, "site-1.crt"))
        try:
            leaf_not_after = cert.not_valid_after_utc
        except AttributeError:
            leaf_not_after = cert.not_valid_after.replace(tzinfo=datetime.timezone.utc)
        assert leaf_not_after <= capped_ca_expiry
        assert abs((leaf_not_after - capped_ca_expiry).total_seconds()) < 5

    def test_random_serials_do_not_require_ca_json_mutation(self, tmp_path):
        ca_dir = _setup_ca(tmp_path)
        ca_json = os.path.join(ca_dir, "ca.json")
        with open(ca_json) as _f:
            initial = json.load(_f)

        csr1 = _setup_csr(tmp_path / "csr1", name="site-a")
        csr2 = _setup_csr(tmp_path / "csr2", name="site-b")
        csr3 = _setup_csr(tmp_path / "csr3", name="site-c")
        serials = set()

        for i, (csr, name) in enumerate([(csr1, "site-a"), (csr2, "site-b"), (csr3, "site-c")]):
            out = str(tmp_path / f"out{i}")
            args = _sign_args(csr_path=csr, ca_dir=ca_dir, output_dir=out, cert_type="client")
            handle_cert_sign(args)
            cert = load_crt(os.path.join(out, f"{name}.crt"))
            serials.add(cert.serial_number)

        with open(ca_json) as _f:
            final = json.load(_f)

        assert len(serials) == 3
        assert final == initial


# ---------------------------------------------------------------------------
# internal CSR helper: proposed role embedded in CSR
# ---------------------------------------------------------------------------


class TestCertCsrWithRole:
    def test_csr_role_embedded_when_type_given(self, tmp_path):
        """The internal CSR helper embeds 'lead' in CSR UNSTRUCTURED_NAME."""
        args = _csr_args(name="alice", output_dir=str(tmp_path), cert_type="lead")
        handle_cert_csr(args)
        csr = _load_csr_file(str(tmp_path / "alice.csr"))
        role_attrs = csr.subject.get_attributes_for_oid(NameOID.UNSTRUCTURED_NAME)
        assert len(role_attrs) == 1
        assert role_attrs[0].value == "lead"

    def test_csr_no_role_when_type_absent(self, tmp_path):
        """The internal CSR helper rejects missing role/type."""
        args = _csr_args(name="h1", output_dir=str(tmp_path))
        with pytest.raises(SystemExit) as exc_info:
            handle_cert_csr(args)
        assert exc_info.value.code == 4

    def test_csr_role_all_valid_types(self, tmp_path):
        """Each valid cert type can be embedded in a CSR."""
        for cert_type in ("client", "server", "org_admin", "lead", "member"):
            out = str(tmp_path / cert_type)
            os.makedirs(out, exist_ok=True)
            args = _csr_args(name="p1", output_dir=out, cert_type=cert_type)
            handle_cert_csr(args)
            csr = _load_csr_file(os.path.join(out, "p1.csr"))
            role_attrs = csr.subject.get_attributes_for_oid(NameOID.UNSTRUCTURED_NAME)
            assert role_attrs[0].value == cert_type


# ---------------------------------------------------------------------------
# internal signing helper: read type from CSR when explicit type is omitted
# ---------------------------------------------------------------------------


class TestCertSignReadsTypeFromCsr:
    def test_sign_accepts_type_from_csr_when_flag_is_set(self, tmp_path):
        """The internal signing helper reads role from CSR UNSTRUCTURED_NAME."""
        ca_dir = _setup_ca(tmp_path)
        # Generate CSR with role embedded
        csr_dir = str(tmp_path / "csr")
        os.makedirs(csr_dir, exist_ok=True)
        args = _csr_args(name="alice", output_dir=csr_dir, cert_type="lead")
        handle_cert_csr(args)
        csr_path = os.path.join(csr_dir, "alice.csr")

        out_dir = str(tmp_path / "signed")
        sign_args = _sign_args(csr_path=csr_path, ca_dir=ca_dir, output_dir=out_dir, accept_csr_role=True)
        rc = handle_cert_sign(sign_args)
        assert rc == 0
        cert = load_crt(os.path.join(out_dir, "alice.crt"))
        role_attrs = cert.subject.get_attributes_for_oid(NameOID.UNSTRUCTURED_NAME)
        assert role_attrs[0].value == "lead"

    def test_sign_t_overrides_csr_role(self, tmp_path):
        """The internal signing helper can override the role proposed in the CSR."""
        ca_dir = _setup_ca(tmp_path)
        csr_dir = str(tmp_path / "csr")
        os.makedirs(csr_dir, exist_ok=True)
        # CSR proposes 'member', Project Admin overrides to 'lead'
        args = _csr_args(name="bob", output_dir=csr_dir, cert_type="member")
        handle_cert_csr(args)
        csr_path = os.path.join(csr_dir, "bob.csr")

        out_dir = str(tmp_path / "signed")
        sign_args = _sign_args(csr_path=csr_path, ca_dir=ca_dir, output_dir=out_dir, cert_type="lead")
        handle_cert_sign(sign_args)
        cert = load_crt(os.path.join(out_dir, "bob.crt"))
        role_attrs = cert.subject.get_attributes_for_oid(NameOID.UNSTRUCTURED_NAME)
        assert role_attrs[0].value == "lead"

    def test_sign_without_explicit_decision_fails(self, tmp_path):
        """The internal signing helper requires an explicit role decision."""
        ca_dir = _setup_ca(tmp_path)
        csr_dir = str(tmp_path / "csr")
        os.makedirs(csr_dir, exist_ok=True)
        args = _csr_args(name="alice", output_dir=csr_dir, cert_type="lead")
        handle_cert_csr(args)
        csr_path = os.path.join(csr_dir, "alice.csr")
        out_dir = str(tmp_path / "signed")
        args = _sign_args(csr_path=csr_path, ca_dir=ca_dir, output_dir=out_dir)
        with pytest.raises(SystemExit) as exc_info:
            handle_cert_sign(args)
        assert exc_info.value.code == 4

    def test_sign_accept_csr_role_without_csr_role_fails(self, tmp_path):
        """--accept-csr-role requires a role embedded in the CSR."""
        ca_dir = _setup_ca(tmp_path)
        csr_dir = tmp_path / "csr"
        csr_dir.mkdir()
        _, pem_csr = _generate_csr("hospital-1")
        csr_path = csr_dir / "hospital-1.csr"
        csr_path.write_bytes(pem_csr)
        out_dir = str(tmp_path / "signed")
        args = _sign_args(csr_path=str(csr_path), ca_dir=ca_dir, output_dir=out_dir, accept_csr_role=True)
        with pytest.raises(SystemExit) as exc_info:
            handle_cert_sign(args)
        assert exc_info.value.code == 4

    def test_sign_type_and_accept_csr_role_are_mutually_exclusive(self, tmp_path):
        ca_dir = _setup_ca(tmp_path)
        csr_dir = str(tmp_path / "csr")
        os.makedirs(csr_dir, exist_ok=True)
        args = _csr_args(name="bob", output_dir=csr_dir, cert_type="member")
        handle_cert_csr(args)
        csr_path = os.path.join(csr_dir, "bob.csr")
        out_dir = str(tmp_path / "signed")
        args = _sign_args(
            csr_path=csr_path,
            ca_dir=ca_dir,
            output_dir=out_dir,
            cert_type="lead",
            accept_csr_role=True,
        )
        with pytest.raises(SystemExit) as exc_info:
            handle_cert_sign(args)
        assert exc_info.value.code == 4

    def test_sign_rejects_duplicate_safe_subject_oid_in_csr(self, tmp_path):
        ca_dir = _setup_ca(tmp_path)
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        subject = x509.Name(
            [
                x509.NameAttribute(NameOID.COMMON_NAME, "alice"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "attacker-org"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "victim-org"),
                x509.NameAttribute(NameOID.UNSTRUCTURED_NAME, "lead"),
            ]
        )
        csr = x509.CertificateSigningRequestBuilder().subject_name(subject).sign(key, hashes.SHA256())
        csr_dir = tmp_path / "csr"
        csr_dir.mkdir()
        csr_path = csr_dir / "alice.csr"
        csr_path.write_bytes(csr.public_bytes(serialization.Encoding.PEM))

        out_dir = str(tmp_path / "signed")
        args = _sign_args(csr_path=str(csr_path), ca_dir=ca_dir, output_dir=out_dir, accept_csr_role=True)
        with pytest.raises(SystemExit) as exc_info:
            handle_cert_sign(args)
        assert exc_info.value.code == 1

    def test_sign_rejects_empty_subject_cn_in_csr(self, tmp_path):
        ca_dir = _setup_ca(tmp_path)
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        subject = x509.Name([x509.NameAttribute(NameOID.UNSTRUCTURED_NAME, "lead")])
        csr = x509.CertificateSigningRequestBuilder().subject_name(subject).sign(key, hashes.SHA256())
        csr_dir = tmp_path / "csr-empty"
        csr_dir.mkdir()
        csr_path = csr_dir / "empty.csr"
        csr_path.write_bytes(csr.public_bytes(serialization.Encoding.PEM))

        out_dir = str(tmp_path / "signed-empty")
        args = _sign_args(csr_path=str(csr_path), ca_dir=ca_dir, output_dir=out_dir, accept_csr_role=True)
        with pytest.raises(SystemExit) as exc_info:
            handle_cert_sign(args)
        assert exc_info.value.code == 4

    def test_sign_rejects_whitespace_only_subject_cn_in_csr(self, tmp_path):
        ca_dir = _setup_ca(tmp_path)
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        subject = x509.Name(
            [
                x509.NameAttribute(NameOID.COMMON_NAME, "   "),
                x509.NameAttribute(NameOID.UNSTRUCTURED_NAME, "lead"),
            ]
        )
        csr = x509.CertificateSigningRequestBuilder().subject_name(subject).sign(key, hashes.SHA256())
        csr_dir = tmp_path / "csr-space"
        csr_dir.mkdir()
        csr_path = csr_dir / "space.csr"
        csr_path.write_bytes(csr.public_bytes(serialization.Encoding.PEM))

        out_dir = str(tmp_path / "signed-space")
        args = _sign_args(csr_path=str(csr_path), ca_dir=ca_dir, output_dir=out_dir, accept_csr_role=True)
        with pytest.raises(SystemExit) as exc_info:
            handle_cert_sign(args)
        assert exc_info.value.code == 4

    def test_sign_rejects_subject_cn_with_newline(self, tmp_path):
        ca_dir = _setup_ca(tmp_path)
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        subject = x509.Name(
            [
                x509.NameAttribute(NameOID.COMMON_NAME, "alice\nbad"),
                x509.NameAttribute(NameOID.UNSTRUCTURED_NAME, "lead"),
            ]
        )
        csr = x509.CertificateSigningRequestBuilder().subject_name(subject).sign(key, hashes.SHA256())
        csr_dir = tmp_path / "csr-newline"
        csr_dir.mkdir()
        csr_path = csr_dir / "newline.csr"
        csr_path.write_bytes(csr.public_bytes(serialization.Encoding.PEM))

        out_dir = str(tmp_path / "signed-newline")
        args = _sign_args(csr_path=str(csr_path), ca_dir=ca_dir, output_dir=out_dir, accept_csr_role=True)
        with pytest.raises(SystemExit) as exc_info:
            handle_cert_sign(args)
        assert exc_info.value.code == 4

    def test_sign_preserves_single_organization_name(self, tmp_path):
        ca_dir = _setup_ca(tmp_path)
        csr_dir = str(tmp_path / "csr")
        os.makedirs(csr_dir, exist_ok=True)
        args = _csr_args(name="alice", output_dir=csr_dir, cert_type="lead", org="valid-org")
        handle_cert_csr(args)
        csr_path = os.path.join(csr_dir, "alice.csr")

        out_dir = str(tmp_path / "signed")
        sign_args = _sign_args(csr_path=csr_path, ca_dir=ca_dir, output_dir=out_dir, accept_csr_role=True)
        handle_cert_sign(sign_args)
        cert = load_crt(os.path.join(out_dir, "alice.crt"))
        org_attrs = cert.subject.get_attributes_for_oid(NameOID.ORGANIZATION_NAME)
        assert [a.value for a in org_attrs] == ["valid-org"]


# ---------------------------------------------------------------------------
# _load_single_site_yaml unit tests
# ---------------------------------------------------------------------------


class TestLoadSingleSiteYaml:
    from nvflare.tool.cert.cert_commands import _load_single_site_yaml

    def _write_yaml(self, tmp_path, content: str) -> str:
        p = tmp_path / "site.yml"
        p.write_text(content)
        return str(p)

    def test_valid_yaml_returns_correct_dict(self, tmp_path):
        from nvflare.tool.cert.cert_commands import _load_single_site_yaml

        path = self._write_yaml(tmp_path, "name: hospital-1\norg: ACME\ntype: client\n")
        result = _load_single_site_yaml(path)
        assert result == {"name": "hospital-1", "org": "ACME", "cert_type": "client"}

    def test_type_key_maps_to_cert_type(self, tmp_path):
        from nvflare.tool.cert.cert_commands import _load_single_site_yaml

        path = self._write_yaml(tmp_path, "name: srv\norg: NVIDIA\ntype: server\n")
        result = _load_single_site_yaml(path)
        assert result["cert_type"] == "server"

    def test_include_key_is_not_resolved(self, tmp_path):
        from nvflare.tool.cert.cert_commands import _load_single_site_yaml

        included = tmp_path / "included.yml"
        included.write_text("name: hospital-1\norg: ACME\ntype: client\n")
        path = self._write_yaml(tmp_path, f"include: {included.name}\n")
        with pytest.raises(SystemExit) as exc_info:
            _load_single_site_yaml(path)
        assert exc_info.value.code == 4

    def test_file_not_found_exits_1(self, tmp_path):
        from nvflare.tool.cert.cert_commands import _load_single_site_yaml

        with pytest.raises(SystemExit) as exc_info:
            _load_single_site_yaml(str(tmp_path / "no-such.yml"))
        assert exc_info.value.code == 1

    def test_not_a_mapping_exits_4(self, tmp_path):
        from nvflare.tool.cert.cert_commands import _load_single_site_yaml

        path = self._write_yaml(tmp_path, "- a\n- b\n")
        with pytest.raises(SystemExit) as exc_info:
            _load_single_site_yaml(path)
        assert exc_info.value.code == 4

    def test_missing_name_exits_4(self, tmp_path):
        from nvflare.tool.cert.cert_commands import _load_single_site_yaml

        path = self._write_yaml(tmp_path, "org: ACME\ntype: client\n")
        with pytest.raises(SystemExit) as exc_info:
            _load_single_site_yaml(path)
        assert exc_info.value.code == 4

    def test_missing_org_exits_4(self, tmp_path):
        from nvflare.tool.cert.cert_commands import _load_single_site_yaml

        path = self._write_yaml(tmp_path, "name: h1\ntype: client\n")
        with pytest.raises(SystemExit) as exc_info:
            _load_single_site_yaml(path)
        assert exc_info.value.code == 4

    def test_missing_type_exits_4(self, tmp_path):
        from nvflare.tool.cert.cert_commands import _load_single_site_yaml

        path = self._write_yaml(tmp_path, "name: h1\norg: ACME\n")
        with pytest.raises(SystemExit) as exc_info:
            _load_single_site_yaml(path)
        assert exc_info.value.code == 4

    def test_invalid_type_exits_4(self, tmp_path):
        from nvflare.tool.cert.cert_commands import _load_single_site_yaml

        path = self._write_yaml(tmp_path, "name: h1\norg: ACME\ntype: gpu\n")
        with pytest.raises(SystemExit) as exc_info:
            _load_single_site_yaml(path)
        assert exc_info.value.code == 4

    def test_empty_name_exits_4(self, tmp_path):
        from nvflare.tool.cert.cert_commands import _load_single_site_yaml

        path = self._write_yaml(tmp_path, "name: ''\norg: ACME\ntype: client\n")
        with pytest.raises(SystemExit) as exc_info:
            _load_single_site_yaml(path)
        assert exc_info.value.code == 4

    @pytest.mark.skipif(
        not hasattr(os, "symlink") or not hasattr(os, "O_NOFOLLOW"), reason="nofollow symlink support required"
    )
    def test_symlink_source_is_rejected(self, tmp_path):
        from nvflare.tool.cert.cert_commands import _load_single_site_yaml

        real_path = tmp_path / "real.yml"
        real_path.write_text("name: hospital-1\norg: ACME\ntype: client\n")
        link_path = tmp_path / "site.yml"
        os.symlink(str(real_path), str(link_path))

        with pytest.raises(SystemExit) as exc_info:
            _load_single_site_yaml(str(link_path))

        assert exc_info.value.code == 4


# ---------------------------------------------------------------------------
# handle_cert_csr --project-file tests
# ---------------------------------------------------------------------------


class TestCertCsrProjectFile:
    def _write_site_yaml(self, tmp_path, name="hospital-1", org="ACME", cert_type="client") -> str:
        content = f"name: {name}\norg: {org}\ntype: {cert_type}\n"
        p = tmp_path / "site.yml"
        p.write_text(content)
        return str(p)

    def test_happy_path_creates_key_and_csr(self, tmp_path):
        site_file = self._write_site_yaml(tmp_path, name="hospital-1")
        out_dir = tmp_path / "out"
        args = _csr_args(name=None, project_file=site_file, output_dir=str(out_dir))
        rc = handle_cert_csr(args)
        assert rc == 0
        assert (out_dir / "hospital-1.key").exists()
        assert (out_dir / "hospital-1.csr").exists()

    def test_csr_cn_taken_from_yaml_name(self, tmp_path):
        site_file = self._write_site_yaml(tmp_path, name="fl-server")
        out_dir = tmp_path / "out"
        args = _csr_args(name=None, project_file=site_file, output_dir=str(out_dir))
        handle_cert_csr(args)
        csr = _load_csr_file(str(out_dir / "fl-server.csr"))
        cn = csr.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value
        assert cn == "fl-server"

    def test_project_file_rejects_name_with_newline(self, tmp_path):
        site_file = self._write_site_yaml(tmp_path, name="hospital-1\nbad")
        out_dir = tmp_path / "out"
        args = _csr_args(name=None, project_file=site_file, output_dir=str(out_dir))
        with pytest.raises(SystemExit) as exc_info:
            handle_cert_csr(args)
        assert exc_info.value.code == 4

    def test_csr_org_taken_from_yaml(self, tmp_path):
        site_file = self._write_site_yaml(tmp_path, org="NVIDIA")
        out_dir = tmp_path / "out"
        args = _csr_args(name=None, project_file=site_file, output_dir=str(out_dir))
        handle_cert_csr(args)
        csr = _load_csr_file(str(out_dir / "hospital-1.csr"))
        org_attrs = csr.subject.get_attributes_for_oid(NameOID.ORGANIZATION_NAME)
        assert len(org_attrs) == 1
        assert org_attrs[0].value == "NVIDIA"

    def test_missing_output_dir_exits_4(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "json")
        site_file = self._write_site_yaml(tmp_path)
        args = _csr_args(name=None, project_file=site_file, output_dir=None)
        with pytest.raises(SystemExit) as exc_info:
            handle_cert_csr(args)
        assert exc_info.value.code == 4

    def test_mutual_exclusivity_with_name_exits_4(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "json")
        site_file = self._write_site_yaml(tmp_path)
        out_dir = tmp_path / "out"
        args = _csr_args(name="conflicting", project_file=site_file, output_dir=str(out_dir))
        with pytest.raises(SystemExit) as exc_info:
            handle_cert_csr(args)
        assert exc_info.value.code == 4

    def test_mutual_exclusivity_with_org_exits_4(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "json")
        site_file = self._write_site_yaml(tmp_path)
        out_dir = tmp_path / "out"
        args = _csr_args(name=None, org="ConflictingOrg", project_file=site_file, output_dir=str(out_dir))
        with pytest.raises(SystemExit) as exc_info:
            handle_cert_csr(args)
        assert exc_info.value.code == 4

    def test_mutual_exclusivity_with_type_exits_4(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "json")
        site_file = self._write_site_yaml(tmp_path)
        out_dir = tmp_path / "out"
        args = _csr_args(name=None, cert_type="server", project_file=site_file, output_dir=str(out_dir))
        with pytest.raises(SystemExit) as exc_info:
            handle_cert_csr(args)
        assert exc_info.value.code == 4

    def test_nonexistent_project_file_exits_1(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "json")
        out_dir = tmp_path / "out"
        args = _csr_args(name=None, project_file=str(tmp_path / "no-such.yml"), output_dir=str(out_dir))
        with pytest.raises(SystemExit) as exc_info:
            handle_cert_csr(args)
        assert exc_info.value.code == 1

    def test_project_file_loader_none_stops_without_missing_flag_fallthrough(self, tmp_path):
        args = _csr_args(name=None, project_file=str(tmp_path / "bad.yml"), output_dir=str(tmp_path / "out"))

        with patch("nvflare.tool.cert.cert_commands._load_single_site_yaml", return_value=None) as load_site:
            with patch("nvflare.tool.cert.cert_commands.output_error_message") as output_error:
                with patch("nvflare.tool.cert.cert_commands.generate_csr_files") as generate_csr:
                    result = handle_cert_csr(args)

        assert result == 1
        load_site.assert_called_once_with(str(tmp_path / "bad.yml"))
        output_error.assert_not_called()
        generate_csr.assert_not_called()

    def test_force_allowed_with_project_file(self, tmp_path):
        site_file = self._write_site_yaml(tmp_path, name="h1")
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        (out_dir / "h1.key").write_bytes(b"old-key")
        args = _csr_args(name=None, project_file=site_file, output_dir=str(out_dir), force=True)
        rc = handle_cert_csr(args)
        assert rc == 0
        assert b"PRIVATE KEY" in (out_dir / "h1.key").read_bytes()

    def test_mutual_exclusivity_checked_before_yaml_load(self, tmp_path, monkeypatch):
        """Conflict with --name must error even when the YAML file doesn't exist."""
        monkeypatch.setattr(cli_output, "_output_format", "json")
        out_dir = tmp_path / "out"
        args = _csr_args(
            name="conflict",
            project_file=str(tmp_path / "no-such.yml"),
            output_dir=str(out_dir),
        )
        with pytest.raises(SystemExit) as exc_info:
            handle_cert_csr(args)
        # Must be INVALID_ARGS exit 4 (mutual exclusivity), not file-not-found exit 1
        assert exc_info.value.code == 4


class TestHandleCertCmdRouting:
    """Tests for handle_cert_cmd top-level dispatch."""

    def test_no_subcommand_exits_4(self, capsys):
        """nvflare cert with no subcommand prints help and exits with INVALID_ARGS."""
        from argparse import ArgumentParser
        from unittest.mock import MagicMock

        from nvflare.tool.cert.cert_cli import def_cert_cli_parser, handle_cert_cmd

        parser = ArgumentParser(prog="nvflare")
        subparsers = parser.add_subparsers(dest="sub_command")
        def_cert_cli_parser(subparsers)
        args = MagicMock()
        args.cert_sub_command = None

        with pytest.raises(SystemExit) as exc_info:
            handle_cert_cmd(args)
        assert exc_info.value.code == 4
        captured = capsys.readouterr()
        assert "usage:" in captured.err
        assert "cert subcommand required" in captured.err
        assert "Code: INVALID_ARGS (exit 4)" in captured.err

    def test_unknown_subcommand_exits_4(self, capsys):
        """nvflare cert <unknown> prints help and exits with INVALID_ARGS."""
        from argparse import ArgumentParser
        from unittest.mock import MagicMock

        from nvflare.tool.cert.cert_cli import def_cert_cli_parser, handle_cert_cmd

        parser = ArgumentParser(prog="nvflare")
        subparsers = parser.add_subparsers(dest="sub_command")
        def_cert_cli_parser(subparsers)
        args = MagicMock()
        args.cert_sub_command = "nonexistent"

        with pytest.raises(SystemExit) as exc_info:
            handle_cert_cmd(args)
        assert exc_info.value.code == 4
        captured = capsys.readouterr()
        assert "usage:" in captured.err
        assert "invalid cert subcommand" in captured.err
        assert "Code: INVALID_ARGS (exit 4)" in captured.err

    def test_no_subcommand_returns_when_error_is_mocked(self):
        from unittest.mock import MagicMock

        from nvflare.tool.cert.cert_cli import handle_cert_cmd

        args = MagicMock()
        args.cert_sub_command = None
        args.compat_output_format = None

        with patch("nvflare.tool.cli_output.output_usage_error") as output_usage_error:
            rc = handle_cert_cmd(args)

        assert rc == 1
        output_usage_error.assert_called_once()

    def test_compat_output_alias_sets_output_format(self, tmp_path):
        from argparse import ArgumentParser

        from nvflare.tool.cert.cert_cli import def_cert_cli_parser, handle_cert_cmd

        parser = ArgumentParser(prog="nvflare")
        subparsers = parser.add_subparsers(dest="sub_command")
        def_cert_cli_parser(subparsers)
        profile_path = tmp_path / "project_profile.yaml"
        _write_project_profile(profile_path, project="Demo")
        args = parser.parse_args(
            ["cert", "init", "--profile", str(profile_path), "-o", str(tmp_path), "--output", "json"]
        )

        with patch("nvflare.tool.cli_output.set_output_format") as set_output_format:
            with patch("nvflare.tool.cert.cert_commands.handle_cert_init", return_value=0) as handle_init:
                rc = handle_cert_cmd(args)

        set_output_format.assert_called_once_with("json")
        handle_init.assert_called_once_with(args)
        assert rc == 0


# ---------------------------------------------------------------------------
# Distributed provisioning public request/approve workflow
# ---------------------------------------------------------------------------


def _cert_root_parser():
    from nvflare.tool.cert.cert_cli import def_cert_cli_parser

    parser = argparse.ArgumentParser(prog="nvflare")
    subparsers = parser.add_subparsers(dest="sub_command")
    parsers = def_cert_cli_parser(subparsers)
    return parser, parsers["cert"]


def _run_cert_cli(argv):
    from nvflare.tool.cert.cert_cli import handle_cert_cmd

    parser, _ = _cert_root_parser()
    args = parser.parse_args(["cert"] + argv)
    return handle_cert_cmd(args)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path) -> str:
    with open(path, "rb") as f:
        return _sha256_bytes(f.read())


def _write_participant_definition(path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def _write_project_profile(
    path,
    project="example_project",
    scheme="grpc",
    connection_security="tls",
    server_host="fl-server",
    fed_learn_port=8002,
    admin_port=8003,
) -> None:
    _write_participant_definition(
        path,
        {
            "name": project,
            "scheme": scheme,
            "connection_security": connection_security,
            "server": {
                "host": server_host,
                "fed_learn_port": fed_learn_port,
                "admin_port": admin_port,
            },
        },
    )


def _participant_request_args(participant_path):
    return ["request", "--participant", str(participant_path)]


def _request_participant_definition(
    project="example_project",
    name="site-3",
    participant_type="client",
    org="nvidia",
    role=None,
):
    participant = {
        "name": name,
        "type": participant_type,
        "org": org,
    }
    if role:
        participant["role"] = role
    return {
        "name": project,
        "participants": [participant],
    }


def _write_request_participant(path, **kwargs):
    _write_participant_definition(path, _request_participant_definition(**kwargs))


def _public_key_sha256_from_csr(csr_path) -> str:
    csr = _load_csr_file(str(csr_path))
    public_key_der = csr.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return _sha256_bytes(public_key_der)


def _write_request_zip(
    tmp_path,
    *,
    name="site-3",
    project="example_project",
    org="nvidia",
    kind="site",
    cert_type="client",
    request_id="11111111111111111111111111111111",
    include_key=False,
    traversal=False,
    csr_member=None,
    hash_mismatch=False,
    omit_fields=(),
    metadata_updates=None,
):
    request_dir = tmp_path / name
    request_dir.mkdir()
    key_pem, csr_pem = _generate_csr(name, org, cert_type)
    key_path = request_dir / f"{name}.key"
    csr_path = request_dir / f"{name}.csr"
    site_yaml_path = request_dir / "site.yaml"
    request_json_path = request_dir / "request.json"
    key_path.write_bytes(key_pem)
    csr_path.write_bytes(csr_pem)
    participant_type = "server" if kind == "server" else "client" if cert_type == "client" else "admin"
    participant = {"name": name, "type": participant_type, "org": org}
    if participant_type == "admin":
        participant["role"] = cert_type
    site_yaml_path.write_text(
        yaml.safe_dump(
            {
                "name": project,
                "participants": [participant],
            },
            sort_keys=False,
        )
    )
    request_metadata = {
        "artifact_type": "nvflare.cert.request",
        "schema_version": "1",
        "request_id": request_id,
        "created_at": "2026-04-24T00:00:00Z",
        "project": project,
        "name": name,
        "org": org,
        "kind": kind,
        "cert_type": cert_type,
        "cert_role": None,
        "csr_sha256": "0" * 64 if hash_mismatch else _sha256_file(csr_path),
        "public_key_sha256": _public_key_sha256_from_csr(csr_path),
        "site_yaml_sha256": _sha256_file(site_yaml_path),
    }
    if metadata_updates:
        request_metadata.update(metadata_updates)
    for field in omit_fields:
        request_metadata.pop(field, None)
    request_json_path.write_text(json.dumps(request_metadata, sort_keys=True))
    zip_path = tmp_path / f"{name}.request.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(request_json_path, "request.json")
        zf.write(site_yaml_path, "site.yaml")
        zf.write(csr_path, csr_member or (f"../{name}.csr" if traversal else f"{name}.csr"))
        if include_key:
            zf.write(key_path, f"{name}.key")
    return zip_path


class TestDistributedCertPublicSurface:
    def test_cert_help_exposes_request_and_approve_without_developer_csr_sign(self):
        _, cert_parser = _cert_root_parser()
        help_text = cert_parser.format_help()

        assert "request" in help_text
        assert "approve" in help_text
        assert " csr " not in help_text
        assert " sign " not in help_text

    @pytest.mark.parametrize("subcommand", ["csr", "sign"])
    def test_removed_developer_subcommands_are_not_parseable(self, subcommand):
        parser, _ = _cert_root_parser()

        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["cert", subcommand, "--schema"])

        assert exc_info.value.code == 2


class TestDistributedCertParticipantWorkflow:
    def test_request_parser_accepts_participant_definition_without_legacy_identity_args(self, tmp_path):
        participant_path = tmp_path / "hospital-a.yaml"
        _write_participant_definition(
            participant_path,
            {
                "name": "hospital_federation",
                "participants": [
                    {
                        "name": "hospital-a",
                        "type": "client",
                        "org": "hospital_alpha",
                    }
                ],
            },
        )
        parser, _ = _cert_root_parser()

        args = parser.parse_args(["cert"] + _participant_request_args(participant_path))

        assert args.cert_sub_command == "request"
        assert args.participant == str(participant_path)

    def test_request_parser_requires_participant_definition(self, capsys):
        parser, _ = _cert_root_parser()

        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["cert", "request"])

        assert exc_info.value.code == 2
        assert "--participant" in capsys.readouterr().err

    def test_request_from_client_participant_definition_derives_identity_and_sanitizes_zip(
        self, tmp_path, capsys, monkeypatch
    ):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.chdir(tmp_path)
        participant_definition = {
            "name": "hospital_federation",
            "description": "Site A - Hospital Alpha",
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
            ],
        }
        participant_path = tmp_path / "hospital-a.yaml"
        _write_participant_definition(participant_path, participant_definition)

        _run_cert_cli(_participant_request_args(participant_path))

        request_dir = tmp_path / "hospital-a"
        assert yaml.safe_load((request_dir / "site.yaml").read_text()) == participant_definition
        with zipfile.ZipFile(request_dir / "hospital-a.request.zip") as zf:
            request_json = json.loads(zf.read("request.json"))
            approval_site = yaml.safe_load(zf.read("site.yaml"))

        assert request_json["project"] == "hospital_federation"
        assert request_json["name"] == "hospital-a"
        assert request_json["org"] == "hospital_alpha"
        assert request_json["kind"] == "site"
        assert request_json["cert_type"] == "client"
        assert request_json.get("cert_role") is None
        assert approval_site["name"] == "hospital_federation"
        assert approval_site["participants"] == participant_definition["participants"]
        assert "builders" not in approval_site
        capsys.readouterr()

    def test_request_accepts_localhost_server_name_like_centralized_provisioning(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.chdir(tmp_path)
        participant_path = tmp_path / "server.yaml"
        _write_participant_definition(
            participant_path,
            {
                "name": "hospital_federation",
                "participants": [
                    {
                        "name": "localhost",
                        "type": "server",
                        "org": "hospital_central",
                    }
                ],
            },
        )

        rc = _run_cert_cli(_participant_request_args(participant_path))

        assert rc == 0
        assert (tmp_path / "localhost" / "localhost.request.zip").exists()
        capsys.readouterr()

    def test_server_request_uses_centralized_long_name_cert_convention(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.chdir(tmp_path)
        long_name = f"server-{'a' * 70}.hospital-central.org"
        ca_dir = tmp_path / "ca"
        profile_path = tmp_path / "project_profile.yaml"
        signed_zip = tmp_path / f"{long_name}.signed.zip"
        participant_path = tmp_path / "server.yaml"
        _write_participant_definition(
            participant_path,
            {
                "name": "hospital_federation",
                "participants": [
                    {
                        "name": long_name,
                        "type": "server",
                        "org": "hospital_central",
                        "host_names": ["server-alias.hospital-central.org"],
                    }
                ],
            },
        )
        _write_participant_definition(
            profile_path,
            {
                "name": "hospital_federation",
                "scheme": "grpc",
                "connection_security": "tls",
                "server": {
                    "host": long_name,
                    "fed_learn_port": 8002,
                    "admin_port": 8003,
                },
            },
        )

        handle_cert_init(_init_args(profile=str(profile_path), org="hospital_central", output_dir=str(ca_dir)))
        _run_cert_cli(_participant_request_args(participant_path))
        request_dir = tmp_path / long_name
        _run_cert_cli(
            [
                "approve",
                str(request_dir / f"{long_name}.request.zip"),
                "--ca-dir",
                str(ca_dir),
                "--profile",
                str(profile_path),
                "--out",
                str(signed_zip),
            ]
        )

        with zipfile.ZipFile(signed_zip) as zf:
            signed_json = json.loads(zf.read("signed.json"))
            signed_cert = x509.load_pem_x509_certificate(zf.read(f"{long_name}.crt"), default_backend())

        assert signed_json["name"] == long_name
        assert signed_json["cert_file"] == f"{long_name}.crt"
        subject_cn = signed_cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value
        assert subject_cn == long_name[:64]
        san = signed_cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        assert long_name in san.get_values_for_type(x509.DNSName)
        assert "server-alias.hospital-central.org" in san.get_values_for_type(x509.DNSName)
        capsys.readouterr()

    def test_request_rejects_invalid_server_host_names_with_centralized_validation(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.chdir(tmp_path)
        participant_path = tmp_path / "server.yaml"
        _write_participant_definition(
            participant_path,
            {
                "name": "hospital_federation",
                "participants": [
                    {
                        "name": "server1.hospital-central.org",
                        "type": "server",
                        "org": "hospital_central",
                        "host_names": ["bad host name"],
                    }
                ],
            },
        )

        with pytest.raises(SystemExit) as exc_info:
            _run_cert_cli(_participant_request_args(participant_path))

        assert exc_info.value.code == 4
        captured = capsys.readouterr()
        assert "INVALID_ARGS" in captured.err
        assert "host_names" in captured.err

    def test_request_rejects_client_listening_host_until_listener_cert_supported(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.chdir(tmp_path)
        participant_path = tmp_path / "site-1.yaml"
        _write_participant_definition(
            participant_path,
            {
                "name": "hospital_federation",
                "participants": [
                    {
                        "name": "hospital-a",
                        "type": "client",
                        "org": "hospital_alpha",
                        "listening_host": "hospital-a.internal",
                    }
                ],
            },
        )

        with pytest.raises(SystemExit) as exc_info:
            _run_cert_cli(_participant_request_args(participant_path))

        assert exc_info.value.code == 4
        captured = capsys.readouterr()
        assert "listening_host" in captured.err
        assert "not supported" in captured.err

    def test_request_from_user_participant_definition_derives_role(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.chdir(tmp_path)
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
        participant_path = tmp_path / "alice.yaml"
        _write_participant_definition(participant_path, participant_definition)

        _run_cert_cli(_participant_request_args(participant_path))

        request_dir = tmp_path / "alice@hospital-alpha.org"
        with zipfile.ZipFile(request_dir / "alice@hospital-alpha.org.request.zip") as zf:
            request_json = json.loads(zf.read("request.json"))
            approval_site = yaml.safe_load(zf.read("site.yaml"))

        assert request_json["project"] == "hospital_federation"
        assert request_json["name"] == "alice@hospital-alpha.org"
        assert request_json["org"] == "hospital_alpha"
        assert request_json["kind"] == "user"
        assert request_json["cert_type"] == "lead"
        assert request_json["cert_role"] == "lead"
        assert approval_site["participants"][0]["role"] == "lead"
        capsys.readouterr()

    def test_server_connection_security_override_stays_local_and_approve_injects_profile(
        self, tmp_path, capsys, monkeypatch
    ):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.chdir(tmp_path)
        ca_dir = tmp_path / "ca"
        signed_zip = tmp_path / "server1.hospital-central.org.signed.zip"
        participant_definition = {
            "name": "hospital_federation",
            "participants": [
                {
                    "name": "server1.hospital-central.org",
                    "type": "server",
                    "org": "hospital_central",
                    "host_names": ["server1.hospital-central.org", "10.0.1.50", "fl-server.internal"],
                    "connection_security": "clear",
                }
            ],
        }
        profile = {
            "name": "hospital_federation",
            "scheme": "grpc",
            "connection_security": "tls",
            "server": {
                "host": "server1.hospital-central.org",
                "fed_learn_port": 8002,
                "admin_port": 8003,
            },
        }
        participant_path = tmp_path / "server.yaml"
        profile_path = tmp_path / "project_profile.yaml"
        _write_participant_definition(participant_path, participant_definition)
        _write_participant_definition(profile_path, profile)

        handle_cert_init(_init_args(profile=str(profile_path), org="hospital_central", output_dir=str(ca_dir)))
        _run_cert_cli(_participant_request_args(participant_path))
        request_dir = tmp_path / "server1.hospital-central.org"
        capsys.readouterr()

        local_site = yaml.safe_load((request_dir / "site.yaml").read_text())
        assert local_site["participants"][0]["connection_security"] == "clear"
        with zipfile.ZipFile(request_dir / "server1.hospital-central.org.request.zip") as zf:
            request_site = yaml.safe_load(zf.read("site.yaml"))
        assert "connection_security" not in request_site["participants"][0]

        _run_cert_cli(
            [
                "approve",
                str(request_dir / "server1.hospital-central.org.request.zip"),
                "--ca-dir",
                str(ca_dir),
                "--profile",
                str(profile_path),
                "--out",
                str(signed_zip),
            ]
        )

        with zipfile.ZipFile(signed_zip) as zf:
            signed_json = json.loads(zf.read("signed.json"))
            signed_site = yaml.safe_load(zf.read("site.yaml"))
            signed_cert = x509.load_pem_x509_certificate(zf.read("server1.hospital-central.org.crt"), default_backend())
        assert signed_json["scheme"] == "grpc"
        assert signed_json["default_connection_security"] == "tls"
        assert signed_json["server"] == {
            "host": "server1.hospital-central.org",
            "fed_learn_port": 8002,
            "admin_port": 8003,
        }
        assert "connection_security" not in signed_site["participants"][0]
        san = signed_cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        assert "server1.hospital-central.org" in san.get_values_for_type(x509.DNSName)
        assert "fl-server.internal" in san.get_values_for_type(x509.DNSName)
        assert "10.0.1.50" in [str(ip) for ip in san.get_values_for_type(x509.IPAddress)]
        capsys.readouterr()

    def test_client_signed_cert_keeps_subject_name_san_like_centralized(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.chdir(tmp_path)
        ca_dir = tmp_path / "ca"
        request_dir = tmp_path / "site-3"
        signed_zip = tmp_path / "site-3.signed.zip"
        participant_path = tmp_path / "site-3.yaml"
        profile_path = tmp_path / "project_profile.yaml"

        handle_cert_init(
            _init_args(profile=_make_profile(tmp_path, "example_project"), org="nvidia", output_dir=str(ca_dir))
        )
        _write_project_profile(profile_path)
        _write_request_participant(participant_path)
        _run_cert_cli(["request", "-p", str(participant_path), "--out", str(request_dir)])
        capsys.readouterr()

        _run_cert_cli(
            [
                "approve",
                str(request_dir / "site-3.request.zip"),
                "--ca-dir",
                str(ca_dir),
                "--profile",
                str(profile_path),
                "--out",
                str(signed_zip),
            ]
        )

        with zipfile.ZipFile(signed_zip) as zf:
            signed_cert = x509.load_pem_x509_certificate(zf.read("site-3.crt"), default_backend())

        san = signed_cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        assert san.get_values_for_type(x509.DNSName) == ["site-3"]
        assert san.get_values_for_type(x509.IPAddress) == []
        capsys.readouterr()

    @pytest.mark.parametrize(
        "extra_kwargs,expected_flag",
        [
            ({"org": "myorg"}, "--org"),
            ({"project": "myproject"}, "--project"),
            ({"name": "myname"}, "--name"),
            ({"cert_type": "client"}, "--type"),
        ],
    )
    def test_request_rejects_legacy_identity_arg_alongside_participant(
        self, tmp_path, capsys, monkeypatch, extra_kwargs, expected_flag
    ):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        participant_path = tmp_path / "participant.yaml"
        participant_path.write_text("name: p\nparticipants:\n- name: p\n")
        args = _request_args(participant=str(participant_path), **extra_kwargs)

        with pytest.raises(SystemExit) as exc_info:
            handle_cert_request(args)

        assert exc_info.value.code == 4
        assert expected_flag in capsys.readouterr().err

    def test_approve_requires_project_profile(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        ca_dir = tmp_path / "ca"
        handle_cert_init(
            _init_args(profile=_make_profile(tmp_path, "example_project"), org="nvidia", output_dir=str(ca_dir))
        )
        request_zip = _write_request_zip(tmp_path)
        capsys.readouterr()

        with pytest.raises(SystemExit) as exc_info:
            _run_cert_cli(["approve", str(request_zip), "--ca-dir", str(ca_dir)])

        assert exc_info.value.code == 2
        assert "--profile" in capsys.readouterr().err

    def test_approve_server_san_fields_use_centralized_participant_validation(self, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        site_meta = {
            "name": "hospital_federation",
            "participants": [
                {
                    "name": "server1.hospital-central.org",
                    "type": "server",
                    "org": "hospital_central",
                    "fed_learn_port": 8002,
                    "admin_port": 8003,
                    "host_names": ["bad host name"],
                }
            ],
        }

        with pytest.raises(SystemExit) as exc_info:
            _server_cert_san_fields(site_meta, {"cert_type": "server"})

        assert exc_info.value.code == 4
        captured = capsys.readouterr()
        assert "host_names" in captured.err

    def test_server_san_field_validation_returns_none_when_error_helper_is_mocked(self):
        site_meta = {
            "name": "hospital_federation",
            "participants": [
                {
                    "name": "server1.hospital-central.org",
                    "type": "server",
                    "org": "hospital_central",
                    "fed_learn_port": 8002,
                    "admin_port": 8003,
                    "host_names": ["bad host name"],
                }
            ],
        }

        with patch("nvflare.tool.cert.cert_commands.output_error_message") as output_error:
            result = _server_cert_san_fields(site_meta, {"cert_type": "server"})

        assert result is None
        output_error.assert_called_once()

    def test_server_san_flat_site_yaml_uses_project_profile_host(self):
        # Flat site.yaml (no participants list) with project_profile: use profile host as SAN.
        flat_site_meta = {"name": "server1.hospital-central.org", "type": "server", "org": "hospital_central"}
        profile = {"server": {"host": "server1.hospital-central.org", "fed_learn_port": 8002, "admin_port": 8003}}

        default_host, host_names = _server_cert_san_fields(
            flat_site_meta, {"cert_type": "server"}, project_profile=profile
        )

        assert default_host == "server1.hospital-central.org"
        assert host_names is None

    def test_server_san_flat_site_yaml_no_profile_returns_none_host(self):
        # Flat site.yaml without a project_profile: default_host is None (falls back to CN).
        flat_site_meta = {"name": "server1.hospital-central.org", "type": "server", "org": "hospital_central"}

        default_host, host_names = _server_cert_san_fields(flat_site_meta, {"cert_type": "server"})

        assert default_host is None
        assert host_names is None

    def test_approve_nonexistent_profile_reports_file_not_found(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        ca_dir = tmp_path / "ca"
        handle_cert_init(
            _init_args(profile=_make_profile(tmp_path, "example_project"), org="nvidia", output_dir=str(ca_dir))
        )
        request_zip = _write_request_zip(tmp_path)
        missing_profile = str(tmp_path / "does_not_exist.yaml")
        capsys.readouterr()

        with pytest.raises(SystemExit) as exc_info:
            _run_cert_cli(["approve", str(request_zip), "--ca-dir", str(ca_dir), "--profile", missing_profile])

        assert exc_info.value.code == 4
        assert "file not found" in capsys.readouterr().err


class TestDistributedCertRequestApprove:
    def test_request_missing_participant_returns_when_error_is_mocked(self, tmp_path):
        args = argparse.Namespace(
            schema=False,
            participant=None,
            output_dir=str(tmp_path / "site-3"),
            force=False,
        )

        with patch("nvflare.tool.cert.cert_commands.output_usage_error") as output_error:
            result = handle_cert_request(args)

        assert result == 1
        output_error.assert_called_once()
        assert not (tmp_path / "site-3").exists()

    def test_resolve_sign_cert_type_returns_after_conflicting_modes_when_error_is_mocked(self):
        with patch("nvflare.tool.cert.cert_commands.output_error_message") as output_error:
            cert_type = _resolve_sign_cert_type(None, "client", True)

        output_error.assert_called_once()
        assert cert_type is None
        assert "use either -t/--type or --accept-csr-role" in output_error.call_args.kwargs["detail"]

    def test_validate_request_kind_cert_type_returns_after_invalid_kind_when_error_is_mocked(self):
        with patch("nvflare.tool.cert.cert_commands.output_error_message") as output_error:
            _validate_request_kind_cert_type("bad-kind", "client")

        output_error.assert_called_once()
        assert "cert request kind must be one of" in output_error.call_args.kwargs["detail"]

    @pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlink support required")
    def test_request_zip_writer_rejects_symlink_source(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        source = tmp_path / "source.json"
        source.write_text('{"ok": true}')
        link = tmp_path / "source-link.json"
        os.symlink(str(source), str(link))
        zip_path = tmp_path / "request.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("sentinel.txt", "keep")

        with pytest.raises(SystemExit) as exc_info:
            _write_zip_nofollow(str(zip_path), {"request.json": str(link)}, force=True)

        assert exc_info.value.code == 4
        captured = capsys.readouterr()
        assert "unsafe zip source" in captured.err
        with zipfile.ZipFile(zip_path) as zf:
            assert zf.read("sentinel.txt") == b"keep"

    def test_request_zip_writer_does_not_truncate_existing_zip_for_private_key_member(
        self, tmp_path, capsys, monkeypatch
    ):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        source = tmp_path / "source.json"
        source.write_text('{"ok": true}')
        zip_path = tmp_path / "request.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("sentinel.txt", "keep")

        with pytest.raises(SystemExit) as exc_info:
            _write_zip_nofollow(str(zip_path), {"site-3.key": str(source)}, force=True)

        assert exc_info.value.code == 4
        assert "private keys" in capsys.readouterr().err
        with zipfile.ZipFile(zip_path) as zf:
            assert zf.read("sentinel.txt") == b"keep"

    def test_request_zip_writer_removes_partial_zip_when_write_fails(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        source = tmp_path / "source.json"
        source.write_text('{"ok": true}')
        zip_path = tmp_path / "request.zip"

        original_writestr = zipfile.ZipFile.writestr

        def _fail_after_first_write(self, zinfo_or_arcname, data, *args, **kwargs):
            original_writestr(self, zinfo_or_arcname, data, *args, **kwargs)
            raise OSError("disk full")

        with patch("zipfile.ZipFile.writestr", _fail_after_first_write):
            with pytest.raises(SystemExit) as exc_info:
                _write_zip_nofollow(str(zip_path), {"request.json": str(source)})

        assert exc_info.value.code == 1
        assert "OUTPUT_DIR_NOT_WRITABLE" in capsys.readouterr().err
        assert not zip_path.exists()

    def test_request_returns_after_zip_write_error_when_error_is_mocked(self, tmp_path):
        request_dir = tmp_path / "site-3"
        participant_path = tmp_path / "site-3.yaml"
        _write_request_participant(participant_path)
        args = argparse.Namespace(
            schema=False,
            participant=str(participant_path),
            output_dir=str(request_dir),
            force=False,
        )

        with patch("nvflare.tool.cert.cert_commands._write_zip_nofollow", side_effect=OSError("blocked")):
            with patch("nvflare.tool.cert.cert_commands.output_error") as output_error:
                with patch("nvflare.tool.cert.cert_commands._try_write_request_audit") as write_audit:
                    rc = handle_cert_request(args)

        assert rc == 1
        output_error.assert_called_once()
        assert output_error.call_args.args[0] == "OUTPUT_DIR_NOT_WRITABLE"
        write_audit.assert_not_called()

    def test_request_returns_when_zip_writer_reports_failure_under_mocked_error(self, tmp_path):
        request_dir = tmp_path / "site-3"
        participant_path = tmp_path / "site-3.yaml"
        _write_request_participant(participant_path)
        args = argparse.Namespace(
            schema=False,
            participant=str(participant_path),
            output_dir=str(request_dir),
            force=False,
        )

        with patch("nvflare.tool.cert.cert_commands._write_zip_nofollow", return_value=False):
            with patch("nvflare.tool.cert.cert_commands._try_write_request_audit") as write_audit:
                rc = handle_cert_request(args)

        assert rc == 1
        write_audit.assert_not_called()

    def test_request_creates_request_zip_without_private_key_and_final_workflow_hint(
        self, tmp_path, capsys, monkeypatch
    ):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        request_dir = tmp_path / "site-3"
        participant_path = tmp_path / "site-3.yaml"
        _write_request_participant(participant_path)

        _run_cert_cli(
            [
                "request",
                "-p",
                str(participant_path),
                "--out",
                str(request_dir),
            ]
        )

        assert (request_dir / "site-3.key").is_file()
        assert (request_dir / "site-3.csr").is_file()
        assert (request_dir / "site.yaml").is_file()
        assert (request_dir / "request.json").is_file()
        request_zip = request_dir / "site-3.request.zip"
        assert request_zip.is_file()

        with zipfile.ZipFile(request_zip) as zf:
            assert sorted(zf.namelist()) == ["request.json", "site-3.csr", "site.yaml"]
            assert not any(name.endswith(".key") for name in zf.namelist())
            request_json = json.loads(zf.read("request.json"))
        assert request_json["name"] == "site-3"
        assert request_json["org"] == "nvidia"
        assert request_json["kind"] == "site"
        assert request_json["cert_type"] == "client"
        assert request_json["project"] == "example_project"
        assert (tmp_path / "home" / ".nvflare" / "cert_requests" / request_json["request_id"] / "audit.json").is_file()

        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert "request_zip" in combined
        assert "private_key" not in combined
        assert "Send site-3.request.zip to your Project Admin" in combined
        assert "Send site-3.csr" not in combined
        assert "cert sign" not in combined

    @pytest.mark.parametrize(
        "cert_role, expected_cert_type",
        [
            ("org-admin", "org_admin"),
            ("lead", "lead"),
            ("member", "member"),
        ],
    )
    def test_request_user_role_email_creates_user_request_metadata(
        self, tmp_path, capsys, monkeypatch, cert_role, expected_cert_type
    ):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        email = f"{cert_role}@nvidia.com"
        request_dir = tmp_path / email
        participant_path = tmp_path / f"{cert_role}.yaml"
        _write_request_participant(participant_path, name=email, participant_type="admin", role=cert_role)

        _run_cert_cli(
            [
                "request",
                "-p",
                str(participant_path),
                "--out",
                str(request_dir),
            ]
        )

        with zipfile.ZipFile(request_dir / f"{email}.request.zip") as zf:
            request_json = json.loads(zf.read("request.json"))
            site_yaml = yaml.safe_load(zf.read("site.yaml"))

        assert request_json["kind"] == "user"
        assert request_json["name"] == email
        assert request_json["cert_role"] == expected_cert_type
        assert request_json["cert_type"] == expected_cert_type
        assert site_yaml["participants"][0]["role"] == expected_cert_type
        assert site_yaml["participants"][0]["type"] == "admin"
        capsys.readouterr()

    def test_request_user_invalid_role_fails(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        participant_path = tmp_path / "alice.yaml"
        _write_request_participant(
            participant_path,
            name="alice@nvidia.com",
            participant_type="admin",
            role="study-lead",
        )
        with pytest.raises(SystemExit) as exc_info:
            _run_cert_cli(
                [
                    "request",
                    "-p",
                    str(participant_path),
                ]
            )

        assert exc_info.value.code == 4
        err = capsys.readouterr().err
        assert "admin participant role must be one of:" in err
        assert "org-admin" in err
        assert "org_admin" in err

    def test_request_user_requires_email_identity(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        participant_path = tmp_path / "alice.yaml"
        _write_request_participant(participant_path, name="alice", participant_type="admin", role="lead")
        with pytest.raises(SystemExit) as exc_info:
            _run_cert_cli(
                [
                    "request",
                    "-p",
                    str(participant_path),
                ]
            )

        assert exc_info.value.code == 4

    def test_request_rejects_unsafe_project_name(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        participant_path = tmp_path / "site-3.yaml"
        _write_request_participant(participant_path, project="../escape")
        with pytest.raises(SystemExit) as exc_info:
            _run_cert_cli(
                [
                    "request",
                    "-p",
                    str(participant_path),
                ]
            )

        assert exc_info.value.code == 4

    def test_request_rejects_invalid_org_name(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        participant_path = tmp_path / "site-3.yaml"
        _write_request_participant(participant_path, org="bad-org")
        with pytest.raises(SystemExit) as exc_info:
            _run_cert_cli(
                [
                    "request",
                    "-p",
                    str(participant_path),
                ]
            )

        assert exc_info.value.code == 4

    def test_approve_creates_signed_zip_without_private_key_and_final_workflow_hint(
        self, tmp_path, capsys, monkeypatch
    ):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        ca_dir = tmp_path / "ca"
        request_dir = tmp_path / "site-3"
        signed_zip = tmp_path / "site-3.signed.zip"
        profile_path = tmp_path / "project_profile.yaml"
        participant_path = tmp_path / "site-3.yaml"

        handle_cert_init(
            _init_args(profile=_make_profile(tmp_path, "example_project"), org="nvidia", output_dir=str(ca_dir))
        )
        _write_project_profile(profile_path)
        _write_request_participant(participant_path)
        _run_cert_cli(
            [
                "request",
                "-p",
                str(participant_path),
                "--out",
                str(request_dir),
            ]
        )
        capsys.readouterr()

        with patch(
            "nvflare.tool.cert.cert_commands.shutil.copyfile",
            side_effect=AssertionError("site.yaml should be written with nofollow semantics"),
        ):
            _run_cert_cli(
                [
                    "approve",
                    str(request_dir / "site-3.request.zip"),
                    "--ca-dir",
                    str(ca_dir),
                    "--profile",
                    str(profile_path),
                    "--out",
                    str(signed_zip),
                ]
            )

        assert signed_zip.is_file()
        with zipfile.ZipFile(signed_zip) as zf:
            assert sorted(zf.namelist()) == ["rootCA.pem", "signed.json", "signed.json.sig", "site-3.crt", "site.yaml"]
            assert not any(name.endswith(".key") for name in zf.namelist())
            signed_json = json.loads(zf.read("signed.json"))
            cert_bytes = zf.read("site-3.crt")
            rootca_bytes = zf.read("rootCA.pem")
            site_yaml_bytes = zf.read("site.yaml")
        assert signed_json["name"] == "site-3"
        assert signed_json["org"] == "nvidia"
        assert signed_json["kind"] == "site"
        assert signed_json["cert_type"] == "client"
        assert signed_json["project"] == "example_project"
        assert signed_json["server"] == {
            "host": "fl-server",
            "fed_learn_port": 8002,
            "admin_port": 8003,
        }
        assert "request" not in signed_json
        assert "project_name" not in signed_json
        assert "serial_number" not in signed_json
        assert "valid_until" not in signed_json
        assert "cert_role" not in signed_json
        assert "requested_cert_role" not in signed_json
        assert "public_key_sha256" not in signed_json
        assert "csr_sha256" not in signed_json
        assert signed_json["certificate"]["serial"]
        assert signed_json["certificate"]["valid_until"]
        assert "public_key_sha256" not in signed_json["certificate"]
        assert signed_json["hashes"]["certificate_sha256"]
        assert signed_json["hashes"]["public_key_sha256"]
        assert signed_json["hashes"]["certificate_sha256"] == _sha256_bytes(cert_bytes)
        assert signed_json["hashes"]["rootca_sha256"] == _sha256_bytes(rootca_bytes)
        assert signed_json["hashes"]["site_yaml_sha256"] == _sha256_bytes(site_yaml_bytes)
        approve_audit_path = tmp_path / "home" / ".nvflare" / "cert_approves" / f"{signed_json['request_id']}.json"
        assert approve_audit_path.is_file()
        approve_audit = json.loads(approve_audit_path.read_text())
        assert approve_audit["schema_version"] == "1"
        assert approve_audit["request"]["name"] == "site-3"

        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert "rootca_fingerprint_sha256" in combined
        assert cert_fingerprint_sha256(load_crt(str(ca_dir / "rootCA.pem"))) in combined
        assert "signed_zip" in combined
        assert "Return site-3.signed.zip to the requester" in combined
        assert "cert sign" not in combined
        assert "hospital-1.crt" not in combined

    @pytest.mark.parametrize(
        "zip_kwargs",
        [
            {"include_key": True},
            {"traversal": True},
            {"hash_mismatch": True},
            {"omit_fields": ("csr_sha256",)},
            {"omit_fields": ("public_key_sha256",)},
            {"omit_fields": ("site_yaml_sha256",)},
            {"csr_member": "subdir/site-3.csr"},
            {"project": "../escape"},
            {"request_id": "../escape"},
            {"metadata_updates": {"project": 123}},
            {"metadata_updates": {"name": 123}},
            {"metadata_updates": {"org": 123}},
            {"metadata_updates": {"kind": "workspace"}},
            {"org": "bad-org"},
        ],
    )
    def test_approve_rejects_unsafe_or_tampered_request_zip(self, tmp_path, capsys, monkeypatch, zip_kwargs):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        ca_dir = tmp_path / "ca"
        profile_path = tmp_path / "project_profile.yaml"
        handle_cert_init(
            _init_args(profile=_make_profile(tmp_path, "example_project"), org="nvidia", output_dir=str(ca_dir))
        )
        _write_project_profile(profile_path)
        request_zip = _write_request_zip(tmp_path, **zip_kwargs)
        capsys.readouterr()

        with pytest.raises(SystemExit) as exc_info:
            _run_cert_cli(
                [
                    "approve",
                    str(request_zip),
                    "--ca-dir",
                    str(ca_dir),
                    "--profile",
                    str(profile_path),
                ]
            )

        assert exc_info.value.code != 0
        assert not (tmp_path / "home" / ".nvflare" / "escape.json").exists()
        assert "required for distributed provisioning approvals" not in capsys.readouterr().err

    def test_approve_rejects_tampered_site_yaml_hash(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        ca_dir = tmp_path / "ca"
        profile_path = tmp_path / "project_profile.yaml"
        handle_cert_init(
            _init_args(profile=_make_profile(tmp_path, "example_project"), org="nvidia", output_dir=str(ca_dir))
        )
        _write_project_profile(profile_path)
        request_zip = _write_request_zip(tmp_path, metadata_updates={"site_yaml_sha256": "0" * 64})
        capsys.readouterr()

        with pytest.raises(SystemExit) as exc_info:
            _run_cert_cli(
                [
                    "approve",
                    str(request_zip),
                    "--ca-dir",
                    str(ca_dir),
                    "--profile",
                    str(profile_path),
                ]
            )

        assert exc_info.value.code == 4
        assert "site.yaml hash does not match request metadata" in capsys.readouterr().err

    def test_approve_rejects_request_for_different_ca_project(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        ca_dir = tmp_path / "ca"
        handle_cert_init(
            _init_args(profile=_make_profile(tmp_path, "example_project"), org="nvidia", output_dir=str(ca_dir))
        )
        request_zip = _write_request_zip(tmp_path, project="other_project")
        capsys.readouterr()

        with pytest.raises(SystemExit) as exc_info:
            _run_cert_cli(["approve", str(request_zip), "--ca-dir", str(ca_dir), "--profile", str(tmp_path / "p.yaml")])

        assert exc_info.value.code == 4
        assert "does not match CA project" in capsys.readouterr().err

    def test_approve_rejects_profile_project_mismatch(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        ca_dir = tmp_path / "ca"
        handle_cert_init(
            _init_args(profile=_make_profile(tmp_path, "example_project"), org="nvidia", output_dir=str(ca_dir))
        )
        request_zip = _write_request_zip(tmp_path, project="example_project")
        profile_path = tmp_path / "project_profile.yaml"
        _write_project_profile(profile_path, project="other_project")
        capsys.readouterr()

        with pytest.raises(SystemExit) as exc_info:
            _run_cert_cli(["approve", str(request_zip), "--ca-dir", str(ca_dir), "--profile", str(profile_path)])

        assert exc_info.value.code == 4
        assert "does not match project profile" in capsys.readouterr().err

    def test_approve_request_zip_read_error_returns_cli_error(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        ca_dir = tmp_path / "ca"
        handle_cert_init(
            _init_args(profile=_make_profile(tmp_path, "example_project"), org="nvidia", output_dir=str(ca_dir))
        )
        request_zip = _write_request_zip(tmp_path)
        capsys.readouterr()

        with patch("nvflare.tool.cert.cert_commands.zipfile.ZipFile", side_effect=PermissionError("blocked")):
            with pytest.raises(SystemExit) as exc_info:
                _run_cert_cli(
                    ["approve", str(request_zip), "--ca-dir", str(ca_dir), "--profile", str(tmp_path / "p.yaml")]
                )

        assert exc_info.value.code == 4
        captured = capsys.readouterr()
        assert "failed to read request zip" in captured.err
        assert "blocked" in captured.err

    def test_approve_missing_request_zip_uses_request_zip_error_code(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        ca_dir = tmp_path / "ca"
        profile_path = tmp_path / "project_profile.yaml"
        handle_cert_init(
            _init_args(profile=_make_profile(tmp_path, "example_project"), org="nvidia", output_dir=str(ca_dir))
        )
        _write_project_profile(profile_path)
        capsys.readouterr()

        with pytest.raises(SystemExit) as exc_info:
            _run_cert_cli(
                [
                    "approve",
                    str(tmp_path / "missing.request.zip"),
                    "--ca-dir",
                    str(ca_dir),
                    "--profile",
                    str(profile_path),
                ]
            )

        assert exc_info.value.code == 1
        assert "REQUEST_ZIP_NOT_FOUND" in capsys.readouterr().err

    @pytest.mark.parametrize("large_member", ["request.json", "site-3.csr"])
    def test_read_request_zip_rejects_member_exceeding_read_limit(self, tmp_path, capsys, monkeypatch, large_member):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        request_zip = tmp_path / "site-3.request.zip"
        request_json = json.dumps({"name": "site-3"})
        with zipfile.ZipFile(request_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("request.json", request_json if large_member != "request.json" else "x" * 513)
            zf.writestr("site.yaml", "name: site-3\norg: nvidia\ntype: client\n")
            zf.writestr("site-3.csr", b"x" * (513 if large_member == "site-3.csr" else 8))
        monkeypatch.setattr("nvflare.tool.cert.cert_commands._MAX_ZIP_MEMBER_SIZE", 512)
        monkeypatch.setattr(
            "nvflare.tool.cert.cert_commands._safe_zip_names",
            lambda _zf: ["request.json", "site.yaml", "site-3.csr"],
        )
        extract_dir = tmp_path / "extract"
        extract_dir.mkdir()

        with pytest.raises(SystemExit) as exc_info:
            _read_request_zip(str(request_zip), str(extract_dir))

        assert exc_info.value.code == 4
        assert "zip member exceeds size limit" in capsys.readouterr().err

    def test_approve_reuses_validated_csr(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        ca_dir = tmp_path / "ca"
        request_zip = _write_request_zip(tmp_path)
        profile_path = tmp_path / "project_profile.yaml"
        handle_cert_init(
            _init_args(profile=_make_profile(tmp_path, "example_project"), org="nvidia", output_dir=str(ca_dir))
        )
        _write_project_profile(profile_path)
        capsys.readouterr()

        from nvflare.tool.cert import cert_commands

        original_load_csr = cert_commands._load_and_validate_csr
        with patch("nvflare.tool.cert.cert_commands._load_and_validate_csr", wraps=original_load_csr) as load_csr:
            _run_cert_cli(["approve", str(request_zip), "--ca-dir", str(ca_dir), "--profile", str(profile_path)])

        assert load_csr.call_count == 1
        capsys.readouterr()

    def test_approve_returns_when_request_metadata_validation_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        ca_dir = tmp_path / "ca"
        request_zip = _write_request_zip(tmp_path)
        handle_cert_init(
            _init_args(profile=_make_profile(tmp_path, "example_project"), org="nvidia", output_dir=str(ca_dir))
        )

        with patch("nvflare.tool.cert.cert_commands._validate_request_metadata", return_value=None):
            with patch("nvflare.tool.cert.cert_commands.sign_csr_files") as sign_csr:
                _run_cert_cli(
                    ["approve", str(request_zip), "--ca-dir", str(ca_dir), "--profile", str(tmp_path / "p.yaml")]
                )

        sign_csr.assert_not_called()

    def test_approve_returns_when_signed_zip_writer_reports_failure_under_mocked_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        ca_dir = tmp_path / "ca"
        request_zip = _write_request_zip(tmp_path)
        signed_zip = tmp_path / "site-3.signed.zip"
        profile_path = tmp_path / "project_profile.yaml"
        handle_cert_init(
            _init_args(profile=_make_profile(tmp_path, "example_project"), org="nvidia", output_dir=str(ca_dir))
        )
        _write_project_profile(profile_path)

        with patch("nvflare.tool.cert.cert_commands._write_zip_nofollow", return_value=False):
            with patch("nvflare.tool.cert.cert_commands._try_write_approve_audit") as write_audit:
                rc = _run_cert_cli(
                    [
                        "approve",
                        str(request_zip),
                        "--ca-dir",
                        str(ca_dir),
                        "--profile",
                        str(profile_path),
                        "--out",
                        str(signed_zip),
                    ]
                )

        assert rc == 1
        write_audit.assert_not_called()
        assert not signed_zip.exists()

    def test_read_request_zip_returns_after_member_mismatch_when_error_is_mocked(self, tmp_path):
        request_zip = tmp_path / "site-3.request.zip"
        with zipfile.ZipFile(request_zip, "w") as zf:
            zf.writestr("request.json", json.dumps({"name": "site-3"}))
            zf.writestr("site.yaml", "name: site-3\norg: nvidia\ntype: client\n")
        extract_dir = tmp_path / "extract"
        extract_dir.mkdir()

        with patch("nvflare.tool.cert.cert_commands.output_error_message") as output_error:
            request_meta = _read_request_zip(str(request_zip), str(extract_dir))

        output_error.assert_called_once()
        assert request_meta is None
        assert not (extract_dir / "site-3.csr").exists()

    def test_read_request_zip_returns_after_missing_member_when_error_is_mocked(self, tmp_path):
        request_zip = tmp_path / "site-3.request.zip"
        with zipfile.ZipFile(request_zip, "w") as zf:
            zf.writestr("site.yaml", "name: site-3\norg: nvidia\ntype: client\n")
        extract_dir = tmp_path / "extract"
        extract_dir.mkdir()

        with patch("nvflare.tool.cert.cert_commands.output_error_message") as output_error:
            request_meta = _read_request_zip(str(request_zip), str(extract_dir))

        output_error.assert_called_once()
        assert request_meta is None

    def test_read_request_zip_returns_after_member_size_error_when_error_is_mocked(self, tmp_path, monkeypatch):
        monkeypatch.setattr("nvflare.tool.cert.cert_commands._MAX_ZIP_MEMBER_SIZE", 512)
        request_zip = tmp_path / "site-3.request.zip"
        with zipfile.ZipFile(request_zip, "w") as zf:
            zf.writestr("request.json", "x" * 513)
            zf.writestr("site.yaml", "name: site-3\norg: nvidia\ntype: client\n")
            zf.writestr("site-3.csr", b"csr")
        monkeypatch.setattr(
            "nvflare.tool.cert.cert_commands._safe_zip_names",
            lambda _zf: ["request.json", "site.yaml", "site-3.csr"],
        )
        extract_dir = tmp_path / "extract"
        extract_dir.mkdir()

        with patch("nvflare.tool.cert.cert_commands.output_error_message") as output_error:
            request_meta = _read_request_zip(str(request_zip), str(extract_dir))

        output_error.assert_called_once()
        assert "zip member exceeds size limit" in output_error.call_args.kwargs["detail"]
        assert request_meta is None
        assert not (extract_dir / "request.json").exists()

    def test_read_request_zip_returns_after_extract_error_when_error_is_mocked(self, tmp_path):
        request_zip = _write_request_zip(tmp_path)
        extract_dir = tmp_path / "extract"
        extract_dir.mkdir()

        with patch("nvflare.tool.cert.cert_commands._write_file_nofollow", side_effect=OSError("blocked")):
            with patch("nvflare.tool.cert.cert_commands.output_error_message") as output_error:
                request_meta = _read_request_zip(str(request_zip), str(extract_dir))

        output_error.assert_called_once()
        assert "failed to read request zip" in output_error.call_args.kwargs["detail"]
        assert request_meta is None
