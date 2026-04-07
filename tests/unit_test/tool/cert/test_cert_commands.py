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
import json
import os
import platform
import stat
import sys

import pytest
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.x509.extensions import BasicConstraints
from cryptography.x509.oid import NameOID

# Ensure parsers are initialized by importing cert_cli (registers module-level parser refs)
import nvflare.tool.cert.cert_cli  # noqa: F401
from nvflare.lighter.utils import load_crt
from nvflare.tool.cert.cert_commands import handle_cert_csr, handle_cert_init, handle_cert_sign

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _init_args(**kwargs):
    defaults = dict(
        project="TestProject",
        output_dir=None,
        org=None,
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


def _sign_args(**kwargs):
    defaults = dict(
        csr_path=None,
        ca_dir=None,
        output_dir=None,
        cert_type=None,
        valid_days=1095,
        force=False,
        schema=False,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _run_init(tmp_path, **kwargs):
    """Run cert init in tmp_path; returns return code."""
    args = _init_args(output_dir=str(tmp_path), **kwargs)
    return handle_cert_init(args)


def _run_csr(tmp_path, name="hospital-1", **kwargs):
    args = _csr_args(name=name, output_dir=str(tmp_path), **kwargs)
    return handle_cert_csr(args)


def _load_csr_file(path):
    with open(path, "rb") as f:
        return x509.load_pem_x509_csr(f.read(), default_backend())


# ---------------------------------------------------------------------------
# cert init tests
# ---------------------------------------------------------------------------


class TestCertInit:
    def test_basic_init(self, tmp_path):
        rc = _run_init(tmp_path)
        assert rc == 0
        assert (tmp_path / "rootCA.pem").exists()
        assert (tmp_path / "rootCA.key").exists()
        assert (tmp_path / "ca.json").exists()

    @pytest.mark.skipif(platform.system() == "Windows", reason="chmod not meaningful on Windows")
    def test_ca_key_permissions(self, tmp_path):
        _run_init(tmp_path)
        key_path = tmp_path / "rootCA.key"
        mode = stat.S_IMODE(os.stat(str(key_path)).st_mode)
        assert mode == 0o600

    def test_ca_json_content(self, tmp_path):
        _run_init(tmp_path, project="MyProject")
        with open(str(tmp_path / "ca.json")) as f:
            meta = json.load(f)
        assert meta["project"] == "MyProject"
        assert meta["next_serial"] == 2
        assert "created_at" in meta

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
        # Key should be different (new key pair generated)
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

    def test_agent_mode_json_envelope(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setenv("NVFLARE_CLI_MODE", "agent")
        rc = handle_cert_init(_init_args(output_dir=str(tmp_path)))
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["schema_version"] == "1"
        assert data["status"] == "ok"
        assert "ca_cert" in data["data"]
        assert "ca_key" in data["data"]
        assert "project" in data["data"]
        assert "valid_until" in data["data"]

    def test_force_flag_overwrites_existing_ca(self, tmp_path):
        (tmp_path / "rootCA.key").write_bytes(b"old-key")
        rc = handle_cert_init(_init_args(output_dir=str(tmp_path), force=True))
        assert rc == 0
        assert (tmp_path / "rootCA.key").exists()

    def test_output_dir_created(self, tmp_path):
        new_dir = str(tmp_path / "new" / "subdir")
        args = _init_args(output_dir=new_dir)
        rc = handle_cert_init(args)
        assert rc == 0
        assert os.path.exists(new_dir)
        assert os.path.exists(os.path.join(new_dir, "rootCA.pem"))

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

    def test_csr_no_role_in_subject(self, tmp_path):
        """cert csr no longer embeds a role — cert sign's -t is authoritative."""
        _run_csr(tmp_path, name="h1")
        csr = _load_csr_file(str(tmp_path / "h1.csr"))
        role_attrs = csr.subject.get_attributes_for_oid(NameOID.UNSTRUCTURED_NAME)
        assert len(role_attrs) == 0

    def test_org_in_subject(self, tmp_path):
        args = _csr_args(name="h1", output_dir=str(tmp_path), org="ACME")
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
        rc = handle_cert_csr(_csr_args(name="h1", output_dir=str(tmp_path), force=True))
        assert rc == 0
        capsys.readouterr()  # discard output
        assert b"PRIVATE KEY" in (tmp_path / "h1.key").read_bytes()

    def test_output_dir_created(self, tmp_path):
        new_dir = str(tmp_path / "newdir")
        rc = handle_cert_csr(_csr_args(name="h1", output_dir=new_dir))
        assert rc == 0
        assert os.path.exists(new_dir)

    def test_agent_mode_json_envelope(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setenv("NVFLARE_CLI_MODE", "agent")
        handle_cert_csr(_csr_args(name="h1", output_dir=str(tmp_path)))
        data = json.loads(capsys.readouterr().out)
        assert data["schema_version"] == "1"
        assert data["status"] == "ok"
        assert "key" in data["data"]
        assert "csr" in data["data"]

    def test_agent_mode_no_key_material_in_output(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setenv("NVFLARE_CLI_MODE", "agent")
        handle_cert_csr(_csr_args(name="h1", output_dir=str(tmp_path)))
        out = capsys.readouterr().out
        assert "BEGIN RSA PRIVATE KEY" not in out
        assert "BEGIN PRIVATE KEY" not in out

    def test_schema_flag(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["nvflare", "cert", "csr", "--schema"])
        with pytest.raises(SystemExit) as exc_info:
            handle_cert_csr(_csr_args())
        assert exc_info.value.code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["schema_version"] == "1"
        assert data["command"] == "nvflare cert csr"


# ---------------------------------------------------------------------------
# cert sign tests
# ---------------------------------------------------------------------------


def _setup_ca(tmp_path):
    """Run cert init and return ca_dir path."""
    ca_dir = str(tmp_path / "ca")
    args = _init_args(output_dir=ca_dir)
    handle_cert_init(args)
    return ca_dir


def _setup_csr(tmp_path, name="hospital-1"):
    """Run cert csr and return csr_path."""
    csr_dir = str(tmp_path / "csr")
    os.makedirs(csr_dir, exist_ok=True)
    args = _csr_args(name=name, output_dir=csr_dir)
    handle_cert_csr(args)
    return os.path.join(csr_dir, f"{name}.csr")


class TestCertSign:
    def test_basic_sign(self, tmp_path):
        ca_dir = _setup_ca(tmp_path)
        csr_path = _setup_csr(tmp_path)
        out_dir = str(tmp_path / "signed")
        args = _sign_args(csr_path=csr_path, ca_dir=ca_dir, output_dir=out_dir, cert_type="client")
        rc = handle_cert_sign(args)
        assert rc == 0

    def test_sign_output_files(self, tmp_path):
        ca_dir = _setup_ca(tmp_path)
        csr_path = _setup_csr(tmp_path)  # name="hospital-1"
        out_dir = str(tmp_path / "signed")
        args = _sign_args(csr_path=csr_path, ca_dir=ca_dir, output_dir=out_dir, cert_type="client")
        handle_cert_sign(args)
        assert os.path.exists(os.path.join(out_dir, "hospital-1.crt"))
        assert os.path.exists(os.path.join(out_dir, "rootCA.pem"))

    def test_sign_updates_ca_json(self, tmp_path):
        ca_dir = _setup_ca(tmp_path)
        csr_path = _setup_csr(tmp_path)
        out_dir = str(tmp_path / "signed")

        def _peek_serial(ca_json_path):
            with open(ca_json_path) as _f:
                return json.load(_f).get("next_serial", 2)

        serial_before = _peek_serial(os.path.join(ca_dir, "ca.json"))
        args = _sign_args(csr_path=csr_path, ca_dir=ca_dir, output_dir=out_dir, cert_type="client")
        handle_cert_sign(args)
        serial_after = _peek_serial(os.path.join(ca_dir, "ca.json"))
        assert serial_after == serial_before + 1

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
        open(os.path.join(out_dir, "hospital-1.crt"), "w").close()
        args = _sign_args(csr_path=csr_path, ca_dir=ca_dir, output_dir=out_dir, cert_type="client")
        with pytest.raises(SystemExit) as exc_info:
            handle_cert_sign(args)
        assert exc_info.value.code == 1

    def test_sign_existing_cert_with_force(self, tmp_path):
        ca_dir = _setup_ca(tmp_path)
        csr_path = _setup_csr(tmp_path)
        out_dir = str(tmp_path / "signed")
        os.makedirs(out_dir, exist_ok=True)
        open(os.path.join(out_dir, "hospital-1.crt"), "w").close()
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

    def test_sign_agent_mode_json_envelope(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setenv("NVFLARE_CLI_MODE", "agent")
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
        assert "signed_cert" in data["data"]
        assert "rootca" in data["data"]
        assert "serial" in data["data"]

    def test_sign_force_overwrites_existing_cert(self, tmp_path):
        ca_dir = _setup_ca(tmp_path)
        csr_path = _setup_csr(tmp_path)
        out_dir = str(tmp_path / "signed")
        os.makedirs(out_dir, exist_ok=True)
        open(os.path.join(out_dir, "hospital-1.crt"), "w").close()
        args = _sign_args(csr_path=csr_path, ca_dir=ca_dir, output_dir=out_dir, cert_type="client", force=True)
        assert handle_cert_sign(args) == 0

    def test_sign_schema_output(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["nvflare", "cert", "sign", "--schema"])
        with pytest.raises(SystemExit) as exc_info:
            handle_cert_sign(_sign_args())
        assert exc_info.value.code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["schema_version"] == "1"
        assert data["command"] == "nvflare cert sign"

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

    def test_sign_multiple_certs_serial_increments(self, tmp_path):
        ca_dir = _setup_ca(tmp_path)
        csr1 = _setup_csr(tmp_path / "csr1", name="client1")
        csr2 = _setup_csr(tmp_path / "csr2", name="client2")

        def _peek_serial(ca_json_path):
            with open(ca_json_path) as _f:
                return json.load(_f).get("next_serial", 2)

        out1 = str(tmp_path / "signed1")
        out2 = str(tmp_path / "signed2")

        args1 = _sign_args(csr_path=csr1, ca_dir=ca_dir, output_dir=out1, cert_type="client")
        handle_cert_sign(args1)
        serial_mid = _peek_serial(os.path.join(ca_dir, "ca.json"))

        args2 = _sign_args(csr_path=csr2, ca_dir=ca_dir, output_dir=out2, cert_type="client")
        handle_cert_sign(args2)
        serial_end = _peek_serial(os.path.join(ca_dir, "ca.json"))

        assert serial_end == serial_mid + 1

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
            now = datetime.datetime.utcnow()
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
            now = datetime.datetime.utcnow()
        days_remaining = (not_after - now).days
        assert 1093 <= days_remaining <= 1097

    def test_serial_incremented_atomically(self, tmp_path):
        """_claim_serial reads and increments in one step; serial advances exactly once per sign."""
        ca_dir = _setup_ca(tmp_path)

        def _peek(ca_json_path):
            with open(ca_json_path) as _f:
                return json.load(_f).get("next_serial", 2)

        ca_json = os.path.join(ca_dir, "ca.json")
        initial = _peek(ca_json)

        csr1 = _setup_csr(tmp_path / "csr1", name="site-a")
        csr2 = _setup_csr(tmp_path / "csr2", name="site-b")
        csr3 = _setup_csr(tmp_path / "csr3", name="site-c")

        for i, csr in enumerate([csr1, csr2, csr3]):
            out = str(tmp_path / f"out{i}")
            args = _sign_args(csr_path=csr, ca_dir=ca_dir, output_dir=out, cert_type="client")
            handle_cert_sign(args)

        assert _peek(ca_json) == initial + 3


# ---------------------------------------------------------------------------
# cert csr -t: proposed role embedded in CSR
# ---------------------------------------------------------------------------


class TestCertCsrWithRole:
    def test_csr_role_embedded_when_type_given(self, tmp_path):
        """cert csr -t lead embeds 'lead' in CSR UNSTRUCTURED_NAME."""
        args = _csr_args(name="alice", output_dir=str(tmp_path), cert_type="lead")
        handle_cert_csr(args)
        csr = _load_csr_file(str(tmp_path / "alice.csr"))
        role_attrs = csr.subject.get_attributes_for_oid(NameOID.UNSTRUCTURED_NAME)
        assert len(role_attrs) == 1
        assert role_attrs[0].value == "lead"

    def test_csr_no_role_when_type_absent(self, tmp_path):
        """cert csr without -t produces no UNSTRUCTURED_NAME in CSR."""
        args = _csr_args(name="h1", output_dir=str(tmp_path))
        handle_cert_csr(args)
        csr = _load_csr_file(str(tmp_path / "h1.csr"))
        role_attrs = csr.subject.get_attributes_for_oid(NameOID.UNSTRUCTURED_NAME)
        assert len(role_attrs) == 0

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
# cert sign: read type from CSR when -t is omitted
# ---------------------------------------------------------------------------


class TestCertSignReadsTypeFromCsr:
    def test_sign_reads_type_from_csr_when_t_absent(self, tmp_path):
        """cert sign without -t reads role from CSR UNSTRUCTURED_NAME."""
        ca_dir = _setup_ca(tmp_path)
        # Generate CSR with role embedded
        csr_dir = str(tmp_path / "csr")
        os.makedirs(csr_dir, exist_ok=True)
        args = _csr_args(name="alice", output_dir=csr_dir, cert_type="lead")
        handle_cert_csr(args)
        csr_path = os.path.join(csr_dir, "alice.csr")

        out_dir = str(tmp_path / "signed")
        sign_args = _sign_args(csr_path=csr_path, ca_dir=ca_dir, output_dir=out_dir)
        rc = handle_cert_sign(sign_args)
        assert rc == 0
        cert = load_crt(os.path.join(out_dir, "alice.crt"))
        role_attrs = cert.subject.get_attributes_for_oid(NameOID.UNSTRUCTURED_NAME)
        assert role_attrs[0].value == "lead"

    def test_sign_t_overrides_csr_role(self, tmp_path):
        """cert sign -t overrides the role proposed in the CSR."""
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

    def test_sign_no_t_no_csr_role_fails(self, tmp_path):
        """cert sign without -t and CSR has no role → error (INVALID_ARGS)."""
        ca_dir = _setup_ca(tmp_path)
        csr_path = _setup_csr(tmp_path)  # no role in CSR
        out_dir = str(tmp_path / "signed")
        args = _sign_args(csr_path=csr_path, ca_dir=ca_dir, output_dir=out_dir)  # cert_type=None
        with pytest.raises(SystemExit) as exc_info:
            handle_cert_sign(args)
        assert exc_info.value.code == 2


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

    def test_file_not_found_exits_1(self, tmp_path):
        from nvflare.tool.cert.cert_commands import _load_single_site_yaml

        with pytest.raises(SystemExit) as exc_info:
            _load_single_site_yaml(str(tmp_path / "no-such.yml"))
        assert exc_info.value.code == 1

    def test_not_a_mapping_exits_2(self, tmp_path):
        from nvflare.tool.cert.cert_commands import _load_single_site_yaml

        path = self._write_yaml(tmp_path, "- a\n- b\n")
        with pytest.raises(SystemExit) as exc_info:
            _load_single_site_yaml(path)
        assert exc_info.value.code == 2

    def test_missing_name_exits_2(self, tmp_path):
        from nvflare.tool.cert.cert_commands import _load_single_site_yaml

        path = self._write_yaml(tmp_path, "org: ACME\ntype: client\n")
        with pytest.raises(SystemExit) as exc_info:
            _load_single_site_yaml(path)
        assert exc_info.value.code == 2

    def test_missing_org_exits_2(self, tmp_path):
        from nvflare.tool.cert.cert_commands import _load_single_site_yaml

        path = self._write_yaml(tmp_path, "name: h1\ntype: client\n")
        with pytest.raises(SystemExit) as exc_info:
            _load_single_site_yaml(path)
        assert exc_info.value.code == 2

    def test_missing_type_exits_2(self, tmp_path):
        from nvflare.tool.cert.cert_commands import _load_single_site_yaml

        path = self._write_yaml(tmp_path, "name: h1\norg: ACME\n")
        with pytest.raises(SystemExit) as exc_info:
            _load_single_site_yaml(path)
        assert exc_info.value.code == 2

    def test_empty_name_exits_2(self, tmp_path):
        from nvflare.tool.cert.cert_commands import _load_single_site_yaml

        path = self._write_yaml(tmp_path, "name: ''\norg: ACME\ntype: client\n")
        with pytest.raises(SystemExit) as exc_info:
            _load_single_site_yaml(path)
        assert exc_info.value.code == 2


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

    def test_csr_org_taken_from_yaml(self, tmp_path):
        site_file = self._write_site_yaml(tmp_path, org="NVIDIA")
        out_dir = tmp_path / "out"
        args = _csr_args(name=None, project_file=site_file, output_dir=str(out_dir))
        handle_cert_csr(args)
        csr = _load_csr_file(str(out_dir / "hospital-1.csr"))
        org_attrs = csr.subject.get_attributes_for_oid(NameOID.ORGANIZATION_NAME)
        assert len(org_attrs) == 1
        assert org_attrs[0].value == "NVIDIA"

    def test_missing_output_dir_exits_2(self, tmp_path, monkeypatch):
        monkeypatch.setenv("NVFLARE_CLI_MODE", "agent")
        site_file = self._write_site_yaml(tmp_path)
        args = _csr_args(name=None, project_file=site_file, output_dir=None)
        with pytest.raises(SystemExit) as exc_info:
            handle_cert_csr(args)
        assert exc_info.value.code == 2

    def test_mutual_exclusivity_with_name_exits_2(self, tmp_path, monkeypatch):
        monkeypatch.setenv("NVFLARE_CLI_MODE", "agent")
        site_file = self._write_site_yaml(tmp_path)
        out_dir = tmp_path / "out"
        args = _csr_args(name="conflicting", project_file=site_file, output_dir=str(out_dir))
        with pytest.raises(SystemExit) as exc_info:
            handle_cert_csr(args)
        assert exc_info.value.code == 2

    def test_mutual_exclusivity_with_org_exits_2(self, tmp_path, monkeypatch):
        monkeypatch.setenv("NVFLARE_CLI_MODE", "agent")
        site_file = self._write_site_yaml(tmp_path)
        out_dir = tmp_path / "out"
        args = _csr_args(name=None, org="ConflictingOrg", project_file=site_file, output_dir=str(out_dir))
        with pytest.raises(SystemExit) as exc_info:
            handle_cert_csr(args)
        assert exc_info.value.code == 2

    def test_mutual_exclusivity_with_type_exits_2(self, tmp_path, monkeypatch):
        monkeypatch.setenv("NVFLARE_CLI_MODE", "agent")
        site_file = self._write_site_yaml(tmp_path)
        out_dir = tmp_path / "out"
        args = _csr_args(name=None, cert_type="server", project_file=site_file, output_dir=str(out_dir))
        with pytest.raises(SystemExit) as exc_info:
            handle_cert_csr(args)
        assert exc_info.value.code == 2

    def test_nonexistent_project_file_exits_1(self, tmp_path, monkeypatch):
        monkeypatch.setenv("NVFLARE_CLI_MODE", "agent")
        out_dir = tmp_path / "out"
        args = _csr_args(name=None, project_file=str(tmp_path / "no-such.yml"), output_dir=str(out_dir))
        with pytest.raises(SystemExit) as exc_info:
            handle_cert_csr(args)
        assert exc_info.value.code == 1

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
        monkeypatch.setenv("NVFLARE_CLI_MODE", "agent")
        out_dir = tmp_path / "out"
        args = _csr_args(
            name="conflict",
            project_file=str(tmp_path / "no-such.yml"),
            output_dir=str(out_dir),
        )
        with pytest.raises(SystemExit) as exc_info:
            handle_cert_csr(args)
        # Must be INVALID_ARGS exit 2 (mutual exclusivity), not file-not-found exit 1
        assert exc_info.value.code == 2
