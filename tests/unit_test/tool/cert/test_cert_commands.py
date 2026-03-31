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

import pytest
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.x509.extensions import BasicConstraints
from cryptography.x509.oid import NameOID

# Ensure parsers are initialized by importing cert_cli (registers module-level parser refs)
import nvflare.tool.cert.cert_cli  # noqa: F401
from nvflare.lighter.utils import load_crt
from nvflare.tool.cert.cert_commands import (
    _build_signed_cert,
    _generate_csr,
    _increment_serial,
    _read_serial,
    handle_cert_csr,
    handle_cert_init,
    handle_cert_sign,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _init_args(**kwargs):
    defaults = dict(
        name="TestProject",
        output_dir=None,
        org=None,
        force=False,
        output_fmt=None,
        schema=False,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _csr_args(**kwargs):
    defaults = dict(
        name="hospital-1",
        cert_type="client",
        output_dir=None,
        org=None,
        force=False,
        output_fmt=None,
        schema=False,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _sign_args(**kwargs):
    defaults = dict(
        csr_path=None,
        ca_dir=None,
        output_dir=None,
        cert_type="client",
        force=False,
        output_fmt=None,
        schema=False,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _run_init(tmp_path, **kwargs):
    """Run cert init in tmp_path; returns return code."""
    args = _init_args(output_dir=str(tmp_path), **kwargs)
    return handle_cert_init(args)


def _run_csr(tmp_path, name="hospital-1", cert_type="client", **kwargs):
    args = _csr_args(name=name, cert_type=cert_type, output_dir=str(tmp_path), **kwargs)
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
        _run_init(tmp_path, name="MyProject")
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

    def test_schema_output(self, tmp_path, capsys):
        args = _init_args(output_dir=str(tmp_path), schema=True)
        rc = handle_cert_init(args)
        assert rc == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["schema_version"] == "1"
        assert data["command"] == "nvflare cert init"
        assert len(data["args"]) > 0

    def test_output_json_success(self, tmp_path, capsys):
        args = _init_args(output_dir=str(tmp_path), output_fmt="json")
        rc = handle_cert_init(args)
        assert rc == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["schema_version"] == "1"
        assert data["status"] == "ok"
        assert "ca_cert" in data["data"]
        assert "ca_key" in data["data"]
        assert "project" in data["data"]
        assert "valid_until" in data["data"]

    def test_output_json_implies_force(self, tmp_path):
        # Pre-create rootCA.key — json mode should NOT error
        (tmp_path / "rootCA.key").write_bytes(b"old-key")
        args = _init_args(output_dir=str(tmp_path), output_fmt="json")
        rc = handle_cert_init(args)
        assert rc == 0
        # New key should be written
        assert (tmp_path / "rootCA.key").exists()

    def test_output_dir_created(self, tmp_path):
        new_dir = str(tmp_path / "new" / "subdir")
        args = _init_args(output_dir=new_dir)
        rc = handle_cert_init(args)
        assert rc == 0
        assert os.path.exists(new_dir)
        assert os.path.exists(os.path.join(new_dir, "rootCA.pem"))

    def test_ca_cert_subject_cn_matches_name(self, tmp_path):
        _run_init(tmp_path, name="FederationX")
        cert = load_crt(str(tmp_path / "rootCA.pem"))
        cn = cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value
        assert cn == "FederationX"

    def test_ca_cert_validity_approx_10_years(self, tmp_path):
        import datetime

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
    def test_basic_client_csr(self, tmp_path):
        rc = _run_csr(tmp_path, name="hospital-1", cert_type="client")
        assert rc == 0
        assert (tmp_path / "hospital-1.key").exists()
        assert (tmp_path / "hospital-1.csr").exists()

    def test_basic_server_csr(self, tmp_path):
        rc = _run_csr(tmp_path, name="fl-server", cert_type="server")
        assert rc == 0
        assert (tmp_path / "fl-server.key").exists()
        assert (tmp_path / "fl-server.csr").exists()

    def test_basic_lead_csr(self, tmp_path):
        rc = _run_csr(tmp_path, name="alice", cert_type="lead")
        assert rc == 0
        assert (tmp_path / "alice.key").exists()
        assert (tmp_path / "alice.csr").exists()

    @pytest.mark.skipif(platform.system() == "Windows", reason="chmod not meaningful on Windows")
    def test_key_permissions(self, tmp_path):
        _run_csr(tmp_path, name="h1", cert_type="client")
        key_path = str(tmp_path / "h1.key")
        mode = stat.S_IMODE(os.stat(key_path).st_mode)
        assert mode == 0o600

    def test_no_cert_generated(self, tmp_path):
        _run_csr(tmp_path, name="h1", cert_type="client")
        assert not (tmp_path / "h1.crt").exists()

    def test_csr_valid_x509(self, tmp_path):
        _run_csr(tmp_path, name="h1", cert_type="client")
        csr = _load_csr_file(str(tmp_path / "h1.csr"))
        assert csr.is_signature_valid

    def test_csr_cn_matches_name(self, tmp_path):
        _run_csr(tmp_path, name="hospital-1", cert_type="client")
        csr = _load_csr_file(str(tmp_path / "hospital-1.csr"))
        cn = csr.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value
        assert cn == "hospital-1"

    def test_csr_type_in_subject(self, tmp_path):
        _run_csr(tmp_path, name="h1", cert_type="lead")
        csr = _load_csr_file(str(tmp_path / "h1.csr"))
        role_attrs = csr.subject.get_attributes_for_oid(NameOID.UNSTRUCTURED_NAME)
        assert len(role_attrs) == 1
        assert role_attrs[0].value == "lead"

    def test_org_in_subject(self, tmp_path):
        args = _csr_args(name="h1", cert_type="client", output_dir=str(tmp_path), org="ACME")
        handle_cert_csr(args)
        csr = _load_csr_file(str(tmp_path / "h1.csr"))
        org_attrs = csr.subject.get_attributes_for_oid(NameOID.ORGANIZATION_NAME)
        assert len(org_attrs) == 1
        assert org_attrs[0].value == "ACME"

    def test_no_org_in_subject(self, tmp_path):
        _run_csr(tmp_path, name="h1", cert_type="client")
        csr = _load_csr_file(str(tmp_path / "h1.csr"))
        org_attrs = csr.subject.get_attributes_for_oid(NameOID.ORGANIZATION_NAME)
        assert len(org_attrs) == 0

    def test_existing_key_no_force(self, tmp_path):
        (tmp_path / "h1.key").write_bytes(b"old-key")
        with pytest.raises(SystemExit) as exc_info:
            _run_csr(tmp_path, name="h1", cert_type="client")
        assert exc_info.value.code == 1

    def test_existing_key_with_force(self, tmp_path):
        (tmp_path / "h1.key").write_bytes(b"old-key")
        rc = _run_csr(tmp_path, name="h1", cert_type="client", force=True)
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
        _run_csr(tmp_path, name="h1", cert_type="client", force=True)
        bak_dirs = list((tmp_path / ".bak").iterdir())
        assert len(bak_dirs) >= 1
        bak_dir = bak_dirs[0]
        assert (bak_dir / "h1.key").exists()
        assert (bak_dir / "h1.csr").exists()

    def test_output_json_implies_force(self, tmp_path, capsys):
        (tmp_path / "h1.key").write_bytes(b"old-key")
        args = _csr_args(name="h1", cert_type="client", output_dir=str(tmp_path), output_fmt="json")
        rc = handle_cert_csr(args)
        assert rc == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["status"] == "ok"

    def test_output_dir_created(self, tmp_path):
        new_dir = str(tmp_path / "newdir")
        args = _csr_args(name="h1", cert_type="client", output_dir=new_dir)
        rc = handle_cert_csr(args)
        assert rc == 0
        assert os.path.exists(new_dir)

    def test_output_json_envelope(self, tmp_path, capsys):
        args = _csr_args(name="h1", cert_type="client", output_dir=str(tmp_path), output_fmt="json")
        handle_cert_csr(args)
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["schema_version"] == "1"
        assert data["status"] == "ok"

    def test_output_json_no_key_material(self, tmp_path, capsys):
        args = _csr_args(name="h1", cert_type="client", output_dir=str(tmp_path), output_fmt="json")
        handle_cert_csr(args)
        out = capsys.readouterr().out
        assert "BEGIN RSA PRIVATE KEY" not in out
        assert "BEGIN PRIVATE KEY" not in out

    def test_schema_flag(self, tmp_path, capsys):
        args = _csr_args(output_dir=str(tmp_path), schema=True)
        with pytest.raises(SystemExit) as exc_info:
            handle_cert_csr(args)
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["schema_version"] == "1"
        assert len(data["args"]) > 0

    def test_schema_contains_type_choices(self, tmp_path, capsys):
        args = _csr_args(output_dir=str(tmp_path), schema=True)
        with pytest.raises(SystemExit):
            handle_cert_csr(args)
        out = capsys.readouterr().out
        data = json.loads(out)
        type_arg = next(a for a in data["args"] if "--type" in a["name"] or a["name"] == "-t")
        assert set(type_arg["choices"]) == {"client", "server", "org_admin", "lead", "member"}

    def test_output_quiet(self, tmp_path, capsys):
        args = _csr_args(name="h1", cert_type="client", output_dir=str(tmp_path), output_fmt="quiet")
        rc = handle_cert_csr(args)
        assert rc == 0
        assert (tmp_path / "h1.key").exists()


# ---------------------------------------------------------------------------
# cert sign tests
# ---------------------------------------------------------------------------


def _setup_ca(tmp_path):
    """Run cert init and return ca_dir path. Uses quiet output to avoid polluting capsys."""
    ca_dir = str(tmp_path / "ca")
    args = _init_args(output_dir=ca_dir, output_fmt="quiet")
    handle_cert_init(args)
    return ca_dir


def _setup_csr(tmp_path, name="hospital-1", cert_type="client"):
    """Run cert csr and return csr_path. Uses quiet output to avoid polluting capsys."""
    csr_dir = str(tmp_path / "csr")
    os.makedirs(csr_dir, exist_ok=True)
    args = _csr_args(name=name, cert_type=cert_type, output_dir=csr_dir, output_fmt="quiet")
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
        csr_path = _setup_csr(tmp_path)
        out_dir = str(tmp_path / "signed")
        args = _sign_args(csr_path=csr_path, ca_dir=ca_dir, output_dir=out_dir, cert_type="client")
        handle_cert_sign(args)
        assert os.path.exists(os.path.join(out_dir, "client.crt"))
        assert os.path.exists(os.path.join(out_dir, "rootCA.pem"))

    def test_sign_updates_ca_json(self, tmp_path):
        ca_dir = _setup_ca(tmp_path)
        csr_path = _setup_csr(tmp_path)
        out_dir = str(tmp_path / "signed")
        # Read serial before
        serial_before = _read_serial(os.path.join(ca_dir, "ca.json"))
        args = _sign_args(csr_path=csr_path, ca_dir=ca_dir, output_dir=out_dir, cert_type="client")
        handle_cert_sign(args)
        serial_after = _read_serial(os.path.join(ca_dir, "ca.json"))
        assert serial_after == serial_before + 1

    def test_sign_cert_is_valid_x509(self, tmp_path):
        ca_dir = _setup_ca(tmp_path)
        csr_path = _setup_csr(tmp_path)
        out_dir = str(tmp_path / "signed")
        args = _sign_args(csr_path=csr_path, ca_dir=ca_dir, output_dir=out_dir, cert_type="client")
        handle_cert_sign(args)
        cert = load_crt(os.path.join(out_dir, "client.crt"))
        assert cert is not None

    def test_sign_cert_not_ca(self, tmp_path):
        ca_dir = _setup_ca(tmp_path)
        csr_path = _setup_csr(tmp_path)
        out_dir = str(tmp_path / "signed")
        args = _sign_args(csr_path=csr_path, ca_dir=ca_dir, output_dir=out_dir, cert_type="client")
        handle_cert_sign(args)
        cert = load_crt(os.path.join(out_dir, "client.crt"))
        bc = cert.extensions.get_extension_for_class(BasicConstraints)
        assert bc.value.ca is False

    def test_sign_cert_type_authoritative(self, tmp_path):
        """The -t arg controls UNSTRUCTURED_NAME in signed cert, not the CSR's value."""
        ca_dir = _setup_ca(tmp_path)
        # CSR says "client" but we sign as "lead"
        csr_path = _setup_csr(tmp_path, name="alice", cert_type="client")
        out_dir = str(tmp_path / "signed")
        args = _sign_args(csr_path=csr_path, ca_dir=ca_dir, output_dir=out_dir, cert_type="lead")
        handle_cert_sign(args)
        cert = load_crt(os.path.join(out_dir, "lead.crt"))
        role_attrs = cert.subject.get_attributes_for_oid(NameOID.UNSTRUCTURED_NAME)
        assert len(role_attrs) == 1
        assert role_attrs[0].value == "lead"

    def test_sign_output_filename_from_type(self, tmp_path):
        ca_dir = _setup_ca(tmp_path)
        csr_path = _setup_csr(tmp_path, name="srv", cert_type="server")
        out_dir = str(tmp_path / "signed")
        args = _sign_args(csr_path=csr_path, ca_dir=ca_dir, output_dir=out_dir, cert_type="server")
        handle_cert_sign(args)
        assert os.path.exists(os.path.join(out_dir, "server.crt"))

    def test_sign_existing_cert_no_force(self, tmp_path):
        ca_dir = _setup_ca(tmp_path)
        csr_path = _setup_csr(tmp_path)
        out_dir = str(tmp_path / "signed")
        os.makedirs(out_dir, exist_ok=True)
        # Pre-create the cert file
        open(os.path.join(out_dir, "client.crt"), "w").close()
        args = _sign_args(csr_path=csr_path, ca_dir=ca_dir, output_dir=out_dir, cert_type="client")
        with pytest.raises(SystemExit) as exc_info:
            handle_cert_sign(args)
        assert exc_info.value.code == 1

    def test_sign_existing_cert_with_force(self, tmp_path):
        ca_dir = _setup_ca(tmp_path)
        csr_path = _setup_csr(tmp_path)
        out_dir = str(tmp_path / "signed")
        os.makedirs(out_dir, exist_ok=True)
        open(os.path.join(out_dir, "client.crt"), "w").close()
        args = _sign_args(
            csr_path=csr_path, ca_dir=ca_dir, output_dir=out_dir, cert_type="client", force=True
        )
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

    def test_sign_output_json(self, tmp_path, capsys):
        ca_dir = _setup_ca(tmp_path)
        csr_path = _setup_csr(tmp_path)
        capsys.readouterr()  # discard setup output
        out_dir = str(tmp_path / "signed")
        args = _sign_args(
            csr_path=csr_path, ca_dir=ca_dir, output_dir=out_dir, cert_type="client", output_fmt="json"
        )
        rc = handle_cert_sign(args)
        assert rc == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["schema_version"] == "1"
        assert data["status"] == "ok"
        assert "signed_cert" in data["data"]
        assert "rootca" in data["data"]
        assert "serial" in data["data"]

    def test_sign_output_json_implies_force(self, tmp_path):
        ca_dir = _setup_ca(tmp_path)
        csr_path = _setup_csr(tmp_path)
        out_dir = str(tmp_path / "signed")
        os.makedirs(out_dir, exist_ok=True)
        open(os.path.join(out_dir, "client.crt"), "w").close()
        args = _sign_args(
            csr_path=csr_path, ca_dir=ca_dir, output_dir=out_dir, cert_type="client", output_fmt="json"
        )
        rc = handle_cert_sign(args)
        assert rc == 0

    def test_sign_schema_output(self, tmp_path, capsys):
        args = _sign_args(schema=True, csr_path="x", ca_dir="y", output_dir="z")
        with pytest.raises(SystemExit) as exc_info:
            handle_cert_sign(args)
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["schema_version"] == "1"
        assert data["command"] == "nvflare cert sign"

    def test_sign_rootca_copied(self, tmp_path):
        ca_dir = _setup_ca(tmp_path)
        csr_path = _setup_csr(tmp_path)
        out_dir = str(tmp_path / "signed")
        args = _sign_args(csr_path=csr_path, ca_dir=ca_dir, output_dir=out_dir, cert_type="client")
        handle_cert_sign(args)
        # rootCA.pem in output dir should match the one in ca_dir
        orig = (open(os.path.join(ca_dir, "rootCA.pem"), "rb").read())
        copy = (open(os.path.join(out_dir, "rootCA.pem"), "rb").read())
        assert orig == copy

    def test_sign_multiple_certs_serial_increments(self, tmp_path):
        ca_dir = _setup_ca(tmp_path)
        csr1 = _setup_csr(tmp_path / "csr1", name="client1", cert_type="client")
        csr2 = _setup_csr(tmp_path / "csr2", name="client2", cert_type="client")

        out1 = str(tmp_path / "signed1")
        out2 = str(tmp_path / "signed2")

        args1 = _sign_args(csr_path=csr1, ca_dir=ca_dir, output_dir=out1, cert_type="client")
        handle_cert_sign(args1)
        serial_mid = _read_serial(os.path.join(ca_dir, "ca.json"))

        args2 = _sign_args(csr_path=csr2, ca_dir=ca_dir, output_dir=out2, cert_type="client")
        handle_cert_sign(args2)
        serial_end = _read_serial(os.path.join(ca_dir, "ca.json"))

        assert serial_end == serial_mid + 1

    def test_sign_admin_role_cert(self, tmp_path):
        ca_dir = _setup_ca(tmp_path)
        csr_path = _setup_csr(tmp_path, name="bob", cert_type="org_admin")
        out_dir = str(tmp_path / "signed")
        args = _sign_args(csr_path=csr_path, ca_dir=ca_dir, output_dir=out_dir, cert_type="org_admin")
        rc = handle_cert_sign(args)
        assert rc == 0
        assert os.path.exists(os.path.join(out_dir, "org_admin.crt"))
        # Admin-role certs must have content_commitment (non-repudiation) set to True.
        # This is required so that job submissions signed with admin certs cannot be repudiated.
        # Regression guard: _build_signed_cert must use content_commitment=True for admin roles.
        cert = load_crt(os.path.join(out_dir, "org_admin.crt"))
        key_usage = cert.extensions.get_extension_for_class(x509.KeyUsage)
        assert key_usage.value.content_commitment is True
