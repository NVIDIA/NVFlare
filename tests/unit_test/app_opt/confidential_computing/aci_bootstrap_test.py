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
import io
import sys
import tarfile
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from nvflare.app_opt.confidential_computing import aci_bootstrap as bootstrap
from nvflare.lighter.constants import ProvFileName
from nvflare.lighter.utils import sign_content, sign_folders


@pytest.fixture
def delivery(tmp_path, monkeypatch):
    inputs = tmp_path / "input"
    inputs.mkdir()
    workspaces = tmp_path / "workspaces"
    workspaces.mkdir()
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    key_path = tmp_path / "public.pem"
    key_path.write_bytes(
        key.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    )
    monkeypatch.setattr(bootstrap, "INPUT_DIR", inputs)
    monkeypatch.setattr(bootstrap, "WORKSPACE_DIR", workspaces)
    monkeypatch.setattr(bootstrap, "PUBLIC_KEY_FILE", key_path)

    def write(data):
        (inputs / "startup-kit.tar").write_bytes(data)
        (inputs / "startup-kit.sig").write_bytes(sign_content(data, key, return_str=False))

    return SimpleNamespace(write=write, key=key, inputs=inputs, workspaces=workspaces)


def _archive(entries):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name, kind, content in entries:
            member = tarfile.TarInfo(name)
            member.type = kind
            member.size = len(content) if kind == tarfile.REGTYPE else 0
            if kind in (tarfile.LNKTYPE, tarfile.SYMTYPE):
                member.linkname = "outside"
            archive.addfile(member, io.BytesIO(content) if kind == tarfile.REGTYPE else None)
    return output.getvalue()


def _signed_kit(tmp_path, key):
    kit = tmp_path / "kit"
    (kit / "startup").mkdir(parents=True)
    (kit / "local").mkdir()
    subject = x509.Name(
        [x509.NameAttribute(NameOID.COMMON_NAME, "server"), x509.NameAttribute(NameOID.ORGANIZATION_NAME, "example")]
    )
    now = datetime.datetime.now(datetime.timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .sign(key, hashes.SHA256())
    )
    pem = certificate.public_bytes(serialization.Encoding.PEM)
    (kit / "startup/rootCA.pem").write_bytes(pem)
    (kit / "startup/server.crt").write_bytes(pem)
    (kit / "startup/fed_server.json").write_text("{}")
    (kit / "local/approved_cce_policy.txt").write_text("approved policy\n")
    sign_folders(str(kit), key, signature_file=ProvFileName.SIGNATURE_JSON)
    entries = [
        (
            path.relative_to(kit).as_posix(),
            tarfile.DIRTYPE if path.is_dir() else tarfile.REGTYPE,
            b"" if path.is_dir() else path.read_bytes(),
        )
        for path in sorted(kit.rglob("*"))
    ]
    return _archive(entries)


def test_valid_signed_delivery_verifies_both_signature_layers(tmp_path, delivery):
    delivery.write(_signed_kit(tmp_path, delivery.key))
    workspace = bootstrap.prepare_workspace()
    assert (workspace / "local/approved_cce_policy.txt").read_text() == "approved policy\n"
    assert workspace.parent == delivery.workspaces
    assert workspace.stat().st_mode & 0o777 == 0o700
    assert (workspace / "startup/server.crt").stat().st_mode & 0o777 == 0o600
    command = bootstrap._server_command(workspace)
    assert command[:5] == [sys.executable, "-I", "-u", "-m", "nvflare.private.fed.app.server.server_train"]
    assert command[-1] == "org=example"


def test_untrusted_archive_is_not_parsed_or_extracted(delivery, monkeypatch):
    delivery.write(b"not even an archive")
    (delivery.inputs / "startup-kit.tar").write_bytes(b"changed after signing")
    parse = Mock(side_effect=AssertionError("must not parse unauthenticated input"))
    monkeypatch.setattr(bootstrap, "_extract_archive", parse)
    with pytest.raises(Exception):
        bootstrap.prepare_workspace()
    parse.assert_not_called()
    assert list(delivery.workspaces.iterdir()) == []


@pytest.mark.parametrize(
    "entries",
    [
        [("../escape", tarfile.REGTYPE, b"bad")],
        [("/absolute", tarfile.REGTYPE, b"bad")],
        [("local/./file", tarfile.REGTYPE, b"bad")],
        [("local//file", tarfile.REGTYPE, b"bad")],
        [("local\\file", tarfile.REGTYPE, b"bad")],
        [("duplicate", tarfile.REGTYPE, b"one"), ("duplicate", tarfile.REGTYPE, b"two")],
        [("link", tarfile.SYMTYPE, b"")],
        [("link", tarfile.LNKTYPE, b"")],
        [("device", tarfile.CHRTYPE, b"")],
        [("pipe", tarfile.FIFOTYPE, b"")],
        [("local", tarfile.REGTYPE, b"bad"), ("local/file", tarfile.REGTYPE, b"bad")],
        [("local/file", tarfile.REGTYPE, b"bad"), ("local", tarfile.REGTYPE, b"bad")],
    ],
)
def test_unsafe_signed_archive_is_rejected_without_residual_files(entries, delivery):
    delivery.write(_archive(entries))
    with pytest.raises(ValueError):
        bootstrap.prepare_workspace()
    assert list(delivery.workspaces.iterdir()) == []


@pytest.mark.parametrize("limit", ["MAX_ARCHIVE_BYTES", "MAX_EXTRACTED_BYTES", "MAX_ARCHIVE_MEMBERS"])
def test_signed_oversized_delivery_is_rejected(limit, delivery, monkeypatch):
    delivery.write(_archive([("one", tarfile.REGTYPE, b"1234"), ("two", tarfile.REGTYPE, b"5678")]))
    monkeypatch.setattr(bootstrap, limit, 1)
    with pytest.raises(ValueError):
        bootstrap.prepare_workspace()
    assert list(delivery.workspaces.iterdir()) == []


def test_signed_archive_with_broken_inner_kit_signature_is_rejected(tmp_path, delivery):
    valid = _signed_kit(tmp_path, delivery.key)
    with tarfile.open(fileobj=io.BytesIO(valid), mode="r:") as archive:
        entries = [
            (
                member.name,
                member.type,
                (
                    b"modified"
                    if member.name == "startup/fed_server.json"
                    else archive.extractfile(member).read() if member.isfile() else b""
                ),
            )
            for member in archive
        ]
    delivery.write(_archive(entries))
    with pytest.raises(ValueError, match="signatures"):
        bootstrap.prepare_workspace()
    assert list(delivery.workspaces.iterdir()) == []


def test_signed_incomplete_kit_is_rejected(delivery):
    delivery.write(_archive([("startup", tarfile.DIRTYPE, b"")]))
    with pytest.raises(ValueError, match="Incomplete"):
        bootstrap.prepare_workspace()


def test_main_launches_fixed_command_with_clean_environment(tmp_path, delivery, monkeypatch):
    delivery.write(_signed_kit(tmp_path, delivery.key))
    monkeypatch.setattr(
        bootstrap,
        "sys",
        SimpleNamespace(
            flags=SimpleNamespace(isolated=True), argv=["bootstrap"], executable=sys.executable, stderr=sys.stderr
        ),
    )
    monkeypatch.setenv("PYTHONPATH", "/malicious")
    monkeypatch.setenv("LD_PRELOAD", "/malicious.so")
    chdir, execute = Mock(), Mock()
    monkeypatch.setattr(bootstrap.os, "chdir", chdir)
    monkeypatch.setattr(bootstrap.os, "execve", execute)
    assert bootstrap.main() == 0
    executable, command, environment = execute.call_args.args
    assert executable == sys.executable
    assert "-I" in command
    assert command[-1] == "org=example"
    assert set(environment) == {"PATH", "HOME", "LANG"}
    assert "PYTHONPATH" not in environment and "LD_PRELOAD" not in environment
    chdir.assert_called_once()


def test_main_refuses_unisolated_start_without_launch(delivery, monkeypatch):
    monkeypatch.setattr(bootstrap, "sys", SimpleNamespace(flags=SimpleNamespace(isolated=False), stderr=sys.stderr))
    execute = Mock()
    monkeypatch.setattr(bootstrap.os, "execve", execute)
    assert bootstrap.main() == 1
    execute.assert_not_called()


def test_main_never_starts_server_on_signature_failure(delivery, monkeypatch):
    delivery.write(b"invalid tar")
    (delivery.inputs / "startup-kit.sig").write_bytes(b"invalid signature")
    monkeypatch.setattr(
        bootstrap,
        "sys",
        SimpleNamespace(
            flags=SimpleNamespace(isolated=True), argv=["bootstrap"], executable=sys.executable, stderr=sys.stderr
        ),
    )
    execute = Mock()
    monkeypatch.setattr(bootstrap.os, "execve", execute)
    assert bootstrap.main() == 1
    execute.assert_not_called()
