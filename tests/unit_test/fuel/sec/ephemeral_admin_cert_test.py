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
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from nvflare.fuel.sec.ephemeral_admin_cert import (
    EphemeralAdminCertError,
    EphemeralAdminCertFiles,
    get_ephemeral_admin_cert_renewal_window,
    obtain_ephemeral_admin_cert_files,
    validate_ephemeral_admin_cert_files,
)
from nvflare.fuel.sec.step_ca_admin_cert import (
    DEFAULT_STEP_CA_REQUEST_NAME,
    obtain_step_ca_admin_cert_files,
    validate_step_ca_admin_cert_config,
)
from nvflare.lighter.tool_consts import NVFLARE_SUBMITTER_CRT_FILE
from nvflare.lighter.utils import (
    Identity,
    generate_cert,
    generate_keys,
    load_crt_chain,
    load_private_key_file,
    serialize_cert,
    serialize_pri_key,
    sign_folders,
    verify_folder_signature,
)
from nvflare.private.fed.utils.identity_utils import IdentityAsserter, IdentityVerifier, load_cert_file


@pytest.fixture(autouse=True)
def _use_test_home(monkeypatch, tmp_path):
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    monkeypatch.setenv("HOME", str(home_dir))


def _make_root_ca(tmp_path):
    ca_key, ca_pub_key = generate_keys()
    ca_cert = generate_cert(
        subject=Identity("root", "nvidia"),
        issuer=Identity("root", "nvidia"),
        signing_pri_key=ca_key,
        subject_pub_key=ca_pub_key,
        ca=True,
    )
    root_ca_path = tmp_path / "rootCA.pem"
    root_ca_path.write_bytes(serialize_cert(ca_cert))
    return ca_key, ca_cert, root_ca_path


def _write_key_cert(key_path, cert_path, private_key, certs):
    with open(key_path, "wb") as f:
        f.write(serialize_pri_key(private_key))
    with open(cert_path, "wb") as f:
        for cert in certs:
            f.write(serialize_cert(cert))


def _make_admin_cert_files(
    tmp_path,
    signing_key,
    issuer_identity,
    chain=(),
    issued_subject="alice@nvidia.com",
    role="lead",
    ca=False,
    extra_extensions=None,
):
    admin_key, admin_pub_key = generate_keys()
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = generate_cert(
        subject=Identity(issued_subject, "nvidia", role),
        issuer=issuer_identity,
        signing_pri_key=signing_key,
        subject_pub_key=admin_pub_key,
        not_valid_before=now - datetime.timedelta(seconds=1),
        not_valid_after=now + datetime.timedelta(hours=1),
        ca=ca,
        extra_extensions=extra_extensions,
    )
    cert_src = tmp_path / "issued.crt"
    key_src = tmp_path / "issued.key"
    _write_key_cert(key_src, cert_src, admin_key, [cert, *chain])
    return cert_src, key_src


def _custom_ephemeral_provider(config, root_ca_file):
    return EphemeralAdminCertFiles(
        client_key=config["key_path"],
        client_cert=config["cert_path"],
        expires_at=config.get("expires_at", 0.0),
    )


def _fake_step(
    monkeypatch,
    tmp_path,
    cert_src=None,
    key_src=None,
    command_log=None,
    exit_code=None,
    sleep=None,
):
    fake_step = tmp_path / "step"
    fake_step.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import json, os, shutil, sys, time",
                "if os.environ.get('NVFLARE_TEST_STEP_SLEEP'):",
                "    time.sleep(float(os.environ['NVFLARE_TEST_STEP_SLEEP']))",
                "if os.environ.get('NVFLARE_TEST_STEP_EXIT'):",
                "    raise SystemExit(int(os.environ['NVFLARE_TEST_STEP_EXIT']))",
                "if os.environ.get('NVFLARE_TEST_STEP_COMMAND_LOG'):",
                "    with open(os.environ['NVFLARE_TEST_STEP_COMMAND_LOG'], 'a') as f:",
                "        f.write(json.dumps(sys.argv[1:]) + '\\n')",
                "if sys.argv[1:3] == ['ca', 'certificate']:",
                "    shutil.copyfile(os.environ['NVFLARE_TEST_STEP_CERT'], sys.argv[-2])",
                "    shutil.copyfile(os.environ['NVFLARE_TEST_STEP_KEY'], sys.argv[-1])",
                "else:",
                "    raise SystemExit(3)",
            ]
        ),
        encoding="utf-8",
    )
    fake_step.chmod(0o755)
    if cert_src:
        monkeypatch.setenv("NVFLARE_TEST_STEP_CERT", str(cert_src))
    if key_src:
        monkeypatch.setenv("NVFLARE_TEST_STEP_KEY", str(key_src))
    if command_log:
        monkeypatch.setenv("NVFLARE_TEST_STEP_COMMAND_LOG", str(command_log))
    if exit_code is not None:
        monkeypatch.setenv("NVFLARE_TEST_STEP_EXIT", str(exit_code))
    if sleep is not None:
        monkeypatch.setenv("NVFLARE_TEST_STEP_SLEEP", str(sleep))
    return fake_step


def _read_command_log(command_log):
    return [json.loads(line) for line in command_log.read_text(encoding="utf-8").splitlines()]


def test_step_ca_source_invokes_step_and_cert_works_with_existing_flare_paths(monkeypatch, tmp_path):
    ca_key, _ca_cert, root_ca_path = _make_root_ca(tmp_path)
    cert_src, key_src = _make_admin_cert_files(tmp_path, signing_key=ca_key, issuer_identity=Identity("root", "nvidia"))
    command_log = tmp_path / "commands.jsonl"
    fake_step = _fake_step(monkeypatch, tmp_path, cert_src=cert_src, key_src=key_src, command_log=command_log)

    files = obtain_ephemeral_admin_cert_files(
        config={
            "provider": "step_ca",
            "provider_config": {
                "ca_url": "https://step-ca.example.com",
                "provisioner": "nvflare-admin-oidc",
                "cert_ttl": "1h",
                "step_bin": str(fake_step),
            },
        },
        root_ca_file=str(root_ca_path),
    )

    try:
        command = _read_command_log(command_log)[0]
        assert command[:2] == ["ca", "certificate"]
        assert "--ca-url" in command
        assert "https://step-ca.example.com" in command
        assert "--provisioner" in command
        assert "nvflare-admin-oidc" in command
        assert "--token" not in command
        assert "--kty" in command
        assert "RSA" in command
        assert command[-3] == DEFAULT_STEP_CA_REQUEST_NAME
        assert os.path.isfile(files.client_key)
        assert os.path.isfile(files.client_cert)
        assert files.expires_at > time.time()

        asserter = IdentityAsserter(private_key_file=files.client_key, cert_file=files.client_cert)
        signature = asserter.sign_common_name(nonce="")
        cert_chain = load_crt_chain(files.client_cert)
        assert IdentityVerifier(root_cert_file=str(root_ca_path)).verify_common_name(
            asserted_cn="alice@nvidia.com",
            nonce="",
            asserter_cert=asserter.cert,
            signature=signature,
            intermediate_certs=cert_chain[1:],
        )

        cert = load_cert_file(files.client_cert)
        assert cert.subject.get_attributes_for_oid(NameOID.ORGANIZATION_NAME)[0].value == "nvidia"
        assert cert.subject.get_attributes_for_oid(NameOID.UNSTRUCTURED_NAME)[0].value == "lead"

        job_dir = tmp_path / "job"
        job_dir.mkdir()
        (job_dir / "config.json").write_text(json.dumps({"name": "job"}), encoding="utf-8")
        sign_folders(str(job_dir), asserter.pri_key, files.client_cert)
        assert (job_dir / NVFLARE_SUBMITTER_CRT_FILE).is_file()
        assert verify_folder_signature(str(job_dir), str(root_ca_path))
    finally:
        files.cleanup()


def test_step_ca_source_accepts_intermediate_chain(monkeypatch, tmp_path):
    root_key, _root_cert, root_ca_path = _make_root_ca(tmp_path)
    intermediate_key, intermediate_pub_key = generate_keys()
    intermediate_cert = generate_cert(
        subject=Identity("step-ca-intermediate", "nvidia"),
        issuer=Identity("root", "nvidia"),
        signing_pri_key=root_key,
        subject_pub_key=intermediate_pub_key,
        ca=True,
    )
    cert_src, key_src = _make_admin_cert_files(
        tmp_path,
        signing_key=intermediate_key,
        issuer_identity=Identity("step-ca-intermediate", "nvidia"),
        chain=(intermediate_cert,),
    )
    command_log = tmp_path / "commands.jsonl"
    fake_step = _fake_step(monkeypatch, tmp_path, cert_src=cert_src, key_src=key_src, command_log=command_log)

    files = obtain_ephemeral_admin_cert_files(
        config={
            "provider": "step_ca",
            "provider_config": {
                "ca_url": "https://step-ca.example.com",
                "provisioner": "nvflare-admin-oidc",
                "step_bin": str(fake_step),
            },
        },
        root_ca_file=str(root_ca_path),
    )

    try:
        command = _read_command_log(command_log)[0]
        assert "24h" in command
        assert len(load_crt_chain(files.client_cert)) == 2
        job_dir = tmp_path / "job"
        job_dir.mkdir()
        (job_dir / "config.json").write_text(json.dumps({"name": "job"}), encoding="utf-8")
        sign_folders(str(job_dir), load_private_key_file(files.client_key), files.client_cert)
        assert verify_folder_signature(str(job_dir), str(root_ca_path))
    finally:
        files.cleanup()


def test_custom_ephemeral_admin_cert_provider_path_is_supported(tmp_path):
    ca_key, _ca_cert, root_ca_path = _make_root_ca(tmp_path)
    cert_path, key_path = _make_admin_cert_files(
        tmp_path,
        signing_key=ca_key,
        issuer_identity=Identity("root", "nvidia"),
    )

    files = obtain_ephemeral_admin_cert_files(
        config={
            "provider": "tests.unit_test.fuel.sec.ephemeral_admin_cert_test:_custom_ephemeral_provider",
            "provider_config": {
                "cert_path": str(cert_path),
                "key_path": str(key_path),
            },
        },
        root_ca_file=str(root_ca_path),
    )

    assert os.path.isfile(files.client_cert)
    assert os.path.isfile(files.client_key)
    assert files.client_cert != str(cert_path)
    assert files.client_key != str(key_path)
    assert files.expires_at > time.time()


def test_ephemeral_admin_cert_uses_validated_certificate_expiry(tmp_path):
    ca_key, _ca_cert, root_ca_path = _make_root_ca(tmp_path)
    cert_path, key_path = _make_admin_cert_files(
        tmp_path,
        signing_key=ca_key,
        issuer_identity=Identity("root", "nvidia"),
    )

    files = obtain_ephemeral_admin_cert_files(
        config={
            "provider": "tests.unit_test.fuel.sec.ephemeral_admin_cert_test:_custom_ephemeral_provider",
            "provider_config": {
                "cert_path": str(cert_path),
                "key_path": str(key_path),
                "expires_at": time.time() + 86400,
            },
        },
        root_ca_file=str(root_ca_path),
    )

    cert = load_cert_file(files.client_cert)
    assert files.expires_at == cert.not_valid_after_utc.timestamp()


def test_ephemeral_admin_cert_cache_reuses_valid_cert(monkeypatch, tmp_path):
    ca_key, _ca_cert, root_ca_path = _make_root_ca(tmp_path)
    cert_src, key_src = _make_admin_cert_files(tmp_path, signing_key=ca_key, issuer_identity=Identity("root", "nvidia"))
    command_log = tmp_path / "commands.jsonl"
    fake_step = _fake_step(monkeypatch, tmp_path, cert_src=cert_src, key_src=key_src, command_log=command_log)
    config = {
        "provider": "step_ca",
        "provider_config": {
            "ca_url": "https://step-ca.example.com",
            "provisioner": "nvflare-admin-oidc",
            "step_bin": str(fake_step),
        },
    }

    first = obtain_ephemeral_admin_cert_files(config=config, root_ca_file=str(root_ca_path))
    second = obtain_ephemeral_admin_cert_files(config=config, root_ca_file=str(root_ca_path))

    assert len(_read_command_log(command_log)) == 1
    assert first.client_cert == second.client_cert
    assert first.client_key == second.client_key


def test_ephemeral_admin_cert_rejects_issued_cert_inside_renewal_window(monkeypatch, tmp_path):
    ca_key, _ca_cert, root_ca_path = _make_root_ca(tmp_path)
    cert_src, key_src = _make_admin_cert_files(tmp_path, signing_key=ca_key, issuer_identity=Identity("root", "nvidia"))
    command_log = tmp_path / "commands.jsonl"
    fake_step = _fake_step(monkeypatch, tmp_path, cert_src=cert_src, key_src=key_src, command_log=command_log)
    config = {
        "provider": "step_ca",
        "renewal_window": 7200,
        "provider_config": {
            "ca_url": "https://step-ca.example.com",
            "provisioner": "nvflare-admin-oidc",
            "step_bin": str(fake_step),
        },
    }

    with pytest.raises(EphemeralAdminCertError, match="remain valid beyond the renewal window"):
        obtain_ephemeral_admin_cert_files(config=config, root_ca_file=str(root_ca_path))

    assert len(_read_command_log(command_log)) == 1


def test_ephemeral_admin_cert_cache_removes_orphaned_staging_directory(monkeypatch, tmp_path):
    ca_key, _ca_cert, root_ca_path = _make_root_ca(tmp_path)
    cert_src, key_src = _make_admin_cert_files(tmp_path, signing_key=ca_key, issuer_identity=Identity("root", "nvidia"))
    command_log = tmp_path / "commands.jsonl"
    fake_step = _fake_step(monkeypatch, tmp_path, cert_src=cert_src, key_src=key_src, command_log=command_log)
    config = {
        "provider": "step_ca",
        "provider_config": {
            "ca_url": "https://step-ca.example.com",
            "provisioner": "nvflare-admin-oidc",
            "step_bin": str(fake_step),
        },
    }

    obtain_ephemeral_admin_cert_files(config=config, root_ca_file=str(root_ca_path))
    cache_root = Path.home() / ".nvflare" / "ephemeral_admin_certs"
    cache_dir = next(path for path in cache_root.iterdir() if path.is_dir())
    orphan = cache_dir / ".new-orphan"
    orphan.mkdir()
    (orphan / "client.key").write_text("partial", encoding="utf-8")

    obtain_ephemeral_admin_cert_files(config=config, root_ca_file=str(root_ca_path))

    assert not orphan.exists()
    assert len(_read_command_log(command_log)) == 1


def test_ephemeral_admin_cert_cache_refreshes_invalid_cache(monkeypatch, tmp_path):
    ca_key, _ca_cert, root_ca_path = _make_root_ca(tmp_path)
    cert_src, key_src = _make_admin_cert_files(tmp_path, signing_key=ca_key, issuer_identity=Identity("root", "nvidia"))
    command_log = tmp_path / "commands.jsonl"
    fake_step = _fake_step(monkeypatch, tmp_path, cert_src=cert_src, key_src=key_src, command_log=command_log)
    config = {
        "provider": "step_ca",
        "provider_config": {
            "ca_url": "https://step-ca.example.com",
            "provisioner": "nvflare-admin-oidc",
            "step_bin": str(fake_step),
        },
    }

    files = obtain_ephemeral_admin_cert_files(config=config, root_ca_file=str(root_ca_path))
    with open(files.client_key, "w", encoding="utf-8") as f:
        f.write("not a private key")

    refreshed_files = obtain_ephemeral_admin_cert_files(config=config, root_ca_file=str(root_ca_path))

    assert len(_read_command_log(command_log)) == 2
    assert os.path.isfile(refreshed_files.client_cert)
    assert os.path.isfile(refreshed_files.client_key)


def test_ephemeral_admin_cert_cache_serializes_concurrent_acquisition(monkeypatch, tmp_path):
    ca_key, _ca_cert, root_ca_path = _make_root_ca(tmp_path)
    cert_src, key_src = _make_admin_cert_files(tmp_path, signing_key=ca_key, issuer_identity=Identity("root", "nvidia"))
    provider_started = threading.Event()
    provider_release = threading.Event()
    provider_calls = 0

    def _provider(config, root_ca_file):
        nonlocal provider_calls
        provider_calls += 1
        provider_started.set()
        assert provider_release.wait(timeout=5)
        return EphemeralAdminCertFiles(client_key=str(key_src), client_cert=str(cert_src))

    monkeypatch.setattr("nvflare.fuel.sec.ephemeral_admin_cert._load_provider", lambda _provider_name: _provider)
    config = {"provider": "test.provider:obtain_certificate", "provider_config": {}}

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(obtain_ephemeral_admin_cert_files, config, str(root_ca_path))
        assert provider_started.wait(timeout=5)
        second_future = executor.submit(obtain_ephemeral_admin_cert_files, config, str(root_ca_path))
        provider_release.set()
        first = first_future.result(timeout=5)
        second = second_future.result(timeout=5)

    assert provider_calls == 1
    assert first.client_cert == second.client_cert
    assert first.client_key == second.client_key


@pytest.mark.parametrize(
    "org, role, missing_field",
    [
        (None, "lead", "organizationName"),
        ("", "lead", "organizationName"),
        ("nvidia", None, "unstructuredName"),
        ("nvidia", "", "unstructuredName"),
    ],
)
def test_validate_ephemeral_admin_cert_files_rejects_missing_identity_field(tmp_path, org, role, missing_field):
    ca_key, _ca_cert, root_ca_path = _make_root_ca(tmp_path)
    admin_key, admin_pub_key = generate_keys()
    cert = generate_cert(
        subject=Identity("alice@nvidia.com", org, role),
        issuer=Identity("root", "nvidia"),
        signing_pri_key=ca_key,
        subject_pub_key=admin_pub_key,
    )
    key_path = tmp_path / "client.key"
    cert_path = tmp_path / "client.crt"
    _write_key_cert(key_path, cert_path, admin_key, [cert])

    with pytest.raises(EphemeralAdminCertError, match=missing_field):
        validate_ephemeral_admin_cert_files(str(cert_path), str(key_path), str(root_ca_path))


def test_validate_ephemeral_admin_cert_files_rejects_non_rsa_key(tmp_path):
    ca_key, _ca_cert, root_ca_path = _make_root_ca(tmp_path)
    admin_key = ec.generate_private_key(ec.SECP256R1())
    cert = generate_cert(
        subject=Identity("alice@nvidia.com", "nvidia", "lead"),
        issuer=Identity("root", "nvidia"),
        signing_pri_key=ca_key,
        subject_pub_key=admin_key.public_key(),
    )
    key_path = tmp_path / "client.key"
    cert_path = tmp_path / "client.crt"
    _write_key_cert(key_path, cert_path, admin_key, [cert])

    with pytest.raises(EphemeralAdminCertError, match="must use RSA"):
        validate_ephemeral_admin_cert_files(str(cert_path), str(key_path), str(root_ca_path))


def test_validate_ephemeral_admin_cert_files_reports_unreadable_private_key(tmp_path):
    ca_key, _ca_cert, root_ca_path = _make_root_ca(tmp_path)
    cert_path, key_path = _make_admin_cert_files(
        tmp_path, signing_key=ca_key, issuer_identity=Identity("root", "nvidia")
    )
    private_key = load_private_key_file(key_path)
    key_path.write_bytes(
        private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.BestAvailableEncryption(b"password"),
        )
    )

    with pytest.raises(EphemeralAdminCertError, match="unencrypted PEM private key"):
        validate_ephemeral_admin_cert_files(str(cert_path), str(key_path), str(root_ca_path))


def test_validate_ephemeral_admin_cert_files_preserves_validity_failure_detail(tmp_path):
    ca_key, _ca_cert, root_ca_path = _make_root_ca(tmp_path)
    admin_key, admin_pub_key = generate_keys()
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = generate_cert(
        subject=Identity("alice@nvidia.com", "nvidia", "lead"),
        issuer=Identity("root", "nvidia"),
        signing_pri_key=ca_key,
        subject_pub_key=admin_pub_key,
        not_valid_before=now - datetime.timedelta(hours=2),
        not_valid_after=now - datetime.timedelta(hours=1),
    )
    key_path = tmp_path / "client.key"
    cert_path = tmp_path / "client.crt"
    _write_key_cert(key_path, cert_path, admin_key, [cert])

    with pytest.raises(EphemeralAdminCertError, match="chain validation failed:.*not valid"):
        validate_ephemeral_admin_cert_files(str(cert_path), str(key_path), str(root_ca_path))


def test_step_ca_source_reports_step_failure(monkeypatch, tmp_path):
    _ca_key, _ca_cert, root_ca_path = _make_root_ca(tmp_path)
    fake_step = _fake_step(monkeypatch, tmp_path, exit_code=1)

    with pytest.raises(EphemeralAdminCertError, match="step ca certificate failed"):
        obtain_ephemeral_admin_cert_files(
            config={
                "provider": "step_ca",
                "provider_config": {
                    "ca_url": "https://step-ca.example.com",
                    "provisioner": "nvflare-admin-oidc",
                    "step_bin": str(fake_step),
                },
            },
            root_ca_file=str(root_ca_path),
        )


def test_step_ca_source_requires_explicit_provisioner(monkeypatch, tmp_path):
    ca_key, _ca_cert, root_ca_path = _make_root_ca(tmp_path)
    cert_src, key_src = _make_admin_cert_files(tmp_path, signing_key=ca_key, issuer_identity=Identity("root", "nvidia"))
    fake_step = _fake_step(monkeypatch, tmp_path, cert_src=cert_src, key_src=key_src)
    monkeypatch.setattr(
        "nvflare.fuel.sec.step_ca_admin_cert.tempfile.TemporaryDirectory",
        lambda **_kwargs: pytest.fail("temporary directory created before config validation"),
    )

    with pytest.raises(EphemeralAdminCertError, match="provisioner"):
        obtain_step_ca_admin_cert_files(
            config={"ca_url": "https://step-ca.example.com", "step_bin": str(fake_step)},
            root_ca_file=str(root_ca_path),
        )


def test_step_ca_source_requires_ca_url():
    with pytest.raises(EphemeralAdminCertError, match="ca_url is required"):
        validate_step_ca_admin_cert_config({"provisioner": "nvflare-admin-oidc"})


def test_step_ca_source_accepts_ipv6_loopback_http():
    config = {"ca_url": "http://[::1]:9000", "provisioner": "nvflare-admin-oidc"}

    assert validate_step_ca_admin_cert_config(config) == {**config, "command_timeout": 300.0}


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), float("-inf")])
def test_ephemeral_admin_cert_rejects_non_finite_renewal_window(value):
    with pytest.raises(EphemeralAdminCertError, match="finite number"):
        get_ephemeral_admin_cert_renewal_window({"renewal_window": value})


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), float("-inf")])
def test_step_ca_source_rejects_non_finite_command_timeout(value):
    with pytest.raises(EphemeralAdminCertError, match="finite number"):
        validate_step_ca_admin_cert_config(
            {
                "ca_url": "https://step-ca.example.com",
                "provisioner": "nvflare-admin-oidc",
                "command_timeout": value,
            }
        )


def test_step_ca_source_times_out_step_command(monkeypatch, tmp_path):
    _ca_key, _ca_cert, root_ca_path = _make_root_ca(tmp_path)
    fake_step = _fake_step(monkeypatch, tmp_path, sleep=2)

    with pytest.raises(EphemeralAdminCertError, match="timed out"):
        obtain_step_ca_admin_cert_files(
            config={
                "ca_url": "https://step-ca.example.com",
                "provisioner": "nvflare-admin-oidc",
                "step_bin": str(fake_step),
                "command_timeout": 0.01,
            },
            root_ca_file=str(root_ca_path),
        )


def test_step_ca_source_reports_unexecutable_step_binary(tmp_path):
    _ca_key, _ca_cert, root_ca_path = _make_root_ca(tmp_path)
    step_bin = tmp_path / "step"
    step_bin.write_text("not executable", encoding="utf-8")
    step_bin.chmod(0o600)

    with pytest.raises(EphemeralAdminCertError, match="cannot execute the 'step' CLI"):
        obtain_step_ca_admin_cert_files(
            config={
                "ca_url": "https://step-ca.example.com",
                "provisioner": "nvflare-admin-oidc",
                "step_bin": str(step_bin),
            },
            root_ca_file=str(root_ca_path),
        )


def test_step_ca_source_rejects_placeholder_common_name(monkeypatch, tmp_path):
    ca_key, _ca_cert, root_ca_path = _make_root_ca(tmp_path)
    cert_src, key_src = _make_admin_cert_files(
        tmp_path,
        signing_key=ca_key,
        issuer_identity=Identity("root", "nvidia"),
        issued_subject=DEFAULT_STEP_CA_REQUEST_NAME,
    )
    fake_step = _fake_step(monkeypatch, tmp_path, cert_src=cert_src, key_src=key_src)

    with pytest.raises(EphemeralAdminCertError, match="commonName"):
        obtain_ephemeral_admin_cert_files(
            config={
                "provider": "step_ca",
                "provider_config": {
                    "ca_url": "https://step-ca.example.com",
                    "provisioner": "nvflare-admin-oidc",
                    "step_bin": str(fake_step),
                },
            },
            root_ca_file=str(root_ca_path),
        )


def test_ephemeral_cert_files_need_renewal_inside_window():
    cert_files = EphemeralAdminCertFiles(
        client_key="client.key",
        client_cert="client.crt",
        expires_at=time.time() + 30.0,
    )

    assert cert_files.needs_renewal(renewal_window=60.0)
    assert not cert_files.needs_renewal(renewal_window=10.0)
