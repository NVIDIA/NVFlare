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
import subprocess
from unittest.mock import patch

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa

from nvflare.app_opt.confidential_computing.cc_authorizer import CCTokenGenerateError
from nvflare.app_opt.confidential_computing.trustee_authorizer import TrusteeAuthorizer


def _public_key():
    return (
        ec.generate_private_key(ec.SECP256R1())
        .public_key()
        .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        .decode()
    )


def _authorizer(provider="verifier", **kwargs):
    return TrusteeAuthorizer(
        trustee_public_key=_public_key(),
        audience="nvflare-trustee:test",
        token_provider=provider,
        **kwargs,
    )


def test_verifier_provider_cannot_generate_guest_evidence():
    with pytest.raises(CCTokenGenerateError, match="Verifier-only"):
        _authorizer()._get_guest_token()


def test_cvm_provider_uses_pinned_guest_kbs_client_and_ephemeral_key():
    authorizer = _authorizer(
        "cvm",
        site_name="site-1",
        kbs_url="https://trustee.example.org:8443",
        kbs_ca="-----BEGIN CERTIFICATE-----\nfixture\n-----END CERTIFICATE-----\n",
    )
    completed = subprocess.CompletedProcess([], 0, stdout="signed-ear\n", stderr="")
    with patch(
        "nvflare.app_opt.confidential_computing.trustee_authorizer.subprocess.run", return_value=completed
    ) as run:
        reply = authorizer._get_guest_token()

    assert reply["token"] == "signed-ear"
    private = serialization.load_pem_private_key(reply["tee_keypair"].encode(), password=None)
    assert isinstance(private, rsa.RSAPrivateKey)
    argv = run.call_args.args[0]
    assert argv[:4] == [
        "/host/lib/ld-linux-x86-64.so.2",
        "--library-path",
        "/host/lib",
        "/host/bin/kbs-client",
    ]
    assert argv[4:7] == ["--url", "https://trustee.example.org:8443", "--cert-file"]
    assert argv[-2] == "--tee-key-file"
    assert argv[-1].startswith("/proc/self/fd/")
    assert run.call_args.kwargs["pass_fds"]
    assert run.call_args.kwargs["env"]["RUST_LOG"] == "off"
    assert run.call_args.kwargs["env"]["LD_LIBRARY_PATH"] == "/host/lib"


def test_cvm_provider_does_not_forward_loader_injection(monkeypatch):
    authorizer = _authorizer(
        "cvm",
        kbs_url="https://trustee.example.org:8443",
        kbs_ca="-----BEGIN CERTIFICATE-----\nfixture\n-----END CERTIFICATE-----\n",
    )
    monkeypatch.setenv("LD_PRELOAD", "/untrusted/injection.so")
    monkeypatch.setenv("LD_AUDIT", "/untrusted/audit.so")
    with patch(
        "nvflare.app_opt.confidential_computing.trustee_authorizer.subprocess.run",
        return_value=subprocess.CompletedProcess([], 0, stdout="signed-ear\n", stderr=""),
    ) as run:
        authorizer._get_guest_token()

    assert "LD_PRELOAD" not in run.call_args.kwargs["env"]
    assert "LD_AUDIT" not in run.call_args.kwargs["env"]
    assert run.call_args.kwargs["env"]["LD_LIBRARY_PATH"] == "/host/lib"


def test_cvm_provider_reads_supervisor_credentials_without_invoking_kbs_client(tmp_path):
    token_file = tmp_path / "trustee_token.json"
    reply = {"token": "signed-ear", "tee_keypair": "private-key"}
    token_file.write_text(json.dumps(reply))
    authorizer = _authorizer("cvm", guest_token_file=str(token_file))

    with patch("nvflare.app_opt.confidential_computing.trustee_authorizer.subprocess.run") as run:
        assert authorizer._get_guest_token() == reply

    run.assert_not_called()


@pytest.mark.parametrize("provider", ["", "legacy", None])
def test_unknown_token_provider_is_rejected(provider):
    with pytest.raises(ValueError, match="token_provider"):
        _authorizer(provider)
