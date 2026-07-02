# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import pytest
from cryptography import x509
from cryptography.x509.oid import ExtendedKeyUsageOID

from nvflare.lighter.impl.cert import serialize_cert
from nvflare.lighter.utils import Identity, generate_cert, generate_keys, sign_content
from nvflare.private.fed.utils.identity_utils import IdentityVerifier, InvalidAsserterCert, get_parent_site_name
from nvflare.private.fed.utils.job_cert_utils import JOB_ID_EXTENSION_OID


class TestIdentityUtils:
    @pytest.mark.parametrize(
        "fqsn, result",
        [
            ("", None),
            ("x", None),
            ("x.", "x"),
            (".x", ""),
            (".", ""),
            ("x.y", "x"),
            ("x.y.z", "y"),
        ],
    )
    def test_get_parent_site_name(self, fqsn, result):
        assert get_parent_site_name(fqsn) == result


def _make_root_and_client_certs(extra_extensions=None):
    root_key, root_pub_key = generate_keys()
    root_cert = generate_cert(
        subject=Identity("root", "nvidia"),
        issuer=Identity("root", "nvidia"),
        signing_pri_key=root_key,
        subject_pub_key=root_pub_key,
        ca=True,
    )
    client_key, client_pub_key = generate_keys()
    client_cert = generate_cert(
        subject=Identity("client", "nvidia"),
        issuer=Identity("root", "nvidia"),
        signing_pri_key=root_key,
        subject_pub_key=client_pub_key,
        extra_extensions=extra_extensions,
    )
    return root_cert, root_key, client_cert, client_key


def test_identity_verifier_accepts_direct_root_signed_cert(tmp_path):
    root_cert, _root_key, client_cert, client_key = _make_root_and_client_certs()
    root_cert_path = tmp_path / "root.crt"
    root_cert_path.write_bytes(serialize_cert(root_cert))
    verifier = IdentityVerifier(str(root_cert_path))
    signature = sign_content("client" + "nonce", client_key, return_str=False)

    assert (
        verifier.verify_common_name(
            "client", "nonce", client_cert, signature, expected_eku=ExtendedKeyUsageOID.CLIENT_AUTH
        )
        is True
    )


def test_identity_verifier_rejects_key_usage_without_digital_signature(tmp_path):
    key_usage = x509.KeyUsage(
        digital_signature=False,
        content_commitment=False,
        key_encipherment=True,
        data_encipherment=False,
        key_agreement=False,
        key_cert_sign=False,
        crl_sign=False,
        encipher_only=False,
        decipher_only=False,
    )
    root_cert, _root_key, client_cert, client_key = _make_root_and_client_certs(extra_extensions=[(key_usage, True)])
    root_cert_path = tmp_path / "root.crt"
    root_cert_path.write_bytes(serialize_cert(root_cert))
    verifier = IdentityVerifier(str(root_cert_path))
    signature = sign_content("client" + "nonce", client_key, return_str=False)

    with pytest.raises(InvalidAsserterCert, match="digitalSignature"):
        verifier.verify_common_name(
            "client", "nonce", client_cert, signature, expected_eku=ExtendedKeyUsageOID.CLIENT_AUTH
        )


def test_identity_verifier_rejects_wrong_extended_key_usage(tmp_path):
    extended_key_usage = x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH])
    root_cert, _root_key, client_cert, client_key = _make_root_and_client_certs(
        extra_extensions=[(extended_key_usage, False)]
    )
    root_cert_path = tmp_path / "root.crt"
    root_cert_path.write_bytes(serialize_cert(root_cert))
    verifier = IdentityVerifier(str(root_cert_path))
    signature = sign_content("client" + "nonce", client_key, return_str=False)

    with pytest.raises(InvalidAsserterCert, match=ExtendedKeyUsageOID.CLIENT_AUTH.dotted_string):
        verifier.verify_common_name(
            "client", "nonce", client_cert, signature, expected_eku=ExtendedKeyUsageOID.CLIENT_AUTH
        )


def test_identity_verifier_accepts_expected_extended_key_usage(tmp_path):
    extended_key_usage = x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CLIENT_AUTH])
    root_cert, _root_key, client_cert, client_key = _make_root_and_client_certs(
        extra_extensions=[(extended_key_usage, False)]
    )
    root_cert_path = tmp_path / "root.crt"
    root_cert_path.write_bytes(serialize_cert(root_cert))
    verifier = IdentityVerifier(str(root_cert_path))
    signature = sign_content("client" + "nonce", client_key, return_str=False)

    assert (
        verifier.verify_common_name(
            "client", "nonce", client_cert, signature, expected_eku=ExtendedKeyUsageOID.CLIENT_AUTH
        )
        is True
    )


def test_identity_verifier_rejects_job_scoped_cert_chain(tmp_path):
    # a leaked job cert (leaf + job CA chaining to root, CN=site) must not be
    # usable to register as the site — the rogue-CP scenario
    root_key, root_pub_key = generate_keys()
    root_cert = generate_cert(
        subject=Identity("root", "nvidia"),
        issuer=Identity("root", "nvidia"),
        signing_pri_key=root_key,
        subject_pub_key=root_pub_key,
        ca=True,
    )
    job_ca_key, job_ca_pub_key = generate_keys()
    job_ca_cert = generate_cert(
        subject=Identity("job_ca.test", "nvidia"),
        issuer=Identity("root", "nvidia"),
        signing_pri_key=root_key,
        subject_pub_key=job_ca_pub_key,
        ca=True,
        ca_path_length=0,
    )
    leaf_key, leaf_pub_key = generate_keys()
    job_id_ext = x509.UnrecognizedExtension(JOB_ID_EXTENSION_OID, b"job-123")
    leaf_cert = generate_cert(
        subject=Identity("client", "nvidia"),
        issuer=Identity("job_ca.test", "nvidia"),
        signing_pri_key=job_ca_key,
        subject_pub_key=leaf_pub_key,
        extra_extensions=[(job_id_ext, False)],
    )
    root_cert_path = tmp_path / "root.crt"
    root_cert_path.write_bytes(serialize_cert(root_cert))
    verifier = IdentityVerifier(str(root_cert_path))
    signature = sign_content("client" + "nonce", leaf_key, return_str=False)

    with pytest.raises(InvalidAsserterCert, match="job-scoped"):
        verifier.verify_common_name(
            "client",
            "nonce",
            leaf_cert,
            signature,
            intermediate_certs=[job_ca_cert],
            expected_eku=ExtendedKeyUsageOID.CLIENT_AUTH,
        )


def test_identity_verifier_rejects_job_extension_even_when_root_issued(tmp_path):
    # the rejection is keyed on the job-id extension, not the issuer
    job_id_ext = x509.UnrecognizedExtension(JOB_ID_EXTENSION_OID, b"job-123")
    root_cert, _root_key, client_cert, client_key = _make_root_and_client_certs(extra_extensions=[(job_id_ext, False)])
    root_cert_path = tmp_path / "root.crt"
    root_cert_path.write_bytes(serialize_cert(root_cert))
    verifier = IdentityVerifier(str(root_cert_path))
    signature = sign_content("client" + "nonce", client_key, return_str=False)

    with pytest.raises(InvalidAsserterCert, match="job-scoped"):
        verifier.verify_common_name(
            "client", "nonce", client_cert, signature, expected_eku=ExtendedKeyUsageOID.CLIENT_AUTH
        )


def test_identity_verifier_wraps_invalid_cert_chain(tmp_path):
    root_cert, _root_key, _client_cert, _client_key = _make_root_and_client_certs()
    other_root_key, _other_root_pub_key = generate_keys()
    client_key, client_pub_key = generate_keys()
    client_cert = generate_cert(
        subject=Identity("client", "nvidia"),
        issuer=Identity("other-root", "nvidia"),
        signing_pri_key=other_root_key,
        subject_pub_key=client_pub_key,
    )
    root_cert_path = tmp_path / "root.crt"
    root_cert_path.write_bytes(serialize_cert(root_cert))
    verifier = IdentityVerifier(str(root_cert_path))

    with pytest.raises(InvalidAsserterCert) as ex_info:
        verifier.verify_common_name("client", "nonce", client_cert, client_key, intermediate_certs=[])

    assert ex_info.value.__cause__ is not None
