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

import pytest
from cryptography import x509
from cryptography.x509.oid import ExtendedKeyUsageOID

from nvflare.fuel.sec.admin_cert import AdminCertValidationError, validate_admin_leaf_cert
from nvflare.lighter.utils import Identity, generate_cert, generate_keys


def _make_admin_cert(role="lead", common_name="alice@nvidia.com", org="nvidia", ca=False, extra_extensions=None):
    root_key, root_pub_key = generate_keys()
    root_cert = generate_cert(
        subject=Identity("root", "nvidia"),
        issuer=Identity("root", "nvidia"),
        signing_pri_key=root_key,
        subject_pub_key=root_pub_key,
        ca=True,
    )
    admin_key, admin_pub_key = generate_keys()
    admin_cert = generate_cert(
        subject=Identity(common_name, org, role),
        issuer=Identity("root", "nvidia"),
        signing_pri_key=root_key,
        subject_pub_key=admin_pub_key,
        ca=ca,
        extra_extensions=extra_extensions,
    )
    return root_cert, admin_cert


@pytest.mark.parametrize("role", ["project_admin", "org_admin", "lead", "member"])
def test_validate_admin_leaf_cert_accepts_flare_roles(role):
    _root_cert, admin_cert = _make_admin_cert(role=role)

    validate_admin_leaf_cert(admin_cert)


def test_validate_admin_leaf_cert_accepts_custom_role():
    _root_cert, admin_cert = _make_admin_cert(role="self_defined")

    validate_admin_leaf_cert(admin_cert)


def test_validate_admin_leaf_cert_accepts_missing_organization():
    _root_cert, admin_cert = _make_admin_cert(org=None)

    validate_admin_leaf_cert(admin_cert)


def test_validate_admin_leaf_cert_accepts_missing_role():
    _root_cert, admin_cert = _make_admin_cert(role=None)

    validate_admin_leaf_cert(admin_cert)


def test_validate_admin_leaf_cert_rejects_ca_cert():
    _root_cert, admin_cert = _make_admin_cert(ca=True)

    with pytest.raises(AdminCertValidationError, match="must not be a CA"):
        validate_admin_leaf_cert(admin_cert)


def test_validate_admin_leaf_cert_rejects_key_usage_without_digital_signature():
    key_usage = x509.KeyUsage(
        digital_signature=False,
        content_commitment=False,
        key_encipherment=False,
        data_encipherment=False,
        key_agreement=False,
        key_cert_sign=False,
        crl_sign=False,
        encipher_only=False,
        decipher_only=False,
    )
    _root_cert, admin_cert = _make_admin_cert(extra_extensions=[(key_usage, True)])

    with pytest.raises(AdminCertValidationError, match="digitalSignature"):
        validate_admin_leaf_cert(admin_cert)


def test_validate_admin_leaf_cert_rejects_eku_without_client_auth():
    _root_cert, admin_cert = _make_admin_cert(
        extra_extensions=[(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), False)]
    )

    with pytest.raises(AdminCertValidationError, match="clientAuth"):
        validate_admin_leaf_cert(admin_cert)
