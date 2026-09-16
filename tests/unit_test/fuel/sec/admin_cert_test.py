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
from cryptography.x509.oid import ExtendedKeyUsageOID, ExtensionOID

from nvflare.fuel.sec.admin_cert import (
    MAX_ADMIN_STUDIES,
    AdminCertValidationError,
    get_admin_study_entitlements,
    validate_admin_leaf_cert,
)
from nvflare.fuel.sec.cert_uri import ADMIN_STUDY_URI_PREFIX, cert_uri_values
from nvflare.lighter.utils import Identity, generate_cert, generate_keys

_PROJECT = "demo"


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


def _study_uri(study, project=_PROJECT):
    return f"{ADMIN_STUDY_URI_PREFIX}{project}/study/{study}"


def _make_cert_with_uris(*uris):
    san = x509.SubjectAlternativeName([x509.UniformResourceIdentifier(uri) for uri in uris])
    extensions = x509.Extensions([x509.Extension(ExtensionOID.SUBJECT_ALTERNATIVE_NAME, False, san)])
    return type("CertWithUriSans", (), {"extensions": extensions})()


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


def test_get_admin_study_entitlements_returns_empty_when_san_is_absent():
    _root_cert, admin_cert = _make_admin_cert()

    assert get_admin_study_entitlements(admin_cert) == ()


def test_get_admin_study_entitlements_reads_uri_sans_and_ignores_unrelated_uris():
    cert = _make_cert_with_uris(
        "https://example.com/not-an-nvflare-claim",
        _study_uri("cancer-research"),
        _study_uri("study_2", project="other-project"),
    )

    assert get_admin_study_entitlements(cert) == ("cancer-research", "study_2")


@pytest.mark.parametrize(
    "project",
    [
        "other-project",
        "demo%2Fproject",
        "demo%2fproject",
        "%64emo",
        "caf%c3%a9",
        "demo%252Fproject",
    ],
)
def test_get_admin_study_entitlements_does_not_restrict_project_label(project):
    uri = _study_uri("study-a", project=project)

    assert get_admin_study_entitlements(_make_cert_with_uris(uri)) == ("study-a",)


@pytest.mark.parametrize(
    "uri",
    [
        "https://nvidia.com/nvflare/v2/project/demo/study/study-a",
        ADMIN_STUDY_URI_PREFIX,
        f"{ADMIN_STUDY_URI_PREFIX}{_PROJECT}/all-studies",
        _study_uri("default"),
        _study_uri("default\n"),
        _study_uri("study-a\n"),
        _study_uri("Invalid"),
        _study_uri("study-a", project=""),
        _study_uri("study-a", project="demo/project"),
        _study_uri("study-a", project="demo%ZZ"),
    ],
)
def test_get_admin_study_entitlements_rejects_invalid_nvflare_uri(uri):
    cert = _make_cert_with_uris(uri)
    with pytest.raises(AdminCertValidationError):
        get_admin_study_entitlements(cert)
    for kind in ("cell", "job", "ca"):
        with pytest.raises(ValueError):
            cert_uri_values(cert, kind)


def test_get_admin_study_entitlements_rejects_duplicates_and_too_many_studies():
    with pytest.raises(AdminCertValidationError, match="duplicate"):
        get_admin_study_entitlements(_make_cert_with_uris(_study_uri("study-a"), _study_uri("study-a")))

    uris = [_study_uri(f"study-{i}") for i in range(MAX_ADMIN_STUDIES + 1)]
    with pytest.raises(AdminCertValidationError, match="too many"):
        get_admin_study_entitlements(_make_cert_with_uris(*uris))
