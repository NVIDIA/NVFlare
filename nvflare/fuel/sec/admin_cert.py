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

from cryptography import x509
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

ADMIN_CERT_PLACEHOLDER_CN = "nvflare-admin"
ALLOWED_FLARE_ADMIN_ROLES = {"project_admin", "org_admin", "lead", "member"}


class AdminCertValidationError(ValueError):
    """Raised when an admin certificate is not acceptable to a FLARE relying party."""


def validate_admin_leaf_cert(cert: x509.Certificate, reject_placeholder_cn: bool = True):
    """Validate the FLARE admin certificate fields that relying parties consume.

    Legacy FLARE leaf certs may omit KeyUsage and EKU. Treat absent extensions
    as unrestricted for compatibility, but enforce admin-client usage
    constraints when the issuer includes them.
    """
    common_name = _require_subject_attr(cert, NameOID.COMMON_NAME, "commonName")
    if reject_placeholder_cn and common_name == ADMIN_CERT_PLACEHOLDER_CN:
        raise AdminCertValidationError("admin certificate commonName must be a real admin identity")

    _require_subject_attr(cert, NameOID.ORGANIZATION_NAME, "organizationName")
    role = _require_subject_attr(cert, NameOID.UNSTRUCTURED_NAME, "unstructuredName")
    if role not in ALLOWED_FLARE_ADMIN_ROLES:
        raise AdminCertValidationError(
            f"admin certificate subject unstructuredName must be one of {sorted(ALLOWED_FLARE_ADMIN_ROLES)}"
        )

    _reject_ca_leaf(cert)
    _validate_key_usage(cert)
    _validate_extended_key_usage(cert)


def _require_subject_attr(cert: x509.Certificate, oid, field_name: str) -> str:
    attrs = cert.subject.get_attributes_for_oid(oid)
    value = attrs[0].value if attrs else ""
    if not value:
        raise AdminCertValidationError(f"admin certificate missing subject {field_name}")
    return value


def _reject_ca_leaf(cert: x509.Certificate):
    try:
        basic_constraints = cert.extensions.get_extension_for_class(x509.BasicConstraints).value
    except x509.ExtensionNotFound:
        return
    if basic_constraints.ca:
        raise AdminCertValidationError("admin certificate must not be a CA certificate")


def _validate_key_usage(cert: x509.Certificate):
    try:
        key_usage = cert.extensions.get_extension_for_class(x509.KeyUsage).value
    except x509.ExtensionNotFound:
        return
    if not key_usage.digital_signature:
        raise AdminCertValidationError("admin certificate keyUsage must allow digitalSignature")


def _validate_extended_key_usage(cert: x509.Certificate):
    try:
        extended_key_usage = cert.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value
    except x509.ExtensionNotFound:
        return
    if ExtendedKeyUsageOID.CLIENT_AUTH not in extended_key_usage:
        raise AdminCertValidationError("admin certificate extendedKeyUsage must allow clientAuth")
