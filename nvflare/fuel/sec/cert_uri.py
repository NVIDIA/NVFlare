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

"""NVFlare certificate URIs.

Certificate attributes travel as https URI Subject Alternative Names under a root the project
owns rather than as private X.509 extensions: a URI under a controlled domain is globally
unique without an OID allocation, and Go, cryptography and OpenSSL parse it natively.
Readers match the root exactly, ignore URIs on other hosts, and reject malformed URIs
under the root.
"""

from typing import Iterable, List
from urllib.parse import quote, unquote

from cryptography import x509

NVFLARE_CERT_URI_ROOT = "https://nvidia.com/nvflare/"
_V1_PREFIX = NVFLARE_CERT_URI_ROOT + "v1/"

# https://nvidia.com/nvflare/v1/<kind>/<percent-encoded value>
JOB_URI_KIND = "job"  # leaf: the job the credential belongs to
CA_URI_KIND = "ca"  # CA certificate: the role of the CA
CELL_URI_KIND = "cell"  # leaf: an FQCN the certificate may claim in cellnet (with its descendants)
JOB_CA_URI_VALUE = "job"


def cert_uri(kind: str, value: str) -> str:
    return f"{_V1_PREFIX}{kind}/{quote(value, safe='')}"


def job_ca_marker_uri() -> str:
    return cert_uri(CA_URI_KIND, JOB_CA_URI_VALUE)


def uri_general_names(uris: Iterable[str]) -> List[x509.UniformResourceIdentifier]:
    return [x509.UniformResourceIdentifier(uri) for uri in uris]


def cert_uri_values(cert: x509.Certificate, kind: str) -> List[str]:
    """Values of one kind carried by the certificate's NVFlare URI SANs.

    Raises ValueError for a URI under the NVFlare root that is not a well-formed v1 entry.
    """
    try:
        san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    except x509.ExtensionNotFound:
        return []
    values = []
    for uri in san.get_values_for_type(x509.UniformResourceIdentifier):
        if not uri.startswith(NVFLARE_CERT_URI_ROOT):
            continue
        if not uri.startswith(_V1_PREFIX):
            raise ValueError(f"unsupported NVFlare certificate URI: {uri}")
        uri_kind, separator, encoded_value = uri[len(_V1_PREFIX) :].partition("/")
        if not separator or not uri_kind or not encoded_value or "/" in encoded_value:
            raise ValueError(f"malformed NVFlare certificate URI: {uri}")
        if uri_kind == kind:
            values.append(unquote(encoded_value))
    return values
