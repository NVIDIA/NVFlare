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
Readers match the root exactly and ignore URIs on other hosts. All readers reject
malformed entries, including study URIs.
"""

import re
from typing import Iterable, List
from urllib.parse import quote, unquote

from cryptography import x509

from nvflare.apis.job_def import DEFAULT_STUDY
from nvflare.apis.utils.format_check import type_pattern_mapping

NVFLARE_CERT_URI_ROOT = "https://nvidia.com/nvflare/"
_V1_PREFIX = NVFLARE_CERT_URI_ROOT + "v1/"
ADMIN_STUDY_URI_PREFIX = _V1_PREFIX + "project/"

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


def parse_admin_study_uri(uri: str) -> str:
    """Validate a study URI and return its study; the project label is informational."""
    if not uri.startswith(ADMIN_STUDY_URI_PREFIX):
        raise ValueError("unsupported study URI")
    project, separator, study = uri[len(ADMIN_STUDY_URI_PREFIX) :].partition("/study/")
    if not separator or not re.fullmatch(r"(?:[A-Za-z0-9._~-]|%[0-9A-Fa-f]{2})+", project):
        raise ValueError("malformed study URI: invalid project URI segment")
    if study == DEFAULT_STUDY or not re.fullmatch(type_pattern_mapping["study"], study):
        raise ValueError("malformed study URI: invalid study name")
    return study


def cert_uri_values(cert: x509.Certificate, kind: str) -> List[str]:
    """Values of one kind carried by the certificate's NVFlare URI SANs.

    Study URIs are validated but are not identity values. Other entries under the NVFlare root must
    be well-formed v1 identity URIs or raise ValueError.
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
        if uri.startswith(ADMIN_STUDY_URI_PREFIX):
            parse_admin_study_uri(uri)
            continue
        uri_kind, separator, encoded_value = uri[len(_V1_PREFIX) :].partition("/")
        if not separator or not uri_kind or not encoded_value or "/" in encoded_value:
            raise ValueError(f"malformed NVFlare certificate URI: {uri}")
        if uri_kind == kind:
            values.append(unquote(encoded_value))
    return values
