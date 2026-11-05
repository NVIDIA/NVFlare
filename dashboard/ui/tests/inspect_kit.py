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

"""Check encrypted startup-kit contents using the PIN displayed by the UI."""

import json
import sys
from zipfile import ZipFile

from cryptography import x509
from cryptography.x509.oid import NameOID

if __name__ == "__main__":
    filename, pin, kind, name, organization = sys.argv[1:]
    with ZipFile(filename) as kit:
        assert kit.getinfo("startup/client.crt").flag_bits & 1, "Certificate ZIP entry must be encrypted"
        certificate = x509.load_pem_x509_certificate(kit.read("startup/client.crt", pwd=pin.encode()))
        assert certificate.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value == name
        assert certificate.subject.get_attributes_for_oid(NameOID.ORGANIZATION_NAME)[0].value == organization
        if kind == "client":
            config = json.loads(kit.read("startup/fed_client.json", pwd=pin.encode()))
            assert config["client"]["connection_security"] == "mtls"
        else:
            assert certificate.subject.get_attributes_for_oid(NameOID.UNSTRUCTURED_NAME)[0].value == "org_admin"
    print(json.dumps({"kind": kind, "name": name, "organization": organization, "encrypted": True}))
