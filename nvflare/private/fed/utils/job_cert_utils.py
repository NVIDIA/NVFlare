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
import os
from typing import Optional, Tuple

from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.x509.oid import NameOID

from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.lighter.constants import ProvFileName
from nvflare.lighter.utils import Identity, generate_cert, generate_keys, serialize_cert, serialize_pri_key

JOB_CERT_DIR_NAME = "job_cert"
JOB_CERT_FILE_NAME = "job.crt"
JOB_KEY_FILE_NAME = "job.key"

JOB_CERT_VALID_DAYS = 30

# keys of the credential dict pushed to clients in the deploy message
PROP_CERT = "cert"
PROP_KEY = "key"

# Private extension under NVIDIA's IANA enterprise arc (1.3.6.1.4.1.5703),
# carrying the job ID a per-job certificate is bound to.
JOB_ID_EXTENSION_OID = x509.ObjectIdentifier("1.3.6.1.4.1.5703.300.1")


def write_job_cert(run_dir: str, cert_chain_pem: bytes, key_pem: bytes):
    cert_dir = os.path.join(run_dir, JOB_CERT_DIR_NAME)
    os.makedirs(cert_dir, exist_ok=True)
    with open(os.path.join(cert_dir, JOB_CERT_FILE_NAME), "wb") as f:
        f.write(cert_chain_pem)
    key_path = os.path.join(cert_dir, JOB_KEY_FILE_NAME)
    fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(key_pem)


def find_job_cert(run_dir: str) -> Optional[Tuple[str, str]]:
    cert_path = os.path.join(run_dir, JOB_CERT_DIR_NAME, JOB_CERT_FILE_NAME)
    key_path = os.path.join(run_dir, JOB_CERT_DIR_NAME, JOB_KEY_FILE_NAME)
    if os.path.isfile(cert_path) and os.path.isfile(key_path):
        return cert_path, key_path
    return None


def get_job_id_from_cert(cert: x509.Certificate) -> Optional[str]:
    try:
        ext = cert.extensions.get_extension_for_oid(JOB_ID_EXTENSION_OID)
    except x509.ExtensionNotFound:
        return None
    return ext.value.value.decode("utf-8")


class JobCertIssuer:
    """Issues short-lived per-job certificates signed by the provisioned job CA.

    The issuer only runs in the server parent process. It is enabled when the server startup kit
    contains job_ca.crt/job_ca.key (provisioned with CertBuilder's enable_job_ca option); otherwise
    job cells fall back to site certificates.
    """

    def __init__(self, startup_dir: str):
        self.logger = get_obj_logger(self)
        self.enabled = False
        self.ca_cert_pem = None
        self.ca_cert = None
        self.ca_key = None
        self.ca_cn = None

        cert_path = os.path.join(startup_dir, ProvFileName.JOB_CA_CERT)
        key_path = os.path.join(startup_dir, ProvFileName.JOB_CA_KEY)
        if not (os.path.isfile(cert_path) and os.path.isfile(key_path)):
            return

        with open(cert_path, "rb") as f:
            self.ca_cert_pem = f.read()
        self.ca_cert = x509.load_pem_x509_certificate(self.ca_cert_pem)
        with open(key_path, "rb") as f:
            self.ca_key = serialization.load_pem_private_key(f.read(), password=None)
        self.ca_cn = self.ca_cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value

        if self.ca_cert.not_valid_after_utc <= datetime.datetime.now(datetime.timezone.utc):
            self.logger.warning(
                f"job CA expired at {self.ca_cert.not_valid_after_utc.isoformat()}; "
                "job cells will fall back to site certificates"
            )
            return
        self.enabled = True

    def issue(self, site_name: str, job_id: str, valid_days: int = JOB_CERT_VALID_DAYS) -> Tuple[bytes, bytes]:
        """Issue a per-job credential for one site.

        Returns:
            (cert_chain_pem, key_pem): leaf cert followed by the job CA cert, and the private key.
        """
        if not self.enabled:
            raise RuntimeError("job cert issuer is not enabled")

        pri_key, pub_key = generate_keys()
        now = datetime.datetime.now(datetime.timezone.utc)
        not_valid_after = min(now + datetime.timedelta(days=valid_days), self.ca_cert.not_valid_after_utc)
        job_id_ext = x509.UnrecognizedExtension(JOB_ID_EXTENSION_OID, job_id.encode("utf-8"))
        cert = generate_cert(
            subject=Identity(site_name),
            issuer=Identity(self.ca_cn),
            signing_pri_key=self.ca_key,
            subject_pub_key=pub_key,
            not_valid_before=now,
            not_valid_after=not_valid_after,
            extra_extensions=[(job_id_ext, False)],
        )
        cert_chain_pem = serialize_cert(cert) + self.ca_cert_pem
        return cert_chain_pem, serialize_pri_key(pri_key)
