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
import logging
import os
from typing import Optional, Tuple

from cryptography import x509
from cryptography.x509.oid import NameOID

from nvflare.apis.fl_constant import SecureTrainConst
from nvflare.lighter.constants import CertExtensionOID, ProvFileName
from nvflare.lighter.utils import (
    Identity,
    generate_cert,
    generate_keys,
    load_crt_bytes,
    load_private_key_file,
    serialize_cert,
    serialize_pri_key,
    write_pri_key_file,
)

logger = logging.getLogger(__name__)

JOB_CERT_DIR_NAME = "job_cert"
JOB_CERT_FILE_NAME = "job.crt"
JOB_KEY_FILE_NAME = "job.key"

JOB_CERT_VALID_DAYS = 30

# leaf notBefore is backdated to tolerate clock skew between the issuing
# server and the sites that validate the cert seconds later
JOB_CERT_BACKDATE = datetime.timedelta(minutes=5)

# below this remaining job-CA validity, refuse to issue so jobs fall back to
# site certificates instead of getting certs that expire mid-run
JOB_CA_MIN_REMAINING = datetime.timedelta(hours=1)

# keys of the credential dict pushed to clients in the deploy message
_PROP_CERT = "cert"
_PROP_KEY = "key"

JOB_ID_EXTENSION_OID = x509.ObjectIdentifier(CertExtensionOID.JOB_ID)
JOB_CA_MARKER_OID = x509.ObjectIdentifier(CertExtensionOID.JOB_CA_MARKER)


def write_job_cert(run_dir: str, cert_chain_pem: bytes, key_pem: bytes):
    cert_dir = os.path.join(run_dir, JOB_CERT_DIR_NAME)
    os.makedirs(cert_dir, exist_ok=True)
    with open(os.path.join(cert_dir, JOB_CERT_FILE_NAME), "wb") as f:
        f.write(cert_chain_pem)
    write_pri_key_file(os.path.join(cert_dir, JOB_KEY_FILE_NAME), key_pem)


def find_job_cert(run_dir: str) -> Optional[Tuple[str, str]]:
    cert_path = os.path.join(run_dir, JOB_CERT_DIR_NAME, JOB_CERT_FILE_NAME)
    key_path = os.path.join(run_dir, JOB_CERT_DIR_NAME, JOB_KEY_FILE_NAME)
    if os.path.isfile(cert_path) and os.path.isfile(key_path):
        return cert_path, key_path
    return None


def job_cert_config_entries(run_dir: str) -> Optional[dict]:
    """Security config entries pointing at the job credential, or None if the job has none."""
    paths = find_job_cert(run_dir)
    if not paths:
        return None
    return {SecureTrainConst.JOB_CERT: paths[0], SecureTrainConst.JOB_PRIVATE_KEY: paths[1]}


def pick_cell_credential(config: dict) -> Tuple[str, str]:
    """Cert/key for a job cell: the per-job credential when present, else the site credential.

    Preferring the job credential means the job cell never needs the site's long-lived key.
    The job credential is used only when both parts are present, so a partial config cannot
    pair a job cert with the site key.
    """
    job_cert = config.get(SecureTrainConst.JOB_CERT)
    job_key = config.get(SecureTrainConst.JOB_PRIVATE_KEY)
    if job_cert and job_key:
        return job_cert, job_key
    return config[SecureTrainConst.SSL_CERT], config[SecureTrainConst.PRIVATE_KEY]


def pack_job_cert_header(cert_chain_pem: bytes, key_pem: bytes) -> dict:
    return {_PROP_CERT: cert_chain_pem.decode("ascii"), _PROP_KEY: key_pem.decode("ascii")}


def unpack_job_cert_header(header) -> Optional[Tuple[bytes, bytes]]:
    """Decode a pushed job credential; None for anything malformed (e.g. version skew)."""
    if not isinstance(header, dict):
        return None
    cert_pem = header.get(_PROP_CERT)
    key_pem = header.get(_PROP_KEY)
    if not (isinstance(cert_pem, str) and isinstance(key_pem, str) and cert_pem and key_pem):
        return None
    try:
        return cert_pem.encode("ascii"), key_pem.encode("ascii")
    except UnicodeEncodeError:
        return None


def has_job_id_extension(cert: x509.Certificate) -> bool:
    try:
        cert.extensions.get_extension_for_oid(JOB_ID_EXTENSION_OID)
        return True
    except x509.ExtensionNotFound:
        return False


def has_job_ca_marker(cert: x509.Certificate) -> bool:
    try:
        cert.extensions.get_extension_for_oid(JOB_CA_MARKER_OID)
        return True
    except x509.ExtensionNotFound:
        return False


def get_job_id_from_cert(cert: x509.Certificate) -> Optional[str]:
    try:
        ext = cert.extensions.get_extension_for_oid(JOB_ID_EXTENSION_OID)
    except x509.ExtensionNotFound:
        return None
    return ext.value.value.decode("utf-8")


class JobCertIssuer:
    """Issues short-lived per-job certificates signed by the provisioned job CA.

    Only the server parent process issues job certs. Use load_job_cert_issuer() to create one
    from a startup kit.
    """

    def __init__(self, ca_cert_pem: bytes, ca_key):
        self.ca_cert_pem = ca_cert_pem
        self.ca_cert = load_crt_bytes(ca_cert_pem)
        self.ca_key = ca_key
        self.ca_cn = self.ca_cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value

    def issue(self, site_name: str, job_id: str) -> Tuple[bytes, bytes]:
        """Issue a per-job credential for one site.

        Returns:
            (cert_chain_pem, key_pem): leaf cert followed by the job CA cert, and the private key.
        """
        pri_key, pub_key = generate_keys()
        now = datetime.datetime.now(datetime.timezone.utc)
        not_valid_after = min(now + datetime.timedelta(days=JOB_CERT_VALID_DAYS), self.ca_cert.not_valid_after_utc)
        job_id_ext = x509.UnrecognizedExtension(JOB_ID_EXTENSION_OID, job_id.encode("utf-8"))
        cert = generate_cert(
            subject=Identity(site_name),
            issuer=Identity(self.ca_cn),
            signing_pri_key=self.ca_key,
            subject_pub_key=pub_key,
            not_valid_before=now - JOB_CERT_BACKDATE,
            not_valid_after=not_valid_after,
            extra_extensions=[(job_id_ext, False)],
        )
        cert_chain_pem = serialize_cert(cert) + self.ca_cert_pem
        return cert_chain_pem, serialize_pri_key(pri_key)


def load_job_cert_issuer(startup_dir: str) -> Optional[JobCertIssuer]:
    """Create a JobCertIssuer from the startup kit's job CA, or None if absent or near expiry.

    The job CA is absent from kits provisioned before this feature or with CertBuilder's
    enable_job_ca option off; job cells then fall back to site certificates.
    """
    cert_path = os.path.join(startup_dir, ProvFileName.JOB_CA_CERT)
    key_path = os.path.join(startup_dir, ProvFileName.JOB_CA_KEY)
    if not (os.path.isfile(cert_path) and os.path.isfile(key_path)):
        return None

    with open(cert_path, "rb") as f:
        ca_cert_pem = f.read()
    ca_cert = load_crt_bytes(ca_cert_pem)
    if ca_cert.not_valid_after_utc <= datetime.datetime.now(datetime.timezone.utc) + JOB_CA_MIN_REMAINING:
        logger.warning(
            f"job CA expires at {ca_cert.not_valid_after_utc.isoformat()}; "
            "job cells will fall back to site certificates"
        )
        return None
    return JobCertIssuer(ca_cert_pem, load_private_key_file(key_path))
