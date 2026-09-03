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
import shutil
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable, List, Optional, Tuple

from cryptography import x509

from nvflare.apis.fl_constant import FLContextKey, SecureTrainConst
from nvflare.fuel.f3.cellnet.identity import get_cert_common_name
from nvflare.fuel.f3.drivers.net_utils import JOB_ID_EXTENSION_OID
from nvflare.lighter.constants import CertExtensionOID, ProvFileName
from nvflare.lighter.utils import (
    Identity,
    bounded_validity,
    generate_cert,
    generate_keys,
    load_crt_bytes,
    load_private_key_file,
    serialize_cert,
    serialize_pri_key,
    write_pri_key_file,
)

JOB_CERT_DIR_NAME = "job_cert"
JOB_CERT_FILE_NAME = "job.crt"
JOB_KEY_FILE_NAME = "job.key"

JOB_CERT_VALID_DAYS = 30

# leaf notBefore is backdated to tolerate clock skew between the issuing
# server and the sites that validate the cert seconds later
JOB_CERT_BACKDATE = datetime.timedelta(minutes=5)

# below this remaining job-CA validity, refuse to issue (and so to deploy) rather
# than hand out certs that expire mid-run
JOB_CA_MIN_REMAINING = datetime.timedelta(hours=1)

NO_JOB_CREDENTIAL = (
    "job has no job credential; secure jobs run only on per-job certificates "
    "(the server must be provisioned with a job CA)"
)

# keys of the credential dict pushed to clients in the deploy message
_PROP_CERT = "cert"
_PROP_KEY = "key"

JOB_CA_MARKER_OID = x509.ObjectIdentifier(CertExtensionOID.JOB_CA_MARKER)


class JobCertError(RuntimeError):
    """A secure-mode job cannot get or use its per-job credential; there is no fallback to site certs."""


def job_cert_paths(run_dir: str) -> Tuple[str, str]:
    cert_dir = os.path.join(run_dir, JOB_CERT_DIR_NAME)
    return os.path.join(cert_dir, JOB_CERT_FILE_NAME), os.path.join(cert_dir, JOB_KEY_FILE_NAME)


def write_job_cert(run_dir: str, cert_chain_pem: bytes, key_pem: bytes):
    cert_path, key_path = job_cert_paths(run_dir)
    os.makedirs(os.path.dirname(cert_path), exist_ok=True)
    with open(cert_path, "wb") as f:
        f.write(cert_chain_pem)
    write_pri_key_file(key_path, key_pem)


def find_job_cert(run_dir: str) -> Optional[Tuple[str, str]]:
    cert_path, key_path = job_cert_paths(run_dir)
    if os.path.isfile(cert_path) and os.path.isfile(key_path):
        return cert_path, key_path
    return None


def read_job_cert(run_dir: str) -> Optional[Tuple[bytes, bytes]]:
    paths = find_job_cert(run_dir)
    if not paths:
        return None
    with open(paths[0], "rb") as f:
        cert_pem = f.read()
    with open(paths[1], "rb") as f:
        key_pem = f.read()
    return cert_pem, key_pem


def require_job_cert(fl_ctx, run_dir: str) -> Optional[Tuple[str, str]]:
    """The job credential paths, or None only in non-secure mode.

    Launchers call this before starting a job process: a secure job without its credential
    is refused instead of being started on site certificates.
    """
    paths = find_job_cert(run_dir)
    if paths is None and fl_ctx.get_prop(FLContextKey.SECURE_MODE, False):
        raise JobCertError(NO_JOB_CREDENTIAL)
    return paths


def apply_job_cert_config(site_config: dict, run_dir: str) -> None:
    """Make the job credential the job process's ssl_cert / ssl_private_key.

    A job process refers to no other credential. Left untouched when the job has none: in
    non-secure mode no certificate is used, and a secure job has already refused to start.
    """
    paths = find_job_cert(run_dir)
    if paths:
        site_config[SecureTrainConst.SSL_CERT], site_config[SecureTrainConst.PRIVATE_KEY] = paths


def job_startup_files(startup_dir: str) -> List[str]:
    """Startup-kit files a job process may see: every regular file except private keys."""
    return sorted(
        f for f in os.listdir(startup_dir) if not f.endswith(".key") and os.path.isfile(os.path.join(startup_dir, f))
    )


def stage_job_startup_dir(startup_dir: str, dest_dir: str) -> str:
    """Copy the startup kit without its private keys to dest_dir, for a launcher to mount into the job."""
    os.makedirs(dest_dir, mode=0o700, exist_ok=True)
    for fname in job_startup_files(startup_dir):
        shutil.copy2(os.path.join(startup_dir, fname), os.path.join(dest_dir, fname))
    return dest_dir


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


def has_job_ca_marker(cert: x509.Certificate) -> bool:
    try:
        cert.extensions.get_extension_for_oid(JOB_CA_MARKER_OID)
        return True
    except x509.ExtensionNotFound:
        return False


class JobCertIssuer:
    """Issues short-lived per-job certificates signed by the provisioned job CA.

    Only the server parent process issues job certs. Use load_job_cert_issuer() to create one
    from a startup kit.
    """

    def __init__(self, ca_cert_pem: bytes, ca_key):
        self.ca_cert_pem = ca_cert_pem
        self.ca_cert = load_crt_bytes(ca_cert_pem)
        self.ca_key = ca_key
        self.ca_cn = get_cert_common_name(self.ca_cert)

    def issue(self, site_name: str, job_id: str, valid_days: int = JOB_CERT_VALID_DAYS) -> Tuple[bytes, bytes]:
        """Issue a per-job credential for one site.

        Returns:
            (cert_chain_pem, key_pem): leaf cert followed by the job CA cert, and the private key.
        """
        pri_key, pub_key = generate_keys()
        not_valid_before, not_valid_after = bounded_validity(self.ca_cert, valid_days, backdate=JOB_CERT_BACKDATE)
        job_id_ext = x509.UnrecognizedExtension(JOB_ID_EXTENSION_OID, job_id.encode("utf-8"))
        cert = generate_cert(
            subject=Identity(site_name),
            issuer=Identity(self.ca_cn),
            signing_pri_key=self.ca_key,
            subject_pub_key=pub_key,
            not_valid_before=not_valid_before,
            not_valid_after=not_valid_after,
            extra_extensions=[(job_id_ext, False)],
        )
        return serialize_cert(cert) + self.ca_cert_pem, serialize_pri_key(pri_key)

    def issue_many(
        self, site_names: Iterable[str], job_id: str, valid_days: int = JOB_CERT_VALID_DAYS
    ) -> Dict[str, Tuple[bytes, bytes]]:
        """Issue credentials for several sites at once; RSA key generation dominates and runs in parallel."""
        names = list(site_names)
        if not names:
            return {}
        with ThreadPoolExecutor(max_workers=min(8, len(names))) as pool:
            return dict(zip(names, pool.map(lambda name: self.issue(name, job_id, valid_days), names)))


def load_job_cert_issuer(startup_dir: str) -> JobCertIssuer:
    """Create a JobCertIssuer from the startup kit's job CA.

    Raises JobCertError when the kit has no job CA (provisioned before this feature or with
    CertBuilder's enable_job_ca off) or the job CA is about to expire; secure-mode jobs must
    not run without a per-job credential.
    """
    cert_path = os.path.join(startup_dir, ProvFileName.JOB_CA_CERT)
    key_path = os.path.join(startup_dir, ProvFileName.JOB_CA_KEY)
    if not (os.path.isfile(cert_path) and os.path.isfile(key_path)):
        raise JobCertError(
            f"server startup kit has no job CA ({ProvFileName.JOB_CA_CERT} / {ProvFileName.JOB_CA_KEY}); "
            "re-provision the project (CertBuilder enable_job_ca) to run jobs in secure mode"
        )

    with open(cert_path, "rb") as f:
        ca_cert_pem = f.read()
    ca_cert = load_crt_bytes(ca_cert_pem)
    if ca_cert.not_valid_after_utc <= datetime.datetime.now(datetime.timezone.utc) + JOB_CA_MIN_REMAINING:
        raise JobCertError(
            f"job CA expires at {ca_cert.not_valid_after_utc.isoformat()} (less than {JOB_CA_MIN_REMAINING} left); "
            "re-provision the project to renew it"
        )
    return JobCertIssuer(ca_cert_pem, load_private_key_file(key_path))
