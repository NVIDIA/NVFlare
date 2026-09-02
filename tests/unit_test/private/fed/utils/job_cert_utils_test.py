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
import stat

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.x509.oid import NameOID

from nvflare.apis.fl_constant import SecureTrainConst
from nvflare.fuel.f3.cellnet.cell_cipher import SimpleCellCipher
from nvflare.lighter.constants import ProvFileName
from nvflare.lighter.utils import (
    Identity,
    generate_cert,
    generate_keys,
    serialize_cert,
    serialize_pri_key,
    verify_cert_chain,
)
from nvflare.private.fed.utils.job_cert_utils import (
    JOB_CA_MARKER_OID,
    JOB_CERT_BACKDATE,
    JOB_CERT_FILE_NAME,
    JOB_CERT_VALID_DAYS,
    JOB_KEY_FILE_NAME,
    apply_job_cert_config,
    find_job_cert,
    get_job_id_from_cert,
    has_job_ca_marker,
    job_startup_files,
    load_job_cert_issuer,
    pack_job_cert_header,
    pick_cell_credential,
    read_job_cert,
    stage_job_startup_dir,
    unpack_job_cert_header,
    write_job_cert,
)


def _write_job_ca(startup_dir, ca_lifetime=datetime.timedelta(days=360), expired=False):
    os.makedirs(startup_dir, exist_ok=True)
    root_key, root_pub = generate_keys()
    root_cert = generate_cert(Identity("root"), Identity("root"), root_key, root_pub, ca=True)

    now = datetime.datetime.now(datetime.timezone.utc)
    if expired:
        not_valid_before = now - datetime.timedelta(days=2)
        not_valid_after = now - datetime.timedelta(days=1)
    else:
        not_valid_before = now
        not_valid_after = now + ca_lifetime

    ca_key, ca_pub = generate_keys()
    marker = x509.UnrecognizedExtension(JOB_CA_MARKER_OID, b"job_ca")
    ca_cert = generate_cert(
        Identity("job_ca.test"),
        Identity("root"),
        root_key,
        ca_pub,
        ca=True,
        ca_path_length=0,
        not_valid_before=not_valid_before,
        not_valid_after=not_valid_after,
        extra_extensions=[(marker, False)],
    )

    with open(os.path.join(startup_dir, ProvFileName.JOB_CA_CERT), "wb") as f:
        f.write(serialize_cert(ca_cert))
    with open(os.path.join(startup_dir, ProvFileName.JOB_CA_KEY), "wb") as f:
        f.write(serialize_pri_key(ca_key))
    return root_cert, ca_cert


def test_no_issuer_without_job_ca(tmp_path):
    assert load_job_cert_issuer(str(tmp_path)) is None


def test_no_issuer_when_job_ca_expired(tmp_path):
    _write_job_ca(str(tmp_path), expired=True)
    assert load_job_cert_issuer(str(tmp_path)) is None


def test_no_issuer_when_job_ca_near_expiry(tmp_path):
    _write_job_ca(str(tmp_path), ca_lifetime=datetime.timedelta(minutes=30))
    assert load_job_cert_issuer(str(tmp_path)) is None


def test_issued_cert_chains_to_root_and_carries_job_id(tmp_path):
    root_cert, ca_cert = _write_job_ca(str(tmp_path))
    issuer = load_job_cert_issuer(str(tmp_path))
    assert issuer is not None

    cert_pem, key_pem = issuer.issue("site-1", "job-123")

    chain = x509.load_pem_x509_certificates(cert_pem)
    assert len(chain) == 2
    leaf, intermediate = chain
    assert intermediate == ca_cert
    verify_cert_chain(leaf_cert=leaf, intermediate_certs=[intermediate], root_ca_cert=root_cert)
    assert leaf.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value == "site-1"
    assert get_job_id_from_cert(leaf) == "job-123"
    assert has_job_ca_marker(intermediate) and not has_job_ca_marker(leaf)
    expected_lifetime = datetime.timedelta(days=JOB_CERT_VALID_DAYS) + JOB_CERT_BACKDATE
    assert leaf.not_valid_after_utc - leaf.not_valid_before_utc == expected_lifetime
    assert b"PRIVATE KEY" in key_pem


def test_issued_cert_validity_clamped_to_job_ca(tmp_path):
    _, ca_cert = _write_job_ca(str(tmp_path), ca_lifetime=datetime.timedelta(days=1))
    issuer = load_job_cert_issuer(str(tmp_path))

    cert_pem, _ = issuer.issue("site-1", "job-123")

    leaf = x509.load_pem_x509_certificates(cert_pem)[0]
    assert leaf.not_valid_after_utc == ca_cert.not_valid_after_utc.replace(microsecond=0)


def test_issue_honors_valid_days(tmp_path):
    _write_job_ca(str(tmp_path))
    issuer = load_job_cert_issuer(str(tmp_path))

    cert_pem, _ = issuer.issue("site-1", "job-123", valid_days=3)

    leaf = x509.load_pem_x509_certificates(cert_pem)[0]
    assert leaf.not_valid_after_utc - leaf.not_valid_before_utc == datetime.timedelta(days=3) + JOB_CERT_BACKDATE
    with pytest.raises(ValueError, match="valid_days"):
        issuer.issue("site-1", "job-123", valid_days=0)


def test_pick_cell_credential_prefers_complete_job_pair():
    config = {
        SecureTrainConst.SSL_CERT: "site.crt",
        SecureTrainConst.PRIVATE_KEY: "site.key",
        SecureTrainConst.JOB_CERT: "job.crt",
        SecureTrainConst.JOB_PRIVATE_KEY: "job.key",
    }
    assert pick_cell_credential(config) == ("job.crt", "job.key")


def test_pick_cell_credential_ignores_partial_job_pair():
    config = {
        SecureTrainConst.SSL_CERT: "site.crt",
        SecureTrainConst.PRIVATE_KEY: "site.key",
        SecureTrainConst.JOB_CERT: "job.crt",
    }
    assert pick_cell_credential(config) == ("site.crt", "site.key")


def test_pack_unpack_job_cert_header_round_trip():
    header = pack_job_cert_header(b"cert-bytes", b"key-bytes")
    assert unpack_job_cert_header(header) == (b"cert-bytes", b"key-bytes")


@pytest.mark.parametrize(
    "header", [None, "not-a-dict", b"bytes", 5, {}, {"cert": "x"}, {"key": "y"}, {"cert": "", "key": "k"}]
)
def test_unpack_job_cert_header_rejects_malformed(header):
    assert unpack_job_cert_header(header) is None


def test_job_id_absent_from_site_cert():
    root_key, root_pub = generate_keys()
    root_cert = generate_cert(Identity("root"), Identity("root"), root_key, root_pub, ca=True)
    assert get_job_id_from_cert(root_cert) is None


def test_write_and_find_job_cert(tmp_path):
    run_dir = str(tmp_path / "run_1")
    assert find_job_cert(run_dir) is None

    write_job_cert(run_dir, b"cert-bytes", b"key-bytes")

    found = find_job_cert(run_dir)
    assert found is not None
    cert_path, key_path = found
    assert cert_path.endswith(JOB_CERT_FILE_NAME)
    assert key_path.endswith(JOB_KEY_FILE_NAME)
    with open(cert_path, "rb") as f:
        assert f.read() == b"cert-bytes"
    assert stat.S_IMODE(os.stat(key_path).st_mode) == 0o600


def test_write_job_cert_overwrite(tmp_path):
    run_dir = str(tmp_path / "run_1")
    write_job_cert(run_dir, b"cert-1", b"key-1")

    write_job_cert(run_dir, b"cert-2", b"key-2")

    cert_path, key_path = find_job_cert(run_dir)
    with open(cert_path, "rb") as f:
        assert f.read() == b"cert-2"
    with open(key_path, "rb") as f:
        assert f.read() == b"key-2"


def test_read_job_cert(tmp_path):
    run_dir = str(tmp_path / "run_1")
    assert read_job_cert(run_dir) is None

    write_job_cert(run_dir, b"cert-bytes", b"key-bytes")

    assert read_job_cert(run_dir) == (b"cert-bytes", b"key-bytes")


def test_apply_job_cert_config_replaces_site_credential(tmp_path):
    run_dir = str(tmp_path / "run_1")
    site_only = {SecureTrainConst.SSL_CERT: "site.crt", SecureTrainConst.PRIVATE_KEY: "site.key"}
    config = dict(site_only)

    assert apply_job_cert_config(config, run_dir) is False
    assert config == site_only

    write_job_cert(run_dir, b"c", b"k")
    assert apply_job_cert_config(config, run_dir) is True
    cert_path, key_path = find_job_cert(run_dir)
    assert config == {
        SecureTrainConst.SSL_CERT: cert_path,
        SecureTrainConst.PRIVATE_KEY: key_path,
        SecureTrainConst.JOB_CERT: cert_path,
        SecureTrainConst.JOB_PRIVATE_KEY: key_path,
    }


def test_job_startup_files_and_staging_exclude_private_keys(tmp_path):
    startup = tmp_path / "startup"
    startup.mkdir()
    for name in ("rootCA.pem", "client.crt", "client.key", "fed_client.json", "job_ca.key", "start.sh"):
        (startup / name).write_text(name)
    (startup / "subdir").mkdir()

    assert job_startup_files(str(startup)) == ["client.crt", "fed_client.json", "rootCA.pem", "start.sh"]

    staged = stage_job_startup_dir(str(startup), str(tmp_path / "job" / "startup"))

    assert sorted(os.listdir(staged)) == ["client.crt", "fed_client.json", "rootCA.pem", "start.sh"]
    assert stat.S_IMODE(os.stat(staged).st_mode) == 0o700
    assert (tmp_path / "job" / "startup" / "rootCA.pem").read_text() == "rootCA.pem"


def test_cell_cipher_works_with_job_cert_chains(tmp_path):
    root_cert, _ = _write_job_ca(str(tmp_path))
    issuer = load_job_cert_issuer(str(tmp_path))

    sj_cert_pem, sj_key_pem = issuer.issue("server", "job-123")
    cj_cert_pem, cj_key_pem = issuer.issue("site-1", "job-123")

    sj_cipher = SimpleCellCipher(
        root_cert,
        serialization.load_pem_private_key(sj_key_pem, password=None),
        x509.load_pem_x509_certificates(sj_cert_pem),
    )
    cj_cipher = SimpleCellCipher(
        root_cert,
        serialization.load_pem_private_key(cj_key_pem, password=None),
        x509.load_pem_x509_certificates(cj_cert_pem),
    )

    cipher_text = sj_cipher.encrypt(b"task data", x509.load_pem_x509_certificates(cj_cert_pem))
    assert cj_cipher.decrypt(cipher_text, x509.load_pem_x509_certificates(sj_cert_pem)) == b"task data"
