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
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from nvflare.apis.fl_constant import SecureTrainConst
from nvflare.fuel.f3.cellnet.cell_cipher import SimpleCellCipher
from nvflare.fuel.sec.cert_uri import ADMIN_STUDY_URI_PREFIX, CELL_URI_KIND, cert_uri_values, job_ca_marker_uri
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
    JOB_CERT_FILE_NAME,
    JOB_CERT_VALID_DAYS,
    JOB_KEY_FILE_NAME,
    JobCertError,
    apply_job_cert_config,
    find_job_cert,
    get_cert_job_id,
    has_job_ca_marker,
    job_cell_scopes,
    job_cert_uris,
    job_startup_files,
    load_job_cert_issuer,
    pack_job_cert_header,
    read_job_cert,
    stage_job_startup_dir,
    unpack_job_cert_header,
    workspace_transfer_cell_name,
    write_job_cert,
)


def _write_job_ca(startup_dir, ca_lifetime=datetime.timedelta(days=360), expired=False):
    os.makedirs(startup_dir, exist_ok=True)
    root_key, root_pub = generate_keys()
    root_cert = generate_cert(Identity("root"), Identity("root"), root_key, root_pub, ca=True)
    with open(os.path.join(startup_dir, "rootCA.pem"), "wb") as f:
        f.write(serialize_cert(root_cert))

    now = datetime.datetime.now(datetime.timezone.utc)
    if expired:
        not_valid_before = now - datetime.timedelta(days=2)
        not_valid_after = now - datetime.timedelta(days=1)
    else:
        not_valid_before = now
        not_valid_after = now + ca_lifetime

    ca_key, ca_pub = generate_keys()
    ca_cert = generate_cert(
        Identity("job_ca.test"),
        Identity("root"),
        root_key,
        ca_pub,
        ca=True,
        ca_path_length=0,
        not_valid_before=not_valid_before,
        not_valid_after=not_valid_after,
        uri_names=[job_ca_marker_uri()],
    )

    with open(os.path.join(startup_dir, ProvFileName.JOB_CA_CERT), "wb") as f:
        f.write(serialize_cert(ca_cert))
    with open(os.path.join(startup_dir, ProvFileName.JOB_CA_KEY), "wb") as f:
        f.write(serialize_pri_key(ca_key))
    return root_cert, ca_cert


@pytest.mark.parametrize("suffix", ["", "demo/study/", "demo/study/study-a\n"])
@pytest.mark.parametrize("cert_type", ["site", "job", "ca"])
def test_malformed_study_uri_rejected_on_non_admin_certificates(cert_type, suffix):
    key, public_key = generate_keys()
    identity = Identity("site-1")
    uris = job_cert_uris("site-1", "job-123") if cert_type == "job" else []
    if cert_type == "ca":
        uris.append(job_ca_marker_uri())
    # The malformed claim follows valid identity URIs so they cannot mask it.
    cert = generate_cert(
        identity, identity, key, public_key, ca=cert_type == "ca", uri_names=uris + [ADMIN_STUDY_URI_PREFIX + suffix]
    )
    with pytest.raises(ValueError, match="malformed study URI"):
        cert_uri_values(cert, CELL_URI_KIND)
    with pytest.raises(ValueError, match="malformed study URI"):
        get_cert_job_id(cert)
    with pytest.raises(ValueError, match="malformed study URI"):
        has_job_ca_marker(cert)


def _write_external_job_ca(startup, failure=None, ec_key=False):
    now = datetime.datetime.now(datetime.timezone.utc)
    root_key, root_pub = generate_keys()
    root = generate_cert(
        Identity("root"),
        Identity("root"),
        root_key,
        root_pub,
        ca=True,
        ca_path_length=1 if failure == "root_path_length" else None,
        not_valid_before=now + datetime.timedelta(days=1) if failure == "future_root" else now,
    )
    issuer_key, issuer_pub = generate_keys()
    issuer = generate_cert(
        Identity("approved-issuer", "IT"),
        root.subject,
        root_key,
        issuer_pub,
        ca=True,
        ca_path_length=0 if failure == "path_length" else 1,
        valid_days=2,
        not_valid_after=now + datetime.timedelta(minutes=30) if failure == "near_expiry_issuer" else None,
    )
    ca_key = ec.generate_private_key(ec.SECP256R1()) if ec_key else generate_keys()[0]
    if failure == "unsupported_key":
        ca_key = ed25519.Ed25519PrivateKey.generate()
    name = x509.Name(
        [
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Federation"),
            x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, "Jobs"),
            x509.NameAttribute(NameOID.COMMON_NAME, "job_ca.test"),
        ]
    )
    builder = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(issuer.subject)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now + datetime.timedelta(days=1) if failure == "future" else now)
        .not_valid_after(now + datetime.timedelta(days=30))
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(issuer_pub), critical=False)
    )
    if failure != "missing_constraints":
        builder = builder.add_extension(
            x509.BasicConstraints(
                ca=failure != "not_ca",
                path_length=None if failure == "not_ca" else (1 if failure == "delegating_ca" else 0),
            ),
            critical=failure != "noncritical_constraints",
        )
    if failure != "missing_usage":
        builder = builder.add_extension(
            x509.KeyUsage(False, False, False, False, False, failure != "bad_usage", True, False, False),
            critical=True,
        )
    if failure != "missing_marker":
        builder = builder.add_extension(
            x509.SubjectAlternativeName([x509.UniformResourceIdentifier(job_ca_marker_uri())]), critical=False
        )
    if failure == "client_only":
        builder = builder.add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CLIENT_AUTH]), critical=False)
    ca = builder.sign(root_key if failure == "bad_signature" else issuer_key, hashes.SHA256())
    chain = serialize_cert(ca) + (b"" if failure == "missing_issuer" else serialize_cert(issuer))
    (startup / "job_ca.crt").write_bytes(b"not PEM" if failure == "malformed" else chain)
    installed_key = generate_keys()[0] if failure == "mismatch" else ca_key
    (startup / "job_ca.key").write_bytes(
        installed_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    if failure == "untrusted":
        other_key, other_pub = generate_keys()
        root = generate_cert(Identity("other"), Identity("other"), other_key, other_pub, ca=True)
    (startup / "rootCA.pem").write_bytes(serialize_cert(root))
    return root, issuer, ca


@pytest.mark.parametrize("ec_key", [False, True])
def test_external_job_ca_preserves_dn_chain_and_ancestor_lifetime(tmp_path, ec_key):
    root, intermediate, ca = _write_external_job_ca(tmp_path, ec_key=ec_key)
    issuer = load_job_cert_issuer(str(tmp_path))
    cert_pem, _ = issuer.issue("site-1", "job-1", "site-1")
    leaf, *chain = x509.load_pem_x509_certificates(cert_pem)
    assert leaf.issuer == ca.subject
    assert chain == [ca, intermediate]
    assert leaf.not_valid_after_utc == intermediate.not_valid_after_utc
    verify_cert_chain(leaf, chain, root)

    # Even a job-CA holder omitting job restrictions cannot assert parent identity.
    from nvflare.private.fed.utils.identity_utils import IdentityVerifier, InvalidAsserterCert

    key, pub = generate_keys()
    forged_parent = generate_cert(Identity("site-1"), ca.subject, issuer.ca_key, pub)
    with pytest.raises(InvalidAsserterCert, match="job CA cannot assert site identity"):
        IdentityVerifier(str(tmp_path / "rootCA.pem")).verify_common_name(
            "site-1",
            "nonce",
            forged_parent,
            b"unused",
            intermediate_certs=chain,
        )


@pytest.mark.parametrize(
    "failure",
    [
        "path_length",
        "unsupported_key",
        "future",
        "missing_constraints",
        "not_ca",
        "delegating_ca",
        "missing_usage",
        "bad_usage",
        "missing_marker",
        "client_only",
        "bad_signature",
        "missing_issuer",
        "malformed",
        "mismatch",
        "untrusted",
        "root_path_length",
        "future_root",
        "near_expiry_issuer",
        "noncritical_constraints",
    ],
)
def test_external_job_ca_rejected_before_issuance(tmp_path, failure):
    _write_external_job_ca(tmp_path, failure=failure)
    with pytest.raises(JobCertError, match="install an approved"):
        load_job_cert_issuer(str(tmp_path))


def test_issuer_requires_job_ca(tmp_path):
    with pytest.raises(JobCertError, match="no job CA"):
        load_job_cert_issuer(str(tmp_path))


def test_issuer_rejects_expired_job_ca(tmp_path):
    _write_job_ca(str(tmp_path), expired=True)
    with pytest.raises(JobCertError, match="expires at"):
        load_job_cert_issuer(str(tmp_path))


def test_issuer_rejects_job_ca_near_expiry(tmp_path):
    _write_job_ca(str(tmp_path), ca_lifetime=datetime.timedelta(minutes=30))
    with pytest.raises(JobCertError, match="expires at"):
        load_job_cert_issuer(str(tmp_path))


def test_issued_cert_chains_to_root_and_carries_job_id(tmp_path):
    root_cert, ca_cert = _write_job_ca(str(tmp_path))
    issuer = load_job_cert_issuer(str(tmp_path))
    assert issuer is not None

    cert_pem, key_pem = issuer.issue("site-1", "job-123", "site-1")

    chain = x509.load_pem_x509_certificates(cert_pem)
    assert len(chain) == 2
    leaf, intermediate = chain
    assert intermediate == ca_cert
    verify_cert_chain(leaf_cert=leaf, intermediate_certs=[intermediate], root_ca_cert=root_cert)
    assert leaf.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value == "site-1"
    assert get_cert_job_id(leaf) == "job-123"
    assert cert_uri_values(leaf, CELL_URI_KIND) == job_cell_scopes("site-1", "job-123")
    assert has_job_ca_marker(intermediate) and not has_job_ca_marker(leaf)
    assert leaf.not_valid_before_utc >= max(root_cert.not_valid_before_utc, ca_cert.not_valid_before_utc)
    assert datetime.timedelta(days=JOB_CERT_VALID_DAYS) <= leaf.not_valid_after_utc - leaf.not_valid_before_utc
    assert b"PRIVATE KEY" in key_pem


def test_issued_cert_validity_clamped_to_job_ca(tmp_path):
    _, ca_cert = _write_job_ca(str(tmp_path), ca_lifetime=datetime.timedelta(days=1))
    issuer = load_job_cert_issuer(str(tmp_path))

    cert_pem, _ = issuer.issue("site-1", "job-123", "site-1")

    leaf = x509.load_pem_x509_certificates(cert_pem)[0]
    assert leaf.not_valid_after_utc == ca_cert.not_valid_after_utc.replace(microsecond=0)


def test_issue_honors_valid_days(tmp_path):
    _write_job_ca(str(tmp_path))
    issuer = load_job_cert_issuer(str(tmp_path))

    cert_pem, _ = issuer.issue("site-1", "job-123", "site-1", valid_days=3)

    leaf = x509.load_pem_x509_certificates(cert_pem)[0]
    assert datetime.timedelta(days=3) <= leaf.not_valid_after_utc - leaf.not_valid_before_utc
    assert leaf.not_valid_after_utc - leaf.not_valid_before_utc < datetime.timedelta(days=3, minutes=5)


def test_issue_many_issues_one_credential_per_site(tmp_path):
    _write_job_ca(str(tmp_path))
    issuer = load_job_cert_issuer(str(tmp_path))

    creds = issuer.issue_many({"site-1": "site-1", "site-2": "relay-1.site-2"}, "job-123")

    assert set(creds) == {"site-1", "site-2"}
    leaves = {name: x509.load_pem_x509_certificates(cert_pem)[0] for name, (cert_pem, _) in creds.items()}
    assert {leaf.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value for leaf in leaves.values()} == set(creds)
    assert leaves["site-1"].public_key() != leaves["site-2"].public_key()
    assert cert_uri_values(leaves["site-2"], CELL_URI_KIND) == job_cell_scopes("relay-1.site-2", "job-123")
    assert issuer.issue_many({}, "job-123") == {}


def test_job_cert_uris_name_the_job_and_its_cells():
    assert workspace_transfer_cell_name("job-1") == "ws_transfer_job-1"
    assert job_cell_scopes("relay-1.site-1", "job-1") == ["relay-1.site-1.job-1", "relay-1.site-1.ws_transfer_job-1"]
    assert job_cert_uris("site-1", "job-1") == [
        "https://nvidia.com/nvflare/v1/job/job-1",
        "https://nvidia.com/nvflare/v1/cell/site-1.job-1",
        "https://nvidia.com/nvflare/v1/cell/site-1.ws_transfer_job-1",
    ]


def test_get_cert_job_id_rejects_a_cert_claiming_several_jobs():
    key, pub_key = generate_keys()
    cert = generate_cert(
        Identity("site-1"),
        Identity("site-1"),
        key,
        pub_key,
        uri_names=["https://nvidia.com/nvflare/v1/job/job-1", "https://nvidia.com/nvflare/v1/job/job-2"],
    )

    with pytest.raises(ValueError, match="several jobs"):
        get_cert_job_id(cert)


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
    assert get_cert_job_id(root_cert) is None


def test_write_find_read_job_cert(tmp_path):
    run_dir = str(tmp_path / "run_1")
    assert find_job_cert(run_dir) is None
    assert read_job_cert(run_dir) is None

    write_job_cert(run_dir, b"cert-1", b"key-1")
    write_job_cert(run_dir, b"cert-2", b"key-2")

    cert_path, key_path = find_job_cert(run_dir)
    assert cert_path.endswith(JOB_CERT_FILE_NAME) and key_path.endswith(JOB_KEY_FILE_NAME)
    assert read_job_cert(run_dir) == (b"cert-2", b"key-2")
    assert stat.S_IMODE(os.stat(key_path).st_mode) == 0o600


def test_apply_job_cert_config_replaces_site_credential(tmp_path):
    run_dir = str(tmp_path / "run_1")
    site_only = {SecureTrainConst.SSL_CERT: "site.crt", SecureTrainConst.PRIVATE_KEY: "site.key"}
    config = dict(site_only)

    apply_job_cert_config(config, run_dir)
    assert config == site_only

    write_job_cert(run_dir, b"c", b"k")
    apply_job_cert_config(config, run_dir)
    cert_path, key_path = find_job_cert(run_dir)
    assert config == {SecureTrainConst.SSL_CERT: cert_path, SecureTrainConst.PRIVATE_KEY: key_path}


def test_job_startup_files_and_staging_exclude_private_keys(tmp_path):
    startup = tmp_path / "startup"
    startup.mkdir()
    for name in (
        "rootCA.pem",
        "customRootCA.pem",
        "client.crt",
        "client.key",
        "fed_client.json",
        "job_ca.key",
        "start.sh",
        "enrollment-token",
        "ca-config.json",
        "credentials.pem",
        "signature.json",
        "client_context.tenseal",
        "authorization.json",
        "app__j_resources.json",
    ):
        (startup / name).write_text(name)
    (startup / "subdir").mkdir()

    expected = [
        "app__j_resources.json",
        "authorization.json",
        "client.crt",
        "client_context.tenseal",
        "customRootCA.pem",
        "fed_client.json",
        "rootCA.pem",
        "signature.json",
    ]
    assert job_startup_files(str(startup)) == expected

    staged = stage_job_startup_dir(str(startup), str(tmp_path / "job" / "startup"))

    assert sorted(os.listdir(staged)) == expected
    assert stat.S_IMODE(os.stat(staged).st_mode) == 0o700
    assert (tmp_path / "job" / "startup" / "rootCA.pem").read_text() == "rootCA.pem"


def test_cell_cipher_works_with_job_cert_chains(tmp_path):
    root_cert, _ = _write_job_ca(str(tmp_path))
    issuer = load_job_cert_issuer(str(tmp_path))

    sj_cert_pem, sj_key_pem = issuer.issue("server", "job-123", "server")
    cj_cert_pem, cj_key_pem = issuer.issue("site-1", "job-123", "site-1")

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
