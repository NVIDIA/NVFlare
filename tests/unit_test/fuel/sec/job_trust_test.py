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
import time

import pytest

from nvflare.apis.job_def import JobMetaKey
from nvflare.fuel.sec.job_trust import (
    JOB_SUBMITTER_PRINCIPAL_META_KEY,
    JobTrustError,
    attach_job_authorization,
    get_job_authorization,
    issue_job_authorization,
    verify_job_authorization,
)
from nvflare.lighter.utils import Identity, generate_cert, generate_keys, serialize_cert, sign_content


class _IdAsserter:
    def __init__(self, cert, pri_key, cn="server"):
        self.cert = cert
        self.cert_data = serialize_cert(cert)
        self.cn = cn
        self.pri_key = pri_key

    def sign(self, content, return_str=True):
        return sign_content(content, self.pri_key, return_str=return_str)


def _certs(tmp_path, server_not_valid_before=None, server_not_valid_after=None):
    root_pri, root_pub = generate_keys()
    root_cert = generate_cert(
        subject=Identity("root"),
        issuer=Identity("root"),
        signing_pri_key=root_pri,
        subject_pub_key=root_pub,
        ca=True,
    )
    server_pri, server_pub = generate_keys()
    server_cert = generate_cert(
        subject=Identity("server"),
        issuer=Identity("root"),
        signing_pri_key=root_pri,
        subject_pub_key=server_pub,
        server_default_host="server",
        not_valid_before=server_not_valid_before,
        not_valid_after=server_not_valid_after,
    )
    root_path = tmp_path / "rootCA.pem"
    root_path.write_bytes(serialize_cert(root_cert))
    return str(root_path), _IdAsserter(server_cert, server_pri)


def _job_meta():
    return {
        JobMetaKey.JOB_ID.value: "job-1",
        JobMetaKey.JOB_NAME.value: "hello",
        JobMetaKey.STUDY.value: "study-a",
        JobMetaKey.SUBMITTER_NAME.value: "admin@nvidia.com",
        JobMetaKey.SUBMITTER_ORG.value: "nvidia",
        JobMetaKey.SUBMITTER_ROLE.value: "project_admin",
        JobMetaKey.SUBMIT_TIME_ISO.value: "2026-05-14T00:00:00+00:00",
        JOB_SUBMITTER_PRINCIPAL_META_KEY: {
            "subject": "keycloak-subject",
            "username": "admin@nvidia.com",
            "email": "admin@nvidia.com",
            "org": "nvidia",
            "effective_role": "project_admin",
            "auth_method": "oidc",
            "issuer": "https://keycloak.example.com/realms/nvflare",
            "token_id": "token-id",
        },
    }


def test_issue_and_verify_job_authorization(tmp_path):
    root_path, id_asserter = _certs(tmp_path)
    app_data = b"app zip bytes"
    # use a realistic clock: verification also checks the signing cert validity window
    issued_at = time.time()
    authorization = issue_job_authorization(
        job_meta=_job_meta(),
        app_name="hello-client",
        app_data=app_data,
        target_sites=["site-a", "site-b"],
        id_asserter=id_asserter,
        issued_at=issued_at,
    )

    manifest = verify_job_authorization(
        authorization=authorization,
        app_data=app_data,
        root_ca_path=root_path,
        site_name="site-a",
        expected_job_id="job-1",
        expected_app_name="hello-client",
        expected_server_identity="server",
        now=issued_at + 1.0,
    )

    assert manifest["app_sha256"]
    assert manifest["target_sites"] == ["site-a", "site-b"]
    assert manifest["submitter"]["subject"] == "keycloak-subject"
    assert manifest["submitter"]["auth_method"] == "oidc"
    assert manifest["server_identity"] == "server"
    assert manifest["expires_at"] == issued_at + 3600.0


def test_job_authorization_rejects_wrong_app_hash(tmp_path):
    root_path, id_asserter = _certs(tmp_path)
    authorization = issue_job_authorization(
        job_meta=_job_meta(),
        app_name="hello-client",
        app_data=b"expected",
        target_sites=["site-a"],
        id_asserter=id_asserter,
    )

    with pytest.raises(JobTrustError, match="app hash"):
        verify_job_authorization(
            authorization=authorization,
            app_data=b"tampered",
            root_ca_path=root_path,
            site_name="site-a",
        )


def test_job_authorization_rejects_untargeted_site(tmp_path):
    root_path, id_asserter = _certs(tmp_path)
    authorization = issue_job_authorization(
        job_meta=_job_meta(),
        app_name="hello-client",
        app_data=b"app",
        target_sites=["site-a"],
        id_asserter=id_asserter,
    )

    with pytest.raises(JobTrustError, match="does not target"):
        verify_job_authorization(
            authorization=authorization,
            app_data=b"app",
            root_ca_path=root_path,
            site_name="site-b",
        )


def test_job_authorization_rejects_bad_server_cert_chain(tmp_path):
    root_path, _id_asserter = _certs(tmp_path)
    rogue_pri, rogue_pub = generate_keys()
    rogue_cert = generate_cert(
        subject=Identity("server"),
        issuer=Identity("server"),
        signing_pri_key=rogue_pri,
        subject_pub_key=rogue_pub,
        ca=True,
    )
    authorization = issue_job_authorization(
        job_meta=_job_meta(),
        app_name="hello-client",
        app_data=b"app",
        target_sites=["site-a"],
        id_asserter=_IdAsserter(rogue_cert, rogue_pri),
    )

    with pytest.raises(Exception):
        verify_job_authorization(
            authorization=authorization,
            app_data=b"app",
            root_ca_path=root_path,
            site_name="site-a",
        )


def test_job_authorization_rejects_non_server_identity(tmp_path):
    root_pri, root_pub = generate_keys()
    root_cert = generate_cert(
        subject=Identity("root"),
        issuer=Identity("root"),
        signing_pri_key=root_pri,
        subject_pub_key=root_pub,
        ca=True,
    )
    client_pri, client_pub = generate_keys()
    client_cert = generate_cert(
        subject=Identity("client-a"),
        issuer=Identity("root"),
        signing_pri_key=root_pri,
        subject_pub_key=client_pub,
    )
    root_path = tmp_path / "rootCA.pem"
    root_path.write_bytes(serialize_cert(root_cert))
    authorization = issue_job_authorization(
        job_meta=_job_meta(),
        app_name="hello-client",
        app_data=b"app",
        target_sites=["site-a"],
        id_asserter=_IdAsserter(client_cert, client_pri, cn="client-a"),
    )

    with pytest.raises(JobTrustError, match="does not match expected server"):
        verify_job_authorization(
            authorization=authorization,
            app_data=b"app",
            root_ca_path=str(root_path),
            site_name="site-a",
            expected_server_identity="server",
        )


def test_job_authorization_rejects_expired_authorization(tmp_path):
    root_path, id_asserter = _certs(tmp_path)
    authorization = issue_job_authorization(
        job_meta=_job_meta(),
        app_name="hello-client",
        app_data=b"app",
        target_sites=["site-a"],
        id_asserter=id_asserter,
        issued_at=100.0,
        expires_at=200.0,
    )

    with pytest.raises(JobTrustError, match="expired"):
        verify_job_authorization(
            authorization=authorization,
            app_data=b"app",
            root_ca_path=root_path,
            site_name="site-a",
            now=600.0,
            clock_skew=0.0,
        )


def test_job_authorization_rejects_future_issued_authorization(tmp_path):
    root_path, id_asserter = _certs(tmp_path)
    authorization = issue_job_authorization(
        job_meta=_job_meta(),
        app_name="hello-client",
        app_data=b"app",
        target_sites=["site-a"],
        id_asserter=id_asserter,
        issued_at=600.0,
        expires_at=700.0,
    )

    with pytest.raises(JobTrustError, match="issued in the future"):
        verify_job_authorization(
            authorization=authorization,
            app_data=b"app",
            root_ca_path=root_path,
            site_name="site-a",
            now=100.0,
            clock_skew=0.0,
        )


def test_expired_authorization_error_reports_clock_skew():
    """The error message must show the computed skew and the allowed bound for diagnosis."""
    root_pri, root_pub = generate_keys()
    root_cert = generate_cert(Identity("root"), Identity("root"), root_pri, root_pub, ca=True)
    server_pri, server_pub = generate_keys()
    server_cert = generate_cert(
        subject=Identity("server"),
        issuer=Identity("root"),
        signing_pri_key=root_pri,
        subject_pub_key=server_pub,
        server_default_host="server",
    )
    authorization = issue_job_authorization(
        job_meta=_job_meta(),
        app_name="hello-client",
        app_data=b"app",
        target_sites=["site-a"],
        id_asserter=_IdAsserter(server_cert, server_pri),
        issued_at=100.0,
        expires_at=200.0,
    )

    with pytest.raises(JobTrustError, match=r"expires_at is 400 seconds before verification time"):
        verify_job_authorization(
            authorization=authorization,
            app_data=b"app",
            root_ca_path="unused",
            site_name="site-a",
            now=600.0,
            clock_skew=30.0,
        )


def test_future_issued_authorization_error_reports_clock_skew():
    root_pri, root_pub = generate_keys()
    server_pri, server_pub = generate_keys()
    server_cert = generate_cert(
        subject=Identity("server"),
        issuer=Identity("root"),
        signing_pri_key=root_pri,
        subject_pub_key=server_pub,
        server_default_host="server",
    )
    authorization = issue_job_authorization(
        job_meta=_job_meta(),
        app_name="hello-client",
        app_data=b"app",
        target_sites=["site-a"],
        id_asserter=_IdAsserter(server_cert, server_pri),
        issued_at=600.0,
        expires_at=700.0,
    )

    with pytest.raises(JobTrustError, match=r"issued_at is 500 seconds ahead of verification time"):
        verify_job_authorization(
            authorization=authorization,
            app_data=b"app",
            root_ca_path="unused",
            site_name="site-a",
            now=100.0,
            clock_skew=30.0,
        )


def test_job_authorization_rejects_expired_server_cert(tmp_path):
    now = datetime.datetime.now(datetime.timezone.utc)
    root_path, id_asserter = _certs(
        tmp_path,
        server_not_valid_before=now - datetime.timedelta(days=30),
        server_not_valid_after=now - datetime.timedelta(days=1),
    )
    authorization = issue_job_authorization(
        job_meta=_job_meta(),
        app_name="hello-client",
        app_data=b"app",
        target_sites=["site-a"],
        id_asserter=id_asserter,
    )

    with pytest.raises(JobTrustError, match="server certificate expired"):
        verify_job_authorization(
            authorization=authorization,
            app_data=b"app",
            root_ca_path=root_path,
            site_name="site-a",
        )


def test_job_authorization_rejects_not_yet_valid_server_cert(tmp_path):
    now = datetime.datetime.now(datetime.timezone.utc)
    root_path, id_asserter = _certs(
        tmp_path,
        server_not_valid_before=now + datetime.timedelta(days=1),
        server_not_valid_after=now + datetime.timedelta(days=30),
    )
    authorization = issue_job_authorization(
        job_meta=_job_meta(),
        app_name="hello-client",
        app_data=b"app",
        target_sites=["site-a"],
        id_asserter=id_asserter,
    )

    with pytest.raises(JobTrustError, match="server certificate is not yet valid"):
        verify_job_authorization(
            authorization=authorization,
            app_data=b"app",
            root_ca_path=root_path,
            site_name="site-a",
        )


def test_job_authorization_accepts_cert_just_expired_within_clock_skew(tmp_path):
    now = datetime.datetime.now(datetime.timezone.utc)
    root_path, id_asserter = _certs(
        tmp_path,
        server_not_valid_before=now - datetime.timedelta(days=30),
        server_not_valid_after=now - datetime.timedelta(seconds=30),
    )
    authorization = issue_job_authorization(
        job_meta=_job_meta(),
        app_name="hello-client",
        app_data=b"app",
        target_sites=["site-a"],
        id_asserter=id_asserter,
    )

    # default clock_skew (5 minutes) covers the 30-second overrun
    manifest = verify_job_authorization(
        authorization=authorization,
        app_data=b"app",
        root_ca_path=root_path,
        site_name="site-a",
    )
    assert manifest["server_identity"] == "server"


def test_attach_and_get_job_authorization():
    root_pri, root_pub = generate_keys()
    root_cert = generate_cert(Identity("root"), Identity("root"), root_pri, root_pub, ca=True)
    job_meta = _job_meta()
    authorization = {"manifest": {}, "signature": "sig", "server_cert": serialize_cert(root_cert).decode("utf-8")}

    attach_job_authorization(job_meta=job_meta, app_name="hello-client", authorization=authorization)

    assert get_job_authorization(job_meta, "hello-client") == authorization
