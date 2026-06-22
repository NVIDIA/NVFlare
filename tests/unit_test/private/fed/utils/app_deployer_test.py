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

import os
from unittest.mock import patch

import pytest

from nvflare.apis.job_def import JobMetaKey
from nvflare.apis.workspace import Workspace
from nvflare.lighter.tool_consts import NVFLARE_SIG_FILE
from nvflare.private.fed.utils.app_deployer import AppDeployer


def _make_workspace(root_dir: str) -> Workspace:
    os.makedirs(os.path.join(root_dir, "startup"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, "local"), exist_ok=True)
    return Workspace(root_dir, site_name="site-1")


def _job_meta(submitter_role: str):
    return {
        JobMetaKey.SUBMITTER_NAME: "server-claimed@nvidia.com",
        JobMetaKey.SUBMITTER_ORG: "nvidia",
        JobMetaKey.SUBMITTER_ROLE: submitter_role,
    }


def test_deploy_rejects_traversing_job_id_before_touching_outside_path(tmp_path):
    workspace = _make_workspace(str(tmp_path / "workspace"))
    outside = tmp_path / "outside"
    outside.mkdir()
    marker = outside / "marker.txt"
    marker.write_text("keep")

    with pytest.raises(Exception, match="invalid job_id"):
        AppDeployer().deploy(
            workspace=workspace,
            job_id="good/../../outside",
            job_meta={},
            app_name="app",
            app_data=b"not-needed",
            fl_ctx=None,
        )

    assert marker.read_text() == "keep"


@pytest.mark.parametrize(
    "metadata_role,signer_role,expected_err",
    [
        ("lead", "member", "BYOC not permitted"),
        ("member", "lead", None),
    ],
)
def test_deploy_signed_app_authorizes_with_signer_role(tmp_path, metadata_role, signer_role, expected_err):
    workspace = _make_workspace(str(tmp_path / "workspace"))
    signer = ("signer@nvidia.com", "nvidia", signer_role)

    def fake_unzip(_app_data, app_path):
        os.makedirs(app_path, exist_ok=True)
        with open(os.path.join(app_path, NVFLARE_SIG_FILE), "wt") as f:
            f.write("{}")

    def authorize_only_lead(**kwargs):
        authorized = kwargs["submitter_role"] == "lead"
        return authorized, "" if authorized else "BYOC not permitted"

    with (
        patch("nvflare.private.fed.utils.app_deployer.unzip_all_from_bytes", side_effect=fake_unzip),
        patch(
            "nvflare.private.fed.utils.app_deployer.verify_folder_signature_and_get_signers",
            return_value=(True, [signer]),
        ) as mock_verify,
        patch(
            "nvflare.private.fed.utils.app_deployer.AppAuthzService.authorize", side_effect=authorize_only_lead
        ) as mock_authz,
    ):
        err = AppDeployer().deploy(
            workspace=workspace,
            job_id="job-1",
            job_meta=_job_meta(submitter_role=metadata_role),
            app_name="app",
            app_data=b"not-needed",
            fl_ctx=None,
        )

    mock_verify.assert_called_once_with(
        workspace.get_app_dir("job-1"), workspace.get_file_path_in_startup("rootCA.pem")
    )
    assert err == expected_err
    assert mock_authz.call_args.kwargs["submitter_role"] == signer_role


def test_deploy_signed_app_reports_missing_signer_identity(tmp_path):
    workspace = _make_workspace(str(tmp_path / "workspace"))

    def fake_unzip(_app_data, app_path):
        os.makedirs(app_path, exist_ok=True)
        with open(os.path.join(app_path, NVFLARE_SIG_FILE), "wt") as f:
            f.write("{}")

    with (
        patch("nvflare.private.fed.utils.app_deployer.unzip_all_from_bytes", side_effect=fake_unzip),
        patch(
            "nvflare.private.fed.utils.app_deployer.verify_folder_signature_and_get_signers",
            return_value=(True, []),
        ),
        patch("nvflare.private.fed.utils.app_deployer.AppAuthzService.authorize") as mock_authz,
    ):
        err = AppDeployer().deploy(
            workspace=workspace,
            job_id="job-1",
            job_meta=_job_meta(submitter_role="lead"),
            app_name="app",
            app_data=b"not-needed",
            fl_ctx=None,
        )

    assert err == "app app: signature verified but no signer identity could be extracted"
    mock_authz.assert_not_called()


def test_deploy_unsigned_app_uses_metadata_submitter(tmp_path):
    workspace = _make_workspace(str(tmp_path / "workspace"))

    with (
        patch("nvflare.private.fed.utils.app_deployer.unzip_all_from_bytes"),
        patch("nvflare.private.fed.utils.app_deployer.verify_folder_signature_and_get_signers") as mock_verify,
        patch(
            "nvflare.private.fed.utils.app_deployer.AppAuthzService.authorize", return_value=(True, "")
        ) as mock_authz,
    ):
        err = AppDeployer().deploy(
            workspace=workspace,
            job_id="job-1",
            job_meta=_job_meta(submitter_role="member"),
            app_name="app",
            app_data=b"not-needed",
            fl_ctx=None,
        )

    assert err is None
    mock_verify.assert_not_called()
    assert mock_authz.call_args.kwargs["submitter_role"] == "member"
