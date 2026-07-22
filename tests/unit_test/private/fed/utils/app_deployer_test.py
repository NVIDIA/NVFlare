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

import io
import json
import os
from unittest.mock import patch
from zipfile import ZipFile

import pytest

from nvflare.apis.app_validation import AppValidationKey
from nvflare.apis.fl_constant import FLContextKey, SiteType
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import JobMetaKey
from nvflare.apis.workspace import Workspace
from nvflare.fuel.sec.authz import AuthorizationService
from nvflare.lighter.tool_consts import NVFLARE_SIG_FILE
from nvflare.private.fed.app.default_app_validator import DefaultAppValidator
from nvflare.private.fed.utils.app_authz import AppAuthzService
from nvflare.private.fed.utils.app_deployer import AppDeployer
from nvflare.private.fed.utils.fed_utils import authorize_build_component
from nvflare.private.json_configer import ConfigContext
from nvflare.security.security import EmptyAuthorizer, FLAuthorizer


def _make_workspace(root_dir: str) -> Workspace:
    os.makedirs(os.path.join(root_dir, "startup"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, "local"), exist_ok=True)
    return Workspace(root_dir, site_name="site-1")


def _make_app_zip(include_custom=False) -> bytes:
    output = io.BytesIO()
    with ZipFile(output, "w") as zip_file:
        zip_file.writestr("config/config_fed_client.json", "{}")
        if include_custom:
            zip_file.writestr("custom/local_code.py", "VALUE = 1\n")
    return output.getvalue()


def _job_meta(submitter_role: str):
    return {
        JobMetaKey.SUBMITTER_NAME: "server-claimed@nvidia.com",
        JobMetaKey.SUBMITTER_ORG: "nvidia",
        JobMetaKey.SUBMITTER_ROLE: submitter_role,
    }


def _launcher_meta(mode: str, source: str, values: dict) -> dict:
    if source == "default":
        return {JobMetaKey.JOB_LAUNCHER_SPEC.value: {"default": {mode: values}}}
    if source == "site":
        return {JobMetaKey.JOB_LAUNCHER_SPEC.value: {"site-1": {mode: values}}}
    if source == "legacy":
        return {JobMetaKey.RESOURCE_SPEC.value: {"site-1": {mode: values}}}
    raise ValueError(f"unsupported launcher metadata source: {source}")


def _byoc_none_authorizer():
    return FLAuthorizer(
        "site-org",
        {
            "format_version": "1.0",
            "permissions": {
                "lead": {
                    "submit_job": "any",
                    "byoc": "none",
                }
            },
        },
    )


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


def test_deploy_strips_server_supplied_byoc_without_local_byoc_app(tmp_path):
    workspace = _make_workspace(str(tmp_path / "workspace"))
    job_meta = {AppValidationKey.BYOC: True}

    with patch("nvflare.private.fed.utils.app_deployer.PrivacyService.is_scope_allowed", return_value=True):
        with patch("nvflare.private.fed.utils.app_deployer.AppAuthzService.validate_app", return_value=("", {})):
            with patch(
                "nvflare.private.fed.utils.app_deployer.AppAuthzService.authorize_app_info",
                return_value=(True, ""),
            ):
                err = AppDeployer().deploy(
                    workspace=workspace,
                    job_id="job-1",
                    job_meta=job_meta,
                    app_name="app",
                    app_data=_make_app_zip(),
                    fl_ctx=None,
                )

    assert err is None
    with open(workspace.get_job_meta_path("job-1")) as f:
        local_meta = json.load(f)
    assert local_meta.get(AppValidationKey.BYOC, False) is False

    resources_file = os.path.join(workspace.get_site_config_dir(), "resources.json")
    with open(resources_file, "w") as f:
        json.dump({"class_allow_list": []}, f)
    fl_ctx = FLContext()
    fl_ctx.set_prop(FLContextKey.WORKSPACE_OBJECT, workspace, sticky=False, private=True)
    fl_ctx.set_prop(FLContextKey.CURRENT_JOB_ID, "job-1", sticky=False, private=False)

    build_err = authorize_build_component(
        {"path": "subprocess.Popen", "args": {}},
        ConfigContext(),
        None,
        fl_ctx=fl_ctx,
        event_handlers=[],
    )
    assert "subprocess.Popen" in build_err
    assert "allow_list" in build_err


def test_deploy_detects_custom_dir_as_local_byoc_for_allow_list(tmp_path):
    workspace = _make_workspace(str(tmp_path / "workspace"))
    AuthorizationService.initialize(EmptyAuthorizer())
    AppAuthzService.initialize(DefaultAppValidator(site_type=SiteType.CLIENT))

    try:
        with patch("nvflare.private.fed.utils.app_deployer.PrivacyService.is_scope_allowed", return_value=True):
            err = AppDeployer().deploy(
                workspace=workspace,
                job_id="job-1",
                job_meta={AppValidationKey.BYOC: False},
                app_name="app",
                app_data=_make_app_zip(include_custom=True),
                fl_ctx=None,
            )
    finally:
        AppAuthzService.initialize(None)

    assert err is None
    with open(workspace.get_job_meta_path("job-1")) as f:
        local_meta = json.load(f)
    assert local_meta[AppValidationKey.BYOC] is True

    resources_file = os.path.join(workspace.get_site_config_dir(), "resources.json")
    with open(resources_file, "w") as f:
        json.dump({"class_allow_list": []}, f)
    fl_ctx = FLContext()
    fl_ctx.set_prop(FLContextKey.WORKSPACE_OBJECT, workspace, sticky=False, private=True)
    fl_ctx.set_prop(FLContextKey.CURRENT_JOB_ID, "job-1", sticky=False, private=False)

    assert (
        authorize_build_component(
            {"path": "subprocess.Popen", "args": {}},
            ConfigContext(),
            None,
            fl_ctx=fl_ctx,
            event_handlers=[],
        )
        == ""
    )


@pytest.mark.parametrize("mode", ["docker", "k8s"])
@pytest.mark.parametrize("source", ["default", "site", "legacy"])
@pytest.mark.parametrize("field", ["image", "python_path"])
def test_deploy_requires_byoc_for_job_selected_launcher_content(tmp_path, monkeypatch, mode, source, field):
    workspace = _make_workspace(str(tmp_path / "workspace"))
    monkeypatch.setattr(AppAuthzService, "app_validator", DefaultAppValidator(site_type=SiteType.CLIENT))
    monkeypatch.setattr(AuthorizationService, "the_authorizer", _byoc_none_authorizer())
    job_meta = _job_meta("lead")
    job_meta[AppValidationKey.BYOC] = False
    job_meta.update(_launcher_meta(mode, source, {field: "attacker.example/value"}))

    with patch("nvflare.private.fed.utils.app_deployer.PrivacyService.is_scope_allowed", return_value=True):
        err = AppDeployer().deploy(workspace, "job-1", job_meta, "app", _make_app_zip(), None)

    assert err == "BYOC not permitted"
    assert not os.path.exists(workspace.get_run_dir("job-1"))


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
            "nvflare.private.fed.utils.app_deployer.AppAuthzService.validate_app",
            return_value=("", {AppValidationKey.BYOC: True}),
        ),
        patch(
            "nvflare.private.fed.utils.app_deployer.AppAuthzService.authorize_app_info",
            side_effect=authorize_only_lead,
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
        patch("nvflare.private.fed.utils.app_deployer.AppAuthzService.authorize_app_info") as mock_authz,
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
        patch("nvflare.private.fed.utils.app_deployer.AppAuthzService.validate_app", return_value=("", {})),
        patch(
            "nvflare.private.fed.utils.app_deployer.AppAuthzService.authorize_app_info", return_value=(True, "")
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
