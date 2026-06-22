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
from nvflare.apis.workspace import Workspace
from nvflare.fuel.sec.authz import AuthorizationService
from nvflare.private.fed.app.default_app_validator import DefaultAppValidator
from nvflare.private.fed.utils.app_authz import AppAuthzService
from nvflare.private.fed.utils.app_deployer import AppDeployer
from nvflare.private.fed.utils.fed_utils import authorize_build_component
from nvflare.private.json_configer import ConfigContext
from nvflare.security.security import EmptyAuthorizer


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


def test_deploy_validation_error_cleans_run_dir_without_writing_meta(tmp_path):
    workspace = _make_workspace(str(tmp_path / "workspace"))

    with patch("nvflare.private.fed.utils.app_deployer.PrivacyService.is_scope_allowed", return_value=True):
        with patch(
            "nvflare.private.fed.utils.app_deployer.AppAuthzService.validate_app",
            return_value=("invalid app", {}),
        ):
            err = AppDeployer().deploy(
                workspace=workspace,
                job_id="job-1",
                job_meta={AppValidationKey.BYOC: True},
                app_name="app",
                app_data=_make_app_zip(),
                fl_ctx=None,
            )

    assert err == "invalid app"
    assert not os.path.exists(workspace.get_run_dir("job-1"))
    assert not os.path.exists(workspace.get_job_meta_path("job-1"))


def test_deploy_authorization_failure_cleans_run_dir(tmp_path):
    workspace = _make_workspace(str(tmp_path / "workspace"))

    with patch("nvflare.private.fed.utils.app_deployer.PrivacyService.is_scope_allowed", return_value=True):
        with patch("nvflare.private.fed.utils.app_deployer.AppAuthzService.validate_app", return_value=("", {})):
            with patch(
                "nvflare.private.fed.utils.app_deployer.AppAuthzService.authorize_app_info",
                return_value=(False, ""),
            ):
                err = AppDeployer().deploy(
                    workspace=workspace,
                    job_id="job-1",
                    job_meta={},
                    app_name="app",
                    app_data=_make_app_zip(),
                    fl_ctx=None,
                )

    assert err == "not authorized"
    assert not os.path.exists(workspace.get_run_dir("job-1"))


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
