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

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from nvflare.apis.fl_constant import SystemComponents, SystemVarName
from nvflare.apis.job_def import JobMetaKey
from nvflare.apis.workspace import Workspace
from nvflare.app_common.hub.hub_app_deployer import HubAppDeployer
from nvflare.lighter.tool_consts import NVFLARE_SIG_FILE

SITE_NAME = "site-1"
JOB_ID = "job-1"
T2_JOB_ID = f"{JOB_ID}_t2"


def _write_json(path: str, data: dict):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data))


def _read_json(path: str) -> dict:
    return json.loads(Path(path).read_text())


def _make_workspace(tmp_path) -> Workspace:
    root = tmp_path / "workspace"
    (root / "startup").mkdir(parents=True)
    (root / "local").mkdir()
    (root / "startup" / "rootCA.pem").write_text("root ca")

    workspace = Workspace(str(root), site_name=SITE_NAME)

    _write_json(workspace.get_client_app_config_file_path(JOB_ID), {"client": "original"})
    _write_json(workspace.get_server_app_config_file_path(JOB_ID), {"server": "original"})
    _write_json(
        workspace.get_job_meta_path(JOB_ID),
        {
            JobMetaKey.SUBMITTER_NAME.value: "admin@nvidia.com",
            JobMetaKey.SUBMITTER_ORG.value: "nvidia",
            JobMetaKey.SUBMITTER_ROLE.value: "lead",
            JobMetaKey.SCOPE.value: "global",
        },
    )
    Path(workspace.get_app_dir(JOB_ID), NVFLARE_SIG_FILE).write_text('{"sig": "originator"}')

    _write_json(
        workspace.get_file_path_in_site_config(HubAppDeployer.HUB_CLIENT_CONFIG_TEMPLATE_NAME),
        {"client": "hub"},
    )
    _write_json(
        workspace.get_file_path_in_site_config(HubAppDeployer.HUB_SERVER_CONFIG_TEMPLATE_NAME),
        {"workflows": [{"id": "hub-controller"}]},
    )
    return workspace


def _make_fl_ctx(workspace: Workspace):
    fed_client = MagicMock()
    fed_client.cell.get_root_url_for_child.return_value = "grpc://t1-root:8002"
    fed_client.cell.is_secure.return_value = True

    validator = MagicMock()
    validator.validate.return_value = (True, "", {JobMetaKey.JOB_ID.value: JOB_ID})

    props = {
        SystemVarName.WORKSPACE: workspace.root_dir,
        SystemComponents.FED_CLIENT: fed_client,
        SystemComponents.JOB_META_VALIDATOR: validator,
    }

    fl_ctx = MagicMock()
    fl_ctx.get_prop.side_effect = lambda key: props.get(key)
    return fl_ctx


class TestHubAppDeployerSignatureHandling:
    def test_prepare_verifies_before_rewrite_and_strips_t2_signature(self, tmp_path):
        workspace = _make_workspace(tmp_path)
        fl_ctx = _make_fl_ctx(workspace)

        def _verify_before_rewrite(app_path, root_ca_path):
            assert app_path == workspace.get_app_dir(JOB_ID)
            assert root_ca_path == str(Path(workspace.get_startup_kit_dir(), "rootCA.pem"))
            assert _read_json(workspace.get_client_app_config_file_path(JOB_ID)) == {"client": "original"}
            assert _read_json(workspace.get_server_app_config_file_path(T2_JOB_ID)) == {"server": "original"}
            return True

        def _load_after_t2_signature_removed(from_path, def_name):
            assert from_path == workspace.root_dir
            assert def_name == T2_JOB_ID
            assert not Path(workspace.get_app_dir(T2_JOB_ID), NVFLARE_SIG_FILE).exists()
            return b"t2-job"

        with (
            patch(
                "nvflare.app_common.hub.hub_app_deployer.verify_folder_signature", side_effect=_verify_before_rewrite
            ),
            patch(
                "nvflare.app_common.hub.hub_app_deployer.load_job_def_bytes",
                side_effect=_load_after_t2_signature_removed,
            ),
        ):
            err, meta, job_def = HubAppDeployer().prepare(fl_ctx, workspace, JOB_ID, remove_tmp_t2_dir=False)

        assert err == ""
        assert meta[JobMetaKey.JOB_ID.value] == JOB_ID
        assert job_def == b"t2-job"
        assert _read_json(workspace.get_client_app_config_file_path(JOB_ID)) == {"client": "hub"}
        assert _read_json(workspace.get_server_app_config_file_path(T2_JOB_ID))["workflows"] == [
            {"id": "hub-controller"}
        ]
        assert Path(workspace.get_app_dir(JOB_ID), NVFLARE_SIG_FILE).exists()

    def test_prepare_rejects_invalid_signature_before_rewrite(self, tmp_path):
        workspace = _make_workspace(tmp_path)
        fl_ctx = _make_fl_ctx(workspace)

        def _verify_before_rewrite(app_path, root_ca_path):
            assert _read_json(workspace.get_client_app_config_file_path(JOB_ID)) == {"client": "original"}
            assert _read_json(workspace.get_server_app_config_file_path(T2_JOB_ID)) == {"server": "original"}
            return False

        with (
            patch(
                "nvflare.app_common.hub.hub_app_deployer.verify_folder_signature", side_effect=_verify_before_rewrite
            ),
            patch("nvflare.app_common.hub.hub_app_deployer.load_job_def_bytes") as load_job_def_bytes,
        ):
            err, meta, job_def = HubAppDeployer().prepare(fl_ctx, workspace, JOB_ID, remove_tmp_t2_dir=False)

        assert err == "hub: job signature verification failed before forwarding"
        assert meta is None
        assert job_def is None
        load_job_def_bytes.assert_not_called()
        assert _read_json(workspace.get_client_app_config_file_path(JOB_ID)) == {"client": "original"}
        assert _read_json(workspace.get_server_app_config_file_path(T2_JOB_ID)) == {"server": "original"}
