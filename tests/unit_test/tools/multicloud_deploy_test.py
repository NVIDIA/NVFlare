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

import importlib.util
import json
import sys
from pathlib import Path

import pytest

DEPLOY_PY = Path(__file__).resolve().parents[3] / "devops" / "multicloud" / "deploy.py"
SPEC = importlib.util.spec_from_file_location("multicloud_deploy", DEPLOY_PY)
DEPLOY_MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = DEPLOY_MODULE
SPEC.loader.exec_module(DEPLOY_MODULE)


class TestPatchResourcesJson:
    def test_replaces_process_launcher_and_writes_study_pvc_file(self, tmp_path):
        kit_dir = tmp_path
        local_dir = kit_dir / "local"
        local_dir.mkdir()
        resources = {
            "components": [
                {
                    "id": "process_launcher",
                    "path": "nvflare.app_common.job_launcher.process_launcher.ProcessJobLauncher",
                    "args": {},
                }
            ]
        }
        (local_dir / "resources.json.default").write_text(json.dumps(resources))

        DEPLOY_MODULE.patch_resources_json(
            kit_dir,
            "nvflare-client-1",
            "nvflare.app_opt.job_launcher.k8s_launcher.ClientK8sJobLauncher",
            {"seLinuxOptions": {"type": "spc_t"}},
        )

        updated = json.loads((local_dir / "resources.json").read_text())
        component = updated["components"][0]
        assert component["id"] == "k8s_launcher"
        assert component["path"] == "nvflare.app_opt.job_launcher.k8s_launcher.ClientK8sJobLauncher"
        assert component["args"]["namespace"] == "nvflare-client-1"
        assert "workspace_pvc" not in component["args"]
        assert component["args"]["security_context"] == {"seLinuxOptions": {"type": "spc_t"}}
        assert (kit_dir / "etc" / "study_data_pvc.yaml").read_text() == "default: nvfldata\n"

    def test_raises_when_process_launcher_component_is_missing(self, tmp_path):
        kit_dir = tmp_path
        local_dir = kit_dir / "local"
        local_dir.mkdir()
        (local_dir / "resources.json.default").write_text(json.dumps({"components": [{"id": "other", "path": "x"}]}))

        with pytest.raises(RuntimeError, match="No ProcessJobLauncher component found"):
            DEPLOY_MODULE.patch_resources_json(
                kit_dir,
                "nvflare-client-1",
                "nvflare.app_opt.job_launcher.k8s_launcher.ClientK8sJobLauncher",
            )
