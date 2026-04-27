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
    def test_replaces_process_launcher_and_writes_study_pvc_file_to_local_workspace(self, tmp_path):
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
        assert component["args"]["study_data_pvc_file_path"] == "/var/tmp/nvflare/workspace/local/study_data_pvc.yaml"
        assert component["args"]["security_context"] == {"seLinuxOptions": {"type": "spc_t"}}
        assert (kit_dir / "local" / "study_data_pvc.yaml").read_text() == "default: nvfldata\n"

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


class TestHelmDeployFlow:
    def test_helm_release_status_parses_status(self, monkeypatch):
        monkeypatch.setattr(
            DEPLOY_MODULE,
            "run_quiet",
            lambda cmd: DEPLOY_MODULE.FakeProc(0, stdout=json.dumps({"info": {"status": "failed"}})),
        )

        assert DEPLOY_MODULE.helm_release_status("/tmp/kubeconfig", "site-1", "ns-1") == "failed"

    def test_failed_release_restage_and_upgrade_install(self, monkeypatch, tmp_path):
        kit_dir = tmp_path / "site-1"
        chart_dir = kit_dir / "helm_chart"
        (kit_dir / "startup").mkdir(parents=True)
        (kit_dir / "local").mkdir()
        chart_dir.mkdir()
        (kit_dir / "local" / "study_data_pvc.yaml").write_text("default: nvfldata\n")

        participant = DEPLOY_MODULE.Participant(
            name="site-1",
            namespace="ns-1",
            kubeconfig="/tmp/kubeconfig",
            role="client",
            cloud="gcp",
            launcher_class="launcher",
            image="repo/image:tag",
            pvc_config={"nvflws": {"sc": "sc", "access": "ReadWriteMany", "size": "10Gi"}},
            pod_annotations={"karpenter.sh/do-not-disrupt": r"value,with=helm\chars"},
        )

        kubectl_calls = []
        helm_calls = []

        monkeypatch.setattr(DEPLOY_MODULE, "check_auth_for", lambda cloud: None)
        monkeypatch.setattr(DEPLOY_MODULE, "namespace_exists", lambda kubeconfig, ns: True)
        monkeypatch.setattr(DEPLOY_MODULE, "helm_release_status", lambda kubeconfig, name, ns: "failed")
        monkeypatch.setattr(DEPLOY_MODULE, "pod_exists", lambda kubeconfig, ns, name: False)
        monkeypatch.setattr(DEPLOY_MODULE, "kubectl", lambda *args: kubectl_calls.append(args))
        monkeypatch.setattr(DEPLOY_MODULE, "helm", lambda kubeconfig, *args: helm_calls.append((kubeconfig, args)))
        monkeypatch.setattr(
            DEPLOY_MODULE,
            "run_quiet",
            lambda cmd: DEPLOY_MODULE.FakeProc(0) if "get" in cmd and "pvc" in cmd else DEPLOY_MODULE.FakeProc(1),
        )
        monkeypatch.setattr(DEPLOY_MODULE, "run", lambda *args, **kwargs: DEPLOY_MODULE.FakeProc(0))

        DEPLOY_MODULE.deploy_participant(participant, tmp_path)

        assert any("cp" in call for call in kubectl_calls)
        assert helm_calls
        kubeconfig, helm_args = helm_calls[0]
        assert kubeconfig == "/tmp/kubeconfig"
        assert helm_args[:4] == ("upgrade", "--install", "site-1", str(chart_dir))
        assert "--set-string" in helm_args
        assert r"podAnnotations.karpenter\.sh\/do-not-disrupt=value\,with\=helm\\chars" in helm_args


class TestAzureIpValidation:
    def test_reserve_ip_azure_requires_resource_group_and_location(self, monkeypatch):
        called = False

        def _run(*args, **kwargs):
            nonlocal called
            called = True
            return DEPLOY_MODULE.FakeProc(0)

        monkeypatch.setattr(DEPLOY_MODULE, "run", _run)

        with pytest.raises(ValueError, match="resource_group"):
            DEPLOY_MODULE._reserve_ip_azure(None, "westus2")
        with pytest.raises(ValueError, match="location"):
            DEPLOY_MODULE._reserve_ip_azure("my-rg", "")

        assert called is False

    def test_release_ip_azure_requires_resource_group(self, monkeypatch):
        called = False

        def _run(*args, **kwargs):
            nonlocal called
            called = True
            return DEPLOY_MODULE.FakeProc(0)

        monkeypatch.setattr(DEPLOY_MODULE, "run", _run)

        with pytest.raises(ValueError, match="resource_group"):
            DEPLOY_MODULE._release_ip_azure("pip-name", None)

        assert called is False

    def test_release_ip_azure_warns_on_delete_failure(self, monkeypatch, capsys):
        monkeypatch.setattr(
            DEPLOY_MODULE,
            "run",
            lambda *args, **kwargs: DEPLOY_MODULE.FakeProc(1, stderr="public IP is still in use"),
        )

        DEPLOY_MODULE._release_ip_azure("pip-name", "my-rg")

        out = capsys.readouterr().out
        assert "Warning: failed to delete Azure Public IP pip-name" in out
        assert "still in use" in out
