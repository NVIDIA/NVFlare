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
import sys
from pathlib import Path

import pytest

_MULTICLOUD_DIR = Path(__file__).resolve().parents[3] / "examples" / "devops" / "multicloud"

_CONFIG = """\
clouds:
  gcp:
    prepare:
      runtime: k8s
      parent:
        docker_image: registry.example.com/nvflare:test
participants:
  - name: server
    cloud: gcp
    namespace: nvflare
    role: server
{extra}"""


def _load_deploy(monkeypatch):
    monkeypatch.syspath_prepend(str(_MULTICLOUD_DIR))
    spec = importlib.util.spec_from_file_location("multicloud_deploy_config", _MULTICLOUD_DIR / "deploy.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "multicloud_deploy_config", module)
    spec.loader.exec_module(module)
    return module


def _write_config(tmp_path, extra=""):
    path = tmp_path / "all-clouds.yaml"
    path.write_text(_CONFIG.format(extra=extra), encoding="utf-8")
    return path


def test_load_config_rejects_removed_study_data_key(monkeypatch, tmp_path):
    deploy = _load_deploy(monkeypatch)
    config = _write_config(tmp_path, extra="    study_data: {study1: {datasets: {d: {source: pvc, mode: ro}}}}\n")
    with pytest.raises(ValueError, match="removed 'study_data' key.*'study_runtime'"):
        deploy.load_config(config)


def test_load_config_accepts_study_runtime_key(monkeypatch, tmp_path):
    deploy = _load_deploy(monkeypatch)
    config = _write_config(tmp_path, extra="    study_runtime: {study1: {datasets: {d: {source: pvc, mode: ro}}}}\n")
    loaded = deploy.load_config(config)
    assert loaded.participants[0].study_datasets == {"study1": {"d": {"source": "pvc", "mode": "ro"}}}
