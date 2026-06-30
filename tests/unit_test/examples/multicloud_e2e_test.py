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
import io
import json
import os
import sys
from types import SimpleNamespace

import pytest


def _repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _load_module(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_verify_module():
    return _load_module(
        "multicloud_e2e_verify",
        os.path.join(_repo_root(), "examples", "devops", "multicloud", "e2e", "verify.py"),
    )


def test_cifar_workflow_uses_class_path():
    verify = _load_verify_module()
    args = SimpleNamespace(
        num_clients=2,
        num_rounds=2,
        data_root="/data/default/data",
        max_train_samples=128,
        max_val_samples=128,
        batch_size=32,
        epochs=1,
        torch_threads=1,
        allow_download=False,
    )

    server_config, _ = verify.create_cifar10_configs(args)

    workflow = server_config["workflows"][0]
    assert workflow["path"] == "nvflare.app_common.workflows.fedavg.FedAvg"
    assert "name" not in workflow


def test_json_get_returns_empty_for_missing_intermediate_key(monkeypatch, capsys):
    verify = _load_verify_module()
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"status": "error"}'))

    verify.json_get(SimpleNamespace(expr="data.job_id"))

    assert capsys.readouterr().out == "\n"


def test_validate_result_accepts_structured_cli_log_values(tmp_path):
    verify = _load_verify_module()
    download_json = tmp_path / "download.json"
    logs_json = tmp_path / "logs.json"
    download_json.write_text(json.dumps({"data": {"artifacts": {"global_model": "server.npy"}}}))
    logs_json.write_text(json.dumps({"data": {"logs": {"server": {"message": "E2E_ROUND current_round=0"}}}}))

    verify.validate_result(
        SimpleNamespace(
            download_json=str(download_json),
            logs_json=str(logs_json),
            k8s_logs="",
            num_rounds=1,
            job_type="numpy",
        )
    )


def test_cifar_evaluate_rejects_empty_loader():
    if any(importlib.util.find_spec(dep) is None for dep in ("torch", "torchvision")):
        pytest.skip("PyTorch example dependencies are not installed")
    job_dir = os.path.join(_repo_root(), "examples", "devops", "multicloud", "e2e", "jobs", "cifar10")
    client_dir = os.path.join(job_dir, "app_client", "custom")
    model_path = os.path.join(job_dir, "app_server", "custom", "e2e_net.py")
    original_model_module = sys.modules.pop("e2e_net", None)
    try:
        sys.modules["e2e_net"] = _load_module("e2e_net", model_path)
        client = _load_module("multicloud_e2e_cifar_client", os.path.join(client_dir, "e2e_cifar10_client.py"))
    except RuntimeError as e:
        if "torchvision" in str(e):
            pytest.skip(f"PyTorch example dependency is unavailable: {e}")
        raise
    finally:
        if original_model_module is not None:
            sys.modules["e2e_net"] = original_model_module
        else:
            sys.modules.pop("e2e_net", None)

    with pytest.raises(ValueError, match="evaluation data loader produced no samples"):
        client._evaluate(model=None, loader=[], criterion=None)
