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


def _validate_args(tmp_path, log_message, num_rounds=1, job_type="numpy"):
    download_json = tmp_path / "download.json"
    logs_json = tmp_path / "logs.json"
    download_json.write_text(json.dumps({"data": {"artifacts": {"global_model": "server.npy"}}}))
    logs_json.write_text(json.dumps({"data": {"logs": {"server": log_message}}}))
    return SimpleNamespace(
        download_json=str(download_json),
        logs_json=str(logs_json),
        k8s_logs="",
        num_rounds=num_rounds,
        job_type=job_type,
    )


def test_create_job_numpy_exports_recipe_job_with_launcher_spec(tmp_path):
    verify = _load_verify_module()
    participants_tsv = tmp_path / "participants.tsv"
    participants_tsv.write_text(
        "server\tserver\tnvflare\t/tmp/kc/gcp.yaml\tgcp\texample.com/flare:1\n"
        "client\tsite-1\tnvflare\t/tmp/kc/aws.yaml\taws\texample.com/flare:2\n"
    )
    selected = tmp_path / "selected_clients.txt"
    selected.write_text("site-1\n")
    job_dir = tmp_path / "job"
    args = SimpleNamespace(
        job_dir=str(job_dir),
        job_name="numpy_e2e_unit",
        job_type="numpy",
        job_image="example.com/flare:1",
        python_path="/usr/local/bin/python3",
        num_rounds=2,
        num_clients=1,
        data_root="/data/default/data",
        allow_download=False,
        max_train_samples=128,
        max_val_samples=128,
        batch_size=32,
        epochs=1,
        torch_threads=1,
        selected_clients=str(selected),
        participants_tsv=str(participants_tsv),
        templates_dir=os.path.join(_repo_root(), "examples", "devops", "multicloud", "e2e", "jobs"),
    )

    verify.create_job(args)

    meta = json.loads((job_dir / "meta.json").read_text())
    assert meta["name"] == "numpy_e2e_unit"
    assert meta["min_clients"] == 1
    assert meta["launcher_spec"]["default"]["k8s"]["image"] == "example.com/flare:1"
    assert meta["launcher_spec"]["site-1"]["k8s"]["image"] == "example.com/flare:2"
    for targets in meta["deploy_map"].values():
        assert "@ALL" not in targets
        assert targets == ["server", "site-1"]

    client_configs = list(job_dir.rglob("config_fed_client.json"))
    assert len(client_configs) == 1
    executor = json.loads(client_configs[0].read_text())["executors"][0]["executor"]
    assert executor["args"]["task_script_path"] == "e2e_numpy_client.py"
    assert list(job_dir.rglob("e2e_numpy_client.py"))


def test_json_get_returns_empty_for_missing_intermediate_key(monkeypatch, capsys):
    verify = _load_verify_module()
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"status": "error"}'))

    verify.json_get(SimpleNamespace(expr="data.job_id"))

    assert capsys.readouterr().out == "\n"


def test_validate_result_accepts_structured_cli_log_values(tmp_path):
    verify = _load_verify_module()
    verify.validate_result(_validate_args(tmp_path, {"message": "E2E_ROUND current_round=0"}))


def test_validate_result_round_marker_is_not_satisfied_by_higher_round(tmp_path):
    verify = _load_verify_module()
    args = _validate_args(
        tmp_path,
        "E2E_ROUND current_round=0 site=x\nE2E_ROUND current_round=10 site=x",
        num_rounds=2,
    )

    with pytest.raises(SystemExit, match="missing round log marker: E2E_ROUND current_round=1"):
        verify.validate_result(args)


def test_validate_result_warns_but_passes_on_error_level_lines(tmp_path, capsys):
    verify = _load_verify_module()
    args = _validate_args(
        tmp_path,
        "2026-06-01 12:00:00 ERROR - connection closed by peer\nE2E_ROUND current_round=0 site=x",
    )

    verify.validate_result(args)

    assert "WARNING" in capsys.readouterr().err


def test_validate_result_fails_on_traceback(tmp_path):
    verify = _load_verify_module()
    args = _validate_args(
        tmp_path,
        "Traceback (most recent call last)\nE2E_ROUND current_round=0 site=x",
    )

    with pytest.raises(SystemExit, match="error marker"):
        verify.validate_result(args)


def test_validate_result_fails_on_execution_exception(tmp_path):
    verify = _load_verify_module()
    args = _validate_args(
        tmp_path,
        "submit result: EXECUTION_EXCEPTION\nE2E_ROUND current_round=0 site=x",
    )

    with pytest.raises(SystemExit, match="error marker"):
        verify.validate_result(args)


def test_compare_restarts_flags_disappeared_pod(tmp_path, capsys):
    verify = _load_verify_module()
    before = tmp_path / "before.tsv"
    after = tmp_path / "after.tsv"
    before.write_text("ns\tpod-a\t0\tsite-1\t1\n")
    after.write_text("ns\tpod-b\t0\tsite-1\t1\n")

    with pytest.raises(SystemExit):
        verify.compare_restarts(SimpleNamespace(before=str(before), after=str(after)))

    assert "disappeared" in capsys.readouterr().err


def test_compare_restarts_ignores_terminating_pod_that_disappears(tmp_path):
    verify = _load_verify_module()
    before = tmp_path / "before.tsv"
    after = tmp_path / "after.tsv"
    # A pod already Terminating/Completed at snapshot time (expect_present=0)
    # is allowed to disappear during the run.
    before.write_text("ns\told-job-pod\t0\tsite-1\t0\n")
    after.write_text("")

    verify.compare_restarts(SimpleNamespace(before=str(before), after=str(after)))


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
