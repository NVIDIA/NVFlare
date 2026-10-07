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
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.client import Client
from nvflare.apis.controller_spec import ClientTask, Task
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal

APPLICATION = Path(__file__).resolve().parents[5] / "examples/devops/coco/acceptance/application"
NONCE = "0123456789abcdef0123456789abcdef"


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, APPLICATION / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


application = load_module("tdx_acceptance")
generator = load_module("generate_job")


def response(site, **changes):
    result = Shareable()
    result.update(client=site, nonce=NONCE, value=application.EXPECTED_VALUES.get(site, 0))
    result.update(changes)
    return result


def receive(controller, site, result):
    client_task = ClientTask(Client(site, "token-" + site), Task(name=application.TASK_NAME, data=Shareable()))
    client_task.result = result
    controller._receive(client_task, Mock())
    assert client_task.result is None


def context(tmp_path):
    return SimpleNamespace(
        get_job_id=lambda: "test-job",
        get_workspace=lambda: SimpleNamespace(get_run_dir=lambda job_id: str(tmp_path)),
    )


def test_complete_job_persists_nonce_bound_exact_results(tmp_path):
    controller = application.AcceptanceController(NONCE)
    controller.system_panic = Mock()

    def broadcast(**kwargs):
        assert kwargs["targets"] == ["site-1", "site-2"]
        assert kwargs["min_responses"] == 2
        assert kwargs["task"].timeout == 540
        assert kwargs["task"].data["nonce"] == NONCE
        receive(controller, "site-2", response("site-2"))
        receive(controller, "site-1", response("site-1"))

    controller.broadcast_and_wait = broadcast
    controller.control_flow(Signal(), context(tmp_path))
    artifact = json.loads((tmp_path / application.RESULT_FILE).read_text())
    assert artifact["status"] == "passed"
    assert artifact["nonce"] == NONCE
    assert artifact["job_id"] == "test-job"
    assert artifact["values"] == {"site-1": 3, "site-2": 7}
    assert artifact["aggregate"] == 10
    assert artifact["errors"] == []
    controller.system_panic.assert_not_called()


@pytest.mark.parametrize(
    "site,result",
    [
        ("site-1", response("site-1", client="site-2")),
        ("site-1", response("site-1", nonce="expired-nonce-000")),
        ("site-1", response("site-1", value=True)),
        ("site-1", response("site-1", value=4)),
        ("site-1", response("site-1", extra="unexpected")),
        ("site-observer", response("site-observer")),
        ("site-1", None),
    ],
)
def test_reject_invalid_authenticated_result(site, result):
    controller = application.AcceptanceController(NONCE)
    receive(controller, site, result)
    assert controller.errors
    assert not controller.values


def test_duplicate_is_failure_even_if_values_are_correct():
    controller = application.AcceptanceController(NONCE)
    receive(controller, "site-1", response("site-1"))
    receive(controller, "site-1", response("site-1"))
    assert controller.errors == ["duplicate response from site-1"]


def test_missing_client_panics_and_writes_failure(tmp_path):
    controller = application.AcceptanceController(NONCE)
    controller.broadcast_and_wait = lambda **kwargs: receive(controller, "site-1", response("site-1"))
    controller.system_panic = Mock()
    controller.control_flow(Signal(), context(tmp_path))
    artifact = json.loads((tmp_path / application.RESULT_FILE).read_text())
    assert artifact["status"] == "failed"
    assert artifact["aggregate"] == 3
    controller.system_panic.assert_called_once()


def test_abort_never_passes(tmp_path):
    controller = application.AcceptanceController(NONCE)
    controller.broadcast_and_wait = Mock()
    controller.system_panic = Mock()
    signal = Signal()
    signal.trigger(True)
    controller.control_flow(signal, context(tmp_path))
    assert "job aborted" in json.loads((tmp_path / application.RESULT_FILE).read_text())["errors"]
    controller.system_panic.assert_called_once()


@pytest.mark.parametrize("site,value", [("site-1", 3), ("site-2", 7)])
def test_executor_returns_distinct_expected_value(site, value):
    executor = application.AcceptanceExecutor(NONCE)
    fl_ctx = SimpleNamespace(get_identity_name=lambda: site)
    result = executor.execute(application.TASK_NAME, Shareable({"nonce": NONCE}), fl_ctx, Signal())
    assert result["client"] == site
    assert result["nonce"] == NONCE
    assert result["value"] == value


@pytest.mark.parametrize("site,nonce", [("site-observer", NONCE), ("site-1", "wrong")])
def test_executor_rejects_observer_and_wrong_nonce(site, nonce):
    executor = application.AcceptanceExecutor(NONCE)
    result = executor.execute(
        application.TASK_NAME, Shareable({"nonce": nonce}), SimpleNamespace(get_identity_name=lambda: site), Signal()
    )
    assert result.get_return_code() == ReturnCode.EXECUTION_RESULT_ERROR


def test_generator_excludes_observer_and_executable_payload(tmp_path):
    job = generator.generate_job(tmp_path / "job", NONCE)
    meta = json.loads((job / "meta.json").read_text())
    assert meta["mandatory_clients"] == ["site-1", "site-2"]
    assert meta["min_clients"] == 2
    assert meta["deploy_map"] == {"app": ["server", "site-1", "site-2"]}
    files = list(job.rglob("*"))
    assert all(path.suffix == ".json" for path in files if path.is_file())
    assert not any(path.name == "custom" for path in files)
    for name in ("server", "client"):
        document = json.loads((job / "app/config" / f"config_fed_{name}.json").read_text())
        component = document["workflows"][0] if name == "server" else document["executors"][0]["executor"]
        assert component["args"]["nonce"] == NONCE
    with pytest.raises(FileExistsError):
        generator.generate_job(job, NONCE)


@pytest.mark.parametrize("nonce", ["short", "a" * 129, "a" * 16 + "/", None])
def test_reject_invalid_nonce_before_writing(tmp_path, nonce):
    with pytest.raises(ValueError):
        generator.generate_job(tmp_path / "job", nonce)
    assert not (tmp_path / "job").exists()
