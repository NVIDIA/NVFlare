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
import os
import tempfile

import pytest

from nvflare.apis.job_def import ALL_SITES
from nvflare.app_common.workflows.cmd_task_controller import CmdTaskController
from nvflare.client.config import ExchangeFormat
from nvflare.fuel.utils.constants import FrameworkType
from nvflare.fuel.utils.secret_utils import PotentialSecretWarning, UnsupportedSecretRefWarning
from nvflare.recipe import FedTaskRecipe, secret_ref


@pytest.fixture
def temp_task_script():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("# Test task script\nimport nvflare.client as flare\n")
        script_path = f.name
    yield script_path
    os.unlink(script_path)


class TestFedTaskRecipe:
    def test_warns_on_secret_in_task_args(self, temp_task_script):
        secret = "ghp_" + "Ab1" * 12

        recipe = FedTaskRecipe(
            name="secret_task",
            task_name="embed",
            min_clients=1,
            task_script=temp_task_script,
            task_args=f"--api-key {secret}",
        )

        with pytest.warns(PotentialSecretWarning) as record:
            recipe._warn_potential_secrets_in_params()

        messages = [str(warning.message) for warning in record]
        assert any("task_args" in message for message in messages)
        assert all(secret not in message for message in messages)

    def test_warns_on_secret_in_task_payload(self, temp_task_script):
        recipe = FedTaskRecipe(
            name="secret_payload",
            task_name="embed",
            min_clients=1,
            task_script=temp_task_script,
            task_data={"auth_token": "abcd1234efgh"},
        )

        with pytest.warns(PotentialSecretWarning, match="task_data"):
            recipe._warn_potential_secrets_in_params()

    def test_warns_when_task_payload_uses_unsupported_secret_ref(self, temp_task_script):
        recipe = FedTaskRecipe(
            name="secret_ref_payload",
            task_name="embed",
            min_clients=1,
            task_script=temp_task_script,
            task_data={"auth_token": secret_ref("API_TOKEN")},
        )

        with pytest.warns(UnsupportedSecretRefWarning, match="task_data"):
            recipe._warn_potential_secrets_in_params()

    def test_initializes_model_free_one_round_task(self, temp_task_script):
        recipe = FedTaskRecipe(
            name="embedding_job",
            task_name="embed",
            min_clients=2,
            num_clients=2,
            min_responses=1,
            timeout=10,
            task_data={"checkpoint": "/tmp/model"},
            task_meta={"stage": "embedding"},
            task_script=temp_task_script,
            task_args="--data-root /tmp/data",
        )

        assert recipe.name == "embedding_job"
        assert recipe.task_name == "embed"
        assert recipe.framework == FrameworkType.RAW
        assert recipe.server_expected_format == ExchangeFormat.RAW

        server_app = recipe._job._deploy_map["server"]
        controller = server_app.app_config.workflows[0].controller
        assert isinstance(controller, CmdTaskController)
        assert controller.task_name == "embed"
        assert controller.task_data == {"checkpoint": "/tmp/model"}
        assert controller.task_meta == {"stage": "embedding"}
        assert controller.num_clients == 2
        assert controller.min_responses == 1
        assert controller.timeout == 10

        client_app = recipe._job._deploy_map[ALL_SITES]
        executor_def = client_app.app_config.executors[0]
        assert executor_def.tasks == ["embed"]
        assert temp_task_script in client_app.app_config.ext_scripts

    def test_defaults_send_task_name_and_request_meta(self, temp_task_script):
        recipe = FedTaskRecipe(
            name="default_task",
            task_name="preprocess",
            min_clients=1,
            task_script=temp_task_script,
        )

        controller = recipe._job._deploy_map["server"].app_config.workflows[0].controller
        assert controller.task_data == {"task_name": "preprocess"}
        assert controller.task_meta == {"status": "request"}

    def test_exported_config_has_no_model_persistor(self, temp_task_script):
        recipe = FedTaskRecipe(
            name="config_task",
            task_name="infer",
            min_clients=2,
            num_clients=2,
            min_responses=1,
            timeout=30,
            task_data={"task": "infer"},
            task_meta={"stage": "embedding"},
            task_script=temp_task_script,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            recipe._job.export_job(tmpdir)
            job_dir = os.path.join(tmpdir, "config_task")

            with open(os.path.join(job_dir, "app", "config", "config_fed_server.json")) as f:
                server_config = json.load(f)
            with open(os.path.join(job_dir, "app", "config", "config_fed_client.json")) as f:
                client_config = json.load(f)

        assert server_config["components"] == []
        assert server_config["workflows"][0]["path"] == (
            "nvflare.app_common.workflows.cmd_task_controller.CmdTaskController"
        )
        assert server_config["workflows"][0]["args"]["task_name"] == "infer"
        assert server_config["workflows"][0]["args"]["task_data"] == {"task": "infer"}
        assert server_config["workflows"][0]["args"]["task_meta"] == {"stage": "embedding"}
        assert server_config["workflows"][0]["args"]["num_clients"] == 2
        assert server_config["workflows"][0]["args"]["min_responses"] == 1
        assert server_config["workflows"][0]["args"]["timeout"] == 30
        assert server_config["workflows"][0]["args"]["persistor_id"] == ""
        assert client_config["executors"][0]["tasks"] == ["infer"]
        executor = client_config["executors"][0]["executor"]
        assert executor["path"].endswith(".ClientAPIExecutor")
        assert executor["args"]["execution_mode"] == "in_process"
        # RAW is ClientAPIExecutor's default and may be omitted by job-config serialization.
        assert executor["args"].get("params_exchange_format", "raw") == "raw"
        assert executor["args"]["server_expected_format"] == "raw"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"min_clients": 0},
            {"min_clients": 1, "num_clients": 0},
            {"min_clients": 1, "min_responses": 0},
            {"min_clients": 1, "num_clients": 2, "min_responses": 3},
            {"min_clients": 1, "timeout": -1},
        ],
    )
    def test_client_and_wait_counts_must_be_valid(self, temp_task_script, kwargs):
        with pytest.raises(ValueError):
            FedTaskRecipe(
                name="bad_task",
                **kwargs,
                task_script=temp_task_script,
            )


class TestCmdTaskController:
    def test_run_defaults_match_existing_task_payload(self):
        controller = CmdTaskController(task_name="etl", persistor_id="")
        captured = {}

        controller.info = lambda _msg: None
        controller.sample_clients = lambda num_clients=None: ["site-1"]

        def capture_send_task_and_wait(task_name, targets, data, min_responses=None, timeout=0):
            captured["task_name"] = task_name
            captured["targets"] = targets
            captured["data"] = data
            captured["min_responses"] = min_responses
            captured["timeout"] = timeout
            return []

        controller.send_task_and_wait = capture_send_task_and_wait

        controller.run()

        assert captured["task_name"] == "etl"
        assert captured["targets"] == ["site-1"]
        assert captured["data"].params == {"task_name": "etl"}
        assert captured["data"].meta == {"status": "request"}
        assert captured["min_responses"] is None
        assert captured["timeout"] == 0

    def test_run_sends_one_round_task_model(self):
        controller = CmdTaskController(
            task_name="infer",
            task_data={"checkpoint": "/tmp/model"},
            task_meta={"stage": "embedding"},
            num_clients=2,
            min_responses=1,
            timeout=10,
            persistor_id="",
        )
        captured = {}

        controller.info = lambda _msg: None
        controller.sample_clients = lambda num_clients=None: ["site-1", "site-2"]

        def capture_send_task_and_wait(task_name, targets, data, min_responses=None, timeout=0):
            captured["task_name"] = task_name
            captured["targets"] = targets
            captured["data"] = data
            captured["min_responses"] = min_responses
            captured["timeout"] = timeout
            return []

        controller.send_task_and_wait = capture_send_task_and_wait

        controller.run()

        assert captured["task_name"] == "infer"
        assert captured["targets"] == ["site-1", "site-2"]
        assert captured["data"].params == {"checkpoint": "/tmp/model"}
        assert captured["data"].meta == {"stage": "embedding"}
        assert captured["data"].current_round == 0
        assert captured["data"].total_rounds == 1
        assert captured["min_responses"] == 1
        assert captured["timeout"] == 10

    def test_run_rejects_unreachable_min_responses(self):
        controller = CmdTaskController(
            task_name="infer",
            min_responses=3,
            timeout=0,
            persistor_id="",
        )
        captured = {}

        controller.info = lambda _msg: None
        controller.panic = lambda msg: captured.setdefault("panic", msg)
        controller.sample_clients = lambda num_clients=None: ["site-1", "site-2"]

        def fail_send_task_and_wait(*_args, **_kwargs):
            raise AssertionError("send_task_and_wait should not be called")

        controller.send_task_and_wait = fail_send_task_and_wait

        with pytest.raises(RuntimeError, match="min_responses=3 exceeds sampled clients=2"):
            controller.run()

        assert "min_responses=3 exceeds sampled clients=2" in captured["panic"]
