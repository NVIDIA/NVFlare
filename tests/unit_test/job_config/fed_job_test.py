# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.app_common.abstract.model_learner import ModelLearner
from nvflare.app_common.executors.model_learner_executor import ModelLearnerExecutor
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.job_config.api import FedJob


class TestFedJob:
    def test_validate_targets(self):
        job = FedJob()
        controller = FedAvg()
        executor = ModelLearnerExecutor(learner_id=job.as_id(ModelLearner()))

        job.to(controller, "server")
        job.to(executor, "site-1")

        with pytest.raises(Exception):
            job.to(executor, "site-/1")

    def test_non_empty_target(self):
        job = FedJob()
        component = FedAvg()
        with pytest.raises(Exception):
            job.to(component, None)

    def test_add_params_functionality(self):
        """Test that add_params functionality works correctly."""
        job = FedJob(name="test_job")

        # Add a controller to server
        controller = FedAvg()
        job.to_server(controller)

        # Add an executor to clients
        executor = ModelLearnerExecutor(learner_id=job.as_id(ModelLearner()))
        job.to_clients(executor)

        # Add additional arguments to server
        server_params = {"timeout": 600, "max_retries": 3, "heartbeat_interval": 30}
        job.to_server(server_params)

        # Add additional arguments to clients
        client_params = {"submit_task_result_timeout": 300}
        job.to_clients(client_params)

        # Export the job to verify the params are included
        with tempfile.TemporaryDirectory() as temp_dir:
            job.export_job(temp_dir)

            # Check server config
            server_config_path = os.path.join(temp_dir, "test_job", "app", "config", "config_fed_server.json")
            assert os.path.exists(server_config_path)

            with open(server_config_path, "r") as f:
                server_config = json.load(f)

            # Verify server params are included
            for key, value in server_params.items():
                assert key in server_config
                assert server_config[key] == value

            # Check client config
            client_config_path = os.path.join(temp_dir, "test_job", "app", "config", "config_fed_client.json")
            assert os.path.exists(client_config_path)

            with open(client_config_path, "r") as f:
                client_config = json.load(f)

            # Verify client params are included
            for key, value in client_params.items():
                assert key in client_config
                assert client_config[key] == value
