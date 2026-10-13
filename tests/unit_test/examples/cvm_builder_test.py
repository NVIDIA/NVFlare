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

"""Regression tests for the unified CVM Builder example."""

import json
from pathlib import Path

import yaml

EXAMPLE = Path(__file__).parents[3] / "examples" / "advanced" / "cvm_builder"
CLIENT_API_EXECUTOR = "nvflare.app_common.executors.client_api_executor.ClientAPIExecutor"


def test_client_api_executor_is_allowed_by_client_cc_config():
    config = yaml.safe_load((EXAMPLE / "cc_site-1.yml").read_text())
    job = json.loads(
        (EXAMPLE / "jobs" / "hello-pt_cifar10_fedavg" / "app_site-1" / "config" / "config_fed_client.json").read_text()
    )
    executor_paths = {
        component["executor"]["path"]
        for component in job["executors"]
        if component.get("executor", {}).get("path") == CLIENT_API_EXECUTOR
    }

    assert executor_paths == {CLIENT_API_EXECUTOR}
    assert config["class_allow_list"] == [CLIENT_API_EXECUTOR]
