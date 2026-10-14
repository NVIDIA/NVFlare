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
import sys
from pathlib import Path

import yaml

from nvflare.app_common.default_component_policy import DEFAULT_CLASS_ALLOW_LIST
from nvflare.app_common.executors.client_api_executor import ExecutionMode
from nvflare.app_common.widgets.component_path_authorizer import ComponentPathAuthorizer

EXAMPLE = Path(__file__).parents[3] / "examples" / "advanced" / "cvm_builder"
CLIENT_API_EXECUTOR = "nvflare.app_common.executors.client_api_executor.ClientAPIExecutor"
SAFE_CVM_EXECUTOR = "cvm_client_api_executor.CVMClientAPIExecutor"


def _component_paths(value):
    if isinstance(value, dict):
        for name, child in value.items():
            if name in ("path", "class_path") and isinstance(child, str):
                yield child
            yield from _component_paths(child)
    elif isinstance(value, list):
        for child in value:
            yield from _component_paths(child)


def _is_allowed(path, allow_list):
    return any(ComponentPathAuthorizer._path_matches_prefix(path, entry) for entry in allow_list)


def test_bundled_job_components_are_allowed_without_command_capable_classes():
    job_root = EXAMPLE / "jobs" / "hello-pt_cifar10_fedavg"
    participants = (
        ("site-1", "app_site-1", "config_fed_client.json"),
        ("server", "app_server", "config_fed_server.json"),
    )
    for participant, app, filename in participants:
        cc_config = yaml.safe_load((EXAMPLE / f"cc_{participant}.yml").read_text())
        job = json.loads((job_root / app / "config" / filename).read_text())
        allow_list = (*DEFAULT_CLASS_ALLOW_LIST, *cc_config.get("class_allow_list", []))
        paths = set(_component_paths(job))
        assert paths
        assert all(_is_allowed(path, allow_list) for path in paths), (participant, paths)

    site_config = yaml.safe_load((EXAMPLE / "cc_site-1.yml").read_text())
    assert site_config["class_allow_list"] == [SAFE_CVM_EXECUTOR]
    assert not _is_allowed(CLIENT_API_EXECUTOR, (*DEFAULT_CLASS_ALLOW_LIST, *site_config["class_allow_list"]))


def test_measured_cvm_executor_fixes_mode_and_script_path():
    sys.path.insert(0, str(EXAMPLE / "docker/code"))
    try:
        from cvm_client_api_executor import CVMClientAPIExecutor

        executor = CVMClientAPIExecutor()
    finally:
        sys.path.pop(0)
        sys.modules.pop("cvm_client_api_executor", None)
    assert executor.execution_mode == ExecutionMode.IN_PROCESS
    assert executor._command is None
    assert executor._task_script_path == "/local/custom/client.py"


def test_example_image_installs_compatible_nvflare_release():
    dockerfile = (EXAMPLE / "docker/Dockerfile").read_text()
    assert "pip install nvflare~=2.9.0" in dockerfile
    assert "ENV PYTHONPATH=/local/custom" in dockerfile
