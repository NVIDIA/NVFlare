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

import argparse
import copy

from nvflare.apis.workspace import Workspace
from nvflare.app_common.widgets.metrics_artifact_writer import MetricsArtifactWriter
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.fuel.utils.config_service import ConfigService
from nvflare.job_config.api import FedJob
from nvflare.private.fed.server.server_json_config import ServerJsonConfigurator


def test_plain_fedjob_server_config_does_not_inject_metrics_writer(tmp_path, monkeypatch):
    for name in ("_sections", "_config_path", "_cmd_args", "_var_dict", "_var_values"):
        monkeypatch.setattr(ConfigService, name, copy.deepcopy(getattr(ConfigService, name)))
    job = FedJob(name="plain-job")
    job.to_server(FedAvg(num_clients=1, num_rounds=1))
    job.export_job(str(tmp_path / "export"))
    server_json = next((tmp_path / "export").rglob("config_fed_server.json"))

    for folder in ("startup", "local"):
        (tmp_path / folder).mkdir()
    config = ServerJsonConfigurator(
        workspace_obj=Workspace(str(tmp_path), site_name="server"),
        config_file_name=str(server_json),
        args=argparse.Namespace(job_id="test-job", workspace=str(tmp_path)),
        app_root=str(server_json.parent.parent),
    )
    config.configure()

    assert "metrics_artifact_writer" not in config.components
    assert not any(isinstance(handler, MetricsArtifactWriter) for handler in config.runner_config.handlers)
