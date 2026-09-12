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

import pytest

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.workspace import Workspace
from nvflare.app_common.widgets.metrics_artifact_writer import MetricsArtifactWriter
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.fuel.utils.config_service import ConfigService
from nvflare.job_config.api import FedJob
from nvflare.job_config.base_fed_job import BaseFedJob
from nvflare.private.fed.server.server_json_config import ServerJsonConfigurator


@pytest.mark.parametrize("writer_kind", ["replacement", "configured", "other_id", "missing"])
def test_real_export_preserves_explicit_metrics_writer(tmp_path, monkeypatch, writer_kind):
    for name in ("_sections", "_config_path", "_cmd_args", "_var_dict", "_var_values"):
        monkeypatch.setattr(ConfigService, name, copy.deepcopy(getattr(ConfigService, name)))
    if writer_kind == "replacement":
        # BaseFedJob permits any FLComponent, including an independent/no-op replacement.
        job = BaseFedJob(name="reporting", metrics_artifact_writer=FLComponent())
    elif writer_kind == "configured":
        job = BaseFedJob(name="reporting", metrics_artifact_writer=MetricsArtifactWriter(results_dir="custom-metrics"))
    else:
        job = FedJob(name="reporting")
        if writer_kind == "other_id":
            job.to_server(MetricsArtifactWriter(results_dir="custom-metrics"), id="my_writer")
    job.to_server(FedAvg(num_clients=1, num_rounds=1))
    job.export_job(str(tmp_path / "export"))
    server_json = next((tmp_path / "export").rglob("config_fed_server.json"))

    for folder in ("startup", "local"):
        (tmp_path / folder).mkdir()
    workspace = Workspace(str(tmp_path), site_name="server")
    config = ServerJsonConfigurator(
        workspace_obj=workspace,
        config_file_name=str(server_json),
        args=argparse.Namespace(job_id="test-job", workspace=str(tmp_path)),
        app_root=str(server_json.parent.parent),
    )
    config.configure()
    writers = [h for h in config.runner_config.handlers if isinstance(h, MetricsArtifactWriter)]
    if writer_kind == "replacement":
        assert type(config.components["metrics_artifact_writer"]) is FLComponent
        assert config.components["metrics_artifact_writer"] in config.runner_config.handlers
        assert writers == []
    else:
        assert len(writers) == 1
        assert writers[0].results_dir == ("metrics" if writer_kind == "missing" else "custom-metrics")
        if writer_kind != "missing":
            component_id = "my_writer" if writer_kind == "other_id" else "metrics_artifact_writer"
            assert writers[0] is config.components[component_id]
