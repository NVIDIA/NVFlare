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
import json
from types import SimpleNamespace

import pytest

from nvflare.private.fed.client.client_json_config import ClientJsonConfigurator
from nvflare.private.fed.server.server_json_config import ServerJsonConfigurator
from nvflare.private.json_configer import ConfigError

FILTERS = [
    {
        "id": "percentile_privacy",
        "path": "nvflare.app_common.filters.percentile_privacy.PercentilePrivacy",
        "args": {"percentile": 95, "gamma": 0.01},
    }
]
CHAIN = {"tasks": ["train"], "filters": FILTERS}
# One real component per side: the shape check runs during the scan, but `configure()` also
# refuses a config with no executor or no workflow, so both have to be present to reach it.
EXECUTORS = [
    {
        "tasks": ["train"],
        "executor": {
            "path": "nvflare.app_common.executors.client_api_executor.ClientAPIExecutor",
            "args": {"execution_mode": "in_process", "task_script_path": "train.py"},
        },
    }
]
WORKFLOWS = [
    {
        "id": "fedavg",
        "path": "nvflare.app_common.workflows.fedavg.FedAvg",
        "args": {"num_clients": 1, "num_rounds": 1},
    }
]


def _workspace(tmp_path):
    return SimpleNamespace(
        get_app_custom_dir=lambda job_id: str(tmp_path / job_id / "custom"),
        get_app_config_dir=lambda job_id: str(tmp_path / job_id / "config"),
    )


def _configure(tmp_path, is_server, key, value):
    config = {"format_version": 2, key: value}
    if is_server:
        config["workflows"] = WORKFLOWS
        name = "config_fed_server.json"
        args = argparse.Namespace(job_id="job-1", workspace=str(tmp_path))
        cls = ServerJsonConfigurator
    else:
        config["executors"] = EXECUTORS
        name = "config_fed_client.json"
        args = argparse.Namespace(
            sp_scheme="grpc",
            sp_target="localhost:8002",
            client_name="site-1",
            parent_url=None,
            job_id="job-1",
            workspace=str(tmp_path),
        )
        cls = ClientJsonConfigurator

    config_file = tmp_path / name
    config_file.write_text(json.dumps(config))
    configurator = cls(
        workspace_obj=_workspace(tmp_path),
        config_file_name=str(config_file),
        args=args,
        app_root=str(tmp_path),
    )
    configurator.configure()
    return configurator


@pytest.mark.parametrize("is_server", [False, True], ids=["client", "server"])
@pytest.mark.parametrize("key", ["task_data_filters", "task_result_filters"])
@pytest.mark.parametrize(
    "value, got",
    [
        (CHAIN, "dict"),
        ({"chain-1": CHAIN}, "dict"),
        ("train", "str"),
    ],
    ids=["one-chain-as-dict", "chains-keyed-by-name", "string"],
)
class TestFilterChainShape:
    def test_non_list_filter_chains_are_refused(self, tmp_path, is_server, key, value, got):
        # A chain is recognised at `task_*_filters.#<n>`, which the scanner only produces for a
        # list. Before this check any other shape matched nothing, so the chain was never built
        # and the job ran with the filters the site asked for silently absent.
        with pytest.raises(ConfigError, match='"%s" must be a list of filter chains but got %s' % (key, got)):
            _configure(tmp_path, is_server, key, value)


@pytest.mark.parametrize("is_server", [False, True], ids=["client", "server"])
@pytest.mark.parametrize("key", ["task_data_filters", "task_result_filters"])
def test_list_of_chains_is_accepted_and_registered(tmp_path, is_server, key):
    configurator = _configure(tmp_path, is_server, key, [CHAIN])

    chains = (
        configurator.task_data_filter_chains if key == "task_data_filters" else configurator.task_result_filter_chains
    )
    assert len(chains) == 1
    assert len(chains[0].filters) == 1


@pytest.mark.parametrize("is_server", [False, True], ids=["client", "server"])
def test_absent_filters_stay_absent(tmp_path, is_server):
    configurator = _configure(tmp_path, is_server, "components", [])

    assert configurator.task_data_filter_chains == []
    assert configurator.task_result_filter_chains == []
