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

"""Contract tests for the agent-facing ``--format json`` CLI outputs.

Packaged agent skills consume the JSON shapes of ``nvflare recipe list``,
``nvflare recipe show``, and the ``nvflare agent inspect source``/``data``
capabilities as product API contracts, not incidental CLI formatting. These golden-field tests fail if a
field the skills depend on is renamed or dropped, forcing an explicit,
versioned migration instead of a silent break in shipped skills.

If a change here is intentional, update the documented contract and the
consuming skill guidance in the same change.
"""

import json
from argparse import Namespace

import pytest

# Fields the packaged conversion skills read from each JSON output. Renaming or
# dropping any of these breaks a shipped skill's documented workflow.
# Every `recipe list` entry carries these; the skill recipe-selection guidance
# matches recipes on algorithm/aggregation/state_exchange/privacy, so a dropped
# or renamed field must fail CI. privacy_compatible is intentionally excluded: it
# is emitted only on recipes that declare it (e.g. HE), not on every entry.
RECIPE_LIST_ENTRY_CONTRACT = {"name", "framework", "algorithm", "aggregation", "state_exchange", "privacy"}
RECIPE_SHOW_DETAIL_CONTRACT = {"name", "framework", "parameters"}
RECIPE_SHOW_PARAMETER_CONTRACT = {"name", "type", "required", "default"}
AGENT_INSPECT_COMMON_CONTRACT = {
    "capability",
    "limits",
    "path",
    "routing",
    "scan",
    "schema_version",
    "static_only",
}
AGENT_INSPECT_SOURCE_CONTRACT = AGENT_INSPECT_COMMON_CONTRACT | {"integration", "ownership"}
AGENT_INSPECT_DATA_CONTRACT = AGENT_INSPECT_COMMON_CONTRACT | {"dataset"}

_SAMPLE_CATALOG = [
    {
        "name": "fedavg-pt",
        "framework": "pytorch",
        "description": "FedAvg",
        "algorithm": "fedavg",
        "aggregation": "weighted_average",
        "state_exchange": "full_model",
        "privacy": [],
    },
    {
        "name": "fedopt-pt",
        "framework": "pytorch",
        "description": "FedOpt",
        "algorithm": "fedopt",
        "aggregation": "server_optimizer",
        "state_exchange": "weight_diff",
        "privacy": [],
    },
]


def _json_mode(monkeypatch):
    from nvflare.tool import cli_output

    monkeypatch.setattr(cli_output, "_output_format", "json")


def test_recipe_list_json_contract(monkeypatch, capsys):
    from nvflare.tool.recipe.recipe_cli import cmd_recipe_list

    _json_mode(monkeypatch)
    monkeypatch.setattr(
        "nvflare.tool.recipe.recipe_cli._load_catalog",
        lambda framework=None: list(_SAMPLE_CATALOG),
    )

    cmd_recipe_list(Namespace(framework=None, filters=None))

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert isinstance(payload["data"], list) and payload["data"]
    for entry in payload["data"]:
        missing = RECIPE_LIST_ENTRY_CONTRACT - entry.keys()
        assert not missing, f"recipe list entry missing contract fields: {missing}"


def test_recipe_list_json_contract_real_catalog(monkeypatch, capsys):
    # Golden test against the real installed catalog: protects the actual
    # skill-facing surface, not just a mocked entry shape.
    pytest.importorskip("torch")
    from nvflare.tool.recipe.recipe_cli import cmd_recipe_list

    _json_mode(monkeypatch)

    cmd_recipe_list(Namespace(framework="pytorch", filters=None))

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert isinstance(payload["data"], list) and payload["data"], "no pytorch recipes in installed catalog"
    for entry in payload["data"]:
        missing = RECIPE_LIST_ENTRY_CONTRACT - entry.keys()
        assert not missing, f"real recipe list entry missing contract fields: {missing}"
    assert any(entry["name"] == "fedavg-pt" for entry in payload["data"])


def test_recipe_show_json_contract(monkeypatch, capsys):
    # Golden test against the real fedavg-pt recipe: the conversion skills read
    # this exact surface when auditing model/constructor args and choosing
    # enable_tensor_disk_offload / server_expected_format.
    pytest.importorskip("torch")
    from nvflare.tool.recipe.recipe_cli import cmd_recipe_show

    _json_mode(monkeypatch)

    # With torch installed, fedavg-pt must be showable. A SystemExit here means
    # the recipe was renamed or its registration broke — exactly the contract
    # break this golden test exists to catch, so let it fail rather than skip.
    cmd_recipe_show(Namespace(name="fedavg-pt"))

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    detail = payload["data"]
    missing = RECIPE_SHOW_DETAIL_CONTRACT - detail.keys()
    assert not missing, f"recipe show detail missing contract fields: {missing}"

    assert isinstance(detail["parameters"], list) and detail["parameters"]
    for param in detail["parameters"]:
        param_missing = RECIPE_SHOW_PARAMETER_CONTRACT - param.keys()
        assert not param_missing, f"recipe show parameter missing contract fields: {param_missing}"

    by_name = {p["name"]: p for p in detail["parameters"]}
    # Parameters the packaged conversion skills reference by name must remain
    # discoverable; renaming any of these breaks the shipped skill guidance.
    for skill_referenced in (
        "model",
        "min_clients",
        "num_rounds",
        "aggregator",
        "enable_tensor_disk_offload",
        "server_expected_format",
    ):
        assert skill_referenced in by_name, f"fedavg-pt no longer exposes '{skill_referenced}'"
    # required/default discovery is what "audit constructor args" depends on; the
    # field must be a usable boolean flag (its value per recipe may change).
    assert all(isinstance(p["required"], bool) for p in detail["parameters"])


def test_agent_inspect_json_contract(tmp_path):
    from nvflare.tool.agent.inspector import inspect_source

    (tmp_path / "train.py").write_text(
        "from torch.optim import Adam\noptimizer = Adam(params)\noptimizer.step()\n",
        encoding="utf-8",
    )

    data = inspect_source(tmp_path)

    top_missing = AGENT_INSPECT_SOURCE_CONTRACT - data.keys()
    assert not top_missing, f"agent inspect output missing contract fields: {top_missing}"
    assert set(data) == AGENT_INSPECT_SOURCE_CONTRACT
    assert data["schema_version"] == "3"
    assert data["capability"] == "source"
    assert data["static_only"] is True
    assert data["routing"] == {"recommended_skill": "nvflare-convert-pytorch", "reason": "clear_owner"}
    assert data["ownership"]["framework"] == "pytorch"


def test_agent_inspect_exported_job_recommends_no_skill(tmp_path):
    # Lifecycle skills are out of scope and not planned; exported jobs must not
    # route to a dropped/nonexistent skill name.
    from nvflare.tool.agent.inspector import inspect_source

    app_cfg = tmp_path / "app" / "config"
    app_cfg.mkdir(parents=True)
    (app_cfg / "config_fed_server.json").write_text("{}", encoding="utf-8")
    (app_cfg / "config_fed_client.json").write_text("{}", encoding="utf-8")
    (tmp_path / "meta.json").write_text("{}", encoding="utf-8")

    data = inspect_source(tmp_path)

    # Assert the classification unconditionally so the contract cannot pass
    # vacuously if exported-job detection drifts.
    assert data["routing"] == {"recommended_skill": None, "reason": "existing_job"}


def test_agent_inspect_data_json_contract(tmp_path):
    from nvflare.tool.agent.inspector import inspect_data

    (tmp_path / "data.csv").write_text("age,income\n1,2\n", encoding="utf-8")

    data = inspect_data(tmp_path)

    assert set(data) == AGENT_INSPECT_DATA_CONTRACT
    assert data["capability"] == "data"
    assert data["dataset"]["modality"] == "tabular"
    assert data["routing"] == {"recommended_skill": "nvflare-fed-stats", "reason": "dataset"}
