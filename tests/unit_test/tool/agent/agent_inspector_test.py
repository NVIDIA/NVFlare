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

from nvflare.tool.agent import inspector as inspector_module
from nvflare.tool.agent.inspection import project, routing
from nvflare.tool.agent.inspector import inspect_path


def test_inspect_static_only_does_not_execute_user_module(tmp_path):
    marker = tmp_path / "import_side_effect"
    script = tmp_path / "train.py"
    script.write_text(
        "import pathlib\n"
        "import torch\n"
        f"pathlib.Path({str(marker)!r}).write_text('executed')\n"
        "\n"
        "def train():\n"
        "    return None\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert not marker.exists()
    assert data["target_type"] == "single_training_script"
    assert data["conversion_state"] == "not_converted"
    assert data["frameworks"][0]["name"] == "pytorch"
    assert data["skill_selection"]["recommended_skills"] == []
    assert data["framework_ownership"] == {"state": "import_only", "owners": [], "candidates": []}


def test_inspect_builds_local_import_graph_once(monkeypatch, tmp_path):
    (tmp_path / "train.py").write_text("from model import Net\n", encoding="utf-8")
    (tmp_path / "model.py").write_text("import torch\n\nclass Net(torch.nn.Module):\n    pass\n", encoding="utf-8")
    original = inspector_module.build_local_import_graph
    calls = []

    def count_builds(facts):
        calls.append(facts)
        return original(facts)

    for module in (inspector_module, project, routing):
        monkeypatch.setattr(module, "build_local_import_graph", count_builds)

    inspect_path(tmp_path)

    assert len(calls) == 1


def test_inspect_does_not_classify_lone_export_marker_as_submit_ready(tmp_path):
    (tmp_path / "config_fed_server.json").write_text("{}", encoding="utf-8")

    data = inspect_path(tmp_path)

    assert data["conversion_state"] == "unknown"
    assert data["target_type"] == "unknown_target"
    assert data["recommended_next_commands"] == []
    assert data["job"]["exported_job_candidates"] == []
    assert data["job"]["nested_candidates"] == [
        {
            "path": ".",
            "markers": ["config_fed_server.json"],
            "reason": "incomplete_exported_job_marker_set",
        }
    ]


def test_inspect_does_not_let_nested_export_marker_hijack_training_repo(tmp_path):
    (tmp_path / "train.py").write_text("import torch\n", encoding="utf-8")
    (tmp_path / "model.py").write_text(
        "import torch\n\nclass Net(torch.nn.Module):\n    pass\n",
        encoding="utf-8",
    )
    marker = tmp_path / "tests" / "fixtures" / "config_fed_server.json"
    marker.parent.mkdir(parents=True)
    marker.write_text("{}", encoding="utf-8")

    data = inspect_path(tmp_path)

    assert data["conversion_state"] == "not_converted"
    assert data["target_type"] == "training_repository"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-pytorch"]
    assert data["recommended_next_commands"] == ["Use the nvflare-convert-pytorch skill before editing."]
    assert data["job"]["exported_job_candidates"] == []
    assert data["job"]["nested_candidates"] == [
        {
            "path": "tests/fixtures",
            "markers": ["config_fed_server.json"],
            "reason": "incomplete_exported_job_marker_set",
        }
    ]


def test_inspect_requires_export_markers_to_form_submit_ready_root(tmp_path):
    (tmp_path / "meta.json").write_text("{}", encoding="utf-8")
    app_config = tmp_path / "app" / "config"
    app_config.mkdir(parents=True)
    (app_config / "config_fed_server.json").write_text("{}", encoding="utf-8")

    data = inspect_path(tmp_path)

    assert data["conversion_state"] == "exported_job"
    assert data["target_type"] == "exported_submit_ready_flare_job"
    assert data["job"]["exported_job_candidates"] == ["."]
    assert data["job"]["nested_candidates"] == []
    assert data["skill_selection"]["recommended_skills"] == []
    assert data["recommended_next_commands"] == ["nvflare job submit <job-folder> --format json"]


def test_inspect_relative_path_does_not_create_false_app_layout(monkeypatch, tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "meta.json").write_text("{}", encoding="utf-8")
    config = project / "config"
    config.mkdir()
    (config / "config_fed_server.json").write_text("{}", encoding="utf-8")

    monkeypatch.chdir(project)
    data = inspect_path(".")

    assert data["path"] == str(project.resolve(strict=False))
    assert data["conversion_state"] == "unknown"
    assert data["target_type"] == "unknown_target"
    assert data["job"]["exported_job_candidates"] == []
    assert data["recommended_next_commands"] == []
    assert data["job"]["nested_candidates"] == [
        {
            "path": ".",
            "markers": ["meta.json"],
            "reason": "incomplete_exported_job_marker_set",
        },
        {
            "path": "config",
            "markers": ["config_fed_server.json"],
            "reason": "incomplete_exported_job_marker_set",
        },
    ]


def test_inspect_reports_valid_nested_exported_job_candidate(tmp_path):
    job = tmp_path / "job"
    job.mkdir()
    (job / "meta.json").write_text("{}", encoding="utf-8")
    (job / "config_fed_server.json").write_text("{}", encoding="utf-8")

    data = inspect_path(tmp_path)

    assert data["conversion_state"] == "unknown"
    assert data["target_type"] == "unknown_target"
    assert data["job"]["exported_job_candidates"] == []
    assert data["job"]["nested_candidates"] == [
        {
            "path": "job",
            "markers": ["config_fed_server.json", "meta.json"],
            "reason": "nested_exported_job_candidate",
        }
    ]


def test_inspect_suppresses_consumed_root_app_configs_but_keeps_unrelated_nested_candidates(tmp_path):
    (tmp_path / "meta.json").write_text("{}", encoding="utf-8")
    app_config = tmp_path / "app" / "config"
    app_config.mkdir(parents=True)
    (app_config / "config_fed_server.json").write_text("{}", encoding="utf-8")
    fixture_config = tmp_path / "tests" / "fixtures" / "config_fed_client.json"
    fixture_config.parent.mkdir(parents=True)
    fixture_config.write_text("{}", encoding="utf-8")

    data = inspect_path(tmp_path)

    assert data["conversion_state"] == "exported_job"
    assert data["target_type"] == "exported_submit_ready_flare_job"
    assert data["job"]["exported_job_candidates"] == ["."]
    assert data["job"]["nested_candidates"] == [
        {
            "path": "tests/fixtures",
            "markers": ["config_fed_client.json"],
            "reason": "incomplete_exported_job_marker_set",
        }
    ]


def test_inspect_does_not_classify_lone_root_meta_json_as_exported_job(tmp_path):
    (tmp_path / "meta.json").write_text("{}", encoding="utf-8")

    data = inspect_path(tmp_path)

    assert data["conversion_state"] == "unknown"
    assert data["target_type"] == "unknown_target"
    assert data["recommended_next_commands"] == []


def test_inspect_does_not_pair_root_meta_with_unrelated_nested_config(tmp_path):
    (tmp_path / "train.py").write_text("import torch\n", encoding="utf-8")
    (tmp_path / "meta.json").write_text("{}", encoding="utf-8")
    fixture_config = tmp_path / "tests" / "fixtures" / "config_fed_server.json"
    fixture_config.parent.mkdir(parents=True)
    fixture_config.write_text("{}", encoding="utf-8")

    data = inspect_path(tmp_path)

    assert data["conversion_state"] == "not_converted"
    assert data["target_type"] == "training_repository"
    assert data["job"]["exported_job_candidates"] == []
    assert data["skill_selection"]["recommended_skills"] == []
    assert data["recommended_next_commands"] == []
    assert data["job"]["nested_candidates"] == [
        {
            "path": ".",
            "markers": ["meta.json"],
            "reason": "incomplete_exported_job_marker_set",
        },
        {
            "path": "tests/fixtures",
            "markers": ["config_fed_server.json"],
            "reason": "incomplete_exported_job_marker_set",
        },
    ]
