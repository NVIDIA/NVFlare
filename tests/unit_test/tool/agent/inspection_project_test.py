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

from nvflare.tool.agent.inspector import inspect_path


def test_inspect_classifies_flare_job_source(tmp_path):
    job_py = tmp_path / "job.py"
    job_py.write_text(
        "from nvflare.recipe import SimEnv\n"
        "\n"
        "def main():\n"
        "    env = SimEnv(num_clients=2)\n"
        "    recipe.export('/tmp/job')\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    assert data["target_type"] == "flare_job_source"
    assert data["conversion_state"] == "flare_job"
    assert data["job"]["job_py"] == "job.py"
    assert data["job"]["sim_env_used"] is True
    assert data["job"]["export_support"] is True
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-autofl"]
    assert data["recommended_next_commands"] == [
        "Use the nvflare-autofl skill before editing.",
        "python job.py --export --export-dir <job-dir>",
    ]


def test_inspect_flare_job_source_recommends_autofl_not_conversion(tmp_path):
    # An existing FLARE job source routes optimization requests to the Auto-FL
    # skill; the conversion skill must not be recommended for an already
    # converted job even though the framework is detected.
    (tmp_path / "job.py").write_text(
        "import torch\n"
        "from nvflare.recipe import SimEnv\n"
        "\n"
        "class Net(torch.nn.Module):\n"
        "    pass\n"
        "\n"
        "def main():\n"
        "    env = SimEnv(num_clients=2)\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    assert data["conversion_state"] == "flare_job"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-autofl"]
    assert "nvflare-convert-pytorch" not in data["skill_selection"]["recommended_skills"]
    assert data["recommended_next_commands"] == ["Use the nvflare-autofl skill before editing."]


def test_nested_flare_job_source_does_not_override_root_pytorch_project(tmp_path):
    (tmp_path / "model.py").write_text(
        "import torch\n\n\nclass Net(torch.nn.Module):\n    pass\n",
        encoding="utf-8",
    )
    (tmp_path / "train.py").write_text(
        "from model import Net\n\n\ndef train():\n    return Net()\n",
        encoding="utf-8",
    )
    fixture = tmp_path / "tests" / "fixture"
    fixture.mkdir(parents=True)
    (fixture / "job.py").write_text(
        "from nvflare.app_common.workflows.fedavg import FedAvg\n"
        "from nvflare.job_config.api import FedJob\n"
        "\n"
        "job = FedJob(name='historical_fixture')\n"
        "controller = FedAvg()\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    assert data["frameworks"][0]["name"] == "pytorch"
    assert data["conversion_state"] == "not_converted"
    assert data["target_type"] == "training_repository"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-pytorch"]
    assert data["job"]["job_py"] == "tests/fixture/job.py"


def test_deeper_flare_job_does_not_override_nested_pytorch_component(tmp_path):
    app = tmp_path / "app"
    app.mkdir()
    (app / "model.py").write_text(
        "import torch\n\n\nclass Net(torch.nn.Module):\n    pass\n",
        encoding="utf-8",
    )
    (app / "train.py").write_text(
        "from model import Net\n\n\ndef train():\n    return Net()\n",
        encoding="utf-8",
    )
    fixture = tmp_path / "tests" / "fixture"
    fixture.mkdir(parents=True)
    (fixture / "job.py").write_text(
        "from nvflare.job_config.api import FedJob\n\njob = FedJob(name='historical_fixture')\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    assert data["frameworks"][0]["name"] == "pytorch"
    assert data["conversion_state"] == "not_converted"
    assert data["target_type"] == "training_repository"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-pytorch"]


def test_nested_converted_entry_point_importing_root_model_is_authoritative(tmp_path):
    (tmp_path / "model.py").write_text(
        "import torch\n\n\nclass Net(torch.nn.Module):\n    pass\n",
        encoding="utf-8",
    )
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    (jobs / "__init__.py").write_text("", encoding="utf-8")
    (jobs / "train.py").write_text(
        "from model import Net\n"
        "import nvflare.client as flare\n"
        "\n"
        "\n"
        "def train():\n"
        "    input_model = flare.receive()\n"
        "    flare.send(input_model)\n"
        "    return Net()\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    assert data["frameworks"][0]["name"] == "pytorch"
    assert data["conversion_state"] == "client_api_converted"
    assert data["target_type"] == "mixed_workspace"
    assert "nvflare-convert-pytorch" not in data["skill_selection"]["recommended_skills"]


def test_same_depth_independent_components_do_not_classify_whole_workspace_as_flare_job(tmp_path):
    (tmp_path / "common.py").write_text("VALUE = 1\n", encoding="utf-8")
    app = tmp_path / "app"
    app.mkdir()
    (app / "train.py").write_text(
        "import torch\n"
        "from common import VALUE\n"
        "\n"
        "\n"
        "def train():\n"
        "    return torch.nn.Linear(VALUE, 1)\n",
        encoding="utf-8",
    )
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "job.py").write_text(
        "from common import VALUE\n"
        "from nvflare.job_config.api import FedJob\n"
        "\n"
        "job = FedJob(name=f'archived_job_{VALUE}')\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    assert data["frameworks"][0]["name"] == "pytorch"
    assert data["conversion_state"] == "ambiguous"
    assert data["target_type"] == "mixed_workspace"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-orient"]
    assert "nvflare-convert-pytorch" not in data["skill_selection"]["recommended_skills"]
    assert data["recommended_next_commands"] == ["Use the nvflare-orient skill before editing."]


def test_equal_depth_fixture_entry_point_does_not_override_framework_project(tmp_path):
    model_dir = tmp_path / "src" / "pkg"
    model_dir.mkdir(parents=True)
    (model_dir / "model.py").write_text(
        "import torch\n\n\nclass Net(torch.nn.Module):\n    pass\n",
        encoding="utf-8",
    )
    fixture_dir = tmp_path / "tests" / "fixture"
    fixture_dir.mkdir(parents=True)
    (fixture_dir / "job.py").write_text(
        "from nvflare.job_config.api import FedJob\n\njob = FedJob(name='historical_fixture')\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    assert data["frameworks"][0]["name"] == "pytorch"
    assert data["conversion_state"] == "ambiguous"
    assert data["target_type"] == "mixed_workspace"
    assert data["job"]["job_py"] == "tests/fixture/job.py"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-orient"]
    assert "nvflare-autofl" not in data["skill_selection"]["recommended_skills"]
    assert "nvflare-convert-pytorch" not in data["skill_selection"]["recommended_skills"]


def test_connected_fixture_job_does_not_gain_authority_by_importing_production_model(tmp_path):
    model_dir = tmp_path / "src" / "pkg"
    model_dir.mkdir(parents=True)
    (model_dir / "model.py").write_text(
        "import torch\n\n\nclass Net(torch.nn.Module):\n    pass\n",
        encoding="utf-8",
    )
    fixture_dir = tmp_path / "tests" / "fixture"
    fixture_dir.mkdir(parents=True)
    (fixture_dir / "job.py").write_text(
        "from pkg.model import Net\n"
        "from nvflare.job_config.api import FedJob\n"
        "\n"
        "job = FedJob(name='historical_fixture')\n"
        "model = Net()\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    assert data["frameworks"][0]["name"] == "pytorch"
    assert data["conversion_state"] == "not_converted"
    assert data["target_type"] == "training_repository"
    assert data["job"]["job_py"] == "tests/fixture/job.py"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-pytorch"]
    assert "nvflare-autofl" not in data["skill_selection"]["recommended_skills"]

    fixture_data = inspect_path(fixture_dir)

    assert fixture_data["conversion_state"] == "flare_job"
    assert fixture_data["target_type"] == "flare_job_source"
    assert fixture_data["skill_selection"]["recommended_skills"] == ["nvflare-autofl"]


def test_same_depth_imported_components_share_source_job_authority(tmp_path):
    app = tmp_path / "app"
    app.mkdir()
    (app / "main.py").write_text(
        "from jobs import job\n\n\ndef main():\n    return job\n",
        encoding="utf-8",
    )
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    (jobs / "__init__.py").write_text("", encoding="utf-8")
    (jobs / "job.py").write_text(
        "from nvflare.job_config.api import FedJob\n\njob = FedJob(name='active_job')\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    assert data["conversion_state"] == "flare_job"
    assert data["target_type"] == "flare_job_source"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-autofl"]


def test_root_launcher_import_makes_nested_flare_job_component_authoritative(tmp_path):
    (tmp_path / "main.py").write_text(
        "from jobs.fedavg import job\n\n\ndef main():\n    return job\n",
        encoding="utf-8",
    )
    job_dir = tmp_path / "jobs" / "fedavg"
    job_dir.mkdir(parents=True)
    (job_dir / "__init__.py").write_text("", encoding="utf-8")
    (job_dir / "job.py").write_text(
        "from nvflare.job_config.api import FedJob\n\njob = FedJob(name='active_nested_job')\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    assert data["conversion_state"] == "flare_job"
    assert data["target_type"] == "flare_job_source"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-autofl"]


def test_root_flare_job_source_remains_authoritative_with_nested_job_candidate(tmp_path):
    (tmp_path / "job.py").write_text(
        "from nvflare.app_common.workflows.fedavg import FedAvg\n"
        "from nvflare.job_config.api import FedJob\n"
        "\n"
        "job = FedJob(name='active_job')\n"
        "controller = FedAvg()\n",
        encoding="utf-8",
    )
    fixture = tmp_path / "tests" / "fixture"
    fixture.mkdir(parents=True)
    (fixture / "job.py").write_text(
        "from nvflare.job_config.api import FedJob\n\njob = FedJob(name='historical_fixture')\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    assert data["conversion_state"] == "flare_job"
    assert data["target_type"] == "flare_job_source"
    assert data["job"]["job_py"] == "job.py"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-autofl"]


def test_root_flare_job_source_can_delegate_nvflare_import_to_local_helper(tmp_path):
    (tmp_path / "job.py").write_text(
        "from job_utils import build_job\n\njob = build_job()\n",
        encoding="utf-8",
    )
    (tmp_path / "job_utils.py").write_text(
        "from nvflare.job_config.api import FedJob\n\n" "def build_job():\n" "    return FedJob(name='active_job')\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    assert data["conversion_state"] == "flare_job"
    assert data["target_type"] == "flare_job_source"
    assert data["job"]["job_py"] == "job.py"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-autofl"]


def test_root_job_source_does_not_use_unreachable_nvflare_import(tmp_path):
    (tmp_path / "job.py").write_text(
        "def build_job():\n" "    return object()\n",
        encoding="utf-8",
    )
    (tmp_path / "unused_helper.py").write_text(
        "from nvflare.job_config.api import FedJob\n\n"
        "def build_job():\n"
        "    return FedJob(name='unrelated_job')\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    assert data["conversion_state"] != "flare_job"
    assert data["target_type"] != "flare_job_source"
    assert data["skill_selection"]["recommended_skills"] != ["nvflare-autofl"]


def test_nested_flare_job_source_is_authoritative_without_competing_root_project(tmp_path):
    job_dir = tmp_path / "jobs" / "fedavg"
    job_dir.mkdir(parents=True)
    (job_dir / "job.py").write_text(
        "from nvflare.app_common.workflows.fedavg import FedAvg\n"
        "from nvflare.job_config.api import FedJob\n"
        "\n"
        "job = FedJob(name='nested_active_job')\n"
        "controller = FedAvg()\n"
        "job.export('/tmp/job')\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    assert data["conversion_state"] == "flare_job"
    assert data["target_type"] == "flare_job_source"
    assert data["job"]["job_py"] == "jobs/fedavg/job.py"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-autofl"]
    assert data["recommended_next_commands"] == [
        "Use the nvflare-autofl skill before editing.",
        "python jobs/fedavg/job.py --export --export-dir <job-dir>",
    ]


def test_src_layout_converted_project_is_not_hidden_by_root_packaging_scaffold(tmp_path):
    (tmp_path / "setup.py").write_text("from setuptools import setup\n\nsetup()\n", encoding="utf-8")
    train_py = tmp_path / "src" / "mypkg" / "train.py"
    train_py.parent.mkdir(parents=True)
    train_py.write_text(
        "import torch\n"
        "import nvflare.client as flare\n"
        "\n"
        "def train():\n"
        "    model = flare.receive()\n"
        "    flare.send(model)\n"
        "    return torch.nn.Linear(1, 1)\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    assert data["frameworks"][0]["name"] == "pytorch"
    assert data["conversion_state"] == "client_api_converted"
    assert data["target_type"] == "mixed_workspace"
    assert "nvflare-convert-pytorch" not in data["skill_selection"]["recommended_skills"]


def test_inspect_classifies_authoritative_flmodel_call_as_client_api_converted(tmp_path):
    (tmp_path / "client.py").write_text(
        "from nvflare.app_common.abstract.fl_model import FLModel\n" "\n" "def main():\n" "    return FLModel()\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    assert data["conversion_state"] == "client_api_converted"
    assert data["flare_integration"]["calls"] == ["FLModel"]


def test_nested_flare_evidence_uses_whole_directory_fallback_without_project_anchors(tmp_path):
    helper = tmp_path / "src" / "mypkg" / "helper.py"
    helper.parent.mkdir(parents=True)
    helper.write_text(
        "import nvflare.client as flare\n\nmodel = flare.receive()\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    assert data["conversion_state"] == "partial_client_api"
    assert data["flare_integration"]["calls"] == ["flare.receive"]


def test_inspect_does_not_treat_pytorch_to_call_as_export_support(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "import torch\n" "\n" "def train(tensor):\n" "    return tensor.to('cpu')\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["job"]["export_support"] is False
    assert "python job.py --export --export-dir <job-dir>" not in data["recommended_next_commands"]
