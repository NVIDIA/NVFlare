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
from pathlib import Path

import pytest

from nvflare.tool.agent import inspector as inspector_module
from nvflare.tool.agent.frameworks.lightning import LightningDetector
from nvflare.tool.agent.inspector import (
    InspectState,
    _entry_point_imports_file,
    _evidence_score,
    _FamilyResolver,
    _framework_evidence_tied_to_entry_context,
    _module_names_for_file,
    _resolve_import_from_module,
    inspect_path,
)


def _should_promote_lightning_over_pytorch(state):
    # The PyTorch-family promotion decision now lives in the Lightning detector;
    # exercise it through the same resolver the engine uses.
    return LightningDetector().promote_over_family("pytorch", _FamilyResolver(state))


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
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-pytorch"]


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
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-pytorch"]
    assert data["recommended_next_commands"] == ["Use the nvflare-convert-pytorch skill before editing."]
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


def test_inspect_detects_pytorch_lightning_and_recommends_lightning_skill(tmp_path):
    script = tmp_path / "train_lightning.py"
    script.write_text(
        "import torch\n" "import pytorch_lightning as pl\n" "\n" "class Net(pl.LightningModule):\n" "    pass\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["frameworks"][0]["name"] == "pytorch_lightning"
    assert data["conversion_state"] == "not_converted"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-lightning"]
    assert data["recommended_next_commands"] == [
        "Use the nvflare-convert-lightning skill before editing.",
    ]
    assert any(item["kind"] == "lightning_class" for item in data["frameworks"][0]["evidence"])


def test_inspect_detects_huggingface_trainer_and_recommends_huggingface_skill(tmp_path):
    script = tmp_path / "train_hf.py"
    script.write_text(
        "import torch\n"
        "from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments\n"
        "\n"
        "model = AutoModelForSequenceClassification.from_pretrained('local-model')\n"
        "args = TrainingArguments(output_dir='outputs')\n"
        "trainer = Trainer(model=model, args=args)\n"
        "trainer.train()\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["frameworks"][0]["name"] == "huggingface"
    assert data["conversion_state"] == "not_converted"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-huggingface"]
    assert any(item["kind"] == "huggingface_trainer" for item in data["frameworks"][0]["evidence"])
    assert all(item["line"] is not None for item in data["frameworks"][0]["evidence"])


def test_inspect_promotes_huggingface_trainer_over_realistic_pytorch_usage(tmp_path):
    script = tmp_path / "train_hf.py"
    script.write_text(
        "import torch\n"
        "from torch.utils.data import DataLoader\n"
        "from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments\n"
        "\n"
        "class Rows(torch.utils.data.Dataset):\n"
        "    def __len__(self):\n"
        "        return 1\n"
        "    def __getitem__(self, index):\n"
        "        return {'input_ids': torch.tensor([index]), 'labels': index}\n"
        "\n"
        "model = AutoModelForSequenceClassification.from_pretrained('local-model')\n"
        "args = TrainingArguments(output_dir='outputs')\n"
        "loader = DataLoader(Rows())\n"
        "trainer = Trainer(model=model, args=args, train_dataset=loader.dataset)\n"
        "trainer.train()\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["frameworks"][0]["name"] == "huggingface"
    assert "pytorch" in {item["name"] for item in data["frameworks"]}
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-huggingface"]


def test_inspect_promotes_huggingface_across_training_and_data_modules(tmp_path):
    (tmp_path / "train.py").write_text(
        "from transformers import Trainer, TrainingArguments\n"
        "from data import train_data\n"
        "\n"
        "trainer = Trainer(model=model, args=TrainingArguments(output_dir='outputs'), train_dataset=train_data)\n"
        "trainer.train()\n",
        encoding="utf-8",
    )
    (tmp_path / "data.py").write_text(
        "import torch\n"
        "from torch.utils.data import DataLoader\n"
        "\n"
        "class Rows(torch.utils.data.Dataset):\n"
        "    pass\n"
        "\n"
        "train_data = DataLoader(Rows()).dataset\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    assert data["frameworks"][0]["name"] == "huggingface"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-huggingface"]


def test_inspect_classifies_huggingface_patch_as_client_api_converted(tmp_path):
    script = tmp_path / "client.py"
    script.write_text(
        "from trl import SFTTrainer\n"
        "from nvflare.client import hf as flare\n"
        "\n"
        "trainer = SFTTrainer(model=model, args=args, train_dataset=train_data)\n"
        "flare.patch(trainer)\n"
        "while flare.is_running():\n"
        "    trainer.train()\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["frameworks"][0]["name"] == "huggingface"
    assert data["conversion_state"] == "client_api_converted"
    assert data["skill_selection"]["recommended_skills"] == []
    assert "flare.patch" in data["flare_integration"]["calls"]


def test_inspect_classifies_direct_huggingface_patch_alias_as_client_api_converted(tmp_path):
    script = tmp_path / "client.py"
    script.write_text(
        "from transformers import Trainer, TrainingArguments\n"
        "from nvflare.client.hf import patch as flare_patch\n"
        "\n"
        "trainer = Trainer(model=model, args=TrainingArguments(output_dir='outputs'))\n"
        "flare_patch(trainer)\n"
        "trainer.train()\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["conversion_state"] == "client_api_converted"
    assert data["flare_integration"]["calls"] == ["flare_patch"]


def test_inspect_does_not_recommend_huggingface_conversion_for_evaluation_only_trainer(tmp_path):
    script = tmp_path / "evaluate.py"
    script.write_text(
        "from transformers import Trainer, TrainingArguments\n"
        "\n"
        "trainer = Trainer(model=model, args=TrainingArguments(output_dir='outputs'), eval_dataset=eval_data)\n"
        "trainer.evaluate()\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["skill_selection"]["recommended_skills"] == ["nvflare-orient"]
    assert data["recommended_next_commands"] == ["Use the nvflare-orient skill before editing."]


def test_inspect_lightning_owner_still_wins_when_huggingface_candidate_declines(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "import torch\n"
        "import pytorch_lightning as pl\n"
        "from transformers import Trainer, TrainingArguments\n"
        "\n"
        "model = object()\n"
        "args = TrainingArguments(output_dir='outputs')\n"
        "hf_trainer = Trainer(model=model, args=args)\n"
        "hf_eval = Trainer(model=model, args=args)\n"
        "lit_trainer = pl.Trainer(max_epochs=1)\n"
        "optimizer = torch.optim.Adam([])\n"
        "lit_trainer.fit(model)\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["skill_selection"]["detected_framework"] == "pytorch_lightning"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-lightning"]


def test_inspect_routes_factory_built_trainer_candidate_to_orient(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "from transformers import TrainingArguments\n"
        "from my_lib import build_trainer\n"
        "\n"
        "trainer = build_trainer(TrainingArguments(output_dir='outputs'))\n"
        "trainer.train()\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["skill_selection"]["detected_framework"] == "huggingface"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-orient"]
    assert data["recommended_next_commands"] == ["Use the nvflare-orient skill before editing."]


def test_inspect_routes_factory_built_trainer_with_torch_import_to_orient(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "import torch\n"
        "from torch.utils.data import DataLoader\n"
        "from transformers import TrainingArguments\n"
        "from my_lib import build_trainer\n"
        "\n"
        "train_data = DataLoader(ds)\n"
        "trainer = build_trainer(TrainingArguments(output_dir='outputs'))\n"
        "trainer.train()\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["skill_selection"]["detected_framework"] == "huggingface"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-orient"]


@pytest.mark.parametrize(
    ("symbol", "alias", "arguments"),
    [
        ("DataLoader", "Loader", "ds"),
        ("DistributedSampler", "Sampler", "ds"),
        ("TensorDataset", "Dataset", "features"),
    ],
)
def test_inspect_treats_aliased_pytorch_data_helpers_as_data_plumbing(tmp_path, symbol, alias, arguments):
    script = tmp_path / "train.py"
    script.write_text(
        f"from torch.utils.data import {symbol} as {alias}\n"
        "from transformers import TrainingArguments\n"
        "from my_lib import build_trainer\n"
        "\n"
        f"train_data = {alias}({arguments})\n"
        "trainer = build_trainer(TrainingArguments(output_dir='outputs'))\n"
        "trainer.train()\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["skill_selection"]["detected_framework"] == "huggingface"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-orient"]
    pytorch = next(item for item in data["frameworks"] if item["name"] == "pytorch")
    assert any(item["kind"] == "pytorch_data_call" for item in pytorch["evidence"])


def test_inspect_does_not_route_huggingface_inference_with_torch_import_to_pytorch(tmp_path):
    script = tmp_path / "infer.py"
    script.write_text(
        "import torch\n" "from transformers import pipeline\n" "\n" "generator = pipeline('text-generation')\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["skill_selection"]["detected_framework"] == "huggingface"
    assert data["skill_selection"]["recommended_skills"] == []


def test_inspect_routes_cross_file_trainer_ownership_to_orient(tmp_path):
    (tmp_path / "train.py").write_text(
        "import torch\n" "from builder import build_trainer\n" "\n" "trainer = build_trainer()\n" "trainer.train()\n",
        encoding="utf-8",
    )
    (tmp_path / "builder.py").write_text(
        "from transformers import Trainer, TrainingArguments\n"
        "\n"
        "def build_trainer():\n"
        "    return Trainer(model=model, args=TrainingArguments(output_dir='outputs'))\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    assert data["skill_selection"]["detected_framework"] == "huggingface"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-orient"]


def test_inspect_routes_cross_file_trainer_with_pytorch_model_to_orient(tmp_path):
    (tmp_path / "train.py").write_text(
        "from builder import build_trainer\n" "\n" "trainer = build_trainer()\n" "trainer.train()\n",
        encoding="utf-8",
    )
    (tmp_path / "builder.py").write_text(
        "import torch\n"
        "from transformers import Trainer, TrainingArguments\n"
        "\n"
        "class Net(torch.nn.Module):\n"
        "    pass\n"
        "\n"
        "def build_trainer():\n"
        "    return Trainer(model=Net(), args=TrainingArguments(output_dir='outputs'))\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    assert data["skill_selection"]["detected_framework"] == "huggingface"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-orient"]


def test_inspect_keeps_pytorch_model_with_huggingface_utility_on_pytorch(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "import torch\n"
        "from transformers import AutoTokenizer\n"
        "\n"
        "class Net(torch.nn.Module):\n"
        "    pass\n"
        "\n"
        "tokenizer = AutoTokenizer.from_pretrained('local-model')\n"
        "model = Net()\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["skill_selection"]["detected_framework"] == "pytorch"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-pytorch"]


def test_inspect_routes_evaluation_only_trainer_with_torch_import_to_orient(tmp_path):
    script = tmp_path / "evaluate.py"
    script.write_text(
        "import torch\n"
        "from torch.utils.data import DataLoader\n"
        "from transformers import Trainer, TrainingArguments\n"
        "\n"
        "train_data = DataLoader(ds)\n"
        "trainer = Trainer(model=model, args=TrainingArguments(output_dir='outputs'), eval_dataset=eval_data)\n"
        "trainer.evaluate()\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["skill_selection"]["detected_framework"] == "huggingface"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-orient"]


def test_inspect_routes_indirect_train_call_to_orient(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "from transformers import Trainer, TrainingArguments\n"
        "\n"
        "def run(trainer):\n"
        "    trainer.train()\n"
        "\n"
        "trainer = Trainer(model=model, args=TrainingArguments(output_dir='outputs'))\n"
        "run(trainer)\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["skill_selection"]["detected_framework"] == "huggingface"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-orient"]


def test_inspect_detects_local_trainer_subclass(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "from transformers import Trainer, TrainingArguments\n"
        "\n"
        "class CustomTrainer(Trainer):\n"
        "    pass\n"
        "\n"
        "trainer = CustomTrainer(model=model, args=TrainingArguments(output_dir='outputs'))\n"
        "trainer.train()\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["skill_selection"]["detected_framework"] == "huggingface"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-huggingface"]


def test_inspect_prefers_active_manual_pytorch_over_evaluation_only_trainer(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "import torch\n"
        "from transformers import Trainer, TrainingArguments\n"
        "\n"
        "model = torch.nn.Linear(2, 1)\n"
        "optimizer = torch.optim.SGD(model.parameters(), lr=0.1)\n"
        "optimizer.step()\n"
        "trainer = Trainer(model=hf_model, args=TrainingArguments(output_dir='outputs'))\n"
        "trainer.evaluate()\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["skill_selection"]["detected_framework"] == "pytorch"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-pytorch"]


def test_inspect_does_not_bind_unrelated_function_parameter_to_trainer(tmp_path):
    script = tmp_path / "evaluate.py"
    script.write_text(
        "from transformers import Trainer, TrainingArguments\n"
        "\n"
        "def train_other(trainer):\n"
        "    trainer.train()\n"
        "\n"
        "trainer = Trainer(model=model, args=TrainingArguments(output_dir='outputs'))\n"
        "trainer.evaluate()\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["skill_selection"]["detected_framework"] == "huggingface"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-orient"]


def test_inspect_invalidates_reassigned_trainer_identity(tmp_path):
    script = tmp_path / "evaluate.py"
    script.write_text(
        "from transformers import Trainer, TrainingArguments\n"
        "\n"
        "trainer = Trainer(model=model, args=TrainingArguments(output_dir='outputs'))\n"
        "trainer = local_worker\n"
        "trainer.train()\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["skill_selection"]["detected_framework"] == "huggingface"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-orient"]


def test_inspect_treats_attribute_trainer_identity_as_unresolved(tmp_path):
    script = tmp_path / "evaluate.py"
    script.write_text(
        "from transformers import Trainer, TrainingArguments\n"
        "\n"
        "class Evaluation:\n"
        "    def __init__(self):\n"
        "        self.trainer = Trainer(model=model, args=TrainingArguments(output_dir='outputs'))\n"
        "    def evaluate(self):\n"
        "        self.trainer.evaluate()\n"
        "\n"
        "class Unrelated:\n"
        "    def train(self):\n"
        "        self.trainer.train()\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["skill_selection"]["detected_framework"] == "huggingface"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-orient"]


def test_inspect_detects_attribute_trainer_in_same_scope(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "from transformers import Trainer, TrainingArguments\n"
        "\n"
        "class Training:\n"
        "    def train(self):\n"
        "        self.trainer = Trainer(model=model, args=TrainingArguments(output_dir='outputs'))\n"
        "        self.trainer.train()\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["skill_selection"]["detected_framework"] == "huggingface"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-huggingface"]


def test_inspect_detects_module_global_trainer_used_in_function(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "from transformers import Trainer, TrainingArguments\n"
        "\n"
        "trainer = Trainer(model=model, args=TrainingArguments(output_dir='outputs'))\n"
        "\n"
        "def run():\n"
        "    trainer.train()\n"
        "\n"
        "run()\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["skill_selection"]["detected_framework"] == "huggingface"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-huggingface"]


def test_inspect_detects_module_global_trainer_constructed_after_function(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "from transformers import Trainer, TrainingArguments\n"
        "\n"
        "def run():\n"
        "    trainer.train()\n"
        "\n"
        "trainer = Trainer(model=model, args=TrainingArguments(output_dir='outputs'))\n"
        "run()\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["skill_selection"]["detected_framework"] == "huggingface"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-huggingface"]


def test_inspect_does_not_keep_enclosing_trainer_rebound_after_function_definition(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "from transformers import Trainer, TrainingArguments\n"
        "\n"
        "trainer = Trainer(model=model, args=TrainingArguments(output_dir='outputs'))\n"
        "\n"
        "def run():\n"
        "    trainer.train()\n"
        "\n"
        "trainer = worker\n"
        "run()\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["skill_selection"]["detected_framework"] == "huggingface"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-orient"]


def test_inspect_detects_closure_captured_trainer(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "from transformers import Trainer, TrainingArguments\n"
        "\n"
        "def run():\n"
        "    trainer = Trainer(model=model, args=TrainingArguments(output_dir='outputs'))\n"
        "\n"
        "    def train():\n"
        "        trainer.train()\n"
        "\n"
        "    train()\n"
        "\n"
        "run()\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["skill_selection"]["detected_framework"] == "huggingface"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-huggingface"]


def test_inspect_detects_closure_trainer_constructed_after_nested_function(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "from transformers import Trainer, TrainingArguments\n"
        "\n"
        "def run():\n"
        "    def train():\n"
        "        trainer.train()\n"
        "\n"
        "    trainer = Trainer(model=model, args=TrainingArguments(output_dir='outputs'))\n"
        "    train()\n"
        "\n"
        "run()\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["skill_selection"]["detected_framework"] == "huggingface"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-huggingface"]


def test_inspect_does_not_resolve_train_call_before_same_scope_assignment(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "from transformers import Trainer, TrainingArguments\n"
        "\n"
        "def run():\n"
        "    trainer.train()\n"
        "    trainer = Trainer(model=model, args=TrainingArguments(output_dir='outputs'))\n"
        "\n"
        "run()\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["skill_selection"]["detected_framework"] == "huggingface"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-orient"]


def test_inspect_resolves_global_trainer_assignment(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "from transformers import Trainer, TrainingArguments\n"
        "\n"
        "trainer = None\n"
        "\n"
        "def configure():\n"
        "    global trainer\n"
        "    trainer = Trainer(model=model, args=TrainingArguments(output_dir='outputs'))\n"
        "\n"
        "def run():\n"
        "    trainer.train()\n"
        "\n"
        "configure()\n"
        "run()\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["skill_selection"]["detected_framework"] == "huggingface"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-huggingface"]


def test_inspect_resolves_nonlocal_trainer_assignment(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "from transformers import Trainer, TrainingArguments\n"
        "\n"
        "def run():\n"
        "    trainer = None\n"
        "\n"
        "    def configure():\n"
        "        nonlocal trainer\n"
        "        trainer = Trainer(model=model, args=TrainingArguments(output_dir='outputs'))\n"
        "\n"
        "    def train():\n"
        "        trainer.train()\n"
        "\n"
        "    configure()\n"
        "    train()\n"
        "\n"
        "run()\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["skill_selection"]["detected_framework"] == "huggingface"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-huggingface"]


def test_inspect_does_not_reuse_rebound_trainer_constructor(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "from transformers import Trainer, TrainingArguments\n"
        "\n"
        "class LocalTrainer:\n"
        "    pass\n"
        "\n"
        "Trainer = LocalTrainer\n"
        "trainer = Trainer()\n"
        "trainer.train()\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["skill_selection"]["detected_framework"] == "huggingface"
    assert data["skill_selection"]["recommended_skills"] == []


def test_inspect_does_not_reuse_trainer_constructor_rebound_by_import(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "from transformers import Trainer\n"
        "from local_training import Trainer\n"
        "\n"
        "trainer = Trainer()\n"
        "trainer.train()\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["skill_selection"]["detected_framework"] == "huggingface"
    assert data["skill_selection"]["recommended_skills"] == []


def test_inspect_does_not_reuse_same_named_trainer_subclass_across_scopes(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "from transformers import Trainer, TrainingArguments\n"
        "\n"
        "def define_hf_trainer():\n"
        "    class Worker(Trainer):\n"
        "        pass\n"
        "\n"
        "def run():\n"
        "    class Worker:\n"
        "        pass\n"
        "\n"
        "    trainer = Worker()\n"
        "    trainer.train()\n"
        "\n"
        "run()\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["skill_selection"]["detected_framework"] == "huggingface"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-orient"]


def test_inspect_invalidates_trainer_identity_for_loop_target(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "from transformers import Trainer, TrainingArguments\n"
        "\n"
        "trainer = Trainer(model=model, args=TrainingArguments(output_dir='outputs'))\n"
        "for trainer in workers:\n"
        "    trainer.train()\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["skill_selection"]["detected_framework"] == "huggingface"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-orient"]


def test_inspect_invalidates_trainer_identity_for_comprehension_target(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "from transformers import Trainer, TrainingArguments\n"
        "\n"
        "trainer = Trainer(model=model, args=TrainingArguments(output_dir='outputs'))\n"
        "results = [trainer.train() for trainer in workers]\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["skill_selection"]["detected_framework"] == "huggingface"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-orient"]


def test_inspect_invalidates_trainer_constructor_rebound_by_function(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "from transformers import Trainer\n"
        "\n"
        "def Trainer():\n"
        "    return worker\n"
        "\n"
        "trainer = Trainer()\n"
        "trainer.train()\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["skill_selection"]["detected_framework"] == "huggingface"
    assert data["skill_selection"]["recommended_skills"] == []


def test_inspect_detects_trl_sft_trainer_submodule_alias(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "from trl.trainer import sft_trainer as trainer_module\n"
        "\n"
        "trainer = trainer_module.SFTTrainer(model=model, args=args, train_dataset=train_data)\n"
        "trainer.train()\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["skill_selection"]["detected_framework"] == "huggingface"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-huggingface"]


def test_inspect_does_not_inflate_huggingface_evidence_for_type_only_imports(tmp_path):
    script = tmp_path / "types.py"
    script.write_text(
        "import torch\n"
        "from transformers import Trainer, TrainingArguments\n"
        "\n"
        "def describe(trainer: Trainer, args: TrainingArguments) -> torch.Tensor:\n"
        "    return torch.tensor([1])\n",
        encoding="utf-8",
    )

    data = inspect_path(script)
    huggingface = next(item for item in data["frameworks"] if item["name"] == "huggingface")

    assert [item["kind"] for item in huggingface["evidence"]] == ["import"]
    assert data["skill_selection"]["detected_framework"] == "huggingface"
    assert data["skill_selection"]["recommended_skills"] == []


def test_inspect_detects_huggingface_trainer_submodule_alias(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "from transformers import trainer as trainer_module\n"
        "\n"
        "trainer = trainer_module.Trainer(model=model, args=args, train_dataset=train_data)\n"
        "trainer.train()\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["frameworks"][0]["name"] == "huggingface"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-huggingface"]


def test_inspect_keeps_manual_transformers_loop_with_pytorch_converter(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "import torch\n"
        "from transformers import AutoModelForSequenceClassification\n"
        "\n"
        "model = AutoModelForSequenceClassification.from_pretrained('local-model')\n"
        "optimizer = torch.optim.SGD(model.parameters(), lr=0.1)\n"
        "loss = model(**batch).loss\n"
        "loss.backward()\n"
        "optimizer.step()\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["frameworks"][0]["name"] == "pytorch"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-pytorch"]


def test_inspect_does_not_recommend_conversion_for_huggingface_inference_only(tmp_path):
    script = tmp_path / "infer.py"
    script.write_text(
        "from transformers import pipeline\n"
        "\n"
        "generator = pipeline('text-generation', model='local-model')\n"
        "print(generator('hello'))\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["frameworks"][0]["name"] == "huggingface"
    assert data["skill_selection"]["recommended_skills"] == []
    assert data["recommended_next_commands"] == []


def test_inspect_keeps_transformers_model_under_lightning_converter(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "import lightning as L\n"
        "from transformers import AutoModel\n"
        "\n"
        "class Model(L.LightningModule):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.encoder = AutoModel.from_pretrained('local-model')\n"
        "\n"
        "trainer = L.Trainer(max_epochs=1)\n"
        "trainer.fit(Model(), train_dataloaders=loader)\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["frameworks"][0]["name"] == "pytorch_lightning"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-lightning"]


def test_inspect_routes_active_lightning_and_huggingface_trainers_to_orient(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "import lightning as L\n"
        "from transformers import Trainer, TrainingArguments\n"
        "\n"
        "lightning_trainer = L.Trainer(max_epochs=1)\n"
        "hf_trainer = Trainer(model=model, args=TrainingArguments(output_dir='outputs'))\n"
        "lightning_trainer.fit(lightning_model, train_dataloaders=loader)\n"
        "hf_trainer.train()\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["skill_selection"]["recommended_skills"] == ["nvflare-orient"]
    assert data["recommended_next_commands"] == ["Use the nvflare-orient skill before editing."]


def test_inspect_routes_independent_lightning_and_huggingface_entrypoints_to_orient(tmp_path):
    (tmp_path / "train_lightning.py").write_text(
        "import lightning as L\n"
        "\n"
        "def main():\n"
        "    trainer = L.Trainer(max_epochs=1)\n"
        "    trainer.fit(model, train_dataloaders=loader)\n"
        "\n"
        "main()\n",
        encoding="utf-8",
    )
    (tmp_path / "train_huggingface.py").write_text(
        "from transformers import Trainer, TrainingArguments\n"
        "\n"
        "def main():\n"
        "    trainer = Trainer(model=model, args=TrainingArguments(output_dir='outputs'))\n"
        "    trainer.train()\n"
        "\n"
        "main()\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    assert data["skill_selection"]["recommended_skills"] == ["nvflare-orient"]
    assert data["recommended_next_commands"] == ["Use the nvflare-orient skill before editing."]


def test_inspect_ignores_unused_lightning_trainer_beside_huggingface_entrypoint(tmp_path):
    (tmp_path / "main.py").write_text(
        "from transformers import Trainer, TrainingArguments\n"
        "\n"
        "trainer = Trainer(model=model, args=TrainingArguments(output_dir='outputs'))\n"
        "trainer.train()\n",
        encoding="utf-8",
    )
    (tmp_path / "unused_lightning.py").write_text(
        "import lightning as L\n" "\n" "trainer = L.Trainer(max_epochs=1)\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-huggingface"]


def test_inspect_ignores_unused_huggingface_trainer_beside_lightning_entrypoint(tmp_path):
    (tmp_path / "main.py").write_text(
        "import lightning as L\n"
        "\n"
        "trainer = L.Trainer(max_epochs=1)\n"
        "trainer.fit(model, train_dataloaders=loader)\n",
        encoding="utf-8",
    )
    (tmp_path / "unused_huggingface.py").write_text(
        "from transformers import Trainer, TrainingArguments\n"
        "\n"
        "trainer = Trainer(model=model, args=TrainingArguments(output_dir='outputs'))\n"
        "trainer.train()\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-lightning"]


def test_inspect_routes_lightning_module_owned_by_huggingface_trainer_to_huggingface(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "import lightning as L\n"
        "from transformers import Trainer, TrainingArguments\n"
        "\n"
        "class Model(L.LightningModule):\n"
        "    pass\n"
        "\n"
        "trainer = Trainer(model=Model(), args=TrainingArguments(output_dir='outputs'))\n"
        "trainer.train()\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-huggingface"]


def test_inspect_detects_lightning_pytorch_trainer_import(tmp_path):
    script = tmp_path / "train_lightning.py"
    script.write_text(
        "from lightning.pytorch import LightningDataModule, Trainer\n"
        "\n"
        "class Data(LightningDataModule):\n"
        "    pass\n"
        "\n"
        "trainer = Trainer(max_epochs=1)\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["frameworks"][0]["name"] == "pytorch_lightning"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-lightning"]
    evidence_kinds = {item["kind"] for item in data["frameworks"][0]["evidence"]}
    assert {"import", "lightning_class", "lightning_trainer"} <= evidence_kinds


def test_inspect_detects_top_level_lightning_alias_and_from_import(tmp_path):
    script = tmp_path / "train_lightning.py"
    script.write_text(
        "import lightning as L\n"
        "from lightning import LightningModule, Trainer\n"
        "\n"
        "class AliasNet(L.LightningModule):\n"
        "    pass\n"
        "\n"
        "class ImportedNet(LightningModule):\n"
        "    pass\n"
        "\n"
        "alias_trainer = L.Trainer(max_epochs=1)\n"
        "imported_trainer = Trainer(max_epochs=1)\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["frameworks"][0]["name"] == "pytorch_lightning"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-lightning"]
    evidence = data["frameworks"][0]["evidence"]
    assert any(item["kind"] == "lightning_class" and item["value"] == "L.LightningModule" for item in evidence)
    assert any(item["kind"] == "lightning_class" and item["value"] == "LightningModule" for item in evidence)
    assert any(item["kind"] == "lightning_trainer" and item["value"] == "L.Trainer" for item in evidence)
    assert any(item["kind"] == "lightning_trainer" and item["value"] == "Trainer" for item in evidence)


# Lightning-patch conversion-state cases: each writes one trainer script that
# differs only in the import/patch-call spelling and must be classified as
# client_api_converted. Fields: source, expected_call, exact_calls (assert
# calls == [expected_call] instead of membership), and check_framework
# (assert frameworks[0] is pytorch_lightning).
_LIGHTNING_PATCH_CONVERTED_CASES = [
    pytest.param(
        "import lightning as L\n"
        "import nvflare.client.lightning as flare\n"
        "\n"
        "trainer = L.Trainer(max_epochs=1)\n"
        "flare.patch(trainer)\n"
        "trainer.fit(model, datamodule=data)\n",
        "flare.patch",
        True,
        True,
        id="classifies_lightning_patched_trainer_as_client_api_converted",
    ),
    pytest.param(
        "import lightning as L\n"
        "from nvflare.client.lightning import patch\n"
        "\n"
        "trainer = L.Trainer(max_epochs=1)\n"
        "patch(trainer)\n"
        "trainer.fit(model, datamodule=data)\n",
        "patch",
        True,
        True,
        id="classifies_imported_lightning_patch_as_client_api_converted",
    ),
    pytest.param(
        "import lightning as L\n"
        "from nvflare.client.lightning import patch as flare_patch\n"
        "\n"
        "trainer = L.Trainer(max_epochs=1)\n"
        "flare_patch(trainer)\n"
        "trainer.fit(model, datamodule=data)\n",
        "flare_patch",
        True,
        True,
        id="classifies_aliased_lightning_patch_import_as_client_api_converted",
    ),
    pytest.param(
        "import lightning as L\n"
        "import nvflare.client.lightning as nfl\n"
        "\n"
        "trainer = L.Trainer(max_epochs=1)\n"
        "nfl.patch(trainer)\n"
        "trainer.fit(model, datamodule=data)\n",
        "nfl.patch",
        False,
        True,
        id="classifies_aliased_lightning_patch_module_as_client_api_converted",
    ),
    pytest.param(
        "import lightning as L\n"
        "import nvflare.client.lightning\n"
        "\n"
        "trainer = L.Trainer(max_epochs=1)\n"
        "nvflare.client.lightning.patch(trainer)\n"
        "trainer.fit(model, datamodule=data)\n",
        "nvflare.client.lightning.patch",
        False,
        True,
        id="classifies_fully_qualified_lightning_patch_as_client_api_converted",
    ),
    pytest.param(
        "from nemo import lightning as nl\n"
        "import nvflare.client.lightning\n"
        "\n"
        "trainer = nl.Trainer(max_steps=10)\n"
        "nvflare.client.lightning.patch(trainer)\n"
        "trainer.fit(model)\n",
        "nvflare.client.lightning.patch",
        False,
        False,
        id="classifies_fully_qualified_lightning_patch_for_wrapper_trainer_as_converted",
    ),
    pytest.param(
        "import lightning as L\n"
        "from nvflare.client import lightning as flare\n"
        "\n"
        "trainer = L.Trainer(max_epochs=1)\n"
        "flare.patch(trainer)\n"
        "trainer.fit(model, datamodule=data)\n",
        "flare.patch",
        False,
        True,
        id="classifies_from_import_lightning_module_alias_as_client_api_converted",
    ),
    pytest.param(
        "import lightning as L\n"
        "from nvflare.client import lightning\n"
        "\n"
        "trainer = L.Trainer(max_epochs=1)\n"
        "lightning.patch(trainer)\n"
        "trainer.fit(model, datamodule=data)\n",
        "lightning.patch",
        False,
        True,
        id="classifies_from_import_lightning_module_as_client_api_converted",
    ),
    # nemo.lightning-style wrapper: the trainer is built via ``nl.Trainer`` which
    # is not a recognized Lightning constructor, but ``flare.patch(trainer)`` is
    # still the definitive conversion signal.
    pytest.param(
        "from nemo import lightning as nl\n"
        "import nvflare.client.lightning as flare\n"
        "\n"
        "trainer = nl.Trainer(max_steps=10)\n"
        "flare.patch(trainer, restore_state=False)\n"
        "trainer.fit(model)\n",
        None,
        False,
        False,
        id="classifies_wrapper_trainer_lightning_patch_as_client_api_converted",
    ),
]


@pytest.mark.parametrize(
    ("source", "expected_call", "exact_calls", "check_framework"), _LIGHTNING_PATCH_CONVERTED_CASES
)
def test_inspect_classifies_lightning_patch_as_client_api_converted(
    tmp_path, source, expected_call, exact_calls, check_framework
):
    script = tmp_path / "client.py"
    script.write_text(source, encoding="utf-8")

    data = inspect_path(script)

    if check_framework:
        assert data["frameworks"][0]["name"] == "pytorch_lightning"
    if exact_calls:
        assert data["flare_integration"]["calls"] == [expected_call]
    elif expected_call is not None:
        assert expected_call in data["flare_integration"]["calls"]
    assert data["conversion_state"] == "client_api_converted"


@pytest.mark.parametrize(
    ("framework", "source", "expected_skill"),
    [
        pytest.param(
            "huggingface",
            "from transformers import Trainer, TrainingArguments\n"
            "from nvflare.client.hf import patch\n"
            "\n"
            "trainer = Trainer(model=model, args=TrainingArguments(output_dir='outputs'))\n"
            "patch = local_patch\n"
            "patch(trainer)\n"
            "trainer.train()\n",
            "nvflare-convert-huggingface",
            id="huggingface-direct-alias-rebound",
        ),
        pytest.param(
            "huggingface",
            "from transformers import Trainer, TrainingArguments\n"
            "from nvflare.client import hf as flare\n"
            "\n"
            "trainer = Trainer(model=model, args=TrainingArguments(output_dir='outputs'))\n"
            "flare = local_module\n"
            "flare.patch(trainer)\n"
            "trainer.train()\n",
            "nvflare-convert-huggingface",
            id="huggingface-module-alias-rebound",
        ),
        pytest.param(
            "pytorch_lightning",
            "import lightning as L\n"
            "from nvflare.client.lightning import patch\n"
            "trainer = L.Trainer(max_epochs=1)\n"
            "patch = local_patch\n"
            "patch(trainer)\n"
            "trainer.fit(model)\n",
            "nvflare-convert-lightning",
            id="lightning-direct-alias-rebound",
        ),
        pytest.param(
            "pytorch_lightning",
            "import lightning as L\n"
            "from nvflare.client import lightning as flare\n"
            "trainer = L.Trainer(max_epochs=1)\n"
            "flare = local_module\n"
            "flare.patch(trainer)\n"
            "trainer.fit(model)\n",
            "nvflare-convert-lightning",
            id="lightning-module-alias-rebound",
        ),
        pytest.param(
            "pytorch_lightning",
            "import lightning as L\n"
            "from nvflare.client.lightning import patch\n"
            "from local_callbacks import patch\n"
            "trainer = L.Trainer(max_epochs=1)\n"
            "patch(trainer)\n"
            "trainer.fit(model)\n",
            "nvflare-convert-lightning",
            id="lightning-direct-alias-reimported",
        ),
    ],
)
def test_inspect_does_not_treat_rebound_patch_alias_as_conversion(tmp_path, framework, source, expected_skill):
    script = tmp_path / "client.py"
    script.write_text(source, encoding="utf-8")

    data = inspect_path(script)

    assert data["frameworks"][0]["name"] == framework
    assert data["conversion_state"] == "partial_client_api"
    assert data["skill_selection"]["recommended_skills"] == [expected_skill]


@pytest.mark.parametrize(
    ("framework", "source", "expected_skill"),
    [
        pytest.param(
            "huggingface",
            "from transformers import Trainer, TrainingArguments\n"
            "from nvflare.client.hf import patch\n"
            "\n"
            "trainer = Trainer(model=model, args=TrainingArguments(output_dir='outputs'))\n"
            "def apply_patch(patch):\n"
            "    patch(trainer)\n"
            "apply_patch(local_patch)\n"
            "trainer.train()\n",
            "nvflare-convert-huggingface",
            id="huggingface-direct-alias-parameter-shadow",
        ),
        pytest.param(
            "huggingface",
            "from transformers import Trainer, TrainingArguments\n"
            "from nvflare.client import hf as flare\n"
            "\n"
            "trainer = Trainer(model=model, args=TrainingArguments(output_dir='outputs'))\n"
            "def apply_patch(flare):\n"
            "    flare.patch(trainer)\n"
            "apply_patch(local_module)\n"
            "trainer.train()\n",
            "nvflare-convert-huggingface",
            id="huggingface-module-alias-parameter-shadow",
        ),
        pytest.param(
            "pytorch_lightning",
            "import lightning as L\n"
            "from nvflare.client.lightning import patch\n"
            "trainer = L.Trainer(max_epochs=1)\n"
            "def apply_patch(patch):\n"
            "    patch(trainer)\n"
            "apply_patch(local_patch)\n"
            "trainer.fit(model)\n",
            "nvflare-convert-lightning",
            id="lightning-direct-alias-parameter-shadow",
        ),
        pytest.param(
            "pytorch_lightning",
            "import lightning as L\n"
            "from nvflare.client import lightning as flare\n"
            "trainer = L.Trainer(max_epochs=1)\n"
            "def apply_patch(flare):\n"
            "    flare.patch(trainer)\n"
            "apply_patch(local_module)\n"
            "trainer.fit(model)\n",
            "nvflare-convert-lightning",
            id="lightning-module-alias-parameter-shadow",
        ),
    ],
)
def test_inspect_does_not_treat_shadowed_patch_alias_as_conversion(tmp_path, framework, source, expected_skill):
    script = tmp_path / "client.py"
    script.write_text(source, encoding="utf-8")

    data = inspect_path(script)

    assert data["frameworks"][0]["name"] == framework
    assert data["conversion_state"] == "partial_client_api"
    assert data["skill_selection"]["recommended_skills"] == [expected_skill]


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            "from transformers import Trainer, TrainingArguments\n"
            "trainer = Trainer(model=model, args=TrainingArguments(output_dir='outputs'))\n"
            "def configure():\n"
            "    patch(trainer)\n"
            "from nvflare.client.hf import patch\n"
            "configure()\n"
            "trainer.train()\n",
            id="huggingface-forward-direct-patch-import",
        ),
        pytest.param(
            "from transformers import Trainer, TrainingArguments\n"
            "trainer = Trainer(model=model, args=TrainingArguments(output_dir='outputs'))\n"
            "def configure():\n"
            "    flare.patch(trainer)\n"
            "from nvflare.client import hf as flare\n"
            "configure()\n"
            "trainer.train()\n",
            id="huggingface-forward-module-patch-import",
        ),
        pytest.param(
            "from transformers import Trainer, TrainingArguments\n"
            "trainer = Trainer(model=model, args=TrainingArguments(output_dir='outputs'))\n"
            "def configure():\n"
            "    nvflare.client.hf.patch(trainer)\n"
            "import nvflare.client.hf\n"
            "configure()\n"
            "trainer.train()\n",
            id="huggingface-forward-fully-qualified-patch-import",
        ),
    ],
)
def test_inspect_resolves_huggingface_patch_imported_after_function_definition(tmp_path, source):
    script = tmp_path / "client.py"
    script.write_text(source, encoding="utf-8")

    data = inspect_path(script)

    assert data["frameworks"][0]["name"] == "huggingface"
    assert data["conversion_state"] == "client_api_converted"


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            "import lightning as L\n"
            "trainer = L.Trainer(max_epochs=1)\n"
            "def configure():\n"
            "    patch(trainer)\n"
            "from nvflare.client.lightning import patch\n"
            "configure()\n"
            "trainer.fit(model)\n",
            id="direct-patch-import",
        ),
        pytest.param(
            "import lightning as L\n"
            "trainer = L.Trainer(max_epochs=1)\n"
            "def configure():\n"
            "    flare.patch(trainer)\n"
            "from nvflare.client import lightning as flare\n"
            "configure()\n"
            "trainer.fit(model)\n",
            id="module-patch-import",
        ),
        pytest.param(
            "import lightning as L\n"
            "trainer = L.Trainer(max_epochs=1)\n"
            "def configure():\n"
            "    nvflare.client.lightning.patch(trainer)\n"
            "import nvflare.client.lightning\n"
            "configure()\n"
            "trainer.fit(model)\n",
            id="fully-qualified-patch-import",
        ),
    ],
)
def test_inspect_resolves_lightning_patch_imported_after_function_definition(tmp_path, source):
    script = tmp_path / "client.py"
    script.write_text(source, encoding="utf-8")

    data = inspect_path(script)

    assert data["frameworks"][0]["name"] == "pytorch_lightning"
    assert data["conversion_state"] == "client_api_converted"


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            "from transformers import Trainer, TrainingArguments\n"
            "from nvflare.client.hf import patch\n"
            "trainer = Trainer(model=model, args=TrainingArguments(output_dir='outputs'))\n"
            "patch = patch(trainer)\n"
            "trainer.train()\n",
            id="direct-patch",
        ),
        pytest.param(
            "from transformers import Trainer, TrainingArguments\n"
            "from nvflare.client import hf as flare\n"
            "trainer = Trainer(model=model, args=TrainingArguments(output_dir='outputs'))\n"
            "flare = flare.patch(trainer)\n"
            "trainer.train()\n",
            id="module-patch",
        ),
    ],
)
def test_inspect_resolves_huggingface_patch_before_assignment_target_rebinding(tmp_path, source):
    script = tmp_path / "client.py"
    script.write_text(source, encoding="utf-8")

    data = inspect_path(script)

    assert data["frameworks"][0]["name"] == "huggingface"
    assert data["conversion_state"] == "client_api_converted"


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            "import lightning as L\n"
            "from nvflare.client.lightning import patch\n"
            "trainer = L.Trainer(max_epochs=1)\n"
            "patch = patch(trainer)\n"
            "trainer.fit(model)\n",
            id="direct-patch",
        ),
        pytest.param(
            "import lightning as L\n"
            "from nvflare.client import lightning as flare\n"
            "trainer = L.Trainer(max_epochs=1)\n"
            "flare = flare.patch(trainer)\n"
            "trainer.fit(model)\n",
            id="module-patch",
        ),
    ],
)
def test_inspect_resolves_lightning_patch_before_assignment_target_rebinding(tmp_path, source):
    script = tmp_path / "client.py"
    script.write_text(source, encoding="utf-8")

    data = inspect_path(script)

    assert data["frameworks"][0]["name"] == "pytorch_lightning"
    assert data["conversion_state"] == "client_api_converted"


@pytest.mark.parametrize(
    "assignment",
    [
        pytest.param("patch, result = patch(trainer), trainer", id="destructured"),
        pytest.param("patch = identity(patch(trainer))", id="nested-call"),
    ],
)
def test_inspect_resolves_nested_huggingface_patch_before_assignment_target_rebinding(tmp_path, assignment):
    script = tmp_path / "client.py"
    script.write_text(
        "from transformers import Trainer, TrainingArguments\n"
        "from nvflare.client.hf import patch\n"
        "trainer = Trainer(model=model, args=TrainingArguments(output_dir='outputs'))\n"
        f"{assignment}\n"
        "trainer.train()\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["frameworks"][0]["name"] == "huggingface"
    assert data["conversion_state"] == "client_api_converted"


def test_inspect_does_not_trust_rebound_fully_qualified_huggingface_patch_root(tmp_path):
    script = tmp_path / "client.py"
    script.write_text(
        "from transformers import Trainer, TrainingArguments\n"
        "import nvflare.client.hf\n"
        "trainer = Trainer(model=model, args=TrainingArguments(output_dir='outputs'))\n"
        "nvflare = local_module\n"
        "nvflare.client.hf.patch(trainer)\n"
        "trainer.train()\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["frameworks"][0]["name"] == "huggingface"
    assert data["conversion_state"] == "partial_client_api"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-huggingface"]


def test_inspect_does_not_trust_rebound_fully_qualified_lightning_patch_root(tmp_path):
    script = tmp_path / "client.py"
    script.write_text(
        "import lightning as L\n"
        "import nvflare.client.lightning\n"
        "trainer = L.Trainer(max_epochs=1)\n"
        "nvflare = local_module\n"
        "nvflare.client.lightning.patch(trainer)\n"
        "trainer.fit(model)\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["frameworks"][0]["name"] == "pytorch_lightning"
    assert data["conversion_state"] == "partial_client_api"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-lightning"]


@pytest.mark.parametrize(
    "shadowing",
    [
        pytest.param("L = local_lightning\nL.Trainer(max_epochs=1)\n", id="assignment"),
        pytest.param("import local_lightning as L\nL.Trainer(max_epochs=1)\n", id="reimport"),
        pytest.param("for L in providers:\n    pass\nL.Trainer(max_epochs=1)\n", id="loop"),
        pytest.param("[L.Trainer(max_epochs=1) for L in providers]\n", id="comprehension"),
        pytest.param(
            "def helper(L):\n    L.Trainer(max_epochs=1)\nhelper(local_lightning)\n",
            id="parameter",
        ),
    ],
)
def test_inspect_does_not_emit_lightning_owner_for_shadowed_alias(tmp_path, shadowing):
    script = tmp_path / "train.py"
    script.write_text(
        "import lightning as L\n"
        "from transformers import Trainer, TrainingArguments\n"
        f"{shadowing}"
        "trainer = Trainer(model=model, args=TrainingArguments(output_dir='outputs'))\n"
        "trainer.train()\n",
        encoding="utf-8",
    )

    data = inspect_path(script)
    lightning = next(item for item in data["frameworks"] if item["name"] == "pytorch_lightning")

    assert not any(item["kind"] == "lightning_trainer" for item in lightning["evidence"])
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-huggingface"]


def test_inspect_does_not_emit_lightning_owner_for_rebound_trainer_symbol(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "from lightning import Trainer\n"
        "from transformers import TrainingArguments\n"
        "from local_factory import build_trainer\n"
        "Trainer = LocalTrainer\n"
        "Trainer(max_epochs=1)\n"
        "trainer = build_trainer(TrainingArguments(output_dir='outputs'))\n"
        "trainer.train()\n",
        encoding="utf-8",
    )

    data = inspect_path(script)
    lightning = next(item for item in data["frameworks"] if item["name"] == "pytorch_lightning")

    assert not any(item["kind"] == "lightning_trainer" for item in lightning["evidence"])
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-orient"]


def test_inspect_does_not_emit_lightning_owner_for_shadowing_trainer_class(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "from lightning import Trainer\n"
        "from transformers import TrainingArguments\n"
        "from local_factory import build_trainer\n"
        "class Trainer:\n"
        "    pass\n"
        "Trainer()\n"
        "trainer = build_trainer(TrainingArguments(output_dir='outputs'))\n"
        "trainer.train()\n",
        encoding="utf-8",
    )

    data = inspect_path(script)
    lightning = next(item for item in data["frameworks"] if item["name"] == "pytorch_lightning")

    assert not any(item["kind"] == "lightning_trainer" for item in lightning["evidence"])
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-orient"]


@pytest.mark.parametrize(
    "shadowing",
    [
        pytest.param("torch = local_torch\ntorch.optim.SGD(params)\n", id="assignment"),
        pytest.param("import local_torch as torch\ntorch.optim.SGD(params)\n", id="reimport"),
        pytest.param("for torch in providers:\n    pass\ntorch.optim.SGD(params)\n", id="loop"),
        pytest.param("[torch.optim.SGD(params) for torch in providers]\n", id="comprehension"),
        pytest.param(
            "def helper(torch):\n    torch.optim.SGD(params)\nhelper(local_torch)\n",
            id="parameter",
        ),
    ],
)
def test_inspect_does_not_emit_pytorch_owner_for_shadowed_alias(tmp_path, shadowing):
    script = tmp_path / "train.py"
    script.write_text(
        "import torch\n"
        "from transformers import TrainingArguments\n"
        "from local_factory import build_trainer\n"
        f"{shadowing}"
        "trainer = build_trainer(TrainingArguments(output_dir='outputs'))\n"
        "trainer.train()\n",
        encoding="utf-8",
    )

    data = inspect_path(script)
    pytorch = next(item for item in data["frameworks"] if item["name"] == "pytorch")

    assert not any(item["kind"] == "pytorch_call" for item in pytorch["evidence"])
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-orient"]


def test_inspect_does_not_emit_pytorch_owner_for_rebound_optimizer_symbol(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "from torch.optim import SGD\n"
        "from transformers import TrainingArguments\n"
        "from local_factory import build_trainer\n"
        "SGD = LocalOptimizer\n"
        "SGD(params)\n"
        "trainer = build_trainer(TrainingArguments(output_dir='outputs'))\n"
        "trainer.train()\n",
        encoding="utf-8",
    )

    data = inspect_path(script)
    pytorch = next(item for item in data["frameworks"] if item["name"] == "pytorch")

    assert not any(item["kind"] == "pytorch_call" for item in pytorch["evidence"])
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-orient"]


def test_inspect_does_not_emit_pytorch_owner_for_shadowing_optimizer_class(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "from torch.optim import SGD\n"
        "from transformers import TrainingArguments\n"
        "from local_factory import build_trainer\n"
        "class SGD:\n"
        "    pass\n"
        "SGD()\n"
        "trainer = build_trainer(TrainingArguments(output_dir='outputs'))\n"
        "trainer.train()\n",
        encoding="utf-8",
    )

    data = inspect_path(script)
    pytorch = next(item for item in data["frameworks"] if item["name"] == "pytorch")

    assert not any(item["kind"] == "pytorch_call" for item in pytorch["evidence"])
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-orient"]


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            "from transformers import Trainer, TrainingArguments\n"
            "from nvflare.client.hf import patch\n"
            "trainer = Trainer(model=model, args=TrainingArguments(output_dir='outputs'))\n"
            "def configure(patch=patch(trainer)):\n"
            "    patch = local_patch\n",
            id="huggingface-default-before-body-local",
        ),
        pytest.param(
            "from transformers import Trainer, TrainingArguments\n"
            "from nvflare.client.hf import patch\n"
            "trainer = Trainer(model=model, args=TrainingArguments(output_dir='outputs'))\n"
            "@patch(trainer)\n"
            "def patch():\n"
            "    pass\n",
            id="huggingface-decorator-before-function-name",
        ),
        pytest.param(
            "from transformers import Trainer, TrainingArguments\n"
            "from nvflare.client.hf import patch\n"
            "trainer = Trainer(model=model, args=TrainingArguments(output_dir='outputs'))\n"
            "def configure(value: patch(trainer)):\n"
            "    pass\n",
            id="huggingface-annotation",
        ),
        pytest.param(
            "from transformers import Trainer, TrainingArguments\n"
            "from nvflare.client.hf import patch\n"
            "trainer = Trainer(model=model, args=TrainingArguments(output_dir='outputs'))\n"
            "configure = lambda patch=patch(trainer): patch\n",
            id="huggingface-lambda-default",
        ),
        pytest.param(
            "import lightning as L\n"
            "from nvflare.client.lightning import patch\n"
            "trainer = L.Trainer(max_epochs=1)\n"
            "async def configure(patch=patch(trainer)):\n"
            "    patch = local_patch\n",
            id="lightning-async-default-before-body-local",
        ),
        pytest.param(
            "import lightning as L\n"
            "from nvflare.client.lightning import patch\n"
            "trainer = L.Trainer(max_epochs=1)\n"
            "@patch(trainer)\n"
            "class patch:\n"
            "    pass\n",
            id="lightning-class-decorator-before-class-name",
        ),
    ],
)
def test_inspect_resolves_patch_in_definition_time_enclosing_scope(tmp_path, source):
    script = tmp_path / "client.py"
    script.write_text(source, encoding="utf-8")

    data = inspect_path(script)

    assert data["conversion_state"] == "client_api_converted"


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            "from transformers import Trainer, TrainingArguments\n"
            "trainer = Trainer(model=model, args=TrainingArguments(output_dir='outputs'))\n"
            "def configure(value=patch(trainer)):\n"
            "    pass\n"
            "from nvflare.client.hf import patch\n",
            id="huggingface",
        ),
        pytest.param(
            "import lightning as L\n"
            "trainer = L.Trainer(max_epochs=1)\n"
            "def configure(value=patch(trainer)):\n"
            "    pass\n"
            "from nvflare.client.lightning import patch\n",
            id="lightning",
        ),
    ],
)
def test_inspect_does_not_finalize_definition_time_patch_before_later_import(tmp_path, source):
    script = tmp_path / "client.py"
    script.write_text(source, encoding="utf-8")

    data = inspect_path(script)

    assert data["conversion_state"] == "partial_client_api"


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            "from transformers import Trainer, TrainingArguments\n"
            "trainer = Trainer(model=model, args=TrainingArguments(output_dir='outputs'))\n"
            "class Config:\n"
            "    converted = patch(trainer)\n"
            "    from nvflare.client.hf import patch\n",
            id="huggingface",
        ),
        pytest.param(
            "import lightning as L\n"
            "trainer = L.Trainer(max_epochs=1)\n"
            "class Config:\n"
            "    converted = patch(trainer)\n"
            "    from nvflare.client.lightning import patch\n",
            id="lightning",
        ),
    ],
)
def test_inspect_does_not_finalize_class_body_patch_before_later_import(tmp_path, source):
    script = tmp_path / "client.py"
    script.write_text(source, encoding="utf-8")

    data = inspect_path(script)

    assert data["conversion_state"] == "partial_client_api"


@pytest.mark.parametrize(
    ("source", "expected_framework", "expected_skill"),
    [
        pytest.param(
            "from transformers import Trainer, TrainingArguments\n"
            "from nvflare.client.hf import patch\n"
            "trainer = Trainer(model=model, args=TrainingArguments(output_dir='outputs'))\n"
            "class patch:\n"
            "    def run(self):\n"
            "        return patch(trainer)\n"
            "trainer.train()\n",
            "huggingface",
            "nvflare-convert-huggingface",
            id="huggingface-method",
        ),
        pytest.param(
            "import lightning as L\n"
            "from nvflare.client.lightning import patch\n"
            "trainer = L.Trainer(max_epochs=1)\n"
            "class patch:\n"
            "    async def run(self):\n"
            "        return patch(trainer)\n"
            "trainer.fit(model)\n",
            "pytorch_lightning",
            "nvflare-convert-lightning",
            id="lightning-async-method",
        ),
        pytest.param(
            "from transformers import Trainer, TrainingArguments\n"
            "from nvflare.client.hf import patch\n"
            "trainer = Trainer(model=model, args=TrainingArguments(output_dir='outputs'))\n"
            "class patch:\n"
            "    run = lambda self: patch(trainer)\n"
            "trainer.train()\n",
            "huggingface",
            "nvflare-convert-huggingface",
            id="huggingface-lambda",
        ),
    ],
)
def test_inspect_class_callable_body_uses_post_definition_binding(tmp_path, source, expected_framework, expected_skill):
    script = tmp_path / "client.py"
    script.write_text(source, encoding="utf-8")

    data = inspect_path(script)

    assert data["frameworks"][0]["name"] == expected_framework
    assert data["conversion_state"] == "partial_client_api"
    assert data["skill_selection"]["recommended_skills"] == [expected_skill]


@pytest.mark.parametrize(
    ("source", "framework", "evidence_kind", "expected_values"),
    [
        pytest.param(
            "import lightning as L\n"
            "class L:\n"
            "    trainer = L.Trainer(max_epochs=1)\n"
            "    def build(self):\n"
            "        return L.Trainer(max_epochs=1)\n",
            "pytorch_lightning",
            "lightning_trainer",
            ["L.Trainer"],
            id="lightning",
        ),
        pytest.param(
            "from torch.optim import SGD\n"
            "class SGD:\n"
            "    optimizer = SGD(params)\n"
            "    def build(self):\n"
            "        return SGD(params)\n",
            "pytorch",
            "pytorch_call",
            ["SGD"],
            id="pytorch",
        ),
    ],
)
def test_inspect_class_immediate_and_deferred_bodies_use_correct_framework_bindings(
    tmp_path, source, framework, evidence_kind, expected_values
):
    script = tmp_path / "train.py"
    script.write_text(source, encoding="utf-8")

    data = inspect_path(script)
    evidence = next(item for item in data["frameworks"] if item["name"] == framework)["evidence"]

    assert [item["value"] for item in evidence if item["kind"] == evidence_kind] == expected_values


def test_inspect_method_resolves_enclosing_huggingface_trainer_subclass(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "from transformers import Trainer\n"
        "class CustomTrainer(Trainer):\n"
        "    def clone(self):\n"
        "        return CustomTrainer(model=model, args=args)\n",
        encoding="utf-8",
    )

    data = inspect_path(script)
    huggingface = next(item for item in data["frameworks"] if item["name"] == "huggingface")

    assert any(
        item["kind"] == "huggingface_trainer" and item["value"] == "CustomTrainer" for item in huggingface["evidence"]
    )


@pytest.mark.parametrize(
    ("source", "expected_framework", "expected_state"),
    [
        pytest.param(
            "from transformers import Trainer, TrainingArguments\n"
            "from nvflare.client.hf import patch\n"
            "trainer = Trainer(model=model, args=TrainingArguments(output_dir='outputs'))\n"
            "class patch:\n"
            "    converted = (lambda: patch(trainer))()\n"
            "trainer.train()\n",
            "huggingface",
            "client_api_converted",
            id="huggingface-immediate-lambda",
        ),
        pytest.param(
            "import lightning as L\n"
            "from nvflare.client.lightning import patch\n"
            "trainer = L.Trainer(max_epochs=1)\n"
            "class patch:\n"
            "    @(lambda method: (patch(trainer), method)[1])\n"
            "    def run(self):\n"
            "        pass\n"
            "trainer.fit(model)\n",
            "pytorch_lightning",
            "client_api_converted",
            id="lightning-lambda-decorator",
        ),
        pytest.param(
            "from transformers import Trainer, TrainingArguments\n"
            "from nvflare.client.hf import patch\n"
            "trainer = Trainer(model=model, args=TrainingArguments(output_dir='outputs'))\n"
            "class patch:\n"
            "    class Inner:\n"
            "        def run(self):\n"
            "            return patch(trainer)\n"
            "trainer.train()\n",
            "huggingface",
            "partial_client_api",
            id="huggingface-nested-class-method",
        ),
        pytest.param(
            "import lightning as L\n"
            "from nvflare.client.lightning import patch\n"
            "trainer = L.Trainer(max_epochs=1)\n"
            "class patch:\n"
            "    class Inner:\n"
            "        async def run(self):\n"
            "            return patch(trainer)\n"
            "trainer.fit(model)\n",
            "pytorch_lightning",
            "partial_client_api",
            id="lightning-nested-class-method",
        ),
        pytest.param(
            "from transformers import Trainer, TrainingArguments\n"
            "from nvflare.client.hf import patch\n"
            "trainer = Trainer(model=model, args=TrainingArguments(output_dir='outputs'))\n"
            "class patch:\n"
            "    callbacks = [lambda: patch(trainer) for _ in range(1)]\n"
            "trainer.train()\n",
            "huggingface",
            "partial_client_api",
            id="huggingface-comprehension-lambda",
        ),
        pytest.param(
            "import lightning as L\n"
            "from nvflare.client.lightning import patch\n"
            "trainer = L.Trainer(max_epochs=1)\n"
            "class patch:\n"
            "    callbacks = [lambda: patch(trainer) for _ in range(1)]\n"
            "trainer.fit(model)\n",
            "pytorch_lightning",
            "partial_client_api",
            id="lightning-comprehension-lambda",
        ),
    ],
)
def test_inspect_class_callable_uses_execution_phase_binding(tmp_path, source, expected_framework, expected_state):
    script = tmp_path / "client.py"
    script.write_text(source, encoding="utf-8")

    data = inspect_path(script)

    assert data["frameworks"][0]["name"] == expected_framework
    assert data["conversion_state"] == expected_state


@pytest.mark.parametrize(
    ("source", "expected_framework", "expected_state"),
    [
        pytest.param(
            "from transformers import Trainer, TrainingArguments\n"
            "from nvflare.client.hf import patch\n"
            "trainer = Trainer(model=model, args=TrainingArguments(output_dir='outputs'))\n"
            "class patch:\n"
            "    def convert():\n"
            "        return patch(trainer)\n"
            "    converted = convert()\n"
            "trainer.train()\n",
            "huggingface",
            "client_api_converted",
            id="huggingface-named-class-call",
        ),
        pytest.param(
            "import lightning as L\n"
            "from nvflare.client.lightning import patch\n"
            "trainer = L.Trainer(max_epochs=1)\n"
            "class patch:\n"
            "    def convert():\n"
            "        return patch(trainer)\n"
            "    converted = convert()\n"
            "trainer.fit(model)\n",
            "pytorch_lightning",
            "client_api_converted",
            id="lightning-named-class-call",
        ),
        pytest.param(
            "from transformers import Trainer, TrainingArguments\n"
            "from nvflare.client.hf import patch\n"
            "trainer = Trainer(model=model, args=TrainingArguments(output_dir='outputs'))\n"
            "class patch:\n"
            "    def converted(method):\n"
            "        patch(trainer)\n"
            "        return method\n"
            "    @converted\n"
            "    def run(self):\n"
            "        pass\n"
            "trainer.train()\n",
            "huggingface",
            "client_api_converted",
            id="huggingface-named-decorator",
        ),
        pytest.param(
            "import lightning as L\n"
            "from nvflare.client.lightning import patch\n"
            "trainer = L.Trainer(max_epochs=1)\n"
            "class patch:\n"
            "    def converted(method):\n"
            "        patch(trainer)\n"
            "        return method\n"
            "    @converted\n"
            "    def run(self):\n"
            "        pass\n"
            "trainer.fit(model)\n",
            "pytorch_lightning",
            "client_api_converted",
            id="lightning-named-decorator",
        ),
        pytest.param(
            "from transformers import Trainer, TrainingArguments\n"
            "from nvflare.client.hf import patch\n"
            "trainer = Trainer(model=model, args=TrainingArguments(output_dir='outputs'))\n"
            "class patch:\n"
            "    convert = lambda: patch(trainer)\n"
            "    converted = convert()\n"
            "trainer.train()\n",
            "huggingface",
            "client_api_converted",
            id="huggingface-lambda-alias",
        ),
        pytest.param(
            "import lightning as L\n"
            "from nvflare.client.lightning import patch\n"
            "trainer = L.Trainer(max_epochs=1)\n"
            "class patch:\n"
            "    convert = lambda: patch(trainer)\n"
            "    converted = convert()\n"
            "trainer.fit(model)\n",
            "pytorch_lightning",
            "client_api_converted",
            id="lightning-lambda-alias",
        ),
        pytest.param(
            "from transformers import Trainer, TrainingArguments\n"
            "from nvflare.client.hf import patch\n"
            "trainer = Trainer(model=model, args=TrainingArguments(output_dir='outputs'))\n"
            "class patch:\n"
            "    def convert():\n"
            "        return patch(trainer)\n"
            "patch.convert()\n"
            "trainer.train()\n",
            "huggingface",
            "partial_client_api",
            id="huggingface-later-method-call",
        ),
        pytest.param(
            "import lightning as L\n"
            "from nvflare.client.lightning import patch\n"
            "trainer = L.Trainer(max_epochs=1)\n"
            "class patch:\n"
            "    def convert():\n"
            "        return patch(trainer)\n"
            "patch.convert()\n"
            "trainer.fit(model)\n",
            "pytorch_lightning",
            "partial_client_api",
            id="lightning-later-method-call",
        ),
    ],
)
def test_inspect_named_class_callable_uses_invocation_phase_binding(
    tmp_path, source, expected_framework, expected_state
):
    script = tmp_path / "client.py"
    script.write_text(source, encoding="utf-8")

    data = inspect_path(script)

    assert data["frameworks"][0]["name"] == expected_framework
    assert data["conversion_state"] == expected_state


@pytest.mark.parametrize(
    ("framework", "source_prefix", "activity"),
    [
        pytest.param(
            "huggingface",
            "from transformers import Trainer, TrainingArguments\n"
            "from nvflare.client.hf import patch\n"
            "trainer = Trainer(model=model, args=TrainingArguments(output_dir='outputs'))\n",
            "trainer.train()\n",
            id="huggingface",
        ),
        pytest.param(
            "pytorch_lightning",
            "import lightning as L\n"
            "from nvflare.client.lightning import patch\n"
            "trainer = L.Trainer(max_epochs=1)\n",
            "trainer.fit(model)\n",
            id="lightning",
        ),
    ],
)
@pytest.mark.parametrize(
    "class_body",
    [
        pytest.param(
            "class patch:\n"
            "    async def convert():\n"
            "        return patch(trainer)\n"
            "    converted = convert()\n",
            id="coroutine",
        ),
        pytest.param(
            "class patch:\n" "    def convert():\n" "        yield patch(trainer)\n" "    converted = convert()\n",
            id="generator",
        ),
    ],
)
def test_inspect_lazy_class_callable_uses_post_class_binding(tmp_path, framework, source_prefix, activity, class_body):
    script = tmp_path / "client.py"
    script.write_text(source_prefix + class_body + activity, encoding="utf-8")

    data = inspect_path(script)

    assert data["frameworks"][0]["name"] == framework
    assert data["conversion_state"] == "partial_client_api"


@pytest.mark.parametrize(
    ("framework", "source_prefix", "activity"),
    [
        pytest.param(
            "huggingface",
            "from transformers import Trainer, TrainingArguments\n"
            "from nvflare.client.hf import patch\n"
            "trainer = Trainer(model=model, args=TrainingArguments(output_dir='outputs'))\n",
            "trainer.train()\n",
            id="huggingface",
        ),
        pytest.param(
            "pytorch_lightning",
            "import lightning as L\n"
            "from nvflare.client.lightning import patch\n"
            "trainer = L.Trainer(max_epochs=1)\n",
            "trainer.fit(model)\n",
            id="lightning",
        ),
    ],
)
@pytest.mark.parametrize(
    "class_body",
    [
        pytest.param(
            "class patch:\n"
            "    def convert():\n"
            "        return patch(trainer)\n"
            "    alias = convert\n"
            "    converted = alias()\n",
            id="function-alias",
        ),
        pytest.param(
            "class patch:\n"
            "    convert = lambda: patch(trainer)\n"
            "    alias = convert\n"
            "    converted = alias()\n",
            id="lambda-alias",
        ),
        pytest.param(
            "class patch:\n"
            "    def convert(method):\n"
            "        patch(trainer)\n"
            "        return method\n"
            "    alias = convert\n"
            "    @alias\n"
            "    def run(self):\n"
            "        pass\n",
            id="decorator-alias",
        ),
        pytest.param(
            "class patch:\n" "    converted = (convert := lambda: patch(trainer))()\n",
            id="walrus-lambda",
        ),
    ],
)
def test_inspect_class_callable_alias_preserves_identity(tmp_path, framework, source_prefix, activity, class_body):
    script = tmp_path / "client.py"
    script.write_text(source_prefix + class_body + activity, encoding="utf-8")

    data = inspect_path(script)

    assert data["frameworks"][0]["name"] == framework
    assert data["conversion_state"] == "client_api_converted"


@pytest.mark.parametrize(
    ("framework", "source_prefix", "activity"),
    [
        pytest.param(
            "huggingface",
            "from transformers import Trainer, TrainingArguments\n"
            "from nvflare.client.hf import patch\n"
            "trainer = Trainer(model=model, args=TrainingArguments(output_dir='outputs'))\n",
            "trainer.train()\n",
            id="huggingface",
        ),
        pytest.param(
            "pytorch_lightning",
            "import lightning as L\n"
            "from nvflare.client.lightning import patch\n"
            "trainer = L.Trainer(max_epochs=1)\n",
            "trainer.fit(model)\n",
            id="lightning",
        ),
    ],
)
@pytest.mark.parametrize(
    "class_body",
    [
        pytest.param(
            "class patch:\n"
            "    def decorate(method):\n"
            "        def wrapper():\n"
            "            return patch(trainer)\n"
            "        return wrapper\n"
            "    @decorate\n"
            "    def convert():\n"
            "        pass\n"
            "    converted = convert()\n",
            id="decorator-wrapper",
        ),
        pytest.param(
            "class patch:\n"
            "    def decorate(method):\n"
            "        return lambda: patch(trainer)\n"
            "    @decorate\n"
            "    def convert():\n"
            "        pass\n"
            "    converted = convert()\n",
            id="decorator-direct-lambda",
        ),
        pytest.param(
            "class patch:\n"
            "    def convert():\n"
            "        return patch(trainer)\n"
            "    convert = staticmethod(convert)\n"
            "    converted = convert()\n",
            id="explicit-staticmethod",
        ),
        pytest.param(
            "class patch:\n"
            "    def decorate(method):\n"
            "        return method\n"
            "    @decorate\n"
            "    def convert():\n"
            "        return patch(trainer)\n"
            "    converted = convert()\n",
            id="identity-decorator",
        ),
    ],
)
def test_inspect_decorated_class_callable_preserves_bound_result(
    tmp_path, framework, source_prefix, activity, class_body
):
    script = tmp_path / "client.py"
    script.write_text(source_prefix + class_body + activity, encoding="utf-8")

    data = inspect_path(script)

    assert data["frameworks"][0]["name"] == framework
    assert data["conversion_state"] == "client_api_converted"


@pytest.mark.parametrize(
    ("framework", "source_prefix", "activity"),
    [
        pytest.param(
            "huggingface",
            "from transformers import Trainer, TrainingArguments\n"
            "from nvflare.client.hf import patch\n"
            "trainer = Trainer(model=model, args=TrainingArguments(output_dir='outputs'))\n",
            "trainer.train()\n",
            id="huggingface",
        ),
        pytest.param(
            "pytorch_lightning",
            "import lightning as L\n"
            "from nvflare.client.lightning import patch\n"
            "trainer = L.Trainer(max_epochs=1)\n",
            "trainer.fit(model)\n",
            id="lightning",
        ),
    ],
)
@pytest.mark.parametrize(
    "consumer",
    [
        pytest.param("converted = list(convert())", id="list"),
        pytest.param("converted = tuple(convert())", id="tuple"),
        pytest.param("converted = set(convert())", id="set"),
        pytest.param("converted = dict(convert())", id="dict"),
        pytest.param("converted = next(convert())", id="next"),
        pytest.param("for converted in convert():\n        pass", id="for"),
        pytest.param("converted = [item for item in convert()]", id="comprehension"),
    ],
)
def test_inspect_eager_generator_consumer_uses_class_construction_binding(
    tmp_path, framework, source_prefix, activity, consumer
):
    script = tmp_path / "client.py"
    script.write_text(
        source_prefix
        + "class patch:\n"
        + "    def convert():\n"
        + "        yield ('model', patch(trainer))\n"
        + f"    {consumer}\n"
        + activity,
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["frameworks"][0]["name"] == framework
    assert data["conversion_state"] == "client_api_converted"


@pytest.mark.parametrize(
    ("framework", "source_prefix", "activity"),
    [
        pytest.param(
            "huggingface",
            "from transformers import Trainer, TrainingArguments\n"
            "from nvflare.client.hf import patch\n"
            "trainer = Trainer(model=model, args=TrainingArguments(output_dir='outputs'))\n",
            "trainer.train()\n",
            id="huggingface",
        ),
        pytest.param(
            "pytorch_lightning",
            "import lightning as L\n"
            "from nvflare.client.lightning import patch\n"
            "trainer = L.Trainer(max_epochs=1)\n",
            "trainer.fit(model)\n",
            id="lightning",
        ),
    ],
)
@pytest.mark.parametrize(
    ("consumer", "expected_state"),
    [
        pytest.param("pending = convert()\n    converted = list(pending)", "client_api_converted", id="assigned"),
        pytest.param(
            "pending: object = convert()\n    converted = tuple(pending)", "client_api_converted", id="annotated"
        ),
        pytest.param(
            "pending = alias = convert()\n    converted = set(alias)",
            "client_api_converted",
            id="chained",
        ),
        pytest.param(
            "pending = convert()\n    alias = pending\n    converted = list(alias)",
            "client_api_converted",
            id="alias",
        ),
        pytest.param("converted = list(pending := convert())", "client_api_converted", id="walrus"),
        pytest.param(
            "pending = convert()\n    pending = None\n    converted = list(pending)",
            "partial_client_api",
            id="invalidated",
        ),
        pytest.param("pending = convert()", "partial_client_api", id="stored-unused"),
    ],
)
def test_inspect_stored_generator_consumer_preserves_lazy_identity(
    tmp_path, framework, source_prefix, activity, consumer, expected_state
):
    script = tmp_path / "client.py"
    script.write_text(
        source_prefix
        + "class patch:\n"
        + "    def convert():\n"
        + "        yield ('model', patch(trainer))\n"
        + f"    {consumer}\n"
        + activity,
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["frameworks"][0]["name"] == framework
    assert data["conversion_state"] == expected_state


@pytest.mark.parametrize(
    ("framework", "source_prefix", "activity"),
    [
        pytest.param(
            "huggingface",
            "from transformers import Trainer, TrainingArguments\n"
            "from nvflare.client.hf import patch\n"
            "trainer = Trainer(model=model, args=TrainingArguments(output_dir='outputs'))\n",
            "trainer.train()\n",
            id="huggingface",
        ),
        pytest.param(
            "pytorch_lightning",
            "import lightning as L\n"
            "from nvflare.client.lightning import patch\n"
            "trainer = L.Trainer(max_epochs=1)\n",
            "trainer.fit(model)\n",
            id="lightning",
        ),
    ],
)
@pytest.mark.parametrize(
    "consumer",
    [
        pytest.param("first, second = convert()", id="direct-assignment-unpack"),
        pytest.param("pending = convert()\n    first, second = pending", id="stored-assignment-unpack"),
        pytest.param("consume(*convert())", id="starred-call"),
        pytest.param("converted = [*convert()]", id="starred-list"),
        pytest.param("converted = (*convert(),)", id="starred-tuple"),
        pytest.param("converted = {*convert()}", id="starred-set"),
    ],
)
def test_inspect_generator_unpacking_uses_class_construction_binding(
    tmp_path, framework, source_prefix, activity, consumer
):
    script = tmp_path / "client.py"
    script.write_text(
        source_prefix
        + "class patch:\n"
        + "    def convert():\n"
        + "        yield patch(trainer)\n"
        + "        yield None\n"
        + f"    {consumer}\n"
        + activity,
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["frameworks"][0]["name"] == framework
    assert data["conversion_state"] == "client_api_converted"


def test_deferred_class_callable_snapshots_only_eager_consumer_names(monkeypatch, tmp_path):
    retained_name_counts = []
    original_defer = inspector_module._PythonInspector._defer_class_callable_body

    def record_deferred_body(self, kind, node):
        deferred_body = original_defer(self, kind, node)
        if deferred_body:
            retained_name_counts.append(sum(len(frame) for frame in deferred_body.bound_name_stack))
        return deferred_body

    monkeypatch.setattr(inspector_module._PythonInspector, "_defer_class_callable_body", record_deferred_body)
    script = tmp_path / "generated.py"
    module_bindings = [f"name_{index} = None" for index in range(300)]
    methods = [f"    def method_{index}():\n        return None" for index in range(300)]
    script.write_text("\n".join(module_bindings + ["list = object()", "class Generated:"] + methods), encoding="utf-8")

    inspect_path(script)

    assert len(retained_name_counts) == 300
    assert max(retained_name_counts) == 1


@pytest.mark.parametrize(
    ("framework", "source_prefix", "activity"),
    [
        pytest.param(
            "huggingface",
            "from transformers import Trainer, TrainingArguments\n"
            "from nvflare.client.hf import patch\n"
            "trainer = Trainer(model=model, args=TrainingArguments(output_dir='outputs'))\n",
            "trainer.train()\n",
            id="huggingface",
        ),
        pytest.param(
            "pytorch_lightning",
            "import lightning as L\n"
            "from nvflare.client.lightning import patch\n"
            "trainer = L.Trainer(max_epochs=1)\n",
            "trainer.fit(model)\n",
            id="lightning",
        ),
    ],
)
@pytest.mark.parametrize(
    "consumer",
    [
        pytest.param("converted = dict(value=convert())", id="dict-keyword"),
        pytest.param("converted = next(iter(()), convert())", id="next-default"),
        pytest.param(
            "def keep(value):\n        return value\n    list = keep\n    converted = list(convert())",
            id="shadowed-list",
        ),
    ],
)
def test_inspect_eager_generator_consumer_does_not_consume_stored_arguments(
    tmp_path, framework, source_prefix, activity, consumer
):
    script = tmp_path / "client.py"
    script.write_text(
        source_prefix
        + "class patch:\n"
        + "    def convert():\n"
        + "        yield ('model', patch(trainer))\n"
        + f"    {consumer}\n"
        + activity,
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["frameworks"][0]["name"] == framework
    assert data["conversion_state"] == "partial_client_api"


@pytest.mark.parametrize(
    "assignment",
    [
        pytest.param("Trainer = trainer = Trainer(model=model, args=args)", id="direct-symbol-first"),
        pytest.param("trainer = Trainer = Trainer(model=model, args=args)", id="direct-symbol-last"),
        pytest.param("tf = trainer = tf.Trainer(model=model, args=args)", id="module-alias-first"),
        pytest.param("trainer = tf = tf.Trainer(model=model, args=args)", id="module-alias-last"),
        pytest.param(
            "trainer = Trainer(model=(Trainer := replacement), args=args)",
            id="constructor-rebound-in-argument",
        ),
    ],
)
def test_inspect_chained_assignment_preserves_huggingface_rhs_provenance(tmp_path, assignment):
    script = tmp_path / "train.py"
    script.write_text(
        "import transformers as tf\n" "from transformers import Trainer\n" f"{assignment}\n" "trainer.train()\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["frameworks"][0]["name"] == "huggingface"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-huggingface"]


@pytest.mark.parametrize(
    ("framework", "source_prefix", "call_target", "expected_state"),
    [
        pytest.param(
            "huggingface",
            "from transformers import Trainer, TrainingArguments\n"
            "from nvflare.client.hf import patch\n"
            "trainer = Trainer(model=model, args=TrainingArguments(output_dir='outputs'))\n",
            "outputs[patch(trainer)], patch",
            "client_api_converted",
            id="huggingface-call-before-bind",
        ),
        pytest.param(
            "huggingface",
            "from transformers import Trainer, TrainingArguments\n"
            "from nvflare.client.hf import patch\n"
            "trainer = Trainer(model=model, args=TrainingArguments(output_dir='outputs'))\n",
            "patch, outputs[patch(trainer)]",
            "partial_client_api",
            id="huggingface-bind-before-call",
        ),
        pytest.param(
            "pytorch_lightning",
            "import lightning as L\n"
            "from nvflare.client.lightning import patch\n"
            "trainer = L.Trainer(max_epochs=1)\n",
            "outputs[patch(trainer)], patch",
            "client_api_converted",
            id="lightning-call-before-bind",
        ),
        pytest.param(
            "pytorch_lightning",
            "import lightning as L\n"
            "from nvflare.client.lightning import patch\n"
            "trainer = L.Trainer(max_epochs=1)\n",
            "patch, outputs[patch(trainer)]",
            "partial_client_api",
            id="lightning-bind-before-call",
        ),
    ],
)
def test_inspect_with_unpacking_resolves_patch_in_target_assignment_order(
    tmp_path, framework, source_prefix, call_target, expected_state
):
    script = tmp_path / "client.py"
    script.write_text(
        source_prefix + f"with manager() as ({call_target}):\n" "    pass\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["frameworks"][0]["name"] == framework
    assert data["conversion_state"] == expected_state


@pytest.mark.parametrize(
    ("call_target", "expected_calls"),
    [
        pytest.param("outputs[SGD(params)], SGD", ["SGD"], id="call-before-bind"),
        pytest.param("SGD, outputs[SGD(params)]", [], id="bind-before-call"),
    ],
)
def test_inspect_with_unpacking_resolves_pytorch_call_in_target_assignment_order(tmp_path, call_target, expected_calls):
    script = tmp_path / "train.py"
    script.write_text(
        "from torch.optim import SGD\n" f"with manager() as ({call_target}):\n" "    pass\n",
        encoding="utf-8",
    )

    data = inspect_path(script)
    pytorch = next(item for item in data["frameworks"] if item["name"] == "pytorch")

    assert [item["value"] for item in pytorch["evidence"] if item["kind"] == "pytorch_call"] == expected_calls


@pytest.mark.parametrize(
    ("assignment_target", "expected_state"),
    [
        pytest.param("outputs[patch(trainer)], patch", "client_api_converted", id="call-before-bind"),
        pytest.param("patch, outputs[patch(trainer)]", "partial_client_api", id="bind-before-call"),
    ],
)
def test_inspect_assignment_unpacking_resolves_patch_in_target_assignment_order(
    tmp_path, assignment_target, expected_state
):
    script = tmp_path / "client.py"
    script.write_text(
        "from transformers import Trainer, TrainingArguments\n"
        "from nvflare.client.hf import patch\n"
        "trainer = Trainer(model=model, args=TrainingArguments(output_dir='outputs'))\n"
        f"{assignment_target} = values\n"
        "trainer.train()\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["conversion_state"] == expected_state


@pytest.mark.parametrize(
    ("loop_target", "expected_calls"),
    [
        pytest.param("outputs[SGD(params)], SGD", ["SGD"], id="call-before-bind"),
        pytest.param("SGD, outputs[SGD(params)]", [], id="bind-before-call"),
    ],
)
def test_inspect_for_unpacking_resolves_pytorch_call_in_target_assignment_order(tmp_path, loop_target, expected_calls):
    script = tmp_path / "train.py"
    script.write_text(
        "from torch.optim import SGD\n" f"for {loop_target} in values:\n" "    pass\n",
        encoding="utf-8",
    )

    data = inspect_path(script)
    pytorch = next(item for item in data["frameworks"] if item["name"] == "pytorch")

    assert [item["value"] for item in pytorch["evidence"] if item["kind"] == "pytorch_call"] == expected_calls


@pytest.mark.parametrize(
    "pattern",
    [
        pytest.param("[trainer]", id="match-as"),
        pytest.param("[*trainer]", id="match-star"),
        pytest.param("{'value': _, **trainer}", id="match-mapping-rest"),
    ],
)
def test_inspect_does_not_reuse_trainer_identity_shadowed_by_match_capture(tmp_path, pattern):
    script = tmp_path / "train.py"
    script.write_text(
        "from transformers import Trainer, TrainingArguments\n"
        "trainer = Trainer(model=model, args=TrainingArguments(output_dir='outputs'))\n"
        "def run(value):\n"
        "    match value:\n"
        f"        case {pattern}:\n"
        "            trainer.train()\n",
        encoding="utf-8",
    )

    data = inspect_path(script)
    huggingface = next(item for item in data["frameworks"] if item["name"] == "huggingface")

    assert not any(item["kind"] == "huggingface_train" for item in huggingface["evidence"])
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-orient"]


def test_inspect_visits_calls_in_with_assignment_target(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "from torch.optim import SGD\n" "with manager() as outputs[SGD(params)]:\n" "    pass\n",
        encoding="utf-8",
    )

    data = inspect_path(script)
    pytorch = next(item for item in data["frameworks"] if item["name"] == "pytorch")

    assert [item["value"] for item in pytorch["evidence"] if item["kind"] == "pytorch_call"] == ["SGD"]


def test_inspect_resolves_pytorch_default_before_parameter_and_body_scope(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "from torch.optim import SGD\n" "def configure(SGD=SGD(params)):\n" "    SGD = LocalOptimizer\n",
        encoding="utf-8",
    )

    data = inspect_path(script)
    pytorch = next(item for item in data["frameworks"] if item["name"] == "pytorch")

    assert [item["value"] for item in pytorch["evidence"] if item["kind"] == "pytorch_call"] == ["SGD"]


def test_inspect_resolves_lightning_default_before_parameter_and_body_scope(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "import lightning as L\n" "def configure(L=L.Trainer(max_epochs=1)):\n" "    L = local_lightning\n",
        encoding="utf-8",
    )

    data = inspect_path(script)
    lightning = next(item for item in data["frameworks"] if item["name"] == "pytorch_lightning")

    assert [item["value"] for item in lightning["evidence"] if item["kind"] == "lightning_trainer"] == ["L.Trainer"]


@pytest.mark.parametrize(
    ("source", "framework", "kind", "expected_value"),
    [
        pytest.param(
            "from torch.optim import SGD\n"
            "class Config:\n"
            "    optimizer = SGD(params)\n"
            "    SGD = LocalOptimizer\n"
            "    other = SGD(params)\n",
            "pytorch",
            "pytorch_call",
            "SGD",
            id="pytorch",
        ),
        pytest.param(
            "import lightning as L\n"
            "class Config:\n"
            "    trainer = L.Trainer(max_epochs=1)\n"
            "    L = local_lightning\n"
            "    other = L.Trainer(max_epochs=1)\n",
            "pytorch_lightning",
            "lightning_trainer",
            "L.Trainer",
            id="lightning",
        ),
    ],
)
def test_inspect_class_body_framework_bindings_are_sequential(tmp_path, source, framework, kind, expected_value):
    script = tmp_path / "train.py"
    script.write_text(source, encoding="utf-8")

    data = inspect_path(script)
    evidence = next(item for item in data["frameworks"] if item["name"] == framework)["evidence"]

    assert [item["value"] for item in evidence if item["kind"] == kind] == [expected_value]


def test_inspect_visits_nested_class_base_calls_in_enclosing_scope(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "import lightning as L\n" "class Config(factory(L.Trainer(max_epochs=1))):\n" "    L = local_lightning\n",
        encoding="utf-8",
    )

    data = inspect_path(script)
    lightning = next(item for item in data["frameworks"] if item["name"] == "pytorch_lightning")

    assert [item["value"] for item in lightning["evidence"] if item["kind"] == "lightning_trainer"] == ["L.Trainer"]


@pytest.mark.parametrize(
    ("module_import", "trainer_call"),
    [
        ("import pytorch_lightning.callbacks", "pytorch_lightning.Trainer(max_epochs=1)"),
        ("import lightning.pytorch.callbacks", "lightning.pytorch.Trainer(max_epochs=1)"),
    ],
)
def test_inspect_dotted_lightning_imports_establish_root_owner_identity(tmp_path, module_import, trainer_call):
    script = tmp_path / "train.py"
    script.write_text(
        f"{module_import}\n"
        "from transformers import Trainer, TrainingArguments\n"
        f"lightning_trainer = {trainer_call}\n"
        "hf_trainer = Trainer(model=model, args=TrainingArguments(output_dir='outputs'))\n"
        "hf_trainer.train()\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["skill_selection"]["recommended_skills"] == ["nvflare-orient"]


def test_inspect_aliased_lightning_submodule_does_not_claim_root_trainer(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "import pytorch_lightning.callbacks as callbacks\n" "callbacks.Trainer(max_epochs=1)\n",
        encoding="utf-8",
    )

    data = inspect_path(script)
    lightning = next(item for item in data["frameworks"] if item["name"] == "pytorch_lightning")

    assert not any(item["kind"] == "lightning_trainer" for item in lightning["evidence"])


def test_inspect_does_not_route_unconverted_nemo_wrapper_as_lightning(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "from nemo import lightning as nl\n" "\n" "trainer = nl.Trainer(max_steps=10)\n" "trainer.fit(model)\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["frameworks"] == []
    assert data["conversion_state"] == "unknown"
    assert data["skill_selection"]["recommended_skills"] == []


def test_inspect_keeps_plain_pytorch_routing_separate_from_lightning(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "import torch\n"
        "from torch.utils.data import DataLoader\n"
        "\n"
        "class Net(torch.nn.Module):\n"
        "    pass\n"
        "\n"
        "loader = DataLoader([])\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert [framework["name"] for framework in data["frameworks"]] == ["pytorch"]
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-pytorch"]


def test_inspect_mixed_pytorch_workspace_with_incidental_lightning_keeps_pytorch(tmp_path):
    # A plain PyTorch entry point plus incidental Lightning imports should
    # surface the mixed workspace without hiding the PyTorch training script,
    # even when the helper has more raw Lightning import evidence.
    (tmp_path / "train.py").write_text(
        "import torch\n" "\n" "class Net(torch.nn.Module):\n" "    pass\n" "\n" "def main():\n" "    model = Net()\n",
        encoding="utf-8",
    )
    (tmp_path / "optional_utils.py").write_text(
        "import pytorch_lightning\n"
        "import lightning.pytorch\n"
        "from pytorch_lightning.callbacks import ModelCheckpoint\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    framework_names = [framework["name"] for framework in data["frameworks"]]
    framework_by_name = {framework["name"]: framework for framework in data["frameworks"]}
    assert framework_names[0] == "pytorch"
    assert "pytorch_lightning" in framework_names
    assert framework_by_name["pytorch_lightning"]["confidence"] > framework_by_name["pytorch"]["confidence"]
    assert len(framework_by_name["pytorch_lightning"]["evidence"]) > len(framework_by_name["pytorch"]["evidence"])
    assert data["target_type"] == "mixed_framework_workspace"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-pytorch"]


def test_inspect_incidental_lightning_does_not_demote_ranked_pytorch(tmp_path):
    # When PyTorch already ranks ahead of Lightning, preserving that order keeps
    # unrelated frameworks from becoming the display primary.
    (tmp_path / "train.py").write_text(
        "import torch\n"
        "import torchvision\n"
        "import torchaudio\n"
        "\n"
        "class Net(torch.nn.Module):\n"
        "    pass\n"
        "\n"
        "def train():\n"
        "    return Net()\n",
        encoding="utf-8",
    )
    (tmp_path / "boost_helper.py").write_text(
        "import xgboost\n" "import xgboost as xgb\n",
        encoding="utf-8",
    )
    (tmp_path / "optional_lightning.py").write_text(
        "import pytorch_lightning\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    framework_names = [framework["name"] for framework in data["frameworks"]]
    assert framework_names[:3] == ["pytorch", "xgboost", "pytorch_lightning"]
    assert data["target_type"] == "mixed_framework_workspace"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-pytorch"]


# PyTorch workspaces with an unrelated or unreachable Lightning helper must
# keep PyTorch routing. Fields: files (relative path -> content) and
# expect_mixed_target (assert target_type == "mixed_framework_workspace",
# only where the original case asserted it).
_KEEPS_PYTORCH_DESPITE_LIGHTNING_HELPER_CASES = [
    # Active PyTorch evidence tied to the entry point keeps unrelated Lightning
    # helpers from taking over routing just because they have more active evidence.
    pytest.param(
        {
            "train.py": (
                "import torch\n"
                "\n"
                "class Net(torch.nn.Module):\n"
                "    pass\n"
                "\n"
                "def train():\n"
                "    return Net()\n"
            ),
            "lit_helper.py": (
                "import pytorch_lightning as pl\n"
                "\n"
                "class Helper(pl.LightningModule):\n"
                "    pass\n"
                "\n"
                "trainer = pl.Trainer(max_epochs=1)\n"
            ),
        },
        True,
        id="unrelated_active_lightning_helper_does_not_outweigh_pytorch_entry_point",
    ),
    pytest.param(
        {
            "train.py": "import torch\n" "\n" "class Net(torch.nn.Module):\n" "    pass\n",
            "lit_helper.py": (
                "import pytorch_lightning as pl\n" "\n" "class Helper(pl.LightningModule):\n" "    pass\n"
            ),
        },
        True,
        id="sparse_pytorch_entry_with_lightning_helper_class_keeps_pytorch",
    ),
    pytest.param(
        {
            "train.py": "import torch\n" "\n" "def main():\n" "    return None\n",
            "lit_helper.py": (
                "import pytorch_lightning as pl\n"
                "\n"
                "class Helper(pl.LightningModule):\n"
                "    pass\n"
                "\n"
                "trainer = pl.Trainer(max_epochs=1)\n"
            ),
        },
        True,
        id="pytorch_entry_import_with_unrelated_active_lightning_helper_keeps_pytorch",
    ),
    pytest.param(
        {
            "train.py": "def main():\n" "    return None\n",
            "model.py": "import torch\n" "\n" "class Net(torch.nn.Module):\n" "    pass\n",
            "lit_helper.py": (
                "import pytorch_lightning as pl\n"
                "\n"
                "class Helper(pl.LightningModule):\n"
                "    pass\n"
                "\n"
                "trainer = pl.Trainer(max_epochs=1)\n"
            ),
        },
        True,
        id="entry_point_blocks_unreachable_active_lightning_helper_from_fallback",
    ),
    pytest.param(
        {
            "models/train.py": (
                "import lightning.pytorch\n"
                "import torch\n"
                "from torch.utils.data import DataLoader\n"
                "\n"
                "def train():\n"
                "    return DataLoader([])\n"
            ),
            "models/lightning.py": (
                "import pytorch_lightning as pl\n" "\n" "class Helper(pl.LightningModule):\n" "    pass\n"
            ),
        },
        False,
        id="external_lightning_import_does_not_reach_local_lightning_file",
    ),
    pytest.param(
        {
            "models/train.py": (
                "import lightning.pytorch\n"
                "import torch\n"
                "from torch.utils.data import DataLoader\n"
                "\n"
                "def train():\n"
                "    return DataLoader([])\n"
            ),
            "models/lightning/__init__.py": (
                "import pytorch_lightning as pl\n" "\n" "class Helper(pl.LightningModule):\n" "    pass\n"
            ),
        },
        False,
        id="external_lightning_import_does_not_reach_local_lightning_package",
    ),
    pytest.param(
        {
            "train.py": (
                "import lightning.pytorch\n"
                "import torch\n"
                "from torch.utils.data import DataLoader\n"
                "\n"
                "def train():\n"
                "    return DataLoader([])\n"
            ),
            "lightning/__init__.py": (
                "import pytorch_lightning as pl\n" "\n" "class Helper(pl.LightningModule):\n" "    pass\n"
            ),
        },
        False,
        id="external_lightning_import_does_not_reach_top_level_lightning_package",
    ),
    pytest.param(
        {
            "experiment.py": (
                "import torch\n"
                "import torch.nn\n"
                "import torch.optim\n"
                "import torch.utils.data\n"
                "import torchaudio\n"
                "import torchvision\n"
                "\n"
                "DEFAULT_EPOCHS = 1\n"
            ),
            "lightning_helper.py": (
                "import pytorch_lightning as pl\n" "\n" "class Helper(pl.LightningModule):\n" "    pass\n"
            ),
        },
        False,
        id="unrelated_lightning_helper_does_not_beat_pytorch_import_heavy_workspace",
    ),
]


@pytest.mark.parametrize(("files", "expect_mixed_target"), _KEEPS_PYTORCH_DESPITE_LIGHTNING_HELPER_CASES)
def test_inspect_keeps_pytorch_despite_lightning_helper(tmp_path, files, expect_mixed_target):
    for rel_path, content in files.items():
        path = tmp_path / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    data = inspect_path(tmp_path)

    framework_names = [framework["name"] for framework in data["frameworks"]]
    assert framework_names[0] == "pytorch"
    assert "pytorch_lightning" in framework_names
    if expect_mixed_target:
        assert data["target_type"] == "mixed_framework_workspace"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-pytorch"]


def _lightning_module_and_trainer(alias):
    return (
        f"class Net({alias}.LightningModule):\n"
        "    def configure_optimizers(self):\n"
        "        return None\n"
        "def main():\n"
        f"    {alias}.Trainer(max_epochs=1).fit(Net())\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )


# Workspaces where the Lightning code is reachable from (or dominates) the
# entry context, so routing goes to the Lightning skill. Fields: files
# (relative path -> content), target (relative path passed to inspect_path,
# or None for the workspace root), and evidence_file (when set, assert a
# lightning_class evidence item from that file on the primary framework).
_ROUTES_TO_LIGHTNING_CASES = [
    # `from lightning import pytorch as pl` (Lightning 2.x form) alongside torch.
    pytest.param(
        {
            "train.py": "import torch\nimport torch.nn as nn\nfrom lightning import pytorch as pl\n"
            + _lightning_module_and_trainer("pl"),
        },
        None,
        None,
        id="from_lightning_import_pytorch_alias_routes_to_lightning",
    ),
    # `import lightning as L` then `L.pytorch.LightningModule` / `L.pytorch.Trainer`.
    pytest.param(
        {
            "train.py": (
                "import torch\nimport lightning as L\n"
                "class Net(L.pytorch.LightningModule):\n"
                "    def configure_optimizers(self):\n"
                "        return None\n"
                "def main():\n"
                "    L.pytorch.Trainer(max_epochs=1).fit(Net())\n"
                "if __name__ == '__main__':\n"
                "    main()\n"
            ),
        },
        None,
        None,
        id="bare_lightning_alias_pytorch_submodule_routes_to_lightning",
    ),
    # PyPA src-layout: entry imports mypkg.loop; the module lives at src/mypkg/loop.py.
    pytest.param(
        {
            "src/mypkg/loop.py": "import lightning.pytorch as pl\n" + _lightning_module_and_trainer("pl"),
            "train.py": (
                "import torch\nimport torch.nn as nn\nfrom mypkg.loop import Net\n"
                "def main():\n    return Net()\nif __name__ == '__main__':\n    main()\n"
            ),
        },
        None,
        None,
        id="src_layout_lightning_reachable_from_entry_routes_to_lightning",
    ),
    # Lightning model in a non-entry module + a torch import (no active torch use)
    # + an unrelated entry point must not default to the PyTorch base.
    pytest.param(
        {
            "litmodel.py": (
                "import torch\nimport lightning.pytorch as pl\n"
                "class LitNet(pl.LightningModule):\n"
                "    def configure_optimizers(self):\n"
                "        return None\n"
            ),
            "run.py": "import json\nif __name__ == '__main__':\n    print(json.dumps({}))\n",
        },
        None,
        None,
        id="dominant_lightning_module_with_unrelated_entry_routes_to_lightning",
    ),
    pytest.param(
        {
            "train.py": "import torch\n" "from model import LitModel\n" "\n" "def main():\n" "    return LitModel()\n",
            "model.py": "import pytorch_lightning as pl\n" "\n" "class LitModel(pl.LightningModule):\n" "    pass\n",
        },
        None,
        None,
        id="split_file_lightning_model_imported_by_entry_point_recommends_lightning",
    ),
    pytest.param(
        {
            "train.py": "import torch\n" "from models import LitModel\n" "\n" "def main():\n" "    return LitModel()\n",
            "models/__init__.py": (
                "import pytorch_lightning as pl\n" "\n" "class LitModel(pl.LightningModule):\n" "    pass\n"
            ),
        },
        None,
        None,
        id="package_lightning_model_imported_by_entry_point_recommends_lightning",
    ),
    pytest.param(
        {
            "train.py": "import torch\n" "from models import LitModel\n" "\n" "def main():\n" "    return LitModel()\n",
            "models/__init__.py": "from .model import LitModel\n",
            "models/model.py": (
                "import pytorch_lightning as pl\n" "\n" "class LitModel(pl.LightningModule):\n" "    pass\n"
            ),
        },
        None,
        None,
        id="package_reexported_lightning_model_with_torch_entry_import_recommends_lightning",
    ),
    pytest.param(
        {
            "train.py": (
                "import torch\n"
                "from torch.utils.data import DataLoader\n"
                "from models import LitModel\n"
                "\n"
                "def main():\n"
                "    loader = DataLoader([])\n"
                "    return LitModel(), loader\n"
            ),
            "models/__init__.py": "from .model import LitModel\n",
            "models/model.py": (
                "import pytorch_lightning as pl\n" "\n" "class LitModel(pl.LightningModule):\n" "    pass\n"
            ),
        },
        None,
        None,
        id="package_reexported_lightning_model_with_active_pytorch_entry_recommends_lightning",
    ),
    pytest.param(
        {
            "models/__init__.py": "",
            "train.py": (
                "import torch\n"
                "from torch.utils.data import DataLoader\n"
                "from models import lightning_model\n"
                "\n"
                "def main():\n"
                "    loader = DataLoader([])\n"
                "    return lightning_model.LitModel(), loader\n"
            ),
            "models/lightning_model.py": (
                "import pytorch_lightning as pl\n" "\n" "class LitModel(pl.LightningModule):\n" "    pass\n"
            ),
        },
        None,
        "models/lightning_model.py",
        id="package_lightning_submodule_imported_by_entry_point_recommends_lightning",
    ),
    pytest.param(
        {
            "train.py": "import torch\n" "from models import *\n" "\n" "def main():\n" "    return LitModel()\n",
            "models/__init__.py": (
                "import pytorch_lightning as pl\n" "\n" "class LitModel(pl.LightningModule):\n" "    pass\n"
            ),
        },
        None,
        None,
        id="package_star_import_can_reach_lightning_model",
    ),
    pytest.param(
        {
            "models/__init__.py": "",
            "models/train.py": (
                "import torch\n"
                "from torch.utils.data import DataLoader\n"
                "from . import model\n"
                "\n"
                "def main():\n"
                "    loader = DataLoader([])\n"
                "    return model.LitModel(), loader\n"
            ),
            "models/model.py": (
                "import pytorch_lightning as pl\n" "\n" "class LitModel(pl.LightningModule):\n" "    pass\n"
            ),
        },
        None,
        "models/model.py",
        id="relative_package_lightning_submodule_imported_by_entry_point_recommends_lightning",
    ),
    # A normal Lightning script imports several torch symbols, so PyTorch import
    # evidence outnumbers Lightning symbols. Lightning still wins.
    pytest.param(
        {
            "train.py": (
                "import torch\n"
                "from torch import nn\n"
                "from torch.utils.data import DataLoader\n"
                "import pytorch_lightning as pl\n"
                "\n"
                "class Net(pl.LightningModule):\n"
                "    pass\n"
                "\n"
                "trainer = pl.Trainer(max_epochs=1)\n"
            ),
        },
        "train.py",
        None,
        id="lightning_script_with_many_torch_imports_recommends_lightning",
    ),
    pytest.param(
        {
            "train.py": (
                "import torch\n"
                "from torch import nn\n"
                "from torch.utils.data import DataLoader\n"
                "import pytorch_lightning as pl\n"
                "\n"
                "class Net(pl.LightningModule):\n"
                "    pass\n"
            ),
        },
        "train.py",
        None,
        id="lightning_module_with_many_torch_imports_recommends_lightning",
    ),
    pytest.param(
        {
            "model.py": (
                "import torch\n"
                "import torch.nn as nn\n"
                "import torch.optim as optim\n"
                "import torchaudio\n"
                "import torchvision\n"
                "from torch import nn\n"
                "from torch.nn import functional as F\n"
                "from torch.optim import Adam\n"
                "from torch.utils.data import DataLoader\n"
                "import pytorch_lightning as pl\n"
                "\n"
                "class LitModel(pl.LightningModule):\n"
                "    pass\n"
            ),
        },
        None,
        None,
        id="lightning_model_file_with_many_torch_imports_recommends_lightning",
    ),
]


@pytest.mark.parametrize(("files", "target", "evidence_file"), _ROUTES_TO_LIGHTNING_CASES)
def test_inspect_routes_to_lightning(tmp_path, files, target, evidence_file):
    for rel_path, content in files.items():
        path = tmp_path / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    data = inspect_path(tmp_path / target if target else tmp_path)

    framework_names = [framework["name"] for framework in data["frameworks"]]
    assert framework_names[0] == "pytorch_lightning"
    assert "pytorch" in framework_names
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-lightning"]
    if evidence_file is not None:
        assert any(
            item["file"] == evidence_file and item["kind"] == "lightning_class"
            for item in data["frameworks"][0]["evidence"]
        )


def test_inspect_cross_family_confidence_tie_prefers_entry_context_framework(tmp_path):
    # sklearn entry point + a torch utility can tie on evidence count; a pure
    # alphabetical tie-break would pick pytorch and recommend the PyTorch skill.
    # The framework whose evidence is tied to the entry point (sklearn) wins, so
    # no conversion skill is recommended for the sklearn-dominant repo.
    (tmp_path / "train.py").write_text(
        "from sklearn.linear_model import LogisticRegression\n"
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.metrics import accuracy_score\n"
        "def main():\n"
        "    LogisticRegression()\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
        encoding="utf-8",
    )
    (tmp_path / "util.py").write_text(
        "import torch\nfrom torch.utils.data import DataLoader\ndef loader(ds):\n    return DataLoader(ds)\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    assert data["skill_selection"]["recommended_skills"] == []
    assert data["frameworks"][0]["name"] == "sklearn"


def test_inspect_higher_count_unreachable_torch_helper_does_not_beat_sklearn_entry(tmp_path):
    # Count-based confidence can rank an unreachable torch helper above the
    # sklearn the entry point actually uses. Reachability must win: the torch
    # helper is never imported from the entry point, so the sklearn-dominant repo
    # stays on sklearn and abstains from a (wrong) PyTorch recommendation.
    (tmp_path / "train.py").write_text(
        "from sklearn.linear_model import LogisticRegression\n"
        "from sklearn.model_selection import train_test_split\n"
        "def main():\n"
        "    LogisticRegression()\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
        encoding="utf-8",
    )
    # data.py is never imported by the entry point but has more torch evidence
    # (import + submodule import + call) than sklearn's two imports.
    (tmp_path / "data.py").write_text(
        "import torch\nfrom torch.utils.data import DataLoader\ndef loader(ds):\n    return DataLoader(ds)\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    confidence = {fw["name"]: fw["confidence"] for fw in data["frameworks"]}
    assert confidence["pytorch"] > confidence["sklearn"]  # torch helper has the higher raw count
    assert data["skill_selection"]["detected_framework"] == "sklearn"  # but entry-tied sklearn wins
    assert data["skill_selection"]["recommended_skills"] == []


def test_inspect_stale_src_layout_copy_does_not_steal_entry_reachability(tmp_path):
    # A src-layout copy (src/mypkg/loop.py) shares the stripped module name
    # "mypkg.loop" with an actively imported root-level mypkg/loop.py. The stale
    # copy (Lightning) must not be scored as entry-reachable via the shared name;
    # the entry point imports the root PyTorch module, so routing stays PyTorch.
    (tmp_path / "train.py").write_text(
        "from mypkg.loop import run\nif __name__ == '__main__':\n    run()\n",
        encoding="utf-8",
    )
    (tmp_path / "mypkg").mkdir()
    (tmp_path / "mypkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "mypkg" / "loop.py").write_text(
        "import torch\nimport torch.nn as nn\nclass Net(nn.Module):\n    pass\ndef run():\n    return Net()\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "mypkg").mkdir(parents=True)
    (tmp_path / "src" / "mypkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "src" / "mypkg" / "loop.py").write_text(
        "import lightning.pytorch as pl\nclass Lit(pl.LightningModule):\n    pass\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-pytorch"]


def test_inspect_src_layout_model_imported_by_root_entry_still_resolves(tmp_path):
    # Guard the src-layout fix does not over-correct: with no root-level
    # collision, an entry point that imports mypkg.loop must still reach the
    # src/mypkg/loop.py Lightning model and route to Lightning.
    (tmp_path / "train.py").write_text(
        "from mypkg.loop import Lit\nif __name__ == '__main__':\n    Lit()\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "mypkg").mkdir(parents=True)
    (tmp_path / "src" / "mypkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "src" / "mypkg" / "loop.py").write_text(
        "import lightning.pytorch as pl\nclass Lit(pl.LightningModule):\n    pass\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-lightning"]


def test_inspect_reachable_lightning_class_wins_over_co_located_torch(tmp_path):
    # DESIGN DECISION: a LightningModule reachable from the entry context routes
    # to the Lightning skill even when co-located with dominant plain-torch code.
    # This deliberately favors the common case (real Lightning projects compose
    # torch models/submodules) over the rare stray-leftover-LightningModule edge,
    # which is low-harm (a Lightning conversion still works). Mis-routing a real
    # Lightning repo to the PyTorch skill would be worse. See the rationale in
    # LightningDetector.promote_over_family. (Previously this asserted PyTorch;
    # the guard that produced that was intentionally removed.)
    (tmp_path / "model.py").write_text(
        "import torch\n"
        "import torch.nn as nn\n"
        "import pytorch_lightning as pl\n"
        "class LegacyLit(pl.LightningModule):\n"
        "    pass\n"
        "class Net(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.fc = nn.Linear(4, 2)\n"
        "    def forward(self, x):\n"
        "        return self.fc(x)\n"
        "def train():\n"
        "    net = Net()\n"
        "    opt = torch.optim.SGD(net.parameters(), lr=0.1)\n"
        "    loss = torch.nn.CrossEntropyLoss()\n"
        "    loader = torch.utils.data.DataLoader([])\n"
        "    return net, opt, loss, loader\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-lightning"]


def test_inspect_reachable_active_torch_model_beats_import_only_sklearn_entry(tmp_path):
    # #4: when the entry reaches BOTH import-only sklearn and an ACTIVE torch
    # model, prefer the framework with real (active) evidence -> recommend the
    # PyTorch conversion rather than abstaining on the sklearn imports.
    (tmp_path / "train.py").write_text(
        "from sklearn.linear_model import LogisticRegression\n"
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.metrics import accuracy_score\n"
        "from net import Net\n"
        "def main():\n"
        "    LogisticRegression()\n"
        "    Net()\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
        encoding="utf-8",
    )
    (tmp_path / "net.py").write_text(
        "import torch.nn as nn\nclass Net(nn.Module):\n    def forward(self, x):\n        return x\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    assert data["skill_selection"]["detected_framework"] == "pytorch"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-pytorch"]


def test_inspect_import_only_sklearn_entry_still_wins_when_torch_unreachable(tmp_path):
    # #4 control (preserves the earlier sklearn-entry decision): when the torch
    # helper is NOT reachable from the entry, the entry-owned sklearn stays
    # primary and no conversion skill is recommended.
    (tmp_path / "train.py").write_text(
        "from sklearn.linear_model import LogisticRegression\n"
        "from sklearn.model_selection import train_test_split\n"
        "def main():\n"
        "    LogisticRegression()\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
        encoding="utf-8",
    )
    (tmp_path / "data.py").write_text(
        "import torch\nfrom torch.utils.data import DataLoader\ndef loader(ds):\n    return DataLoader(ds)\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    assert data["skill_selection"]["detected_framework"] == "sklearn"
    assert data["skill_selection"]["recommended_skills"] == []


def test_inspect_frameworks_list_leads_with_detected_primary(tmp_path):
    # #5: frameworks[0] must match detected_framework even when a non-detected
    # framework has higher raw confidence (here incidental Lightning imports
    # outrank the entry-tied active PyTorch model by count).
    (tmp_path / "train.py").write_text(
        "import torch\nimport torch.nn as nn\nclass Net(nn.Module):\n    pass\n"
        "def main():\n    Net()\nif __name__ == '__main__':\n    main()\n",
        encoding="utf-8",
    )
    (tmp_path / "unused.py").write_text(
        "import pytorch_lightning\nimport pytorch_lightning.callbacks\nimport pytorch_lightning.loggers\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    detected = data["skill_selection"]["detected_framework"]
    assert detected == "pytorch"
    assert data["frameworks"][0]["name"] == detected


def test_inspect_ranks_on_full_evidence_beyond_display_cap(tmp_path):
    # #3: framework ranking uses the true evidence count, not the display cap of
    # 12. A file with more torch imports than a competing framework's imports
    # ranks PyTorch higher even when both exceed the display cap; the displayed
    # evidence list stays capped.
    torch_imports = "".join(f"import torch.pkg{i}\n" for i in range(20))
    sklearn_imports = "".join(f"import sklearn.pkg{i}\n" for i in range(13))
    (tmp_path / "a.py").write_text(torch_imports, encoding="utf-8")
    (tmp_path / "b.py").write_text(sklearn_imports, encoding="utf-8")

    data = inspect_path(tmp_path)

    confidence = {fw["name"]: fw["confidence"] for fw in data["frameworks"]}
    assert confidence["pytorch"] > confidence["sklearn"]  # 20 vs 13, not a 12-capped tie
    for fw in data["frameworks"]:
        assert len(fw["evidence"]) <= 12  # display still bounded


def test_inspect_incidental_numpy_entry_does_not_suppress_dynamically_loaded_pytorch(tmp_path):
    # An incidental `import numpy` in the entry must not win primary-framework
    # selection over the real PyTorch code, even when that code is loaded
    # dynamically (no static import chain) and lives in a non-entry-point
    # submodule. numpy is a numerical utility, not the training framework.
    (tmp_path / "main.py").write_text(
        "import numpy as np\n"
        "import importlib\n"
        "def main():\n"
        "    importlib.import_module('pkg.net')\n"
        "    return np.array([1])\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
        encoding="utf-8",
    )
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "pkg" / "net.py").write_text(
        "import torch\nimport torch.nn as nn\nNET = torch.nn.Linear(4, 2)\nOPT = torch.optim.SGD(NET.parameters(), lr=0.1)\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    assert data["skill_selection"]["detected_framework"] == "pytorch"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-pytorch"]


def test_inspect_tied_numpy_entry_fallback_prefers_dynamically_loaded_pytorch(tmp_path):
    # A single incidental numpy import and a single dynamically-loaded torch
    # import tie on confidence. The fallback must not route to numpy just
    # because it sorts alphabetically before pytorch.
    (tmp_path / "main.py").write_text(
        "import importlib\n"
        "import numpy as np\n"
        "def main():\n"
        "    importlib.import_module('pkg.net')\n"
        "    return np.array([1])\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
        encoding="utf-8",
    )
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "pkg" / "net.py").write_text("import torch\n", encoding="utf-8")

    data = inspect_path(tmp_path)

    assert {framework["name"]: framework["confidence"] for framework in data["frameworks"]} == {
        "numpy": 0.7,
        "pytorch": 0.7,
    }
    assert data["skill_selection"]["detected_framework"] == "pytorch"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-pytorch"]


def test_inspect_reverse_src_layout_prefers_importing_files_packaging_root(tmp_path):
    # Reverse of the stale-src-copy case: entry and real code live under src/,
    # and a stale copy sits at the root. The import from src/pkg/main.py must
    # resolve to the src/ copy (sharing its packaging root), not the stale
    # root-level copy, so routing follows the real (src/) PyTorch code.
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    (tmp_path / "src" / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "src" / "pkg" / "main.py").write_text(
        "from pkg.loop import run\nif __name__ == '__main__':\n    run()\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "pkg" / "loop.py").write_text(
        "import torch\nimport torch.nn as nn\nclass Net(nn.Module):\n    pass\ndef run():\n    return Net()\n",
        encoding="utf-8",
    )
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "pkg" / "loop.py").write_text(
        "import lightning.pytorch as pl\nclass Lit(pl.LightningModule):\n    pass\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-pytorch"]


def test_inspect_stray_lightning_import_is_mixed_framework_not_flare_mixed_workspace(tmp_path):
    # A plain PyTorch repo with a stray, unused `import pytorch_lightning` is a
    # mixed-framework workspace, not the FLARE conversion "mixed_workspace".
    (tmp_path / "train.py").write_text(
        "import torch\nimport torch.nn as nn\nclass Net(nn.Module):\n    pass\n"
        "def main():\n    Net()\nif __name__ == '__main__':\n    main()\n",
        encoding="utf-8",
    )
    (tmp_path / "misc.py").write_text("import pytorch_lightning\n", encoding="utf-8")

    data = inspect_path(tmp_path)

    assert data["target_type"] == "mixed_framework_workspace"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-pytorch"]


def test_inspect_torch_ops_inside_lightning_module_with_unrelated_entry_routes_to_lightning(tmp_path):
    # A realistic LightningModule calls several torch APIs (optimizer, loss,
    # dataloader). Those live in the same file as the active Lightning evidence,
    # so they are Lightning code, not standalone PyTorch usage. Even though the
    # raw torch-call count exceeds the single LightningModule class, an unrelated
    # entry point must not let that in-Lightning torch usage force the PyTorch
    # base and misroute a genuine Lightning repo.
    (tmp_path / "litmodel.py").write_text(
        "import torch\nimport lightning.pytorch as pl\n"
        "from torch.optim import SGD\nfrom torch.utils.data import DataLoader\n"
        "class LitNet(pl.LightningModule):\n"
        "    def train_dataloader(self):\n"
        "        return DataLoader([])\n"
        "    def training_step(self, batch, batch_idx):\n"
        "        return torch.nn.functional.cross_entropy(batch, batch)\n"
        "    def configure_optimizers(self):\n"
        "        return SGD(self.parameters(), lr=0.1)\n",
        encoding="utf-8",
    )
    (tmp_path / "run.py").write_text(
        "import json\nif __name__ == '__main__':\n    print(json.dumps({}))\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-lightning"]


def test_inspect_split_file_pytorch_model_with_unrelated_lightning_helper_keeps_pytorch(tmp_path):
    (tmp_path / "train.py").write_text(
        "from model import Net\n" "\n" "def main():\n" "    return Net()\n",
        encoding="utf-8",
    )
    (tmp_path / "model.py").write_text(
        "import torch\n" "\n" "class Net(torch.nn.Module):\n" "    pass\n",
        encoding="utf-8",
    )
    (tmp_path / "lit_helper.py").write_text(
        "import pytorch_lightning as pl\n"
        "\n"
        "class Helper(pl.LightningModule):\n"
        "    pass\n"
        "\n"
        "trainer = pl.Trainer(max_epochs=1)\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    framework_names = [framework["name"] for framework in data["frameworks"]]
    assert framework_names[0] == "pytorch"
    assert "pytorch_lightning" in framework_names
    assert data["target_type"] == "mixed_framework_workspace"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-pytorch"]


def test_inspect_top_level_model_import_does_not_reach_nested_model_file(tmp_path):
    helpers = tmp_path / "helpers"
    helpers.mkdir()
    (tmp_path / "train.py").write_text(
        "import torch\n" "import model\n" "\n" "def train():\n" "    return model.Net()\n",
        encoding="utf-8",
    )
    (tmp_path / "model.py").write_text(
        "import torch\n" "\n" "class Net(torch.nn.Module):\n" "    pass\n",
        encoding="utf-8",
    )
    (helpers / "model.py").write_text(
        "import pytorch_lightning as pl\n" "\n" "class Helper(pl.LightningModule):\n" "    pass\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    framework_names = [framework["name"] for framework in data["frameworks"]]
    assert framework_names[0] == "pytorch"
    assert "pytorch_lightning" in framework_names
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-pytorch"]


def test_inspect_external_lightning_import_does_not_promote_shadowing_lightning_package(tmp_path):
    # An unreachable local ``lightning`` package can carry far more active
    # Lightning evidence than the PyTorch entry point. The dotted external import
    # ``import lightning.pytorch`` must not resolve to that local package, so the
    # entry-context guard -- not the weighted fallback score -- must keep routing
    # on PyTorch even though the helper would otherwise win on raw evidence.
    package = tmp_path / "models"
    package.mkdir()
    (package / "train.py").write_text(
        "import lightning.pytorch\n"
        "import torch\n"
        "from torch.utils.data import DataLoader\n"
        "\n"
        "def train():\n"
        "    return DataLoader([])\n",
        encoding="utf-8",
    )
    lightning_package = package / "lightning"
    lightning_package.mkdir()
    (lightning_package / "__init__.py").write_text(
        "import pytorch_lightning as pl\n"
        "import lightning.pytorch\n"
        "from pytorch_lightning.callbacks import ModelCheckpoint\n"
        "\n"
        "class HelperA(pl.LightningModule):\n"
        "    pass\n"
        "\n"
        "class HelperB(pl.LightningModule):\n"
        "    pass\n"
        "\n"
        "trainer = pl.Trainer(max_epochs=1)\n"
        "second_trainer = pl.Trainer(max_epochs=2)\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    framework_by_name = {framework["name"]: framework for framework in data["frameworks"]}
    framework_names = [framework["name"] for framework in data["frameworks"]]
    assert framework_names[0] == "pytorch"
    assert "pytorch_lightning" in framework_names
    # The helper genuinely carries more raw evidence; routing still stays on
    # PyTorch because the helper is unreachable from the entry point.
    assert len(framework_by_name["pytorch_lightning"]["evidence"]) > len(framework_by_name["pytorch"]["evidence"])
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-pytorch"]


def test_inspect_same_directory_model_import_can_reach_lightning_helper(tmp_path):
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "train.py").write_text(
        "import torch\n" "from model import LitModel\n" "\n" "def main():\n" "    return LitModel()\n",
        encoding="utf-8",
    )
    (package / "model.py").write_text(
        "import pytorch_lightning as pl\n" "\n" "class LitModel(pl.LightningModule):\n" "    pass\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    framework_names = [framework["name"] for framework in data["frameworks"]]
    assert framework_names[0] == "pytorch_lightning"
    assert "pytorch" in framework_names
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-lightning"]


def test_inspect_nested_local_dotted_import_can_reach_lightning_helper(tmp_path):
    package = tmp_path / "models"
    package.mkdir()
    (package / "train.py").write_text(
        "import torch\n" "from layers.block import LitModel\n" "\n" "def main():\n" "    return LitModel()\n",
        encoding="utf-8",
    )
    layers = package / "layers"
    layers.mkdir()
    (layers / "__init__.py").write_text("", encoding="utf-8")
    (layers / "block.py").write_text(
        "import pytorch_lightning as pl\n" "\n" "class LitModel(pl.LightningModule):\n" "    pass\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    framework_names = [framework["name"] for framework in data["frameworks"]]
    assert framework_names[0] == "pytorch_lightning"
    assert "pytorch" in framework_names
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-lightning"]


def test_inspect_nested_dotted_import_follows_context_resolved_package_init(tmp_path):
    # The Lightning evidence lives in ``models/layers/__init__.py`` and is only reachable through
    # the context-resolved ``models.layers`` package prefix of ``from layers.block import ...`` in
    # ``models/train.py``. An unrelated top-level ``layers/`` package (matching the raw prefix) must
    # not be followed.
    package = tmp_path / "models"
    package.mkdir()
    (package / "train.py").write_text(
        "import torch\n" "from layers.block import LitModel\n" "\n" "def main():\n" "    return LitModel()\n",
        encoding="utf-8",
    )
    nested_layers = package / "layers"
    nested_layers.mkdir()
    (nested_layers / "__init__.py").write_text(
        "import pytorch_lightning as pl\n" "\n" "class LitModel(pl.LightningModule):\n" "    pass\n",
        encoding="utf-8",
    )
    # block.py is neutral so the Lightning evidence in ``models/layers/__init__.py`` is reachable
    # only through the package-prefix follow of the resolved ``models.layers.block`` module.
    (nested_layers / "block.py").write_text("import torch\n", encoding="utf-8")
    unrelated_layers = tmp_path / "layers"
    unrelated_layers.mkdir()
    (unrelated_layers / "__init__.py").write_text("import tensorflow\n", encoding="utf-8")

    data = inspect_path(tmp_path)

    framework_names = [framework["name"] for framework in data["frameworks"]]
    assert framework_names[0] == "pytorch_lightning"
    assert "pytorch" in framework_names
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-lightning"]


def test_inspect_nested_dotted_import_does_not_follow_raw_top_level_package_init(tmp_path):
    # Complement to the test above: the raw top-level prefix ``layers`` now carries the active
    # Lightning evidence, while the context-resolved ``models.layers`` package is plain PyTorch.
    # Following the raw top-level prefix would incorrectly reach the Lightning evidence, so routing
    # must stay on PyTorch to prove only the context-resolved package prefix is traversed.
    package = tmp_path / "models"
    package.mkdir()
    (package / "train.py").write_text(
        "import torch\n" "from layers.block import Model\n" "\n" "def main():\n" "    return Model()\n",
        encoding="utf-8",
    )
    nested_layers = package / "layers"
    nested_layers.mkdir()
    # The context-resolved ``models.layers`` package is plain PyTorch.
    (nested_layers / "__init__.py").write_text(
        "import torch.nn as nn\n" "\n" "class Model(nn.Module):\n" "    pass\n",
        encoding="utf-8",
    )
    (nested_layers / "block.py").write_text("import torch\n", encoding="utf-8")
    # The unrelated top-level ``layers/`` package matches the raw prefix and holds Lightning
    # evidence; it must not be followed from ``models/train.py``.
    unrelated_layers = tmp_path / "layers"
    unrelated_layers.mkdir()
    (unrelated_layers / "__init__.py").write_text(
        "import pytorch_lightning as pl\n" "\n" "class Helper(pl.LightningModule):\n" "    pass\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    framework_names = [framework["name"] for framework in data["frameworks"]]
    assert framework_names[0] == "pytorch"
    # Lightning is still detected globally but stays unreachable, so routing remains PyTorch.
    assert "pytorch_lightning" in framework_names
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-pytorch"]


def test_inspect_split_file_lightning_trainer_helper_beats_pytorch_entry_point(tmp_path):
    (tmp_path / "train.py").write_text(
        "import torch\n"
        "from torch.utils.data import DataLoader\n"
        "from lightning_helper import build_trainer\n"
        "\n"
        "def main():\n"
        "    loader = DataLoader([])\n"
        "    return build_trainer(loader)\n",
        encoding="utf-8",
    )
    (tmp_path / "lightning_helper.py").write_text(
        "import pytorch_lightning as pl\n"
        "\n"
        "class LitModel(pl.LightningModule):\n"
        "    pass\n"
        "\n"
        "def build_trainer(_loader):\n"
        "    trainer = pl.Trainer(max_epochs=1)\n"
        "    return trainer, LitModel()\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    framework_names = [framework["name"] for framework in data["frameworks"]]
    assert framework_names[0] == "pytorch_lightning"
    assert "pytorch" in framework_names
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-lightning"]


def test_inspect_unqualified_lightning_symbol_without_from_import_stays_import_only(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "import pytorch_lightning\n" "\n" "class LitModel(LightningModule):\n" "    pass\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["frameworks"][0]["name"] == "pytorch_lightning"
    assert all(item["kind"] == "import" for item in data["frameworks"][0]["evidence"])


def test_inspect_lightning_subscripted_base_recommends_lightning(tmp_path):
    script = tmp_path / "model.py"
    script.write_text(
        "import pytorch_lightning as pl\n" "\n" "class LitModel(pl.LightningModule[int]):\n" "    pass\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["frameworks"][0]["name"] == "pytorch_lightning"
    assert any(item["kind"] == "lightning_class" for item in data["frameworks"][0]["evidence"])


def test_module_names_for_file_handles_package_and_invalid_paths():
    assert _module_names_for_file("model.py") == {"model"}
    assert _module_names_for_file("pkg/model.py") == {"pkg.model"}
    assert _module_names_for_file("pkg/__init__.py") == {"pkg"}
    assert _module_names_for_file("notes.txt") == set()
    assert _module_names_for_file("../model.py") == set()


def test_resolve_import_from_module_handles_absolute_and_relative_imports():
    assert _resolve_import_from_module("train.py", "models", 0) == "models"
    assert _resolve_import_from_module("pkg/train.py", "", 1) == "pkg"
    assert _resolve_import_from_module("pkg/train.py", "model", 1) == "pkg.model"
    assert _resolve_import_from_module("pkg/sub/train.py", "model", 2) == "pkg.model"


def test_lightning_routing_helper_defensive_branches(tmp_path):
    state = InspectState(root=tmp_path / "train.py", redact=True)
    state.framework_evidence["pytorch_lightning"] = [
        {"file": "helper.py", "line": 1, "kind": "lightning_class", "value": "pl.LightningModule"}
    ]
    state.framework_evidence["pytorch"] = [
        {"file": "model.py", "line": 1, "kind": "pytorch_class", "value": "torch.nn.Module"}
    ]

    assert not _should_promote_lightning_over_pytorch(state)
    assert not _framework_evidence_tied_to_entry_context(state, state.framework_evidence["pytorch_lightning"])
    assert not _framework_evidence_tied_to_entry_context(state, state.framework_evidence["pytorch"])
    assert not _entry_point_imports_file(state, "README.md")
    assert _evidence_score([{"kind": "unknown"}]) == 1


def test_lightning_routing_fallback_prefers_active_lightning_over_pytorch_imports(tmp_path):
    state = InspectState(root=tmp_path, redact=True)
    state.framework_evidence["pytorch_lightning"] = [
        {"file": "model.py", "line": 1, "kind": "import", "value": "pytorch_lightning"},
        {"file": "model.py", "line": 6, "kind": "lightning_class", "value": "pl.LightningModule"},
    ]
    state.framework_evidence["pytorch"] = [
        {"file": "model.py", "line": 2, "kind": "import", "value": "torch"},
        {"file": "model.py", "line": 3, "kind": "import", "value": "torch.nn"},
        {"file": "model.py", "line": 4, "kind": "import", "value": "torch.optim"},
        {"file": "model.py", "line": 5, "kind": "import", "value": "torch.utils.data"},
    ]

    assert _should_promote_lightning_over_pytorch(state)


def test_lightning_routing_fallback_keeps_pytorch_import_threshold_for_unrelated_helpers(tmp_path):
    state = InspectState(root=tmp_path, redact=True)
    state.framework_evidence["pytorch_lightning"] = [
        {"file": "lightning_helper.py", "line": 1, "kind": "import", "value": "pytorch_lightning"},
        {"file": "lightning_helper.py", "line": 4, "kind": "lightning_class", "value": "pl.LightningModule"},
    ]
    state.framework_evidence["pytorch"] = [
        {"file": "experiment.py", "line": 1, "kind": "import", "value": "torch"},
        {"file": "experiment.py", "line": 2, "kind": "import", "value": "torch.nn"},
        {"file": "experiment.py", "line": 3, "kind": "import", "value": "torch.optim"},
        {"file": "experiment.py", "line": 4, "kind": "import", "value": "torch.utils.data"},
    ]

    assert not _should_promote_lightning_over_pytorch(state)


def test_inspect_unrelated_entry_ignores_pytorch_calls_inside_lightning_module(tmp_path):
    (tmp_path / "main.py").write_text(
        "def main():\n" "    return 'unrelated entry point'\n",
        encoding="utf-8",
    )
    (tmp_path / "helper.py").write_text("import torch\n", encoding="utf-8")
    (tmp_path / "lit.py").write_text(
        "import torch\n"
        "import pytorch_lightning as pl\n"
        "\n"
        "class LitModel(pl.LightningModule):\n"
        "    def configure_optimizers(self):\n"
        "        return torch.optim.SGD(self.parameters(), lr=0.1)\n"
        "\n"
        "    def training_step(self, batch, batch_idx):\n"
        "        loss = torch.nn.CrossEntropyLoss()\n"
        "        return loss\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    framework_names = [framework["name"] for framework in data["frameworks"]]
    assert framework_names[0] == "pytorch_lightning"
    assert "pytorch" in framework_names
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-lightning"]
    pytorch_evidence = next(framework["evidence"] for framework in data["frameworks"] if framework["name"] == "pytorch")
    assert any(item["file"] == "lit.py" and item["kind"] == "pytorch_call" for item in pytorch_evidence)


def test_inspect_lightning_fallback_ignores_in_module_torch_calls_for_outside_import(tmp_path):
    # No entry point: active Lightning and standalone PyTorch tie, so the final
    # weighted fallback must not count torch calls from inside the Lightning file.
    (tmp_path / "litmodel.py").write_text(
        "import torch\n"
        "import lightning\n"
        "import pytorch_lightning as pl\n"
        "from pytorch_lightning.callbacks import ModelCheckpoint\n"
        "\n"
        "class LitModel(pl.LightningModule):\n"
        "    def configure_optimizers(self):\n"
        "        return torch.optim.SGD(self.parameters(), lr=0.1)\n"
        "\n"
        "    def train_dataloader(self):\n"
        "        return torch.utils.data.DataLoader([])\n"
        "\n"
        "    def training_step(self, batch, batch_idx):\n"
        "        loss_fn = torch.nn.CrossEntropyLoss()\n"
        "        return loss_fn(batch[0], batch[1])\n",
        encoding="utf-8",
    )
    (tmp_path / "torch_import_only.py").write_text("import torch\n", encoding="utf-8")
    (tmp_path / "base_model.py").write_text(
        "import torch\n" "\n" "class Net(torch.nn.Module):\n" "    pass\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    framework_names = [framework["name"] for framework in data["frameworks"]]
    assert framework_names[0] == "pytorch_lightning"
    assert "pytorch" in framework_names
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-lightning"]
    pytorch_evidence = next(framework["evidence"] for framework in data["frameworks"] if framework["name"] == "pytorch")
    assert any(
        item["file"] == "torch_import_only.py" and item["kind"] == "import" and item["value"] == "torch"
        for item in pytorch_evidence
    )
    assert (
        sum(
            1
            for item in pytorch_evidence
            if item["file"] == "litmodel.py" and item["kind"] in {"pytorch_call", "pytorch_data_call"}
        )
        >= 3
    )


@pytest.mark.parametrize(
    ("expected_framework", "training_imports"),
    [
        ("tensorflow", "import tensorflow\nimport keras\nfrom tensorflow.keras import layers\n"),
        ("jax", "import jax\nimport flax\nimport optax\n"),
    ],
)
def test_inspect_non_pytorch_workspace_with_incidental_lightning_import_is_not_lightning(
    tmp_path, expected_framework, training_imports
):
    # The Lightning-over-PyTorch preference is a PyTorch-family rule only. A
    # non-PyTorch workspace with an incidental pytorch_lightning import
    # must not be routed to the Lightning conversion skill.
    (tmp_path / "train.py").write_text(
        training_imports,
        encoding="utf-8",
    )
    (tmp_path / "optional_utils.py").write_text(
        "import pytorch_lightning\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    framework_names = [framework["name"] for framework in data["frameworks"]]
    assert framework_names[0] == expected_framework
    assert data["skill_selection"]["detected_framework"] == expected_framework
    assert "nvflare-convert-lightning" not in data["skill_selection"]["recommended_skills"]


def test_inspect_lightning_with_other_frameworks_recommends_lightning(tmp_path):
    # Lightning wins over PyTorch and is surfaced first for display even when a
    # third, higher-import-count framework is present in the workspace.
    (tmp_path / "train.py").write_text(
        "import torch\n"
        "import pytorch_lightning as pl\n"
        "\n"
        "class Net(pl.LightningModule):\n"
        "    pass\n"
        "\n"
        "trainer = pl.Trainer(max_epochs=1)\n",
        encoding="utf-8",
    )
    (tmp_path / "tf_helper.py").write_text(
        "import tensorflow\n" "import keras\n" "from tensorflow.keras import layers\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    framework_names = [framework["name"] for framework in data["frameworks"]]
    assert framework_names[0] == "pytorch_lightning"
    assert "tensorflow" in framework_names
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-lightning"]


def test_inspect_exported_job_priority_over_lightning_routing(tmp_path):
    (tmp_path / "meta.json").write_text("{}\n", encoding="utf-8")
    app_config = tmp_path / "app_server" / "config"
    app_config.mkdir(parents=True)
    (app_config / "config_fed_server.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "client.py").write_text(
        "import pytorch_lightning as pl\n" "\n" "class Net(pl.LightningModule):\n" "    pass\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    assert data["frameworks"][0]["name"] == "pytorch_lightning"
    assert data["conversion_state"] == "exported_job"
    assert data["target_type"] == "exported_submit_ready_flare_job"
    assert data["job"]["nested_candidates"] == []
    assert data["skill_selection"]["recommended_skills"] == []


def test_inspect_directory_reports_inspected_target_path(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "train.py").write_text("import torch\n", encoding="utf-8")

    data = inspect_path(root)

    assert data["path"] == str(root.resolve(strict=False))
    assert data["path"] != "."


def test_inspect_file_reports_inspected_target_path(tmp_path):
    script = tmp_path / "train.py"
    script.write_text("import torch\n", encoding="utf-8")

    data = inspect_path(script)

    assert data["path"] == str(script.resolve(strict=False))


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks are not supported on this platform")
def test_inspect_symlink_reports_link_path_without_resolving_target(tmp_path):
    target_dir = tmp_path / "outside"
    target_dir.mkdir()
    (target_dir / "train.py").write_text("import tensorflow\n", encoding="utf-8")
    link_dir = tmp_path / "linked-repo"
    link_dir.symlink_to(target_dir, target_is_directory=True)

    data = inspect_path(link_dir)

    assert data["path"] == os.path.abspath(os.path.normpath(str(link_dir)))
    assert data["path"] != str(target_dir.resolve(strict=False))
    assert data["scan"]["files_skipped"][0]["code"] == "SYMLINK_SKIPPED"


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks are not supported on this platform")
def test_inspect_symlinked_file_does_not_classify_target(tmp_path):
    target_file = tmp_path / "outside.py"
    target_file.write_text("import torch\n", encoding="utf-8")
    link_file = tmp_path / "linked-train.py"
    link_file.symlink_to(target_file)

    data = inspect_path(link_file)

    assert data["target_type"] == "unknown_target"
    assert data["frameworks"] == []
    assert data["scan"]["files_skipped"] == [
        {
            "code": "SYMLINK_SKIPPED",
            "path": link_file.name,
            "target": "<REDACTED_PATH>",
            "message": "symlink was not followed during static inspection",
        }
    ]


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks are not supported on this platform")
def test_inspect_dangling_symlink_is_reported_as_skipped(tmp_path):
    link_file = tmp_path / "dangling-train.py"
    link_file.symlink_to(tmp_path / "missing.py")

    data = inspect_path(link_file)

    assert data["target_type"] == "unknown_target"
    assert data["scan"]["files_skipped"][0]["code"] == "SYMLINK_SKIPPED"
    assert data["scan"]["files_skipped"][0]["path"] == link_file.name


def test_inspect_redacts_secret_literals_and_absolute_paths_by_default(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "API_TOKEN = 'super-secret-value'\n" "DATA_ROOT = '/Users/alice/private/data'\n" "import tensorflow as tf\n",
        encoding="utf-8",
    )

    data = inspect_path(script)
    dumped = json.dumps(data)

    assert "super-secret-value" not in dumped
    assert "/Users/alice/private/data" not in dumped
    assert "<REDACTED>" in dumped
    assert "<REDACTED_PATH>" in dumped
    assert data["frameworks"][0]["name"] == "tensorflow"
    assert {finding["code"] for finding in data["findings"]} == {"SECRET_LITERAL_REDACTED"}
    assert data["patterns"]["absolute_data_paths"][0]["code"] == "ABSOLUTE_DATA_PATH"


def test_inspect_redaction_can_be_disabled_for_local_debugging(tmp_path):
    script = tmp_path / "train.py"
    script.write_text("PASSWORD = 'local-debug-secret'\nDATA_ROOT = '/opt/data'\n", encoding="utf-8")

    data = inspect_path(script, redact=False)
    dumped = json.dumps(data)

    assert "local-debug-secret" in dumped
    assert "/opt/data" in dumped


def test_inspect_skips_symlink_without_scanning_target(tmp_path):
    target = tmp_path / "outside.py"
    target.write_text("import tensorflow\n", encoding="utf-8")
    root = tmp_path / "repo"
    root.mkdir()
    (root / "linked.py").symlink_to(target)
    (root / "train.py").write_text("import torch\n", encoding="utf-8")

    data = inspect_path(root)

    assert [framework["name"] for framework in data["frameworks"]] == ["pytorch"]
    assert data["scan"]["files_skipped"][0]["code"] == "SYMLINK_SKIPPED"


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


def test_inspect_bom_prefixed_source_still_detects_framework(tmp_path):
    # A leading UTF-8 BOM (Windows/Notepad-authored source) must not blind the
    # inspector: it should still parse and detect the framework, not degrade to a
    # parse error with no evidence.
    script = tmp_path / "train.py"
    script.write_text(
        "﻿import torch\n"
        "\n"
        "\n"
        "class Net(torch.nn.Module):\n"
        "    def forward(self, x):\n"
        "        return x\n"
        "\n"
        "\n"
        'if __name__ == "__main__":\n'
        "    Net()\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    assert data["skill_selection"]["detected_framework"] == "pytorch"
    assert "nvflare-convert-pytorch" in data["skill_selection"]["recommended_skills"]
    assert not any(finding["code"] == "PYTHON_PARSE_ERROR" for finding in data["findings"])


def test_inspect_skips_deep_ast_and_continues_classifying_other_files(tmp_path, monkeypatch):
    (tmp_path / "generated.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "train.py").write_text(
        "import torch\n\n" "class Net(torch.nn.Module):\n" "    pass\n",
        encoding="utf-8",
    )
    original_visit = inspector_module._PythonInspector.visit

    def raise_for_generated_file(visitor, tree):
        if visitor.rel_path == "generated.py":
            raise RecursionError("AST depth exceeded")
        return original_visit(visitor, tree)

    monkeypatch.setattr(inspector_module._PythonInspector, "visit", raise_for_generated_file)

    data = inspect_path(tmp_path)

    assert data["classification_incomplete"] is True
    assert data["skill_selection"]["detected_framework"] == "pytorch"
    assert "nvflare-convert-pytorch" in data["skill_selection"]["recommended_skills"]
    findings = [finding for finding in data["findings"] if finding["code"] == "PYTHON_AST_DEPTH_LIMIT"]
    assert findings == [
        {
            "code": "PYTHON_AST_DEPTH_LIMIT",
            "severity": "warning",
            "file": "generated.py",
            "line": None,
            "message": "Python file exceeds the safe static-inspection AST depth.",
        }
    ]


def test_inspect_name_only_job_py_without_flare_evidence_is_not_flare_job(tmp_path):
    # A plain training repo that happens to have a launcher named job.py (a common
    # SLURM filename) and no nvflare imports must route to conversion, not be
    # misclassified as an existing FLARE job.
    (tmp_path / "job.py").write_text(
        "import torch\n\n\ndef main():\n    torch.nn.Linear(1, 1)\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    assert data["conversion_state"] == "not_converted"
    assert data["target_type"] == "training_repository"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-pytorch"]


def test_inspect_simenv_call_without_flare_evidence_is_not_flare_job(tmp_path):
    # SimEnv is a natural class name in RL/robotics code; a call to a local SimEnv
    # with no nvflare imports must not be classified as a FLARE job.
    (tmp_path / "train.py").write_text(
        "class SimEnv:\n    pass\n\n\ndef main():\n    env = SimEnv()\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    assert data["job"]["sim_env_used"] is True
    assert data["conversion_state"] != "flare_job"
    assert data["target_type"] != "flare_job_source"


def test_inspect_export_command_requires_flare_evidence(tmp_path):
    # `.export` calls over-match (torch.onnx.export); without nvflare evidence the
    # inspector must not ship a `job.py --export` command that would fail argparse.
    (tmp_path / "job.py").write_text(
        "import torch\n\n\ndef main():\n    torch.onnx.export(None, (), 'm.onnx')\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    assert data["job"]["export_support"] is True
    assert "python job.py --export --export-dir <job-dir>" not in data["recommended_next_commands"]


def test_inspect_does_not_treat_builtin_compile_as_torch_compile(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "import torch\n" "\n" "def build_code():\n" "    return compile('x = 1', '<inline>', 'exec')\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["frameworks"][0]["name"] == "pytorch"
    assert not any(item["kind"] == "torch_compile" for item in data["patterns"]["dynamic"])


def test_inspect_stops_and_caps_skips_after_file_limit(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    for index in range(20):
        (root / f"train_{index:02d}.py").write_text("import torch\n", encoding="utf-8")

    data = inspect_path(root, max_files=3)

    assert data["classification_incomplete"] is True
    assert data["scan"]["entries_visited"] == 3
    assert data["scan"]["files_considered"] == 20
    assert data["scan"]["files_scanned"] == 3
    assert data["scan"]["files_skipped_count"] == 17
    assert data["scan"]["files_skipped_count_approximate"] is False
    assert data["scan"]["files_skipped_truncated"] is True
    assert data["scan"]["files_skipped_evidence_truncated"] is True
    assert len(data["scan"]["files_skipped"]) == 12
    assert data["scan"]["files_skipped"][0] == {
        "code": "FILE_LIMIT_REACHED",
        "path": "train_03.py",
        "message": "file scan limit reached",
    }
    assert data["scan"]["files_skipped"][-1]["path"] == "train_14.py"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-pytorch", "nvflare-orient"]


def test_inspect_exact_file_limit_without_unvisited_files_is_complete(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    for index in range(3):
        (root / f"train_{index:02d}.py").write_text("import torch\n", encoding="utf-8")

    data = inspect_path(root, max_files=3)

    assert data["classification_incomplete"] is False
    assert data["scan"]["entries_visited"] == 3
    assert data["scan"]["files_considered"] == 3
    assert data["scan"]["files_scanned"] == 3
    assert data["scan"]["files_skipped_count"] == 0
    assert data["scan"]["files_skipped_count_approximate"] is False
    assert data["scan"]["files_skipped_truncated"] is False
    assert data["scan"]["files_skipped_evidence_truncated"] is False


def test_inspect_file_limit_accounting_is_bounded(monkeypatch, tmp_path):
    monkeypatch.setattr("nvflare.tool.agent.inspector.MAX_FILE_LIMIT_ACCOUNTED_SKIPS", 3)
    root = tmp_path / "repo"
    root.mkdir()
    for index in range(10):
        (root / f"train_{index:02d}.py").write_text("import torch\n", encoding="utf-8")

    data = inspect_path(root, max_files=1)

    assert data["classification_incomplete"] is True
    assert data["scan"]["entries_visited"] == 1
    assert data["scan"]["files_considered"] == 4
    assert data["scan"]["files_scanned"] == 1
    assert data["scan"]["files_skipped_count"] == 3
    assert data["scan"]["files_skipped_count_approximate"] is True
    assert data["scan"]["files_skipped_truncated"] is True
    assert data["scan"]["files_skipped"] == [
        {"code": "FILE_LIMIT_REACHED", "path": "train_01.py", "message": "file scan limit reached"},
        {"code": "FILE_LIMIT_REACHED", "path": "train_02.py", "message": "file scan limit reached"},
        {"code": "FILE_LIMIT_REACHED", "path": "train_03.py", "message": "file scan limit reached"},
    ]


def test_inspect_file_limit_unreadable_directory_accounting_is_bounded(monkeypatch, tmp_path):
    monkeypatch.setattr("nvflare.tool.agent.inspector.MAX_FILE_LIMIT_ACCOUNTED_SKIPS", 3)
    root = tmp_path / "repo"
    root.mkdir()
    for index in range(5):
        (root / f"a_{index:02d}").mkdir()
    (root / "train.py").write_text("import torch\n", encoding="utf-8")

    original_iterdir = Path.iterdir

    def fake_iterdir(path):
        if path.name.startswith("a_"):
            raise OSError("blocked")
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", fake_iterdir)

    data = inspect_path(root, max_files=1)

    assert data["classification_incomplete"] is True
    assert data["scan"]["entries_visited"] == 1
    assert data["scan"]["files_considered"] == 1
    assert data["scan"]["files_scanned"] == 1
    assert data["scan"]["files_skipped_count"] == 3
    assert data["scan"]["files_skipped_count_approximate"] is True
    assert data["scan"]["files_skipped_truncated"] is True
    assert data["scan"]["files_skipped"] == [
        {
            "code": "DIRECTORY_NOT_SCANNED_FILE_LIMIT",
            "path": "a_00",
            "message": "directory not scanned because file scan limit was reached",
        },
        {
            "code": "UNREADABLE_DIRECTORY",
            "path": "a_00",
            "message": "could not read directory",
            "error_type": "OSError",
        },
        {
            "code": "DIRECTORY_NOT_SCANNED_FILE_LIMIT",
            "path": "a_01",
            "message": "directory not scanned because file scan limit was reached",
        },
    ]


def test_inspect_file_limit_records_unvisited_stack_directories(tmp_path):
    root = tmp_path / "repo"
    nested = root / "a_nested"
    nested.mkdir(parents=True)
    (nested / "train_nested.py").write_text("import torch\n", encoding="utf-8")
    for index in range(5):
        (root / f"train_{index:02d}.py").write_text("import torch\n", encoding="utf-8")

    data = inspect_path(root, max_files=3)

    skipped = {(entry["code"], entry["path"]) for entry in data["scan"]["files_skipped"]}
    assert ("FILE_LIMIT_REACHED", "train_03.py") in skipped
    assert ("FILE_LIMIT_REACHED", "train_04.py") in skipped
    assert ("DIRECTORY_NOT_SCANNED_FILE_LIMIT", "a_nested") in skipped
    assert ("FILE_LIMIT_REACHED", "a_nested/train_nested.py") in skipped
    assert data["classification_incomplete"] is True
    assert data["scan"]["files_considered"] == 6
    assert data["scan"]["files_skipped_count"] == 4


def test_inspect_file_limit_records_pending_directories_when_last_child_reaches_limit(tmp_path):
    root = tmp_path / "repo"
    nested = root / "a_nested"
    nested.mkdir(parents=True)
    (nested / "train_nested.py").write_text("import torch\n", encoding="utf-8")
    for index in range(3):
        (root / f"train_{index:02d}.py").write_text("import torch\n", encoding="utf-8")

    data = inspect_path(root, max_files=3)

    skipped = {(entry["code"], entry["path"]) for entry in data["scan"]["files_skipped"]}
    assert ("DIRECTORY_NOT_SCANNED_FILE_LIMIT", "a_nested") in skipped
    assert ("FILE_LIMIT_REACHED", "a_nested/train_nested.py") in skipped
    assert data["scan"]["entries_visited"] == 3
    assert data["scan"]["files_considered"] == 4
    assert data["scan"]["files_scanned"] == 3
    assert data["scan"]["files_skipped_count"] == 2


def test_inspect_file_limit_counts_non_python_entries(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    for index in range(5):
        (root / f"metadata_{index:02d}.json").write_text("{}\n", encoding="utf-8")
    (root / "train.py").write_text("import torch\n", encoding="utf-8")

    data = inspect_path(root, max_files=3)

    assert data["classification_incomplete"] is True
    assert data["scan"]["entries_visited"] == 3
    assert data["scan"]["files_considered"] == 6
    assert data["scan"]["files_scanned"] == 0
    assert data["scan"]["files_skipped"] == [
        {"code": "FILE_LIMIT_REACHED", "path": "metadata_03.json", "message": "file scan limit reached"},
        {"code": "FILE_LIMIT_REACHED", "path": "metadata_04.json", "message": "file scan limit reached"},
        {"code": "FILE_LIMIT_REACHED", "path": "train.py", "message": "file scan limit reached"},
    ]


def test_inspect_benign_directory_skip_does_not_self_recommend_orient(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "train.py").write_text("import torch\n", encoding="utf-8")
    git_dir = root / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")

    data = inspect_path(root)

    assert data["classification_incomplete"] is False
    assert data["scan"]["files_skipped_count"] == 1
    assert data["scan"]["files_skipped"] == [
        {"code": "DIRECTORY_SKIPPED", "path": ".git", "message": "directory skipped"}
    ]
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-pytorch"]
