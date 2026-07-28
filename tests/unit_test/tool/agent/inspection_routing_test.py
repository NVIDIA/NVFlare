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

from pathlib import Path

import pytest

from nvflare.tool.agent.frameworks.lightning import LightningDetector
from nvflare.tool.agent.inspection.models import InspectionFacts
from nvflare.tool.agent.inspection.project import (
    _entry_point_imports_file,
    _module_names_for_file,
    _resolve_import_from_module,
)
from nvflare.tool.agent.inspection.project import (
    evidence_tied_to_entry_context as _framework_evidence_tied_to_entry_context,
)
from nvflare.tool.agent.inspection.routing import FamilyResolver as _FamilyResolver
from nvflare.tool.agent.inspection.routing import evidence_score as _evidence_score
from nvflare.tool.agent.inspection.routing import routing_decision as _routing_decision
from nvflare.tool.agent.inspector import inspect_path


def _should_promote_lightning_over_pytorch(state):
    # The PyTorch-family promotion decision now lives in the Lightning detector;
    # exercise it through the same resolver the engine uses.
    return LightningDetector().promote_over_family("pytorch", _FamilyResolver(state))


@pytest.mark.parametrize(
    "detected_framework,conversion_state,dataset,family_member_conflict,expected_skill",
    [
        (None, "flare_job", None, False, "nvflare-autofl"),
        (None, "ambiguous", None, False, "nvflare-orient"),
        ("pytorch", "not_converted", {"modality": "tabular"}, False, "nvflare-fed-stats"),
        ("pytorch", "not_converted", {"modality": "mixed"}, False, "nvflare-orient"),
        ("pytorch", "not_converted", None, True, "nvflare-orient"),
        ("pytorch", "not_converted", None, False, "nvflare-convert-pytorch"),
    ],
)
def test_routing_decision_emits_a_matching_command_for_every_skill(
    detected_framework,
    conversion_state,
    dataset,
    family_member_conflict,
    expected_skill,
):
    framework_evidence = {}
    if detected_framework == "pytorch":
        framework_evidence["pytorch"] = (
            {"file": "train.py", "line": 1, "kind": "pytorch_call", "value": "torch.optim.SGD"},
        )
    facts = InspectionFacts(root=Path("project"), redact=True, framework_evidence=framework_evidence)

    decision = _routing_decision(
        detected_framework,
        conversion_state,
        None,
        facts,
        dataset,
        family_member_conflict,
    )

    assert decision.recommended_skills == (expected_skill,)
    assert decision.recommended_next_commands == (f"Use the {expected_skill} skill before editing.",)


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
    assert data["framework_ownership"] == {"state": "clear", "owners": ["huggingface"], "candidates": []}
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
    assert data["framework_ownership"] == {"state": "candidate", "owners": [], "candidates": ["huggingface"]}
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
    "source",
    [
        pytest.param(
            "from transformers import Trainer\n"
            "from nvflare.client.hf import patch\n"
            "trainer = Trainer(model=model, args=args)\n"
            "class patch:\n"
            "    def run(self):\n"
            "        patch(trainer)\n"
            "trainer.train()\n",
            id="huggingface",
        ),
        pytest.param(
            "import lightning as L\n"
            "from nvflare.client.lightning import patch\n"
            "trainer = L.Trainer(max_epochs=1)\n"
            "class patch:\n"
            "    def run(self):\n"
            "        patch(trainer)\n",
            id="lightning",
        ),
    ],
)
def test_inspect_revalidates_method_patch_after_enclosing_class_rebinding(tmp_path, source):
    script = tmp_path / "client.py"
    script.write_text(source, encoding="utf-8")

    assert inspect_path(script)["conversion_state"] == "partial_client_api"


@pytest.mark.parametrize(
    ("source", "framework", "kind", "expected"),
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
def test_inspect_revalidates_method_framework_call_after_enclosing_class_rebinding(
    tmp_path, source, framework, kind, expected
):
    script = tmp_path / "train.py"
    script.write_text(source, encoding="utf-8")

    evidence = next(item for item in inspect_path(script)["frameworks"] if item["name"] == framework)["evidence"]

    assert [item["value"] for item in evidence if item["kind"] == kind] == expected


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
            "train.py": (
                "import torch\n"
                "from torch.utils.data import DataLoader\n"
                "\n"
                "def main():\n"
                "    return DataLoader([])\n"
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
                "class Net(torch.nn.Module): pass\n"
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


def test_inspect_manual_pytorch_owner_wins_over_co_located_lightning_class(tmp_path):
    # A LightningModule candidate must not steal routing from a direct manual
    # PyTorch owner. PyTorch calls inside the LightningModule itself remain
    # attributable to Lightning.
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

    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-pytorch"]


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
    assert data["skill_selection"]["recommended_skills"] == []


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
    state = InspectionFacts(
        root=tmp_path / "train.py",
        redact=True,
        framework_evidence={
            "pytorch_lightning": (
                {"file": "helper.py", "line": 1, "kind": "lightning_class", "value": "pl.LightningModule"},
            ),
            "pytorch": ({"file": "model.py", "line": 1, "kind": "pytorch_class", "value": "torch.nn.Module"},),
        },
    )

    assert not _should_promote_lightning_over_pytorch(state)
    assert not _framework_evidence_tied_to_entry_context(state, state.framework_evidence["pytorch_lightning"])
    assert not _framework_evidence_tied_to_entry_context(state, state.framework_evidence["pytorch"])
    assert not _entry_point_imports_file(state, "README.md")
    assert _evidence_score([{"kind": "unknown"}]) == 1


def test_lightning_routing_fallback_prefers_active_lightning_over_pytorch_imports(tmp_path):
    state = InspectionFacts(
        root=tmp_path,
        redact=True,
        framework_evidence={
            "pytorch_lightning": (
                {"file": "model.py", "line": 1, "kind": "import", "value": "pytorch_lightning"},
                {"file": "model.py", "line": 6, "kind": "lightning_class", "value": "pl.LightningModule"},
            ),
            "pytorch": (
                {"file": "model.py", "line": 2, "kind": "import", "value": "torch"},
                {"file": "model.py", "line": 3, "kind": "import", "value": "torch.nn"},
                {"file": "model.py", "line": 4, "kind": "import", "value": "torch.optim"},
                {"file": "model.py", "line": 5, "kind": "import", "value": "torch.utils.data"},
            ),
        },
    )

    assert _should_promote_lightning_over_pytorch(state)


def test_lightning_routing_fallback_keeps_pytorch_import_threshold_for_unrelated_helpers(tmp_path):
    state = InspectionFacts(
        root=tmp_path,
        redact=True,
        framework_evidence={
            "pytorch_lightning": (
                {"file": "lightning_helper.py", "line": 1, "kind": "import", "value": "pytorch_lightning"},
                {"file": "lightning_helper.py", "line": 4, "kind": "lightning_class", "value": "pl.LightningModule"},
            ),
            "pytorch": (
                {"file": "experiment.py", "line": 1, "kind": "import", "value": "torch"},
                {"file": "experiment.py", "line": 2, "kind": "import", "value": "torch.nn"},
                {"file": "experiment.py", "line": 3, "kind": "import", "value": "torch.optim"},
                {"file": "experiment.py", "line": 4, "kind": "import", "value": "torch.utils.data"},
            ),
        },
    )

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
