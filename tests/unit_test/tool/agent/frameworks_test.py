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

"""Unit tests for the modular framework-detector registry."""

import ast

from nvflare.tool.agent import frameworks
from nvflare.tool.agent.frameworks.base import DetectContext, EvidenceStrength
from nvflare.tool.agent.frameworks.huggingface import HuggingFaceDetector
from nvflare.tool.agent.frameworks.lightning import LightningDetector
from nvflare.tool.agent.frameworks.pytorch import PyTorchDetector
from nvflare.tool.agent.inspector import inspect_path


def test_framework_for_import_covers_detectors_and_import_only_roots():
    assert frameworks.framework_for_import("torch.nn") == "pytorch"
    assert frameworks.framework_for_import("pytorch_lightning") == "pytorch_lightning"
    assert frameworks.framework_for_import("lightning.pytorch") == "pytorch_lightning"
    assert frameworks.framework_for_import("transformers.trainer") == "huggingface"
    assert frameworks.framework_for_import("trl") == "huggingface"
    # Import-only frameworks (no active detector yet) still rank from imports.
    assert frameworks.framework_for_import("xgboost") == "xgboost"
    assert frameworks.framework_for_import("sklearn.svm") == "sklearn"
    assert frameworks.framework_for_import("unrelated") is None


def test_evidence_weights_are_aggregated_from_detectors():
    weights = frameworks.evidence_weights()
    assert weights["import"] == 1
    assert weights["pytorch_class"] == 3
    assert weights["pytorch_data_call"] == 2
    assert weights["lightning_trainer"] == 3
    assert weights["huggingface_train"] == 4


def test_detectors_share_explicit_evidence_strengths():
    pytorch = PyTorchDetector()
    lightning = LightningDetector()
    huggingface = HuggingFaceDetector()

    assert pytorch.evidence_strength({"kind": "import"}) == EvidenceStrength.IMPORT
    assert pytorch.evidence_strength({"kind": "pytorch_class"}) == EvidenceStrength.CANDIDATE
    assert pytorch.evidence_strength({"kind": "pytorch_call"}) == EvidenceStrength.TRAINING_OWNER
    assert lightning.evidence_strength({"kind": "lightning_class"}) == EvidenceStrength.CANDIDATE
    assert lightning.evidence_strength({"kind": "lightning_trainer"}) == EvidenceStrength.TRAINING_OWNER
    assert huggingface.evidence_strength({"kind": "huggingface_trainer"}) == EvidenceStrength.CANDIDATE
    assert huggingface.evidence_strength({"kind": "huggingface_train"}) == EvidenceStrength.TRAINING_OWNER


def test_recommended_skill_for():
    assert frameworks.recommended_skill_for("pytorch") is None
    assert frameworks.recommended_skill_for("huggingface") is None
    assert frameworks.recommended_skill_for("pytorch", []) is None
    assert frameworks.recommended_skill_for("pytorch_lightning", []) is None
    assert frameworks.recommended_skill_for("pytorch", [{"kind": "pytorch_class"}]) == "nvflare-convert-pytorch"
    assert (
        frameworks.recommended_skill_for("pytorch_lightning", [{"kind": "lightning_class"}])
        == "nvflare-convert-lightning"
    )
    assert frameworks.recommended_skill_for("huggingface", [{"kind": "import"}]) is None
    assert (
        frameworks.recommended_skill_for("huggingface", [{"kind": "huggingface_train"}])
        == "nvflare-convert-huggingface"
    )
    assert frameworks.recommended_skill_for("xgboost", []) is None
    assert frameworks.recommended_skill_for(None, []) is None


def test_huggingface_candidate_falls_back_to_orient():
    assert frameworks.fallback_skill_for("huggingface", [{"kind": "huggingface_training_config"}]) == "nvflare-orient"
    assert frameworks.fallback_skill_for("huggingface", [{"kind": "import"}]) is None


def test_inspect_finalizes_framework_calls_defined_before_later_imports(tmp_path):
    cases = {
        "pytorch": (
            "def run():\n    return SGD(params)\n\nfrom torch.optim import SGD\nrun()\n",
            "nvflare-convert-pytorch",
        ),
        "pytorch_lightning": (
            "def run():\n    pl.Trainer(max_epochs=1)\n\nimport pytorch_lightning as pl\nrun()\n",
            "nvflare-convert-lightning",
        ),
        "huggingface": (
            "def run():\n"
            "    args = TrainingArguments(output_dir='outputs')\n"
            "    trainer = Trainer(model=model, args=args)\n"
            "    trainer.train()\n"
            "\n"
            "from transformers import Trainer, TrainingArguments\n"
            "run()\n",
            "nvflare-convert-huggingface",
        ),
    }
    for framework, (source, expected_skill) in cases.items():
        script = tmp_path / f"{framework}.py"
        script.write_text(source, encoding="utf-8")
        data = inspect_path(script)
        assert data["skill_selection"]["detected_framework"] == framework
        assert data["skill_selection"]["recommended_skills"] == [expected_skill]
        assert data["framework_ownership"]["state"] == "clear"


def test_inspect_does_not_recommend_converter_for_lightning_import_only(tmp_path):
    script = tmp_path / "train.py"
    script.write_text("import lightning\n", encoding="utf-8")
    data = inspect_path(script)
    assert data["skill_selection"]["detected_framework"] == "pytorch_lightning"
    assert data["skill_selection"]["recommended_skills"] == []
    assert data["framework_ownership"] == {"state": "import_only", "owners": [], "candidates": []}


def test_later_import_finalization_honors_same_line_trainer_rebinding(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "def run():\n    trainer = Trainer(model=model); trainer = object(); trainer.train()\n"
        "from transformers import Trainer\nrun()\n",
        encoding="utf-8",
    )
    data = inspect_path(script)
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-orient"]
    assert data["framework_ownership"]["state"] == "candidate"


def test_manual_pytorch_owner_beats_lightning_model_candidate(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "import torch\n"
        "import pytorch_lightning as pl\n"
        "class Helper(pl.LightningModule): pass\n"
        "model = torch.nn.Linear(2, 1)\n"
        "optimizer = torch.optim.SGD(model.parameters(), lr=0.1)\n"
        "optimizer.step()\n",
        encoding="utf-8",
    )
    data = inspect_path(script)
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-pytorch"]
    assert data["framework_ownership"] == {
        "state": "clear",
        "owners": ["pytorch"],
        "candidates": ["pytorch_lightning"],
    }


def test_active_family_member_conflict_requires_two_specialized_trainers():
    class Resolver:
        @staticmethod
        def tied_to_entry_context(evidence):
            return bool(evidence)

    resolver = Resolver()
    assert frameworks.has_active_family_member_conflict(
        {
            "pytorch_lightning": [{"kind": "lightning_trainer"}],
            "huggingface": [{"kind": "huggingface_train"}],
        },
        resolver,
    )
    assert not frameworks.has_active_family_member_conflict(
        {
            "pytorch": [{"kind": "pytorch_call"}],
            "huggingface": [{"kind": "huggingface_train"}],
        },
        resolver,
    )
    assert not frameworks.has_active_family_member_conflict(
        {
            "pytorch_lightning": [{"kind": "lightning_class"}],
            "huggingface": [{"kind": "huggingface_train"}],
        },
        resolver,
    )


def test_resolve_primary_framework_consults_sibling_family_member_after_decline():
    evidence_by_framework = {
        "pytorch": [{"kind": "pytorch_call", "file": "train.py", "line": 9, "value": "Adam"}],
        "huggingface": [
            {"kind": "huggingface_trainer", "file": "train.py", "line": 5, "value": "Trainer"},
            {"kind": "huggingface_training_config", "file": "train.py", "line": 4, "value": "TrainingArguments"},
        ],
        "pytorch_lightning": [{"kind": "lightning_trainer", "file": "train.py", "line": 8, "value": "Trainer"}],
    }

    class Resolver:
        @staticmethod
        def score(evidence):
            return len(evidence)

        @staticmethod
        def tied_to_entry_context(evidence):
            return any(item.get("kind") == "lightning_trainer" for item in evidence)

        @staticmethod
        def evidence_outside_files(evidence, reference_evidence):
            return evidence

        @staticmethod
        def evidence_outside_class_bodies(evidence, class_evidence):
            return evidence

        @staticmethod
        def has_inspected_file_or_entry_point():
            return True

        def evidence(self, framework):
            return evidence_by_framework.get(framework, [])

        def active_evidence(self, framework):
            return [item for item in self.evidence(framework) if frameworks.is_active_evidence(framework, item)]

        def training_owner_evidence(self, framework):
            return [item for item in self.evidence(framework) if frameworks.is_training_owner_evidence(framework, item)]

    assert frameworks.resolve_primary_framework("huggingface", evidence_by_framework, Resolver()) == "pytorch_lightning"


def test_resolve_primary_framework_prefers_single_owner_over_sibling_candidates():
    evidence_by_framework = {
        "pytorch": [{"kind": "import", "file": "train.py", "line": 1, "value": "torch"}],
        "pytorch_lightning": [
            {"kind": "lightning_class", "file": "train.py", "line": line, "value": "LightningModule"}
            for line in range(2, 8)
        ],
        "huggingface": [{"kind": "huggingface_train", "file": "train.py", "line": 12, "value": "trainer.train"}],
    }

    class Resolver:
        def training_owner_evidence(self, framework):
            return [
                item
                for item in evidence_by_framework.get(framework, [])
                if frameworks.is_training_owner_evidence(framework, item)
            ]

        @staticmethod
        def tied_to_entry_context(evidence):
            return bool(evidence)

    assert frameworks.resolve_primary_framework("pytorch_lightning", evidence_by_framework, Resolver()) == "huggingface"


def _emit_collector():
    evidence = []
    flare_calls = []
    signals = []
    ctx = DetectContext(
        lambda fw, kind, value, lineno: evidence.append((fw, kind, value)),
        flare_calls.append,
        lambda fw, name: signals.append((fw, name)),
    )
    return ctx, evidence, flare_calls, signals


def test_pytorch_detector_records_class_evidence():
    detector = PyTorchDetector()
    state = detector.new_file_state()
    ctx, evidence, _, _ = _emit_collector()

    # from torch import nn ; class Net(nn.Module)
    detector.on_import_from("torch", [ast.alias(name="nn", asname=None)], state, ctx)
    detector.on_class_base("nn.Module", 3, state, ctx)

    assert ("pytorch", "pytorch_class", "nn.Module") in evidence


def test_pytorch_detector_preserves_aliased_data_helper_kind():
    detector = PyTorchDetector()
    state = detector.new_file_state()
    ctx, evidence, _, _ = _emit_collector()

    detector.on_import_from(
        "torch.utils.data",
        [ast.alias(name="DataLoader", asname="Loader")],
        state,
        ctx,
    )
    detector.on_call("Loader", 3, state, ctx)

    assert ("pytorch", "pytorch_data_call", "Loader") in evidence


def test_lightning_detector_records_patch_integration_signal():
    detector = LightningDetector()
    state = detector.new_file_state()
    ctx, evidence, flare_calls, signals = _emit_collector()

    # import nvflare.client.lightning as flare ; flare.patch(trainer)
    detector.on_import(ast.alias(name="nvflare.client.lightning", asname="flare"), state, ctx)
    detector.on_call("flare.patch", 5, state, ctx)

    assert "flare.patch" in flare_calls
    assert ("pytorch_lightning", "flare.patch") in signals


def test_huggingface_detector_records_trainer_and_patch_signals():
    detector = HuggingFaceDetector()
    state = detector.new_file_state()
    ctx, evidence, flare_calls, signals = _emit_collector()

    detector.on_import_from("trl", [ast.alias(name="SFTTrainer", asname="Trainer")], state, ctx)
    detector.on_import(ast.alias(name="nvflare.client.hf", asname="flare"), state, ctx)
    detector.on_assignment(["trainer"], "Trainer", 4, state, ctx)
    detector.on_call("Trainer", 4, state, ctx)
    detector.on_call("trainer.train", 5, state, ctx)
    detector.on_call("flare.patch", 6, state, ctx)

    assert ("huggingface", "huggingface_trainer", "Trainer") in evidence
    assert ("huggingface", "huggingface_train", "trainer.train") in evidence
    assert "flare.patch" in flare_calls
    assert ("huggingface", "flare.patch") in signals


def test_huggingface_detector_tracks_local_trainer_subclass():
    detector = HuggingFaceDetector()
    state = detector.new_file_state()
    ctx, evidence, _, _ = _emit_collector()

    detector.on_import_from("transformers", [ast.alias(name="Trainer", asname=None)], state, ctx)
    detector.on_class_definition("Custom", ["Trainer"], 3, state, ctx)
    detector.on_class_base("Trainer", 3, state, ctx)
    detector.on_assignment(["trainer"], "Custom", 5, state, ctx)
    detector.on_call("Custom", 5, state, ctx)
    detector.on_call("trainer.train", 6, state, ctx)

    assert ("huggingface", "huggingface_trainer_class", "Trainer") in evidence
    assert ("huggingface", "huggingface_trainer", "Custom") in evidence
    assert ("huggingface", "huggingface_train", "trainer.train") in evidence
