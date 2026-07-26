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
from nvflare.tool.agent.frameworks.base import DetectContext
from nvflare.tool.agent.frameworks.huggingface import HuggingFaceDetector
from nvflare.tool.agent.frameworks.lightning import LightningDetector
from nvflare.tool.agent.frameworks.pytorch import PyTorchDetector


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


def test_recommended_skill_for():
    assert frameworks.recommended_skill_for("pytorch") == "nvflare-convert-pytorch"
    assert frameworks.recommended_skill_for("huggingface") is None
    assert frameworks.recommended_skill_for("pytorch", []) == "nvflare-convert-pytorch"
    assert frameworks.recommended_skill_for("pytorch_lightning", []) == "nvflare-convert-lightning"
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
