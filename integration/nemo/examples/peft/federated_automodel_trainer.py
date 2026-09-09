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
"""AutoModel trainer with strict model-only adapter handoff between FL rounds."""

from __future__ import annotations

import json
import os
from pathlib import Path

import adapter_checkpoint
import torch
from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction


def _is_rank_zero() -> bool:
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def _last_training_record(checkpoint_dir: str) -> dict:
    log_path = os.path.join(checkpoint_dir, "training.jsonl")
    if not os.path.isfile(log_path):
        return {}
    last = {}
    with open(log_path) as f:
        for line in f:
            if line.strip():
                last = json.loads(line)
    return last


class FederatedTrainFinetuneRecipeForNextTokenPrediction(TrainFinetuneRecipeForNextTokenPrediction):
    """Restore only the adapter, leaving every local optimizer and scheduler fresh."""

    def _assert_trainable_parameters(self) -> dict[str, int]:
        trainable = []
        frozen_count = 0
        for model_part in self.model_parts:
            for name, parameter in model_part.named_parameters():
                if parameter.requires_grad:
                    trainable.append(name)
                else:
                    frozen_count += parameter.numel()
        if not trainable:
            raise RuntimeError("The Lightning profile did not create any trainable LoRA parameters.")
        unexpected = sorted(name for name in trainable if "lora_" not in name)
        if unexpected:
            raise RuntimeError(f"Base parameters are trainable: {unexpected[:10]}")
        return {"trainable_tensor_count": len(trainable), "frozen_parameter_count": frozen_count}

    def _export_adapter(self, output_root: str, *, is_final_checkpoint: bool) -> str:
        self.checkpointer.save_model(
            self.model_parts,
            output_root,
            peft_config=self.peft_config,
            tokenizer=self.tokenizer,
            is_final_checkpoint=is_final_checkpoint,
        )
        self.checkpointer.async_wait()
        return os.path.join(output_root, "model")

    def _write_report(self, report: dict) -> None:
        report_path = os.environ.get("NVFLARE_AUTOMODEL_REPORT")
        if report_path and _is_rank_zero():
            os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2, sort_keys=True)

    def load_checkpoint(self, restore_from: str | None = None):
        trainable = self._assert_trainable_parameters()
        if not restore_from:
            self._federated_load_report = {"restore_from": None, **trainable}
            return

        incoming_state = adapter_checkpoint.load_adapter_state(restore_from)
        adapter_checkpoint.align_adapter_state_strict(incoming_state, incoming_state)
        manifest = adapter_checkpoint.load_adapter_manifest(restore_from)
        expected_manifest = {
            "model_profile": os.environ.get("NVFLARE_MODEL_PROFILE"),
            "base_model_name_or_path": self.cfg.get("model.pretrained_model_name_or_path"),
            "profile_settings": json.loads(os.environ["NVFLARE_PROFILE_SETTINGS"]),
        }
        adapter_checkpoint.validate_adapter_manifest(manifest, incoming_state, expected=expected_manifest)

        self.checkpointer.load_model(self.model_parts, restore_from)

        verification_root = os.environ.get("NVFLARE_LOADED_ADAPTER_DIR")
        if not verification_root:
            verification_root = str(Path(self.checkpointer.config.checkpoint_dir) / "loaded_adapter_verification")
        verification_dir = self._export_adapter(verification_root, is_final_checkpoint=False)
        loaded_state = adapter_checkpoint.load_adapter_state(verification_dir)
        loaded_state = adapter_checkpoint.align_adapter_state_strict(loaded_state, incoming_state)
        received_hash = adapter_checkpoint.state_hash(incoming_state)
        loaded_hash = adapter_checkpoint.state_hash(loaded_state)
        mismatches = []
        for key, received_value in incoming_state.items():
            expected_loaded = received_value.to(loaded_state[key].dtype)
            if not torch.equal(loaded_state[key], expected_loaded):
                mismatches.append(key)
        if mismatches:
            raise RuntimeError(f"Native adapter reload changed values for {len(mismatches)} tensors: {mismatches[:5]}")
        self._federated_load_report = {
            "restore_from": os.path.abspath(restore_from),
            "received_adapter_hash": received_hash,
            "loaded_adapter_hash": loaded_hash,
            "received_tensor_count": len(incoming_state),
            "loaded_tensor_count": len(loaded_state),
            "loaded_checkpoint_dir": verification_dir,
            "loaded_matches_received_after_dtype_cast": True,
            **trainable,
        }
        self._write_report(self._federated_load_report)

    def run_train_validation_loop(self):
        output_root = os.environ.get("NVFLARE_OUTPUT_ADAPTER_DIR")
        if os.environ.get("NVFLARE_INITIALIZE_ONLY") == "1":
            if not output_root:
                raise RuntimeError("NVFLARE_OUTPUT_ADAPTER_DIR is required for adapter initialization.")
            adapter_dir = self._export_adapter(output_root, is_final_checkpoint=True)
            state = adapter_checkpoint.load_adapter_state(adapter_dir)
            report = {
                **getattr(self, "_federated_load_report", {}),
                "initialized_adapter_hash": adapter_checkpoint.state_hash(state),
                "initialized_tensor_count": len(state),
                "output_adapter_dir": adapter_dir,
                "actual_optimizer_steps": 0,
            }
            self._write_report(report)
            return 0

        result = super().run_train_validation_loop()
        if not output_root:
            raise RuntimeError("NVFLARE_OUTPUT_ADAPTER_DIR is required for federated training.")
        adapter_dir = self._export_adapter(output_root, is_final_checkpoint=True)
        outgoing_state = adapter_checkpoint.load_adapter_state(adapter_dir)
        adapter_checkpoint.align_adapter_state_strict(outgoing_state, outgoing_state)
        report = {
            **getattr(self, "_federated_load_report", {}),
            "outgoing_adapter_hash": adapter_checkpoint.state_hash(outgoing_state),
            "outgoing_tensor_count": len(outgoing_state),
            "output_adapter_dir": adapter_dir,
            "actual_optimizer_steps": int(self.step_scheduler.step),
            "last_training_record": _last_training_record(str(self.checkpointer.config.checkpoint_dir)),
        }
        self._write_report(report)
        return result
