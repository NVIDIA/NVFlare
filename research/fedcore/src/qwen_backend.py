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

"""Qwen3-VL feature extraction for paired image-present/image-removed views."""

import importlib.util
from pathlib import Path

import torch
from src.data import make_question


def _load_upstream_qwen_helpers():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "examples" / "advanced" / "qwen3-vl" / "model.py"
    if not module_path.exists():
        raise FileNotFoundError(f"The upstream Qwen3-VL helper was not found at {module_path}")
    spec = importlib.util.spec_from_file_location("nvflare_qwen3_vl_model", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import Qwen3-VL helper from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_nvflare_lora_checkpoint(model, checkpoint_path: Path, helpers, lora_r: int, lora_alpha: int):
    from peft import LoraConfig, TaskType, get_peft_model

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError(f"Expected a state dict in {checkpoint_path}")
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(lora_r),
            lora_alpha=int(lora_alpha),
            lora_dropout=0.0,
            target_modules=helpers.DEFAULT_LORA_TARGET_MODULES,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        ),
    )
    adapter_state = {}
    for key, value in state_dict.items():
        clean = key[6:] if key.startswith("model.") else key
        if "lora" in clean.lower():
            adapter_state[clean] = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    mapped, unmatched = helpers.map_adapter_state_dict_for_peft_model(model, adapter_state)
    if not mapped:
        raise ValueError(f"No LoRA adapter weights from {checkpoint_path} matched Qwen3-VL.")
    if unmatched:
        raise ValueError(f"Could not map {len(unmatched)} LoRA weights from {checkpoint_path}.")
    model.load_state_dict(mapped, strict=False)
    return model


class QwenFeatureExtractor:
    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda:0",
        batch_size: int = 2,
        adapter_checkpoint: str = "",
        lora_r: int = 64,
        lora_alpha: int = 128,
    ) -> None:
        from transformers import AutoProcessor

        helpers = _load_upstream_qwen_helpers()
        self.model_name_or_path = model_name_or_path
        self.device = torch.device(device)
        self.batch_size = max(1, int(batch_size))
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model = helpers.load_qwen_vl_from_pretrained(
            model_name_or_path,
            dtype=dtype,
            attn_implementation="sdpa",
        )
        if adapter_checkpoint:
            self.model = _load_nvflare_lora_checkpoint(
                self.model,
                Path(adapter_checkpoint).expanduser().resolve(),
                helpers,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
            )
        self.model.to(self.device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        self.class_token_ids = {label: self._single_token_id(label) for label in ("A", "B")}

    def _single_token_id(self, label: str) -> int:
        tokenizer = self.processor.tokenizer
        candidates = (label, f" {label}")
        for candidate in candidates:
            token_ids = tokenizer.encode(candidate, add_special_tokens=False)
            if len(token_ids) == 1:
                return int(token_ids[0])
        raise ValueError(f"Class label {label!r} is not a single tokenizer token.")

    @staticmethod
    def _messages(record: dict, data_dir: Path, include_image: bool) -> list[dict]:
        content = []
        if include_image:
            image_path = data_dir / record["image"]
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            content.append({"type": "image", "image": str(image_path)})
        content.append(
            {
                "type": "text",
                "text": make_question(record["context"], include_image=include_image),
            }
        )
        return [{"role": "user", "content": content}]

    def _forward(self, records: list[dict], data_dir: Path, include_image: bool) -> tuple[torch.Tensor, torch.Tensor]:
        all_features = []
        all_scores = []
        for start in range(0, len(records), self.batch_size):
            batch = records[start : start + self.batch_size]
            conversations = [self._messages(record, data_dir, include_image=include_image) for record in batch]
            inputs = self.processor.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=True,
                padding=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
            with torch.inference_mode():
                outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
            attention_mask = inputs["attention_mask"]
            last_positions = []
            for row in attention_mask:
                nonzero = torch.nonzero(row, as_tuple=False).flatten()
                last_positions.append(int(nonzero[-1].item()))
            row_indices = torch.arange(len(batch), device=self.device)
            position_indices = torch.tensor(last_positions, device=self.device)
            final_hidden = outputs.hidden_states[-1][row_indices, position_indices]
            final_logits = outputs.logits[row_indices, position_indices]
            score = final_logits[:, self.class_token_ids["A"]] - final_logits[:, self.class_token_ids["B"]]
            all_features.append(final_hidden.detach().float().cpu())
            all_scores.append(score.detach().float().cpu())
        return torch.cat(all_features), torch.cat(all_scores)

    def extract(self, records: list[dict], data_dir: Path) -> dict:
        missing_features, missing_logits = self._forward(records, data_dir, include_image=False)
        image_available = torch.tensor([record["image_available"] for record in records], dtype=torch.bool)
        full_logits = torch.full((len(records),), torch.nan, dtype=torch.float32)
        paired_indices = image_available.nonzero(as_tuple=False).flatten().tolist()
        if paired_indices:
            paired_records = [records[index] for index in paired_indices]
            _, paired_logits = self._forward(paired_records, data_dir, include_image=True)
            full_logits[paired_indices] = paired_logits
        return {
            "schema_version": 1,
            "example_ids": [record["example_id"] for record in records],
            "labels": torch.tensor([record["label"] for record in records], dtype=torch.long),
            "image_available": image_available,
            "paired_mask": image_available.clone(),
            "missing_features": missing_features,
            "missing_logits": missing_logits,
            "full_logits": full_logits,
        }

    def metadata(self) -> dict:
        return {
            "model_name_or_path": self.model_name_or_path,
            "class_token_ids": self.class_token_ids,
            "feature_pooling": "last_nonpadding_prompt_token",
            "classifier_score": "logit(A)-logit(B)",
        }
