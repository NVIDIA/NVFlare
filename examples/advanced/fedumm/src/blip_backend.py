# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""BLIP-VQA backend: LoRA on text_encoder + text_decoder."""

from typing import Any, Dict, List, Tuple

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import BlipForQuestionAnswering, BlipProcessor

from .common import vqa_soft_score
from .model_registry import register_backend

_DEFAULT_MODEL = "Salesforce/blip-vqa-base"


class BLIPVQADataset(Dataset):
    def __init__(self, hf_ds, processor, max_q_len=64, max_a_len=16):
        self.ds, self.proc = hf_ds, processor
        self.max_q, self.max_a = max_q_len, max_a_len

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        img = ex["image"].convert("RGB")
        enc = self.proc(images=img, text=ex["question"],
                        padding="max_length", truncation=True,
                        max_length=self.max_q, return_tensors="pt")
        lab = self.proc.tokenizer(ex["multiple_choice_answer"],
                                  padding="max_length", truncation=True,
                                  max_length=self.max_a, return_tensors="pt")
        labels = lab["input_ids"].squeeze(0).clone()
        labels[labels == self.proc.tokenizer.pad_token_id] = -100
        return {
            "pixel_values": enc["pixel_values"].squeeze(0),
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": labels,
            "decoder_attention_mask": lab["attention_mask"].squeeze(0),
            "gt_answers": [a["answer"] for a in ex["answers"]],
        }


class BLIPBackend:
    name = "blip_vqa"

    def build_model_and_processor(self, model_name_or_path, lora_r, lora_alpha,
                                  lora_dropout, device):
        model_id = model_name_or_path or _DEFAULT_MODEL
        processor = BlipProcessor.from_pretrained(model_id)
        model = BlipForQuestionAnswering.from_pretrained(model_id)

        cfg = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=lora_r,
                         lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                         bias="none", target_modules=["query", "key", "value"])
        model.text_encoder = get_peft_model(model.text_encoder, cfg)
        model.text_decoder = get_peft_model(model.text_decoder, cfg)
        for n, p in model.named_parameters():
            p.requires_grad = "lora_" in n

        model.to(device)
        return model, processor

    def build_dataset(self, hf_ds, processor, max_q_len, max_a_len):
        return BLIPVQADataset(hf_ds, processor, max_q_len, max_a_len)

    def collate_fn(self, batch):
        out = {}
        for k in ["pixel_values", "input_ids", "attention_mask",
                   "labels", "decoder_attention_mask"]:
            out[k] = torch.stack([b[k] for b in batch])
        out["gt_answers"] = [b["gt_answers"] for b in batch]
        return out

    def train_step(self, model, batch, device):
        return model(
            pixel_values=batch["pixel_values"].to(device, non_blocking=True),
            input_ids=batch["input_ids"].to(device, non_blocking=True),
            attention_mask=batch["attention_mask"].to(device, non_blocking=True),
            labels=batch["labels"].to(device, non_blocking=True),
            decoder_attention_mask=batch["decoder_attention_mask"].to(device, non_blocking=True),
        ).loss

    @torch.no_grad()
    def evaluate(self, model, dataloader, processor, device):
        model.eval()
        total_score, total = 0.0, 0
        for batch in dataloader:
            gen = model.generate(
                input_ids=batch["input_ids"].to(device),
                pixel_values=batch["pixel_values"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                max_new_tokens=5,
            )
            preds = processor.tokenizer.batch_decode(gen, skip_special_tokens=True)
            for p, gt in zip(preds, batch["gt_answers"]):
                total_score += vqa_soft_score(p, gt)
                total += 1
        return total_score / max(total, 1)

    def hf_dataset_name(self):
        return "HuggingFaceM4/VQAv2"

    def hf_train_split(self):
        return "train"

    def hf_eval_split(self):
        return "validation[:50%]"

    def keep_columns(self):
        return ["image", "question", "multiple_choice_answer", "answers"]


register_backend("blip_vqa", BLIPBackend())
