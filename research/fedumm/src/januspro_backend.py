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

"""JanusPro backend: LoRA on the DeepSeek language_model only.

JanusPro architecture (MultiModalityCausalLM):
  - vision_model       (SigLIP-L, frozen)
  - aligner            (MLP projector, frozen)
  - language_model      (DeepSeek-LLM, LoRA target)
  - gen_vision_model   (generation vision encoder, frozen)
  - gen_head           (generation head, frozen)

We inject LoRA into ``language_model`` targeting q_proj / k_proj / v_proj,
keeping everything else frozen.  Only LoRA weights are exchanged in FL.

Note: JanusPro requires ``trust_remote_code=True`` and the ``janus``
      package (``pip install -e .`` from the Janus repo).
"""

from typing import Any, Dict, List, Tuple

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset

from .common import vqa_soft_score
from .model_registry import register_backend

_DEFAULT_MODEL = "deepseek-ai/Janus-Pro-1B"


class JanusProVQADataset(Dataset):
    """Wraps VQAv2 for JanusPro's conversation-style input.

    JanusPro uses ``VLChatProcessor`` which expects a list-of-dict
    conversation format.  We pre-tokenize here and store the tensors.
    """

    def __init__(self, hf_ds, processor, max_q_len=128, max_a_len=32):
        self.ds = hf_ds
        self.proc = processor
        self.tokenizer = processor.tokenizer
        self.max_q = max_q_len
        self.max_a = max_a_len

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        image = ex["image"].convert("RGB")
        question = ex["question"]
        answer = ex["multiple_choice_answer"]
        gt_answers = [a["answer"] for a in ex["answers"]]

        # Build conversation in JanusPro format
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{question}",
                "images": [image],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        # Use VLChatProcessor to build inputs
        pil_images = [image]
        prepare = self.proc(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
        )

        # Tokenize the answer for teacher-forcing labels
        ans_enc = self.tokenizer(
            answer,
            padding="max_length",
            truncation=True,
            max_length=self.max_a,
            return_tensors="pt",
        )
        labels = ans_enc["input_ids"].squeeze(0).clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": prepare.input_ids.squeeze(0),
            "attention_mask": prepare.attention_mask.squeeze(0),
            "pixel_values": prepare.pixel_values.squeeze(0)
            if hasattr(prepare, "pixel_values") and prepare.pixel_values is not None
            else prepare.images.squeeze(0),
            "images_seq_mask": prepare.images_seq_mask.squeeze(0)
            if hasattr(prepare, "images_seq_mask")
            else torch.zeros(prepare.input_ids.shape[-1], dtype=torch.bool),
            "images_emb_mask": prepare.images_emb_mask.squeeze(0)
            if hasattr(prepare, "images_emb_mask")
            else torch.zeros(1, dtype=torch.bool),
            "labels": labels,
            "labels_mask": ans_enc["attention_mask"].squeeze(0),
            "gt_answers": gt_answers,
        }


class JanusProBackend:
    name = "januspro"

    def build_model_and_processor(self, model_name_or_path, lora_r, lora_alpha,
                                  lora_dropout, device):
        from transformers import AutoModelForCausalLM

        model_id = model_name_or_path or _DEFAULT_MODEL

        # JanusPro uses custom model code
        from janus.models import MultiModalityCausalLM, VLChatProcessor

        processor = VLChatProcessor.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True
        )

        # Apply LoRA to the language_model (DeepSeek LLM) only
        cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj"],
        )
        model.language_model = get_peft_model(model.language_model, cfg)

        # Freeze everything except LoRA
        for n, p in model.named_parameters():
            p.requires_grad = "lora_" in n

        model.to(torch.bfloat16).to(device)
        return model, processor

    def build_dataset(self, hf_ds, processor, max_q_len, max_a_len):
        return JanusProVQADataset(hf_ds, processor, max_q_len, max_a_len)

    def collate_fn(self, batch):
        out = {}
        tensor_keys = [
            "input_ids", "attention_mask", "pixel_values",
            "images_seq_mask", "images_emb_mask", "labels", "labels_mask",
        ]
        for k in tensor_keys:
            out[k] = torch.stack([b[k] for b in batch])
        out["gt_answers"] = [b["gt_answers"] for b in batch]
        return out

    def train_step(self, model, batch, device):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        images_seq_mask = batch["images_seq_mask"].to(device)
        images_emb_mask = batch["images_emb_mask"].to(device)
        labels = batch["labels"].to(device)

        # Prepare multimodal embeddings
        inputs_embeds = model.prepare_inputs_embeds(
            input_ids=input_ids,
            pixel_values=pixel_values,
            images_seq_mask=images_seq_mask,
            images_emb_mask=images_emb_mask,
        )

        outputs = model.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs.loss

    @torch.no_grad()
    def evaluate(self, model, dataloader, processor, device):
        model.eval()
        tokenizer = processor.tokenizer
        total_score, total = 0.0, 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            images_seq_mask = batch["images_seq_mask"].to(device)
            images_emb_mask = batch["images_emb_mask"].to(device)

            inputs_embeds = model.prepare_inputs_embeds(
                input_ids=input_ids,
                pixel_values=pixel_values,
                images_seq_mask=images_seq_mask,
                images_emb_mask=images_emb_mask,
            )

            gen_ids = model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=10,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )

            preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            for p, gt in zip(preds, batch["gt_answers"]):
                # Take last line as the answer (after assistant prompt)
                answer = p.strip().split("\n")[-1].strip()
                total_score += vqa_soft_score(answer, gt)
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


register_backend("januspro", JanusProBackend())
