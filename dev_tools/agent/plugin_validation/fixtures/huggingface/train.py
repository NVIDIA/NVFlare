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

"""Tiny locally initialized BERT and synthetic token IDs; no Hub access."""

import json

import torch
from torch.utils.data import Dataset
from transformers import BertConfig, BertForSequenceClassification, Trainer, TrainingArguments


class Tokens(Dataset):
    def __len__(self):
        return 16

    def __getitem__(self, index):
        return {
            "input_ids": torch.tensor([(index + offset) % 30 + 1 for offset in range(8)]),
            "attention_mask": torch.ones(8, dtype=torch.long),
            "labels": torch.tensor(index % 2),
        }


def create_model():
    return BertForSequenceClassification(
        BertConfig(
            vocab_size=32,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=16,
            num_labels=2,
        )
    )


def main():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    trainer = Trainer(
        model=create_model(),
        args=TrainingArguments(
            output_dir="outputs",
            use_cpu=True,
            max_steps=2,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            report_to=[],
            save_strategy="no",
            disable_tqdm=True,
            seed=7,
        ),
        train_dataset=Tokens(),
        eval_dataset=Tokens(),
    )
    trainer.train()
    result = trainer.evaluate()
    print(json.dumps({"framework": "huggingface", "loss": result["eval_loss"], "examples": len(Tokens())}))


if __name__ == "__main__":
    main()
