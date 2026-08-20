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

import numpy as np
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

MODEL_NAME = "hf-internal-testing/tiny-random-distilbert"


def metrics(prediction):
    logits, labels = prediction
    return {"accuracy": float((np.argmax(logits, axis=-1) == labels).mean())}


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    data = load_dataset("json", data_files={"train": "train.jsonl", "validation": "valid.jsonl"})
    data = data.map(
        lambda rows: tokenizer(rows["text"], truncation=True, padding="max_length", max_length=32), batched=True
    )
    arguments = TrainingArguments("outputs", num_train_epochs=1, per_device_train_batch_size=2, report_to=[])
    trainer = Trainer(
        model=AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2),
        args=arguments,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        compute_metrics=metrics,
    )
    trainer.evaluate()
    trainer.train()
