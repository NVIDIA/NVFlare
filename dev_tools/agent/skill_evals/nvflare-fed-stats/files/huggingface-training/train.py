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

from datasets import Dataset
from model import create_model
from transformers import AutoTokenizer, Trainer, TrainingArguments


def main():
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
    dataset = Dataset.from_dict({"text": ["alpha", "beta"], "label": [0, 1]})
    tokenized = dataset.map(lambda row: tokenizer(row["text"], truncation=True), batched=True)
    trainer = Trainer(
        model=create_model(),
        args=TrainingArguments(output_dir="outputs", num_train_epochs=1, report_to=[]),
        train_dataset=tokenized,
    )
    trainer.train()


if __name__ == "__main__":
    main()
