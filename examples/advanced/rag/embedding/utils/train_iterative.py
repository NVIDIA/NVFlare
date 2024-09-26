# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import argparse

from datasets import load_dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers


def main():
    # argparse
    parser = argparse.ArgumentParser(description="Train a model on a dataset")
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/mpnet-base",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="nli",
    )
    args = parser.parse_args()
    model_name = args.model_name
    dataset_name = args.dataset_name

    # Load a model to finetune with
    model = SentenceTransformer(model_name)

    # Load training datasets
    # (anchor, positive, negative)
    dataset_nli_train = load_dataset("sentence-transformers/all-nli", "triplet", split="train[:16000]")
    # (question, answer)
    dataset_squad_train = load_dataset("sentence-transformers/squad", split="train[:16000]")
    # (anchor, positive)
    dataset_quora_train = load_dataset("sentence-transformers/quora-duplicates", "pair", split="train[:16000]")

    # Combine all datasets into a dictionary with dataset names to datasets
    dataset_all_train = {
        "all-nli-triplet": dataset_nli_train,
        "squad": dataset_squad_train,
        "quora": dataset_quora_train,
    }

    # Load validation datasets
    # (anchor, positive, negative)
    dataset_nli_val = load_dataset("sentence-transformers/all-nli", "triplet", split="dev")
    # (question, answer)
    dataset_squad_val = load_dataset("sentence-transformers/squad", split="train[16000:18000]")
    # (anchor, positive)
    dataset_quora_val = load_dataset("sentence-transformers/quora-duplicates", "pair", split="train[16000:18000]")
    # Combine all datasets into a dictionary with dataset names to datasets
    dataset_all_val = {
        "all-nli-triplet": dataset_nli_val,
        "squad": dataset_squad_val,
        "quora": dataset_quora_val,
    }

    # Load loss function
    loss_mnrl = MultipleNegativesRankingLoss(model)
    # Create a mapping with dataset names to loss functions
    loss_all = {
        "all-nli-triplet": loss_mnrl,
        "squad": loss_mnrl,
        "quora": loss_mnrl,
    }

    if dataset_name == "all":
        dataset_train = dataset_all_train
        dataset_val = dataset_all_val
        loss_func = loss_all
    elif dataset_name == "nli":
        dataset_train = dataset_nli_train
        dataset_val = dataset_nli_val
        loss_func = loss_mnrl
    elif dataset_name == "squad":
        dataset_train = dataset_squad_train
        dataset_val = dataset_squad_val
        loss_func = loss_mnrl
    elif dataset_name == "quora":
        dataset_train = dataset_quora_train
        dataset_val = dataset_quora_val
        loss_func = loss_mnrl
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    base_model_name = model_name.split("/")[-1]
    output_dir = f"/tmp/embed/cen/models_iter/{base_model_name}-{dataset_name}"
    unit_train_epochs = 0.25
    # Specify training arguments
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=output_dir,
        # Optional training parameters:
        num_train_epochs=unit_train_epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=1e-6,
        lr_scheduler_type="constant",
        bf16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=1,
        # logging parameters:
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=50,
        report_to="tensorboard",
    )

    # Define trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        loss=loss_func,
    )

    for round in range(6):
        # Train the model
        # First round: start from scratch
        if round == 0:
            trainer.train()
        # Subsequent rounds: start from the previous model
        else:
            args.num_train_epochs += unit_train_epochs
            trainer.train(resume_from_checkpoint=True)

    # Save the trained model
    model.save_pretrained(f"{output_dir}/final")


if __name__ == "__main__":
    main()
