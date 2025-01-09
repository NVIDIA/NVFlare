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
import copy

from datasets import load_dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from transformers import trainer_utils

import nvflare.client as flare


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
    if dataset_name == "nli":
        # (anchor, positive, negative)
        dataset_train = load_dataset("sentence-transformers/all-nli", "triplet", split="train[:16000]")
        dataset_val = load_dataset("sentence-transformers/all-nli", "triplet", split="dev")
    elif dataset_name == "squad":
        # (question, answer)
        dataset_train = load_dataset("sentence-transformers/squad", split="train[:16000]")
        dataset_val = load_dataset("sentence-transformers/squad", split="train[16000:18000]")
    elif dataset_name == "quora":
        # (anchor, positive)
        dataset_train = load_dataset("sentence-transformers/quora-duplicates", "pair", split="train[:16000]")
        dataset_val = load_dataset("sentence-transformers/quora-duplicates", "pair", split="train[16000:18000]")
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    # Load loss function
    loss = MultipleNegativesRankingLoss(model)

    base_model_name = model_name.split("/")[-1]
    output_dir = f"./models/{base_model_name}-{dataset_name}"
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
        loss=loss,
    )

    # initializes NVFlare client API
    flare.init()

    while flare.is_running():
        # receives FLModel from NVFlare
        input_model = flare.receive()
        curr_round = input_model.current_round
        print(f"current_round={curr_round}")

        # Update the key name received from global model if using model def file
        global_model = copy.deepcopy(input_model.params)
        for key in list(global_model.keys()):
            global_model[key.replace("model.", "", 1)] = global_model.pop(key)

        # evaluate on received global model
        trainer.model.load_state_dict(global_model)
        eval_loss_dict = trainer.evaluate()
        eval_loss = float(eval_loss_dict["eval_loss"])
        print(f"Evaluation loss: {eval_loss}")
        # Save the global model
        model.save_pretrained(f"{output_dir}/global")

        # Train the model
        if curr_round == 0:
            # First round: start from scratch
            trainer.train()
        else:
            # Subsequent rounds: start from the previous model
            # Since we perform iterative training by using "resume" functionality
            # we need to replace the resume weights with global weights every round
            resume_from_checkpoint_folder = trainer_utils.get_last_checkpoint(trainer.args.output_dir)
            # update local record with global model weights
            trainer.model.save_pretrained(resume_from_checkpoint_folder)
            # increment the number of training epochs so that the trainer will continue training
            args.num_train_epochs += unit_train_epochs
            # continue training
            trainer.train(resume_from_checkpoint=True)

        # update the key name sent to global model
        out_param = trainer.model.state_dict()
        for key in list(out_param.keys()):
            out_param["model." + key] = out_param.pop(key).cpu()
        num_steps = trainer.train_dataset.num_rows * unit_train_epochs

        # construct trained FL model
        output_model = flare.FLModel(
            params=out_param,
            metrics={"eval_loss": eval_loss},
            meta={"NUM_STEPS_CURRENT_ROUND": num_steps},
        )
        # send model back to NVFlare
        flare.send(output_model)


if __name__ == "__main__":
    main()
