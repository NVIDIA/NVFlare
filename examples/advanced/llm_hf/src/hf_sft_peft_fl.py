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
import os

# Add deterministic seed for reproducibility illustration
import random

import datasets
import numpy as np
import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, utils
from transformers import AutoModelForCausalLM, trainer_utils
from trl import SFTConfig, SFTTrainer

import nvflare.client as flare

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


def format_instruction(example):
    output_texts = []
    for i in range(len(example["input"])):
        text = f"### Instruction: Generate Output according to the information and question given by Input. ### Input:{example['input'][i]} ### Response: {example['output'][i]}"
        output_texts.append(text)
    return output_texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="meta-llama/llama-3.2-1b",
    )
    parser.add_argument(
        "--data_path_train",
        type=str,
        default="./dataset/dolly/training.jsonl",
    )
    parser.add_argument(
        "--data_path_valid",
        type=str,
        default="./dataset/dolly/validation.jsonl",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./workspace_federated/llama-3.2-1b-dolly-sft",
    )
    parser.add_argument(
        "--train_mode",
        type=str,
        default="SFT",
        help="training mode, SFT or PEFT, default to SFT",
    )
    parser.add_argument(
        "--message_mode",
        type=str,
        default="numpy",
        help="message mode, numpy or tensor, default to numpy",
    )
    parser.add_argument("--local_epoch", type=int, default=1)
    parser.add_argument("--clean_up", type=int, default=0)
    args = parser.parse_args()

    # Dataset
    dataset_train = datasets.load_dataset("json", data_files=args.data_path_train, split="train")
    dataset_valid = datasets.load_dataset("json", data_files=args.data_path_valid, split="train")
    # Print dataset info
    print(f"Dataset size: training {len(dataset_train)}, validation {len(dataset_valid)}")
    # record every 5% of the dataset
    batch_size = 4
    gra_accu_steps = 10
    logging_steps = int(len(dataset_train) / (20 * batch_size * gra_accu_steps))
    print(f"logging_steps: {logging_steps}")

    # Model configs
    model_name_or_path = args.model_name_or_path
    peft_config = None

    # Load model
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        use_cache=False,
        torch_dtype=torch.bfloat16,
    )
    torch.set_default_dtype(default_dtype)

    # Train mode
    if args.train_mode.lower() == "sft":
        train_mode = 0
    elif args.train_mode.lower() == "peft":
        train_mode = 1
    else:
        raise ValueError(f"Invalid train_mode: {args.train_mode}, only SFT and PEFT are supported.")

    # PEFT specific
    if train_mode:
        # PEFT configs
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
    model.config.pretraining_tp = 1

    # Training arguments
    train_args = SFTConfig(
        output_dir=args.output_path,
        num_train_epochs=args.local_epoch,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gra_accu_steps,
        gradient_checkpointing=False,
        optim="paged_adamw_32bit",
        logging_steps=logging_steps,
        save_strategy="epoch",
        learning_rate=5e-4,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        disable_tqdm=True,
        max_seq_length=1024,
        save_total_limit=2,
        # safetensors has some issues in saving lm_head.weight, disable it for now
        save_safetensors=False,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset_train,
        eval_dataset=dataset_valid,
        peft_config=peft_config,
        formatting_func=format_instruction,
        args=train_args,
    )

    # initializes NVFlare client API
    flare.init()

    # Train federated rounds
    # start with global model at the beginning of each round
    while flare.is_running():
        # receives FLModel from NVFlare
        input_model = flare.receive()
        curr_round = input_model.current_round
        print(f"current_round={curr_round}")

        # Update the key name received from global model if using model def file
        global_model = copy.deepcopy(input_model.params)
        for key in list(global_model.keys()):
            global_model[key.replace("model.", "", 1)] = global_model.pop(key)

        # wraps evaluation logic into a method to re-use for
        # evaluation on both trained and received model
        def evaluate(input_weights, mode):
            # Special load func for PEFT
            if train_mode:
                set_peft_model_state_dict(trainer.model, input_weights)
            else:
                trainer.model.load_state_dict(input_weights)
            metric_score = trainer.evaluate()
            print(f"Evaluation metric score: {metric_score}")
            return metric_score

        # evaluate on received global model
        eval_loss = evaluate(global_model, train_mode)
        eval_loss = float(eval_loss["eval_loss"])

        # Load global model and previous training states
        # Since we perform iterative training by using "resume" functionality
        # we need to replace the resume weights with global weights every round
        if curr_round == 0:
            # First round, start from pretrained model
            trainer.train()
        else:
            # replace local resume weights with global weights
            resume_from_checkpoint_folder = trainer_utils.get_last_checkpoint(trainer.args.output_dir)
            if train_mode:
                # PEFT model small, directly save via torch.save
                resume_model_file_path = os.path.join(resume_from_checkpoint_folder, utils.WEIGHTS_NAME)
                torch.save(global_model, resume_model_file_path)
            else:
                # SFT model can be large, save via HF API
                # Disable safetensor for now
                trainer.model.save_pretrained(resume_from_checkpoint_folder, safe_serialization=False)
            # increment num_train_epochs so that the trainer will continue training
            if args.clean_up:
                # runner got cleaned up, set num_train_epochs with curr_round
                trainer.args.num_train_epochs = (curr_round + 1) * args.local_epoch
            else:
                # runner still alive, increment num_train_epochs with local_epoch
                trainer.args.num_train_epochs += args.local_epoch
            print(f"Increment num_train_epochs to {trainer.args.num_train_epochs}")
            # continue training
            trainer.train(resume_from_checkpoint=True)

        # compose output model to send back to server
        if train_mode:
            # PEFT, load PEFT part from trainer model
            out_param = get_peft_model_state_dict(trainer.model)
        else:
            # SFT, load whole model state_dict
            out_param = trainer.model.state_dict()

        # update the key name sent to global model
        if not train_mode:
            for key in list(out_param.keys()):
                out_param["model." + key] = out_param.pop(key).cpu()

        if args.message_mode.lower() == "numpy":
            # cast out_param to float32 preparing for communication with numpy
            # otherwise do nothing
            out_param = {k: v.to(torch.float32) for k, v in out_param.items()}

        # construct trained FL model
        output_model = flare.FLModel(
            params=out_param,
            metrics={"eval_loss": eval_loss},
            meta={"NUM_STEPS_CURRENT_ROUND": trainer.train_dataset.num_rows},
        )
        # send model back to NVFlare
        flare.send(output_model)


if __name__ == "__main__":
    main()
