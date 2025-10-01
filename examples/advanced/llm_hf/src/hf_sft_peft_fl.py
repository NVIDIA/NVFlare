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
import shutil

import datasets
import numpy as np
import torch
import torch.distributed as dist
from accelerate import PartialState
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, utils
from transformers import AutoModelForCausalLM, TrainerCallback, trainer_utils
from trl import SFTConfig, SFTTrainer

import nvflare.client as flare


# Add callback to stop at each epoch
class StopCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, logs=None, **kwargs):
        control.should_training_stop = True


# set deterministic seed for reproducibility
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


def format_instruction(example):
    return f"### Instruction: Generate Output according to the information and question given by Input. ### Input:{example['input']} ### Response: {example['output']}"


def setup_distributed_training():
    """Setup distributed training environment."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # Set device
        torch.cuda.set_device(local_rank)

        print(f"Distributed training initialized: rank={rank}, world_size={world_size}, local_rank={local_rank}")
        return rank, world_size, local_rank
    else:
        print("No distributed training environment detected, running in single GPU mode")
        return 0, 1, 0


def cleanup_distributed_training():
    """Cleanup distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


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
        "--lr_scheduler",
        type=str,
        default="constant",
        help="learning rate scheduler type, default to 'constant'",
    )
    parser.add_argument(
        "--message_mode",
        type=str,
        default="numpy",
        help="message mode, numpy or tensor, default to numpy",
    )
    parser.add_argument("--local_epoch", type=int, default=1)
    parser.add_argument("--num_rounds", type=int, default=3)
    args = parser.parse_args()

    # Setup distributed training
    rank, world_size, local_rank = setup_distributed_training()

    # Set up device for DDP
    device_string = PartialState().process_index
    device_map = {"": device_string}

    # If output path exists, remove it (only on main process)
    if local_rank == 0:
        try:
            print(f"Attempting to remove output path {args.output_path}.")
            shutil.rmtree(args.output_path)
        except FileNotFoundError:
            print(f"Output path {args.output_path} does not exist, skipping removal.")

    # Wait for main process to finish cleanup
    if dist.is_initialized():
        dist.barrier()

    # Dataset
    dataset_train = datasets.load_dataset("json", data_files=args.data_path_train, split="train")
    dataset_valid = datasets.load_dataset("json", data_files=args.data_path_valid, split="train")
    # Print dataset info
    if local_rank == 0:
        print(f"Dataset size: training {len(dataset_train)}, validation {len(dataset_valid)}")
    # record every 5% of the dataset
    batch_size = 4
    gra_accu_steps = 10
    logging_steps = int(len(dataset_train) / (20 * batch_size * gra_accu_steps))
    if local_rank == 0:
        print(f"logging_steps: {logging_steps}")

    # Model configs
    model_name_or_path = args.model_name_or_path
    peft_config = None

    # Load model with device_map
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map=device_map,
        use_cache=False,
        dtype=torch.bfloat16,
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
        # Using callback, stop at each epoch, so specify num_train_epochs
        # the same as the total epoch in one-call training
        num_train_epochs=args.local_epoch * args.num_rounds,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gra_accu_steps,
        gradient_checkpointing=False,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # optimizers using bitsandbytes like "paged_adamw_32bit" have an issue with
        # multi-gpu training, to be consistent, use regular optimizer
        optim="adamw_torch",
        logging_steps=logging_steps,
        save_strategy="epoch",
        learning_rate=5e-4,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        # use cosine_with_restarts scheduler to check the iterative behavior
        lr_scheduler_type=args.lr_scheduler,
        lr_scheduler_kwargs={"num_cycles": 2},
        disable_tqdm=True,
        max_length=1024,
        save_total_limit=2,
        # safetensors will remove shared layers, e.g. lm_head.weight
        # disable for local checkpointing
        save_safetensors=False,
        seed=0,
        data_seed=0,
        # Multi-GPU specific settings
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=False,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset_train,
        eval_dataset=dataset_valid,
        peft_config=peft_config,
        formatting_func=format_instruction,
        args=train_args,
        # Add a callback to stop training after one epoch
        callbacks=[StopCallback()],
    )

    # initializes NVFlare client API
    flare.init()

    # Train federated rounds
    # start with global model at the beginning of each round
    while flare.is_running():
        # receives golobal model from NVFlare (only on main process)
        if local_rank == 0:
            input_model = flare.receive()
            curr_round = input_model.current_round
            print(f"current_round={curr_round}")
            # Update the key name received from global model if using model def file
            global_model = copy.deepcopy(input_model.params)
            for key in list(global_model.keys()):
                global_model[key.replace("model.", "", 1)] = global_model.pop(key)
        else:
            curr_round = None
            global_model = None

        # broadcast current round and global_model to all processes
        if dist.is_initialized():
            curr_round_list = [curr_round]
            global_model_list = [global_model]
            dist.broadcast_object_list(curr_round_list, src=0)
            dist.broadcast_object_list(global_model_list, src=0)
            curr_round = curr_round_list[0]
            global_model = global_model_list[0]

        if dist.is_initialized():
            dist.barrier()

        # Load state dict
        if train_mode:
            set_peft_model_state_dict(trainer.model, global_model)
        else:
            trainer.model.load_state_dict(global_model)
        # Wait for main process to finish model loading
        if dist.is_initialized():
            dist.barrier()

        # Evaluate the global model
        eval_loss = trainer.evaluate()
        eval_loss = float(eval_loss["eval_loss"])

        # Train
        if curr_round == 0:
            # First round, start from pretrained model
            for epoch in range(args.local_epoch):
                print(f"Training local epoch {epoch + 1}/{args.local_epoch}")
                # train for one epoch
                if epoch == 0:
                    trainer.train()
                else:
                    # continue training
                    trainer.train(resume_from_checkpoint=True)
        else:
            # replace local resume weights with global weights (only on main process)
            if local_rank == 0:
                resume_from_checkpoint_folder = trainer_utils.get_last_checkpoint(trainer.args.output_dir)
                if train_mode:
                    # PEFT model small, directly save via torch.save
                    resume_model_file_path = os.path.join(resume_from_checkpoint_folder, utils.WEIGHTS_NAME)
                    torch.save(global_model, resume_model_file_path)
                else:
                    # SFT model can be large, save via HF API
                    # Disable safetensor for now
                    trainer.model.save_pretrained(
                        resume_from_checkpoint_folder, state_dict=global_model, safe_serialization=False
                    )

            # Wait for main process to finish saving before continuing
            if dist.is_initialized():
                dist.barrier()

            # continue training
            # as we used callback, no need to increment num_train_epochs
            for epoch in range(args.local_epoch):
                print(f"Training local epoch {epoch + 1}/{args.local_epoch}")
                trainer.train(resume_from_checkpoint=True)

        # Wait for all process to finish training before continuing
        if dist.is_initialized():
            dist.barrier()

        # compose output model to send back to server (only on main process)
        if local_rank == 0:
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

            # print the dict size
            print(f"In total {len(out_param.keys())} params to be sent to server.")

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
