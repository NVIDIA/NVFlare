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

"""
HuggingFace LLM client for multi-GPU federated learning.

Supports both SFT (Supervised Fine-Tuning) and PEFT (Parameter-Efficient Fine-Tuning).
Launch with:
    - Single GPU: python client.py [args]
    - Multi GPU: python -m torch.distributed.run --nnodes=1 --nproc_per_node=N --master_port=7777 client.py [args]
    - Multi-node: via client_wrapper.sh
"""

import argparse
import math
import os

# Add deterministic seed for reproducibility illustration
import random

import datasets
import numpy as np
import torch
import torch.distributed as dist
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer

# (1) import NVFlare HuggingFace Client API
import nvflare.client.hf as flare

# set deterministic seed for reproducibility
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


def format_instruction(example):
    """Format training examples for instruction tuning."""
    return f"### Instruction: Generate Output according to the information and question given by Input. ### Input:{example['input']} ### Response: {example['output']}"


def setup_distributed_training():
    """Setup distributed training environment.

    Returns:
        tuple: (rank, world_size, local_rank)
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # Set device for DDP
        torch.cuda.set_device(local_rank)

        print(f"DDP rank {rank} initialized: world_size={world_size}, local_rank={local_rank}")
        return rank, world_size, local_rank
    else:
        print("No distributed training environment detected, running in single GPU mode")
        return 0, 1, 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="EleutherAI/gpt-neo-1.3B",
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
        default="./workspace_federated/gpt-neo-1.3b-dolly-sft",
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
    parser.add_argument("--local_epoch", type=int, default=1)
    parser.add_argument("--num_rounds", type=int, default=3)
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="nvflare_llm",
        help="WandB project name (default: nvflare_llm). WandB is enabled if WANDB_API_KEY is set.",
    )
    parser.add_argument(
        "--wandb_run_name", type=str, default="nvflare_llm", help="WandB run name, default to nvflare_llm"
    )
    args = parser.parse_args()

    # Setup distributed training
    rank, world_size, local_rank = setup_distributed_training()

    # Set up device for DDP
    # Use local_rank which is 0-7 on each node, not global rank which is 0-15 across all nodes
    device_map = {"": local_rank}

    # Optimize PyTorch memory allocation to reduce fragmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Set WandB environment variables (only if API key is set and rank 0 will actually log)
    wandb_enabled = bool(os.environ.get("WANDB_API_KEY"))
    if wandb_enabled:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        os.environ["WANDB_NAME"] = args.wandb_run_name
        # Add FL-specific tags
        os.environ["WANDB_TAGS"] = (
            f"nvflare,multi-node,{world_size}gpus,{os.environ.get('SLURM_JOB_NUM_NODES', '1')}nodes"
        )
        if rank == 0:
            print(f"Rank 0: WandB enabled - project: {args.wandb_project}, run: {args.wandb_run_name}")
    elif rank == 0:
        print("Rank 0: WandB disabled (WANDB_API_KEY not set), using TensorBoard for logging")

    # Create output path on rank 0; do not remove checkpoints because flare.patch()
    # uses them for round-to-round Trainer state restoration.
    if rank == 0:
        os.makedirs(args.output_path, exist_ok=True)

    # Wait for main process to finish output directory setup
    if dist.is_initialized():
        dist.barrier()

    # Dataset
    dataset_train = datasets.load_dataset("json", data_files=args.data_path_train, split="train")
    dataset_valid = datasets.load_dataset("json", data_files=args.data_path_valid, split="train")
    # Print dataset info
    if rank == 0:
        print(f"Dataset size: training {len(dataset_train)}, validation {len(dataset_valid)}")
    # record every 5% of the dataset
    # Adjust batch size based on training mode
    batch_size = 2 if args.train_mode.lower() == "sft" else 4
    gra_accu_steps = 20 if args.train_mode.lower() == "sft" else 10
    logging_steps = max(1, int(len(dataset_train) / (20 * batch_size * gra_accu_steps)))
    if rank == 0:
        print(f"logging_steps: {logging_steps}")

    # Model configs
    model_name_or_path = args.model_name_or_path

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
        # Don't wrap model with get_peft_model here - let SFTTrainer handle it
        # This is required for TRL >= 0.18 compatibility
    else:
        peft_config = None
    model.config.pretraining_tp = 1

    # Calculate warmup_steps (replacing deprecated warmup_ratio for future compatibility)
    # Total training steps = (dataset_size / (batch_size * grad_accum * world_size)) * num_epochs
    # Use ceiling division to ensure at least 1 step per epoch for small datasets
    steps_per_epoch = math.ceil(len(dataset_train) / (batch_size * gra_accu_steps * world_size))
    total_train_steps = steps_per_epoch * (args.local_epoch * args.num_rounds)
    warmup_steps = int(total_train_steps * 0.03)  # 3% warmup

    # Set TensorBoard logging directory via environment variable
    if not wandb_enabled:
        os.environ["TENSORBOARD_LOGGING_DIR"] = os.path.join(args.output_path, "logs")

    # Training arguments
    train_args = SFTConfig(
        output_dir=args.output_path,
        # flare.patch() treats num_train_epochs as the per-round local budget and
        # expands max_steps from the server's total_rounds at the first train call.
        num_train_epochs=args.local_epoch,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gra_accu_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # optimizers using bitsandbytes like "paged_adamw_32bit" have an issue with
        # multi-gpu training, to be consistent, use regular optimizer
        optim="adamw_torch",
        logging_steps=logging_steps,
        save_strategy="epoch",
        learning_rate=5e-4,
        bf16=True,
        max_grad_norm=0.3,
        warmup_steps=warmup_steps,  # Using warmup_steps instead of deprecated warmup_ratio
        # use cosine_with_restarts scheduler to check the iterative behavior
        lr_scheduler_type=args.lr_scheduler,
        lr_scheduler_kwargs={"num_cycles": 2},
        disable_tqdm=True,
        max_length=1024,
        save_total_limit=2,
        seed=0,
        data_seed=0,
        # Multi-GPU specific settings
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=False,
        # WandB integration (if API key is set), otherwise use TensorBoard for logging
        # HuggingFace Trainer automatically handles multi-process logging (only rank 0 logs)
        report_to="wandb" if wandb_enabled else "tensorboard",
        run_name=args.wandb_run_name if wandb_enabled else None,
    )

    # Trainer
    # For PEFT mode, pass base model + peft_config, let SFTTrainer handle PEFT wrapping
    # For SFT mode, pass model directly without peft_config
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset_train,
        eval_dataset=dataset_valid,
        peft_config=peft_config,
        formatting_func=format_instruction,
        args=train_args,
    )

    # Verify PEFT wrapping in PEFT mode
    if train_mode and not isinstance(trainer.model, PeftModel):
        raise RuntimeError(
            "PEFT mode is enabled but trainer.model is not a PeftModel. "
            "SFTTrainer may have failed to wrap the model with PEFT."
        )

    # (2) Patch the Trainer for federated rounds.
    # Rank 0 uses the Client API; other ranks receive task/control data through torch.distributed.
    flare.patch(trainer, server_key_prefix=None if train_mode else "model.")

    # (3) Train federated rounds. For train tasks, evaluate() captures metrics for server-side model selection,
    # and train() loads the global model, runs the local budget, and sends the result on rank 0.
    while flare.is_running():
        metrics = trainer.evaluate()
        if rank == 0 and metrics and "eval_loss" in metrics:
            print(f"Rank 0: Global model eval_loss: {float(metrics['eval_loss'])}")
        trainer.train()

    # Cleanup distributed training environment
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
