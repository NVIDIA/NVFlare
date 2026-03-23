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
"""
Client script for federated MedGemma fine-tuning with TRL SFTTrainer and LoRA-only exchange.
"""

from __future__ import annotations

import argparse
import gc
import os
import shutil
import signal
import sys

import torch
from data_utils import DEFAULT_MODEL_NAME_OR_PATH, format_training_example, resolve_image_path
from datasets import Image, load_dataset
from model import MEDGEMMA_IMAGE_TOKEN_ID, apply_adapter_state, create_peft_medgemma_model, get_adapter_state_dict
from transformers import AutoProcessor
from trl import SFTConfig, SFTTrainer

import nvflare.client as flare


def _abs_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def _params_size_mb(params) -> float:
    if not params:
        return 0.0
    nbytes = 0
    for value in params.values():
        if isinstance(value, torch.Tensor):
            nbytes += value.numel() * value.element_size()
        elif hasattr(value, "nbytes"):
            nbytes += value.nbytes
    return nbytes / (1024.0 * 1024.0)


def _free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _require_supported_gpu() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires a CUDA GPU because MedGemma QLoRA uses bitsandbytes 4-bit loading.")
    if torch.cuda.get_device_capability()[0] < 8:
        raise RuntimeError("MedGemma fine-tuning requires a GPU with bfloat16 support (compute capability >= 8.0).")


def _load_site_split(json_path: str, image_root: str):
    dataset = load_dataset("json", data_files=json_path, split="train")
    dataset = dataset.map(lambda example: {"image": resolve_image_path(example["image"], image_root)})
    dataset = dataset.cast_column("image", Image())
    return dataset.map(format_training_example)


def _build_collate_fn(processor):
    tokenizer = processor.tokenizer
    pad_token_id = tokenizer.pad_token_id
    boi_token = tokenizer.special_tokens_map.get("boi_token")
    boi_token_id = tokenizer.convert_tokens_to_ids(boi_token) if boi_token else None

    def collate_fn(examples):
        texts = []
        images = []
        for example in examples:
            images.append([example["image"].convert("RGB")])
            texts.append(
                processor.apply_chat_template(example["messages"], add_generation_prompt=False, tokenize=False).strip()
            )

        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100
        if boi_token_id is not None:
            labels[labels == boi_token_id] = -100
        labels[labels == MEDGEMMA_IMAGE_TOKEN_ID] = -100
        batch["labels"] = labels
        return batch

    return collate_fn


def _build_training_args(args, output_dir: str, do_eval: bool) -> SFTConfig:
    training_args = dict(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        logging_steps=args.logging_steps,
        save_strategy="no",
        learning_rate=args.learning_rate,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="linear",
        report_to=args.report_to,
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        label_names=["labels"],
    )
    if args.max_steps is not None:
        training_args["max_steps"] = args.max_steps
    if do_eval:
        training_args["eval_strategy"] = "steps"
        training_args["eval_steps"] = args.eval_steps
    else:
        training_args["eval_strategy"] = "no"
    return SFTConfig(**training_args)


def main():
    parser = argparse.ArgumentParser(description="Federated MedGemma client with QLoRA-based local training.")
    parser.add_argument(
        "--data_path", type=str, default="./data/site-1", help="Site data directory containing train.json."
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default="./NCT-CRC-HE-100K",
        help="Root directory used to resolve image paths stored in train.json and validation.json.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=DEFAULT_MODEL_NAME_OR_PATH,
        help="MedGemma Hugging Face model ID or local path.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Optional max steps per round. If omitted, one local epoch is used.",
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=1, help="Local epochs per round when max_steps is not set."
    )
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Peak learning rate (default: 2e-4).")
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Training batch size per device (default: 4).",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=4,
        help="Evaluation batch size per device (default: 4).",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4).",
    )
    parser.add_argument("--logging_steps", type=int, default=25, help="Logging interval for trainer.train().")
    parser.add_argument("--eval_steps", type=int, default=50, help="Evaluation interval when validation data exists.")
    parser.add_argument(
        "--eval_subset_size",
        type=int,
        default=200,
        help="Maximum number of validation samples evaluated per round (default: 200). Use 0 for full validation.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="none",
        help='Trainer reporting backend, e.g. "none", "tensorboard", or "wandb".',
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default=None,
        help="Working directory for round-specific trainer outputs (default: ./medgemma_checkpoints under the client workspace).",
    )
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, lambda _signum, _frame: sys.exit(0))

    example_dir = os.path.dirname(os.path.abspath(__file__))
    if example_dir not in sys.path:
        sys.path.insert(0, example_dir)

    data_path = _abs_path(args.data_path)
    image_root = _abs_path(args.image_root)
    train_json = os.path.join(data_path, "train.json")
    validation_json = os.path.join(data_path, "validation.json")
    if not os.path.isfile(train_json):
        raise FileNotFoundError(f"Expected train.json at {train_json}. Run prepare_data.py first.")

    _require_supported_gpu()
    flare.init()
    client_name = flare.system_info().get("site_name", "unknown")

    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    processor.tokenizer.padding_side = "right"
    collate_fn = _build_collate_fn(processor)

    train_dataset = _load_site_split(train_json, image_root)
    eval_dataset = _load_site_split(validation_json, image_root) if os.path.isfile(validation_json) else None

    print(f"site={client_name}, train_samples={len(train_dataset)}, validation_samples={len(eval_dataset or [])}")
    model = create_peft_medgemma_model(model_name_or_path=args.model_name_or_path, quantized=True, device_map={"": 0})

    if args.work_dir is None:
        work_dir = os.path.join(os.getcwd(), "medgemma_checkpoints")
    else:
        work_dir = _abs_path(args.work_dir)
    os.makedirs(work_dir, exist_ok=True)

    while flare.is_running():
        input_model = flare.receive()
        if input_model is None:
            break

        current_round = input_model.current_round
        received_mb = _params_size_mb(input_model.params)
        print(f"site={client_name}, round={current_round}, received adapter size: {received_mb:.2f} MB")
        apply_adapter_state(model, input_model.params)
        model.train()

        if eval_dataset is not None and args.eval_subset_size > 0 and len(eval_dataset) > args.eval_subset_size:
            round_eval_dataset = eval_dataset.shuffle(seed=42 + current_round).select(range(args.eval_subset_size))
        else:
            round_eval_dataset = eval_dataset

        round_output_dir = os.path.join(work_dir, f"round-{current_round}")
        if os.path.isdir(round_output_dir):
            shutil.rmtree(round_output_dir)

        trainer = SFTTrainer(
            model=model,
            args=_build_training_args(args, round_output_dir, round_eval_dataset is not None),
            train_dataset=train_dataset,
            eval_dataset=round_eval_dataset,
            processing_class=processor,
            data_collator=collate_fn,
        )

        train_result = trainer.train()
        train_loss = float(getattr(train_result, "training_loss", float("nan")))
        metrics = {"loss": train_loss}
        if round_eval_dataset is not None:
            eval_metrics = trainer.evaluate()
            eval_loss = float(eval_metrics["eval_loss"])
            metrics["eval_loss"] = eval_loss
            metrics["neg_eval_loss"] = -eval_loss

        params = {"model." + key: value for key, value in get_adapter_state_dict(model).items()}
        sent_mb = _params_size_mb(params)
        meta = (
            {"NUM_STEPS_CURRENT_ROUND": args.max_steps}
            if args.max_steps is not None
            else {"NUM_TRAIN_EPOCHS_CURRENT_ROUND": args.num_train_epochs}
        )
        flare.send(flare.FLModel(params=params, metrics=metrics, meta=meta))
        print(f"site={client_name}, round={current_round}, sent updated adapter size: {sent_mb:.2f} MB")

        del trainer, params, input_model
        _free_memory()


if __name__ == "__main__":
    main()
