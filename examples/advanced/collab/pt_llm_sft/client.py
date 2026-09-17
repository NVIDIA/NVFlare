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

"""Stateful client-side supervised fine-tuning for the Collab example."""

import inspect
import random
from pathlib import Path

import torch
from model import cpu_model_state, format_example, load_model_and_tokenizer, precision_config

from nvflare.collab import collab


def filtered_kwargs(callable_obj, values: dict) -> dict:
    """Keep the example compatible with the SFTConfig options available locally."""
    parameters = inspect.signature(callable_obj).parameters
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return values
    return {name: value for name, value in values.items() if name in parameters}


class LLMSFTClient:
    """Own the model, data iterator, and optimizer that persist on each client."""

    def __init__(
        self,
        model_name_or_path: str,
        data_root: str,
        output_root: str,
        syncs_per_epoch: int,
        learning_rate: float,
        max_length: int,
        evaluate_global_model: bool,
        model_revision: str | None = None,
        trust_remote_code: bool = False,
        precision: str = "auto",
        site_name: str | None = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.model_revision = model_revision
        self.trust_remote_code = trust_remote_code
        self.data_root = data_root
        self.output_root = output_root
        self.syncs_per_epoch = syncs_per_epoch
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.evaluate_global_model = evaluate_global_model
        self.precision = precision
        self.site_name = site_name
        self._uses_collab_context = site_name is None
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.eval_dataset = None
        self.sft_trainer = None
        self.train_dataloader = None
        self.train_iterator = None
        self.optimizer = None
        self.completed_syncs = 0

    # @collab.init runs once in each client process before published methods can be called.
    @collab.init
    def initialize(self) -> None:
        import datasets
        import numpy as np

        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

        # collab.site_name identifies which runtime site owns this replicated client object.
        if self.site_name is None:
            self.site_name = collab.site_name
        site_data = Path(self.data_root) / self.site_name
        train_path = site_data / "train.jsonl"
        valid_path = site_data / "valid.jsonl"
        if not train_path.is_file() or not valid_path.is_file():
            raise FileNotFoundError(f"missing prepared data under {site_data}; run prepare_data.py first")

        self.train_dataset = datasets.load_dataset("json", data_files=str(train_path), split="train")
        self.eval_dataset = datasets.load_dataset("json", data_files=str(valid_path), split="train")
        self.model, self.tokenizer = load_model_and_tokenizer(
            model_name_or_path=self.model_name_or_path,
            model_revision=self.model_revision,
            trust_remote_code=self.trust_remote_code,
            precision=self.precision,
        )
        self.sft_trainer = self.make_trainer()
        self.train_dataloader = self.sft_trainer.get_train_dataloader()
        if len(self.train_dataloader) < self.syncs_per_epoch:
            raise ValueError(
                f"{self.site_name} has {len(self.train_dataloader)} training batches, "
                f"fewer than --syncs-per-epoch={self.syncs_per_epoch}"
            )
        self.train_iterator = iter(self.train_dataloader)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        trainable = sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad)
        print(
            f"[{self.site_name}] loaded {self.model_name_or_path}: {trainable:,} parameters trainable, "
            f"{len(self.train_dataloader)} batches per epoch"
        )

    def make_sft_config(self):
        from trl import SFTConfig

        bf16_enabled, fp16_enabled, _ = precision_config(self.precision)
        output_dir = Path(self.output_root) / self.site_name
        values = {
            "output_dir": str(output_dir),
            "num_train_epochs": 1,
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": self.learning_rate,
            "lr_scheduler_type": "constant",
            "logging_strategy": "steps",
            "logging_steps": 1,
            "save_strategy": "no",
            "disable_tqdm": True,
            "report_to": [],
            "max_length": self.max_length,
            "bf16": bf16_enabled,
            "fp16": fp16_enabled,
            "use_cpu": not torch.cuda.is_available(),
            "remove_unused_columns": False,
            "seed": 0,
            "data_seed": 0,
        }
        return SFTConfig(**filtered_kwargs(SFTConfig.__init__, values))

    def make_trainer(self):
        from trl import SFTTrainer

        values = {
            "model": self.model,
            "args": self.make_sft_config(),
            "train_dataset": self.train_dataset,
            "eval_dataset": self.eval_dataset,
            "processing_class": self.tokenizer,
            "tokenizer": self.tokenizer,
            "formatting_func": format_example,
        }
        return SFTTrainer(**filtered_kwargs(SFTTrainer.__init__, values))

    # @collab.publish exposes this method through the server's client proxy.
    @collab.publish
    def get_model_state(self) -> dict[str, torch.Tensor]:
        """Return the initial full model as native PyTorch tensors."""
        return cpu_model_state(self.model)

    # @collab.publish exposes train so collab.clients.train(...) can invoke it remotely.
    @collab.publish
    def train(self, sync_number: int, global_weights: dict[str, torch.Tensor]) -> dict:
        # collab.is_aborted lets local work stop promptly when the federated job is aborted.
        if self._uses_collab_context and collab.is_aborted:
            raise RuntimeError("training aborted")
        if sync_number != self.completed_syncs + 1:
            raise ValueError(f"expected sync {self.completed_syncs + 1}, received {sync_number}")

        self.model.load_state_dict(global_weights, strict=True)
        eval_loss = None
        if self.evaluate_global_model:
            metrics = self.sft_trainer.evaluate()
            eval_loss = float(metrics["eval_loss"])

        sync_in_epoch = (sync_number - 1) % self.syncs_per_epoch
        steps_before = sync_in_epoch * len(self.train_dataloader) // self.syncs_per_epoch
        steps_after = (sync_in_epoch + 1) * len(self.train_dataloader) // self.syncs_per_epoch
        local_steps = steps_after - steps_before

        self.model.train()
        loss_sum = 0.0
        num_examples = 0
        for _ in range(local_steps):
            # Recheck the abort signal between local steps so a long interval can stop early.
            if self._uses_collab_context and collab.is_aborted:
                raise RuntimeError("training aborted")
            try:
                batch = next(self.train_iterator)
            except StopIteration:
                self.train_iterator = iter(self.train_dataloader)
                batch = next(self.train_iterator)
            batch = {
                name: value.to(self.model.device) if isinstance(value, torch.Tensor) else value
                for name, value in batch.items()
            }
            self.optimizer.zero_grad(set_to_none=True)
            loss = self.model(**batch).loss
            loss.backward()
            self.optimizer.step()
            loss_sum += float(loss.detach())
            num_examples += int(batch["input_ids"].shape[0])

        self.completed_syncs = sync_number
        train_loss = loss_sum / local_steps
        result = {
            "weights": cpu_model_state(self.model),
            "num_examples": num_examples,
            "train_loss": train_loss,
            "eval_loss": eval_loss,
        }
        message = f"[{self.site_name}] sync={sync_number} local_steps={local_steps} train_loss={train_loss:.4f}"
        if eval_loss is not None:
            message += f" global_eval_loss={eval_loss:.4f}"
        print(message)
        return result
