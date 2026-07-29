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

"""Full-parameter federated SFT with Hugging Face and the Collab API."""

import argparse
import inspect
import json
import random
from pathlib import Path

import torch

from nvflare.collab import CollabRecipe, collab, simple_logging
from nvflare.recipe import SimEnv

DEFAULT_DATA_ROOT = "/tmp/nvflare/collab/pt_llm_sft/data"
DEFAULT_OUTPUT_ROOT = "/tmp/nvflare/collab/pt_llm_sft/results"
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def format_example(example: dict) -> str:
    if example.get("text"):
        return example["text"]
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")
    if input_text:
        return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
    return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"


def filtered_kwargs(callable_obj, values: dict) -> dict:
    """Keep the example compatible with the SFTConfig options available locally."""
    parameters = inspect.signature(callable_obj).parameters
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return values
    return {name: value for name, value in values.items() if name in parameters}


def precision_config(precision: str = "auto") -> tuple[bool, bool, torch.dtype]:
    if precision not in ("auto", "float32", "bfloat16"):
        raise ValueError(f"unsupported precision: {precision}")
    cuda_available = torch.cuda.is_available()
    bf16_supported = cuda_available and bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    if precision == "bfloat16" and not bf16_supported:
        raise RuntimeError("bfloat16 precision requires a CUDA device with BF16 support")
    if precision == "bfloat16" or (precision == "auto" and bf16_supported):
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    return dtype == torch.bfloat16, False, dtype


def cpu_model_state(model) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def average_model_states(updates: dict[str, dict], min_clients: int) -> tuple[dict[str, torch.Tensor], float]:
    if len(updates) < min_clients:
        raise RuntimeError(f"received {len(updates)} successful client updates; need at least {min_clients}")

    first_update = next(iter(updates.values()))
    expected_keys = set(first_update["weights"])
    total_examples = sum(update["num_examples"] for update in updates.values())
    if total_examples <= 0:
        raise RuntimeError("client updates contain no training examples")

    for site_name, update in updates.items():
        if set(update["weights"]) != expected_keys:
            raise ValueError(f"{site_name} returned a different set of model parameters")
        if update["num_examples"] <= 0:
            raise ValueError(f"{site_name} returned an invalid example count")

    averaged = {}
    for name in sorted(expected_keys):
        reference = first_update["weights"][name]
        for site_name, update in updates.items():
            if update["weights"][name].shape != reference.shape:
                raise ValueError(f"{site_name} returned an incompatible shape for {name}")

        if reference.is_floating_point():
            accumulator = torch.zeros_like(reference, device="cpu", dtype=torch.float32)
            for update in updates.values():
                tensor = update["weights"][name].detach().to(device="cpu", dtype=torch.float32)
                accumulator.add_(tensor, alpha=update["num_examples"])
            averaged[name] = (accumulator / total_examples).to(dtype=reference.dtype)
        else:
            averaged[name] = reference.detach().cpu().clone()

    average_loss = sum(update["train_loss"] * update["num_examples"] for update in updates.values()) / total_examples
    return averaged, average_loss


class LLMSFTClient:
    def __init__(
        self,
        model_name_or_path: str,
        data_root: str,
        output_root: str,
        syncs_per_epoch: int,
        learning_rate: float,
        max_length: int,
        evaluate_global_model: bool,
        precision: str = "auto",
        site_name: str | None = None,
    ):
        self.model_name_or_path = model_name_or_path
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

    @collab.init
    def initialize(self) -> None:
        import datasets
        import numpy as np
        from transformers import AutoModelForCausalLM, AutoTokenizer

        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

        if self.site_name is None:
            self.site_name = collab.site_name
        site_data = Path(self.data_root) / self.site_name
        train_path = site_data / "train.jsonl"
        valid_path = site_data / "valid.jsonl"
        if not train_path.is_file() or not valid_path.is_file():
            raise FileNotFoundError(f"missing prepared data under {site_data}; run prepare_data.py first")

        self.train_dataset = datasets.load_dataset("json", data_files=str(train_path), split="train")
        self.eval_dataset = datasets.load_dataset("json", data_files=str(valid_path), split="train")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        _, _, dtype = precision_config(self.precision)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
            dtype=dtype,
        )
        self.model.config.use_cache = False
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

    @collab.publish
    def get_model_state(self) -> dict[str, torch.Tensor]:
        """Return the initial full model as native PyTorch tensors."""
        return cpu_model_state(self.model)

    @collab.publish
    def train(self, sync_number: int, global_weights: dict[str, torch.Tensor]) -> dict:
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


class SFTFedAvg:
    def __init__(
        self,
        num_epochs: int,
        syncs_per_epoch: int,
        min_clients: int,
        output_root: str,
        call_timeout: float,
        save_every_sync: bool,
    ):
        self.num_epochs = num_epochs
        self.syncs_per_epoch = syncs_per_epoch
        self.min_clients = min_clients
        self.output_root = output_root
        self.call_timeout = call_timeout
        self.save_every_sync = save_every_sync

    @collab.main
    def run(self) -> dict[str, torch.Tensor]:
        output_dir = Path(self.output_root) / "server"
        output_dir.mkdir(parents=True, exist_ok=True)

        # The server does not load the LLM. It obtains the initial full model
        # state from one initialized client as an ordinary tensor dictionary.
        global_weights = collab.clients[0](timeout=self.call_timeout).get_model_state()

        total_syncs = self.num_epochs * self.syncs_per_epoch
        for sync_number in range(1, total_syncs + 1):
            epoch_number = (sync_number - 1) // self.syncs_per_epoch + 1
            sync_in_epoch = (sync_number - 1) % self.syncs_per_epoch + 1
            print(f"=== Epoch {epoch_number}/{self.num_epochs}, " f"sync {sync_in_epoch}/{self.syncs_per_epoch} ===")
            call_results = collab.clients(timeout=self.call_timeout).train(sync_number, global_weights)
            updates = dict(call_results)
            for site_name, error in call_results.failures.items():
                print(f"Warning: {site_name} failed: {error}")

            global_weights, average_loss = average_model_states(updates, self.min_clients)
            message = f"Aggregated {len(updates)} clients: train_loss={average_loss:.4f}"
            if self.save_every_sync:
                checkpoint = output_dir / f"model_sync_{sync_number}.pt"
                torch.save(global_weights, checkpoint)
                message += f"; saved {checkpoint}"
            print(message)

        final_checkpoint = output_dir / "model_final.pt"
        torch.save(global_weights, final_checkpoint)
        print(f"Training complete. Final model: {final_checkpoint}")
        return global_weights


def validate_prepared_data(data_root: Path, num_clients: int) -> None:
    manifest_path = data_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing {manifest_path}; run prepare_data.py first")
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)

    prepared_sites = set(manifest.get("sites", []))
    requested_sites = {f"site-{site_number}" for site_number in range(1, num_clients + 1)}
    missing_sites = sorted(requested_sites - prepared_sites)
    if missing_sites:
        raise ValueError(f"data is not prepared for: {', '.join(missing_sites)}")
    for site_name in requested_sites:
        for filename in ("train.jsonl", "valid.jsonl"):
            if not (data_root / site_name / filename).is_file():
                raise FileNotFoundError(f"missing {data_root / site_name / filename}; run prepare_data.py again")


def make_recipe(args) -> CollabRecipe:
    trainer = LLMSFTClient(
        model_name_or_path=args.model_name_or_path,
        data_root=str(args.data_root),
        output_root=str(args.output_root),
        syncs_per_epoch=args.syncs_per_epoch,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        evaluate_global_model=not args.skip_evaluation,
        precision=args.precision,
    )
    server = SFTFedAvg(
        num_epochs=args.num_epochs,
        syncs_per_epoch=args.syncs_per_epoch,
        min_clients=args.num_clients,
        output_root=str(args.output_root),
        call_timeout=args.call_timeout,
        save_every_sync=args.save_every_sync,
    )
    return CollabRecipe(
        job_name="pt_llm_sft",
        server=server,
        client=trainer,
        min_clients=args.num_clients,
        sync_task_timeout=args.call_timeout,
    )


def define_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workspace-root", default="/tmp/nvflare/collab/pt_llm_sft/workspace")
    parser.add_argument("--model-name-or-path", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--num-clients", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--syncs-per-epoch", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--precision", choices=("auto", "float32", "bfloat16"), default="auto")
    parser.add_argument("--call-timeout", type=float, default=1800)
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--save-every-sync", action="store_true")
    return parser


def main() -> None:
    args = define_parser().parse_args()
    if args.num_clients < 1 or args.num_epochs < 1 or args.syncs_per_epoch < 1:
        raise ValueError("--num-clients, --num-epochs, and --syncs-per-epoch must be at least 1")

    args.data_root = Path(args.data_root).expanduser().resolve()
    args.output_root = Path(args.output_root).expanduser().resolve()
    validate_prepared_data(args.data_root, args.num_clients)

    simple_logging()
    print(
        "Starting Collab full-parameter SFT simulation\n"
        f"  model: {args.model_name_or_path}\n"
        f"  clients: {args.num_clients}\n"
        f"  epochs: {args.num_epochs}\n"
        f"  syncs per epoch: {args.syncs_per_epoch}\n"
        f"  data: {args.data_root}"
    )
    run = make_recipe(args).execute(SimEnv(num_clients=args.num_clients, workspace_root=args.workspace_root))
    print("Job Status:", run.get_status())
    print("Results at:", run.get_result())


if __name__ == "__main__":
    main()
