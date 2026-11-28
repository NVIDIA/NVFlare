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

"""Native FSDP2 language-model training with rank-zero NVFlare communication."""

import argparse
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import load_from_disk
from model import DEFAULT_MODEL, CausalLMModel
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.utils.data import DataLoader, DistributedSampler

import nvflare.client as flare


def rank_zero_call(operation):
    """Run a Client API operation on rank zero and share errors with every rank.

    The result stays on rank zero; model tensors use the distributed state-dict APIs.
    """
    result = None
    error = [None]
    if dist.get_rank() == 0:
        try:
            result = operation()
        except Exception as exc:
            error[0] = f"{type(exc).__name__}: {exc}"
    dist.broadcast_object_list(error, src=0)
    if error[0]:
        raise RuntimeError(f"rank-zero Client API operation failed: {error[0]}")
    return result


def load_global_state(model, params):
    """All ranks load; only rank zero supplies the full CPU state."""
    # PyTorch replaces entries during broadcast; keep the Client API payload on CPU.
    set_model_state_dict(
        model,
        dict(params) if dist.get_rank() == 0 else {},
        options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True),
    )


def gather_global_state(model):
    """All ranks gather; only rank zero gets a full CPU state dictionary."""
    return get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))


def make_loader(path, batch_size, training):
    dataset = load_from_disk(str(path)).with_format("torch")
    if len(dataset) < batch_size * dist.get_world_size():
        raise ValueError(f"{path} needs at least one full batch per rank; rerun prepare_data.py or lower batch_size.")
    sampler = DistributedSampler(dataset, shuffle=training, drop_last=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=True, pin_memory=True)


@torch.no_grad()
def evaluate(model, loader, device, eval_steps):
    model.eval()
    totals = torch.zeros(2, device=device)
    for step, batch in enumerate(loader):
        if step == eval_steps:
            break
        totals[0] += model(batch["input_ids"].to(device))
        totals[1] += 1
    dist.all_reduce(totals)
    return (totals[0] / totals[1]).item()


def train(model, loader, device, local_steps):
    # Build AdamW after sharding. Reset its local state at each federated round.
    # foreach=False avoids a parameter-sized temporary allocation on CUDA.
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, foreach=False)
    model.train()
    batches = iter(loader)
    total_loss = torch.zeros((), device=device)
    for _ in range(local_steps):
        try:
            batch = next(batches)
        except StopIteration:
            batches = iter(loader)
            batch = next(batches)
        optimizer.zero_grad(set_to_none=True)
        loss = model(batch["input_ids"].to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.detach()
    optimizer.zero_grad(set_to_none=True)
    dist.all_reduce(total_loss)
    return (total_loss / (local_steps * dist.get_world_size())).item()


def run_rounds(model, device, args):
    rank = dist.get_rank()
    site = [rank_zero_call(flare.get_site_name)]
    dist.broadcast_object_list(site, src=0)
    train_loader = make_loader(Path(args.data_dir) / site[0] / "train", args.batch_size, training=True)
    val_loader = make_loader(Path(args.data_dir) / "validation", args.batch_size, training=False)

    while True:
        running = [rank_zero_call(flare.is_running)]
        dist.broadcast_object_list(running, src=0)
        if not running[0]:
            break
        input_model = rank_zero_call(flare.receive)
        current_round = [input_model.current_round if rank == 0 and input_model is not None else None]
        dist.broadcast_object_list(current_round, src=0)
        if current_round[0] is None:
            break

        load_global_state(model, input_model.params if rank == 0 else None)
        del input_model
        train_loader.sampler.set_epoch(current_round[0])
        torch.cuda.reset_peak_memory_stats(device)
        # Evaluate the received global model on the same held-out blocks each round.
        val_loss = evaluate(model, val_loader, device, args.eval_steps)
        train_loss = train(model, train_loader, device, args.local_steps)
        params = gather_global_state(model)
        peak_memory = torch.tensor(torch.cuda.max_memory_allocated(device) / 1024**3, device=device)
        dist.all_reduce(peak_memory, op=dist.ReduceOp.MAX)
        metrics = {"train_loss": train_loss, "val_loss": val_loss, "peak_gpu_memory_gib": peak_memory.item()}
        try:
            rank_zero_call(
                lambda: flare.send(
                    flare.FLModel(params=params, metrics=metrics, meta={"NUM_STEPS_CURRENT_ROUND": args.local_steps})
                )
            )
        finally:
            params.clear()
        if rank == 0:
            print(f"[Site={site[0]}, Round={current_round[0]}] {metrics}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=DEFAULT_MODEL)
    parser.add_argument("--data_dir", default="/tmp/nvflare/fsdp2/data")
    parser.add_argument("--local_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()
    if min(args.local_steps, args.eval_steps, args.batch_size) < 1:
        parser.error("steps and batch_size must be positive")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("This example requires two visible CUDA GPUs per client.")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This example requires GPUs with bfloat16 support.")
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=10))
    try:
        if dist.get_world_size() != 2:
            raise RuntimeError("Launch this example with --nproc_per_node=2.")
        rank_zero_call(flare.init)
        model = CausalLMModel(args.model_name, pretrained=False).to(device)
        # Keep the tied embedding/output weights together in the root FSDP group.
        model.model.tie_weights()
        model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        mesh = init_device_mesh("cuda", (2,))
        policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        for layer in model.model.model.layers:
            fully_shard(layer, mesh=mesh, mp_policy=policy)
        fully_shard(model, mesh=mesh, mp_policy=policy)
        run_rounds(model, device, args)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
