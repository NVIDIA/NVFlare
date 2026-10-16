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

"""Train one assignment, explicitly checkpointing PyTorch state between workers."""

import argparse

import torch
from client import (
    DEFAULT_CIFAR_LEARNING_RATE,
    DEFAULT_SYNTHETIC_LEARNING_RATE,
    LOCAL_MODEL_PATH,
    create_data_loaders,
    create_parser,
    evaluate,
    train_one_round,
)
from model import create_model
from prepare_data import stable_seed
from torch.optim import SGD

import nvflare.client as flare
from nvflare.client.tracking import SummaryWriter


def main():
    parser = create_parser()
    parser.add_argument("--resume_training", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    torch.set_num_threads(args.torch_threads)
    flare.init()
    site_name = flare.get_site_name()
    job_id = flare.get_job_id()
    torch.manual_seed(stable_seed(site_name, "training", args.seed))

    # This task execution profile uses CPU workers.
    device = torch.device("cpu")
    model = create_model(seed=args.seed).to(device)
    learning_rate = args.learning_rate
    if learning_rate is None:
        learning_rate = DEFAULT_SYNTHETIC_LEARNING_RATE if args.dataset == "synthetic" else DEFAULT_CIFAR_LEARNING_RATE
    optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    train_loader, test_loader = create_data_loaders(
        dataset=args.dataset,
        site_name=site_name,
        train_size=args.train_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_root=args.data_root,
        seed=args.seed,
    )
    summary_writer = SummaryWriter()
    input_model = flare.receive()
    print(f"site = {site_name}, current_round={input_model.current_round}")
    checkpoint = flare.system_info().get("task_checkpoint")

    if flare.is_train() and args.resume_training:
        if checkpoint is None:
            raise RuntimeError("Resuming training requires task checkpoint paths from the runtime")
        # The runtime supplies an isolated copy of the last committed file.
        # PyTorch owns the checkpoint contents and how they are restored.
        if checkpoint["input"] is not None:
            state = torch.load(checkpoint["input"], map_location="cpu", weights_only=True)
            if state["job_id"] != job_id or state["site_name"] != site_name:
                raise RuntimeError("Training checkpoint belongs to a different job or site")
            if state["round"] != input_model.current_round - 1:
                raise RuntimeError("Training checkpoint must come from the immediately preceding round")
            optimizer.load_state_dict(state["optimizer"])
            train_loader.generator.set_state(state["train_rng"])
            torch.set_rng_state(state["torch_rng"])
        elif input_model.current_round != 0:
            raise RuntimeError("Missing training checkpoint; use --no-resume_training to start fresh")

    # Every assignment starts from the server's global weights. Only local
    # optimizer and RNG state come from the checkpoint.
    model.load_state_dict(input_model.params)
    accuracy_before_training = evaluate(model, test_loader, device)
    if flare.is_evaluate():
        flare.send(flare.FLModel(metrics={"accuracy": accuracy_before_training}))
        return
    if not flare.is_train():
        raise RuntimeError("This task script supports training and global-model evaluation")

    steps = train_one_round(
        model, optimizer, train_loader, device, args.epochs, input_model.current_round, site_name, summary_writer
    )
    trained_accuracy = evaluate(model, test_loader, device)
    params = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    torch.save(params, LOCAL_MODEL_PATH)
    if args.resume_training:
        # Write a candidate after training. The runtime commits it together
        # with the completed result, after finalization and worker settlement.
        # A failed task leaves the previously committed checkpoint untouched.
        state = {
            "job_id": job_id,
            "site_name": site_name,
            "round": input_model.current_round,
            "optimizer": optimizer.state_dict(),
            "train_rng": train_loader.generator.get_state(),
            "torch_rng": torch.get_rng_state(),
        }
        torch.save(state, checkpoint["output"])
    flare.send(
        flare.FLModel(
            params=params,
            metrics={"accuracy": accuracy_before_training, "accuracy_after_local_training": trained_accuracy},
            meta={"NUM_STEPS_CURRENT_ROUND": steps},
        )
    )


if __name__ == "__main__":
    main()
