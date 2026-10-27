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

"""Training and evaluation client used by the Chapter 1 jobs."""

from src.network import create_model
from src.training import copy_weights, evaluate, make_loaders, train, training_parser

import nvflare.client as flare
from nvflare.app_common.abstract.fl_model import FLModel


def create_writer(tracking):
    # Writers relay analytics through FLARE; the server receiver stores them.
    if tracking == "tensorboard":
        from nvflare.client.tracking import SummaryWriter

        return SummaryWriter()
    if tracking == "mlflow":
        from nvflare.client.tracking import MLflowWriter

        return MLflowWriter()
    return None


def log_metrics(writer, tracking, metrics, round_number):
    if tracking == "tensorboard":
        for name, value in metrics.items():
            writer.add_scalar(name, value, global_step=round_number)
    elif tracking == "mlflow":
        writer.log_metrics(metrics, step=round_number)


def main():
    args = training_parser().parse_args()
    flare.init()
    model = create_model()
    site = flare.get_site_name()
    train_loader, eval_loader = make_loaders(site, args.batch_size)
    writer = create_writer(args.tracking)
    print(f"{site}: batch_size={args.batch_size}, learning_rate={args.learning_rate}, local_epochs={args.local_epochs}")

    while flare.is_running():
        incoming = flare.receive()
        # The configured workflows issue train and evaluate, not submit_model.
        if not (flare.is_train() or flare.is_evaluate()):
            raise RuntimeError("This client supports train and evaluate tasks")
        model.load_state_dict(incoming.params)
        accuracy = evaluate(model, eval_loader)
        if flare.is_evaluate():
            flare.send(FLModel(metrics={"accuracy": accuracy}, meta={"num_eval_samples": len(eval_loader.dataset)}))
            continue

        steps, loss = train(model, train_loader, args.local_epochs, args.learning_rate)
        local_accuracy = evaluate(model, eval_loader)
        log_metrics(
            writer,
            args.tracking,
            {
                "global_accuracy_before_train": accuracy,
                "local_accuracy_after_train": local_accuracy,
                "local_train_loss": loss,
            },
            incoming.current_round,
        )
        print(
            f"Round {incoming.current_round}: received accuracy={accuracy:.1f}%, local accuracy={local_accuracy:.1f}%"
        )
        flare.send(
            FLModel(
                params=copy_weights(model),
                metrics={"accuracy": accuracy},
                meta={"NUM_STEPS_CURRENT_ROUND": steps},
            )
        )


if __name__ == "__main__":
    main()
